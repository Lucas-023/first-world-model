"""
Rollout imaginado (World Model congelado + politica) e GAE.

Ver a nota em WorldModel.encode_state: reward/done de uma acao so aparecem na
chamada de encode_state SEGUINTE, depois que a acao entra na janela
deslizante. Por isso o buffer de reward/done coletado aqui e deslocado em 1
passo em relacao a state/action/value -- reward_t (o reward atribuido a
action_t) vem de reward_logits_{t+1}, nao de reward_logits_t.
"""

import torch

REWARD_LOOKUP = torch.tensor([-1.0, 0.0, 1.0])  # classe {neg,neutro,pos} -> escalar


def compute_active_mask(dones):
    """
    dones: (H,B) float {0,1}.
    active[0] = 1; active[t] = active[t-1] * (1 - dones[t-1]) para t>=1.
    O proprio passo terminal continua ativo (conta na loss); so os passos
    DEPOIS dele (onde o World Model so imagina lixo, sem mecanismo de parada)
    ficam com peso 0.
    """
    H, B = dones.shape
    active = torch.ones_like(dones)
    for t in range(1, H):
        active[t] = active[t - 1] * (1.0 - dones[t - 1])
    return active


def compute_gae(rewards, values, dones, active_mask, bootstrap_value, gamma=0.99, gae_lambda=0.95):
    """
    rewards, values, dones, active_mask: (H,B). bootstrap_value: (B,) = V(s_H).
    Retorna (advantages, returns), ambos (H,B), com advantages ja zeradas nos
    passos inativos (active_mask==0) -- defesa extra alem da ponderacao por
    active_mask que o loop de treino tambem deve aplicar na loss (value/
    entropy nao sao automaticamente zeradas so por advantage=0).
    """
    H, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    next_value = bootstrap_value
    for t in reversed(range(H)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    advantages = advantages * active_mask
    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def collect_rollout(world_model, actor_critic, obs_ctx, act_ctx, horizon, gamma=0.99, gae_lambda=0.95):
    """
    obs_ctx: (B,context_len,64) long -- seed real (janela de contexto).
    act_ctx: (B,context_len) long -- seed real.

    Roda `horizon` passos de imaginacao guiados pela politica. Retorna dict
    de tensores (H,B,...): states, actions, log_probs, values, rewards,
    dones, active_mask, advantages, returns.
    """
    device = obs_ctx.device
    reward_lookup = REWARD_LOOKUP.to(device)

    states, actions, log_probs, values = [], [], [], []
    reward_logits_list, done_logits_list = [], []

    for _ in range(horizon):
        state_t, reward_logits_t, done_logits_t = world_model.encode_state(obs_ctx, act_ctx)
        action_t, log_prob_t, value_t, _ = actor_critic.act(state_t)
        next_frame, _, _ = world_model.imagine_next_frame(obs_ctx, act_ctx, action_t)

        states.append(state_t)
        actions.append(action_t)
        log_probs.append(log_prob_t)
        values.append(value_t)
        reward_logits_list.append(reward_logits_t)
        done_logits_list.append(done_logits_t)

        obs_ctx = torch.cat([obs_ctx[:, 1:], next_frame.unsqueeze(1)], dim=1)
        act_ctx = torch.cat([act_ctx[:, 1:], action_t.unsqueeze(1)], dim=1)

    # chamada extra: reward/done de action_{H-1} + bootstrap value de s_H
    state_H, reward_logits_H, done_logits_H = world_model.encode_state(obs_ctx, act_ctx)
    _, bootstrap_value = actor_critic.forward(state_H)
    reward_logits_list.append(reward_logits_H)
    done_logits_list.append(done_logits_H)

    horizon_ = len(actions)
    rewards = torch.stack([
        reward_lookup[reward_logits_list[t + 1].argmax(dim=-1)] for t in range(horizon_)
    ])
    dones = torch.stack([
        done_logits_list[t + 1].argmax(dim=-1) for t in range(horizon_)
    ]).float()

    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)

    active_mask = compute_active_mask(dones)
    advantages, returns = compute_gae(rewards, values, dones, active_mask, bootstrap_value, gamma, gae_lambda)

    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "active_mask": active_mask,
        "advantages": advantages,
        "returns": returns,
        "final_obs_ctx": obs_ctx,
        "final_act_ctx": act_ctx,
    }

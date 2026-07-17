"""
Teste de encanamento (nao de qualidade) do pipeline de RL via imaginacao.
Modelos minusculos, pesos aleatorios, roda em segundos na CPU. Sem pytest
(repo nao usa), so asserts + prints, igual aos test_*.py existentes.

Uso: python -m models.policy.smoke_test
"""

import torch

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.policy.modules import ActorCritic
from models.policy.rollout import collect_rollout, compute_active_mask, compute_gae
from models.policy.train_dream import ppo_update


def tiny_world_model():
    config = WorldModelConfig(
        obs_vocab_size=16, act_vocab_size=5, img_tokens=8,
        context_len=3, n_embd=32, n_head=2, n_layer=1, dropout=0.0,
    )
    model = WorldModel(config)
    model.eval()
    return model, config


def test_shapes():
    model, config = tiny_world_model()
    ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)
    B, T, K = 3, config.context_len, config.img_tokens

    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    state, rlog, dlog = model.encode_state(obs_ctx, act_ctx)
    assert state.shape == (B, config.n_embd)
    assert rlog.shape == (B, 3)
    assert dlog.shape == (B, 2)

    action, log_prob, value, entropy = ac.act(state)
    assert action.shape == (B,) and action.dtype == torch.long
    assert bool((action >= 0).all()) and bool((action < config.act_vocab_size).all())
    assert log_prob.shape == (B,) and value.shape == (B,) and entropy.shape == (B,)

    next_frame, reward_pred, done_pred = model.imagine_next_frame(obs_ctx, act_ctx, action)
    assert next_frame.shape == (B, K) and next_frame.dtype == torch.long
    assert bool((next_frame >= 0).all()) and bool((next_frame < config.obs_vocab_size).all())
    assert reward_pred.shape == (B,) and done_pred.shape == (B,)
    print("OK: 1) formas")


def test_act_token_inertness():
    """Trava de regressao: act_token nao tem efeito causal DENTRO da mesma
    chamada de imagine_next_frame -- so passa a importar na chamada seguinte,
    depois de entrar na janela deslizante (ver docstring de encode_state)."""
    model, config = tiny_world_model()
    B, T, K = 2, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    action_A = torch.zeros(B, dtype=torch.long)
    action_B = torch.full((B,), config.act_vocab_size - 1, dtype=torch.long)

    torch.manual_seed(7)
    frame_A, rew_A, done_A = model.imagine_next_frame(obs_ctx, act_ctx, action_A)
    torch.manual_seed(7)
    frame_B, rew_B, done_B = model.imagine_next_frame(obs_ctx, act_ctx, action_B)

    assert torch.equal(frame_A, frame_B), "next_frame nao deveria depender de act_token"
    assert torch.equal(rew_A, rew_B), "reward_pred nao deveria depender de act_token"
    assert torch.equal(done_A, done_B), "done_pred nao deveria depender de act_token"
    print("OK: 2) act_token e causalmente inerte dentro da mesma chamada (confirmado)")


def test_shift_correctness():
    """A consequencia de action_0 (reward/done) so aparece no passo t=1
    (via encode_state pos-slide), nunca no proprio t=0."""
    model, config = tiny_world_model()
    B, T, K = 2, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    action_A = torch.zeros(B, dtype=torch.long)
    action_B = torch.full((B,), config.act_vocab_size - 1, dtype=torch.long)

    torch.manual_seed(11)
    next_frame, _, _ = model.imagine_next_frame(obs_ctx, act_ctx, action_A)  # inerte a acao, so precisa de 1

    obs_ctx_1 = torch.cat([obs_ctx[:, 1:], next_frame.unsqueeze(1)], dim=1)
    act_ctx_1_A = torch.cat([act_ctx[:, 1:], action_A.unsqueeze(1)], dim=1)
    act_ctx_1_B = torch.cat([act_ctx[:, 1:], action_B.unsqueeze(1)], dim=1)

    _, rlog_1_A, dlog_1_A = model.encode_state(obs_ctx_1, act_ctx_1_A)
    _, rlog_1_B, dlog_1_B = model.encode_state(obs_ctx_1, act_ctx_1_B)
    assert not torch.equal(rlog_1_A, rlog_1_B), "reward_logits do passo seguinte deveria depender de action_0"

    # determinismo: mesma chamada, mesma entrada -> mesma saida (sem no_grad/dropout escondendo ruido)
    _, rlog_1_A2, dlog_1_A2 = model.encode_state(obs_ctx_1, act_ctx_1_A)
    assert torch.equal(rlog_1_A, rlog_1_A2) and torch.equal(dlog_1_A, dlog_1_A2)
    print("OK: 3) deslocamento reward/done por 1 passo confirmado")


def test_gradient_isolation():
    model, config = tiny_world_model()
    ac = ActorCritic(state_dim=config.n_embd, n_actions=config.act_vocab_size, hidden_dim=16)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)

    B, T, K = 4, config.context_len, config.img_tokens
    obs_ctx = torch.randint(0, config.obs_vocab_size, (B, T, K))
    act_ctx = torch.randint(0, config.act_vocab_size, (B, T))

    buffer = collect_rollout(model, ac, obs_ctx, act_ctx, horizon=4)
    ppo_update(ac, opt, buffer, n_epochs=1, minibatch_size=4, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)

    for p in model.parameters():
        assert p.grad is None, "gradiente vazou pro World Model (deveria estar congelado)"
    has_grad = False
    for p in ac.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
    assert has_grad, "ActorCritic deveria ter recebido gradiente"
    print("OK: 4) isolamento de gradiente (World Model intocado, ActorCritic atualizado)")


def test_gae_masking():
    H, B = 5, 2
    dones = torch.zeros(H, B)
    dones[2, 0] = 1.0  # linha 0 termina em t=2; linha 1 nunca termina

    def make_buffer(garbage_reward, garbage_value):
        rewards = torch.full((H, B), 0.1)
        values = torch.full((H, B), 0.5)
        rewards[3, 0], rewards[4, 0] = garbage_reward, -garbage_reward
        values[3, 0], values[4, 0] = garbage_value, -garbage_value
        return rewards, values

    bootstrap_value = torch.zeros(B)
    active_mask = compute_active_mask(dones)
    assert torch.equal(active_mask[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(active_mask[:, 1], torch.ones(H))

    rewards1, values1 = make_buffer(1e6, 1e6)
    rewards2, values2 = make_buffer(999.0, 42.0)

    adv1, ret1 = compute_gae(rewards1, values1, dones, active_mask, bootstrap_value)
    adv2, ret2 = compute_gae(rewards2, values2, dones, active_mask, bootstrap_value)

    assert torch.allclose(adv1, adv2), "advantage nao deveria mudar com lixo pos-done (firewall do GAE + mascara)"
    assert torch.allclose(ret1 * active_mask, ret2 * active_mask), "returns ponderados pela mascara deveriam ser identicos"
    # linha 1 (nunca termina) deve continuar contribuindo normalmente nos 5 passos
    assert bool((active_mask[:, 1] == 1.0).all())
    print("OK: 5) mascara pos-done protege GAE/returns de lixo (linha ativa continua intacta)")


if __name__ == "__main__":
    test_shapes()
    test_act_token_inertness()
    test_shift_correctness()
    test_gradient_isolation()
    test_gae_masking()
    print("\nTodos os smoke tests passaram.")

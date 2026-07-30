"""
Treino hibrido: interacao real no CarRacing-v3 + ramos imaginados pelo World
Model a partir dos estados reais visitados -- o desenho original do Dreamer
(Hafner et al.): a politica age de verdade no ambiente pra visitar estados,
mas o World Model congelado "sonha" H passos a frente a partir desses estados
reais pra gerar sinal de treino extra sem gastar mais passos reais.

Diferenca pra train_real_latent.py: aquele so usa o World Model como
codificador do frame atual (encode_state uma vez por passo, nunca chama
imagine_next_frame). Aqui, a cada --branch_every passos reais, a janela de
contexto daquele instante vira uma "ancora" e o World Model imagina
--horizon passos a frente dali (reaproveitando models/policy/rollout.py::
collect_rollout sem modificacao) -- os dois buffers (real e imaginado) sao
combinados num so buffer (H,B,...) e treinados juntos via ppo_update.

encode_state/imagine_next_frame sao stateless por chamada (kv_cache e
variavel local, recriada do zero a cada chamada -- confirmado lendo
gptdynamics.py e git log -p 5de30b8f6), entao branch/interleaving entre a
trajetoria real e os ramos imaginados nao precisa de nenhum gerenciamento de
cache.

Escopo desta primeira versao: so state_repr (h, 256-d), igual
train_real_latent.py. A variante [h,z] (train_real_latent_zh.py) fica pra
depois, se este hibrido mostrar ganho real -- precisaria de uma variante de
collect_rollout que use encode_state_and_frame nos ramos imaginados.

Uso:
    python -m models.policy.train_real_dreamer --updates 500 --episodes_per_update 8
    python -m models.policy.train_real_dreamer --benchmark_only
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch

from models.policy.modules import ActorCritic
from models.policy.train_dream import load_world_model, ppo_update
from models.policy.train_real_latent import compute_episode_gae, pack_real_batch
from models.policy.rollout import collect_rollout
from models.policy.eval_real_env import make_eval_env, run_episode_policy, frame_to_tensor, _warmup
from models.encoder.modules import VQVAE
from models.encoder.board import Board

# seeds do mini-eval periodico -- bem acima de qualquer seed que
# collect_batch_hybrid possa gerar (seed_base = args.seed + update*episodes_per_update),
# pra nunca reutilizar uma pista ja vista no treino
EVAL_SEED_BASE = 10_000_000


@torch.no_grad()
def collect_episode_with_anchors(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, seed, branch_every):
    """Roda 1 episodio real completo (mesma logica de
    train_real_latent.py::collect_episode), guardando alem disso uma "ancora"
    (obs_ctx, act_ctx) a cada `branch_every` passos reais -- a janela de
    contexto real naquele instante, capturada ANTES de escolher a acao do
    passo (semente valida pra um ramo imaginado a partir dali). Retorna None
    se o episodio terminou durante o warmup."""
    obs, _ = env.reset(seed=seed)
    obs, ended = _warmup(env, skip_frames)
    if ended:
        return None

    obs_ctx_list, act_ctx_list = [], []
    for _ in range(context_len):
        tok = vqvae.encode_indices(frame_to_tensor(obs, device)).view(1, -1)
        obs_ctx_list.append(tok)
        act_ctx_list.append(torch.zeros(1, dtype=torch.long, device=device))
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            return None

    obs_ctx = torch.stack(obs_ctx_list, dim=1)
    act_ctx = torch.stack(act_ctx_list, dim=1)

    states, actions, log_probs, values, rewards = [], [], [], [], []
    anchors = []
    step_idx = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if step_idx % branch_every == 0:
            anchors.append((obs_ctx.clone(), act_ctx.clone()))

        state, _, _ = world_model.encode_state(obs_ctx, act_ctx)
        action, log_prob, value, _ = actor_critic.act(state)

        a = int(action.item())
        obs, reward, terminated, truncated, _ = env.step(a)

        states.append(state.squeeze(0))
        actions.append(action.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(reward)

        next_tok = vqvae.encode_indices(frame_to_tensor(obs, device)).view(1, -1)
        obs_ctx = torch.cat([obs_ctx[:, 1:], next_tok.unsqueeze(1)], dim=1)
        act_ctx = torch.cat([act_ctx[:, 1:], action.unsqueeze(1)], dim=1)
        step_idx += 1

    if terminated:
        bootstrap_value = torch.zeros((), device=device)
    else:
        state_T, _, _ = world_model.encode_state(obs_ctx, act_ctx)
        _, bootstrap_value = actor_critic.forward(state_T)
        bootstrap_value = bootstrap_value.squeeze()

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "values": torch.stack(values).float(),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "bootstrap_value": bootstrap_value.float(),
        "anchors": anchors,
    }


def pad_and_concat_batches(buf_a, buf_b, device):
    """Combina dois buffers (H,B,...) com chaves states/actions/log_probs/
    advantages/returns/active_mask/rewards -- faz padding do eixo H de ambos
    pro maximo comum (zeros + active_mask=0 no preenchimento, mesmo padrao
    ja usado em pack_real_batch) e concatena no eixo do batch (dim=1).
    ppo_update nao precisa mudar: ja trata (H,B,...) generico via
    active_mask."""
    H_a, B_a = buf_a["actions"].shape
    H_b, B_b = buf_b["actions"].shape
    H = max(H_a, H_b)
    B = B_a + B_b
    state_dim = buf_a["states"].shape[-1]

    states = torch.zeros(H, B, state_dim, device=device)
    actions = torch.zeros(H, B, dtype=torch.long, device=device)
    log_probs = torch.zeros(H, B, device=device)
    advantages = torch.zeros(H, B, device=device)
    returns = torch.zeros(H, B, device=device)
    active_mask = torch.zeros(H, B, device=device)
    rewards = torch.zeros(H, B, device=device)

    states[:H_a, :B_a] = buf_a["states"]
    actions[:H_a, :B_a] = buf_a["actions"]
    log_probs[:H_a, :B_a] = buf_a["log_probs"]
    advantages[:H_a, :B_a] = buf_a["advantages"]
    returns[:H_a, :B_a] = buf_a["returns"]
    active_mask[:H_a, :B_a] = buf_a["active_mask"]
    rewards[:H_a, :B_a] = buf_a["rewards"]

    states[:H_b, B_a:] = buf_b["states"]
    actions[:H_b, B_a:] = buf_b["actions"]
    log_probs[:H_b, B_a:] = buf_b["log_probs"]
    advantages[:H_b, B_a:] = buf_b["advantages"]
    returns[:H_b, B_a:] = buf_b["returns"]
    active_mask[:H_b, B_a:] = buf_b["active_mask"]
    rewards[:H_b, B_a:] = buf_b["rewards"]

    return {
        "states": states, "actions": actions, "log_probs": log_probs,
        "advantages": advantages, "returns": returns, "active_mask": active_mask,
        "rewards": rewards,
    }


def collect_batch_hybrid(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, n_episodes, gamma, gae_lambda, seed_base, branch_every, horizon):
    """Roda `n_episodes` episodios reais completos (coletando ancoras pelo
    caminho), empacota o buffer real, imagina `horizon` passos a frente de
    TODAS as ancoras coletadas numa unica chamada batched de collect_rollout,
    e combina os dois buffers pra um so update de PPO. Retorna o buffer
    combinado + estatisticas separadas (reais vs. imaginadas) pra log."""
    episodes = []
    tries = 0
    while len(episodes) < n_episodes:
        data = collect_episode_with_anchors(
            env, world_model, vqvae, actor_critic, context_len, device, skip_frames,
            seed=seed_base + tries, branch_every=branch_every,
        )
        tries += 1
        if data is None:
            continue
        adv, ret = compute_episode_gae(data["rewards"], data["values"], data["bootstrap_value"], gamma, gae_lambda)
        data["advantages"] = adv
        data["returns"] = ret
        episodes.append(data)

    real_buffer = pack_real_batch(episodes, device)

    anchors = [a for ep in episodes for a in ep["anchors"]]
    obs_ctx_batch = torch.cat([a[0] for a in anchors], dim=0)
    act_ctx_batch = torch.cat([a[1] for a in anchors], dim=0)
    imagined_buffer = collect_rollout(world_model, actor_critic, obs_ctx_batch, act_ctx_batch, horizon, gamma, gae_lambda)

    combined = pad_and_concat_batches(real_buffer, imagined_buffer, device)

    active_sum = imagined_buffer["active_mask"].sum().clamp(min=1.0)
    imagined_mean_reward = (imagined_buffer["rewards"] * imagined_buffer["active_mask"]).sum().item() / active_sum.item()

    combined["episode_rewards"] = real_buffer["episode_rewards"]
    combined["episode_lengths"] = real_buffer["episode_lengths"]
    combined["n_anchors"] = len(anchors)
    combined["imagined_mean_reward"] = imagined_mean_reward
    return combined


def train(args):
    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "policy_ckpt.pt")
    best_path = os.path.join(args.save_dir, "policy_best.pt")

    world_model, wm_config = load_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado (context_len={wm_config.context_len}, congelado).")

    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=wm_config.obs_vocab_size).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print("VQ-VAE carregado (congelado).")

    actor_critic = ActorCritic(
        state_dim=wm_config.n_embd, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, render=False)

    if args.benchmark_only:
        is_cuda = device.startswith("cuda")

        start = time.time()
        episodes = []
        tries = 0
        while len(episodes) < args.episodes_per_update:
            data = collect_episode_with_anchors(
                env, world_model, vqvae, actor_critic, wm_config.context_len, device,
                args.skip_frames, seed=args.seed + tries, branch_every=args.branch_every,
            )
            tries += 1
            if data is None:
                continue
            adv, ret = compute_episode_gae(data["rewards"], data["values"], data["bootstrap_value"], args.gamma, args.gae_lambda)
            data["advantages"] = adv
            data["returns"] = ret
            episodes.append(data)
        if is_cuda:
            torch.cuda.synchronize()
        real_elapsed = time.time() - start
        real_steps = sum(ep["states"].shape[0] for ep in episodes)
        print(f"Real: {args.episodes_per_update} episodios ({real_steps} passos) em {real_elapsed:.2f}s -> {real_steps / real_elapsed:.2f} passos/s")

        anchors = [a for ep in episodes for a in ep["anchors"]]
        obs_ctx_batch = torch.cat([a[0] for a in anchors], dim=0)
        act_ctx_batch = torch.cat([a[1] for a in anchors], dim=0)
        start = time.time()
        collect_rollout(world_model, actor_critic, obs_ctx_batch, act_ctx_batch, args.horizon, args.gamma, args.gae_lambda)
        if is_cuda:
            torch.cuda.synchronize()
        imag_elapsed = time.time() - start
        imag_steps = len(anchors) * args.horizon
        print(f"Imaginado: {len(anchors)} ancoras x {args.horizon} horizon ({imag_steps} passos) em {imag_elapsed:.2f}s -> {imag_steps / imag_elapsed:.2f} passos/s")
        return

    board = Board(args.run_name)
    print(f"TensorBoard: tensorboard --logdir runs/{args.run_name}")

    start_update = 0
    best_eval_reward = float("-inf")
    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = ckpt["update"] + 1
        best_eval_reward = ckpt.get("best_eval_reward", float("-inf"))
        print(f"Continuando do update {start_update} | melhor reward de eval: {best_eval_reward:.4f}")
    else:
        print("Iniciando do zero.")

    print("\nIniciando treino hibrido (interacao real + ramos imaginados)...")
    for update in range(start_update, args.updates):
        buffer = collect_batch_hybrid(
            env, world_model, vqvae, actor_critic, wm_config.context_len, device,
            args.skip_frames, args.episodes_per_update, args.gamma, args.gae_lambda,
            seed_base=args.seed + update * args.episodes_per_update,
            branch_every=args.branch_every, horizon=args.horizon,
        )

        stats = ppo_update(
            actor_critic, optimizer, buffer,
            n_epochs=args.n_epochs, minibatch_size=args.minibatch_size,
            clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )

        mean_ep_reward = sum(buffer["episode_rewards"]) / len(buffer["episode_rewards"])
        mean_ep_len = sum(buffer["episode_lengths"]) / len(buffer["episode_lengths"])

        print(
            f"[Update {update:>5}] policy_loss {stats['policy_loss']:.4f} | "
            f"value_loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.4f} | "
            f"reward_real_medio {mean_ep_reward:.2f} | reward_imaginado_medio {buffer['imagined_mean_reward']:.4f} | "
            f"n_ancoras {buffer['n_anchors']}"
        )
        board.log_scalar("train/policy_loss", stats["policy_loss"], update)
        board.log_scalar("train/value_loss", stats["value_loss"], update)
        board.log_scalar("train/entropy", stats["entropy"], update)
        board.log_scalar("train/real_mean_reward", mean_ep_reward, update)
        board.log_scalar("train/real_mean_episode_length", mean_ep_len, update)
        board.log_scalar("train/imagined_mean_reward", buffer["imagined_mean_reward"], update)
        board.log_scalar("train/n_anchors", buffer["n_anchors"], update)

        # mesma logica de robustez ja provada necessaria em train_real_latent.py:
        # o metric de treino (media de poucos episodios/ancoras, seeds que mudam
        # a cada update) e ruidoso demais pra decidir o "melhor" checkpoint --
        # em vez disso, mini-eval deterministico periodico com seeds fixas nunca
        # vistas no treino.
        if update % args.eval_every == 0 or update == args.updates - 1:
            eval_rewards = []
            for i in range(args.eval_episodes):
                r, _ = run_episode_policy(
                    env, world_model, vqvae, actor_critic, wm_config.context_len, device,
                    args.skip_frames, deterministic=True, seed=EVAL_SEED_BASE + i,
                )
                eval_rewards.append(r)
            eval_reward = sum(eval_rewards) / len(eval_rewards)
            print(f"  [Eval seeds fixas] reward_medio {eval_reward:.2f} ({args.eval_episodes} episodios)")
            board.log_scalar("eval/mean_reward", eval_reward, update)

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(
                    {"update": update, "actor_critic_state_dict": actor_critic.state_dict(), "best_eval_reward": best_eval_reward},
                    best_path,
                )
                print(f"  -> Novo melhor checkpoint (eval determinístico: {best_eval_reward:.2f})")

        if args.ckpt_every > 0 and update % args.ckpt_every == 0:
            torch.save(
                {"update": update, "actor_critic_state_dict": actor_critic.state_dict(), "best_eval_reward": best_eval_reward},
                os.path.join(args.save_dir, f"policy_update{update}.pt"),
            )

        torch.save(
            {
                "update": update,
                "actor_critic_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eval_reward": best_eval_reward,
            },
            ckpt_path,
        )

    print("\nTreino finalizado!")
    env.close()
    board.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt.pt")
    p.add_argument("--save_dir", type=str, default="models/policy_real_dreamer")
    p.add_argument("--run_name", type=str, default="POLICY_REAL_DREAMER")
    p.add_argument("--updates", type=int, default=500, help="interacao real e sequencial (1 env) -- bem mais lento em wall-clock que train_dream.py; comece pequeno e calibre com --benchmark_only")
    p.add_argument("--episodes_per_update", type=int, default=8, help="episodios reais completos coletados por update")
    p.add_argument("--branch_every", type=int, default=16, help="a cada quantos passos reais tira uma ancora pra imaginar -- com episodios de ~219 passos, da ~13 ancoras/episodio")
    p.add_argument("--horizon", type=int, default=8, help="passos de imaginacao por ancora -- mesmo valor/justificativa de train_dream.py (techreport.tex: confiabilidade do reward previsto cai depois de ~10-11 passos)")
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_rows", type=int, default=12)
    p.add_argument("--skip_frames", type=int, default=12, help="frames de introducao (zoom) ignorados, igual eval_real_env.py")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.05, help="default menor que train_dream.py (0.5) -- mesmo diagnostico de colapso de entropia que motivou essa mudanca em train_real_latent.py")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=50, help="a cada N updates, roda um mini-eval deterministico com seeds fixas (nunca vistas no treino) pra decidir o checkpoint 'melhor'")
    p.add_argument("--eval_episodes", type=int, default=10, help="episodios do mini-eval periodico")
    p.add_argument("--ckpt_every", type=int, default=100, help="salva um snapshot separado (nao sobrescrito, policy_update{N}.pt) a cada N updates")
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)

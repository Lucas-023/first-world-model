"""
Variante de train_real_latent.py que aproxima o desenho do controlador do
artigo original (Ha & Schmidhuber, 2018): la, C recebe z_t (latente da VAE do
frame atual) CONCATENADO com h_t (estado oculto da RNN, usado pra prever o
proximo z) -- nao pixel bruto. Aqui usamos WorldModel.encode_state_and_frame,
que retorna state_repr (h_t -- ja e o vetor usado pra prever reward/done/
proximo frame) e frame_repr (aproximacao de z_t, pooled dos 64 tokens visuais
do ultimo frame do contexto). A politica recebe [state_repr, frame_repr]
concatenados (512-d em vez dos 256-d de train_real_latent.py).

Mesma interacao real, mesmo reward real que train_real_latent.py -- a
diferenca e SO a entrada da politica (256-d "h" vs. 512-d "[h,z]"). Compara
direto contra train_real_latent.py pra isolar se o z extra ajuda.

Uso:
    python -m models.policy.train_real_latent_zh --updates 500 --episodes_per_update 8
    python -m models.policy.train_real_latent_zh --benchmark_only
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch

from models.policy.modules import ActorCritic
from models.policy.train_dream import load_world_model, ppo_update
from models.policy.eval_real_env import make_eval_env, frame_to_tensor, _warmup, run_episode_policy
from models.encoder.modules import VQVAE
from models.encoder.board import Board

# seeds do mini-eval periodico -- bem acima de qualquer seed que collect_batch
# possa gerar (seed_base = args.seed + update*episodes_per_update), pra nunca
# reutilizar uma pista ja vista no treino
EVAL_SEED_BASE = 10_000_000


@torch.no_grad()
def collect_episode(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, seed):
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
    terminated = truncated = False
    while not (terminated or truncated):
        h, z, _, _ = world_model.encode_state_and_frame(obs_ctx, act_ctx)
        state = torch.cat([h, z], dim=-1)
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

    if terminated:
        bootstrap_value = torch.zeros((), device=device)
    else:
        h_T, z_T, _, _ = world_model.encode_state_and_frame(obs_ctx, act_ctx)
        _, bootstrap_value = actor_critic.forward(torch.cat([h_T, z_T], dim=-1))
        bootstrap_value = bootstrap_value.squeeze()

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "values": torch.stack(values).float(),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "bootstrap_value": bootstrap_value.float(),
    }


def compute_episode_gae(rewards, values, bootstrap_value, gamma, gae_lambda):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    next_value = bootstrap_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def collect_batch(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, n_episodes, gamma, gae_lambda, seed_base):
    episodes = []
    tries = 0
    while len(episodes) < n_episodes:
        data = collect_episode(env, world_model, vqvae, actor_critic, context_len, device, skip_frames, seed=seed_base + tries)
        tries += 1
        if data is None:
            continue
        adv, ret = compute_episode_gae(data["rewards"], data["values"], data["bootstrap_value"], gamma, gae_lambda)
        data["advantages"] = adv
        data["returns"] = ret
        episodes.append(data)

    H = max(ep["states"].shape[0] for ep in episodes)
    B = len(episodes)
    state_dim = episodes[0]["states"].shape[-1]

    states = torch.zeros(H, B, state_dim, device=device)
    actions = torch.zeros(H, B, dtype=torch.long, device=device)
    log_probs = torch.zeros(H, B, device=device)
    advantages = torch.zeros(H, B, device=device)
    returns = torch.zeros(H, B, device=device)
    active_mask = torch.zeros(H, B, device=device)
    rewards = torch.zeros(H, B, device=device)

    for b, ep in enumerate(episodes):
        T = ep["states"].shape[0]
        states[:T, b] = ep["states"]
        actions[:T, b] = ep["actions"]
        log_probs[:T, b] = ep["log_probs"]
        advantages[:T, b] = ep["advantages"]
        returns[:T, b] = ep["returns"]
        active_mask[:T, b] = 1.0
        rewards[:T, b] = ep["rewards"]

    return {
        "states": states, "actions": actions, "log_probs": log_probs,
        "advantages": advantages, "returns": returns, "active_mask": active_mask,
        "rewards": rewards,
        "episode_rewards": [ep["rewards"].sum().item() for ep in episodes],
        "episode_lengths": [ep["states"].shape[0] for ep in episodes],
    }


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
        state_dim=wm_config.n_embd * 2, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    env = make_eval_env(args.frame_skip, args.img_size, args.crop_rows, render=False)

    if args.benchmark_only:
        start = time.time()
        buf = collect_batch(
            env, world_model, vqvae, actor_critic, wm_config.context_len, device,
            args.skip_frames, args.episodes_per_update, args.gamma, args.gae_lambda, seed_base=args.seed,
        )
        elapsed = time.time() - start
        total_steps = sum(buf["episode_lengths"])
        print(
            f"Benchmark: {args.episodes_per_update} episodios reais "
            f"({total_steps} passos de politica) em {elapsed:.2f}s -> {total_steps / elapsed:.2f} passos/s"
        )
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

    print("\nIniciando treino de politica com interacao real ([h,z] do World Model)...")
    for update in range(start_update, args.updates):
        buffer = collect_batch(
            env, world_model, vqvae, actor_critic, wm_config.context_len, device,
            args.skip_frames, args.episodes_per_update, args.gamma, args.gae_lambda,
            seed_base=args.seed + update * args.episodes_per_update,
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
            f"reward_real_medio {mean_ep_reward:.2f} | duracao_media {mean_ep_len:.1f}"
        )
        board.log_scalar("train/policy_loss", stats["policy_loss"], update)
        board.log_scalar("train/value_loss", stats["value_loss"], update)
        board.log_scalar("train/entropy", stats["entropy"], update)
        board.log_scalar("train/mean_episode_reward", mean_ep_reward, update)
        board.log_scalar("train/mean_episode_length", mean_ep_len, update)

        # mean_ep_reward acima e a media de so --episodes_per_update episodios,
        # com as MESMAS seeds do update de treino -- ruidoso demais e tendencioso
        # pra decidir o "melhor" checkpoint (visto na pratica em train_real_latent.py:
        # um treino de 2290 updates com policy_best.pt escolhido assim colapsou sem
        # que o metric de treino avisasse). Em vez disso, a cada eval_every updates
        # roda um mini-eval deterministico com seeds fixas nunca vistas no treino.
        if update % args.eval_every == 0 or update == args.updates - 1:
            eval_rewards = []
            for i in range(args.eval_episodes):
                r, _ = run_episode_policy(
                    env, world_model, vqvae, actor_critic, wm_config.context_len, device,
                    args.skip_frames, deterministic=True, seed=EVAL_SEED_BASE + i, use_zh=True,
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

        # snapshot periodico, nunca sobrescrito -- ponto de recuperacao caso o
        # treino desestabilize mais adiante (best_path so atualiza em novo recorde,
        # entao sozinho nao bastava pra recuperar um "meio termo" razoavel)
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
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt_best.pt", help="tem que ser o MESMO checkpoint que extract_tokens.py usou pra gerar dataset_tokens/ (default ckpt_best.pt) -- ckpt.pt e um checkpoint DIFERENTE, com um codebook diferente; usar o errado aqui tokeniza os frames reais num vocabulario que o World Model nunca viu")
    p.add_argument("--save_dir", type=str, default="models/policy_real_latent_zh")
    p.add_argument("--run_name", type=str, default="POLICY_REAL_LATENT_ZH")
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--episodes_per_update", type=int, default=8)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_rows", type=int, default=12)
    p.add_argument("--skip_frames", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=50, help="a cada N updates, roda um mini-eval deterministico com seeds fixas (nunca vistas no treino) pra decidir o checkpoint 'melhor' -- mais confiavel que a media ruidosa de --episodes_per_update episodios de treino")
    p.add_argument("--eval_episodes", type=int, default=10, help="episodios do mini-eval periodico")
    p.add_argument("--ckpt_every", type=int, default=100, help="salva um snapshot separado (nao sobrescrito, policy_update{N}.pt) a cada N updates -- ponto de recuperacao caso o treino desestabilize mais adiante")
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)

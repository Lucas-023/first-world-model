"""
Treino de politica via RL puramente imaginado: o World Model (congelado,
ja treinado) "sonha" a consequencia das acoes da politica, e a politica e
treinada via PPO sobre esses rollouts imaginados. O ambiente real nunca e
tocado aqui.

Uso:
    python -m models.policy.train_dream --dataset_path dataset_tokens
    python -m models.policy.train_dream --dataset_path dataset_tokens --benchmark_only
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.dynamics.dataset import CarRacingTokenDataset
from models.policy.modules import ActorCritic
from models.policy.rollout import collect_rollout
from models.encoder.board import Board


def load_world_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    model = WorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def seed_iterator(dataset_path, context_len, batch_size, seed, stride, num_workers, device):
    """Gerador infinito de sementes reais (obs_ctx, act_ctx) pra iniciar cada rollout imaginado."""
    ds = CarRacingTokenDataset(dataset_path, split="train", context_len=context_len, seed=seed, stride=stride)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True,
    )
    while True:
        for obs_ctx, act_ctx, _, _, _ in loader:
            yield obs_ctx.to(device), act_ctx.to(device)


def save_policy_dream(world_model, vqvae, actor_critic, obs_ctx, act_ctx, save_path, n_frames, board, step):
    """Sonho guiado pela propria politica sendo treinada -- a cada passo a acao
    vem de actor_critic.act(), nao de uma acao real repetida (diferente do
    save_dream de traingpt.py, que so testa o World Model)."""
    obs_ctx = obs_ctx[0:1].clone()
    act_ctx = act_ctx[0:1].clone()
    frames, rewards, dones = [], [], []
    with torch.no_grad():
        for _ in range(n_frames):
            state, _, _ = world_model.encode_state(obs_ctx, act_ctx)
            action, _, _, _ = actor_critic.act(state)
            next_frame, reward_pred, done_cls = world_model.imagine_next_frame(obs_ctx, act_ctx, action)

            frames.append(next_frame)
            rewards.append(reward_pred.item())
            dones.append(done_cls.item())

            obs_ctx = torch.cat([obs_ctx[:, 1:], next_frame.unsqueeze(1)], dim=1)
            act_ctx = torch.cat([act_ctx[:, 1:], action.unsqueeze(1)], dim=1)

        all_tok = torch.cat(frames, dim=0)
        decoded = vqvae.decode_indices(all_tok)
        grid = make_grid(decoded.cpu(), nrow=10, normalize=True, value_range=(0, 1))
        save_image(grid, save_path)
        if board is not None:
            board.log_image("policy_dream/frames", grid, step)

    print("  -> Sonho guiado pela politica (reward/done previstos):")
    print("     reward:", [f"{r:.3f}" for r in rewards])
    print("     done  :", dones)


def ppo_update(actor_critic, optimizer, buffer, n_epochs, minibatch_size, clip_range, ent_coef, vf_coef, max_grad_norm):
    H, B = buffer["actions"].shape
    states = buffer["states"].reshape(H * B, -1)
    actions = buffer["actions"].reshape(H * B)
    old_log_probs = buffer["log_probs"].reshape(H * B)
    advantages = buffer["advantages"].reshape(H * B)
    returns = buffer["returns"].reshape(H * B)
    active_mask = buffer["active_mask"].reshape(H * B)

    # normaliza advantage so sobre os passos ativos
    active_sum = active_mask.sum().clamp(min=1.0)
    adv_mean = (advantages * active_mask).sum() / active_sum
    adv_var = ((advantages - adv_mean) ** 2 * active_mask).sum() / active_sum
    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

    n = H * B
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_steps = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=states.device)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            mask = active_mask[idx]
            mask_sum = mask.sum()
            if mask_sum.item() == 0:
                continue  # minibatch inteiro pos-done, sem passo ativo -- pular em vez de dar um step com gradiente zero

            new_log_probs, values, entropy = actor_critic.evaluate(states[idx], actions[idx])

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
            policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask_sum
            value_loss = ((values - returns[idx]) ** 2 * mask).sum() / mask_sum
            entropy_bonus = (entropy * mask).sum() / mask_sum

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_bonus.item()
            n_steps += 1

    return {k: v / n_steps for k, v in totals.items()}


def train(args):
    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    img_dir = os.path.join(args.save_dir, "dreams")
    os.makedirs(img_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "policy_ckpt.pt")
    best_path = os.path.join(args.save_dir, "policy_best.pt")

    world_model, wm_config = load_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado (context_len={wm_config.context_len}, congelado).")

    vqvae = None
    if os.path.exists(args.vqvae_path):
        try:
            from models.encoder.modules import VQVAE
            vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=wm_config.obs_vocab_size).to(device)
            vqvae.load_state_dict(
                torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"]
            )
            vqvae.eval()
            for p in vqvae.parameters():
                p.requires_grad_(False)
            print("VQ-VAE carregado (so para visualizacao).")
        except Exception as e:
            print(f"Aviso VQ-VAE: {e}")

    actor_critic = ActorCritic(
        state_dim=wm_config.n_embd, n_actions=wm_config.act_vocab_size, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    seeds = seed_iterator(
        args.dataset_path, wm_config.context_len, args.batch_size, args.seed, args.seed_stride, args.num_workers, device
    )

    if args.benchmark_only:
        obs_ctx, act_ctx = next(seeds)
        is_cuda = device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        n_bench = 5
        for _ in range(n_bench):
            collect_rollout(world_model, actor_critic, obs_ctx, act_ctx, args.horizon, args.gamma, args.gae_lambda)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start
        steps = n_bench * args.horizon
        print(
            f"Benchmark: {steps} passos imaginados (batch={args.batch_size}, horizon={args.horizon}) "
            f"em {elapsed:.2f}s -> {steps / elapsed:.2f} passos/s"
        )
        return

    board = Board(args.run_name)
    print(f"TensorBoard: tensorboard --logdir runs/{args.run_name}")

    start_update = 0
    best_reward = float("-inf")
    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = ckpt["update"] + 1
        best_reward = ckpt.get("best_reward", float("-inf"))
        print(f"Continuando do update {start_update} | melhor reward: {best_reward:.4f}")
    else:
        print("Iniciando do zero.")

    print("\nIniciando treino de politica via imaginacao...")
    for update in range(start_update, args.updates):
        obs_ctx, act_ctx = next(seeds)
        buffer = collect_rollout(world_model, actor_critic, obs_ctx, act_ctx, args.horizon, args.gamma, args.gae_lambda)

        stats = ppo_update(
            actor_critic, optimizer, buffer,
            n_epochs=args.n_epochs, minibatch_size=args.minibatch_size,
            clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )

        active_sum = buffer["active_mask"].sum().clamp(min=1).item()
        mean_reward = (buffer["rewards"] * buffer["active_mask"]).sum().item() / active_sum
        mean_active = buffer["active_mask"].mean().item()

        print(
            f"[Update {update:>5}] policy_loss {stats['policy_loss']:.4f} | "
            f"value_loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.4f} | "
            f"reward_medio {mean_reward:.4f} | fracao_ativa {mean_active:.4f}"
        )
        board.log_scalar("train/policy_loss", stats["policy_loss"], update)
        board.log_scalar("train/value_loss", stats["value_loss"], update)
        board.log_scalar("train/entropy", stats["entropy"], update)
        board.log_scalar("train/mean_reward", mean_reward, update)
        board.log_scalar("train/active_fraction", mean_active, update)

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(
                {"update": update, "actor_critic_state_dict": actor_critic.state_dict(), "best_reward": best_reward},
                best_path,
            )
            print(f"  -> Novo melhor reward medio ({best_reward:.4f})")

        if vqvae is not None and update % args.viz_every == 0:
            try:
                save_path = os.path.join(img_dir, f"policy_dream_{update:05d}.png")
                save_policy_dream(world_model, vqvae, actor_critic, obs_ctx, act_ctx, save_path, args.viz_frames, board, update)
                print(f"  -> Imagem: {save_path}")
            except Exception as e:
                print(f"  Aviso viz: {e}")

        torch.save(
            {
                "update": update,
                "actor_critic_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_reward": best_reward,
            },
            ckpt_path,
        )

    print("\nTreino finalizado!")
    board.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True, help="dataset_tokens -- so usado pra sementes de contexto real")
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt_best.pt", help="so usado pra decodificar os sonhos em imagem no TensorBoard -- usar o checkpoint diferente do que extract_tokens.py usou (ckpt.pt em vez de ckpt_best.pt) so deixa a visualizacao com aparencia errada, nao afeta o treino (os tokens de treino ja vem prontos de dataset_tokens/)")
    p.add_argument("--save_dir", type=str, default="models/policy")
    p.add_argument("--run_name", type=str, default="POLICY_DREAM")
    p.add_argument("--updates", type=int, default=5000)
    p.add_argument("--horizon", type=int, default=8, help="passos de imaginacao por rollout -- eval_rollout.py mostrou reward confiavel (acima do baseline de persistencia) so ate ~passo 10-11, consistente em 3 checkpoints diferentes")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed_stride", type=int, default=8, help="stride ao amostrar sementes reais (so precisa de diversidade, nao exaustividade)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--viz_every", type=int, default=20)
    p.add_argument("--viz_frames", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)

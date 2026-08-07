"""
Inferencia para Dynamics GPT (imagem + reward + done).
"""

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image, make_grid

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.modules import VQVAE


def load_world_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    model = WorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def load_vqvae(ckpt_path, vocab_size, device):
    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=vocab_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    vqvae.load_state_dict(state)
    vqvae.eval()
    return vqvae


def get_test_files(dataset_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return files[n_train + n_val:]


@torch.no_grad()
def dream(model, seed_obs, seed_act, future_act, temperature, top_k):
    ctx_obs = seed_obs.unsqueeze(0)
    ctx_act = seed_act.unsqueeze(0)
    dreamed_obs, dreamed_rew, dreamed_done = [], [], []
    for i in range(future_act.shape[0]):
        act_token = future_act[i:i + 1]
        next_obs, rew_cls, done_cls = model.imagine_next_frame(
            ctx_obs, ctx_act, act_token, temperature=temperature, top_k=top_k
        )
        dreamed_obs.append(next_obs)
        dreamed_rew.append(rew_cls)
        dreamed_done.append(done_cls)

        ctx_obs = torch.cat([ctx_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
        ctx_act = torch.cat([ctx_act[:, 1:], act_token.unsqueeze(1)], dim=1)

    return (
        torch.cat(dreamed_obs, dim=0),
        torch.cat(dreamed_rew, dim=0),
        torch.cat(dreamed_done, dim=0),
    )


def tensor_to_pil_frames(decoded, upscale):
    decoded = decoded.clamp(0, 1).cpu()
    frames = []
    for img in decoded:
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        if upscale > 1:
            pil = pil.resize((pil.width * upscale, pil.height * upscale), resample=Image.NEAREST)
        frames.append(pil)
    return frames


def save_gif(frames, path, fps):
    duration = int(1000 / max(fps, 1))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False, disposal=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default="dataset_tokens")
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/gpt_best.pt")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt.pt")
    p.add_argument("--output_dir", type=str, default="results/dreams_gpt")
    p.add_argument("--episode_idx", type=int, default=None)
    p.add_argument("--n_dream", type=int, default=40)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model, config = load_world_model(args.dynamics_ckpt, args.device)
    vqvae = load_vqvae(args.vqvae_path, config.obs_vocab_size, args.device)

    files = get_test_files(args.dataset_path, seed=args.seed)
    if not files:
        raise FileNotFoundError(f"Nenhum episodio no split de teste: {args.dataset_path}")

    if args.episode_idx is None:
        args.episode_idx = int(rng.integers(0, len(files)))
    ep = np.load(files[args.episode_idx], allow_pickle=False)

    tokens = torch.from_numpy(ep["tokens"].astype(np.int64)).to(args.device)
    actions = torch.from_numpy(ep["actions"].astype(np.int64)).to(args.device)
    rewards = torch.from_numpy((np.sign(ep["rewards"].astype(np.float32)) + 1).astype(np.int64)).to(args.device)
    dones = torch.from_numpy(ep["dones"].astype(np.int64)).to(args.device)

    T = tokens.shape[0]
    n_dream = min(args.n_dream, T - config.context_len)
    seed_obs = tokens[:config.context_len]
    seed_act = actions[:config.context_len]
    future_act = actions[config.context_len:config.context_len + n_dream]
    real_obs = tokens[config.context_len:config.context_len + n_dream]
    real_rew = rewards[config.context_len:config.context_len + n_dream]
    real_done = dones[config.context_len:config.context_len + n_dream]

    dream_obs, dream_rew, dream_done = dream(
        model, seed_obs, seed_act, future_act, args.temperature, args.top_k
    )

    seed_imgs = vqvae.decode_indices(seed_obs)
    dream_imgs = vqvae.decode_indices(dream_obs)
    real_imgs = vqvae.decode_indices(real_obs)

    dream_seq = torch.cat([seed_imgs, dream_imgs], dim=0)
    real_seq = torch.cat([seed_imgs, real_imgs], dim=0)
    save_gif(tensor_to_pil_frames(dream_seq, args.upscale), os.path.join(args.output_dir, "dream.gif"), args.fps)
    save_gif(tensor_to_pil_frames(real_seq, args.upscale), os.path.join(args.output_dir, "real.gif"), args.fps)

    n_show = min(20, n_dream)
    combined = torch.cat([dream_imgs[:n_show], real_imgs[:n_show]], dim=0)
    grid = make_grid(combined.cpu(), nrow=n_show, normalize=True, value_range=(0, 1))
    save_image(grid, os.path.join(args.output_dir, "comparison.png"))

    obs_acc = (dream_obs == real_obs).float().mean().item()
    rew_acc = (dream_rew == real_rew).float().mean().item()
    done_acc = (dream_done == real_done).float().mean().item()
    print(f"Token accuracy: {obs_acc * 100:.2f}%")
    print(f"Reward class accuracy: {rew_acc * 100:.2f}%")
    print(f"Done accuracy: {done_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
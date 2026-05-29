"""
Inferencia do Dynamics Model treinado (gptimage + trainimage).

Pega um episodio tokenizado, usa os primeiros `context_len` frames como seed
e sonha `n_dream` frames a frente usando as acoes reais do episodio.

Salva:
    dream.gif        — sonho do modelo
    real.gif         — continuacao real do episodio
    comparison.png   — grid (sonho na linha de cima, real embaixo)

Uso (a partir da raiz do projeto):
    python -m models.dynamics.inferimage \\
        --dataset_path dataset_tokens \\
        --dynamics_ckpt models/dynamics/dynamics_best.pt \\
        --vqvae_path    models/VQVAE/ckpt.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image, make_grid

from models.dynamics.gptimage import DynamicsModel, DynamicsConfig
from models.encoder.modules   import VQVAE


def load_dynamics(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # O checkpoint salvo por trainimage.py guarda config.__dict__ em "config"
    cfg_dict = ckpt.get("config")
    if cfg_dict is None:
        print("Aviso: checkpoint sem 'config' — usando defaults de DynamicsConfig.")
        config = DynamicsConfig()
    else:
        # Reconstroi config sem os campos derivados (tokens_per_block / block_size)
        init_keys = {
            "obs_vocab_size", "act_vocab_size", "img_tokens", "context_len",
            "n_embd", "n_head", "n_layer", "dropout",
        }
        kept = {k: v for k, v in cfg_dict.items() if k in init_keys}
        config = DynamicsConfig(**kept)

    model = DynamicsModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Dynamics carregado de {ckpt_path}")
    if "epoch" in ckpt:
        print(f"  epoca: {ckpt['epoch']} | val_loss: {ckpt.get('val_loss','?')}")
    return model, config


def load_vqvae(ckpt_path, vocab_size, device):
    vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=vocab_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    vqvae.load_state_dict(state)
    vqvae.eval()
    print(f"VQ-VAE carregado de {ckpt_path}")
    return vqvae


def get_test_files(dataset_path, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Replica o mesmo split do DynamicsDataset para garantir que a inferencia
    use apenas episodios do conjunto de teste."""
    files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
    rng   = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    n       = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return files[n_train + n_val:]


def pick_episode(dataset_path, episode_idx, context_len, n_dream, rng):
    files = get_test_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"Nenhum .npz no split de teste em {dataset_path}")

    if episode_idx is None:
        # Tenta achar um episodio longo o suficiente
        for _ in range(20):
            idx = int(rng.integers(0, len(files)))
            d   = np.load(files[idx], allow_pickle=False)
            if d["tokens"].shape[0] >= context_len + n_dream:
                episode_idx = idx
                break
        if episode_idx is None:
            raise RuntimeError(
                f"Nao achei episodio com >= {context_len + n_dream} frames. "
                f"Reduza --n_dream ou --context_len."
            )

    path = files[episode_idx]
    d    = np.load(path, allow_pickle=False)
    print(f"Episodio (test split): {os.path.basename(path)}  ({d['tokens'].shape[0]} frames)")
    return d, path


@torch.no_grad()
def dream(model, seed_obs, seed_act, future_act, temperature, top_k):
    """
    seed_obs   : (context_len, 64)  tokens reais de seed
    seed_act   : (context_len,)     acoes reais do seed
    future_act : (n_dream,)         acoes reais que viriam no futuro
    Retorna    : (n_dream, 64)      tokens sonhados
    """
    device  = seed_obs.device
    ctx_obs = seed_obs.unsqueeze(0)   # (1, T, 64)
    ctx_act = seed_act.unsqueeze(0)   # (1, T)

    dreamed = []
    for i in range(future_act.shape[0]):
        act_token = future_act[i:i+1].to(device)                       # (1,)
        next_obs  = model.imagine_next_frame(
            ctx_obs, ctx_act, act_token,
            temperature=temperature, top_k=top_k,
        )                                                              # (1, 64)
        dreamed.append(next_obs)

        ctx_obs = torch.cat([ctx_obs[:, 1:], next_obs.unsqueeze(1)], dim=1)
        ctx_act = torch.cat([ctx_act[:, 1:], act_token.unsqueeze(1)], dim=1)

    return torch.cat(dreamed, dim=0)   # (n_dream, 64)


def tensor_to_pil_frames(decoded, upscale):
    """decoded: (N, 3, H, W) em [0,1] → lista de PIL.Image RGB."""
    decoded = decoded.clamp(0, 1).cpu()
    frames  = []
    for img in decoded:
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        if upscale > 1:
            pil = pil.resize(
                (pil.width * upscale, pil.height * upscale),
                resample=Image.NEAREST,
            )
        frames.append(pil)
    return frames


def save_gif(frames, path, fps):
    duration = int(1000 / max(fps, 1))   # ms por frame
    frames[0].save(
        path,
        save_all       = True,
        append_images  = frames[1:],
        duration       = duration,
        loop           = 0,
        optimize       = False,
        disposal       = 2,
    )
    print(f"  -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",   type=str,   default="dataset_tokens")
    p.add_argument("--dynamics_ckpt",  type=str,   default="models/dynamics/dynamics_best.pt")
    p.add_argument("--vqvae_path",     type=str,   default="models/VQVAE/ckpt.pt")
    p.add_argument("--output_dir",     type=str,   default="results/dreams")
    p.add_argument("--episode_idx",    type=int,   default=None,
                   help="Indice do episodio (apos sort). Se None, sorteia.")
    p.add_argument("--n_dream",        type=int,   default=40,
                   help="Quantos frames sonhar apos o seed.")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--fps",            type=int,   default=10)
    p.add_argument("--upscale",        type=int,   default=4,
                   help="Multiplicador de pixel-art para os frames salvos.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # -- Modelos --------------------------------------------------------------
    model, config = load_dynamics(args.dynamics_ckpt, device)
    vqvae         = load_vqvae(args.vqvae_path, config.obs_vocab_size, device)

    # -- Episodio -------------------------------------------------------------
    ep, ep_path = pick_episode(
        args.dataset_path, args.episode_idx,
        config.context_len, args.n_dream, rng,
    )
    tokens  = torch.from_numpy(ep["tokens"].astype(np.int64))    # (T, 64)
    actions = torch.from_numpy(ep["actions"].astype(np.int64))   # (T,)

    T          = tokens.shape[0]
    n_dream    = min(args.n_dream, T - config.context_len)
    seed_obs   = tokens [:config.context_len].to(device)
    seed_act   = actions[:config.context_len].to(device)
    future_act = actions[config.context_len : config.context_len + n_dream].to(device)
    real_obs   = tokens [config.context_len : config.context_len + n_dream].to(device)

    print(f"Seed: {config.context_len} frames | Sonhando: {n_dream} frames "
          f"(temp={args.temperature}, top_k={args.top_k})")

    # -- Sonho ----------------------------------------------------------------
    dream_tokens = dream(
        model, seed_obs, seed_act, future_act,
        temperature=args.temperature, top_k=args.top_k,
    )                                                            # (n_dream, 64)

    # -- Decode ---------------------------------------------------------------
    seed_imgs  = vqvae.decode_indices(seed_obs)                  # (T, 3, 64, 64)
    dream_imgs = vqvae.decode_indices(dream_tokens)              # (n_dream, ...)
    real_imgs  = vqvae.decode_indices(real_obs)                  # (n_dream, ...)

    # -- GIFs -----------------------------------------------------------------
    # Cada GIF mostra o seed (real) + a continuacao (sonhada ou real)
    dream_seq = torch.cat([seed_imgs, dream_imgs], dim=0)
    real_seq  = torch.cat([seed_imgs, real_imgs],  dim=0)

    save_gif(tensor_to_pil_frames(dream_seq, args.upscale),
             os.path.join(args.output_dir, "dream.gif"), args.fps)
    save_gif(tensor_to_pil_frames(real_seq,  args.upscale),
             os.path.join(args.output_dir, "real.gif"),  args.fps)

    # -- Grid comparativo (sonho em cima, real embaixo) -----------------------
    n_show   = min(20, n_dream)
    combined = torch.cat([dream_imgs[:n_show], real_imgs[:n_show]], dim=0)
    grid     = make_grid(combined.cpu(), nrow=n_show,
                         normalize=True, value_range=(0, 1))
    grid_path = os.path.join(args.output_dir, "comparison.png")
    save_image(grid, grid_path)
    print(f"  -> {grid_path}  (linha 1: sonho | linha 2: real)")

    # -- Token accuracy (sanity) ----------------------------------------------
    acc = (dream_tokens == real_obs).float().mean().item()
    print(f"\nToken-accuracy sonho vs real: {acc*100:.2f}%")
    print(f"Episodio usado: {os.path.basename(ep_path)}")


if __name__ == "__main__":
    main()

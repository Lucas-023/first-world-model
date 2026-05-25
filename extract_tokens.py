"""
Extrai tokens VQ-VAE de cada episodio.

Saida por episodio (dataset_tokens/episode_XXXX.npz):
    tokens  : (T, 64) uint16  — indices do codebook por frame (grid 8x8)
    actions : (T,)    int32   — acoes discretas (0-4)
    rewards : (T,)    float32 — rewards continuos BRUTOS (sem discretizar)
    dones   : (T,)    uint8   — 0 ou 1

A conversao de reward para 3 classes {-1,0,+1} e feita no dataset,
nao aqui — assim este arquivo fica agnostico sobre como o reward sera usado.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from models.encoder.modules import VQVAE


def extract_tokens(
    dataset_in:  str,
    dataset_out: str,
    vqvae_ckpt:  str,
    vocab_size:  int = 512,
    skip_frames: int = 12,
    device:      str = "cuda",
):
    os.makedirs(dataset_out, exist_ok=True)

    print(f"Carregando VQ-VAE de: {vqvae_ckpt}")
    vqvae = VQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=vocab_size,
    ).to(device)

    ckpt = torch.load(vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    files   = sorted([f for f in os.listdir(dataset_in) if f.endswith(".npz")])
    skipped = 0

    print(f"Extraindo tokens de {len(files)} episodios...")

    with torch.no_grad():
        for f in tqdm(files):
            data    = np.load(os.path.join(dataset_in, f))
            obs     = data["obs"]       # (T, 3, H, W) float32
            actions = data["actions"]   # (T,)
            rewards = data["rewards"]   # (T,) float32 continuo
            dones   = data["dones"]     # (T,)

            if len(obs) <= skip_frames:
                skipped += 1
                continue

            obs     = obs    [skip_frames:]
            actions = actions[skip_frames:]
            rewards = rewards[skip_frames:]
            dones   = dones  [skip_frames:]

            obs_tensor = torch.from_numpy(obs).float().to(device)
            _, _, indices = vqvae(obs_tensor)

            # indices: (T, 8, 8) -> (T, 64)
            tokens = indices.view(obs_tensor.size(0), -1).cpu().numpy().astype(np.uint16)

            np.savez_compressed(
                os.path.join(dataset_out, f),
                tokens  = tokens,                       # (T, 64) uint16
                actions = actions.astype(np.int32),     # (T,)
                rewards = rewards.astype(np.float32),   # (T,) continuo bruto
                dones   = dones.astype(np.uint8),       # (T,)
            )

    print(f"Extracao concluida! ({skipped} episodios ignorados)")


if __name__ == "__main__":
    extract_tokens(
        dataset_in  = "dataset",
        dataset_out = "dataset_tokens",
        vqvae_ckpt  = "models/VQVAE/ckpt.pt",
    )
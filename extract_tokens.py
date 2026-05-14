"""
Extrai tokens VQ-VAE de cada episodio e discretiza rewards.

Saida por episodio (dataset_tokens/episode_XXXX.npz):
    tokens  : (T, 64)  uint16  — indices do codebook por frame
    actions : (T,)     int32   — acoes discretas (0-4)
    rewards : (T,)     int32   — rewards discretizados em `vocab_reward` bins
    dones   : (T,)     uint8   — 0 ou 1

Layout de vocabulario (deve coincidir com WorldModelConfig):
    [img(512) | action(5) | reward(21) | done(2)]
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from models.encoder.modules import VQVAE


# ---------------------------------------------------------------------------
# Discretizacao de reward — padrao IRIS: 21 bins em [-1, 1]
# ---------------------------------------------------------------------------

def discretize_rewards(rewards: np.ndarray, n_bins: int = 21) -> np.ndarray:
    """
    Mapeia rewards continuos para bins inteiros em [0, n_bins-1].
    Rewards sao clampados em [-1, 1] antes do binning.

    bin 0  → reward ~ -1.0
    bin 10 → reward ~  0.0  (bin central)
    bin 20 → reward ~ +1.0
    """
    clipped = np.clip(rewards, -1.0, 1.0)
    # Normaliza para [0, 1] e mapeia para [0, n_bins-1]
    bins = ((clipped + 1.0) / 2.0 * (n_bins - 1)).round().astype(np.int32)
    return bins


def extract_tokens(
    dataset_in:  str,
    dataset_out: str,
    vqvae_ckpt:  str,
    vocab_size:  int = 512,
    vocab_reward: int = 21,
    skip_frames: int = 12,   # remove zoom inicial (12 steps * frame_skip=4 = ~1s)
    device:      str = "cuda",
):
    os.makedirs(dataset_out, exist_ok=True)

    # -- Carrega VQ-VAE -------------------------------------------------------
    print(f"Carregando VQ-VAE de: {vqvae_ckpt}")
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=vocab_size).to(device)
    ckpt  = torch.load(vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    files = sorted([f for f in os.listdir(dataset_in) if f.endswith(".npz")])
    print(f"Extraindo tokens de {len(files)} episodios...")

    skipped = 0
    with torch.no_grad():
        for f in tqdm(files):
            data    = np.load(os.path.join(dataset_in, f))
            obs     = data["obs"]       # (T, 3, 64, 64) float32
            actions = data["actions"]   # (T,)
            rewards = data["rewards"]   # (T,) float32 continuo
            dones   = data["dones"]     # (T,) bool

            # Remove frames iniciais de zoom
            if len(obs) <= skip_frames:
                skipped += 1
                continue
            obs     = obs[skip_frames:]
            actions = actions[skip_frames:]
            rewards = rewards[skip_frames:]
            dones   = dones[skip_frames:]

            # -- Extrai tokens visuais via VQ-VAE -----------------------------
            obs_tensor = torch.from_numpy(obs).float().to(device)
            _, _, indices = vqvae(obs_tensor)
            # indices: (T, 8, 8) → (T, 64)
            tokens = indices.view(obs_tensor.size(0), -1).cpu().numpy().astype(np.uint16)

            # -- Discretiza rewards -------------------------------------------
            rewards_disc = discretize_rewards(rewards, n_bins=vocab_reward)

            np.savez_compressed(
                os.path.join(dataset_out, f),
                tokens  = tokens,                           # (T, 64) uint16
                actions = actions.astype(np.int32),         # (T,)
                rewards = rewards_disc,                     # (T,) int32 — JA DISCRETIZADO
                dones   = dones.astype(np.uint8),           # (T,)
            )

    print(f"Extracao concluida! ({skipped} episodios ignorados por serem curtos demais)")


if __name__ == "__main__":
    extract_tokens(
        dataset_in  = "dataset",
        dataset_out = "dataset_tokens",
        vqvae_ckpt  = "models/VQVAE/ckpt.pt",
    )
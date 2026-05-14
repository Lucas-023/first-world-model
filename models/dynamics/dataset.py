"""
CarRacingTokenDataset
=====================
Carrega episodios ja tokenizados pelo extract_tokens.py e os divide
em janelas de comprimento fixo para treinar o World Model.

IMPORTANTE: os rewards chegam JA DISCRETIZADOS do extract_tokens.py.
Nao discretizar de novo aqui.

Cada item devolvido pelo __getitem__:
    tokens  : (seq_len, 64)  torch.long  — indices visuais [0, vocab_img)
    actions : (seq_len,)     torch.long  — [0, vocab_action) = [0, 4]
    rewards : (seq_len,)     torch.long  — [0, vocab_reward) = [0, 20]
    dones   : (seq_len,)     torch.long  — 0 ou 1
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingTokenDataset(Dataset):

    def __init__(self, dataset_path: str, seq_len: int = 20):
        self.seq_len = seq_len
        self.chunks  = []

        files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
        print(f"Carregando {len(files)} episodios de tokens...")

        for f in files:
            npz = np.load(f, allow_pickle=False)

            tokens  = npz["tokens"].astype(np.int64)   # (T, 64)
            actions = npz["actions"].astype(np.int64)  # (T,)
            rewards = npz["rewards"].astype(np.int64)  # (T,) — JA discretizado
            dones   = npz["dones"].astype(np.int64)    # (T,)

            T = tokens.shape[0]
            if T < seq_len:
                continue

            # Janelas sem sobreposicao de tamanho seq_len
            for start in range(0, T - seq_len + 1, seq_len):
                end = start + seq_len
                self.chunks.append((
                    tokens [start:end],   # (seq_len, 64)
                    actions[start:end],   # (seq_len,)
                    rewards[start:end],   # (seq_len,)
                    dones  [start:end],   # (seq_len,)
                ))

        print(f"Total de sequencias: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tok, act, rew, don = self.chunks[idx]
        return (
            torch.from_numpy(tok).long(),   # (seq_len, 64)
            torch.from_numpy(act).long(),   # (seq_len,)
            torch.from_numpy(rew).long(),   # (seq_len,)
            torch.from_numpy(don).long(),   # (seq_len,)
        )
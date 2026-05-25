"""
CarRacingTokenDataset
=====================
Carrega episodios tokenizados e devolve janelas de tamanho seq_len
para treinar o World Model.

Cada item devolvido:
    obs_tokens   : (seq_len, 16)  long  — indices visuais [0, obs_vocab)
    act_tokens   : (seq_len,)     long  — acoes [0, 4]
    rewards_sign : (seq_len,)     long  — {0=negativo, 1=neutro, 2=positivo}
    dones        : (seq_len,)     long  — {0, 1}

A conversao de reward continuo para 3 classes e feita aqui via sign(),
identica ao IRIS original:
    rewards.sign() + 1  →  -1→0, 0→1, qualquer positivo→2
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingTokenDataset(Dataset):

    def __init__(self, folder: str, seq_len: int = 20):
        self.seq_len = seq_len
        self.chunks  = []

        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        print(f"Carregando {len(files)} episodios de tokens...")

        for f in files:
            d = np.load(f, allow_pickle=False)

            tokens  = d["tokens"].astype(np.int64)    # (T, 16)
            actions = d["actions"].astype(np.int64)   # (T,)
            dones   = d["dones"].astype(np.int64)     # (T,)

            # Converte reward continuo para 3 classes
            rewards = d["rewards"].astype(np.float32) # (T,) continuo
            rewards_sign = (np.sign(rewards) + 1).astype(np.int64)  # {0,1,2}

            T = tokens.shape[0]
            if T < seq_len:
                continue

            for start in range(0, T - seq_len + 1, seq_len):
                end = start + seq_len
                self.chunks.append((
                    tokens      [start:end],   # (seq_len, 16)
                    actions     [start:end],   # (seq_len,)
                    rewards_sign[start:end],   # (seq_len,)
                    dones       [start:end],   # (seq_len,)
                ))

        print(f"Total de sequencias: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tok, act, rew, don = self.chunks[idx]
        return (
            torch.from_numpy(tok).long(),
            torch.from_numpy(act).long(),
            torch.from_numpy(rew).long(),
            torch.from_numpy(don).long(),
        )
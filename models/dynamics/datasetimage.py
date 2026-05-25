"""
Dataset para o Dynamics Model com janela deslizante.

Estrutura de cada janela:
    contexto : (context_len, 64)  — ultimos context_len frames
    target   : (64,)              — proximo frame a prever
    actions  : (context_len,)     — acoes do contexto

    A acao usada para prever o target e a ultima acao do contexto,
    ou seja actions[-1] e a acao tomada no ultimo frame do contexto
    que levou ao target.

Isso e consistente com a inferencia:
    "dados os ultimos 19 frames e suas acoes, preve o frame 20"

Split por episodio (sem data leakage):
    train : 80%
    val   : 10%
    test  : 10%
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicsDataset(Dataset):

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        folder:      str,
        split:       str   = "train",
        context_len: int   = 19,
        train_ratio: float = 0.7,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
    ):
        assert split in self.SPLITS, f"split deve ser um de {self.SPLITS}"
        assert train_ratio + val_ratio < 1.0

        self.context_len = context_len
        self.chunks      = []   # cada item: (obs_ctx, act_ctx, obs_target)

        # -- Split por episodio -----------------------------------------------
        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        rng   = np.random.default_rng(seed)
        files = [files[i] for i in rng.permutation(len(files))]

        n       = len(files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            selected = files[:n_train]
        elif split == "val":
            selected = files[n_train : n_train + n_val]
        else:
            selected = files[n_train + n_val:]

        print(f"[{split:>5}] {len(selected)}/{n} episodios")

        # -- Janela deslizante de passo 1 ------------------------------------
        # Para cada episodio de T frames, gera T - context_len janelas.
        # Janela i: contexto = frames[i .. i+context_len-1]
        #           target   = frames[i + context_len]
        #           acoes    = actions[i .. i+context_len-1]
        for f in selected:
            d       = np.load(f, allow_pickle=False)
            tokens  = d["tokens"].astype(np.int64)   # (T, 64)
            actions = d["actions"].astype(np.int64)  # (T,)

            T = tokens.shape[0]
            # Precisa de pelo menos context_len + 1 frames
            if T < context_len + 1:
                continue

            for i in range(T - context_len):
                obs_ctx    = tokens [i : i + context_len]       # (context_len, 64)
                act_ctx    = actions[i : i + context_len]       # (context_len,)
                obs_target = tokens [i + context_len]           # (64,)
                self.chunks.append((obs_ctx, act_ctx, obs_target))

        print(f"[{split:>5}] {len(self.chunks)} janelas "
              f"(contexto={context_len} frames, target=1 frame)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        obs_ctx, act_ctx, obs_target = self.chunks[idx]
        return (
            torch.from_numpy(obs_ctx).long(),     # (context_len, 64)
            torch.from_numpy(act_ctx).long(),     # (context_len,)
            torch.from_numpy(obs_target).long(),  # (64,)
        )
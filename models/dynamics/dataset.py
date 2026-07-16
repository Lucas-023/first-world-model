"""
Dataset para Dynamics GPT com contexto -> proximo passo.

Cada janela devolve:
    obs_ctx      : (context_len, 64)
    act_ctx      : (context_len,)
    obs_target   : (64,)
    reward_class : ()  em {0,1,2} = {neg, neutro, pos}
    done_target  : ()  em {0,1}
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingTokenDataset(Dataset):
    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        folder: str,
        split: str = "train",
        context_len: int = 19,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        stride: int = 1,
    ):
        assert split in self.SPLITS, f"split deve ser um de {self.SPLITS}"
        assert train_ratio + val_ratio < 1.0
        assert stride >= 1

        self.context_len = context_len
        self.chunks = []

        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        rng = np.random.default_rng(seed)
        files = [files[i] for i in rng.permutation(len(files))]

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            selected = files[:n_train]
        elif split == "val":
            selected = files[n_train:n_train + n_val]
        else:
            selected = files[n_train + n_val:]

        print(f"[{split:>5}] {len(selected)}/{n} episodios")

        for f in selected:
            d = np.load(f, allow_pickle=False)
            tokens = d["tokens"].astype(np.int64)        # (T, 64)
            actions = d["actions"].astype(np.int64)      # (T,)
            dones = d["dones"].astype(np.int64)          # (T,)
            rewards = d["rewards"].astype(np.float32)    # (T,)
            rewards_sign = (np.sign(rewards) + 1).astype(np.int64)  # {0,1,2}

            T = tokens.shape[0]
            if T < context_len + 1:
                continue

            for i in range(0, T - context_len, stride):
                obs_ctx = tokens[i:i + context_len]                   # (context_len, 64)
                act_ctx = actions[i:i + context_len]                  # (context_len,)
                obs_target = tokens[i + context_len]                  # (64,)
                reward_target = rewards_sign[i + context_len]         # ()
                done_target = dones[i + context_len]                  # ()
                self.chunks.append((obs_ctx, act_ctx, obs_target, reward_target, done_target))

        print(f"[{split:>5}] {len(self.chunks)} janelas (contexto={context_len}, stride={stride})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        obs_ctx, act_ctx, obs_target, reward_target, done_target = self.chunks[idx]
        return (
            torch.from_numpy(obs_ctx).long(),
            torch.from_numpy(act_ctx).long(),
            torch.from_numpy(obs_target).long(),
            torch.tensor(reward_target, dtype=torch.long),
            torch.tensor(done_target, dtype=torch.long),
        )


DynamicsDataset = CarRacingTokenDataset
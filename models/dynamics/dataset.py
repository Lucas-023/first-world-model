import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CarRacingTokenDataset(Dataset):
    def __init__(self, folder_path, seq_len=20, max_files=None):
        super().__init__()
        self.seq_len = seq_len

        self.NUM_REWARD_BINS = 21
        self.NUM_ACTION_BINS = 11

        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        if max_files:
            files = files[:max_files]

        self.episodes = []
        self.valid_indices = []

        for ep_id, f in enumerate(files):
            data = np.load(os.path.join(folder_path, f))

            tokens = data['tokens']
            actions = data['actions']
            rewards = data['rewards']
            dones = data['dones'].astype(np.int64)

            # reward
            rewards = np.clip(rewards, -10, 10)
            rewards = ((rewards + 10) / 20 * (self.NUM_REWARD_BINS - 1)).astype(np.int64)

            # action
            act = actions.copy()
            act[:, 0] = np.clip(act[:, 0], -1, 1)
            act[:, 1:] = np.clip(act[:, 1:], 0, 1)

            act_bins = np.zeros_like(act, dtype=np.int64)
            act_bins[:, 0] = ((act[:, 0] + 1) / 2 * (self.NUM_ACTION_BINS - 1)).astype(np.int64)
            act_bins[:, 1:] = (act[:, 1:] * (self.NUM_ACTION_BINS - 1)).astype(np.int64)

            self.episodes.append({
                "tokens": tokens,
                "actions": act_bins,
                "rewards": rewards,
                "dones": dones
            })

            T = len(tokens)
            for start in range(T - seq_len):
                self.valid_indices.append((ep_id, start))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_id, start = self.valid_indices[idx]
        end = start + self.seq_len

        ep = self.episodes[ep_id]

        z = torch.tensor(ep["tokens"][start:end], dtype=torch.long)
        a = torch.tensor(ep["actions"][start:end], dtype=torch.long)
        r = torch.tensor(ep["rewards"][start:end], dtype=torch.long)
        d = torch.tensor(ep["dones"][start:end], dtype=torch.long)

        return z, a, r, d
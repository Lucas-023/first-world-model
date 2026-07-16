import os
import bisect
import numpy as np
import torch

from torch.utils.data import Dataset


class CarRacingDataset(Dataset):

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        folder_path,
        split:       str   = "train",
        train_ratio: float = 0.15,
        val_ratio:   float = 0.05,
        seed:        int   = 42,
        max_files:   int   = None,
    ):

        assert split in self.SPLITS
        assert train_ratio + val_ratio < 1.0

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".npz")
        ])

        rng = np.random.default_rng(seed)
        files = [files[i] for i in rng.permutation(len(files))]

        if max_files is not None:
            files = files[:max_files]

        n       = len(files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            selected = files[:n_train]
        elif split == "val":
            selected = files[n_train:n_train+n_val]
        else:
            selected = files[n_train+n_val:]

        print(f"[{split:>5}] {len(selected)}/{n} episodios")

        self.folder_path      = folder_path
        self.files            = selected
        self.episode_offsets  = [0]

        for fname in selected:
            path = os.path.join(folder_path, fname)
            data = np.load(path, mmap_mode="r")
            T    = data["obs"].shape[0]
            self.episode_offsets.append(self.episode_offsets[-1] + T)

        print(f"✅ {self.episode_offsets[-1]} frames indexados")

    def __len__(self):
        return self.episode_offsets[-1]

    def __getitem__(self, idx):
        ep_idx    = bisect.bisect_right(self.episode_offsets, idx) - 1
        frame_idx = idx - self.episode_offsets[ep_idx]

        path = os.path.join(self.folder_path, self.files[ep_idx])
        data = np.load(path, mmap_mode="r")

        frame = np.array(data["obs"][frame_idx]).astype(np.float32)

        if frame.max() > 1.0:
            frame /= 255.0

        return torch.from_numpy(frame).float()
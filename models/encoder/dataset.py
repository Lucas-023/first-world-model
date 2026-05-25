import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingDataset(Dataset):

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        folder_path,
        split:       str   = "train",
        train_ratio: float = 0.7,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
        max_files:   int   = None,
    ):
        assert split in self.SPLITS, f"split deve ser um de {self.SPLITS}"
        assert train_ratio + val_ratio < 1.0

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])

        rng   = np.random.default_rng(seed)
        files = [files[i] for i in rng.permutation(len(files))]

        if max_files is not None:
            files = files[:max_files]

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
        print(f"⏳ Carregando {len(selected)} arquivos para RAM...")

        all_frames = []

        for f in selected:
            path = os.path.join(folder_path, f)
            data = np.load(path)["obs"]  # (T, 3, H, W)
            data = data.astype(np.float32)
            if data.max() > 1.0:
                data = data / 255.0
            all_frames.append(data)

        self.data = np.concatenate(all_frames, axis=0)
        mb = self.data.nbytes / (1024 ** 2)
        print(f"✅ {len(self.data)} frames carregados ({mb:.1f} MB)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

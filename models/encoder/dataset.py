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

        self.folder_path = folder_path
        self.index_map   = []

        print(f"[{split:>5}] {len(selected)}/{n} episodios")

        # --------------------------------------------------
        # Cria mapeamento global:
        #
        # idx global -> (arquivo, frame_idx)
        # --------------------------------------------------

        for fname in selected:

            path = os.path.join(folder_path, fname)

            # mmap_mode evita carregar tudo
            data = np.load(path, mmap_mode="r")

            T = data["obs"].shape[0]

            for frame_idx in range(T):
                self.index_map.append(
                    (path, frame_idx)
                )

        print(f"✅ {len(self.index_map)} frames indexados")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        path, frame_idx = self.index_map[idx]

        # carrega somente UM frame
        data = np.load(path, mmap_mode="r")

        frame = data["obs"][frame_idx]

        frame = frame.astype(np.float32)

        if frame.max() > 1.0:
            frame = frame / 255.0

        return torch.from_numpy(frame).float()
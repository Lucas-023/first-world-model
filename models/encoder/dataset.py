import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingDataset(Dataset):

    def __init__(self, folder_path, max_files=None):
        super().__init__()

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".npz")
        ])

        if max_files is not None:
            files = files[:max_files]

        print(f"⏳ Carregando {len(files)} arquivos para RAM...")

        all_frames = []

        for f in files:

            path = os.path.join(folder_path, f)

            data = np.load(path)["obs"]  # (T, 3, 64, 64)

            data = data.astype(np.float32)

            # garante [0,1]
            if data.max() > 1.0:
                data = data / 255.0

            all_frames.append(data)

        self.data = np.concatenate(all_frames, axis=0)

        mb = self.data.nbytes / (1024 ** 2)

        print(
            f"✅ {len(self.data)} frames carregados "
            f"({mb:.1f} MB)"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = torch.from_numpy(
            self.data[idx]
        ).float()

        return x
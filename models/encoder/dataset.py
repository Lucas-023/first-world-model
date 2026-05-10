import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CarRacingDataset(Dataset):
    def __init__(self, folder_path, max_files=None):
        super().__init__()

        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

        if max_files is not None:
            files = files[:max_files]

        print(f"⏳ Carregando {len(files)} arquivos para a RAM...")

        all_frames = []
        for f in files:
            data = np.load(os.path.join(folder_path, f))['obs']  # (T, 3, 64, 64) float32
            all_frames.append(data)

        # (N_total, 3, 64, 64) float32 — ja normalizado [0,1], ja canal-first
        self.data = np.concatenate(all_frames, axis=0)

        mb = self.data.nbytes / (1024 ** 2)
        print(f"✅ {len(self.data)} imagens na RAM. Tamanho total: {mb:.1f} MB.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Converte para tensor — sem nenhuma transform, shape ja esta certo
        return torch.from_numpy(self.data[idx])  # (3, 64, 64) float32
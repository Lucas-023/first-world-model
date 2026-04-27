import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class GrayToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1)

class CarRacingDataset(Dataset):
    def __init__(self, folder_path, max_files=100):
        super().__init__()
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        
        if max_files is not None:
            files = files[:max_files]
            
        print(f"⏳ Carregando {len(files)} arquivos para a RAM...")
        
        all_frames = []
        for f in files:
            file_path = os.path.join(folder_path, f)
            data = np.load(file_path)['obs'] 
            all_frames.append(data)
            
        self.data = np.concatenate(all_frames, axis=0)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            GrayToRGB(),
        ])
        
        tamanho_mb = self.data.nbytes / (1024 * 1024)
        print(f"✅ {len(self.data)} imagens na RAM. Tamanho total: {tamanho_mb:.1f} MB.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
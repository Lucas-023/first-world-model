import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CarRacingDataset(Dataset):
    def __init__(self, folder_path, max_files=100):
        super().__init__()
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        
        # 1. LIMITA O DATASET: Carrega apenas os primeiros 'max_files'
        if max_files is not None:
            files = files[:max_files]
            
        print(f"⏳ Carregando {len(files)} arquivos para a RAM...")
        
        all_frames = []
        for f in files:
            file_path = os.path.join(folder_path, f)
            data = np.load(file_path)['obs'] 
            all_frames.append(data)
            
        # 2. OTIMIZAÇÃO DE MEMÓRIA: Mantemos como numpy array 'uint8'
        # Isso ocupa 4x menos espaço na sua RAM!
        self.data = np.concatenate(all_frames, axis=0)
        
        # O ToTensor() já converte de uint8 para float32 (0.0 a 1.0) automaticamente
        self.transform = transforms.ToTensor()
        
        tamanho_mb = self.data.nbytes / (1024 * 1024)
        print(f"✅ {len(self.data)} imagens na RAM. Tamanho total: {tamanho_mb:.1f} MB.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 3. A conversão pesada (float32) só acontece na hora de ir para a GPU
        # Isso resolve a lentidão de 13 minutos E resolve a morte do terminal!
        return self.transform(self.data[idx])
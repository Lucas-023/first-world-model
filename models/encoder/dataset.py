import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CarRacingDataset(Dataset):
    def __init__(self, folder_path, max_frames_per_episode=300):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.max_frames = max_frames_per_episode
        
        # O ToTensor() do PyTorch converte arrays numpy de shape (A, L, Canais) em uint8 (0-255)
        # diretamente para tensores de shape (Canais, A, L) em float32 (0.0 a 1.0).
        self.transform = transforms.ToTensor()
        
        # Sistema de Cache: Evita que o HD leia o mesmo arquivo 300 vezes seguidas
        self.cached_file_idx = -1
        self.cached_data = None

    def __len__(self):
        # O DataLoader precisa saber o total de imagens disponíveis
        return len(self.files) * self.max_frames

    def __getitem__(self, idx):
        # A matemática mágica para descobrir em qual arquivo o "idx" está
        file_idx = idx // self.max_frames
        frame_idx = idx % self.max_frames
        
        if file_idx >= len(self.files):
            file_idx = len(self.files) - 1

        # Carrega do HD apenas se o DataLoader pediu um arquivo diferente do atual
        if file_idx != self.cached_file_idx:
            file_path = os.path.join(self.folder_path, self.files[file_idx])
            # Extraímos APENAS a matriz 'obs' (as imagens), ignorando actions/rewards no VQ-VAE
            self.cached_data = np.load(file_path)['obs'] 
            self.cached_file_idx = file_idx

        # Se o carro bateu e o episódio salvou menos de 300 frames, evitamos o erro "Out of Bounds"
        if frame_idx >= len(self.cached_data):
            frame_idx = len(self.cached_data) - 1
            
        # Pega a imagem numpy original (64, 64, 3)
        image_np = self.cached_data[frame_idx]
        
        # Transforma para Tensor PyTorch (3, 64, 64)
        image_tensor = self.transform(image_np)
        
        return image_tensor
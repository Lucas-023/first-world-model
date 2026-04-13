import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarRacingSequenceDataset(Dataset):
    def __init__(self, folder_path, seq_len=5, max_frames_per_episode=300):
        """
        seq_len: Quantos frames o Transformer vai ver de uma vez (janela de contexto)
        """
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.seq_len = seq_len
        self.max_frames = max_frames_per_episode
        
        # Se o vídeo tem 300 frames e a sequência tem 5, 
        # temos 296 "janelas" possíveis de 5 frames dentro de um único vídeo.
        self.samples_per_file = self.max_frames - self.seq_len + 1
        
        self.transform = transforms.ToTensor()
        
        # Sistema de Cache para não explodir a leitura do HD
        self.cached_file_idx = -1
        self.cached_data = {}

    def __len__(self):
        # Total de "vídeos curtos" que podemos extrair de todo o dataset
        return len(self.files) * self.samples_per_file

    def __getitem__(self, idx):
        # Descobre de qual arquivo tirar e em qual frame começar
        file_idx = idx // self.samples_per_file
        start_frame = idx % self.samples_per_file
        
        if file_idx >= len(self.files):
            file_idx = len(self.files) - 1

        # Abre o arquivo .npz apenas se mudamos de episódio
        if file_idx != self.cached_file_idx:
            file_path = os.path.join(self.folder_path, self.files[file_idx])
            data = np.load(file_path)
            
            # Carrega tudo para a memória RAM (Cache)
            self.cached_data['obs'] = data['obs']
            self.cached_data['actions'] = data['actions']
            self.cached_data['rewards'] = data['rewards']
            self.cached_data['dones'] = data['dones']
            self.cached_file_idx = file_idx

        # Fim da nossa janela (ex: começa no 10, termina no 15)
        end_frame = start_frame + self.seq_len
        
        # Proteção caso o carro tenha batido e o episódio seja menor que 300 frames
        total_real_frames = len(self.cached_data['obs'])
        if end_frame > total_real_frames:
            end_frame = total_real_frames
            start_frame = max(0, end_frame - self.seq_len)

        # Corta a fatia exata de tempo (Sequência)
        obs_seq = self.cached_data['obs'][start_frame:end_frame]
        act_seq = self.cached_data['actions'][start_frame:end_frame]
        rew_seq = self.cached_data['rewards'][start_frame:end_frame]
        done_seq = self.cached_data['dones'][start_frame:end_frame]
        
        # Converter Imagens (T, 64, 64, 3) para Tensor (T, 3, 64, 64) entre 0.0 e 1.0
        obs_tensors = torch.stack([self.transform(img) for img in obs_seq])
        
        # Converter Ações (T, 3) para Tensor
        act_tensors = torch.tensor(act_seq, dtype=torch.float32)
        
        # Converter Recompensas e Dones (T,) para Tensor
        rew_tensors = torch.tensor(rew_seq, dtype=torch.float32)
        done_tensors = torch.tensor(done_seq, dtype=torch.float32)

        # O DataLoader agora cospe as 4 variáveis para o train_gpt.py
        return obs_tensors, act_tensors, rew_tensors, done_tensors
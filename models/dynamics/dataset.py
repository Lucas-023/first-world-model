import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CarRacingTokenDataset(Dataset):
    def __init__(self, folder_path, seq_len=20, max_files=None):
        super().__init__()
        self.seq_len = seq_len
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        if max_files: files = files[:max_files]
            
        print(f"⏳ Carregando {len(files)} episódios convertidos em TOKENS para a RAM...")
        
        self.episodes = []
        self.valid_indices = [] 
        
        for ep_id, f in enumerate(files):
            data = np.load(os.path.join(folder_path, f))
            self.episodes.append({
                'tokens': data['tokens'],  # Agora carregamos 'tokens' ao invés de 'obs'
                'actions': data['actions'],
                'rewards': data['rewards'],
                'dones': data['dones']
            })
            
            total_frames = len(data['tokens'])
            if total_frames >= seq_len:
                for start_frame in range(total_frames - seq_len + 1):
                    self.valid_indices.append((ep_id, start_frame))
                    
        print(f"✅ Dataset na RAM! Total de sequências de {seq_len} frames: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_id, start_frame = self.valid_indices[idx]
        end_frame = start_frame + self.seq_len
        
        ep_data = self.episodes[ep_id]
        
        # O Tensor de imagens sumiu! Agora passamos os tokens diretamente
        tokens_seq = ep_data['tokens'][start_frame:end_frame]
        act_seq = ep_data['actions'][start_frame:end_frame]
        rew_seq = ep_data['rewards'][start_frame:end_frame]
        done_seq = ep_data['dones'][start_frame:end_frame]
        
        tokens_tensors = torch.tensor(tokens_seq, dtype=torch.long)
        act_tensors = torch.tensor(act_seq, dtype=torch.float32)
        rew_tensors = torch.tensor(rew_seq, dtype=torch.float32)
        done_tensors = torch.tensor(done_seq, dtype=torch.float32)

        return tokens_tensors, act_tensors, rew_tensors, done_tensors
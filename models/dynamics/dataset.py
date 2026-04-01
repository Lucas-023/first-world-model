import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class PongSequenceDataset(Dataset):
    def __init__(self, folder_path, vqvae_model, frames_per_seq=5, device="cuda", cache_file="latents_cache.pt"):
        self.frames_per_seq = frames_per_seq
        self.seq_len = frames_per_seq * 64
        self.cache_file = cache_file
        
        if os.path.exists(cache_file):
            print(f"Carregando cache de: {cache_file}")
            cache = torch.load(cache_file, map_location='cpu', weights_only=True)
            self.tokens = cache["tokens"]
            self.actions = cache["actions"]
            
            self.rewards = cache["rewards"]
            self.dones = cache["dones"]
        else:
            self.tokens, self.actions, self.rewards, self.dones = self._encode_dataset(folder_path, vqvae_model, device)
            
    def _encode_dataset(self, folder_path, vqvae, device):
        vqvae.eval()
        vqvae.to(device)
        
        csv_path = os.path.join(folder_path, "metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Ficheiro {csv_path} não encontrado!")
            
        all_tokens = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        print("A processar imagens e a criar tokens...")
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            
            for row in tqdm(reader):
                img_name, action, reward, done = row
                
                # O VQ-VAE extrai os tokens
                img_path = os.path.join(folder_path, img_name)
                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 1. Passa a imagem pelo Encoder
                    z = vqvae.encoder(img_tensor)
                    # 2. Passa o resultado pelo Quantizador para pegar os índices
                    _, _, indices = vqvae.vq(z)
                    
                    all_tokens.append(indices.cpu())
                
                all_actions.append(int(action))
                
                # Mapear a recompensa: -1.0 -> 0,  0.0 -> 1,  1.0 -> 2
                r = float(reward)
                if r < 0: r_class = 0
                elif r > 0: r_class = 2
                else: r_class = 1
                
                all_rewards.append(r_class)
                all_dones.append(int(done))
                
        all_tokens = torch.cat(all_tokens, dim=0).flatten()
        all_actions = torch.tensor(all_actions, dtype=torch.long)
        all_rewards = torch.tensor(all_rewards, dtype=torch.long)
        all_dones = torch.tensor(all_dones, dtype=torch.long)
        
        torch.save({
            "tokens": all_tokens, 
            "actions": all_actions,
            "rewards": all_rewards,
            "dones": all_dones
        }, self.cache_file)
        
        print(f"✅ Cache concluída!")
        return all_tokens, all_actions, all_rewards, all_dones
    

    def __len__(self):
        total_frames = len(self.actions)
        # Em vez de avançar 1 por 1, dividimos pelo tamanho da sequência (frames_per_seq)
        return (total_frames - 1) // self.frames_per_seq

    def __getitem__(self, idx):
        start_token_idx = idx * 64
        start_frame_idx = idx

        x_tokens = self.tokens[start_token_idx : start_token_idx + self.seq_len]
        y_tokens = self.tokens[start_token_idx + 1 : start_token_idx + self.seq_len + 1]

        actions = self.actions[start_frame_idx : start_frame_idx + self.frames_per_seq]
        
        # MUDANÇA: Pegamos as recompensas e dones de TODOS os frames da sequência
        # O shape retornado será [frames_per_seq], ex: um vetor de 5 valores
        rewards = self.rewards[start_frame_idx : start_frame_idx + self.frames_per_seq]
        dones = self.dones[start_frame_idx : start_frame_idx + self.frames_per_seq]

        return x_tokens, y_tokens, actions, rewards, dones
import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class PongSequenceDataset(Dataset):
    def __init__(self, folder_path, vqvae_model=None, frames_per_seq=5, device='cuda', cache_file="latents_cache.pt"):
        self.frames_per_seq = frames_per_seq
        self.seq_len = frames_per_seq * 64 # 64 tokens por frame
        self.cache_file = cache_file
        
        # Se a cache já existir, carregamos a visão e as ações instantaneamente
        if os.path.exists(cache_file):
            print(f"Carregando cache com ações de: {cache_file}")
            cache = torch.load(cache_file, map_location='cpu')
            self.tokens = cache["tokens"]
            self.actions = cache["actions"]
        else:
            print(f"⚙️ Cache não encontrada. Extraindo tokens e lendo o CSV...")
            if vqvae_model is None:
                raise ValueError("VQ-VAE precisa ser fornecido para criar a cache pela primeira vez!")
            self.tokens, self.actions = self._encode_dataset(folder_path, vqvae_model, device)
            
    def _encode_dataset(self, folder_path, vqvae, device):
        vqvae.eval()
        vqvae.to(device)
        
        csv_path = os.path.join(folder_path, "metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo {csv_path} não encontrado! Rode o script de gerar dataset novamente.")
            
        # 1. Lê o CSV para saber a ordem correta e as ações tomadas
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "frame": row["frame"],
                    "acao": int(row["acao"])
                })
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        all_tokens = []
        all_actions = []
        batch_size = 128
        
        # 2. Extrai as peças visuais usando o VQ-VAE
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Extraindo Visão + Ações"):
                batch_data = data[i:i+batch_size]
                tensors = []
                
                for item in batch_data:
                    img_path = os.path.join(folder_path, item["frame"])
                    img = Image.open(img_path).convert('RGB')
                    tensors.append(transform(img))
                    
                    # Guarda a ação do joystick
                    all_actions.append(item["acao"])
                
                batch_tensor = torch.stack(tensors).to(device)
                
                # Passamos pelo VQ-VAE
                _, _, indices = vqvae(batch_tensor)
                
                # Os índices vêm no formato [Batch, 8, 8]. Achatamos para [Batch, 64]
                indices = indices.view(batch_tensor.size(0), -1)
                all_tokens.append(indices.cpu())
        
        # 3. Junta tudo em vetores gigantes e contínuos
        all_tokens = torch.cat(all_tokens, dim=0).flatten()
        all_actions = torch.tensor(all_actions, dtype=torch.long)
        
        torch.save({"tokens": all_tokens, "actions": all_actions}, self.cache_file)
        print(f"✅ Cache concluída! {len(all_actions)} frames processados e salvos.")
        
        return all_tokens, all_actions

    def __len__(self):
        # Desconta os frames da janela para não dar erro de "index out of bounds" no final
        total_frames = len(self.actions)
        return total_frames - self.frames_per_seq - 1

    def __getitem__(self, idx):
        # Agora o dataset avança frame a frame de forma deslizante
        start_token_idx = idx * 64
        
        # x_tokens: A tela atual (Ex: 320 pecinhas visuais)
        x_tokens = self.tokens[start_token_idx : start_token_idx + self.seq_len]
        
        # y_tokens: O que o Transformer tem que adivinhar (As mesmas pecinhas, deslocadas 1 passo no futuro)
        y_tokens = self.tokens[start_token_idx + 1 : start_token_idx + self.seq_len + 1]
        
        # x_actions: Os botões do joystick que estavam sendo apertados nesses frames
        x_actions = self.actions[idx : idx + self.frames_per_seq]
        
        # Entregamos o combo completo!
        return x_tokens.long(), y_tokens.long(), x_actions.long()
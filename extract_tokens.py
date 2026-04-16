import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from models.encoder.modules import VQVAE

def extract_tokens(dataset_in, dataset_out, vqvae_ckpt, vocab_size=512, device="cuda"):
    os.makedirs(dataset_out, exist_ok=True)
    
    print(f"Carregando VQ-VAE de: {vqvae_ckpt}")
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=vocab_size).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device)["model_state_dict"])
    vqvae.eval()
    
    transform = transforms.ToTensor()
    files = [f for f in os.listdir(dataset_in) if f.endswith('.npz')]
    
    print(f"Extraindo tokens de {len(files)} episódios...")
    with torch.no_grad():
        for f in tqdm(files):
            # Lê a imagem do HD
            data = np.load(os.path.join(dataset_in, f))
            obs_np = data['obs']  # Formato (300, 64, 64, 3) em uint8
            
            # Converte as 300 imagens do episódio para Tensor e joga para a GPU
            obs_tensor = torch.stack([transform(img) for img in obs_np]).to(device)
            
            # Passa no VQ-VAE para pegar os índices (tokens)
            _, _, indices = vqvae(obs_tensor)
            
            # Formato de saída: (300, 64) - 300 frames, 64 tokens por frame
            tokens_np = indices.view(obs_tensor.size(0), -1).cpu().numpy().astype(np.uint16)
            
            # Salva no novo diretório, descartando as imagens pesadas!
            np.savez_compressed(
                os.path.join(dataset_out, f),
                tokens=tokens_np,
                actions=data['actions'],
                rewards=data['rewards'],
                dones=data['dones']
            )
            
    print("✅ Extração concluída! Seus dados agora estão minúsculos e prontos para o Transformer.")

if __name__ == '__main__':
    # Altere os caminhos se necessário
    extract_tokens(
        dataset_in="dataset_carracing", 
        dataset_out="dataset_tokens", 
        vqvae_ckpt="models/VQVAE_CARRACING/ckpt.pt"
    )
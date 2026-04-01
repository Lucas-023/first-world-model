import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from models.dynamics.dataset import PongSequenceDataset
from models.dynamics.modules import GPTDynamicsModel

from models.encoder.modules import VQVAE 

def train(args):
    device = args.device
    
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=512).to(device)
    vqvae_ckpt_path = "models/VQVAE_PONG/ckpt.pt"
    
    if os.path.exists(vqvae_ckpt_path):
        print(f"Carregando 'olhos' do modelo (VQ-VAE) de: {vqvae_ckpt_path}")
        checkpoint = torch.load(vqvae_ckpt_path, map_location=device, weights_only=True)
        vqvae.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Aviso: Checkpoint do VQ-VAE não encontrado. Certifique-se de que a cache já existe.")
    
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    print("Carregando Dataset (Isso pode demorar na 1ª vez para extrair a cache)...")
    dataset = PongSequenceDataset(
        folder_path=args.dataset_path, 
        vqvae_model=vqvae, 
        frames_per_seq=args.seq_len // 64,  # 320 // 64 = 5 frames
        device=device
    )
    
    # MUDANÇA: num_workers=0 para evitar o "Multiprocessing Deadlock" no Linux
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = GPTDynamicsModel(
        vocab_size=512, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers,
        num_actions=6
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    
    token_counts = torch.bincount(dataset.tokens.long(), minlength=512).float()
    token_counts = torch.where(token_counts == 0, torch.tensor(1.0), token_counts)
    total_tokens = token_counts.sum()
    
    weights_pixels = total_tokens / (512 * token_counts)
    weights_pixels = torch.clamp(weights_pixels, min=0.1, max=10.0).to(device)
    
    criterion_pixels = nn.CrossEntropyLoss(weight=weights_pixels)
    
    
    # 3. Loss para o Fim de Jogo (Pesos Manuais Muito Fortes)
    weights_dones = torch.tensor([1.0, 100.0], dtype=torch.float32).to(device)
    criterion_dones = nn.CrossEntropyLoss(weight=weights_dones)

    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "dynamics_ckpt.pt")

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint do Transformer encontrado! Retomando de: {ckpt_path}")
        # MUDANÇA: Adicionado weights_only=True
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0

        for x, y, a, r, d in pbar:
            x, y, a = x.to(device), y.to(device), a.to(device)
            r, d = r.to(device), d.to(device)
            
            optimizer.zero_grad()
            
            logits_pixels, logits_dones = model(x, actions=a) 
            
            # 1. Loss dos Pixels (Avalia todos os 320 tokens)
            loss_pixels = criterion_pixels(logits_pixels.view(-1, 512), y.view(-1))
            
            frame_ends = torch.arange(63, x.size(1), 64, device=device)
            
            pred_dones = logits_dones[:, frame_ends, :]
            
            loss_dones = criterion_dones(pred_dones.reshape(-1, 2), d.view(-1))


            lambda_pixels = 1.0
            lambda_dones = 20.0 

            loss = (lambda_pixels * loss_pixels) + (lambda_dones * loss_dones)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Época {epoch} - Loss Médio: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DYNAMICS_PONG")
    parser.add_argument('--dataset_path', type=str, required=True, help="Caminho para os .png do Pong")
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--seq_len', type=int, default=320)  # 5 frames * 64 tokens/frame
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default="cuda")
    
    args = parser.parse_args()
    train(args)
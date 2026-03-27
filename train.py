import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Importa os módulos que criámos
from models.dynamics.dataset import PongSequenceDataset
from models.dynamics.modules import GPTDynamicsModel
from models.encoder.modules import VQVAE 

def train(args):
    device = args.device
    
    # 1. Preparar os "Olhos" (VQ-VAE)
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=512).to(device)
    vqvae_ckpt_path = "models/VQVAE_PONG/ckpt.pt"
    
    if os.path.exists(vqvae_ckpt_path):
        print(f"👁️ A carregar 'olhos' do modelo (VQ-VAE) de: {vqvae_ckpt_path}")
        checkpoint = torch.load(vqvae_ckpt_path, map_location=device, weights_only=False)
        vqvae.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError("Checkpoint do VQ-VAE não encontrado! Treine o VQ-VAE primeiro.")
    
    # Congelar o VQ-VAE (não queremos treiná-lo de novo, apenas usá-lo)
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    # 2. Carregar o Dataset (A nova versão com Ações do comando!)
    print("📚 A carregar o Dataset e a Cache...")
    dataset = PongSequenceDataset(
        folder_path=args.dataset_path, 
        vqvae_model=vqvae, 
        frames_per_seq=5, 
        device=device,
        cache_file="latents_cache.pt"
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Preparar o "Cérebro" (Transformer)
    print("🧠 A iniciar o Transformer...")
    model = GPTDynamicsModel(
        vocab_size=512, 
        d_model=args.d_model, 
        n_heads=8, 
        n_layers=4, 
        num_actions=6 # 6 botões possíveis no Pong do Atari
    ).to(device)
    
    # 4. Ferramentas de Estudo (Otimizador e Corretor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- A MÁGICA DA LOSS PONDERADA (INVERSO DA FREQUÊNCIA) ---
    print("⚖️ A calcular os pesos da Loss Ponderada para focar na bola e raquetes...")
    todos_os_tokens = dataset.tokens 
    
    # Conta quantas vezes cada token (de 0 a 511) aparece no dataset
    frequencias = torch.bincount(todos_os_tokens, minlength=512).float()
    
    # Soma 1 para evitar o erro de dividir por zero (caso um token nunca tenha sido usado)
    frequencias += 1.0 
    
    # O peso é o inverso da frequência (quem aparece muito, pesa pouco)
    pesos = 1.0 / frequencias
    
    # Normalizamos para que o peso médio fique em torno de 1.0
    pesos = pesos / pesos.mean()
    
    # Trava de segurança: impede que o peso passe de 50.0 
    # (Senão um token que apareceu só 1 vez iria explodir a matemática do treino)
    pesos = torch.clamp(pesos, min=0.1, max=50.0) 
    
    # Passamos os pesos matemáticos para a função de Loss!
    criterion = nn.CrossEntropyLoss(weight=pesos.to(device))
    # -----------------------------------------------------------------

    # Criar a pasta para guardar o modelo
    os.makedirs(f"models/{args.run_name}", exist_ok=True)
    ckpt_path = f"models/{args.run_name}/dynamics_ckpt.pt"

    # 5. O Loop de Treinamento
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{args.epochs}")
        
        # Agora o dataloader entrega 3 coisas: Visão Atual(x), Visão Futura(y) e Ações(a)
        for x, y, a in pbar:
            x = x.to(device)
            y = y.to(device)
            a = a.to(device)
            
            optimizer.zero_grad()
            
            # O modelo tenta adivinhar o futuro (y) a olhar para o presente (x) e para o botão (a)
            logits = model(x, actions=a) 
            
            # Compara a previsão com o gabarito. Achatamos as matrizes para o PyTorch entender
            loss = criterion(logits.view(-1, 512), y.view(-1))
            
            # Aprende com o erro e atualiza os neurónios
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        # Final da época: Guardar o progresso
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
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    train(args)
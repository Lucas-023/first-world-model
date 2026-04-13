import os
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.encoder.modules import VQVAE
from models.dynamics.dataset import CarRacingSequenceDataset  # O novo dataset
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig # O novo modelo
from models.encoder.utils import setup_logging
from models.encoder.board import Board

def train_gpt(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    # 1. Carregar o VQ-VAE Congelado
    print(f"Carregando VQ-VAE de: {args.vqvae_ckpt}")
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=args.vocab_size).to(device)
    vqvae_checkpoint = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False # Não treinamos o VQ-VAE aqui!

    # 2. DataLoader (agora devolve 4 coisas)
    dataset = CarRacingSequenceDataset(args.dataset_path, seq_len=args.frames_per_seq)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. Inicializar o Modelo GPT (World Model)
    config = WorldModelConfig(
        vocab_size=args.vocab_size,
        tokens_per_frame=64,
        frames_per_seq=args.frames_per_seq
    )
    model = WorldModel(config).to(device)
    
    # Usamos o seu excelente configure_optimizers!
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95), device_type=device)
    scaler = GradScaler()

    board = Board(run_name=args.run_name, enabled=True)
    global_step = 0

    print("🚀 Iniciando Treinamento do World Model...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in pbar:
            # Desempacotar as 4 variáveis e enviar para a GPU
            imagens, acoes, recompensas, dones = [b.to(device) for b in batch]
            B, T_seq, C, H, W = imagens.size()

            # --- PASSO A: EXTRAIR TOKENS DO VQ-VAE ---
            with torch.no_grad():
                # Achatar a dimensão do tempo para passar tudo no VQ-VAE de uma vez
                imgs_flat = imagens.view(-1, C, H, W)
                _, _, indices = vqvae(imgs_flat)
                # Voltar para o formato [Batch, Tempo, 64 tokens]
                img_tokens = indices.view(B, T_seq, -1)

            # --- PASSO B: CONSTRUIR OS ALVOS (TARGETS) ---
            # Tamanho total da sequência: T_seq * 65
            L = T_seq * (config.tokens_per_frame + 1)
            
            # Criamos um tensor cheio de -100 (para ignorar as posições de ação)
            targets_seq = torch.full((B, L), -100, dtype=torch.long, device=device)
            
            # Preenchemos apenas as posições da imagem com os tokens reais
            for t in range(T_seq):
                start_idx = t * (config.tokens_per_frame + 1)
                end_idx = start_idx + config.tokens_per_frame
                targets_seq[:, start_idx:end_idx] = img_tokens[:, t, :]
            
            # O alvo é deslocado 1 posição para a direita (prever o próximo!)
            targets_img = targets_seq[:, 1:]

            # --- PASSO C: FORWARD E BACKPROP ---
            optimizer.zero_grad()
            with autocast():
                logits, loss = model(
                    img_tokens=img_tokens,
                    actions=acoes,
                    targets_img=targets_img,
                    targets_reward=recompensas,
                    targets_done=dones
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- PASSO D: LOGS ---
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            board.log_scalar("Loss/Total", loss.item(), global_step)
            global_step += 1

        # Salvar Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

    board.close()
    print("🎉 Treinamento finalizado com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',       type=str,   default="GPT_CARRACING")
    parser.add_argument('--vqvae_ckpt',     type=str,   required=True)
    parser.add_argument('--dataset_path',   type=str,   required=True)
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=32) # Ajuste de acordo com a sua VRAM
    parser.add_argument('--vocab_size',     type=int,   default=512)
    parser.add_argument('--frames_per_seq', type=int,   default=5)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--device',         type=str,   default="cuda")
    args = parser.parse_args()
    
    train_gpt(args)
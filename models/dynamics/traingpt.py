import os
import torch
import logging
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Ajuste os imports para o caminho das suas pastas
from models.encoder.modules import VQVAE
from models.dynamics.dataset import PongSequenceDataset  # O dataset que fizemos
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig # O modelo GPT que fizemos
from models.encoder.utils import setup_logging
from models.encoder.board import Board

def train_gpt(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    # 1. Diretórios
    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    # 2. Carregar o VQ-VAE (Apenas para criar o cache caso não exista)
    print(f"Carregando VQ-VAE de: {args.vqvae_ckpt}")
    vqvae = VQVAE(in_channels=3, latent_dim=128, num_embeddings=args.vocab_size).to(device)
    vqvae_checkpoint = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    # 3. Dataset e Dataloader (O Dataset vai gerar ou ler o cache)
    dataset = PongSequenceDataset(
        folder_path=args.dataset_path,
        vqvae_model=vqvae,
        frames_per_seq=args.frames_per_seq,
        device=device,
        cache_file=os.path.join(args.dataset_path, "latents_cache.pt")
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 4. Configurar e Instanciar o GPT
    # Matemática do contexto: frames * 64 tokens por frame
    seq_len = args.frames_per_seq * 64 
    
    gpt_config = WorldModelConfig(
        block_size=seq_len,
        vocab_size=args.vocab_size,
        action_vocab_size=args.num_actions,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=0.1
    )
    gpt = WorldModel(gpt_config).to(device)

    # Otimizador com decaimento de peso (Weight Decay)
    optimizer = gpt.configure_optimizers(
        weight_decay=0.1, 
        learning_rate=args.lr, 
        betas=(0.9, 0.95), 
        device_type=device
    )
    scaler = GradScaler()

    board = Board(run_name=args.run_name, enabled=True)
    global_step = 0
    start_epoch = 0

    # Retomar treino se houver checkpoint
    if os.path.exists(ckpt_path):
        print(f"🔄 Retomando treino do checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gpt.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # 5. O Loop de Treinamento
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Iniciando GPT Época {epoch}:")
        gpt.train()
        pbar = tqdm(dataloader, desc=f"Época {epoch}/{args.epochs}")
        epoch_losses = []

        for x_tokens, y_tokens, actions, dones in pbar:
            # Mover dados para a GPU
            x_tokens = x_tokens.to(device)
            y_tokens = y_tokens.to(device)
            actions = actions.to(device)
            dones = dones.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                # O Forward calcula tudo: Loss da imagem (tokens) + Loss do Done
                logits, loss = gpt(
                    idx=x_tokens, 
                    actions=actions, 
                    targets=y_tokens, 
                    targets_done=dones
                )

            scaler.scale(loss).backward()
            
            # Gradient clipping para estabilidade do Transformer
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            board.log_scalar("GPT_Loss/Batch", loss.item(), global_step)
            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n📊 Época {epoch} — Loss Média: {avg_loss:.4f}")
        board.log_scalar("GPT_Loss/Epoca", avg_loss, epoch)

        # 6. Salvar Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": gpt.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Salva um backup histórico a cada 50 épocas
        if epoch % 50 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"gpt_ckpt_epoch_{epoch}.pt"))

    board.close()
    print("Treinamento finalizado com sucesso! 🎉")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',       type=str,   default="GPT_PONG_SEQ", help="Nome do run")
    parser.add_argument('--vqvae_ckpt',     type=str,   required=True,          help="Caminho do VQ-VAE")
    parser.add_argument('--dataset_path',   type=str,   required=True,          help="Pasta com imagens e metadata.csv")
    
    # Hiperparâmetros
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=64,             help="Cuidado com VRAM no block_size de 320")
    parser.add_argument('--vocab_size',     type=int,   default=512,            help="num_embeddings do VQ-VAE")
    parser.add_argument('--num_actions',    type=int,   default=6,              help="Quantas ações diferentes o seu Pong tem?")
    parser.add_argument('--frames_per_seq', type=int,   default=5,              help="Quantos frames de contexto o modelo vai ver")
    parser.add_argument('--lr',             type=float, default=5e-4)
    
    parser.add_argument('--device',         type=str,   default="cuda")
    parser.add_argument('--num_workers',    type=int,   default=8)
    
    args = parser.parse_args()
    train_gpt(args)

if __name__ == '__main__':
    main()
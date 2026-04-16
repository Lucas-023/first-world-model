import os
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.dynamics.dataset import CarRacingTokenDataset 
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.utils import setup_logging
from models.encoder.board import Board

def train_gpt(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    board = Board(args.run_name)

    # 1. DataLoader
    print(f"A carregar dataset de tokens de: {args.dataset_path}")
    dataset = CarRacingTokenDataset(args.dataset_path, seq_len=args.frames_per_seq)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True 
    )

    # 2. Configurar o Modelo GPT
    config = WorldModelConfig(
        vocab_size=args.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=4,
        tokens_per_frame=64,
        frames_per_seq=args.frames_per_seq
    )
    
    print("A inicializar o World Model (Transformer)...")
    model = WorldModel(config).to(device)

    # Otimizador
    optimizer = model.configure_optimizers(
        weight_decay=0.1, 
        learning_rate=3e-4, 
        betas=(0.9, 0.95), 
        device_type=device
    )
    
    scaler = GradScaler()
    
    # --- LÓGICA DE RESUME (CARREGAR CHECKPOINT) ---
    start_epoch = 0
    global_step = 0

    if os.path.exists(ckpt_path):
        print(f"Relançando treino a partir de: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1  # Começa na próxima
        # Estimativa do global_step para o TensorBoard não sobrescrever o gráfico antigo
        global_step = start_epoch * len(dataloader)
        
        print(f"Continuando da Época {start_epoch}")
    else:
        print("Nenhum checkpoint encontrado. Iniciando do zero.")

    # 3. Loop de Treino (ajustado com start_epoch)
    print("🚀 A iniciar o Treino do World Model...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in pbar:
            img_tokens, acoes, recompensas, dones = [b.to(device) for b in batch]
            B, T_seq, N_TOKENS = img_tokens.size()

            # --- CONSTRUIR OS ALVOS ---
            L = T_seq * (config.tokens_per_frame + 1)
            targets_seq = torch.full((B, L), -100, dtype=torch.long, device=device)
            
            for t in range(T_seq):
                start_idx = t * (config.tokens_per_frame + 1)
                end_idx = start_idx + config.tokens_per_frame
                targets_seq[:, start_idx:end_idx] = img_tokens[:, t, :]
            
            targets_img = targets_seq[:, 1:]

            # --- FORWARD E BACKWARD PASS ---
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, (loss, l_img, l_rew, l_done) = model(
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

            # --- LOGS ---
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            board.log_scalar("Loss/Total", loss.item(), global_step)
            board.log_scalar("Loss/Image_Entropy", l_img.item(), global_step)
            board.log_scalar("Loss/Reward_MSE", l_rew.item(), global_step)
            board.log_scalar("Loss/Done_BCE", l_done.item(), global_step)
            
            global_step += 1           
             
        # --- FIM DA ÉPOCA ---
        model.eval()
        board.log_layer_gradients(model, epoch)
        
        # Imaginação
        print(f"✨ Gerando imaginação para a época {epoch}...")
        with torch.no_grad():
            seed_img = img_tokens[0:1, 0:1, :] 
            seed_actions = acoes[0:1, :, :]
            tokens_imaginados = model.generate(seed_img, seed_actions, max_steps=19)
            
            tokens_vis = tokens_imaginados[0].float() / args.vocab_size
            board.log_image("Imagination/Token_Map", tokens_vis.unsqueeze(0), epoch)
        
        # Guardar Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

    board.close()
    print("Treino finalizado com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',       type=str,   default="GPT_CARRACING")
    parser.add_argument('--dataset_path',   type=str,   required=True)
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--vocab_size',     type=int,   default=512)
    parser.add_argument('--frames_per_seq', type=int,   default=20)
    parser.add_argument('--device',         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train_gpt(args)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from models.encoder.dataset import CarRacingDataset
from models.encoder.board import Board
from models.encoder.utils import save_images, setup_logging
from models.encoder.modules import VQVAE  # ajuste o caminho se necessário


def train(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    # Diretórios
    save_dir     = os.path.join("models",   args.run_name)
    results_dir  = os.path.join("results",  args.run_name)
    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "ckpt.pt")

    # Dataset & Dataloader
    dataset    = CarRacingDataset(args.dataset_path, max_files=200)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers = False if os.name == 'nt' else True,  # Windows tem problemas com workers persistentes
    )

    # Modelo
    model = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
    ).to(device)

    # Otimizador + loss + scaler
    optimizer    = optim.AdamW(model.parameters(), lr=args.lr)
    recon_loss_fn = nn.MSELoss()          # ou BCELoss se preferir pixel-wise
    scaler = torch.amp.GradScaler('cuda')


    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"🔄 Checkpoint encontrado em: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Treino retomado da época {start_epoch}")

    # TensorBoard
    board       = Board(run_name=args.run_name, enabled=True)
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        model.train()
        pbar         = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_losses = []

        for images in pbar:
            images = images.to(device)          
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):

                x_recon, vq_loss, _ = model(images)

                # Loss total = reconstrução + penalidade do codebook
                recon_loss = recon_loss_fn(x_recon, images)
                loss       = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # ── Logging por batch ──────────────────────────────────
            epoch_losses.append(loss.item())
            pbar.set_postfix(
                Loss=f"{loss.item():.5f}",
                Recon=f"{recon_loss.item():.5f}",
                VQ=f"{vq_loss.item():.5f}",
            )
            board.log_scalar("Loss/Batch",       loss.item(),        global_step)
            board.log_scalar("Loss/Recon_Batch",  recon_loss.item(), global_step)
            board.log_scalar("Loss/VQ_Batch",     vq_loss.item(),    global_step)
            global_step += 1

        # ── Logging por época ──────────────────────────────────────
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n📊 Época {epoch} — Loss Médio: {avg_loss:.6f}")
        board.log_scalar("Loss/Epoca", avg_loss, epoch)

        #if epoch % 10 == 0:
            #board.log_layer_gradients(model, epoch)

        # ── Salvar checkpoint ──────────────────────────────────────
        checkpoint = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "loss":                 avg_loss,
        }
        torch.save(checkpoint, ckpt_path)

        # ── Imagens de teste a cada 25 épocas ─────────────────────
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print("🎨 Gerando imagens de reconstrução...")
            model.eval()
            with torch.no_grad():
                # Pega um batch fixo do dataloader para visualizar
                sample_images = next(iter(dataloader))[:16].to(device)
                x_recon, _, _ = model(sample_images)

            # Salva lado a lado: original | reconstruída
            comparison = torch.cat([sample_images[:8], x_recon[:8]], dim=0)
            save_images(comparison, os.path.join(results_dir, f"{epoch}.jpg"))
            torch.save(checkpoint, os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt"))

            # TensorBoard: grid de reconstruções
            grid = make_grid(comparison, nrow=8, normalize=True, value_range=(0, 1))
            #board.log_image("Reconstrucao/Teste", grid, epoch)

    board.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',     type=str,   default="VQVAE_PONG",  help="Nome do run")
    parser.add_argument('--epochs',       type=int,   default=400,           help="Total de épocas")
    parser.add_argument('--batch_size',   type=int,   default=512,            help="Batch size")
    parser.add_argument('--image_size',   type=int,   default=64,            help="Resolução da imagem")
    parser.add_argument('--latent_dim',   type=int,   default=128,           help="Dimensão do espaço latente")
    parser.add_argument('--num_embeddings', type=int, default=512,           help="Tamanho do codebook")
    parser.add_argument('--lr',           type=float, default=2e-4,          help="Learning rate")
    parser.add_argument('--device',       type=str,   default="cuda",        help="Device (cuda/cpu)")
    parser.add_argument('--dataset_path', type=str,   required=True,         help="Caminho para a pasta com os .png")
    parser.add_argument('--num_workers',  type=int,   default=8,             help="Workers do DataLoader")
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
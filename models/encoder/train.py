import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import logging

from tqdm import tqdm

import argparse

from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision.utils import save_image

from models.encoder.dataset import CarRacingDataset
from models.encoder.modules import VQVAE


def setup_logging(run_name):

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S"
    )


def save_images(images, path):

    images = images.clamp(0, 1)

    save_image(images, path)


def train(args):

    setup_logging(args.run_name)

    device = args.device

    torch.backends.cudnn.benchmark = True

    save_dir = os.path.join(
        "models",
        args.run_name
    )

    results_dir = os.path.join(
        "results",
        args.run_name
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(
        save_dir,
        "ckpt.pt"
    )

    dataset = CarRacingDataset(
        args.dataset_path,
        max_files=args.max_files
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    model = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95)
    )

    recon_loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0

    if os.path.exists(ckpt_path):

        print(f"\n🔄 Checkpoint encontrado: {ckpt_path}")

        checkpoint = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"✅ Retomando da época {start_epoch}"
        )

    for epoch in range(start_epoch, args.epochs):

        model.train()

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}"
        )

        losses = []

        for images in pbar:

            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):

                x_recon, vq_loss, _ = model(images)

                recon_loss = recon_loss_fn(
                    x_recon,
                    images
                )

                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            losses.append(loss.item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "vq": f"{vq_loss.item():.4f}"
            })

        avg_loss = sum(losses) / len(losses)

        print(
            f"\n📊 Epoch {epoch} "
            f"| loss={avg_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict()
        }

        torch.save(
            checkpoint,
            ckpt_path
        )

        if epoch % 5 == 0:

            model.eval()

            with torch.no_grad():

                sample_images = next(
                    iter(dataloader)
                )[:8].to(device)

                x_recon, _, _ = model(sample_images)

                sample_vis = sample_images.clamp(0, 1)

                recon_vis = x_recon.clamp(0, 1)

                comparison = torch.cat(
                    [sample_vis, recon_vis],
                    dim=0
                )

                save_images(
                    comparison,
                    os.path.join(
                        results_dir,
                        f"{epoch}.png"
                    )
                )

    print("\n✅ Treino finalizado")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default="VQVAE"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=400
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=512
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
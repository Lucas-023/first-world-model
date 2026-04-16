import os
import logging
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from models.encoder.dataset import CarRacingDataset  # dataset.py já existe no seu encoder/


def setup_logging(run_name):
    """Cria os diretórios necessários e configura o logger."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_data(args):
    """
    Cria o DataLoader a partir do PongDataset.
    Normaliza as imagens para [-1, 1] para ficar consistente
    com o save_images e facilitar treinos futuros (ex: difusão latente).
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()                              # [0, 1]
    ])

    dataset = CarRacingDataset(args.dataset_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


def save_images(images, path, **kwargs):
    # Imagens já estão em [0, 1], só converte para uint8
    images = (images.clamp(0, 1) * 255).type(torch.uint8)
    grid   = torchvision.utils.make_grid(images, **kwargs)
    ndarr  = grid.permute(1, 2, 0).to("cpu").numpy()
    im     = Image.fromarray(ndarr)
    im.save(path)


def save_reconstruction_grid(originals, reconstructions, path, n=8):
    """
    Salva um grid comparativo: linha de cima = originais, linha de baixo = reconstruções.
    Útil para monitorar a qualidade do VQVAE visualmente.

    Args:
        originals:       tensor [B, C, H, W] em [-1, 1]
        reconstructions: tensor [B, C, H, W] em [-1, 1] (saída do decoder)
        path:            caminho para salvar a imagem
        n:               quantas amostras mostrar (pega as primeiras n)
    """
    originals       = originals[:n].detach().cpu()
    reconstructions = reconstructions[:n].detach().cpu()

    # Empilha: [orig1, orig2, ..., recon1, recon2, ...]
    comparison = torch.cat([originals, reconstructions], dim=0)
    save_images(comparison, path, nrow=n)
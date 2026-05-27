"""
Treino do Dynamics Model com validacao periodica.

Uso:
    python train_dynamics.py --dataset_path dataset_tokens
    python train_dynamics.py --dataset_path dataset_tokens --overfit_test
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from gptimage   import DynamicsModel, DynamicsConfig
from datasetimage import DynamicsDataset


def evaluate(model, dataloader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for obs_ctx, act_ctx, obs_target in dataloader:
            obs_ctx    = obs_ctx.to(device)
            act_ctx    = act_ctx.to(device)
            obs_target = obs_target.to(device)
            total += model.compute_loss(obs_ctx, act_ctx, obs_target).item()
            n     += 1
    return total / max(n, 1)


def save_dream(model, vqvae, obs_ctx, act_ctx, save_path, device, n_frames=20):
    model.eval()
    with torch.no_grad():
        ctx_obs = obs_ctx[0:1]    # (1, context_len, 64)
        ctx_act = act_ctx[0:1]    # (1, context_len)
        frames  = []

        for i in range(n_frames):
            next_act = act_ctx[0:1, min(i, act_ctx.shape[1]-1)]
            next_obs = model.imagine_next_frame(ctx_obs, ctx_act, next_act)

            frames.append(next_obs)
            # Desliza a janela: remove o frame mais antigo, adiciona o novo
            ctx_obs = torch.cat([ctx_obs[:, 1:, :], next_obs.unsqueeze(1)], dim=1)
            ctx_act = torch.cat([ctx_act[:, 1:],    next_act.unsqueeze(1)], dim=1)

        all_tok = torch.cat(frames, dim=0)         # (N, 64)
        decoded = vqvae.decode_indices(all_tok)     # (N, 3, 64, 64)
        grid    = make_grid(decoded.cpu(), nrow=10, normalize=True, value_range=(0,1))
        save_image(grid, save_path)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    img_dir   = os.path.join(args.save_dir, "dreams")
    os.makedirs(img_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "dynamics_ckpt.pt")
    device    = args.device

    # -- Config ---------------------------------------------------------------
    config = DynamicsConfig(
        obs_vocab_size = args.vocab_size,
        act_vocab_size = 5,
        img_tokens     = 64,
        context_len    = args.context_len,
        n_embd         = args.n_embd,
        n_head         = args.n_head,
        n_layer        = args.n_layer,
        dropout        = args.dropout,
    )
    print(f"Contexto         : {config.context_len} frames")
    print(f"Tokens por bloco : {config.tokens_per_block}")
    print(f"Block size       : {config.block_size}")

    # -- Datasets -------------------------------------------------------------
    train_ds = DynamicsDataset(args.dataset_path, split="train",
                               context_len=args.context_len, seed=args.seed)
    val_ds   = DynamicsDataset(args.dataset_path, split="val",
                               context_len=args.context_len, seed=args.seed)
    test_ds  = DynamicsDataset(args.dataset_path, split="test",
                               context_len=args.context_len, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # -- Modelo ---------------------------------------------------------------
    device_type = device.split(":")[0]   # "cuda" ou "cpu"
    model     = DynamicsModel(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr)
    scaler    = GradScaler(device_type)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros: {n_params:,}")

    # -- Checkpoint -----------------------------------------------------------
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Continuando da epoca {start_epoch} | melhor val: {best_val_loss:.4f}")
    else:
        print("Iniciando do zero.")

    # -- Overfit test ---------------------------------------------------------
    if args.overfit_test:
        print("\nModo OVERFIT (sanity check)...")
        model.train()
        obs_ctx, act_ctx, obs_target = [b.to(device) for b in next(iter(train_loader))]

        for step in range(3000):
            loss = model.compute_loss(obs_ctx, act_ctx, obs_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(f"  step {step:>4} | loss {loss.item():.4f}")

        print("Overfit test concluido. Esperado: loss < 1.0")
        return

    # -- VQ-VAE para visualizacao ---------------------------------------------
    vqvae = None
    if os.path.exists(args.vqvae_path):
        try:
            from models.encoder.modules import VQVAE
            vqvae = VQVAE(in_channels=3, latent_dim=256,
                          num_embeddings=args.vocab_size).to(device)
            vqvae.load_state_dict(
                torch.load(args.vqvae_path, map_location=device,
                           weights_only=True)["model_state_dict"]
            )
            vqvae.eval()
            print("VQ-VAE carregado.")
        except Exception as e:
            print(f"Aviso VQ-VAE: {e}")

    # -- Loop principal -------------------------------------------------------
    print(f"\nIniciando treino...")
    for epoch in range(start_epoch, args.epochs):

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        last_batch = None

        for obs_ctx, act_ctx, obs_target in pbar:
            obs_ctx    = obs_ctx.to(device)
            act_ctx    = act_ctx.to(device)
            obs_target = obs_target.to(device)
            last_batch = (obs_ctx, act_ctx)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type):
                loss = model.compute_loss(obs_ctx, act_ctx, obs_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(train=f"{loss.item():.4f}")
            global_step += 1

        # -- Validacao --------------------------------------------------------
        val_loss = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:>5}]  val: {val_loss:.4f}  (melhor: {best_val_loss:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "config":           config.__dict__,
            }, os.path.join(args.save_dir, "dynamics_best.pt"))
            print(f"  -> Novo melhor modelo! val: {val_loss:.4f}")

        # -- Visualizacao a cada 5 epocas -------------------------------------
        if epoch % 5 == 0 and vqvae is not None and last_batch is not None:
            try:
                obs_ctx, act_ctx = last_batch
                save_path = os.path.join(img_dir, f"dream_{epoch:05d}.png")
                save_dream(model, vqvae, obs_ctx, act_ctx, save_path, device)
                print(f"  -> Imagem: {save_path}")
            except Exception as e:
                print(f"  Aviso viz: {e}")

        # -- Checkpoint -------------------------------------------------------
        torch.save({
            "epoch":                epoch,
            "global_step":          global_step,
            "best_val_loss":        best_val_loss,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    # -- Test set final -------------------------------------------------------
    print("\nAvaliacao no test set...")
    best = torch.load(os.path.join(args.save_dir, "dynamics_best.pt"),
                      map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss (epoca {best['epoch']}): {test_loss:.4f}")
    print("Treino finalizado!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",   type=str,   required=True)
    p.add_argument("--save_dir",       type=str,   default="models/dynamics")
    p.add_argument("--vqvae_path",     type=str,   default="models/VQVAE/ckpt.pt")
    p.add_argument("--epochs",         type=int,   default=5000)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--vocab_size",     type=int,   default=512)
    p.add_argument("--context_len",    type=int,   default=19)
    p.add_argument("--n_embd",         type=int,   default=256)
    p.add_argument("--n_head",         type=int,   default=4)
    p.add_argument("--n_layer",        type=int,   default=6)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--overfit_test",   action="store_true")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)
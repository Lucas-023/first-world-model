"""
Treino do Dynamics GPT (imagem + reward + done), com validacao periodica.

Uso:
    python -m models.dynamics.traingpt --dataset_path dataset_tokens
    python -m models.dynamics.traingpt --dataset_path dataset_tokens --overfit_test
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.dynamics.dataset import CarRacingTokenDataset
from models.encoder.board import Board


REWARD_LABEL = {0: "neg (-1)", 1: "neutro (0)", 2: "pos (+1)"}


def evaluate(model, dataloader, device):
    model.eval()
    device_type = device.split(":")[0]
    tot = {"loss": 0.0, "obs": 0.0, "rew": 0.0, "done": 0.0}
    n = 0
    with torch.no_grad():
        for obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt in dataloader:
            obs_ctx = obs_ctx.to(device)
            act_ctx = act_ctx.to(device)
            obs_tgt = obs_tgt.to(device)
            rew_tgt = rew_tgt.to(device)
            done_tgt = done_tgt.to(device)
            with autocast(device_type=device_type):
                loss, l_obs, l_rew, l_done = model.compute_loss(obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt)
            tot["loss"] += loss.item()
            tot["obs"] += l_obs.item()
            tot["rew"] += l_rew.item()
            tot["done"] += l_done.item()
            n += 1
    if n == 0:
        return {k: 0.0 for k in tot}
    return {k: v / n for k, v in tot.items()}


def save_dream(model, vqvae, obs_ctx, act_ctx, save_path, device, n_frames=20, board=None, step=0):
    """Sonha n_frames a partir do contexto, salva o grid de imagens decodificadas
    pelo VQ-VAE (e loga no TensorBoard, se um Board for passado) e imprime
    a sequencia de reward/done previstos junto."""
    model.eval()
    with torch.no_grad():
        ctx_obs = obs_ctx[0:1]    # (1, context_len, 64)
        ctx_act = act_ctx[0:1]    # (1, context_len)
        frames, rewards, dones = [], [], []

        for i in range(n_frames):
            # acao aleatoria a cada passo -- antes repetia pra sempre a ultima
            # acao real do contexto, o que so testava "segurar um botao"
            next_act = torch.randint(0, model.config.act_vocab_size, (1,), device=device)
            next_obs, rew_cls, done_cls = model.imagine_next_frame(ctx_obs, ctx_act, next_act)

            frames.append(next_obs)
            rewards.append(rew_cls.item())
            dones.append(done_cls.item())

            ctx_obs = torch.cat([ctx_obs[:, 1:, :], next_obs.unsqueeze(1)], dim=1)
            ctx_act = torch.cat([ctx_act[:, 1:], next_act.unsqueeze(1)], dim=1)

        all_tok = torch.cat(frames, dim=0)         # (N, 64)
        decoded = vqvae.decode_indices(all_tok)    # (N, 3, 64, 64)
        grid = make_grid(decoded.cpu(), nrow=10, normalize=True, value_range=(0, 1))
        save_image(grid, save_path)

        if board is not None:
            board.log_image("dream/frames", grid, step)

        print("  -> Sonho (reward/done previstos):")
        print("     reward:", [REWARD_LABEL[r] for r in rewards])
        print("     done  :", dones)


def train_gpt(args):
    os.makedirs(args.save_dir, exist_ok=True)
    img_dir = os.path.join(args.save_dir, "dreams", args.run_name)
    os.makedirs(img_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "gpt_ckpt.pt")
    best_path = os.path.join(args.save_dir, "gpt_best.pt")
    device = args.device
    device_type = device.split(":")[0]

    board = Board(args.run_name)
    print(f"TensorBoard: tensorboard --logdir runs/{args.run_name}")

    config = WorldModelConfig(
        obs_vocab_size=args.vocab_size,
        act_vocab_size=5,
        img_tokens=64,
        context_len=args.context_len,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    print(f"Contexto         : {config.context_len} frames")
    print(f"Tokens por bloco : {config.tokens_per_block}")
    print(f"Block size       : {config.block_size}")

    train_ds = CarRacingTokenDataset(args.dataset_path, split="train", context_len=args.context_len, seed=args.seed, stride=args.stride)
    val_ds = CarRacingTokenDataset(args.dataset_path, split="val", context_len=args.context_len, seed=args.seed)
    test_ds = CarRacingTokenDataset(args.dataset_path, split="test", context_len=args.context_len, seed=args.seed)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = WorldModel(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=args.lr)
    scaler = GradScaler(device_type)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros: {n_params:,}")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"Continuando da epoca {start_epoch} | melhor val: {best_val:.4f}")
    else:
        print("Iniciando do zero.")

    if args.overfit_test:
        print("\nModo OVERFIT (sanity check)...")
        model.train()
        obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt = [b.to(device) for b in next(iter(train_loader))]
        for step in range(3000):
            loss, l_obs, l_rew, l_done = model.compute_loss(obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(
                    f"  step {step:>4} | total {loss.item():.4f} | "
                    f"obs {l_obs.item():.4f} | rew {l_rew.item():.4f} | done {l_done.item():.4f}"
                )
        print("Overfit test concluido. Esperado: obs/rew/done proximos de 0.")
        board.close()
        return

    vqvae = None
    if os.path.exists(args.vqvae_path):
        try:
            from models.encoder.modules import VQVAE
            vqvae = VQVAE(in_channels=3, latent_dim=256, num_embeddings=args.vocab_size).to(device)
            vqvae.load_state_dict(
                torch.load(args.vqvae_path, map_location=device, weights_only=True)["model_state_dict"]
            )
            vqvae.eval()
            print("VQ-VAE carregado.")
        except Exception as e:
            print(f"Aviso VQ-VAE: {e}")

    print("\nIniciando treino...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        last_batch = None

        for obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt in pbar:
            obs_ctx = obs_ctx.to(device)
            act_ctx = act_ctx.to(device)
            obs_tgt = obs_tgt.to(device)
            rew_tgt = rew_tgt.to(device)
            done_tgt = done_tgt.to(device)
            last_batch = (obs_ctx, act_ctx)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type):
                loss, l_obs, l_rew, l_done = model.compute_loss(obs_ctx, act_ctx, obs_tgt, rew_tgt, done_tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                obs=f"{l_obs.item():.4f}",
                rew=f"{l_rew.item():.4f}",
                done=f"{l_done.item():.4f}",
            )
            board.log_scalar("train/total", loss.item(), global_step)
            board.log_scalar("train/obs", l_obs.item(), global_step)
            board.log_scalar("train/rew", l_rew.item(), global_step)
            board.log_scalar("train/done", l_done.item(), global_step)
            global_step += 1

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch:>5}] val total {val_metrics['loss']:.4f} | "
            f"obs {val_metrics['obs']:.4f} | rew {val_metrics['rew']:.4f} | done {val_metrics['done']:.4f}"
        )
        board.log_scalar("val/total", val_metrics["loss"], epoch)
        board.log_scalar("val/obs", val_metrics["obs"], epoch)
        board.log_scalar("val/rew", val_metrics["rew"], epoch)
        board.log_scalar("val/done", val_metrics["done"], epoch)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val,
                    "config": config.__dict__,
                },
                best_path,
            )
            print(f"  -> Novo melhor modelo ({best_val:.4f})")

        if epoch % 5 == 0 and vqvae is not None and last_batch is not None:
            try:
                obs_ctx, act_ctx = last_batch
                save_path = os.path.join(img_dir, f"dream_{epoch:05d}.png")
                save_dream(model, vqvae, obs_ctx, act_ctx, save_path, device, board=board, step=epoch)
                print(f"  -> Imagem: {save_path}")
            except Exception as e:
                print(f"  Aviso viz: {e}")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )

    print("\nAvaliacao no test set...")
    if os.path.exists(best_path):
        best = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test total {test_metrics['loss']:.4f} | obs {test_metrics['obs']:.4f} | "
        f"rew {test_metrics['rew']:.4f} | done {test_metrics['done']:.4f}"
    )
    print("Treino finalizado!")
    board.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="models/dynamics")
    p.add_argument("--run_name", type=str, default="DYNAMICS_GPT")
    p.add_argument("--vqvae_path", type=str, default="models/VQVAE/ckpt.pt")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--context_len", type=int, default=19)
    p.add_argument("--stride", type=int, default=1, help="passo da janela deslizante no split de treino (val/test sempre stride=1)")
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit_test", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train_gpt(args)

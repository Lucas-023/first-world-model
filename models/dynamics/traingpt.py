"""

Uso:
    python -m models.dynamics.traingpt --dataset_path dataset_tokens
    python -m models.dynamics.traingpt --dataset_path dataset_tokens --overfit_test
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.utils import setup_logging
from models.encoder.board import Board
from models.dynamics.dataset import CarRacingTokenDataset




def train_gpt(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    save_dir  = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    board = Board(args.run_name)

    # Config
    config = WorldModelConfig(
        obs_vocab_size = args.vocab_size,
        act_vocab_size = 5,
        img_tokens     = 16,
        frames_per_seq = args.frames_per_seq,
        n_embd         = args.n_embd,
        n_head         = args.n_head,
        n_layer        = args.n_layer,
        dropout        = args.dropout,
    )
    print(f"Tokens por bloco  : {config.tokens_per_block}")
    print(f"Block size        : {config.block_size}")
    print(f"Heads             : obs({config.obs_vocab_size}) | reward(3) | done(2)")

    # Dataset
    dataset    = CarRacingTokenDataset(args.dataset_path, seq_len=args.frames_per_seq)
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
    )

    # Modelo
    model     = WorldModel(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95)
    )
    scaler = GradScaler()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {n_params:,}")

    # Checkpoint
    start_epoch = 0
    global_step = 0
    if os.path.exists(ckpt_path):
        print(f"Retomando de: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * len(dataloader))
        print(f"Continuando da epoca {start_epoch}")
    else:
        print("Iniciando do zero.")

    # Overfit test
    if args.overfit_test:
        print("\nModo OVERFIT em 1 batch (sanity check)...")
        model.train()
        obs_tok, act_tok, rew_sign, dones = [b.to(device) for b in next(iter(dataloader))]

        for step in range(3000):
            loss, l_obs, l_rew, l_end = model.compute_loss(obs_tok, act_tok, rew_sign, dones)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  step {step:>4} | loss {loss.item():.4f} "
                      f"(obs {l_obs.item():.4f} | rew {l_rew.item():.4f} | end {l_end.item():.4f})")

        print("Overfit test concluido.")
        return

    # Loop principal
    print(f"\nIniciando treino...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar          = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        last_obs_tok  = None
        last_act_tok  = None

        for batch in pbar:
            obs_tok, act_tok, rew_sign, dones = [b.to(device) for b in batch]
            last_obs_tok = obs_tok
            last_act_tok = act_tok

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                loss, l_obs, l_rew, l_end = model.compute_loss(
                    obs_tok, act_tok, rew_sign, dones
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                Obs=f"{l_obs.item():.4f}",
                Rew=f"{l_rew.item():.4f}",
            )
            board.log_scalar("Loss/Total",   loss.item(),  global_step)
            board.log_scalar("Loss/Obs",     l_obs.item(), global_step)
            board.log_scalar("Loss/Reward",  l_rew.item(), global_step)
            board.log_scalar("Loss/Done",    l_end.item(), global_step)
            global_step += 1

        # Visualizacao
        model.eval()
        if (epoch % 5 == 0 or epoch == start_epoch) and last_obs_tok is not None:
            try:
                from models.encoder.modules import VQVAE
                vqvae_path = "models/VQVAE/ckpt.pt"
                if os.path.exists(vqvae_path):
                    vqvae = VQVAE(
                        in_channels=3, latent_dim=128,
                        num_embeddings=args.vocab_size
                    ).to(device)
                    vqvae_ckpt = torch.load(vqvae_path, map_location=device, weights_only=True)
                    vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
                    vqvae.eval()

                    with torch.no_grad():
                        # Usa primeiro sample do batch como contexto
                        seed_obs = last_obs_tok[0:1, :10, :]  # (1, 10, 64)
                        seed_act = last_act_tok[0:1, :10]     # (1, 10)

                        dream_frames = []
                        ctx_obs = seed_obs
                        ctx_act = seed_act

                        # Imagina 20 frames autorregressivamente
                        for _ in range(20):
                            # Usa acao 0 para imaginar
                            next_act = torch.zeros(1, dtype=torch.long, device=device)
                            next_obs, _, done = model.imagine_next_frame(
                                ctx_obs, next_act, temperature=1.0, top_k=50
                            )
                            dream_frames.append(next_obs)
                            ctx_obs = torch.cat([ctx_obs, next_obs.unsqueeze(1)], dim=1)
                            ctx_act = torch.cat([ctx_act, next_act.unsqueeze(0).unsqueeze(0)], dim=1)
                            if done.item():
                                break

                        if dream_frames:
                            all_tokens = torch.cat(dream_frames, dim=0)  # (N, 64)
                            decoded    = vqvae.decode_indices(all_tokens)  # (N, 3, 64, 64)
                            grid = make_grid(decoded.cpu(), nrow=10, normalize=True, value_range=(0, 1))
                            board.log_image("Imagination/Dream", grid, epoch)
                            print(f"[Epoch {epoch}] {len(dream_frames)} frames imaginados gerados.")

                    del vqvae
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Aviso: falha na visualizacao: {e}")

        board.log_layer_gradients(model, epoch)

        torch.save({
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    board.close()
    print("Treino finalizado!")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",       type=str,   default="GPT_CARRACING")
    p.add_argument("--dataset_path",   type=str,   required=True)
    p.add_argument("--epochs",         type=int,   default=15000)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--vocab_size",     type=int,   default=512)
    p.add_argument("--frames_per_seq", type=int,   default=20)
    p.add_argument("--n_embd",         type=int,   default=512)
    p.add_argument("--n_head",         type=int,   default=8)
    p.add_argument("--n_layer",        type=int,   default=8)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--overfit_test",   action="store_true")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train_gpt(args)
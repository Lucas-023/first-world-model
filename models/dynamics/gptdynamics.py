"""
Treino do World Model (Transformer estilo IRIS) para CarRacing-v3.

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarRacingTokenDataset(Dataset):
    """
    Carrega episodios tokenizados e devolve subsequencias de comprimento fixo.

    Cada item retornado:
        img_tokens : (seq_len, 64)  int64
        actions    : (seq_len,)     int64
        rewards    : (seq_len,)     int64  — JA DISCRETIZADOS em extract_tokens.py
        dones      : (seq_len,)     int64
    """

    def __init__(self, folder: str, seq_len: int = 20):
        self.seq_len = seq_len
        self.samples = []   # lista de (tokens, actions, rewards, dones)

        files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
        print(f"Carregando {len(files)} episodios de tokens para RAM...")

        for f in files:
            d       = np.load(os.path.join(folder, f))
            tokens  = d["tokens"].astype(np.int64)   # (T, 64)
            actions = d["actions"].astype(np.int64)  # (T,)
            rewards = d["rewards"].astype(np.int64)  # (T,) — ja discretizado
            dones   = d["dones"].astype(np.int64)    # (T,)

            T = len(actions)
            if T < seq_len:
                continue

            # Divide o episodio em janelas sem sobreposicao
            for start in range(0, T - seq_len + 1, seq_len):
                self.samples.append((
                    tokens [start : start + seq_len],
                    actions[start : start + seq_len],
                    rewards[start : start + seq_len],
                    dones  [start : start + seq_len],
                ))

        print(f"Total de sequencias: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tok, act, rew, don = self.samples[idx]
        return (
            torch.from_numpy(tok),
            torch.from_numpy(act),
            torch.from_numpy(rew),
            torch.from_numpy(don),
        )


# ---------------------------------------------------------------------------
# Montagem da sequencia IRIS
# ---------------------------------------------------------------------------

def build_sequence(img_tokens, actions, rewards, dones, config: WorldModelConfig):
    """
    Monta a sequencia plana para o transformer.

    Por frame: [z_0..z_63 | a | r | d]  = 67 tokens
    Cada token e deslocado pelo offset do seu tipo no vocabulario unificado.

    Args:
        img_tokens : (B, T, 64)  int64  — indices do codebook [0, vocab_img)
        actions    : (B, T)      int64  — [0, vocab_action)
        rewards    : (B, T)      int64  — ja discretizados [0, vocab_reward)
        dones      : (B, T)      int64  — 0 ou 1

    Returns:
        seq : (B, T*67) int64
    """
    B, T = actions.shape

    # Aplica offsets para colocar cada tipo no seu intervalo do vocabulario
    img_off = img_tokens + config.img_offset    # (B, T, 64)
    act_off = actions.unsqueeze(2) + config.act_offset   # (B, T, 1)
    rew_off = rewards.unsqueeze(2) + config.rew_offset   # (B, T, 1)
    don_off = dones.unsqueeze(2)   + config.done_offset  # (B, T, 1)

    # Concatena: (B, T, 67)
    frame = torch.cat([img_off, act_off, rew_off, don_off], dim=2)

    # Achata para (B, T*67)
    return frame.view(B, T * config.tokens_per_frame)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_gpt(args):
    setup_logging(args.run_name)
    device = args.device
    torch.backends.cudnn.benchmark = True

    save_dir  = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    board = Board(args.run_name)

    # -- Config ---------------------------------------------------------------
    config = WorldModelConfig(
        vocab_img        = args.vocab_size,
        vocab_action     = 5,
        vocab_reward     = 21,
        tokens_per_frame = 64,
        frames_per_seq   = args.frames_per_seq,
        n_embd           = args.n_embd,
        n_head           = args.n_head,
        n_layer          = args.n_layer,
        dropout          = args.dropout,
    )
    print(f"Vocabulario total : {config.vocab_total}")
    print(f"Tokens por frame  : {config.tokens_per_frame}")
    print(f"Block size        : {config.block_size}")

    # -- Dataset --------------------------------------------------------------
    dataset    = CarRacingTokenDataset(args.dataset_path, seq_len=args.frames_per_seq)
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,      # 0 no Windows para evitar erro de pickle
        pin_memory  = True,
    )

    # -- Modelo ---------------------------------------------------------------
    model     = WorldModel(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95)
    )
    scaler    = GradScaler()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {n_params:,}")

    # -- Carrega checkpoint ---------------------------------------------------
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

    # -- Overfit test ---------------------------------------------------------
    if args.overfit_test:
        print("Modo OVERFIT em 1 batch (sanity check)...")
        model.train()
        batch = next(iter(dataloader))
        img_tokens, actions, rewards, dones = [b.to(device) for b in batch]

        for step in range(300):
            seq    = build_sequence(img_tokens, actions, rewards, dones, config)
            inputs  = seq[:, :-1]   # (B, T*67 - 1)
            targets = seq[:, 1:]    # (B, T*67 - 1) — proximo token para cada posicao

            logits, loss = model(idx=inputs, targets=targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  step {step:>3} | loss {loss.item():.4f}")

        print("Overfit test concluido. Se a loss caiu para < 1.0, o modelo esta aprendendo.")
        return

    # -- Loop principal -------------------------------------------------------
    print(f"\nIniciando treino do World Model...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar      = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        last_seq  = None

        for batch in pbar:
            img_tokens, actions, rewards, dones = [b.to(device) for b in batch]
            seq     = build_sequence(img_tokens, actions, rewards, dones, config)
            last_seq = seq

            inputs  = seq[:, :-1]   # (B, T*67 - 1)
            targets = seq[:, 1:]    # (B, T*67 - 1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                logits, loss = model(idx=inputs, targets=targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            board.log_scalar("Loss/Total", loss.item(), global_step)
            global_step += 1

        # -- Visualizacao a cada 5 epocas -------------------------------------
        model.eval()
        if (epoch % 5 == 0 or epoch == start_epoch) and last_seq is not None:
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
                        # Usa os primeiros 10 frames como seed e gera mais 20
                        seed_len  = 10 * config.tokens_per_frame
                        seed_seq  = last_seq[0:1, :seed_len]
                        generated = model.generate(seed_seq, max_new_tokens=20 * config.tokens_per_frame)

                        T_gen    = generated.shape[1]
                        n_frames = T_gen // config.tokens_per_frame
                        reshaped = generated[0, :n_frames * config.tokens_per_frame].view(
                            n_frames, config.tokens_per_frame
                        )
                        # Extrai so os 64 tokens visuais e remove offset
                        img_tok = reshaped[:, :64] - config.img_offset
                        img_tok = torch.clamp(img_tok, 0, config.vocab_img - 1)

                        dream = vqvae.decode_indices(img_tok)  # (n_frames, 3, 64, 64)
                        grid  = make_grid(dream.cpu(), nrow=10, normalize=True, value_range=(0, 1))
                        board.log_image("Imagination/Dream", grid, epoch)

                    del vqvae
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Aviso: falha na visualizacao: {e}")

        board.log_layer_gradients(model, epoch)

        # -- Checkpoint -------------------------------------------------------
        torch.save({
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    board.close()
    print("Treino finalizado!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",       type=str,   default="GPT_CARRACING")
    p.add_argument("--dataset_path",   type=str,   required=True)
    p.add_argument("--epochs",         type=int,   default=10000)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--vocab_size",     type=int,   default=512)
    p.add_argument("--frames_per_seq", type=int,   default=20)
    p.add_argument("--n_embd",         type=int,   default=256)
    p.add_argument("--n_head",         type=int,   default=8)
    p.add_argument("--n_layer",        type=int,   default=6)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--overfit_test",   action="store_true")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train_gpt(args)
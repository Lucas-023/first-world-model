import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.dynamics.dataset import CarRacingTokenDataset
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig
from models.encoder.utils import setup_logging
from models.encoder.board import Board


def build_sequence(img_tokens, actions, rewards, dones, config):
    """
    Sequência IRIS:
    [z(64), a(3), r(1), d(1)] por frame
    """

    B, T_seq, _ = img_tokens.size()

    IMG_OFFSET = 0
    REW_OFFSET = config.vocab_img
    DONE_OFFSET = REW_OFFSET + config.vocab_reward
    ACT_OFFSET = DONE_OFFSET + config.vocab_done

    seq_list = []

    for t in range(T_seq):
        z = img_tokens[:, t, :]                  # [B, 64]
        a = actions[:, t, :] + ACT_OFFSET        # [B, 3]
        r = rewards[:, t] + REW_OFFSET           # [B]
        d = dones[:, t] + DONE_OFFSET            # [B]

        frame = torch.cat([
            z,
            a,
            r.unsqueeze(1),
            d.unsqueeze(1)
        ], dim=1)  # [B, 69]

        seq_list.append(frame)

    return torch.cat(seq_list, dim=1)  # [B, T_total]


def train_gpt(args):
    setup_logging(args.run_name)

    device = args.device
    torch.backends.cudnn.benchmark = True

    save_dir = os.path.join("models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "gpt_ckpt.pt")

    board = Board(args.run_name)

    # =========================
    # DATASET
    # =========================
    print(f"A carregar dataset de tokens de: {args.dataset_path}")

    dataset = CarRacingTokenDataset(
        args.dataset_path,
        seq_len=args.frames_per_seq
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    if args.overfit_test:
        print("🧪 Modo OVERFIT ativado")
        small_batch = next(iter(dataloader))
        small_batch = [b.to(device) for b in small_batch]
    # =========================
    # MODEL (IMPORTANTE)
    # =========================
    config = WorldModelConfig(
        vocab_img=args.vocab_size,
        frames_per_seq=args.frames_per_seq
    )

    print("A inicializar o World Model (Transformer)...")
    model = WorldModel(config).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95)
    )

    scaler = GradScaler()

    start_epoch = 0
    global_step = 0

    # =========================
    # LOAD CHECKPOINT
    # =========================
    if os.path.exists(ckpt_path):
        print(f"Relançando treino a partir de: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        global_step = start_epoch * len(dataloader)

        print(f"Continuando da Época {start_epoch}")
    else:
        print("Nenhum checkpoint encontrado. Iniciando do zero.")


    print("Shape da sequência:", full_seq.shape)
    print("Tokens por frame:", config.tokens_per_frame)
    # =========================
    # TRAIN LOOP
    # =========================
    print("🚀 A iniciar o Treino do World Model...")
    if args.overfit_test:
        print("🧪 Modo OVERFIT ativado")

        small_batch = next(iter(dataloader))
        small_batch = [b.to(device) for b in small_batch]

        model.train()

        for step in range(300):
            img_tokens, actions, rewards, dones = small_batch

            full_seq = build_sequence(img_tokens, actions, rewards, dones, config)

            # DEBUG IMPORTANTE
            if step == 0:
                print("Shape full_seq:", full_seq.shape)
                print("Tokens/frame:", config.tokens_per_frame)

            inputs = full_seq[:, :-1]
            targets = full_seq[:, 1:]

            logits, loss = model(idx=inputs, targets=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f}")

        print("✅ Overfit test finalizado")
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            img_tokens, actions, rewards, dones = [b.to(device) for b in batch]

            # =========================
            # BUILD SEQUENCE (IRIS)
            # =========================
            full_seq = build_sequence(img_tokens, actions, rewards, dones, config)

            inputs = full_seq[:, :-1]
            targets = full_seq[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            # =========================
            # FORWARD
            # =========================
            with autocast():
                logits, loss = model(
                    idx=inputs,
                    targets=targets
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # =========================
            # LOGGING
            # =========================
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            board.log_scalar("Loss/Total", loss.item(), global_step)

            global_step += 1

        # =========================
        # EVAL / IMAGINATION
        # =========================
        model.eval()

        print(f"✨ Gerando imaginação para a época {epoch}...")

        with torch.no_grad():
            seed_seq = full_seq[0:1, :50]

            generated = model.generate(
                seed_seq,
                max_new_tokens=200
            )

            tokens_vis = generated[0].float() / config.vocab_total
            board.log_image("Imagination/Token_Map", tokens_vis.unsqueeze(0), epoch)

        board.log_layer_gradients(model, epoch)

        # =========================
        # SAVE CHECKPOINT
        # =========================
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

    parser.add_argument('--run_name',       type=str, default="GPT_CARRACING")
    parser.add_argument('--dataset_path',   type=str, required=True)
    parser.add_argument('--epochs',         type=int, default=200)
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--vocab_size',     type=int, default=512)
    parser.add_argument('--frames_per_seq', type=int, default=20)
    parser.add_argument('--device',         type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--overfit_test', action='store_true')
    args = parser.parse_args()

    train_gpt(args)
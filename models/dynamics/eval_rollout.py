"""
Avalia a qualidade do World Model em rollouts autoregressivos multi-passo,
usando as ACOES REAIS gravadas pra "andar" a janela (isola o erro do world
model do erro de uma politica que ainda nao existe). Responde: "meu
transformer imagina bem o suficiente pra treinar RL em cima, ou so ate
quantos passos?"

Uso:
    python -m models.dynamics.eval_rollout --dataset_path dataset_tokens --horizon 20
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import time
import numpy as np
import torch
from torch.amp import autocast

from models.dynamics.gptdynamics import WorldModel, WorldModelConfig


def get_test_files(dataset_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return files[n_train + n_val:]


def load_world_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    init_keys = {"obs_vocab_size", "act_vocab_size", "img_tokens", "context_len", "n_embd", "n_head", "n_layer", "dropout"}
    cfg = {k: v for k, v in cfg.items() if k in init_keys}
    config = WorldModelConfig(**cfg) if cfg else WorldModelConfig()
    model = WorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def _collect_windows(files, file_order, context_len, horizon, n_windows):
    obs_ctx_list, act_ctx_list = [], []
    real_obs_list, real_act_list, real_rew_list, real_done_list = [], [], [], []
    persist_rew_list, persist_done_list = [], []

    rng = np.random.default_rng(0)  # so pra escolher o offset dentro de cada episodio
    for fi in file_order:
        if len(obs_ctx_list) >= n_windows:
            break
        d = np.load(files[fi], allow_pickle=False)
        tokens = d["tokens"].astype(np.int64)
        actions = d["actions"].astype(np.int64)
        dones_raw = d["dones"].astype(np.int64)
        rewards_raw = d["rewards"].astype(np.float32)  # escala real, nao classe

        T = tokens.shape[0]
        C = context_len
        if T < C + horizon:
            continue

        i = int(rng.integers(0, T - C - horizon + 1))
        obs_ctx_list.append(tokens[i:i + C])
        act_ctx_list.append(actions[i:i + C])
        real_obs_list.append(tokens[i + C:i + C + horizon])
        real_act_list.append(actions[i + C:i + C + horizon])
        real_rew_list.append(rewards_raw[i + C:i + C + horizon])
        real_done_list.append(dones_raw[i + C:i + C + horizon])
        persist_rew_list.append(rewards_raw[i + C - 1])
        persist_done_list.append(dones_raw[i + C - 1])

    return (
        np.stack(obs_ctx_list), np.stack(act_ctx_list),
        np.stack(real_obs_list), np.stack(real_act_list),
        np.stack(real_rew_list), np.stack(real_done_list),
        np.array(persist_rew_list), np.array(persist_done_list),
    )


@torch.no_grad()
def _run_chunk(model, device, obs_ctx_np, act_ctx_np, real_obs, real_act, real_rew, real_done, horizon):
    """Roda o rollout de `horizon` passos pra um chunk de janelas (ja em memoria, batch pequeno o
    suficiente pra caber na GPU). Autocast igual ao treino -- sem isso, o forward roda em fp32
    puro e a matriz de atencao (batch x 1300 x 1300) estoura a memoria em batches grandes."""
    device_type = device.split(":")[0]
    obs_ctx = torch.from_numpy(obs_ctx_np).long().to(device)
    act_ctx = torch.from_numpy(act_ctx_np).long().to(device)

    B = obs_ctx_np.shape[0]
    obs_correct = np.zeros(horizon)
    obs_total = np.zeros(horizon)
    rew_abs_err = np.zeros(horizon)
    done_correct = np.zeros(horizon)
    done_confusion = np.zeros((2, 2), dtype=np.int64)

    for s in range(horizon):
        act_token = torch.from_numpy(real_act[:, s]).long().to(device)
        with autocast(device_type=device_type):
            next_obs, reward_pred, done_pred = model.imagine_next_frame(obs_ctx, act_ctx, act_token)

        next_obs_np = next_obs.cpu().numpy()
        reward_pred_np = reward_pred.float().cpu().numpy()
        done_pred_np = done_pred.cpu().numpy()

        obs_correct[s] += (next_obs_np == real_obs[:, s]).sum()
        obs_total[s] += real_obs[:, s].size
        rew_abs_err[s] += np.abs(reward_pred_np - real_rew[:, s]).sum()
        done_correct[s] += (done_pred_np == real_done[:, s]).sum()

        for b in range(B):
            done_confusion[real_done[b, s], done_pred_np[b]] += 1

        obs_ctx = torch.cat([obs_ctx[:, 1:], next_obs.unsqueeze(1)], dim=1)
        act_ctx = torch.cat([act_ctx[:, 1:], act_token.unsqueeze(1)], dim=1)

    return obs_correct, obs_total, rew_abs_err, done_correct, done_confusion


def evaluate_rollout(model, config, files, device, horizon, n_windows, seed, eval_batch_size=64):
    """
    Junta ate n_windows janelas de episodios distintos e imagina os `horizon`
    passos em chunks de `eval_batch_size` (em vez de n_windows*horizon
    chamadas sequenciais de imagine_next_frame, sao so
    horizon * ceil(n_windows/eval_batch_size) -- e cada chunk cabe na GPU).
    """
    C = config.context_len
    rng = np.random.default_rng(seed)
    file_order = rng.permutation(len(files))

    obs_ctx, act_ctx, real_obs, real_act, real_rew, real_done, persist_rew, persist_done = _collect_windows(
        files, file_order, C, horizon, n_windows
    )
    B = obs_ctx.shape[0]

    obs_correct = np.zeros(horizon)
    obs_total = np.zeros(horizon)
    rew_abs_err = np.zeros(horizon)
    done_correct = np.zeros(horizon)
    persist_rew_abs_err = np.zeros(horizon)
    persist_done_correct = np.zeros(horizon)
    done_confusion = np.zeros((2, 2), dtype=np.int64)

    for start in range(0, B, eval_batch_size):
        end = min(start + eval_batch_size, B)
        c_obs_correct, c_obs_total, c_rew_abs_err, c_done_correct, c_done_conf = _run_chunk(
            model, device,
            obs_ctx[start:end], act_ctx[start:end],
            real_obs[start:end], real_act[start:end], real_rew[start:end], real_done[start:end],
            horizon,
        )
        obs_correct += c_obs_correct
        obs_total += c_obs_total
        rew_abs_err += c_rew_abs_err
        done_correct += c_done_correct
        done_confusion += c_done_conf

        chunk_rew = real_rew[start:end]
        chunk_done = real_done[start:end]
        chunk_persist_rew = persist_rew[start:end]
        chunk_persist_done = persist_done[start:end]
        for s in range(horizon):
            persist_rew_abs_err[s] += np.abs(chunk_persist_rew - chunk_rew[:, s]).sum()
            persist_done_correct[s] += (chunk_persist_done == chunk_done[:, s]).sum()

    return {
        "obs_acc": obs_correct / np.maximum(obs_total, 1),
        "rew_mae": rew_abs_err / max(B, 1),
        "done_acc": done_correct / max(B, 1),
        "persist_rew_mae": persist_rew_abs_err / max(B, 1),
        "persist_done_acc": persist_done_correct / max(B, 1),
        "done_confusion": done_confusion,
        "n_windows": B,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/gpt_best.pt")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--n_windows", type=int, default=200)
    p.add_argument("--eval_batch_size", type=int, default=64, help="quantas janelas processar de uma vez na GPU (chunk); reduza se der OOM")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_file", type=str, default=None, help="onde salvar o .txt com o resultado; se omitido, gera um nome com timestamp")
    args = p.parse_args()

    lines = []

    def emit(line=""):
        print(line)
        lines.append(str(line))

    model, config = load_world_model(args.dynamics_ckpt, args.device)
    files = get_test_files(args.dataset_path)
    emit(f"Checkpoint: {args.dynamics_ckpt}")
    emit(f"Avaliando em {len(files)} episodios de teste (ate {args.n_windows} janelas, horizon={args.horizon}, context_len={config.context_len})")

    result = evaluate_rollout(model, config, files, args.device, args.horizon, args.n_windows, args.seed, args.eval_batch_size)

    emit(f"\n{result['n_windows']} janelas avaliadas.\n")
    emit(f"{'passo':>5} | {'obs_acc':>8} | {'rew_mae':>8} | {'rew_mae_persist':>15} | {'done_acc':>8} | {'done_persist':>12}")
    for s in range(args.horizon):
        emit(
            f"{s + 1:>5} | {result['obs_acc'][s]:>8.3f} | {result['rew_mae'][s]:>8.3f} | "
            f"{result['persist_rew_mae'][s]:>15.3f} | {result['done_acc'][s]:>8.3f} | {result['persist_done_acc'][s]:>12.3f}"
        )
    emit("\nrew_mae = erro absoluto medio do reward previsto (escala real), rew_mae_persist = baseline de "
         "repetir o ultimo reward real antes do horizonte -- rew_mae menor que rew_mae_persist indica sinal util.")

    emit("\nMatriz de confusao de done (linha=real, coluna=previsto; 0/1):")
    emit(str(result["done_confusion"]))

    tp = result["done_confusion"][1, 1]
    fn = result["done_confusion"][1, 0]
    fp = result["done_confusion"][0, 1]
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    emit(f"\ndone=1 -> recall={recall:.3f} precision={precision:.3f} (se os dois forem ~0, o modelo nunca preve fim de jogo)")

    output_file = args.output_file or f"eval_rollout_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResultado salvo em: {output_file}")


if __name__ == "__main__":
    main()

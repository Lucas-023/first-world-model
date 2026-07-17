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
import numpy as np
import torch

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


@torch.no_grad()
def evaluate_rollout(model, config, files, device, horizon, n_windows, seed):
    C = config.context_len
    rng = np.random.default_rng(seed)

    obs_correct = np.zeros(horizon)
    obs_total = np.zeros(horizon)
    rew_correct = np.zeros(horizon)
    done_correct = np.zeros(horizon)
    step_total = np.zeros(horizon)
    persist_rew_correct = np.zeros(horizon)
    persist_done_correct = np.zeros(horizon)

    reward_confusion = np.zeros((3, 3), dtype=np.int64)
    done_confusion = np.zeros((2, 2), dtype=np.int64)

    collected = 0
    file_order = rng.permutation(len(files))

    for fi in file_order:
        if collected >= n_windows:
            break
        d = np.load(files[fi], allow_pickle=False)
        tokens = d["tokens"].astype(np.int64)
        actions = d["actions"].astype(np.int64)
        dones_raw = d["dones"].astype(np.int64)
        rewards_sign = (np.sign(d["rewards"].astype(np.float32)) + 1).astype(np.int64)

        T = tokens.shape[0]
        if T < C + horizon:
            continue

        i = int(rng.integers(0, T - C - horizon + 1))
        obs_ctx = torch.from_numpy(tokens[i:i + C]).long().unsqueeze(0).to(device)
        act_ctx = torch.from_numpy(actions[i:i + C]).long().unsqueeze(0).to(device)

        persist_rew = int(rewards_sign[i + C - 1])
        persist_done = int(dones_raw[i + C - 1])

        for s in range(horizon):
            idx = i + C + s
            act_token = torch.tensor([actions[idx]], dtype=torch.long, device=device)
            next_obs, reward_pred, done_pred = model.imagine_next_frame(obs_ctx, act_ctx, act_token)

            real_obs = tokens[idx]
            real_rew = int(rewards_sign[idx])
            real_done = int(dones_raw[idx])
            pred_rew = int(reward_pred.item())
            pred_done = int(done_pred.item())

            obs_correct[s] += (next_obs.cpu().numpy()[0] == real_obs).sum()
            obs_total[s] += real_obs.shape[0]
            rew_correct[s] += int(pred_rew == real_rew)
            done_correct[s] += int(pred_done == real_done)
            step_total[s] += 1

            reward_confusion[real_rew, pred_rew] += 1
            done_confusion[real_done, pred_done] += 1
            persist_rew_correct[s] += int(persist_rew == real_rew)
            persist_done_correct[s] += int(persist_done == real_done)

            obs_ctx = torch.cat([obs_ctx[:, 1:], next_obs.unsqueeze(1)], dim=1)
            act_ctx = torch.cat([act_ctx[:, 1:], act_token.unsqueeze(1)], dim=1)

        collected += 1

    return {
        "obs_acc": obs_correct / np.maximum(obs_total, 1),
        "rew_acc": rew_correct / np.maximum(step_total, 1),
        "done_acc": done_correct / np.maximum(step_total, 1),
        "persist_rew_acc": persist_rew_correct / np.maximum(step_total, 1),
        "persist_done_acc": persist_done_correct / np.maximum(step_total, 1),
        "reward_confusion": reward_confusion,
        "done_confusion": done_confusion,
        "n_windows": collected,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/gpt_best.pt")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--n_windows", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, config = load_world_model(args.dynamics_ckpt, args.device)
    files = get_test_files(args.dataset_path)
    print(f"Avaliando em {len(files)} episodios de teste (ate {args.n_windows} janelas, horizon={args.horizon}, context_len={config.context_len})")

    result = evaluate_rollout(model, config, files, args.device, args.horizon, args.n_windows, args.seed)

    print(f"\n{result['n_windows']} janelas avaliadas.\n")
    print(f"{'passo':>5} | {'obs_acc':>8} | {'rew_acc':>8} | {'rew_persist':>11} | {'done_acc':>8} | {'done_persist':>12}")
    for s in range(args.horizon):
        print(
            f"{s + 1:>5} | {result['obs_acc'][s]:>8.3f} | {result['rew_acc'][s]:>8.3f} | "
            f"{result['persist_rew_acc'][s]:>11.3f} | {result['done_acc'][s]:>8.3f} | {result['persist_done_acc'][s]:>12.3f}"
        )

    print("\nMatriz de confusao de reward (linha=real, coluna=previsto; 0=neg,1=neutro,2=pos):")
    print(result["reward_confusion"])
    print("\nMatriz de confusao de done (linha=real, coluna=previsto; 0/1):")
    print(result["done_confusion"])

    tp = result["done_confusion"][1, 1]
    fn = result["done_confusion"][1, 0]
    fp = result["done_confusion"][0, 1]
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    print(f"\ndone=1 -> recall={recall:.3f} precision={precision:.3f} (se os dois forem ~0, o modelo nunca preve fim de jogo)")


if __name__ == "__main__":
    main()

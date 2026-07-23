"""
Baseline: politica ALEATORIA sonhando -- roda o mesmo collect_rollout usado no
treino (models/policy/rollout.py), mas com acoes uniformes aleatorias em vez
da politica treinada, pra medir o reward medio por passo que o World Model
atribui a um comportamento sem nenhum aprendizado.

Serve pra comparar com --random_policy de eval_real_env.py (aleatoria no
mundo REAL) e com a politica treinada no mundo real -- as tres pontas juntas
respondem "o World Model agrega alguma coisa, ou o sinal de reward dele e so
ruido descolado da realidade?" e "treinar via sonho supera o acaso puro?".

IMPORTANTE: usa o mesmo horizon curto do treino (default 8), NAO um episodio
completo -- eval_rollout.py ja mostrou que o World Model degrada bem antes do
passo ~10-11 mesmo com acoes reais; com acoes aleatorias (fora da distribuicao
que a politica treinada visita) a degradacao tende a ser ainda mais rapida.
Um rollout longo aqui so acumularia lixo, nao sinal -- por isso a metrica e
reward MEDIO POR PASSO, nao total do episodio (nao da pra comparar total de
um rollout de 8 passos com um episodio real de ~219 passos).

Uso:
    python -m models.policy.eval_random_dream --dataset_path dataset_tokens --n_rollouts 50
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math
import torch

from models.policy.train_dream import load_world_model, seed_iterator
from models.policy.rollout import collect_rollout


class RandomPolicy:
    """Mesma interface de ActorCritic (.act / .forward), pra reusar
    collect_rollout sem duplicar logica de rollout -- ignora o estado e
    sorteia acao uniforme entre as n_actions."""

    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device

    def act(self, state):
        B = state.shape[0]
        action = torch.randint(0, self.n_actions, (B,), device=self.device)
        log_prob = torch.full((B,), -math.log(self.n_actions), device=self.device)
        value = torch.zeros(B, device=self.device)
        entropy = torch.full((B,), math.log(self.n_actions), device=self.device)
        return action, log_prob, value, entropy

    def forward(self, state):
        # so o bootstrap_value de collect_rollout usa isso -- valor 0 (nao
        # ha critico nenhum treinado pra um stub aleatorio)
        B = state.shape[0]
        return None, torch.zeros(B, device=self.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True, help="dataset_tokens -- sementes reais de contexto pra iniciar cada rollout")
    p.add_argument("--dynamics_ckpt", type=str, default="models/dynamics/DYNAMICS_GPT/pesos/gpt_ckpt.pt")
    p.add_argument("--n_rollouts", type=int, default=50, help="quantos rollouts (cada um com --batch_size sementes em paralelo)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8, help="mesmo default de train_dream.py -- nao aumentar sem revisar eval_rollout.py")
    p.add_argument("--seed_stride", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    world_model, wm_config = load_world_model(args.dynamics_ckpt, device)
    print(f"World model carregado (context_len={wm_config.context_len}, congelado).")

    random_policy = RandomPolicy(wm_config.act_vocab_size, device)

    seeds = seed_iterator(
        args.dataset_path, wm_config.context_len, args.batch_size, args.seed, args.seed_stride, args.num_workers, device
    )

    print(f"\nRodando {args.n_rollouts} rollouts imaginados com acao aleatoria "
          f"(batch={args.batch_size}, horizon={args.horizon})...\n")

    per_rollout_reward = []
    for i in range(args.n_rollouts):
        obs_ctx, act_ctx = next(seeds)
        buffer = collect_rollout(world_model, random_policy, obs_ctx, act_ctx, args.horizon)
        active_sum = buffer["active_mask"].sum().clamp(min=1).item()
        mean_reward = (buffer["rewards"] * buffer["active_mask"]).sum().item() / active_sum
        per_rollout_reward.append(mean_reward)
        if (i + 1) % 10 == 0:
            print(f"[Rollout {i + 1:>4}/{args.n_rollouts}] reward medio por passo (ate agora): "
                  f"{sum(per_rollout_reward) / len(per_rollout_reward):.4f}")

    rewards = torch.tensor(per_rollout_reward)
    print(f"\n{'='*60}")
    print(f"  Politica ALEATORIA sonhando (World Model, horizon={args.horizon})")
    print(f"  Rollouts: {args.n_rollouts}  |  Sementes por rollout: {args.batch_size}")
    print(f"  Reward medio por passo imaginado: {rewards.mean():.4f}  |  Std: {rewards.std():.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

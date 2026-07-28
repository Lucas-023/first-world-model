#!/bin/bash
# Roda os dois treinos via interacao real (mesmo orcamento do PPO em pixels,
# ~2M passos reais = 1150 updates * 8 episodios/update) e depois avalia os
# dois policy_best.pt resultantes, 30 episodios cada, mesmo protocolo.
#
# Uso: bash run_experimentos_vm.sh

set -e

echo "=== [1/4] treinando train_real_latent.py (state_repr, 1150 updates) ==="
python -m models.policy.train_real_latent --updates 1150 --episodes_per_update 8

echo "=== [2/4] treinando train_real_latent_zh.py ([h,z], 1150 updates) ==="
python -m models.policy.train_real_latent_zh --updates 1150 --episodes_per_update 8

echo "=== [3/4] avaliando policy_best.pt de train_real_latent.py ==="
python -m models.policy.eval_real_env --n_episodes 30 --deterministic \
  --policy_ckpt models/policy_real_latent/policy_best.pt \
  --dynamics_ckpt models/dynamics/gpt_best.pt \
  --out_csv eval_results/real_env.csv --label real_latent_1150up_deterministica

echo "=== [4/4] avaliando policy_best.pt de train_real_latent_zh.py ==="
python -m models.policy.eval_real_env --n_episodes 30 --deterministic --zh \
  --policy_ckpt models/policy_real_latent_zh/policy_best.pt \
  --dynamics_ckpt models/dynamics/gpt_best.pt \
  --out_csv eval_results/real_env.csv --label real_latent_zh_1150up_deterministica

echo "=== concluido -- resultado final em eval_results/real_env.csv ==="
cat eval_results/real_env.csv

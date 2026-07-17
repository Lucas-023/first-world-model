#!/usr/bin/env bash
# Roda a avaliacao do World Model (eval_rollout.py) e, se ela terminar sem
# erro, retoma o treino de onde parou (traingpt.py --resume_ckpt).
#
# Uso: bash avaliar_e_retomar.sh
set -euo pipefail

DATASET_PATH="dataset_tokens"
DYNAMICS_CKPT="models/dynamics/gpt_best.pt"
RESUME_CKPT="models/dynamics/gpt_ckpt.pt"
RUN_NAME="DYNAMICS_GPT"
HORIZON=25
N_WINDOWS=500

echo "==> Rodando avaliacao do World Model (eval_rollout.py)..."
python -m models.dynamics.eval_rollout \
    --dataset_path "$DATASET_PATH" \
    --dynamics_ckpt "$DYNAMICS_CKPT" \
    --horizon "$HORIZON" \
    --n_windows "$N_WINDOWS"

echo "==> Avaliacao concluida. Retomando treino de $RESUME_CKPT..."
python -m models.dynamics.traingpt \
    --dataset_path "$DATASET_PATH" \
    --batch_size 128 \
    --lr 2e-4 \
    --num_workers 8 \
    --stride 1 \
    --resume_ckpt "$RESUME_CKPT" \
    --run_name "$RUN_NAME"

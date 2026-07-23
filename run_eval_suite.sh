#!/usr/bin/env bash
# Roda a suite completa de avaliacao da politica (treinada + baselines) e
# documenta tudo em eval_results/:
#   - CSVs com 1 linha de resumo por execucao (real_env.csv, random_dream.csv)
#   - logs .txt completos (stdout linha a linha de cada execucao)
#
# Uso:
#   bash run_eval_suite.sh [pasta_da_politica]
#   pasta_da_politica: onde estao policy_best.pt / policy_ckpt.pt
#                       (default: models/policy_vf_low)
set -euo pipefail

POLICY_DIR="${1:-models/policy_vf_low}"
DATASET_PATH="dataset_tokens"
N_EPISODES=30
N_ROLLOUTS=50
RESULTS_DIR="eval_results"
LOGS_DIR="$RESULTS_DIR/logs"
REAL_CSV="$RESULTS_DIR/real_env.csv"
DREAM_CSV="$RESULTS_DIR/random_dream.csv"
TAG="$(basename "$POLICY_DIR")_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOGS_DIR"

run() {
    local label="$1"; shift
    local logfile="$LOGS_DIR/${TAG}_${label}.txt"
    echo "==> $label"
    "$@" 2>&1 | tee "$logfile"
    echo
}

echo "===================================================================="
echo "  Suite de avaliacao -- $TAG"
echo "  Politica: $POLICY_DIR"
echo "  Resultados em: $RESULTS_DIR/"
echo "===================================================================="
echo

run "aleatoria_sonhando" \
    python -m models.policy.eval_random_dream \
        --dataset_path "$DATASET_PATH" \
        --n_rollouts "$N_ROLLOUTS" \
        --out_csv "$DREAM_CSV" \
        --label "aleatoria_sonhando"

run "aleatoria_real" \
    python -m models.policy.eval_real_env \
        --random_policy \
        --n_episodes "$N_EPISODES" \
        --out_csv "$REAL_CSV" \
        --label "aleatoria_real"

run "treinada_best_estocastica" \
    python -m models.policy.eval_real_env \
        --policy_ckpt "$POLICY_DIR/policy_best.pt" \
        --n_episodes "$N_EPISODES" \
        --out_csv "$REAL_CSV" \
        --label "treinada_best_estocastica"

run "treinada_best_deterministica" \
    python -m models.policy.eval_real_env \
        --policy_ckpt "$POLICY_DIR/policy_best.pt" \
        --n_episodes "$N_EPISODES" \
        --deterministic \
        --out_csv "$REAL_CSV" \
        --label "treinada_best_deterministica"

run "treinada_final_estocastica" \
    python -m models.policy.eval_real_env \
        --policy_ckpt "$POLICY_DIR/policy_ckpt.pt" \
        --n_episodes "$N_EPISODES" \
        --out_csv "$REAL_CSV" \
        --label "treinada_final_estocastica"

run "treinada_final_deterministica" \
    python -m models.policy.eval_real_env \
        --policy_ckpt "$POLICY_DIR/policy_ckpt.pt" \
        --n_episodes "$N_EPISODES" \
        --deterministic \
        --out_csv "$REAL_CSV" \
        --label "treinada_final_deterministica"

echo "===================================================================="
echo "  Concluido. Resumo:"
echo "    $REAL_CSV"
echo "    $DREAM_CSV"
echo "  Logs completos: $LOGS_DIR/${TAG}_*.txt"
echo "===================================================================="

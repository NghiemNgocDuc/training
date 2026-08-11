#!/usr/bin/env bash
# Full neighbor-regularization sweep (run on the GPU box, detached).
# Two formulations share ONE lambda=0 baseline:
#   raw:        exact L_neighbor,          lambdas 0.001 0.003 0.01 0.03
#   normalized: L_neighbor / var(pred),    lambdas 0.05 0.1 0.3 1.0
# Usage: bash run_sweep.sh            # grid above, seed 42
#        bash run_sweep.sh --seeds 42 123 7
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PY=python
RAW_LAMBDAS=(0.001 0.003 0.01 0.03)
NORM_LAMBDAS=(0.05 0.1 0.3 1.0)
SEEDS=(42)
if [ "$1" = "--seeds" ]; then shift; SEEDS=("$@"); fi

for S in "${SEEDS[@]}"; do
  # shared baseline (raw and normalized both compare against it)
  OUT="aqm-spice2/freesolv/neighbor_regularization/baseline/lambda0_seed${S}"
  LOG="aqm-spice2/freesolv/neighbor_regularization/logs/baseline_lambda0_seed${S}.log"
  mkdir -p "$(dirname "$LOG")"
  echo "=== baseline lambda=0 seed=$S -> $LOG (detached) ==="
  nohup $PY aqm-spice2/freesolv/neighbor_regularization/finetune_nbr.py \
    --lambda_nbr 0 --seed "$S" --out "$OUT" --epochs 200 --patience 30 \
    > "$LOG" 2>&1 &
  echo "  pid $!"

  for L in "${RAW_LAMBDAS[@]}"; do
    OUT="aqm-spice2/freesolv/neighbor_regularization/raw/lambda${L}_seed${S}"
    LOG="aqm-spice2/freesolv/neighbor_regularization/logs/raw_lambda${L}_seed${S}.log"
    mkdir -p "$(dirname "$LOG")"
    echo "=== raw lambda=$L seed=$S -> $LOG (detached) ==="
    nohup $PY aqm-spice2/freesolv/neighbor_regularization/finetune_nbr.py \
      --lambda_nbr "$L" --seed "$S" --out "$OUT" --epochs 200 --patience 30 \
      > "$LOG" 2>&1 &
    echo "  pid $!"
  done

  for L in "${NORM_LAMBDAS[@]}"; do
    OUT="aqm-spice2/freesolv/neighbor_regularization/normalized/lambda${L}_seed${S}"
    LOG="aqm-spice2/freesolv/neighbor_regularization/logs/normalized_lambda${L}_seed${S}.log"
    mkdir -p "$(dirname "$LOG")"
    echo "=== normalized lambda=$L seed=$S -> $LOG (detached) ==="
    nohup $PY aqm-spice2/freesolv/neighbor_regularization/finetune_nbr.py \
      --lambda_nbr "$L" --normalize_nbr --seed "$S" --out "$OUT" \
      --epochs 200 --patience 30 \
      > "$LOG" 2>&1 &
    echo "  pid $!"
  done
done
echo "All launched. Tail: tail -f aqm-spice2/freesolv/neighbor_regularization/logs/raw_lambda0.01_seed42.log"
echo "Aggregate: python aqm-spice2/freesolv/neighbor_regularization/report_results.py --baseline-dir baseline/lambda0_seed42 --runs raw/lambda0.001_seed42 raw/lambda0.003_seed42 raw/lambda0.01_seed42 raw/lambda0.03_seed42 normalized/lambda0.05_seed42 normalized/lambda0.1_seed42 normalized/lambda0.3_seed42 normalized/lambda1.0_seed42"
echo "Curves:    python aqm-spice2/freesolv/neighbor_regularization/plot_l_nbr.py --baseline-dir baseline/lambda0_seed42 --runs raw/lambda0.001_seed42 raw/lambda0.01_seed42 normalized/lambda0.1_seed42 normalized/lambda1.0_seed42"

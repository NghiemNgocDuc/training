#!/usr/bin/env bash
# Neighbor-regularization SMOKE test (box-side, GPU, detached).
# Stability/sanity only: seed 42, 15 epochs, both formulations:
#   raw:        lambda {0, 0.001, 0.01}
#   normalized: lambda {0, 0.1, 1.0}
# v2 latent variant (--neighbor_source latent, k=5, min_sim=0.5) runs the same
# grid with the latent-cosine graph + GMM-NLL uncertainty/trust (DESIGN_v2.md).
# Per-epoch task loss + L_neighbor tracking (epoch_history.csv), NaN/Inf checks,
# and --track_groups (per-epoch test-set MAE for isolated6/gradient12/wrong18/
# certain47 -> epoch_test_groups.csv) for the early isolated-6 read.
# Usage: bash run_smoke_box.sh
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PY=python
EPOCHS=15
SEED=42
V2="--neighbor_source latent --k_nbr 5 --min_sim 0.5"

# v2 latent graph + GMM-NLL signals (idempotent; reuses cached z_train/z_test,
# regenerates z_val.npz if missing)
echo "=== building latent graph (if stale) ==="
$PY aqm-spice2/freesolv/neighbor_regularization/latent_graph.py --k 5 --min-sim 0.5 --out aqm-spice2/freesolv/neighbor_regularization/graph_cache 2>&1 | tail -20

declare -a RUNS=(
  "smoke_test/baseline/lambda0_seed42|--lambda_nbr 0"
  "smoke_test/raw/lambda0.001_seed42|--lambda_nbr 0.001"
  "smoke_test/raw/lambda0.01_seed42|--lambda_nbr 0.01"
  "smoke_test/normalized/lambda0.1_seed42|--lambda_nbr 0.1 --normalize_nbr"
  "smoke_test/normalized/lambda1.0_seed42|--lambda_nbr 1.0 --normalize_nbr"
  "smoke_test/v2_latent/baseline_lambda0_seed42|--lambda_nbr 0 $V2"
  "smoke_test/v2_latent/raw_lambda0.001_seed42|--lambda_nbr 0.001 $V2"
  "smoke_test/v2_latent/raw_lambda0.01_seed42|--lambda_nbr 0.01 $V2"
  "smoke_test/v2_latent/norm_lambda0.1_seed42|--lambda_nbr 0.1 --normalize_nbr $V2"
  "smoke_test/v2_latent/norm_lambda1.0_seed42|--lambda_nbr 1.0 --normalize_nbr $V2"
)

for ENTRY in "${RUNS[@]}"; do
  OUT="${ENTRY%%|*}"; EXTRA="${ENTRY#*|}"
  NAME=$(basename "$OUT")
  LOG="aqm-spice2/freesolv/neighbor_regularization/logs/smoke_${NAME}.log"
  mkdir -p "$(dirname "$LOG")"
  echo "=== $OUT -> $LOG (detached) ==="
  nohup $PY aqm-spice2/freesolv/neighbor_regularization/finetune_nbr.py \
    --seed "$SEED" --epochs "$EPOCHS" --patience 30 --track_groups \
    $EXTRA --out "$OUT" \
    > "$LOG" 2>&1 &
  echo "  pid $!"
done

echo "All smoke runs launched (cuda auto-detect). ETA ~5-15 min total on GPU."
echo "Watch:   tail -f aqm-spice2/freesolv/neighbor_regularization/logs/smoke_raw_lambda0.01_seed42.log"
echo "Done?   ls aqm-spice2/freesolv/neighbor_regularization/smoke_test/*/*/metrics.json"
echo "Plots:  python aqm-spice2/freesolv/neighbor_regularization/plot_l_nbr.py --runs smoke_test/raw/lambda0.001_seed42 smoke_test/raw/lambda0.01_seed42 smoke_test/normalized/lambda0.1_seed42 smoke_test/normalized/lambda1.0_seed42"
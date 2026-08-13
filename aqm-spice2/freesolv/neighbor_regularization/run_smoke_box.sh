#!/usr/bin/env bash
# Neighbor-regularization SMOKE test (box-side, GPU, SEQUENTIAL).
# Runs the 10-run grid ONE AT A TIME: 10 concurrent DimeNet++ models OOM'd a
# 23.55 GiB GPU (each model ~3 GiB), so runs are serialized here. The whole
# script itself is detached with nohup; per-run logs are still written.
#   raw:        lambda {0, 0.001, 0.01}
#   normalized: lambda {0, 0.1, 1.0}
# v2 latent variant (--neighbor_source latent, k=5, min_sim=0.5) runs the same
# grid with the latent-cosine graph + GMM-NLL uncertainty/trust (DESIGN_v2.md).
# Per-epoch task loss + L_neighbor tracking (epoch_history.csv), NaN/Inf checks,
# and --track_groups (per-epoch test-set MAE for isolated6/gradient12/wrong18/
# certain47 -> epoch_test_groups.csv) for the early isolated-6 read.
#
# Usage (detached):  nohup bash run_smoke_box.sh > smoke_v2_seq.log 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PY=python
EPOCHS=15
SEED=42
NR="aqm-spice2/freesolv/neighbor_regularization"
V2="--neighbor_source latent --k_nbr 5 --min_sim 0.5"

# v2 latent graph + GMM-NLL signals. Build ONLY if the committed graph meta is
# missing/stale (the box does not have sklearn; the committed artifacts are the
# source of truth and finetune_nbr.py never needs the builder).
GRAPH_META="$NR/graph_cache/latent_k5_sim0.5.json.meta.json"
if [ ! -f "$GRAPH_META" ]; then
  echo "=== latent graph artifacts missing: rebuilding ==="
  $PY "$NR/latent_graph.py" --k 5 --min-sim 0.5 --out "$NR/graph_cache" 2>&1 | tail -30
else
  echo "=== latent graph artifacts present: skipping builder (no sklearn needed) ==="
fi

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

N_RUNS=${#RUNS[@]}
I=0
for ENTRY in "${RUNS[@]}"; do
  I=$((I + 1))
  OUT="${ENTRY%%|*}"; EXTRA="${ENTRY#*|}"
  NAME=$(basename "$OUT")
  LOG="$NR/logs/smoke_${NAME}.log"
  mkdir -p "$(dirname "$LOG")"
  echo "=== [$I/$N_RUNS] $OUT -> $LOG (sequential) ==="
  rm -rf "$OUT"   # clear any partial results from the failed concurrent batch
  if ! $PY "$NR/finetune_nbr.py" --seed "$SEED" --epochs "$EPOCHS" \
      --patience 30 --track_groups $EXTRA --out "$OUT" > "$LOG" 2>&1; then
    echo "=== RUN FAILED [$I/$N_RUNS] $OUT (see $LOG) ==="
    tail -20 "$LOG"
    exit 1
  fi
  echo "=== done [$I/$N_RUNS] $(grep -c DONE "$LOG") DONE marker ==="
done

echo "ALL $N_RUNS smoke runs finished OK (sequential)."
echo "Results: ls $NR/smoke_test/*/*/metrics.json"
echo "Plots:  python $NR/plot_l_nbr.py --runs $NR/smoke_test/v2_latent/raw_lambda0.001_seed42 $NR/smoke_test/v2_latent/raw_lambda0.01_seed42 $NR/smoke_test/v2_latent/norm_lambda0.1_seed42 $NR/smoke_test/v2_latent/norm_lambda1.0_seed42"
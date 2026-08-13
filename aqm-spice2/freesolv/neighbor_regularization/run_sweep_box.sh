#!/usr/bin/env bash
# Full neighbor-regularization SWEEP (box-side, GPU, SEQUENTIAL).
# 10 concurrent DimeNet++ models OOM a 23.55 GiB GPU (~3 GiB each), so runs
# are serialized (17 runs x ~10-30 min ~= 3-8 h total; watch smoke_v2_seq.log
# style progress below).
#
# One SHARED lambda=0 baseline (lambda 0 skips the graph pass -> identical
# for v1 tanimoto and v2 latent sources).
#   v1 tanimoto: raw {0.001 0.003 0.01 0.03}, normalized {0.05 0.1 0.3 1.0}
#   v2 latent:   same grid, --neighbor_source latent (k=5, min_sim=0.5)
#
# Usage (detached):  nohup bash run_sweep_box.sh > sweep_v2_seq.log 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PY=python
EPOCHS=200
PATIENCE=30
SEED=42
NR="aqm-spice2/freesolv/neighbor_regularization"
V2="--neighbor_source latent --k_nbr 5 --min_sim 0.5"
RAW_LAMBDAS=(0.001 0.003 0.01 0.03)
NORM_LAMBDAS=(0.05 0.1 0.3 1.0)

GRAPH_META="$NR/graph_cache/latent_k5_sim0.5.json.meta.json"
if [ ! -f "$GRAPH_META" ]; then
  echo "=== latent graph artifacts missing: rebuilding ==="
  $PY "$NR/latent_graph.py" --k 5 --min-sim 0.5 --out "$NR/graph_cache" 2>&1 | tail -30
else
  echo "=== latent graph artifacts present: skipping builder ==="
fi

declare -a RUNS=(
  "baseline/lambda0_seed42|--lambda_nbr 0"
)
for L in "${RAW_LAMBDAS[@]}"; do
  RUNS+=("raw/lambda${L}_seed42|--lambda_nbr $L")
done
for L in "${NORM_LAMBDAS[@]}"; do
  RUNS+=("normalized/lambda${L}_seed42|--lambda_nbr $L --normalize_nbr")
done
for L in "${RAW_LAMBDAS[@]}"; do
  RUNS+=("v2_latent/raw_lambda${L}_seed42|--lambda_nbr $L $V2")
done
for L in "${NORM_LAMBDAS[@]}"; do
  RUNS+=("v2_latent/normalized/lambda${L}_seed42|--lambda_nbr $L --normalize_nbr $V2")
done

N_RUNS=${#RUNS[@]}
I=0
for ENTRY in "${RUNS[@]}"; do
  I=$((I + 1))
  OUT="${ENTRY%%|*}"; EXTRA="${ENTRY#*|}"
  NAME=$(basename "$OUT")
  LOG="$NR/logs/sweep_${NAME}.log"
  mkdir -p "$(dirname "$LOG")"
  echo "=== [$I/$N_RUNS] $OUT -> $LOG (sequential) ==="
  rm -rf "$NR/$OUT"   # no partials from any earlier attempt
  if ! $PY "$NR/finetune_nbr.py" --seed "$SEED" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --track_groups $EXTRA --out "$NR/$OUT" \
      > "$LOG" 2>&1; then
    echo "=== RUN FAILED [$I/$N_RUNS] $OUT (see $LOG) ==="
    tail -20 "$LOG"
    exit 1
  fi
  echo "=== done [$I/$N_RUNS] $(grep -c DONE "$LOG") DONE marker ==="
done

echo "ALL $N_RUNS sweep runs finished OK (sequential)."
echo "Results: ls $NR/baseline $NR/raw $NR/normalized $NR/v2_latent"
echo "Aggregate: python $NR/report_results.py --baseline-dir $NR/baseline/lambda0_seed42 --runs $NR/raw/lambda0.001_seed42 $NR/raw/lambda0.003_seed42 $NR/raw/lambda0.01_seed42 $NR/raw/lambda0.03_seed42 $NR/normalized/lambda0.05_seed42 $NR/normalized/lambda0.1_seed42 $NR/normalized/lambda0.3_seed42 $NR/normalized/lambda1.0_seed42"
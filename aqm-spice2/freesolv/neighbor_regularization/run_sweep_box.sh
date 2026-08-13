#!/usr/bin/env bash
# Full neighbor-regularization SWEEP (box-side, GPU, parallel-by-default-3).
# 10 concurrent DimeNet++ models OOM a 23.55 GiB GPU (~3 GiB each); 3 at a
# time (~9 GiB) is safe and keeps the GPU busy. Tune with PARALLEL=N env.
# 17 runs x ~10-30 min / 3 = ~3-8 h total.
#
# One SHARED lambda=0 baseline (lambda 0 skips the graph pass -> identical
# for v1 tanimoto and v2 latent sources).
#   v1 tanimoto: raw {0.001 0.003 0.01 0.03}, normalized {0.05 0.1 0.3 1.0}
#   v2 latent:   same grid, --neighbor_source latent (k=5, min_sim=0.5)
#
# Usage (detached):  nohup bash run_sweep_box.sh > sweep_v2_seq.log 2>&1 &
# PARALLEL=4 nohup bash run_sweep_box.sh > sweep_v2_seq.log 2>&1 &
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

mkdir -p "$NR"
LOCK="$NR/.sweep.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "ERROR: run_sweep_box.sh is already running (PID $(cat "$LOCK"))."
  echo "Kill it first (pkill -f run_sweep_box) or remove $LOCK, then relaunch."
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

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
PARALLEL="${PARALLEL:-3}"        # ~3 GiB per model; 3 concurrent is safe on 23.55 GiB
PIDS=()
launch_one() {
  local entry="$1" out extra name log
  out="${entry%%|*}"; extra="${entry#*|}"
  name=$(basename "$out")
  log="$NR/logs/sweep_${name}.log"
  mkdir -p "$(dirname "$log")"
  echo "=== [$I/$N_RUNS] $out -> $log (parallel) ==="
  rm -rf "$NR/$out"   # no partials from any earlier attempt
  $PY "$NR/finetune_nbr.py" --seed "$SEED" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --track_groups $extra --out "$NR/$out" \
      > "$log" 2>&1 &
  PIDS+=("$!")
}

reap() {
  local new=() pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new+=("$pid")
    else
      echo "=== done [pid $pid] ==="
    fi
  done
  PIDS=("${new[@]}")
}

for ENTRY in "${RUNS[@]}"; do
  I=$((I + 1))
  while [ "${#PIDS[@]}" -ge "$PARALLEL" ]; do
    reap
    if [ "${#PIDS[@]}" -ge "$PARALLEL" ]; then sleep 15; fi
  done
  launch_one "$ENTRY"
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "=== done [pid $pid] ==="
  else
    FAILED=$((FAILED + 1))
    echo "=== done [pid $pid] FAILED ==="
  fi
done
if [ "$FAILED" -gt 0 ]; then
  echo "=== $FAILED RUN(S) FAILED - inspect logs under $NR/logs/sweep_*.log ==="
  grep -l "Traceback\|CUDA out of memory" "$NR"/logs/sweep_*.log 2>/dev/null || true
  exit 1
fi

echo "ALL $N_RUNS sweep runs finished OK (parallel=$PARALLEL)."
echo "Results: ls $NR/baseline $NR/raw $NR/normalized $NR/v2_latent"
echo "Aggregate: python $NR/report_results.py --baseline-dir $NR/baseline/lambda0_seed42 --runs $NR/raw/lambda0.001_seed42 $NR/raw/lambda0.003_seed42 $NR/raw/lambda0.01_seed42 $NR/raw/lambda0.03_seed42 $NR/normalized/lambda0.05_seed42 $NR/normalized/lambda0.1_seed42 $NR/normalized/lambda0.3_seed42 $NR/normalized/lambda1.0_seed42"
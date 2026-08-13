#!/usr/bin/env bash
# Rerun of the runs lost to startup OOM in the PARALLEL=2 sweep (10,11,12,13,17).
# The killer was the INITIAL graph_pass forward (line 422, before epoch 1):
# a whole-graph batch spike that OOMs when two runs hit it simultaneously.
# Sequential (PARALLEL=1) gives one process the full 23.55 GiB - safe.
#
# Auto-skips any run whose dir already has metrics.json (written only on a
# clean completion), so rerunning is safe even if the dead list is imperfect.
#
# Usage (detached):  nohup bash run_rerun_box.sh > rerun_v2.log 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PY=python
EPOCHS=200
PATIENCE=30
SEED=42
NR="aqm-spice2/freesolv/neighbor_regularization"
V2="--neighbor_source latent --k_nbr 5 --min_sim 0.5"

mkdir -p "$NR"
LOCK="$NR/.sweep.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "ERROR: another box launcher is already running (PID $(cat "$LOCK"))."
  echo "Kill it first or remove $LOCK, then relaunch."
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

declare -a RUNS=(
  "v2_latent/raw_lambda0.001_seed42|--lambda_nbr 0.001 $V2"
  "v2_latent/raw_lambda0.003_seed42|--lambda_nbr 0.003 $V2"
  "v2_latent/raw_lambda0.01_seed42|--lambda_nbr 0.01 $V2"
  "v2_latent/raw_lambda0.03_seed42|--lambda_nbr 0.03 $V2"
  "v2_latent/normalized/lambda1.0_seed42|--lambda_nbr 1.0 --normalize_nbr $V2"
)

N_RUNS=${#RUNS[@]}
I=0
PARALLEL="${PARALLEL:-1}"        # sequential: proven safe against startup OOM
declare -a RPIDS=() RIDX=() ROUT=()
launch_one() {
  local entry="$1" out extra name log
  out="${entry%%|*}"; extra="${entry#*|}"
  name="${out//\//_}"
  log="$NR/logs/rerun_${name}.log"
  mkdir -p "$(dirname "$log")"
  if [ -f "$NR/$NR/$out/metrics.json" ]; then
    echo "=== [$I/$N_RUNS] SKIP (already complete): $out ==="
    echo "=== done [$I/$N_RUNS] $out ==="
    return
  fi
  echo "=== [$I/$N_RUNS] $out -> $log (rerun) ==="
  rm -rf "$NR/$NR/$out"   # trainer resolves relative --out against its own dir: nested at $NR/$NR
  $PY "$NR/finetune_nbr.py" --seed "$SEED" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --track_groups $extra --out "$NR/$out" \
      > "$log" 2>&1 &
  RPIDS+=("$!")
  RIDX+=("$I")
  ROUT+=("$out")
}

reap() {
  local new=() nidx=() nout=() k pid idx out
  for k in "${!RPIDS[@]}"; do
    pid="${RPIDS[$k]}"; idx="${RIDX[$k]}"; out="${ROUT[$k]}"
    if kill -0 "$pid" 2>/dev/null; then
      new+=("$pid"); nidx+=("$idx"); nout+=("$out")
    else
      echo "=== done [$idx/$N_RUNS] $out ==="
    fi
  done
  RPIDS=("${new[@]}"); RIDX=("${nidx[@]}"); ROUT=("${nout[@]}")
}

for ENTRY in "${RUNS[@]}"; do
  I=$((I + 1))
  while [ "${#RPIDS[@]}" -ge "$PARALLEL" ]; do
    reap
    if [ "${#RPIDS[@]}" -ge "$PARALLEL" ]; then sleep 15; fi
  done
  launch_one "$ENTRY"
done

FAILED=0
for k in "${!RPIDS[@]}"; do
  pid="${RPIDS[$k]}"; idx="${RIDX[$k]}"; out="${ROUT[$k]}"
  if wait "$pid"; then
    echo "=== done [$idx/$N_RUNS] $out ==="
  else
    FAILED=$((FAILED + 1))
    echo "=== done [$idx/$N_RUNS] $out FAILED ==="
  fi
done
if [ "$FAILED" -gt 0 ]; then
  echo "=== $FAILED RERUN(S) FAILED - inspect logs under $NR/logs/rerun_*.log ==="
  grep -l "Traceback\|CUDA out of memory" "$NR"/logs/rerun_*.log 2>/dev/null || true
  exit 1
fi

echo "ALL $N_RUNS rerun runs finished OK (parallel=$PARALLEL)."
echo "Results: ls $NR/v2_latent"
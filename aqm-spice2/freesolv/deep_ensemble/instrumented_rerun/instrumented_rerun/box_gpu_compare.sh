#!/usr/bin/env bash
# Phase A (Vast GPU box): 5-epoch same-box comparison, orig x2 vs fixed x1.
#
#   git pull
#   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/box_gpu_compare.sh \
#       > box_gpu_compare.log 2>&1 &
#   tail -f box_gpu_compare.log
#
# On PASS the full seed_42 rerun + same-box original baseline (Phase B) is
# launched automatically in the background. On FAIL the script stops and the
# verdict report must be reviewed before anything else runs (per protocol).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RERUN="$ROOT/aqm-spice2/freesolv/deep_ensemble/instrumented_rerun"
cd "$ROOT"

echo "[cmp] root: $ROOT"
echo "[cmp] python: $(which python) | torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

echo "[cmp] 1/4 orig_run1: original deep_ensemble.py seed 42, 5 epochs"
python aqm-spice2/freesolv/deep_ensemble.py --mode train --seed 42 --epochs 5 \
    --device cuda --output_dir "$ROOT/cmp_orig1" > "$ROOT/cmp_orig1.log" 2>&1
echo "[cmp] 2/4 orig_run2: same command again"
python aqm-spice2/freesolv/deep_ensemble.py --mode train --seed 42 --epochs 5 \
    --device cuda --output_dir "$ROOT/cmp_orig2" > "$ROOT/cmp_orig2.log" 2>&1
echo "[cmp] 3/4 fixed_run1: instrumented (RNG-guarded) seed 42, 5 epochs"
python "$RERUN/instrument_finetune.py" --seed 42 --epochs 5 --device cuda \
    --out "$ROOT/cmp_inst" > "$ROOT/cmp_inst.log" 2>&1

echo "[cmp] 4/4 verdict"
if python "$RERUN/compare_short_runs.py" \
        --orig1-log "$ROOT/cmp_orig1.log" \
        --orig2-log "$ROOT/cmp_orig2.log" \
        --fixed-csv "$ROOT/cmp_inst/seed_42/val_history.csv" \
        --report "$ROOT/cmp_verdict.json"; then
    echo "[cmp] PASS - launching Phase B (full seed_42 rerun + same-box original"
    echo "[cmp] baseline + dual-reference sanity) in the background:"
    echo "[cmp]   tail -f $ROOT/box_full_rerun.log"
    nohup bash "$RERUN/box_full_rerun.sh" > "$ROOT/box_full_rerun.log" 2>&1 &
    echo "[cmp] Phase B pid: $!"
else
    echo "[cmp] FAIL - STOP before Phase B per protocol."
    echo "[cmp] Review: $ROOT/cmp_verdict.json"
    exit 1
fi

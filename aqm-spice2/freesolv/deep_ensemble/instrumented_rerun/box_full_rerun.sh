#!/usr/bin/env bash
# Phase B (Vast GPU box, launched automatically by box_gpu_compare.sh on PASS):
#   1. full instrumented seed_42 rerun (200 epochs, RNG-guarded) -> instrumented_rerun/seed_42
#   2. full ORIGINAL deep_ensemble.py seed 42 on THIS box -> box_orig_full/seed_42
#      (the same-box reference that absorbs torch/cuDNN/GPU drift)
#   3. sanity_check vs BOTH references (recorded + same-box) with tolerances that
#      account for the baseline's own self-noise
#   4. Stage A early read
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RERUN="$ROOT/aqm-spice2/freesolv/deep_ensemble/instrumented_rerun"
RECORDED="$ROOT/aqm-spice2/freesolv/deep_ensemble/seed_42"
cd "$ROOT"

echo "[full] 1/4 instrumented seed_42 (200 epochs, RNG-guarded)"
python "$RERUN/instrument_finetune.py" --seed 42 --device cuda --out "$RERUN"

echo "[full] 2/4 same-box ORIGINAL seed_42 (200 epochs) -> $ROOT/box_orig_full"
python aqm-spice2/freesolv/deep_ensemble.py --mode train --seed 42 --device cuda \
    --output_dir "$ROOT/box_orig_full"

echo "[full] 3/4 sanity check vs recorded + same-box originals"
python "$RERUN/sanity_check.py" --seed 42 \
    --references "$RECORDED" "$ROOT/box_orig_full/seed_42"

echo "[full] 4/4 Stage A early read"
python "$RERUN/analyze_stageA.py" --seed 42

echo "[full] Phase B complete."
echo "[full] Review:"
echo "[full]   $RERUN/seed_42/sanity_report_ref0.json  (vs recorded)"
echo "[full]   $RERUN/seed_42/sanity_report_ref1.json  (vs same-box original)"
echo "[full]   $RERUN/seed_42/sanity_summary.json      (self-noise-aware verdict)"
echo "[full]   $RERUN/seed_42/stageA_early_read.json"
echo "[full]   $RERUN/seed_42/stageA_curves.png"

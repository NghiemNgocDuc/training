#!/usr/bin/env bash
# Phase C (Vast GPU box, manual launch after Phase B):
#   1. a SECOND full ORIGINAL deep_ensemble.py seed 42 run on THIS box
#      -> box_orig_full2/seed_42   (exact same command as Phase B step 2)
#   2. Phase C verdict: the instrumented full rerun vs the box's OWN
#      two-original-run noise floor (same rule as Phase A, at full scale):
#        self_noise[m] = |orig_full1 - orig_full2| per metric
#        fixed vs closer baseline = min(|fixed-orig1|, |fixed-orig2|)
#        PASS iff fixed's deviation <= max(2*self_noise, floor) on ALL metrics
#      -> instrumented_rerun/compare_runs/cmp_verdict_full.json
# On PASS -> proceed to full Stage B (4 more seeds).
# On FAIL -> the instrumented run perturbs training beyond this box's own
#            hardware nondeterminism floor; investigate before further runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RERUN="$ROOT/aqm-spice2/freesolv/deep_ensemble/instrumented_rerun"
COMP="$RERUN/compare_runs"
mkdir -p "$COMP"
cd "$ROOT"

echo "[c] 1/2 second same-box ORIGINAL seed_42 (200 epochs) -> $ROOT/box_orig_full2"
python aqm-spice2/freesolv/deep_ensemble.py --mode train --seed 42 --device cuda \
    --output_dir "$ROOT/box_orig_full2"

echo "[c] 2/2 Phase C verdict: fixed vs two same-box originals (self-noise floor)"
python "$RERUN/compare_full_runs.py" \
    --orig1-dir "$ROOT/box_orig_full/seed_42" \
    --orig2-dir "$ROOT/box_orig_full2/seed_42" \
    --fixed-dir "$RERUN/seed_42" \
    --report "$COMP/cmp_verdict_full.json"

echo "[c] Phase C complete."
echo "[c] Review: $COMP/cmp_verdict_full.json"
echo "[c] On PASS: proceed to full Stage B (4 more seeds)."
echo "[c] On FAIL: instrumentation still perturbs training -> investigate."

#!/usr/bin/env bash
# Stage B: instrumented runs for the remaining 4 seeds (123, 7, 2024, 999).
# RUN ONLY AFTER Stage A was reviewed and sanity PASSED on seed 42.
#
#   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_all_seeds.sh \
#       > aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_all_seeds.log 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RERUN="$ROOT/aqm-spice2/freesolv/deep_ensemble/instrumented_rerun"
cd "$ROOT"

for SEED in 123 7 2024 999; do
  echo "[run] ===== seed $SEED ====="
  python "$RERUN/instrument_finetune.py" --seed "$SEED" --device cuda
  python "$RERUN/sanity_check.py" --seed "$SEED"
done

echo "[run] Stage B complete. Combine per-seed epoch_predictions.csv files for the "
echo "     5-seed trajectory analysis (Stage B report follows, see analyze_stageB.py)."

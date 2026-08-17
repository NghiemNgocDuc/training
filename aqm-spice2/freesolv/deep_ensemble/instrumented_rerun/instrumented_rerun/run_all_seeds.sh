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
done

echo "[run] all 4 seeds trained; running the 5-seed trajectory analysis"
python "$RERUN/analyze_stageB.py"

echo "[run] Stage B complete. Review analysis_stageB/stageB_report.json + PNGs."
echo "     (per-seed sanity check is skipped: no original references exist for"
echo "      seeds 123/7/2024/999 - only seed 42 has recorded originals)"

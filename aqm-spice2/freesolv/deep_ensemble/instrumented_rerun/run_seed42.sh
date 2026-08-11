#!/usr/bin/env bash
# Stage A: instrumented single-seed run (seed 42) + sanity check + early read.
# Run on the Vast AI GPU box from the repo root (REPO_ROOT, where
# freesolv_conformers.hdf5 and Data/FreeSolv live).
#
#   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_seed42.sh \
#       > aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_seed42.log 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RERUN="$ROOT/aqm-spice2/freesolv/deep_ensemble/instrumented_rerun"
cd "$ROOT"

echo "[run] root: $ROOT"
echo "[run] python: $(which python) | torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# 1) instrumented fine-tune, seed 42, identical config to the original run
python "$RERUN/instrument_finetune.py" --seed 42 --device cuda

# 2) sanity check vs the ORIGINAL seed_42 metrics/predictions on record
python "$RERUN/sanity_check.py" --seed 42

# 3) early read: gradient-12 vs random-12 from certain-47
python "$RERUN/analyze_stageA.py" --seed 42

echo "[run] Stage A complete. Review seed_42/sanity_report.json + stageA_early_read.json"

#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  FreeSolv Full Pipeline — end-to-end
#  Run from repo root:  bash aqm-spice2/scripts/run_freesolv_pipeline.sh
# ================================================================

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  STEP 1 — Generate conformers (if needed)"
echo "========================================"
if [ ! -f freesolv_conformers.hdf5 ]; then
    # xTB not available on this machine — use MMFF only
    python aqm-spice2/freesolv/freesolv_dataset.py --no_xtb
else
    echo "  freesolv_conformers.hdf5 already exists — skipping"
fi

echo ""
echo "========================================"
echo "  STEP 2 — Zero-shot prediction (Option B)"
echo "========================================"
python aqm-spice2/freesolv/predict_freesolv.py --method B

echo ""
echo "========================================"
echo "  STEP 3 — Analysis & baseline comparison"
echo "========================================"
python aqm-spice2/freesolv/analyze_freesolv.py

echo ""
echo "========================================"
echo "  STEP 4 — Fine-tune (80/20 split)"
echo "========================================"
python aqm-spice2/freesolv/finetune_freesolv.py

echo ""
echo "========================================"
echo "  STEP 5 — 5-fold CV + ensemble + conformer aug"
echo "           + comparison table + parity plot"
echo "========================================"
python aqm-spice2/freesolv/cv_finetune.py --n_conformers 20

echo ""
echo "========================================"
echo "  DONE — all results in aqm-spice2/freesolv/cv_results/"
echo "========================================"

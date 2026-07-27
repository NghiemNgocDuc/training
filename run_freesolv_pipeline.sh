#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  FreeSolv Full Pipeline — end-to-end
#  Usage:  bash run_freesolv_pipeline.sh
# ================================================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "  STEP 1 — Generate conformers (if needed)"
echo "========================================"
if [ ! -f freesolv_conformers.hdf5 ]; then
    # xTB not available on this machine — use MMFF only
    python solvation-gnn/freesolv_dataset.py --no_xtb
else
    echo "  freesolv_conformers.hdf5 already exists — skipping"
fi

echo ""
echo "========================================"
echo "  STEP 2 — Zero-shot prediction (Option B)"
echo "========================================"
python solvation-gnn/predict_freesolv.py --method B

echo ""
echo "========================================"
echo "  STEP 3 — Analysis & baseline comparison"
echo "========================================"
python solvation-gnn/analyze_freesolv.py

echo ""
echo "========================================"
echo "  STEP 4 — Fine-tune (80/20 split)"
echo "========================================"
python solvation-gnn/finetune_freesolv.py

echo ""
echo "========================================"
echo "  STEP 5 — 5-fold CV + ensemble + conformer aug"
echo "           + comparison table + parity plot"
echo "========================================"
python solvation-gnn/cv_finetune.py --n_conformers 20

echo ""
echo "========================================"
echo "  DONE — all results in cv_results/"
echo "========================================"
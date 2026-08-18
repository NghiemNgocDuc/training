#!/usr/bin/env bash
# Exp-DB external generalization from a fresh repo clone (run on Vast GPU).
# Usage:  bash run_expdb_box.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> aqm-spice2/freesolv (script dir resolution base)
RESULTS=verification/expdb_external/results
mkdir -p "$RESULTS"

echo "=== [1/5] split reproducibility gate ==="
python verification/expdb_external/run_vast.py md5check

echo "=== [2/5] FreeSolv 5-fold retrain (surviving regime) ==="
# NOTE: cv_finetune_se.py chdir's to aqm-spice2/ at import time, so --conformers
# and --cache_dir are relative to aqm-spice2/ (--checkpoint_dir/--output_dir are
# script-dir-relative).
python cv_finetune_se.py --no_se --no_multi_agg --n_conformers 5 \
    --conformers ../freesolv_conformers.hdf5 \
    --cache_dir Data/FreeSolv \
    --checkpoint_dir ../aqm-spice2/pipeline/results_full \
    --correction_ckpt stage2_correction.pt \
    --output_dir verification/expdb_external/results/cv_results_retrain \
    2>&1 | tee "$RESULTS/train.log"

echo "=== [3/5] FreeSolv anchor (TTA-5, headline-comparable) ==="
python verification/expdb_external/run_vast.py anchor --model_dir "$RESULTS/cv_results_retrain"

echo "=== [4/5] Exp-DB conformer archive (620 mols, ETKDGv3+MMFF) ==="
python verification/expdb_external/run_vast.py generate

echo "=== [5/5] Exp-DB inference (TTA-5 x 5 folds) ==="
python verification/expdb_external/run_vast.py infer --model_dir "$RESULTS/cv_results_retrain"

echo "=== DONE. Results in aqm-spice2/freesolv/verification/expdb_external/results ==="
ls -la "$RESULTS"

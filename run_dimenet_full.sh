#!/bin/bash
# =============================================================================
#  DimeNet+ FULL PIPELINE (authorized plan, per scratch/fix_report.md):
#    Stage 1: vacuum DimeNetPlus from scratch on full AQM-gas, 4-fold CV
#             (per-fold train/val disjointness + fold stats logged at launch)
#    Stage 2: correction DimeNetPlus on full AQM-sol, dG = E_sol - E_gas target,
#             refs SELF-FIT on its own train split (no stage-1 refs needed),
#             gas-paired conformers cached in memory once
#    Stage 3: FreeSolv single 5-fold CV fine-tune + 20-conformer ensemble
#             (fold-mean target stratification logged at start)
#
#  Usage (Vast Jupyter terminal, repo root):  bash run_dimenet_full.sh [quick]
#  Self-detaches via nohup -> safe to close the laptop. PID + log printed.
#
#  Full dep list audited against STAGE_AB_PROTOCOL.md + actual imports:
#  torch (cu128) | torch_geometric | scikit-learn (stage1 imports KFold) |
#  h5py | numpy | tqdm | scipy (predict/analyze) | pandas+matplotlib
#  (analyze_freesolv) | requests
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QUICK="$1"
LAUNCHER=launcher_dimenet.log
R="$ROOT/aqm-spice2/pipeline/results_full"

# ---- self-detach on first (interactive) call -------------------------------
if [ -z "$DETACHED" ]; then
    DETACHED=1 nohup bash "$0" "$@" > "$LAUNCHER" 2>&1 &
    echo "DimeNet pipeline launched (PID $!) -> logs: $LAUNCHER + stage logs"
    echo "monitor: tail -f $LAUNCHER   |   stage logs: tail -f stage1.log / stage2.log / freesolv_ft.log"
    exit 0
fi

echo "=== $(date) DimeNet+ full pipeline start"
nvidia-smi | head -12
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"

# ---- repo + installs --------------------------------------------------------
git pull -q || echo "  git pull skipped (dirty/offline), continuing with local code"
pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
pip install torch_geometric scikit-learn h5py numpy tqdm scipy pandas matplotlib requests rdkit

# ---- full AQM data (Zenodo, ~3.0 GB; skip if already downloaded) ------------
[ -f AQM-sol-full.hdf5 ] || wget -O AQM-sol-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-sol.hdf5?download=1'
[ -f AQM-gas-full.hdf5 ] || wget -O AQM-gas-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-gas.hdf5?download=1'
ls -lh AQM-*-full.hdf5

# ---- Stage 1: vacuum DimeNetPlus from scratch on AQM-gas, 1-fold (1-day) ----
echo "=== $(date) STAGE 1 (vacuum, 20k structurally, 1-fold <1-day config) start"
S1=""; [ "$QUICK" = "quick" ] && S1="--max_structures 4000 --epochs 30"
if python aqm-spice2/pipeline/train_stage1_vacuum.py \
        --hdf5 AQM-gas-full.hdf5 --k_folds 1 \
        --epochs 24 --lr 0.001 --batchsize 32 --max_structures 20000 \
        --output_dir "$R" $S1 > stage1.log 2>&1; then
    echo "STAGE 1 DONE rc=0"
else
    echo "STAGE 1 FAILED rc=$? -> tail -f stage1.log"; exit 1
fi

# ---- Stage 2: correction DimeNetPlus on AQM-sol (dG = E_sol - E_gas) --------
echo "=== $(date) STAGE 2 (correction, full AQM-sol, dG target) start"
S2=""; [ "$QUICK" = "quick" ] && S2="--max_structures 4000 --epochs 30"
if python aqm-spice2/pipeline/train_stage2_correction.py \
        --hdf5 AQM-sol-full.hdf5 --gas_hdf5 AQM-gas-full.hdf5 \
        --vacuum_ckpt "$R/stage1_fold_1.pt" \
        --epochs 14 --lr 0.001 --batchsize 16 --max_structures 20000 \
        --output_dir "$R" $S2 > stage2.log 2>&1; then
    echo "STAGE 2 DONE rc=0"
else
    echo "STAGE 2 FAILED rc=$? -> tail -f stage2.log"; exit 1
fi

# ---- Stage 3: FreeSolv 5-fold CV fine-tune + 20-conformer ensemble ----------
echo "=== $(date) STAGE 3 (FreeSolv 5-fold CV fine-tune) start"
S3=""; [ "$QUICK" = "quick" ] && S3="--quick_test"
if python aqm-spice2/freesolv/cv_finetune.py \
        --conformers "$ROOT/freesolv_conformers.hdf5" \
        --correction_ckpt "$R/stage2_correction.pt" \
        --checkpoint_dir "$R" \
        --output_dir "$ROOT/aqm-spice2/freesolv/cv_results_full" \
        --n_conformers 5 $S3 > freesolv_ft.log 2>&1; then
    echo "STAGE 3 DONE rc=0"
else
    echo "STAGE 3 FAILED rc=$? -> tail -f freesolv_ft.log"; exit 1
fi

echo "=== $(date) DIMENET PIPELINE DONE"
echo "results: $R/stage1_fold_1.pt + $R/stage2_correction.pt + aqm-spice2/freesolv/cv_results_full/"

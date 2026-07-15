#!/bin/bash
set -e

PYTHON=$(command -v python3 || command -v python)

echo "===== 1. Clone repo ====="
if [ ! -f "solvation-gnn/train_stage1_vacuum.py" ]; then
    if [ ! -d "training" ]; then
        git clone https://github.com/NghiemNgocDuc/training.git
        cd training
    else
        cd training
        git pull
    fi
fi

echo "===== 2. Check GPU ====="
$PYTHON -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| gpus:', torch.cuda.device_count()); torch.cuda.empty_cache()"

echo "===== 3. Install deps ====="
pip install torch_geometric h5py
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

echo "===== 4. Download data ====="
# wget -nc https://zenodo.org/records/10208010/files/AQM-gas.hdf5
# wget -nc https://zenodo.org/records/10208010/files/AQM-sol.hdf5
wget -nc https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5

RESULTS="solvation-gnn/results"
mkdir -p "$RESULTS"

echo "===== 5. Stage 1: Vacuum (quick) ====="
$PYTHON train.py train_stage1_vacuum.py \
    --hdf5 AQM-gas.hdf5 \
    --max_structures 10000 --epochs 30 --batchsize 256 \
    --lr 0.001 --k_folds 1 \
    --output_dir "$RESULTS"

echo "===== 6. Stage 2a: Implicit correction (quick) ====="
$PYTHON train.py train_stage2_correction.py \
    --hdf5 AQM-sol.hdf5 \
    --vacuum_ckpt "$RESULTS/stage1_fold_1.pt" \
    --max_structures 10000 --epochs 30 --batchsize 128 \
    --lr 0.001 \
    --output_dir "$RESULTS"

echo "===== 7. Option A: Scratch baseline (quick) ====="
$PYTHON train.py train_option_a.py \
    --hdf5 AQM-sol.hdf5 \
    --option_b_checkpoint "$RESULTS/stage2_correction.pt" \
    --option_b_vacuum_ckpt "$RESULTS/stage1_fold_1.pt" \
    --max_structures 10000 --epochs 30 --batchsize 128 \
    --lr 0.001 \
    --output_dir "$RESULTS"

echo "===== 8. Stage 2b: Explicit water (quick) ====="
$PYTHON train.py train_stage2b_explicit.py \
    --hdf5 SPICE-2.0.1.hdf5 \
    --vacuum_ckpt "$RESULTS/stage1_fold_1.pt" \
    --implicit_ckpt "$RESULTS/stage2_correction.pt" \
    --max_molecules 100 --max_conformers 5 --epochs 15 \
    --batchsize 128 --lr 0.001 \
    --output_dir "$RESULTS"

echo "===== 9. Evaluate ====="
$PYTHON solvation-gnn/evaluate_model.py \
    --checkpoint "$RESULTS/stage1_fold_1.pt" \
    --hdf5 AQM-gas.hdf5 \
    --max_structures 500 --md_steps 500 \
    --output_dir "$RESULTS"

echo "===== DONE ====="

#!/bin/bash
# =============================================================================
#  MACE (from scratch) FULL PIPELINE — Stage A (random-init MACE on full AQM dG)
#  then Stage B (FreeSolv 5-fold CV fine-tune from the scratch Stage-A weights).
#
#  Usage (Vast Jupyter terminal, repo root):  bash run_mace_full.sh [quick]
#  Self-detaches via nohup -> safe to close the laptop. PID + log printed.
#
#  Full dep list audited against STAGE_AB_PROTOCOL.md + actual imports:
#  torch (cu128) | mace-torch (pulls e3nn/scipy/pyyaml) | h5py | tqdm |
#  numpy | scipy | e3nn | requests   (nothing else is imported at train time)
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
QUICK="$1"
LAUNCHER=launcher_mace.log

# ---- self-detach on first (interactive) call -------------------------------
if [ -z "$DETACHED" ]; then
    DETACHED=1 nohup bash "$0" "$@" > "$LAUNCHER" 2>&1 &
    echo "MACE pipeline launched (PID $!) -> logs: $LAUNCHER + stage logs"
    echo "monitor: tail -f $LAUNCHER   |   stage logs: tail -f stage_a_scratch.log / stage_b.log"
    exit 0
fi

echo "=== $(date) MACE full pipeline (from scratch) start"
nvidia-smi | head -12
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"

# ---- repo + installs --------------------------------------------------------
git pull -q || echo "  git pull skipped (dirty/offline), continuing with local code"
pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
pip install mace-torch h5py tqdm numpy scipy e3nn requests

# ---- full AQM data (Zenodo, ~3.0 GB; skip if already downloaded) ------------
[ -f AQM-sol-full.hdf5 ] || wget -O AQM-sol-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-sol.hdf5?download=1'
[ -f AQM-gas-full.hdf5 ] || wget -O AQM-gas-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-gas.hdf5?download=1'
ls -lh AQM-*-full.hdf5

# ---- Stage A: full training from scratch (random init) ----------------------
echo "=== $(date) STAGE A (from-scratch MACE on full AQM dG) start"
QA=""; [ "$QUICK" = "quick" ] && QA="--quick_test"
if python mace_freesolv/train_stage_a_scratch.py \
        --hdf5_sol AQM-sol-full.hdf5 --hdf5_gas AQM-gas-full.hdf5 \
        --device cuda --epochs 300 --lr 1e-2 --warmup_epochs 20 \
        --patience 40 --batch_size 32 \
        --output_dir mace_freesolv/results_stage_a_scratch \
        $QA > stage_a_scratch.log 2>&1; then
    echo "STAGE A DONE rc=0"
else
    echo "STAGE A FAILED rc=$? -> tail -f stage_a_scratch.log"; exit 1
fi

# ---- Stage B: FreeSolv fine-tune (from scratch Stage-A checkpoint) ----------
echo "=== $(date) STAGE B (FreeSolv 5-fold CV fine-tune) start"
QB=""; [ "$QUICK" = "quick" ] && QB="--quick_test"
if python mace_freesolv/main.py \
        --init_checkpoint mace_freesolv/results_stage_a_scratch/stage_a.pt \
        --device cuda $QB > stage_b.log 2>&1; then
    echo "STAGE B DONE rc=0"
else
    echo "STAGE B FAILED rc=$? -> tail -f stage_b.log"; exit 1
fi

echo "=== $(date) MACE PIPELINE DONE"
echo "results: mace_freesolv/results_stage_a_scratch/stage_a.pt + mace_freesolv/results/ (fold_N/*)"

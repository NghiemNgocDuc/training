#!/bin/bash
# =============================================================================
#  MACE fold-0 ENSEMBLE (GIMS-ready) — 3-seed Stage-B + per-atom dump
#  Trains fold-0 ONLY from the SAME Stage-A checkpoint (only seed changes),
#  then dumps per-atom P_mi^k for all 642 FreeSolv molecules.
#
#  Vast GPU usage (repo root /workspace/training):
#    bash run_mace_ensemble.sh              # full 42,123,999 (self-detaches)
#    bash run_mace_ensemble.sh quick        # 1 seed, 2 epochs, 20 mols (~15 min)
#
#  Monitor (new terminal, same dir):
#    tail -f launcher_mace_ensemble.log     # self-detach + stage markers
#    tail -f mace_ensemble.log              # live training + tqdm bars
#    tail -n 100 mace_ensemble.log          # last 100 lines
#    grep -E "SEED|TEST MAE|dump seed|done" mace_ensemble.log | tail -20
#    nvidia-smi -l 5                        # GPU util
#
#  Results:
#    mace_freesolv/fold0_ensemble/seed_{42,123,999}/model.pt + metrics.json
#    mace_freesolv/fold0_ensemble/peratom_mace_seed{S}.pkl
#    mace_freesolv/fold0_ensemble/mace_node_contributions.csv
#    mace_freesolv/fold0_ensemble/mace_seed_predictions_all642.csv
# =============================================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
MODE="$1"

if [ -z "$DETACHED" ] && [ "$MODE" != "quick" ]; then
    DETACHED=1 nohup bash "$0" "$@" > launcher_mace_ensemble.log 2>&1 &
    echo "MACE ensemble launched (PID $!) -> launcher_mace_ensemble.log + mace_ensemble.log"
    echo "monitor: tail -f launcher_mace_ensemble.log"
    echo "         tail -f mace_ensemble.log"
    exit 0
fi

echo "=== $(date) MACE fold-0 ensemble start (mode=${MODE:-full}) ==="
nvidia-smi | head -14 || true
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"

pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 || pip install --quiet torch
pip install --quiet "mace-torch==0.3.16" h5py tqdm numpy scipy pandas scikit-learn e3nn requests

ls -lh freesolv_conformers.hdf5 || { echo "MISSING freesolv_conformers.hdf5"; exit 1; }
ls -lh mace_freesolv/results_stage_a_scratch/stage_a.pt || { echo "MISSING Stage-A checkpoint"; exit 1; }
ls aqm-spice2/aqm-spice2/freesolv/cv_results_full/fold_0/*_ids.json || echo "WARNING: frozen split missing -> script reconstructs"

if [ "$MODE" = "quick" ]; then
  echo "=== QUICK TEST: 1 seed, 2 epochs, 20 mols ==="
  python3 mace_freesolv/fold0_ensemble.py \
    --seeds 42 --quick_test --device cuda \
    --output_dir mace_freesolv/fold0_ensemble_quick \
    2>&1 | tee mace_ensemble_quick.log
  echo "=== QUICK DONE ==="
  echo "check: tail -n 60 mace_ensemble_quick.log"
  echo "expect: [probe] raw MACE output keys + node key + max|sum(P)-E| < 1e-3"
  exit 0
fi

echo "=== FULL: seeds 42,123,999 ==="
python3 mace_freesolv/fold0_ensemble.py \
  --seeds 42,123,999 --device cuda \
  --epochs 500 --patience 50 --lr 1e-4 --batch_size 32 --warmup_epochs 10 \
  --output_dir mace_freesolv/fold0_ensemble \
  2>&1 | tee mace_ensemble.log

echo "=== $(date) MACE ENSEMBLE DONE ==="
echo "results: mace_freesolv/fold0_ensemble/"
ls -lh mace_freesolv/fold0_ensemble/ | tee -a mace_ensemble.log
ls -lh mace_freesolv/fold0_ensemble/seed_*/model.pt | tee -a mace_ensemble.log

#!/bin/bash
# =============================================================================
#  AIMNet2-2025 fold-0 ENSEMBLE — 5-seed fine-tune + per-atom dump
#  Fine-tunes official aimnet2-2025 member-0 weights (HF isayevlab/aimnet2-2025,
#  MIT license; auto-downloaded) on fold-0 ONLY, seeds 42,123,7,2024,999,
#  energy-only MSE, then dumps per-atom P_mi^k for all 642 FreeSolv molecules.
#
#  Vast GPU usage (repo root /workspace/training):
#    bash run_aimnet_ensemble.sh              # full 5 seeds (self-detaches)
#    bash run_aimnet_ensemble.sh quick        # 5 seeds, 2 epochs, 20 mols (~15 min)
#
#  Monitor (new terminal, same dir):
#    tail -f launcher_aimnet_ensemble.log     # self-detach + stage markers
#    tail -f aimnet_ensemble.log              # live training + tqdm bars
#    tail -n 100 aimnet_ensemble.log          # last 100 lines
#    grep -E "SEED|TEST MAE|dump seed|done" aimnet_ensemble.log | tail -20
#    nvidia-smi -l 5                        # GPU util
#
#  Results:
#    aimnet_freesolv/fold0_ensemble/seed_{42,123,7,2024,999}/model.pt + metrics.json
#    aimnet_freesolv/fold0_ensemble/peratom_aimnet_seed{S}.pkl
#    aimnet_freesolv/fold0_ensemble/aimnet_node_contributions.csv
#    aimnet_freesolv/fold0_ensemble/aimnet_seed_predictions_all642.csv
# =============================================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
MODE="$1"

if [ -z "$DETACHED" ] && [ "$MODE" != "quick" ]; then
    DETACHED=1 nohup bash "$0" "$@" > launcher_aimnet_ensemble.log 2>&1 &
    echo "AIMNet2 ensemble launched (PID $!) -> launcher_aimnet_ensemble.log + aimnet_ensemble.log"
    echo "monitor: tail -f launcher_aimnet_ensemble.log"
    echo "         tail -f aimnet_ensemble.log"
    exit 0
fi

echo "=== $(date) AIMNet2 fold-0 ensemble start (mode=${MODE:-full}) ==="
nvidia-smi | head -14 || true
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"

pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 || pip install --quiet torch
pip install --quiet "aimnet[hf,train]" h5py tqdm numpy scipy pandas scikit-learn

ls -lh freesolv_conformers.hdf5 || { echo "MISSING freesolv_conformers.hdf5"; exit 1; }
ls aqm-spice2/aqm-spice2/freesolv/cv_results_full/fold_0/*_ids.json || echo "WARNING: frozen split missing"
python3 -c "from aimnet.calculators import AIMNet2Calculator; print('aimnet import OK')"

if [ "$MODE" = "quick" ]; then
  echo "=== QUICK TEST: 5 seeds, 2 epochs, 20 mols ==="
  python3 aimnet_freesolv/aimnet_ensemble.py \
    --seeds 42,123,7,2024,999 --quick_test --device cuda \
    --output_dir aimnet_freesolv/fold0_ensemble_quick \
    2>&1 | tee aimnet_ensemble_quick.log
  echo "=== QUICK DONE ==="
  echo "check: tail -n 60 aimnet_ensemble_quick.log"
  echo "expect: gate max|sum(P)-E| < 1e-3"
  exit 0
fi

echo "=== FULL: seeds 42,123,7,2024,999 ==="
python3 aimnet_freesolv/aimnet_ensemble.py \
  --seeds 42,123,7,2024,999 --device cuda \
  --epochs 200 --patience 30 --lr 1e-4 --weight_decay 1e-5 \
  --output_dir aimnet_freesolv/fold0_ensemble \
  2>&1 | tee aimnet_ensemble.log

echo "=== $(date) AIMNet2 ENSEMBLE DONE ==="
echo "results: aimnet_freesolv/fold0_ensemble/"
ls -lh aimnet_freesolv/fold0_ensemble/ | tee -a aimnet_ensemble.log
ls -lh aimnet_freesolv/fold0_ensemble/seed_*/model.pt | tee -a aimnet_ensemble.log

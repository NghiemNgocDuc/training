#!/usr/bin/env bash
# Transfer fine-tune (FlexiSol + Guthrie-subset) for vast.ai GPU instance.
# Mirrors run_dimenet_full.sh pattern: self-detaches, progress via tqdm+logs.
# Usage:
#   bash run_transfer_vast.sh [guthrie_csv_path]
#   tail -f transfer_launcher.log
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GUTHRIE_CSV="${1:-$ROOT/guthrie_subset_445.csv}"
LAUNCHER="$ROOT/transfer_launcher.log"

if [ -z "$DETACHED" ]; then
  DETACHED=1 nohup bash "$0" "$@" > "$LAUNCHER" 2>&1 &
  echo "transfer launched (PID $!) -> $LAUNCHER"
  echo "monitor: tail -f $LAUNCHER; splits: ls out_transfer/*/ | head"
  exit 0
fi

echo "=== $(date) transfer fine-tune start"
nvidia-smi | head -12 || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"
git pull -q || echo "git pull skipped, continuing local code"
pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
pip install torch_geometric scikit-learn h5py numpy tqdm scipy pandas matplotlib rdkit seaborn 2>&1 | tail -2

echo "=== STEP 1 splits only (inspect before training)"
python sandbox_transfer_finetune.py --dataset both --guthrie_csv "$GUTHRIE_CSV" \
  --outdir "$ROOT/out_transfer" --device cuda --make_splits_only
echo "INSPECT NOW: cat out_transfer/flexisol/flexisol_train_ids.json | head -c 300; echo"
ls -lh "$ROOT/out_transfer"/*/*.json

echo "=== STEP 2 full fine-tune (detached, tqdm in transfer.log)"
python sandbox_transfer_finetune.py --dataset both --guthrie_csv "$GUTHRIE_CSV" \
  --outdir "$ROOT/out_transfer" --device cuda \
  --epochs 200 --patience 30 --lr 1e-4 --weight_decay 1e-5 --batch_size 8 \
  --n_boot 10000 > transfer.log 2>&1

echo "=== STEP 3 view results"
cat out_transfer/flexisol/flexisol_summary.json
cat out_transfer/guthrie/guthrie_summary.json
echo "csvs: out_transfer/flexisol/flexisol_finetuned_test.csv out_transfer/guthrie/guthrie_finetuned_test.csv"
echo "=== DONE $(date)"

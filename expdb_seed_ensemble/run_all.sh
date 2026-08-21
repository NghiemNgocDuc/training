#!/bin/bash
# Exp-DB seed-ensemble GIMS — end-to-end (train 5 seeds -> per-atom inference -> GIMS)
# Run from inside expdb_seed_ensemble/ on a CUDA machine.
set -e
cd "$(dirname "$0")"

echo "============================================================="
echo "[0/4] preflight $(date)"
echo "============================================================="
python preflight.py

echo "============================================================="
echo "[1/4] training 5 seed variants of fold-0  $(date)"
echo "============================================================="
for SEED in 42 123 7 2024 999; do
  if [ -f "results_seeds/finetuned_seed${SEED}.pt" ]; then
    echo "[skip] finetuned_seed${SEED}.pt already exists"
  else
    echo "--- training seed ${SEED} ---"
    python train_seed.py --seed ${SEED} 2>&1 | tee results_seeds_train_seed${SEED}.log
  fi
done

echo "============================================================="
echo "[2/4] per-atom, per-seed inference  $(date)"
echo "============================================================="
if [ -f "results_seeds/peratom_seed999.pkl" ]; then
  echo "[skip] per-atom pickles already exist (delete to redo)"
else
  python infer_peratom.py --seeds 42,123,7,2024,999 2>&1 | tee infer_peratom.log
fi

echo "============================================================="
echo "[3/4] GIMS on Exp-DB  $(date)"
echo "============================================================="
python gims_expdb.py 2>&1 | tee gims_expdb.log

echo "============================================================="
echo "[4/4] DONE  $(date)"
echo "Outputs in results_seeds/:"
echo "  gims_expdb_report.json / gims_expdb_results.csv"
echo "  peratom_seed*.pkl / finetuned_seed*.pt / train_meta_seed*.json"
echo "============================================================="

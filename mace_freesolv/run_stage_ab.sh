#!/bin/bash
# Stage A + Stage B: MACE-OFF23 two-stage transfer for FreeSolv.
# Run on the Vast instance from the repo root:
#   bash mace_freesolv/run_stage_ab.sh
# Logs: stage_a.log, stage_b.log (repo root)
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SOL="AQM-sol-full.hdf5"
GAS="AQM-gas-full.hdf5"

if [ ! -f "$SOL" ] || [ ! -f "$GAS" ]; then
    echo "ERROR: missing $SOL / $GAS in repo root."
    echo "Upload from local machine:"
    echo "  scp $SOL $GAS battery:/workspace/" 2>/dev/null || echo "  (adjust host if not 'battery')"
    exit 1
fi

echo "=== STAGE A: MACE-OFF23 fine-tune on AQM (dG = E_sol - E_gas) ==="
nohup python -u mace_freesolv/train_stage_a.py \
    --hdf5_sol "$SOL" --hdf5_gas "$GAS" --device cuda \
    > stage_a.log 2>&1
echo "Stage A done: mace_freesolv/results_stage_a/stage_a.pt (see stage_a.log)"

echo "=== STAGE B: FreeSolv 5-fold CV from Stage-A weights ==="
nohup python -u mace_freesolv/main.py \
    --init_checkpoint mace_freesolv/results_stage_a/stage_a.pt --device cuda \
    > stage_b.log 2>&1
echo "Stage B done (see stage_b.log)"

tail -5 stage_b.log

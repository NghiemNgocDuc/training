#!/bin/bash
# Two-stage MACE-OFF23 pipeline for FreeSolv (run on the Vast instance).
# Stage A: fine-tune MACE-OFF23 on AQM hydration free energies (dG = E_sol - E_gas).
# Stage B: FreeSolv 5-fold CV initialized from the Stage-A checkpoint.
#
# Usage (from repo root, either interactively or detached):
#   bash mace_freesolv/run_stage_ab.sh [SOL_HDF5] [GAS_HDF5]
#
# The whole pipeline runs detached (nohup) and logs to <repo>/stage_ab.log,
# so you can close the SSH session. Watch progress with:
#   tail -f stage_ab.log
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SOL="${1:-AQM-sol-full.hdf5}"
GAS="${2:-AQM-gas-full.hdf5}"

for f in "$SOL" "$GAS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found in $REPO"
        echo ""
        echo "Option 1 - download the full AQM files directly on the instance:"
        echo "  wget -O AQM-gas-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-gas.hdf5?download=1'"
        echo "  wget -O AQM-sol-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-sol.hdf5?download=1'"
        echo ""
        echo "Option 2 - upload from your local machine (slower):"
        echo "  scp AQM-sol-full.hdf5 AQM-gas-full.hdf5 battery:/workspace/training/"
        exit 1
    fi
done

LOG="$REPO/stage_ab.log"
export SOL GAS

nohup bash -c '
    set -e
    echo "===== STAGE A START ($(date)) ====="
    python -u mace_freesolv/train_stage_a.py --hdf5_sol "$SOL" --hdf5_gas "$GAS" --device cuda
    echo "===== STAGE B START ($(date)) ====="
    python -u mace_freesolv/main.py --init_checkpoint mace_freesolv/results_stage_a/stage_a.pt --device cuda
    echo "===== PIPELINE DONE ($(date)) ====="
' > "$LOG" 2>&1 &

echo "Pipeline launched (PID $!) - logged to $LOG"
echo "Watch:  tail -f $LOG"
echo "Result: grep 'test:' $LOG | tail -20"

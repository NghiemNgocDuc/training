#!/bin/bash
# Two-stage MACE-OFF23 pipeline for FreeSolv (run on the Vast instance).
# Stage A: fine-tune MACE-OFF23 on AQM hydration free energies (dG = E_sol - E_gas).
# Stage B: FreeSolv 5-fold CV initialized from the Stage-A checkpoint.
#
# From-scratch run: pulls latest code, wipes stale Stage A/B outputs, relaunches.
#
# Usage (from repo root, either interactively or detached):
#   bash mace_freesolv/run_stage_ab.sh [SOL_HDF5] [GAS_HDF5]
#   NORMALIZE=1 bash mace_freesolv/run_stage_ab.sh   # Run B: --normalize_targets (scale pinned to 1.0)
#
# The whole pipeline runs detached (nohup) and logs to <repo>/stage_ab.log,
# so you can close the SSH session. Watch progress with:
#   tail -f stage_ab.log
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "===== FROM-SCRATCH LAUNCH ($(date)) ====="
git pull --ff-only

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

# From-scratch: remove any stale Stage A/B outputs from previous runs.
rm -rf mace_freesolv/results_stage_a mace_freesolv/results_stage_a_b mace_freesolv/results

NORMALIZE="${NORMALIZE:-0}"
EXTRA_ARGS=""
STAGE_A_PT="mace_freesolv/results_stage_a/stage_a.pt"
if [ "$NORMALIZE" = "1" ]; then
    EXTRA_ARGS="--normalize_targets --output_dir mace_freesolv/results_stage_a_b"
    STAGE_A_PT="mace_freesolv/results_stage_a_b/stage_a.pt"
    echo "Mode: Run B (--normalize_targets)"
else
    echo "Mode: Run A (plain calibration)"
fi

LOG="$REPO/stage_ab.log"
export SOL GAS EXTRA_ARGS STAGE_A_PT

nohup bash -c '
    set -e
    echo "===== STAGE A START ($(date)) ====="
    python -u mace_freesolv/train_stage_a.py --hdf5_sol "$SOL" --hdf5_gas "$GAS" --device cuda \
        --lr 3e-4 --warmup_epochs 20 --patience 20 $EXTRA_ARGS
    echo "===== STAGE B START ($(date)) ====="
    python -u mace_freesolv/main.py --init_checkpoint "$STAGE_A_PT" --device cuda
    echo "===== PIPELINE DONE ($(date)) ====="
' > "$LOG" 2>&1 &

echo "Pipeline launched (PID $!) - logged to $LOG"
echo "Watch:  tail -f $LOG"
echo "Result: grep 'test:' $LOG | tail -20"

#!/usr/bin/env bash
#
# Main-run only: launches the AQSOL S(=O) pipeline DETACHED under nohup.
# Preconditions (done before this, per the doc):
#   cd /workspace/training && git pull
#   pip install tqdm h5py numpy torch torch_geometric rdkit
# Then just:   bash aqm-spice2/freesolv/experimental_aqsol_so/run_aqsol_so.sh
# Monitor:     tail -f aqsol_so.so.log
# Rollback if unwanted: rm -rf aqm-spice2/freesolv/experimental_aqsol_so

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/training}"
LOG="$REPO_DIR/aqsol_so.log"
SB="$REPO_DIR/aqm-spice2/freesolv/experimental_aqsol_so"

cd "$REPO_DIR"
[ -f "$SB/prepare_aqsol_so.py" ] || { echo "cannot find sandbox at $SB - run git pull first"; exit 1; }

nohup bash -c '
    set -euo pipefail
    cd "$1"
    sb="$2"
    echo ">>> [1/3] prepare (SMILES filter + tar scan + hdf5 build)"
    python "$sb/prepare_aqsol_so.py"
    echo ">>> [2/3] train (detached; per-epoch batch bar + loss, then TTA bar)"
    python "$sb/finetune_aqsol_so.py" --mode train --device cuda
    echo ">>> [3/3] re-report from saved best ckpt"
    python "$sb/finetune_aqsol_so.py" --mode eval --device cuda
    echo ">>> DONE - full log saved to this file"
    ' _ "$REPO_DIR" "$SB" > "$LOG" 2>&1 &

echo ">>> detached nohup run started (pid $!). log: $LOG"
echo ">>> monitor with: tail -f $LOG"
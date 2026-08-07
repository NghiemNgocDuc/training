#!/usr/bin/env bash
#
# One-shot runner for the AQSOL S(=O) supplement experiment.
# Usage on the GPU box:  bash aqm-spice2/freesolv/experimental_aqsol_so/run_aqsol_so.sh
#
# Does, in order, DETACHED under nohup (safe to close the laptop):
#   0. pip install any missing libs
#   1. git pull (grabs the latest code incl. this script)
#   2. prepare (100k-row SMILES filter + tar scan + ~4k geo extractions -> data/aqsol_so.hdf5)
#   3. train on GPU (best-val ckpt saved to output/)
#   4. re-report from the saved best ckpt
# Then tails the log so you can watch (and re-tail later with
#   tail -f aqsol_so.sh.log ).
#
# Rollback if unwanted: rm -rf aqm-spice2/freesolv/experimental_aqsol_so

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/training}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_DIR/aqsol_so.log"

echo ">>> repo: $REPO_DIR"
echo ">>> script dir: $SCRIPT_DIR"
echo ">>> log: $LOG"
cd "$REPO_DIR"

# Fresh repo? clone it. Otherwise refresh so the script itself is current.
if [ ! -d .git ]; then
    echo ">>> no repo at $REPO_DIR, cloning training.git"
    git clone https://github.com/NghiemNgocDuc/training.git .
fi
git pull --ff-only || true

# 0. libs (tqdm is the only new one; everything else is from the Frag20 run)
echo ">>> [0/4] installing libs"
pip install -q tqdm h5py numpy torch torch_geometric rdkit

SB="$SCRIPT_DIR"
# Repo clone may place the sandbox elsewhere; resolve relative to repo root.
[ -f "$SB/finetune_aqsol_so.py" ] || SB="$REPO_DIR/aqm-spice2/freesolv/experimental_aqsol_so"
[ -f "$SB/finetune_aqsol_so.py" ] || { echo "cannot locate sandbox scripts"; exit 1; }

# Whole pipeline detached under nohup: prepare -> train -> re-report.
nohup bash -c '
    set -euo pipefail
    cd "$1"
    sb="$2"
    [ -f "$sb/prepare_aqsol_so.py" ] || sb="$1/aqm-spice2/freesolv/experimental_aqsol_so"
    [ -f "$sb/prepare_aqsol_so.py" ] || { echo "cannot locate sandbox scripts"; exit 1; }
    echo ">>> [1/4] prepare (SMILES filter + tar scan + hdf5 build)"
    python "$sb/prepare_aqsol_so.py"
    echo ">>> [2/4] train (detached; per-epoch batch bar + loss, then TTA bar)"
    python "$sb/finetune_aqsol_so.py" --mode train --device cuda
    echo ">>> [3/4] re-report from saved best ckpt"
    python "$sb/finetune_aqsol_so.py" --mode eval --device cuda
    echo ">>> [4/4] DONE - full log saved to this file"
    ' _ "$REPO_DIR" "$SCRIPT_DIR" > "$LOG" 2>&1 &

echo ">>> started detached (pid $!). Monitoring the log..."
tail -f "$LOG"
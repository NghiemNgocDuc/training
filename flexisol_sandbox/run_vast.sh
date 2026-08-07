#!/usr/bin/env bash
# FlexiSol sandbox: train the 5-seed DimeNet+ ensemble on FlexiSol-water (cuda),
# then run the Approach-3 coverage diagnostic on the aggregate.
#
# Usage on Vast (run detached, never block the shell):
#   cd flexisol_sandbox
#   nohup bash run_vast.sh > all.log 2>&1 &
#   tail -f all.log
#
# If out/ensemble_full already exists, only the coverage step runs.

set -u
cd "$(dirname "$0")"

ENSEMBLE_DIR=out/ensemble_full
AGG=out/ensemble_full/aggregate/per_molecule.csv

if [ ! -f "$AGG" ]; then
    echo "[run_vast] training 5-seed ensemble -> $ENSEMBLE_DIR"
    python train_flexisol_ensemble.py --device cuda --out "$ENSEMBLE_DIR"
else
    echo "[run_vast] $AGG already exists, skipping training"
fi

echo "[run_vast] coverage diagnostic"
python approach3_coverage.py --aggregate "$AGG"

echo "[run_vast] done"

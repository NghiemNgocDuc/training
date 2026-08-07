#!/usr/bin/env bash
# FlexiSol sandbox: build dataset (if missing), train the 5-seed DimeNet+
# ensemble on FlexiSol-water (cuda), then run the Approach-3 coverage
# diagnostic on the aggregate.
#
# Usage on Vast (run detached, never block the shell):
#   cd flexisol_sandbox
#   nohup bash run_vast.sh > all.log 2>&1 &
#   tail -f all.log
#
# The built dataset (out/, data/) is gitignored, so the first run also
# fetches the grimme-lab/flexisol repo and rebuilds flexisol_water.hdf5.

set -u
cd "$(dirname "$0")"

LABELS=out/labels.json
ENSEMBLE_DIR=out/ensemble_full
AGG=out/ensemble_full/aggregate/per_molecule.csv

if [ ! -f "$LABELS" ]; then
    echo "[run_vast] dataset missing -> fetch + build + inspect"
    python fetch_flexisol.py --dest data/flexisol_repo
    python build_hdf5.py --repo data/flexisol_repo --out out
    python inspect_data.py --out out
else
    echo "[run_vast] $LABELS exists, skipping dataset build"
fi

if [ ! -f "$AGG" ]; then
    echo "[run_vast] training 5-seed ensemble -> $ENSEMBLE_DIR"
    python train_flexisol_ensemble.py --device cuda --out "$ENSEMBLE_DIR" --data out
else
    echo "[run_vast] $AGG already exists, skipping training"
fi

echo "[run_vast] coverage diagnostic"
python approach3_coverage.py --aggregate "$AGG"

echo "[run_vast] done"

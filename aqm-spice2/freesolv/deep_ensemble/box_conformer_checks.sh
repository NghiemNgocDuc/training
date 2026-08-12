#!/bin/bash
# Run the gradient-12 conformer-instability + provenance checks on the Vast box,
# where the training-time conformers, RDKit, and environment live.
# Usage (from repo root /workspace/training): bash aqm-spice2/freesolv/deep_ensemble/box_conformer_checks.sh
set -e
cd "$(dirname "$0")/../../.."   # repo root (Data/FreeSolv + freesolv_conformers.hdf5 live here)
echo "repo root: $(pwd)"
ls Data/FreeSolv/database.json freesolv_conformers.hdf5
python -c "import rdkit; print('rdkit', rdkit.__version__)"

D=aqm-spice2/freesolv/deep_ensemble
O=$D/gradient12_conformer_provenance_check
python "$D/check1_conformer_instability.py" > "$O/check1_box.log" 2>&1
python "$D/check3_provenance.py" > "$O/check3_box.log" 2>&1
python "$D/check2_conformer_seed_sensitivity.py" > "$O/check2_box.log" 2>&1
echo "--- check1 ---"; grep -E "^(calibration|\[ck1\])" "$O/check1_box.log" || true
echo "--- check3 ---"; grep -E "^\[prov\]" "$O/check3_box.log" || true
echo "--- check2 ---"; grep -E "^\[ck2\]" "$O/check2_box.log" || true
echo "done -> $O/"

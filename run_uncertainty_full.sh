#!/bin/bash
# =============================================================================
#  UNCERTAINTY-REFINEMENT FULL RUN (Approach 0 + Approach 1)
#  ======================================================
#  What this DOES:
#    1. install everything needed (pip)
#    2. clone/pull the repo
#    3. make sure ALL datasets/checkpoints exist (download/copy what we can,
#       clearly error on what must be brought up by hand)
#    4. run Approach 0 (iteration-variance go/no-go) + Approach 1
#       (uncertainty-weighted re-train, alpha sweep) on GPU
#  What this does NOT do: it does NOT re-train stage 1 / stage 2 / the 5
#  ensemble members. All of those are already-trained artifacts that are
#  LOADED (the "use what we have" rule).
#
#  Usage (Vast Jupyter terminal):  bash run_uncertainty_full.sh [quick]
#  Self-detaches via nohup -> safe to close the laptop. PID + log printed.
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QUICK="$1"
LAUNCHER=launcher_uncertainty.log

# ---- self-detach on first (interactive) call -------------------------------
if [ -z "$DETACHED" ]; then
    DETACHED=1 nohup bash "$0" "$@" > "$LAUNCHER" 2>&1 &
    echo "Uncertainty run launched (PID $!) -> logs: $LAUNCHER"
    echo "monitor: tail -f $LAUNCHER"
    exit 0
fi

echo "=== $(date) uncertainty full run start"
nvidia-smi | head -12
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"

# ---- 1. install everything needed -------------------------------------------
echo "=== $(date) pip installs"
pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
pip install torch_geometric scikit-learn h5py numpy tqdm scipy rdkit

# ---- 2. clone/pull the repo -------------------------------------------------
if [ ! -d "$ROOT/.git" ]; then
    echo "no .git in $ROOT -> cloning repo into $ROOT/training"
    git clone https://github.com/NghiemNgocDuc/training.git "$ROOT/training"
    ROOT="$ROOT/training"
    cd "$ROOT"
fi
git pull -q || echo "  git pull skipped (dirty/offline), continuing with local code"

# ---- paths (after cluster/pull so $ROOT is final) ----------------------------
EXP="$ROOT/aqm-spice2/freesolv/experimental_uncertainty_refine"
ENS_DIR="$ROOT/aqm-spice2/freesolv/deep_ensemble"
AGG_DIR="$ENS_DIR/aggregate"
SPLIT_DIR="$ROOT/aqm-spice2/aqm-spice2/freesolv/cv_results_full/fold_0"
CORR_CKPT="$ROOT/aqm-spice2/aqm-spice2/pipeline/results_full/stage2_correction.pt"
CONFORMERS="$ROOT/freesolv_conformers.hdf5"
LABELS="$ROOT/Data/FreeSolv/database.json"
PER_MOL="$AGG_DIR/per_molecule.csv"

# ---- 3. datasets/checkpoints: ensure everything is present -------------------
MISSING=""
note() { echo "    [check] $1"; }

echo "=== $(date) artifact checks (nothing is trained here)"
note "experiment folder ($EXP)"
[ -f "$EXP/common.py" ] && [ -f "$EXP/approach0_iteration_variance.py" ] && [ -f "$EXP/approach1_weighted_retrain.py" ] \
    || MISSING="$MISSING experimental_uncertainty_refine/ scripts"

note "FreeSolv labels ($LABELS)"
if [ ! -f "$LABELS" ]; then
    mkdir -p "$ROOT/Data/FreeSolv"
    # preferred: copy from the committed copy in the repo; else download canonical
    if [ -f "$ROOT/aqm-spice2/Data/FreeSolv/database.json" ]; then
        cp "$ROOT/aqm-spice2/Data/FreeSolv/database.json" "$LABELS"
        echo "    copied labels from committed aqm-spice2/Data/FreeSolv copy"
    else
        echo "    downloading canonical FreeSolv labels (MobleyLab)"
        wget -O "$LABELS" 'https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json'
    fi
fi
[ -f "$LABELS" ] || MISSING="$MISSING database.json"

note "conformers ($CONFORMERS)"
[ -f "$CONFORMERS" ] || MISSING="$MISSING freesolv_conformers.hdf5"

note "frozen fold-0 split ($SPLIT_DIR)"
[ -f "$SPLIT_DIR/train_ids.json" ] && [ -f "$SPLIT_DIR/val_ids.json" ] && [ -f "$SPLIT_DIR/test_ids.json" ] \
    || MISSING="$MISSING fold_0/*_ids.json"

note "stage-2 init ckpt ($CORR_CKPT)"
[ -f "$CORR_CKPT" ] || MISSING="$MISSING stage2_correction.pt"

note "5 ensemble member ckpts ($ENS_DIR/seed_*/ensemble_seed*.pt)"
[ -f "$ENS_DIR/seed_42/ensemble_seed42.pt" ] && [ -f "$ENS_DIR/seed_123/ensemble_seed123.pt" ] \
    && [ -f "$ENS_DIR/seed_7/ensemble_seed7.pt" ] && [ -f "$ENS_DIR/seed_2024/ensemble_seed2024.pt" ] \
    && [ -f "$ENS_DIR/seed_999/ensemble_seed999.pt" ] \
    || MISSING="$MISSING 5x ensemble_seed*.pt (already-trained members)"

note "per-molecule aggregate ($PER_MOL)"
[ -f "$PER_MOL" ] || MISSING="$MISSING per_molecule.csv"

if [ -n "$MISSING" ]; then
    echo ""
    echo "ERROR: these required artifacts are missing:$MISSING"
    echo "  The 5 ensemble checkpoints + per_molecule.csv are TRAINED ARTIFACTS"
    echo "  with no public URL - bring them up from the local machine, e.g.:"
    echo "    scp -r aqm-spice2/freesolv/deep_ensemble/ user@box:.../aqm-spice2/freesolv/"
    echo "  (everything else above either comes with the repo or was auto-fetched)"
    exit 1
fi
echo "    all artifacts present - using existing checkpoints (no re-training)"

# ---- 4. run Approach 0 -------------------------------------------------------
echo "=== $(date) APPROACH 0: iteration-variance (loads seed-42 member + adapter)"
A0=""; [ "$QUICK" = "quick" ] && A0="--smoke"
if python "$EXP/approach0_iteration_variance.py" \
        --device cuda \
        --ensemble_dir "$ENS_DIR" \
        --conformers "$CONFORMERS" \
        --labels_json "$LABELS" \
        --per_molecule "$PER_MOL" \
        --split_dir "$SPLIT_DIR" \
        --output_dir "$EXP/output/approach0" $A0 > approach0.log 2>&1; then
    echo "APPROACH 0 DONE rc=0"
else
    echo "APPROACH 0 FAILED rc=$? -> tail -f approach0.log"; exit 1
fi

# ---- 5. run Approach 1 -------------------------------------------------------
echo "=== $(date) APPROACH 1: uncertainty-weighted re-train (alpha sweep)"
A1=""; [ "$QUICK" = "quick" ] && A1="--smoke"
if python "$EXP/approach1_weighted_retrain.py" \
        --device cuda \
        --ensemble_dir "$ENS_DIR" \
        --conformers "$CONFORMERS" \
        --labels_json "$LABELS" \
        --per_molecule "$PER_MOL" \
        --split_dir "$SPLIT_DIR" \
        --correction_ckpt "$CORR_CKPT" \
        --output_dir "$EXP/output/approach1" \
        --seeds 42 \
        --alphas 0.0 0.5 1.0 2.0 $A1 > approach1.log 2>&1; then
    echo "APPROACH 1 DONE rc=0"
else
    echo "APPROACH 1 FAILED rc=$? -> tail -f approach1.log"; exit 1
fi

echo "=== $(date) UNCERTAINTY RUN DONE"
echo "results: $EXP/output/approach0/report.json + $EXP/output/approach1/report.json"

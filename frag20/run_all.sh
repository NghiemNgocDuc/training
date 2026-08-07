#!/usr/bin/env bash
# ==============================================================================
# Frag20 from-scratch pretraining -> FreeSolv fine-tune : FULL PIPELINE (1 bash)
# ==============================================================================
# ONE-TIME RUN (fresh Linux/Vast box):
#
#   git clone https://github.com/NghiemNgocDuc/training.git
#   cd training
#   nohup bash frag20/run_all.sh > frag20/pipeline.log 2>&1 &
#   tail -f frag20/pipeline.log
#
# Steps (self-contained; the Frag20 dataset + FreeSolv database.json are
# downloaded here / auto-downloaded by the scripts - nothing else required):
#   0) FreeSolv database.json   (curl if missing; NOT in git)
#   1) prepare_frag20_scratch   (downloads Frag20 tar + split CSVs, builds
#                                data/frag20_full.hdf5 + labels + report)
#   2) pretrain_stage1_frag20   (vacuum / gas energy)   -> output/stage1_scratch.pt
#   3) pretrain_stage2_frag20   (solvation correction)  -> output/stage2_scratch.pt
#   4) finetune_freesolv        (fold-0 + 5-conf TTA)   -> output_finetune/
#
# Resume behaviour: each step is skipped when its output already exists, so
# rerunning after a crash continues from where it left off. Optional first
# argument starts at a given step: prepare|stage1|stage2|finetune.
set -euo pipefail

cd "$(dirname "$0")"              # frag20/
REPO_ROOT="$(cd .. && pwd)"       # training/
LOG_DIR="$REPO_ROOT/frag20/logs"
mkdir -p "$LOG_DIR"
echo "[$(date +%H:%M:%S)] repo root: $REPO_ROOT"

FROM_STEP="${1:-prepare}"
case "$FROM_STEP" in
  prepare|stage1|stage2|finetune) ;;
  *) echo "usage: $0 [prepare|stage1|stage2|finetune]"; exit 2 ;;
esac

# ---------- device detection ----------
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE=cpu
  if python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null | grep -q cuda; then
    DEVICE=cuda
  fi
fi
echo "[$(date +%H:%M:%S)] device: $DEVICE (override with DEVICE=cuda|cpu bash frag20/run_all.sh)"

# ---------- per-step launcher (foreground; whole script runs under nohup) ----
run_step() {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] ===== STEP $name ===== log: logs/${name}.log"
  echo "[$(date +%H:%M:%S)] cmd: $*"
  python "$@" > "$logf" 2>&1 || {
    echo "[$(date +%H:%M:%S)] FAILED: $name - tail -f logs/${name}.log"; exit 1; }
  echo "[$(date +%H:%M:%S)] done: $name"
}

# ---------- 0) FreeSolv labels (not in git -> curl once) ----------
if [[ "$FROM_STEP" == prepare || "$FROM_STEP" == stage1 ]]; then
  mkdir -p "$REPO_ROOT/Data/FreeSolv"
  if [[ ! -f "$REPO_ROOT/Data/FreeSolv/database.json" ]]; then
    echo "[$(date +%H:%M:%S)] downloading FreeSolv database.json"
    curl -fsSL -o "$REPO_ROOT/Data/FreeSolv/database.json" \
      https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json
  fi
  echo "[$(date +%H:%M:%S)] FreeSolv labels OK: Data/FreeSolv/database.json"
fi

# ---------- 1) prepare dataset (auto-downloads Frag20 tar + CSVs) ------------
if [[ "$FROM_STEP" == prepare || "$FROM_STEP" == stage1 ]]; then
  if [[ -f "data/frag20_full.hdf5" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP prepare: data/frag20_full.hdf5 exists"
  else
    run_step prepare python prepare_frag20_scratch.py --geom qm
  fi
fi

# ---------- 2) stage 1: vacuum (gas energy) from scratch ---------------------
if [[ "$FROM_STEP" == prepare || "$FROM_STEP" == stage1 ]]; then
  if [[ -f "output/stage1_scratch.pt" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP stage1: output/stage1_scratch.pt exists"
  else
    run_step stage1 python pretrain_stage1_frag20.py --device "$DEVICE" --output_dir output
  fi
fi

# ---------- 3) stage 2: solvation correction (wat-gas), frozen vacuum --------
if [[ "$FROM_STEP" == prepare || "$FROM_STEP" == stage1 || "$FROM_STEP" == stage2 ]]; then
  if [[ -f "output/stage2_scratch.pt" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP stage2: output/stage2_scratch.pt exists"
  else
    run_step stage2 python pretrain_stage2_frag20.py --device "$DEVICE" \
      --stage1_ckpt output/stage1_scratch.pt \
      --stage1_refs output/stage1_scratch_refs.json \
      --output_dir output
  fi
fi

# ---------- 4) fine-tune on frozen fold-0 + TTA eval -------------------------
if [[ "$FROM_STEP" == prepare || "$FROM_STEP" == stage1 || "$FROM_STEP" == stage2 || "$FROM_STEP" == finetune ]]; then
  if [[ -f "output_finetune/metrics.json" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP finetune: output_finetune/metrics.json exists"
  else
    run_step finetune python finetune_freesolv.py --device "$DEVICE" \
      --init_ckpt output/stage2_scratch.pt \
      --output_dir output_finetune \
      --n_conformers 5
  fi
fi

echo "[$(date +%H:%M:%S)] ============ ALL DONE ============"
echo "results:"
echo "  output/stage1_scratch.pt + refs"
echo "  output/stage2_scratch.pt"
echo "  output_finetune/metrics.json + predictions.csv"
echo "logs: logs/ (prepare.log stage1.log stage2.log finetune.log)"

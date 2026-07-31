"""One-shot progress snapshot for the two-stage pipeline.

Reads <repo>/stage_ab.log (written by mace_freesolv/run_stage_ab.sh) and prints
current stage, epoch progress, epoch time, ETA, last val MAE and fold results.

Usage (repo root, anytime, even in a fresh terminal):
    python mace_freesolv/status.py
"""

import os
import re
import sys

EV_TO_KCAL = 23.0605

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "stage_ab.log")

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\(\s*([\d.]+)%\)\s*\|.*?Val MAE:\s*([\d.]+).*?([\d.]+)s/epoch \| ETA ~([\d.]+)([hm])")
FOLD_RE = re.compile(r"\s*FOLD (\d+)")
FOLD_TEST_RE = re.compile(r"Fold (\d+) test:\s+MAE=([\d.]+) RMSE=([\d.]+)")
MARKER_RE = re.compile(r"===== (STAGE [AB]|PIPELINE) (START|DONE) ")


def main():
    if not os.path.exists(LOG):
        print(f"No {LOG} found.")
        print("Start the pipeline first: bash mace_freesolv/run_stage_ab.sh")
        sys.exit(1)

    with open(LOG, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    stage = "starting"
    epochs = []
    folds_started = []
    fold_results = []

    for line in lines:
        m = MARKER_RE.search(line)
        if m:
            kind, verb = m.group(1), m.group(2)
            if kind == "STAGE A" and verb == "START":
                stage = "STAGE A (AQM fine-tune)"
            elif kind == "STAGE B" and verb == "START":
                stage = "STAGE B (FreeSolv CV)"
            elif kind == "PIPELINE":
                stage = "DONE"
        m = EPOCH_RE.search(line)
        if m:
            cur, total, pct, val_mae, sec, eta_val, eta_unit = m.groups()
            eta_h = float(eta_val) / 60.0 if eta_unit == "m" else float(eta_val)
            epochs.append((int(cur), int(total), float(pct), float(val_mae),
                           float(sec), eta_h))
        m = FOLD_RE.search(line)
        if m and line.strip().startswith("FOLD"):
            folds_started.append(int(m.group(1)))
        m = FOLD_TEST_RE.search(line)
        if m:
            fold_results.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    print("=" * 60)
    print(f"  Status: {stage}")
    print("=" * 60)

    if epochs:
        cur, total, pct, val_mae, sec, eta_h = epochs[-1]
        print(f"  Epoch {cur}/{total} ({pct:.1f}%)")
        print(f"  {sec:.1f}s/epoch | ETA ~{eta_h:.1f}h (early stopping may end sooner)")
        print(f"  Last val MAE: {val_mae:.3f} kcal/mol")
    else:
        print("  No epoch finished yet (still loading data / model, or calibrating).")
        print("  Check the raw tail: tail -n 20 stage_ab.log")

    if stage == "STAGE B (FreeSolv CV)":
        print(f"  Folds completed: {len(fold_results)}/5")
    if fold_results:
        maes = [r[1] for r in fold_results]
        print(f"  Fold test MAEs: " + " ".join(f"fold{r[0]}: {r[1]:.3f}" for r in fold_results))
        if len(maes) > 1:
            mean = sum(maes) / len(maes)
            std = (sum((m - mean) ** 2 for m in maes) / (len(maes) - 1)) ** 0.5
            print(f"  Mean {mean:.3f} +/- {std:.3f} kcal/mol (so far)")

    if stage == "DONE":
        print("\n  PIPELINE COMPLETE - results in mace_freesolv/results/ and results_stage_a/")
    elif folds_started:
        print(f"\n  Tail for details: tail -n 20 stage_ab.log")


if __name__ == "__main__":
    main()

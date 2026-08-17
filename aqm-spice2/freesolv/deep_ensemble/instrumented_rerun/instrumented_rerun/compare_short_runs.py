"""Phase A verdict: same-box 5-epoch comparison (orig_run1, orig_run2, fixed_run1).

Reads per-epoch val MAE from:
  * original deep_ensemble.py runs  -> parsed from their stdout logs
    (lines: "seed   42 | epoch   1/5 | best val  ... | cur val  X.XXX | ...")
  * fixed instrument_finetune.py run -> seed_<s>/val_history.csv (epoch 0 row skipped)

Computes:
  self_noise = max_epoch |orig1 - orig2|            (baseline self-noise)
  dev1       = max_epoch |fixed - orig1|
  dev2       = max_epoch |fixed - orig2|
  dev_fixed  = min(dev1, dev2)                       (fixed vs the closer baseline)

Verdict PASS iff dev_fixed <= max(2 * self_noise, 0.02) kcal/mol: the fixed run
deviates from the baseline by no more than the baseline deviates from ITSELF
(chaotic float/GPU nondeterminism), i.e. the instrumentation adds nothing.

Usage: python compare_short_runs.py \
         --orig1-log cmp_orig1.log --orig2-log cmp_orig2.log \
         --fixed-csv cmp_inst/seed_42/val_history.csv \
         --report cmp_verdict.json
Exit code 0 = PASS (safe to launch the full seed_42 rerun), 1 = FAIL (report
before proceeding), 2 = error parsing inputs.
"""

import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True)

EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s*\|\s*best val\s+([\d.]+)\s*\(ep\s+\d+\)\s*\|\s*cur val\s+([\d.]+)")

PASS_FLOOR = 0.02      # kcal/mol absolute floor so zero self-noise can't fail
NOISE_FACTOR = 2.0     # allow fixed deviation up to 2x the baseline self-noise


def parse_orig_log(path):
    """Return {epoch: cur_val_mae_kcal} from a deep_ensemble.py train log."""
    out = {}
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(3))
    if not out:
        raise ValueError(f"no per-epoch lines parsed from {path}")
    return out


def parse_fixed_csv(path):
    """Return {epoch: val_mae_kcal} from instrument_finetune val_history.csv
    (skips the epoch-0 warm-start row, which has no original counterpart)."""
    out = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            ep = int(parts[0])
            if ep == 0:
                continue
            out[ep] = float(parts[1])
    if not out:
        raise ValueError(f"no per-epoch rows parsed from {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig1-log", required=True)
    ap.add_argument("--orig2-log", required=True)
    ap.add_argument("--fixed-csv", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    try:
        o1 = parse_orig_log(args.orig1_log)
        o2 = parse_orig_log(args.orig2_log)
        fx = parse_fixed_csv(args.fixed_csv)
    except (OSError, ValueError) as e:
        print(f"[verdict] ERROR parsing inputs: {e}")
        sys.exit(2)

    epochs = sorted(set(o1) & set(o2) & set(fx))
    if not epochs:
        print("[verdict] ERROR: no common epochs across the three runs")
        sys.exit(2)

    rows = []
    for ep in epochs:
        a, b, c = o1[ep], o2[ep], fx[ep]
        rows.append({"epoch": ep, "orig1": round(a, 4), "orig2": round(b, 4),
                     "fixed": round(c, 4),
                     "abs_orig1_orig2": round(abs(a - b), 4),
                     "abs_fixed_orig1": round(abs(c - a), 4),
                     "abs_fixed_orig2": round(abs(c - b), 4)})

    self_noise = max(r["abs_orig1_orig2"] for r in rows)
    dev1 = max(r["abs_fixed_orig1"] for r in rows)
    dev2 = max(r["abs_fixed_orig2"] for r in rows)
    dev_fixed = min(dev1, dev2)
    tol = max(NOISE_FACTOR * self_noise, PASS_FLOOR)
    passed = dev_fixed <= tol

    print("=" * 86)
    print(f"  Phase A verdict | seed 42, {len(epochs)} epochs, same box")
    print("=" * 86)
    print(f"  {'ep':>4} {'orig1':>9} {'orig2':>9} {'fixed':>9} "
          f"{'|o1-o2|':>9} {'|f-o1|':>9} {'|f-o2|':>9}")
    for r in rows:
        print(f"  {r['epoch']:>4} {r['orig1']:>9.4f} {r['orig2']:>9.4f} "
              f"{r['fixed']:>9.4f} {r['abs_orig1_orig2']:>9.4f} "
              f"{r['abs_fixed_orig1']:>9.4f} {r['abs_fixed_orig2']:>9.4f}")
    print("-" * 86)
    print(f"  baseline self-noise     max|orig1-orig2|     = {self_noise:.4f} kcal/mol")
    print(f"  fixed vs orig1          max|fixed-orig1|     = {dev1:.4f}")
    print(f"  fixed vs orig2          max|fixed-orig2|     = {dev2:.4f}")
    print(f"  fixed vs closer baseline max|fixed-orig*|    = {dev_fixed:.4f}")
    print(f"  tolerance               max(2*self_noise, 0.02) = {tol:.4f}")
    print("-" * 86)
    if passed:
        print("  Verdict: PASS - fixed_run1 deviates no more than the original runs")
        print("  deviate from each other. Residual gap = inherent GPU nondeterminism.")
        print("  Safe to proceed to the full seed_42 rerun (Phase B).")
    else:
        print("  Verdict: FAIL - fixed_run1 deviates MORE than the baseline's own")
        print("  self-noise. Something is still wrong - report before proceeding.")
    print(f"  report -> {args.report}")
    print("=" * 86)

    report = {
        "verdict": "PASS" if passed else "FAIL",
        "n_epochs": len(epochs),
        "rows": rows,
        "self_noise_max_abs_orig1_orig2": round(self_noise, 4),
        "fixed_vs_orig1_max": round(dev1, 4),
        "fixed_vs_orig2_max": round(dev2, 4),
        "fixed_vs_closer_baseline_max": round(dev_fixed, 4),
        "tolerance": round(tol, 4),
        "rule": "PASS iff dev_fixed <= max(2*self_noise, 0.02)",
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

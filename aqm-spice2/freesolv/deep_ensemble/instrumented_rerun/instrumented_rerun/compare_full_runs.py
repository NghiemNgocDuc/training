"""Phase C verdict: is the instrumented full seed_42 rerun within the SAME BOX's
own run-to-run noise floor, measured by TWO full ORIGINAL (unmodified)
deep_ensemble.py runs on this box?

Same rule as Phase A (compare_short_runs.py), applied at full scale per metric:

    self_noise[m]    = |orig1[m] - orig2[m]|        (same-box original vs itself)
    dev1/dev2        = |fixed - orig1|, |fixed - orig2|
    dev_nearest[m]   = min(dev1, dev2)
    tolerance[m]     = max(2 * self_noise[m], floor[m])
    PASS per metric  = dev_nearest[m] <= tolerance[m]
    VERDICT          = PASS iff ALL metrics pass

Floors keep a metric from passing on an unmeasurably tiny self-noise, same
spirit as Phase A's max(2*self_noise, 0.02). Metrics come from metrics.json +
predictions.csv in each run dir (final metrics, not per-epoch: the ORIGINAL
runs only log best-val/early-stop, and per-epoch MAE early in training is
dominated by the chaotic regime, which the originals themselves don't
reproduce - the verdict should rest on what the run finally delivers).

Usage:
  python compare_full_runs.py \
      --orig1-dir <box_orig_full/seed_42> \
      --orig2-dir <box_orig_full2/seed_42> \
      --fixed-dir <instrumented_rerun/seed_42> \
      --report cmp_verdict_full.json

Exit 0 = PASS (proceed to full Stage B), 1 = FAIL (instrumentation still
perturbs training beyond hardware nondeterminism -> investigate before any
further runs), 2 = input error.
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

# (metrics.json key, short tag, floor)
METRICS = [
    ("test_mae_single_conf_kcal",  "mae_single",     0.02),
    ("test_rmse_single_conf_kcal", "rmse_single",    0.03),
    ("test_mae_tta_kcal",          "mae_tta",        0.02),
    ("test_rmse_tta_kcal",         "rmse_tta",       0.03),
    ("best_val_mae_kcal",          "best_val_mae",   0.02),
    ("best_val_epoch",             "best_val_epoch", 20.0),
    ("early_stop_epoch",           "early_stop_epoch", 20.0),
]

# predictions.csv-derived per-molecule TTA metrics (tag, floor)
PRED_METRICS = [
    ("preds_mean_abs_diff", 0.05),
    ("preds_max_abs_diff",  0.25),
    ("preds_spearman",      0.01),   # dev = |rho1 - rho2|, smaller is better
]


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def load_preds(path):
    rows = {}
    with open(path) as f:
        next(f)
        for line in f:
            mid, p, e = line.rstrip("\n").split(",")
            rows[mid] = (float(p), float(e))
    return rows


def pred_stats(path):
    import numpy as np
    from scipy.stats import spearmanr
    rows = load_preds(path)
    mids = sorted(rows)
    vals = np.array([rows[m][0] for m in mids])
    # only the mean-abs-diff / max-abs-diff / spearman are used; the reference
    # "original vs original" and "fixed vs nearest original" deltas are computed
    # by the caller over the same molecule set.
    return vals, mids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig1-dir", required=True)
    ap.add_argument("--orig2-dir", required=True)
    ap.add_argument("--fixed-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    for d in (args.orig1_dir, args.orig2_dir, args.fixed_dir):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            print(f"ERROR: no metrics.json in {d}")
            sys.exit(2)
        if not os.path.exists(os.path.join(d, "predictions.csv")):
            print(f"ERROR: no predictions.csv in {d}")
            sys.exit(2)

    m1 = load_metrics(os.path.join(args.orig1_dir, "metrics.json"))
    m2 = load_metrics(os.path.join(args.orig2_dir, "metrics.json"))
    mf = load_metrics(os.path.join(args.fixed_dir, "metrics.json"))

    v1, mids1 = pred_stats(os.path.join(args.orig1_dir, "predictions.csv"))
    v2, mids2 = pred_stats(os.path.join(args.orig2_dir, "predictions.csv"))
    vf, midsf = pred_stats(os.path.join(args.fixed_dir, "predictions.csv"))
    common = sorted(set(mids1) & set(mids2) & set(midsf))
    if len(common) < len(mids1):
        print(f"ERROR: molecule overlap orig1/orig2/fixed = {len(common)}/{len(mids1)}")
        sys.exit(2)
    idx1 = [mids1.index(m) for m in common]
    idx2 = [mids2.index(m) for m in common]
    idxf = [midsf.index(m) for m in common]

    rows = []
    for key, tag, floor in METRICS:
        if key not in m1 or key not in m2 or key not in mf:
            print(f"ERROR: metric {key} missing in a metrics.json")
            sys.exit(2)
        o1, o2, fx = m1[key], m2[key], mf[key]
        self_noise = abs(o1 - o2)
        dev1, dev2 = abs(fx - o1), abs(fx - o2)
        dev_nearest = min(dev1, dev2)
        tol = max(2 * self_noise, floor)
        rows.append((tag, o1, o2, fx, self_noise, dev1, dev2, dev_nearest, tol))

    import numpy as np
    from scipy.stats import spearmanr
    r1 = float(spearmanr(v1[idx1], vf[idxf]).statistic)   # fixed vs orig1
    r2 = float(spearmanr(v2[idx2], vf[idxf]).statistic)   # fixed vs orig2
    r12 = float(spearmanr(v1[idx1], v2[idx2]).statistic)  # orig1 vs orig2 (self-noise)
    md1 = float(np.abs(v1[idx1] - vf[idxf]).mean())
    md2 = float(np.abs(v2[idx2] - vf[idxf]).mean())
    mx1 = float(np.abs(v1[idx1] - vf[idxf]).max())
    mx2 = float(np.abs(v2[idx2] - vf[idxf]).max())
    pm_self = float(np.abs(v1[idx1] - v2[idx2]).mean())
    px_self = float(np.abs(v1[idx1] - v2[idx2]).max())
    sn_rho = 1.0 - r12                    # originals' self-noise on agreement scale
    dev_rho1, dev_rho2 = 1.0 - r1, 1.0 - r2
    p_rows = [
        ("preds mean|d|", 0.0, 0.0, 0.0, pm_self, md1, md2, min(md1, md2),
         max(2 * pm_self, 0.05)),
        ("preds max|d|", 0.0, 0.0, 0.0, px_self, mx1, mx2, min(mx1, mx2),
         max(2 * px_self, 0.25)),
        ("preds spearman", 0.0, 0.0, 0.0, sn_rho, dev_rho1, dev_rho2,
         min(dev_rho1, dev_rho2), max(2 * sn_rho, 0.01)),
    ]

    print("=" * 90)
    print("  Phase C verdict | full seed 42, same box | self-noise = 2 full ORIGINAL runs")
    print("=" * 90)
    print(f"    {'metric':<16} {'orig1':>10} {'orig2':>10} {'fixed':>10} "
          f"{'|o1-o2|':>9} {'|f-o1|':>9} {'|f-o2|':>9} {'nearest':>9} {'tol':>9} {'within':>7}")
    for tag, o1, o2, fx, sn, d1, d2, dn, tol in rows:
        w = dn <= tol
        print(f"    {tag:<16} {o1:>10.4f} {o2:>10.4f} {fx:>10.4f} {sn:>9.4f} "
              f"{d1:>9.4f} {d2:>9.4f} {dn:>9.4f} {tol:>9.4f} {'PASS' if w else 'FAIL':>7}")
    for tag, o1, o2, fx, sn, d1, d2, dn, tol in p_rows:
        w = dn <= tol
        print(f"    {tag:<16} {o1:>10.4f} {o2:>10.4f} {fx:>10.4f} {sn:>9.4f} "
              f"{d1:>9.4f} {d2:>9.4f} {dn:>9.4f} {tol:>9.4f} {'PASS' if w else 'FAIL':>7}")
    print("-" * 90)
    print("  baseline self-noise        max|orig1-orig2|   (per metric, above)")
    print("  fixed vs closer baseline   min(|fixed-orig1|, |fixed-orig2|) per metric")
    print("  tolerance                  max(2*self_noise, floor) per metric (Phase A rule)")
    print("  note: preds rows - only |o1-o2| / |f-o1| / |f-o2| / nearest / tol columns apply")
    print("-" * 90)

    results = rows + [(tag, o1, o2, fx, sn, d1, d2, dn, tol)
                      for tag, o1, o2, fx, sn, d1, d2, dn, tol in p_rows]
    passed = all(dn <= tol for _, _, _, _, _, _, _, dn, tol in results)
    fails = [tag for tag, _, _, _, _, _, _, dn, tol in results if dn > tol]
    if passed:
        verdict = ("PASS - fixed full rerun deviates from the nearer same-box "
                   "original no more than the box's own two original runs deviate "
                   "from each other. Instrumentation is within hardware noise - "
                   "safe to proceed to full Stage B (4 more seeds).")
    else:
        verdict = ("FAIL - fixed full rerun deviates beyond the same box's "
                   "two-original-run noise floor on: " + ", ".join(fails) +
                   ". Instrumentation still perturbs training beyond hardware "
                   "nondeterminism - investigate before any further runs.")
    print(f"  Verdict: {verdict}")
    print(f"  report -> {args.report}")
    print("=" * 90)

    report = {
        "protocol": "Phase C - full-scale self-noise-aware verdict (same rule as Phase A)",
        "orig1_dir": args.orig1_dir, "orig2_dir": args.orig2_dir,
        "fixed_dir": args.fixed_dir,
        "per_metric": {
            tag: {"orig1": round(o1, 4), "orig2": round(o2, 4), "fixed": round(fx, 4),
                  "self_noise": round(sn, 4), "dev_orig1": round(d1, 4),
                  "dev_orig2": round(d2, 4), "dev_nearest": round(dn, 4),
                  "tolerance": round(tol, 4), "within_tolerance": dn <= tol}
            for tag, o1, o2, fx, sn, d1, d2, dn, tol in results},
        "verdict": "PASS" if passed else "FAIL",
        "failing_metrics": fails,
        "note": ("tolerance = max(2*self_noise, floor) per metric; dev_nearest = "
                 "min(|fixed-orig1|, |fixed-orig2|)"),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

"""Stage A sanity check: instrumented seed_42 vs the ORIGINAL seed_42 on record.

Compares final-epoch results:
  * frozen-split md5 (must match exactly)
  * single-conf test MAE / RMSE
  * TTA test MAE / RMSE
  * best_val_epoch, early_stop_epoch
  * per-molecule TTA predictions (mean/max abs diff, Spearman)

Verdict FAIL => the instrumentation (or the environment) changed training
behavior - report and STOP before Stage B, per protocol.

Usage: python sanity_check.py --seed 42 [--instrumented <dir>]
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))          # .../deep_ensemble/instrumented_rerun
_deep_ensemble = os.path.dirname(_script_dir)                     # .../deep_ensemble (original outputs)
_freesolv = os.path.dirname(_deep_ensemble)                       # .../freesolv (deep_ensemble.py lives here)
if _freesolv not in sys.path:
    sys.path.insert(0, _freesolv)

ORIGINAL_DIR = os.path.join(_deep_ensemble, "seed_42")

# thresholds for "very closely reproduce" (kcal/mol unless noted)
T_MAE_DIFF = 0.05      # |single-conf MAE diff|
T_RMSE_DIFF = 0.08
T_TTA_MAE_DIFF = 0.05
T_TTA_RMSE_DIFF = 0.08
T_EPOCH_DIFF = 20      # best-val epoch may shift a few epochs (nondeterminism)
T_PRED_MEAN_DIFF = 0.05
T_PRED_MAX_DIFF = 0.25
T_SPEARMAN = 0.98


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


def main():
    import numpy as np
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instrumented", default=_script_dir,
                    help="instrumented_rerun root (contains seed_<seed>/)")
    ap.add_argument("--original_root", default=_deep_ensemble,
                    help="deep_ensemble/ root (contains original seed_<seed>/)")
    args = ap.parse_args()

    orig_seed = os.path.join(args.original_root, f"seed_{args.seed}")
    new_seed = os.path.join(args.instrumented, f"seed_{args.seed}")

    m_orig = load_metrics(os.path.join(orig_seed, "metrics.json"))
    m_new = load_metrics(os.path.join(new_seed, "metrics.json"))
    p_orig = load_preds(os.path.join(orig_seed, "predictions.csv"))
    p_new = load_preds(os.path.join(new_seed, "predictions.csv"))

    checks = {}
    checks["split_md5_exact"] = m_orig["split_md5"] == m_new["split_md5"]

    def diff(name, a, b, tol, higher_worse=True):
        d = abs(a - b)
        checks[name] = {
            "original": a, "instrumented": b, "abs_diff": round(d, 4),
            "within_tolerance": d <= tol, "tolerance": tol,
        }

    diff("test_mae_single_conf_kcal", m_orig["test_mae_single_conf_kcal"],
         m_new["test_mae_single_conf_kcal"], T_MAE_DIFF)
    diff("test_rmse_single_conf_kcal", m_orig["test_rmse_single_conf_kcal"],
         m_new["test_rmse_single_conf_kcal"], T_RMSE_DIFF)
    diff("test_mae_tta_kcal", m_orig["test_mae_tta_kcal"],
         m_new["test_mae_tta_kcal"], T_TTA_MAE_DIFF)
    diff("test_rmse_tta_kcal", m_orig["test_rmse_tta_kcal"],
         m_new["test_rmse_tta_kcal"], T_TTA_RMSE_DIFF)
    diff("best_val_mae_kcal", m_orig["best_val_mae_kcal"],
         m_new["best_val_mae_kcal"], 0.05)
    diff("best_val_epoch", m_orig["best_val_epoch"], m_new["best_val_epoch"],
         T_EPOCH_DIFF)
    diff("early_stop_epoch", m_orig["early_stop_epoch"], m_new["early_stop_epoch"],
         T_EPOCH_DIFF)

    mids = sorted(set(p_orig) & set(p_new))
    po = np.array([p_orig[m][0] for m in mids])
    pn = np.array([p_new[m][0] for m in mids])
    md = np.abs(po - pn)
    rho, pval = spearmanr(po, pn)
    checks["per_molecule_tta_preds"] = {
        "n": len(mids), "mean_abs_diff": round(float(md.mean()), 4),
        "max_abs_diff": round(float(md.max()), 4),
        "spearman": round(float(rho), 4), "spearman_p": float(pval),
        "mean_within_tol": bool(md.mean() <= T_PRED_MEAN_DIFF),
        "max_within_tol": bool(md.max() <= T_PRED_MAX_DIFF),
        "spearman_within_tol": bool(rho >= T_SPEARMAN),
    }

    hard = checks["split_md5_exact"]
    soft = [checks[k]["within_tolerance"] for k in
            ("test_mae_single_conf_kcal", "test_rmse_single_conf_kcal",
             "test_mae_tta_kcal", "test_rmse_tta_kcal", "best_val_mae_kcal",
             "best_val_epoch", "early_stop_epoch")]
    pm = checks["per_molecule_tta_preds"]
    pred_ok = pm["mean_within_tol"] and pm["max_within_tol"] and pm["spearman_within_tol"]
    passed = hard and all(soft) and pred_ok
    checks["verdict"] = "PASS" if passed else "FAIL"
    checks["fail_reason"] = ("split md5 mismatch - wrong split loaded!" if not hard else
                             "training behavior diverged from the original run "
                             "(see per-check flags)" if not all(soft) or not pred_ok
                             else "")

    with open(os.path.join(new_seed, "sanity_report.json"), "w") as f:
        json.dump(checks, f, indent=2)

    print(f"=== Sanity check seed_{args.seed}: {"PASS" if passed else "FAIL"} ===")
    print(f"  split_md5 exact match: {checks['split_md5_exact']}")
    print(f"  single-conf test: MAE {m_orig['test_mae_single_conf_kcal']:.4f} -> "
          f"{m_new['test_mae_single_conf_kcal']:.4f} "
          f"(|d|={checks['test_mae_single_conf_kcal']['abs_diff']})")
    print(f"  RMSE             : {m_orig['test_rmse_single_conf_kcal']:.4f} -> "
          f"{m_new['test_rmse_single_conf_kcal']:.4f}")
    print(f"  TTA test: MAE {m_orig['test_mae_tta_kcal']:.4f} -> "
          f"{m_new['test_mae_tta_kcal']:.4f}  RMSE {m_orig['test_rmse_tta_kcal']:.4f} "
          f"-> {m_new['test_rmse_tta_kcal']:.4f}")
    print(f"  best val MAE {m_orig['best_val_mae_kcal']:.4f} (ep "
          f"{m_orig['best_val_epoch']}) -> {m_new['best_val_mae_kcal']:.4f} (ep "
          f"{m_new['best_val_epoch']}); early stop ep "
          f"{m_orig['early_stop_epoch']} -> {m_new['early_stop_epoch']}")
    print(f"  per-mol TTA preds (n={pm['n']}): mean|d|={pm['mean_abs_diff']:.4f} "
          f"max|d|={pm['max_abs_diff']:.4f} spearman={pm['spearman']}")
    if passed:
        print("  Verdict: PASS - instrumentation did not change training behavior.")
    else:
        print(f"  Verdict: FAIL - {checks['fail_reason']}")
        print("  STOP before Stage B per protocol.")
    print(f"  report -> {os.path.join(new_seed, 'sanity_report.json')}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
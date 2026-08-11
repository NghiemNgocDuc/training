"""Stage A/B sanity check: instrumented seed_42 vs one or more ORIGINAL runs.

Single-reference usage (unchanged protocol):
  python sanity_check.py --seed 42            # vs deep_ensemble/seed_42 (recorded)

Dual-reference usage (same-box Phase B, self-noise-aware):
  python sanity_check.py --seed 42 \
      --references <recorded_seed_dir> <same_box_orig_seed_dir>

With two references the baseline is no longer treated as noiseless: the drift
between the two ORIGINAL runs measures the environment's self-noise (torch/
cuDNN/GPU float nondeterminism), and the recorded-reference tolerances are
loosened to max(T, 1.5 * drift) per metric. The SAME-BOX reference keeps the
original tight tolerances - if the instrumented run can't match a run of the
original script in the same environment, something is really wrong.

Reports written under the instrumented seed dir:
  sanity_report.json       vs reference[0] (recorded; backward-compatible name)
  sanity_report_box.json   vs reference[1] (same-box original)
  sanity_summary.json      self-noise-aware verdict (when 2 references)

Verdict PASS requires: split md5 exact vs both refs, same-box fixed-tolerance
pass, recorded adaptive-tolerance pass, and per-molecule prediction agreement
vs both refs.
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

# thresholds for "very closely reproduce" (kcal/mol unless noted)
T_MAE_DIFF = 0.05      # |single-conf MAE diff|
T_RMSE_DIFF = 0.08
T_TTA_MAE_DIFF = 0.05
T_TTA_RMSE_DIFF = 0.08
T_EPOCH_DIFF = 20      # best-val epoch may shift a few epochs (nondeterminism)
T_PRED_MEAN_DIFF = 0.05
T_PRED_MAX_DIFF = 0.25
T_SPEARMAN = 0.98

SOFT_METRICS = [("test_mae_single_conf_kcal", "mae_single", T_MAE_DIFF),
                ("test_rmse_single_conf_kcal", "rmse_single", T_RMSE_DIFF),
                ("test_mae_tta_kcal", "mae_tta", T_TTA_MAE_DIFF),
                ("test_rmse_tta_kcal", "rmse_tta", T_TTA_RMSE_DIFF),
                ("best_val_mae_kcal", "best_val_mae", 0.05),
                ("best_val_epoch", "best_val_epoch", T_EPOCH_DIFF),
                ("early_stop_epoch", "early_stop_epoch", T_EPOCH_DIFF)]


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


def compare_one(orig_seed, new_seed, tol_scale=1.0, drift=None):
    """Compare instrumented run in new_seed against original run in orig_seed.

    drift: dict {metric_key: abs_diff_between_originals} for self-noise-aware
    loosening; effective tolerance = max(base_tol, 1.5 * drift[key]).
    Returns a checks dict (schema-compatible with the single-ref protocol).
    """
    import numpy as np
    from scipy.stats import spearmanr

    m_orig = load_metrics(os.path.join(orig_seed, "metrics.json"))
    m_new = load_metrics(os.path.join(new_seed, "metrics.json"))
    p_orig = load_preds(os.path.join(orig_seed, "predictions.csv"))
    p_new = load_preds(os.path.join(new_seed, "predictions.csv"))

    checks = {}
    checks["reference"] = os.path.basename(orig_seed)
    checks["split_md5_exact"] = m_orig["split_md5"] == m_new["split_md5"]

    def diff(name, a, b, base_tol):
        d = abs(a - b)
        eff_tol = base_tol
        if drift is not None:
            dr = drift.get(name, 0.0)
            eff_tol = max(base_tol, 1.5 * dr)
        checks[name] = {
            "original": a, "instrumented": b, "abs_diff": round(d, 4),
            "within_tolerance": d <= eff_tol,
            "tolerance_base": base_tol,
            "tolerance_effective": round(eff_tol, 4),
            "drift_between_originals": None if drift is None else round(drift.get(name, 0.0), 4),
        }

    for key, tag, tol in SOFT_METRICS:
        if key in m_orig and key in m_new:
            diff(key, m_orig[key], m_new[key], tol)

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
    return checks


def eval_checks(checks, adaptive=False):
    """PASS/FAIL for one reference comparison. adaptive=True uses effective
    (self-noise-loosened) tolerances for soft metrics; fixed otherwise."""
    hard = checks["split_md5_exact"]
    soft = [checks[k]["within_tolerance"] for k, _, _ in SOFT_METRICS
            if k in checks]
    pm = checks["per_molecule_tta_preds"]
    pred_ok = pm["mean_within_tol"] and pm["max_within_tol"] and pm["spearman_within_tol"]
    passed = hard and all(soft) and pred_ok
    reason = ("split md5 mismatch - wrong split loaded!" if not hard else
              "training behavior diverged from the original run "
              "(see per-check flags)" if not all(soft) or not pred_ok else "")
    return passed, reason


def print_checks(checks, label):
    m = checks
    pm = m["per_molecule_tta_preds"]
    print(f"  [{label}] reference={m['reference']}  split_md5_exact={m['split_md5_exact']}")
    for key, tag, _ in SOFT_METRICS:
        if key in m:
            c = m[key]
            print(f"    {tag:<22} {c['original']:>10.4f} -> {c['instrumented']:>10.4f} "
                  f"|d|={c['abs_diff']:>8.4f} within={c['within_tolerance']} "
                  f"(tol={c['tolerance_effective']})")
    print(f"    per-mol TTA preds (n={pm['n']}): mean|d|={pm['mean_abs_diff']:.4f} "
          f"max|d|={pm['max_abs_diff']:.4f} spearman={pm['spearman']} "
          f"ok={pm['mean_within_tol'] and pm['max_within_tol'] and pm['spearman_within_tol']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instrumented", default=_script_dir,
                    help="instrumented_rerun root (contains seed_<seed>/)")
    ap.add_argument("--original_root", default=_deep_ensemble,
                    help="deep_ensemble/ root (contains original seed_<seed>/)")
    ap.add_argument("--references", nargs="+", default=None,
                    help="seed dirs to compare against (each with metrics.json + "
                         "predictions.csv). Default: <original_root>/seed_<seed>.")
    args = ap.parse_args()

    new_seed = os.path.join(args.instrumented, f"seed_{args.seed}")
    if args.references:
        refs = [os.path.abspath(r) for r in args.references]
    else:
        refs = [os.path.join(args.original_root, f"seed_{args.seed}")]
    if not all(os.path.exists(os.path.join(r, "metrics.json")) for r in refs):
        print(f"ERROR: missing metrics.json in references: "
              f"{[r for r in refs if not os.path.exists(os.path.join(r, 'metrics.json'))]}")
        sys.exit(2)

    # baseline self-noise: drift between the ORIGINAL runs themselves
    drift = None
    if len(refs) >= 2:
        m_a = load_metrics(os.path.join(refs[0], "metrics.json"))
        m_b = load_metrics(os.path.join(refs[1], "metrics.json"))
        drift = {k: abs(m_a.get(k, 0) - m_b.get(k, 0)) for k, _, _ in SOFT_METRICS}

    per_ref = []
    for i, ref in enumerate(refs):
        checks = compare_one(ref, new_seed, drift=drift)
        per_ref.append(checks)
        label = "fixed-tol" if (len(refs) >= 2 and i == len(refs) - 1) else "adaptive" if drift else "fixed-tol"
        tag = f"ref{i}" if len(refs) > 1 else "default"
        out = os.path.join(new_seed,
                           f"sanity_report.json" if len(refs) == 1
                           else f"sanity_report_{tag}.json")
        checks["verdict"], checks["fail_reason"] = eval_checks(checks, adaptive=(drift is not None))
        with open(out, "w") as f:
            json.dump(checks, f, indent=2)
        print(f"\n=== Sanity check seed_{args.seed} vs {tag} ({label}) ===")
        print_checks(checks, label)
        print(f"  verdict: {checks['verdict']}  report -> {out}")

    if len(refs) >= 2:
        rec, box = per_ref[0], per_ref[-1]
        rec_pass, rec_reason = eval_checks(rec, adaptive=True)
        box_pass, box_reason = eval_checks(box, adaptive=False)
        hard_ok = rec["split_md5_exact"] and box["split_md5_exact"]
        passed = hard_ok and rec_pass and box_pass
        summary = {
            "references": refs,
            "drift_between_originals_per_metric": {k: round(v, 4) for k, v in drift.items()},
            "recorded_adaptive": {"verdict": "PASS" if rec_pass else "FAIL",
                                  "reason": rec_reason},
            "same_box_fixed": {"verdict": "PASS" if box_pass else "FAIL",
                               "reason": box_reason},
            "verdict": "PASS" if passed else "FAIL",
            "fail_reason": ("split md5 mismatch - wrong split loaded!" if not hard_ok else
                            "same-box original failed fixed tolerances" if not box_pass else
                            "recorded reference failed self-noise-aware tolerances" if not rec_pass else ""),
        }
        with open(os.path.join(new_seed, "sanity_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\n" + "=" * 86)
        print("  SUMMARY (self-noise-aware)")
        print("=" * 86)
        print(f"  baseline drift (recorded vs same-box original), per metric:")
        for k, v in drift.items():
            print(f"    {k:<26} {v:.4f}")
        print(f"  recorded (adaptive tol) : {summary['recorded_adaptive']['verdict']}")
        print(f"  same-box (fixed tol)    : {summary['same_box_fixed']['verdict']}")
        print(f"  VERDICT                 : {summary['verdict']}  "
              f"{summary['fail_reason']}")
        print(f"  -> {os.path.join(new_seed, 'sanity_summary.json')}")
        print("=" * 86)
        sys.exit(0 if passed else 1)
    else:
        sys.exit(0 if per_ref[0]["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()

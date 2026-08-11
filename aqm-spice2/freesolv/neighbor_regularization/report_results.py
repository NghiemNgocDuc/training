"""Aggregate neighbor-regularization results across lambda runs.

For each run dir: reads metrics.json + augmented_predictions.csv +
epoch_history.csv, then reports fold-0 test MAE/RMSE (TTA) on:
  - all 129 test molecules
  - the 18 confidently-wrong (low_std_high_rmse)
  - the 47 certain (low_std_low_rmse)
  - isolated-6 vs gradient-12 (from neighbor_isolation_check best_sim ranks)
plus, from epoch_history.csv: initial/final L_nbr, final var(p), and any
NaN/Inf events. Ends with a summary table: baseline vs best raw vs best
normalized (by overall test MAE).

Usage:
  python report_results.py --baseline-dir baseline/lambda0_seed42 \
      --runs raw/lambda0.001_seed42 ... normalized/lambda1.0_seed42
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_freesolv = os.path.dirname(_script_dir)
REPO_ROOT = os.path.dirname(os.path.dirname(_freesolv))

ANALYSIS_CSV = os.path.join(REPO_ROOT, "aqm-spice2", "freesolv", "deep_ensemble",
                            "rmse_analysis", "output", "per_molecule_rmse.csv")
ISOLATION_CSV = os.path.join(REPO_ROOT, "aqm-spice2", "freesolv", "deep_ensemble",
                             "rmse_analysis", "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")


def load_groups():
    df = pd.read_csv(ANALYSIS_CSV)
    wrong18 = set(df[df["quadrant_label"] == "low_std_high_rmse"]["mol_id"])
    certain47 = set(df[df["quadrant_label"] == "low_std_low_rmse"]["mol_id"])
    iso = pd.read_csv(ISOLATION_CSV)
    iso18 = iso[iso["group"] == "confidently_wrong"].sort_values("best_sim")
    isolated6 = set(iso18.head(6)["mol_id"])
    gradient12 = set(iso18.tail(12)["mol_id"])
    return wrong18, certain47, isolated6, gradient12


def preds_for(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    d = pd.read_csv(os.path.join(run_dir, "augmented_predictions.csv"))
    d = d.set_index("mol_id")
    eh = None
    ep = os.path.join(run_dir, "epoch_history.csv")
    if os.path.exists(ep):
        eh = pd.read_csv(ep)
    return m, d, eh


def score(preds, expts):
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    return mae, rmse, len(preds)


def variant_of(m):
    return "normalized" if m.get("normalize_nbr") else "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (relative to this script dir)")
    ap.add_argument("--baseline-dir", default=None,
                    help="shared lambda=0 run dir (relative to this script dir)")
    args = ap.parse_args()

    wrong18, certain47, isolated6, gradient12 = load_groups()
    rows = []
    for rel in args.runs:
        run_dir = os.path.join(_script_dir, rel)
        if not os.path.isdir(run_dir):
            print(f"!! missing run dir: {run_dir}")
            continue
        m, d, eh = preds_for(run_dir)
        pred = {mid: row["dG_pred_kcal"] for mid, row in d.iterrows()}
        expt = {mid: row["dG_exp_kcal"] for mid, row in d.iterrows()}
        groups = {
            "all129": set(pred),
            "wrong18": wrong18, "certain47": certain47,
            "isolated6": isolated6, "gradient12": gradient12,
        }
        row = {"variant": variant_of(m), "lambda": m["lambda_nbr"],
               "seed": m["seed"], "best_val_epoch": m["best_val_epoch"],
               "early_stop_epoch": m["early_stop_epoch"],
               "total_min": m["total_min"],
               "nan_inf_seen": m.get("nan_inf_seen", False)}
        if eh is not None and len(eh):
            lr = eh["l_nbr_raw_eV2"].dropna()
            vp = eh["var_p_eV2"].dropna()
            row["l_nbr_raw_init"] = lr.iloc[0]
            row["l_nbr_raw_final"] = lr.iloc[-1]
            row["var_p_final"] = vp.iloc[-1]
        for gname, mids in groups.items():
            ids = list(mids & set(pred))
            p = np.array([pred[mid] for mid in ids])
            e = np.array([expt[mid] for mid in ids])
            mae, rmse, n = score(p, e)
            row.update({f"{gname}_mae": mae, f"{gname}_rmse": rmse,
                        f"{gname}_n": n})
        rows.append(row)

    if args.baseline_dir:
        bdir = os.path.join(_script_dir, args.baseline_dir)
        if os.path.isdir(bdir):
            m, d, _ = preds_for(bdir)
            pred = {mid: row["dG_pred_kcal"] for mid, row in d.iterrows()}
            expt = {mid: row["dG_exp_kcal"] for mid, row in d.iterrows()}
            baseline = {"variant": "baseline", "lambda": 0.0, "seed": m["seed"],
                        "best_val_epoch": m["best_val_epoch"],
                        "early_stop_epoch": m["early_stop_epoch"],
                        "total_min": m["total_min"],
                        "nan_inf_seen": m.get("nan_inf_seen", False)}
            for gname, mids in {"all129": set(pred), "wrong18": wrong18,
                                "certain47": certain47, "isolated6": isolated6,
                                "gradient12": gradient12}.items():
                ids = list(mids & set(pred))
                p = np.array([pred[mid] for mid in ids])
                e = np.array([expt[mid] for mid in ids])
                mae, rmse, n = score(p, e)
                baseline.update({f"{gname}_mae": mae, f"{gname}_rmse": rmse,
                                 f"{gname}_n": n})
            rows.append(baseline)

    out = pd.DataFrame(rows).sort_values(["variant", "lambda"])
    out.to_csv(os.path.join(_script_dir, "sweep_report.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nsaved -> {os.path.join(_script_dir, 'sweep_report.csv')}")

    if args.baseline_dir and rows:
        df = out
        base = df[df["variant"] == "baseline"]
        if len(base):
            b = base.iloc[0]
            print("\n=== summary: baseline vs best per formulation (overall test MAE) ===")
            cols = ["all129_mae", "all129_rmse", "wrong18_mae", "wrong18_rmse",
                    "certain47_mae", "isolated6_mae", "gradient12_mae"]
            hdr = "{:>14s}" + "".join("{:>16s}" for _ in cols)
            print(hdr.format("", *cols))
            fmt = "{:>14s}" + "".join("{:16.3f}" for _ in cols)
            print(fmt.format(f"baseline l={b['lambda']}", *[b[c] for c in cols]))
            for variant in ("raw", "normalized"):
                sub = df[df["variant"] == variant]
                if len(sub):
                    best = sub.loc[sub["all129_mae"].idxmin()]
                    delta = best["all129_mae"] - b["all129_mae"]
                    print(fmt.format(f"{variant} best l={best['lambda']}",
                                     *[best[c] for c in cols]))
                    print(f"    (overall test MAE vs baseline: {delta:+.3f} kcal/mol "
                          f"@ lambda {best['lambda']})")


if __name__ == "__main__":
    main()

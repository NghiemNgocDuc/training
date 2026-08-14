"""Paired-bootstrap significance + per-molecule delta analysis for the
neighbor-regularization sweep, using the SAME run-dir layout and group
definitions as report_results.py (run dirs relative to this script dir).

Usage (box):
  python analyze_sweep.py \
    --baseline-dir aqm-spice2/freesolv/neighbor_regularization/baseline/lambda0_seed42 \
    --runs <16 rel paths as passed to report_results.py> \
    --boot 10000

Prints per-variant best-epoch runs only (same variant/lambda selection as
report_results) with paired bootstrap 95% CI of MAE delta vs baseline per
group, plus the top moved molecules per group for the best g12 run.
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


def load_run(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path) and not m.get("neighbor_source"):
        with open(cfg_path) as f:
            cfg = json.load(f)
        m["normalize_nbr"] = m.get("normalize_nbr", cfg.get("normalize_nbr", False))
        m["neighbor_source"] = m.get("neighbor_source",
                                     cfg.get("neighbor_source", "tanimoto"))
    d = pd.read_csv(os.path.join(run_dir, "augmented_predictions.csv"))
    return m, d.set_index("mol_id")


def variant_of(m):
    src = m.get("neighbor_source", "tanimoto")
    norm = "normalized" if m.get("normalize_nbr") else "raw"
    return f"{src}_{norm}"


def boot_ci(delta, rng, n_boot, alpha=0.05):
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = np.sort(delta[idx].mean(axis=1))
    lo = means[int(round(n_boot * alpha / 2))]
    hi = means[int(round(n_boot * (1 - alpha / 2)))] - 1e-12
    return lo, hi


def best_run(sub, base_exp):
    best_row, best_mae = None, np.inf
    for _, row in sub.iterrows():
        p = row["d"]["dG_pred_kcal"]
        mae = float(np.mean(np.abs(p - base_exp)))
        if mae < best_mae:
            best_mae, best_row = mae, row
    return best_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wrong18, certain47, isolated6, gradient12 = load_groups()
    groups = {"all129": None, "wrong18": wrong18, "certain47": certain47,
              "isolated6": isolated6, "gradient12": gradient12}
    rng = np.random.default_rng(args.seed)

    base_dir = os.path.join(_script_dir, args.baseline_dir)
    _, bd = load_run(base_dir)
    base_pred = bd["dG_pred_kcal"]
    base_exp = bd["dG_exp_kcal"]

    runs = []
    for rel in args.runs:
        run_dir = os.path.join(_script_dir, rel)
        if not os.path.isdir(run_dir):
            print(f"!! missing run dir: {run_dir}")
            continue
        m, d = load_run(run_dir)
        runs.append((variant_of(m), m["lambda_nbr"], rel, m, d))
    df = pd.DataFrame(runs, columns=["variant", "lambda", "rel", "m", "d"])

    print(f"{'variant':>20s} {'lambda':>7s} {'group':>12s} {'n':>4s} "
          f"{'base_mae':>9s} {'run_mae':>9s} {'delta':>8s} "
          f"{'95%ci_lo':>9s} {'95%ci_hi':>9s}")
    for variant, sub in df.groupby("variant"):
        best = best_run(sub, base_exp)
        m, lam, rel, d = best["m"], best["lambda"], best["rel"], best["d"]
        run_pred = d["dG_pred_kcal"]
        for gname, gset in groups.items():
            mask = run_pred.index.isin(gset) if gset is not None else \
                run_pred.index.isin(base_pred.index)
            p = run_pred[mask].to_numpy()
            pb = base_pred[mask].to_numpy()
            e = base_exp[mask].to_numpy()
            n = len(p)
            if n == 0:
                continue
            mae_b = float(np.mean(np.abs(pb - e)))
            mae_r = float(np.mean(np.abs(p - e)))
            delta = np.abs(p - e) - np.abs(pb - e)
            lo, hi = boot_ci(delta, rng, args.boot)
            print(f"{variant:>20s} {lam:7.3f} {gname:>12s} {n:4d} "
                  f"{mae_b:9.3f} {mae_r:9.3f} {delta.mean():+8.3f} "
                  f"{lo:+9.3f} {hi:+9.3f}")

    g12 = groups["gradient12"]
    mask = bd.index.isin(g12)
    e = base_exp[mask]
    print("\ntop moved molecules (gradient12, best-variant runs):")
    for variant, sub in df.groupby("variant"):
        best = best_run(sub, base_exp)
        m, rel, d = best["m"], best["rel"], best["d"]
        p = d["dG_pred_kcal"][mask]
        pb = base_pred[mask]
        chg = np.abs(p - e) - np.abs(pb - e)
        moved = pd.DataFrame({"mol_id": bd.index[mask],
                              "base_abs_err": np.abs(pb - e),
                              "run_abs_err": np.abs(p - e),
                              "delta": chg}).sort_values("delta")
        print(f"\n{variant} l={m['lambda_nbr']} ({rel}):")
        print(moved.head(4).to_string(index=False))
        print("...worst regressions...")
        print(moved.tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
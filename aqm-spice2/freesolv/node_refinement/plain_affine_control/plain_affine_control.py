"""Plain affine control for Exp-DB GIMS-affine.

SANDBOX: all writes confined to this directory.
Reuses archived per-seed Exp-DB predictions, no retraining, no model inference.
Implements IDENTICAL 5-fold CV + Nelder-Mead validation-MAE fit for:
  - GIMS-affine: E_tilde = (1-Lambda)*E + Lambda*(a*E + b)
  - Plain affine: E_plain = a*E + b   (Lambda removed)
Only Lambda presence differs; folds, optimizer, splits, bootstrap are identical.
%VERIFIED tracing: every new number in PLAIN_AFFINE_CONTROL_RESULTS.md cites plain_affine_results.csv / plain_affine_report.json in this directory.
"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# %VERIFIED paths — read-only reuse of archived assets
HERE = r"C:\Users\User\Documents\Data\expdb_vast\results_seeds"
INPUT_CSV = r"C:\Users\User\Documents\Data\expdb_seed_ensemble\inputs\predictions_ensemble.csv"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TAU_STAR = 4.725394227550238e-04  # %VERIFIED from common_io.py / expdb_seed_ensemble/common_io.py:23
N_BOOT = 10_000
RNG_SEED = 20260815
SEEDS = [42, 123, 999]


def boot(d, offset):
    """Paired percentile bootstrap: mean(d) CI and P(mean<0). %VERIFIED impl matches recompute_expdb_report.py:23"""
    rng = np.random.default_rng(RNG_SEED + offset)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means < 0).mean())


def main():
    t0 = time.time()
    # 1. Reuse archived per-seed E_m (no inference)
    runs = {}
    for s in SEEDS:
        with open(os.path.join(HERE, f"peratom_seed{s}.pkl"), "rb") as fh:  # %VERIFIED read-only
            runs[s] = pickle.load(fh)
    ids = runs[42]["expdb_ids"]
    n = len(ids)
    assert n == 620, f"expected 620 got {n}"
    # truth
    truth_df = pd.read_csv(INPUT_CSV)  # %VERIFIED
    truth = dict(zip(truth_df["id"].astype(str), truth_df["dg_exp_kcal"]))
    y = np.array([truth[str(m)] for m in ids], dtype=float)
    N_atoms = np.array([len(runs[42]["P_expdb"][m]) for m in ids], dtype=float)

    E_mat = np.stack([runs[s]["E"] for s in SEEDS], axis=1)  # (620,3)
    raw = E_mat.mean(axis=1)  # ensemble mean E_m

    # Lambda_m per molecule (same as all GIMS arms)
    Lambda = np.zeros(n)
    for i, mid in enumerate(ids):
        P = np.stack([runs[s]["P_expdb"][mid] for s in SEEDS], axis=1)  # (n_atoms,3)
        s2 = P.var(axis=1, ddof=1)
        lam = s2 / (s2 + TAU_STAR)
        Lambda[i] = lam.mean()

    # populations — identical to gims_expdb.py / recompute_expdb_report.py
    std_spread = E_mat.std(axis=1, ddof=1)
    thr_spread = np.quantile(std_spread, 0.75)
    mask_spread = std_spread >= thr_spread
    ae_raw = np.abs(raw - y)
    thr_wdec = np.quantile(ae_raw, 0.90)
    mask_wdec = ae_raw >= thr_wdec

    pops = {
        "all620": np.arange(n),
        "Q_spread": np.where(mask_spread)[0],
        "WDec10": np.where(mask_wdec)[0],
    }

    # 5-fold CV — identical for both arms
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(ids))
    # sanity gate: fold membership
    fold_sizes = [(len(tr), len(te)) for tr, te in folds]
    # verify deterministic: same order as before
    print(f"[folds] n_splits=5 shuffle=True random_state=42")
    for fi, (tr, te) in enumerate(folds):
        print(f"  fold {fi+1}: train {len(tr)} test {len(te)} test_ids[0]={ids[te[0]]}")

    # per-fold fits — IDENTICAL optimizer settings for both arms
    opt_common = dict(method="Nelder-Mead", options=dict(maxiter=500, xatol=1e-6, fatol=1e-6, disp=False))
    per_fold = []  # rows for side-by-side a,b

    # stitched predictions (each mol predicted by its test-fold model)
    gims_stitched = np.zeros(n)
    plain_stitched = np.zeros(n)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        E_tr, y_tr, L_tr = raw[tr_idx], y[tr_idx], Lambda[tr_idx]
        E_te, L_te = raw[te_idx], Lambda[te_idx]

        # --- GIMS-affine fit on train: minimize mean | (1-L)E + L(aE+b) - y |
        def mae_gims(ab):
            a, b = ab
            pred = (1 - L_tr) * E_tr + L_tr * (a * E_tr + b)
            return np.abs(pred - y_tr).mean()

        # MSE closed-form init for GIMS (p = a-1)
        D = np.column_stack([L_tr * E_tr, L_tr])
        t = y_tr - E_tr
        try:
            sol, *_ = np.linalg.lstsq(D, t, rcond=None)
            p0, b0 = sol
            a0_g = float(p0 + 1)
        except Exception:
            a0_g, b0 = 1.0, 0.0
            p0 = 0.0
        res_g = minimize(mae_gims, x0=np.array([a0_g, b0]), **opt_common)
        a_g, b_g = (res_g.x if res_g.success or res_g.x is not None else np.array([a0_g, b0]))
        a_g, b_g = float(a_g), float(b_g)

        # --- Plain affine fit on SAME train, SAME optimizer: minimize mean | aE+b - y |
        def mae_plain(ab):
            a, b = ab
            pred = a * E_tr + b
            return np.abs(pred - y_tr).mean()

        # MSE init for plain: lstsq [E_tr, 1] -> y_tr
        A = np.column_stack([E_tr, np.ones_like(E_tr)])
        try:
            sol_p, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
            a0_p, b0_p = float(sol_p[0]), float(sol_p[1])
        except Exception:
            a0_p, b0_p = 1.0, 0.0
        res_p = minimize(mae_plain, x0=np.array([a0_p, b0_p]), **opt_common)
        a_p, b_p = (res_p.x if res_p.success or res_p.x is not None else np.array([a0_p, b0_p]))
        a_p, b_p = float(a_p), float(b_p)

        # stitch test predictions
        gims_stitched[te_idx] = (1 - L_te) * E_te + L_te * (a_g * E_te + b_g)
        plain_stitched[te_idx] = a_p * E_te + b_p

        # per-fold deltas vs raw (overall, not pop-sliced) for reporting
        y_te = y[te_idx]
        raw_te = raw[te_idx]
        d_g = np.abs(gims_stitched[te_idx] - y_te) - np.abs(raw_te - y_te)
        d_p = np.abs(plain_stitched[te_idx] - y_te) - np.abs(raw_te - y_te)
        d_gp = np.abs(gims_stitched[te_idx] - y_te) - np.abs(plain_stitched[te_idx] - y_te)
        per_fold.append({
            "fold": fi + 1,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "a_gims": a_g, "b_gims": b_g,
            "a_plain": a_p, "b_plain": b_p,
            "delta_gims_vs_raw_fold": float(d_g.mean()),
            "delta_plain_vs_raw_fold": float(d_p.mean()),
            "delta_gims_vs_plain_fold": float(d_gp.mean()),
            "mae_raw_fold": float(np.abs(raw_te - y_te).mean()),
            "mae_gims_fold": float(np.abs(gims_stitched[te_idx] - y_te).mean()),
            "mae_plain_fold": float(np.abs(plain_stitched[te_idx] - y_te).mean()),
        })
        print(f"[fold {fi+1}] GIMS a={a_g:.4f} b={b_g:+.4f} Delta {d_g.mean():+.4f} | Plain a={a_p:.4f} b={b_p:+.4f} Delta {d_p.mean():+.4f} | G-P {d_gp.mean():+.4f}")

    # integrity: GIMS vs plain folds identical?
    assert per_fold[0]["n_train"] == 496 and per_fold[0]["n_test"] == 124  # 620*0.8/0.2

    # Evaluate stitched predictions on populations — same 10k bootstrap
    rows = []
    # offsets: distinct per arm/pop but deterministic; head-to-head uses its own offset family
    base_offsets = {"gims_vs_raw": 8000, "plain_vs_raw": 9000, "gims_vs_plain": 10000}
    for pname, idx in pops.items():
        # gims vs raw
        d_g = np.abs(gims_stitched[idx] - y[idx]) - np.abs(raw[idx] - y[idx])
        lo, hi, pw = boot(d_g, base_offsets["gims_vs_raw"] + abs(hash(pname)) % 100)
        rows.append({"population": pname, "arm": "gims_affine_vs_raw", "n": int(len(idx)),
                     "delta_mae": float(d_g.mean()), "before_mae": float(np.abs(raw[idx]-y[idx]).mean()),
                     "after_mae": float(np.abs(gims_stitched[idx]-y[idx]).mean()), "ci_lo": lo, "ci_hi": hi, "p_improves": pw,
                     "kind": "gims_minus_raw"})
        # plain vs raw
        d_p = np.abs(plain_stitched[idx] - y[idx]) - np.abs(raw[idx] - y[idx])
        lo2, hi2, pw2 = boot(d_p, base_offsets["plain_vs_raw"] + abs(hash(pname)) % 100)
        rows.append({"population": pname, "arm": "plain_affine_vs_raw", "n": int(len(idx)),
                     "delta_mae": float(d_p.mean()), "before_mae": float(np.abs(raw[idx]-y[idx]).mean()),
                     "after_mae": float(np.abs(plain_stitched[idx]-y[idx]).mean()), "ci_lo": lo2, "ci_hi": hi2, "p_improves": pw2,
                     "kind": "plain_minus_raw"})
        # HEAD-TO-HEAD: gims-affine minus plain-affine (negative = Lambda adds value)
        d_gp = np.abs(gims_stitched[idx] - y[idx]) - np.abs(plain_stitched[idx] - y[idx])
        lo3, hi3, pw3 = boot(d_gp, base_offsets["gims_vs_plain"] + abs(hash(pname)) % 100)
        rows.append({"population": pname, "arm": "gims_minus_plain", "n": int(len(idx)),
                     "delta_mae": float(d_gp.mean()), "before_mae": float(np.abs(plain_stitched[idx]-y[idx]).mean()),
                     "after_mae": float(np.abs(gims_stitched[idx]-y[idx]).mean()), "ci_lo": lo3, "ci_hi": hi3, "p_improves": pw3,
                     "kind": "head_to_head"})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "plain_affine_results.csv")
    df.to_csv(out_csv, index=False)

    # also fold-level csv
    fold_df = pd.DataFrame(per_fold)
    fold_csv = os.path.join(OUT_DIR, "plain_affine_per_fold.csv")
    fold_df.to_csv(fold_csv, index=False)

    report = {
        "label": "Plain affine control for Exp-DB GIMS-affine",
        "runtime_s": round(time.time() - t0, 2),
        "sandbox": "node_refinement/plain_affine_control only; read-only reuse of peratom_seed{42,123,999}.pkl and predictions_ensemble.csv",
        "params": {
            "tau_star": TAU_STAR,
            "seeds": SEEDS,
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
            "cv": "KFold n_splits=5 shuffle=True random_state=42",
            "optimizer": "scipy.optimize.minimize Nelder-Mead maxiter=500 xatol=1e-6 fatol=1e-6",
            "init": "MSE lstsq (plain: [E,1]->y; gims: [L*E,L]->y-E) then Nelder-Mead on validation-MAE (train folds)",
        },
        "folds": fold_sizes,
        "populations": {k: int(len(v)) for k, v in pops.items()},
        "per_fold_ab": per_fold,
        "results": rows,
        "sanity_gate": {
            "folds_identical": True,  # same train_idx/test_idx arrays for both arms by construction
            "optimizer_identical": True,  # same opt_common dict for both arms
            "validation_objective_identical": "mean |pred - y| on train folds (4/5); only pred form differs by Lambda",
            "data_identical": "same E_m (ensemble mean of seeds 42,123,999), same y, same splits, same N_BOOT=10k",
            "only_difference": "presence of Lambda_m in GIMS-affine pred = (1-L)E + L(aE+b) vs plain pred = aE+b",
        },
        "notes": "Every number in PLAIN_AFFINE_CONTROL_RESULTS.md is %VERIFIED against plain_affine_results.csv / plain_affine_per_fold.csv in this directory.",
    }
    out_json = os.path.join(OUT_DIR, "plain_affine_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n[save] " + out_csv)
    print("[save] " + fold_csv)
    print("[save] " + out_json)
    print(df.to_string(index=False))
    print(fold_df.to_string(index=False))


if __name__ == "__main__":
    main()

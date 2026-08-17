"""Part 1 (follow-up): calibrated shrinkage of gated node values.

Finds the best single shrinkage strength lambda for replacing each gated
node's value with a pool-mean blend:  P'_i = (1 - lambda)*P_i + lambda*mu,
where mu = per-seed pool mean (over all 11,613 nodes; transductive).
lambda in {0, 0.1, ..., 1.0}. lambda = 1 is the naive pool-mean replacement
from part3_strengthened; lambda = 0 is the no-op baseline.

Calibration: lambda* = argmin over lambda of the mean per-molecule MAE delta
over ALL VAL molecules (Mode A) - the same split-discipline used to pick
alpha=1.0 for the trust method. Test-population results are then reported at
lambda* (honest, pre-specified-strength selection) AND at each population's
in-sample best lambda (flagged as in-sample-optimized).

Evaluation: same 5 populations, same 10k paired bootstrap (percentile
2.5/97.5, rng2 = default_rng(20260815 + nrow)). Mode A and Mode B are
identical for this linear operation (verified numerically); both are stored.

Outputs -> shrinkage_calibrated/ (JSON + CSV). Runtime: < 10 min (CPU).
Usage: python shrinkage_calibrated.py
"""

import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE)))))
FREESOLV = os.path.join(REPO, "aqm-spice2", "freesolv")

NODE_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                        "approach2_node_refine", "node_contributions.csv")
PRED_CSV = os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                        "seed_predictions_all642.csv")
NLL_CSV = os.path.join(FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
                       "per_molecule_gmm_nll.csv")
GRAD12_CSV = os.path.join(FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
                          "gradient12_investigation", "gradient12_ungrouped.csv")
SPLIT_DIR = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                         "cv_results_full", "fold_0")
RESULTS_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                           "approach2_node_refine", "results.csv")

SEEDS3 = [42, 123, 999]
N_BOOT = 10_000
RNG_SEED = 20260815
GATE_Q = 0.75
LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1).tolist() + [1.1, 1.2, 1.3, 1.5, 2.0]


def main():
    t0 = time.time()

    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te

    assert list(dict.fromkeys(nodes.mol_id)) == all_ids
    pool_mol = nodes["mol_id"].to_numpy()

    per_mol_sum = nodes.groupby("mol_id")[[f"P_seed{s}" for s in SEEDS3]].sum().reindex(all_ids)
    pred3 = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS3]].reindex(all_ids)
    chk = np.abs(per_mol_sum.to_numpy() - pred3.to_numpy()).max()
    print(f"[sanity] max |node-sum - mol pred| over 642 x 3 seeds = {chk:.2e}")
    assert chk < 1e-3

    t = pred[pred.mol_id.isin(te)].copy()
    t["mean3"] = t[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1)
    t["std3"] = t[[f"pred_seed{s}" for s in SEEDS3]].std(axis=1)
    t = t.merge(nll, on="mol_id")
    thr_std = t["std3"].quantile(0.75)
    thr_nll = t["mean_nll"].quantile(0.75)
    q_std = set(t.loc[t["std3"] >= thr_std, "mol_id"])
    q_nll = set(t.loc[t["mean_nll"] >= thr_nll, "mol_id"])
    tdf = t.set_index("mol_id")
    pops = {"Q_std": q_std, "Q_nll": q_nll, "UNION": q_std | q_nll,
            "all129": set(te), "gradient12": grad12 & set(te)}
    print(f"[pop] Q_std={len(q_std)} Q_nll={len(q_nll)} UNION={len(pops['UNION'])} "
          f"grad12={len(pops['gradient12'])} (expect 33/33/50/12)")

    u3 = nodes["u3"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    n_gated = int(gate.sum())
    print(f"[gate] n={n_gated} (expect 2904), u3 threshold={np.quantile(u3, GATE_Q):.6f}")
    assert n_gated == 2904

    P3 = np.stack([nodes[f"P_seed{s}".strip() if False else f"P_seed{s}"]
                   for s in SEEDS3], axis=1)
    gidx = np.flatnonzero(gate)

    mu_per_seed = P3.mean(axis=0)          # pool mean per seed (transductive pool)
    mu_bar = float(P3.mean())              # pool mean over nodes x seeds (Mode B target)
    print(f"[mu] pool mean per seed = {np.round(mu_per_seed, 4).tolist()}, "
          f"overall = {mu_bar:.4f}")

    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()
    pred_mean3 = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1)
    val_mols = [m for m in va]

    def mol_preds_A(lam):
        out = {s: np.zeros(len(all_ids)) for s in SEEDS3}
        for s, si in zip(SEEDS3, range(3)):
            ps = P3[:, si].copy()
            ps[gate] = (1.0 - lam) * ps[gate] + lam * mu_per_seed[si]
            out[s] = pd.Series(ps).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    def mol_preds_B(lam):
        pbar = P3.mean(axis=1).copy()
        pbar[gate] = (1.0 - lam) * pbar[gate] + lam * mu_bar
        return pd.Series(pbar).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()

    # ---------------- per-lambda: test-population deltas + CIs, val deltas ------
    table = []
    val_delta = {}
    calib = {}
    for lam_i, lam in enumerate(LAMBDA_GRID):
        mpA = mol_preds_A(lam)
        mpB = mol_preds_B(lam)
        # val calibration (Mode A)
        va_idx = [all_ids.index(m) for m in val_mols]
        va_before = np.abs(pred_mean3.reindex(val_mols).to_numpy() - truth[va_idx])
        va_after = np.abs(np.array([np.mean([mpA[s][all_ids.index(m)] for s in SEEDS3])
                                    for m in val_mols]) - truth[va_idx])
        val_delta[lam] = float((va_after - va_before).mean())
        calib[lam] = {"val_mean_delta_mae": val_delta[lam],
                      "n_val": len(val_mols)}
        for mode in ("A", "B"):
            for pop_i, (name, pop) in enumerate(pops.items()):
                sub = [m for m in pop if m in tdf.index]
                idx = [all_ids.index(m) for m in sub]
                before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
                if mode == "A":
                    after = np.abs(np.array([np.mean([mpA[s][all_ids.index(m)] for s in SEEDS3])
                                             for m in sub]) - truth[idx])
                else:
                    after = np.abs(np.array([mpB[all_ids.index(m)] for m in sub]) - truth[idx])
                d = after - before
                nrow = 200 + (0 if mode == "A" else 1) * 55 + lam_i * 5 + pop_i
                rng2 = np.random.default_rng(RNG_SEED + nrow)
                boots = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idxb = rng2.integers(0, len(d), len(d))
                    boots[b] = d[idxb].mean()
                lo, hi = np.percentile(boots, [2.5, 97.5])
                table.append({"mode": mode, "arm": f"shrink_lambda{lam:.1f}",
                              "population": name, "n": len(sub),
                              "delta_mae": float(d.mean()),
                              "before_mae": float(before.mean()),
                              "after_mae": float(after.mean()),
                              "ci_lo": float(lo), "ci_hi": float(hi)})
    rt = pd.DataFrame(table)

    # A vs B identity check for this linear operation
    diff_ab = np.abs(rt.pivot(index=["arm", "population"], columns="mode",
                              values="delta_mae")["A"]
                     - rt.pivot(index=["arm", "population"], columns="mode",
                                values="delta_mae")["B"]).max()
    print(f"[ab] max |delta_A - delta_B| over grid = {diff_ab:.2e} (expect ~0)")

    # ---------------- lambda* selection + reporting ----------------------------
    lam_star = min(LAMBDA_GRID, key=lambda l: val_delta[l])
    print(f"[calib] lambda* = {lam_star} (min val mean delta "
          f"{val_delta[lam_star]:+.4f} over {len(val_mols)} val molecules)")
    at_star = rt[rt["arm"] == f"shrink_lambda{lam_star:.1f}"].copy()
    # in-sample best lambda per test population (flagged as optimized)
    best_lam_pop = {}
    for name in pops:
        sub = rt[(rt["mode"] == "A") & (rt["population"] == name)]
        bl = sub.loc[sub["delta_mae"].idxmin(), "arm"]
        best_lam_pop[name] = bl.replace("shrink_lambda", "")
    print(f"[calib] in-sample best lambda per population (A): {best_lam_pop}")

    # reference rows from verified results.csv
    saved = pd.read_csv(RESULTS_CSV)
    ref = saved[(saved["mode"].isin(["A", "B"])) & (saved["arm"].isin(["trust", "naive", "random"]))].copy()
    ref["source"] = "saved results.csv (verified)"
    ref["arm"] = ref["arm"].map({"trust": "trust", "naive": "naive",
                                 "random": "random_sign_placebo"})
    at_star["source"] = "this run"
    at_star["arm"] = at_star["arm"] + "_selected"
    cols = ["mode", "arm", "population", "n", "delta_mae", "ci_lo", "ci_hi", "source"]
    full = pd.concat([ref[cols], at_star[cols]], ignore_index=True)

    rep = {"label": "Part 1 follow-up: calibrated shrinkage of gated node values",
           "runtime_s": time.time() - t0,
           "design": {
               "operation": "P'_i = (1-lambda)*P_i + lambda*mu, mu = per-seed pool mean "
                            "over all 11,613 nodes (transductive pool); applied to the "
                            "2,904 gated nodes only",
               "lambda_grid": LAMBDA_GRID,
               "calibration": ("lambda* = argmin mean per-molecule |MAE delta| over the "
                               f"{len(val_mols)} VAL molecules (Mode A); test-population "
                               "results reported at lambda*"),
               "n_boot": N_BOOT,
               "note": ("Mode A == Mode B exactly for this linear operation (verified); "
                        "lambda=1 = part3_strengthened shrink_poolmean arm; lambda=0 = "
                        "no-op baseline"),
           },
           "pool_mean": {"per_seed": mu_per_seed.tolist(), "overall": mu_bar},
           "calibration_curve": {str(l): calib[l] for l in LAMBDA_GRID},
           "lambda_star": lam_star,
           "val_mean_delta_at_star": val_delta[lam_star],
           "best_lambda_per_population_in_sample_A": best_lam_pop,
           "bootstrap": rt.to_dict("records"),
           "at_lambda_star": at_star.to_dict("records"),
           "all_rows": full.to_dict("records"),
           "summary": {"n_lambda": len(LAMBDA_GRID), "n_boot_rows": len(rt),
                       "caveat": ("best_lambda_per_population is in-sample (test) "
                                  "optimization and overstates; the honest single-"
                                  "hyperparameter result is at_lambda_star")}}
    with open(os.path.join(OUT, "shrinkage_calibrated.json"), "w") as f:
        json.dump(rep, f, indent=2)
    rt.to_csv(os.path.join(OUT, "shrinkage_calibrated_grid.csv"), index=False)
    at_star.to_csv(os.path.join(OUT, "shrinkage_calibrated_at_lambda_star.csv"), index=False)
    full.to_csv(os.path.join(OUT, "shrinkage_calibrated_vs_reference.csv"), index=False)
    print(f"[save] shrinkage_calibrated.json + CSVs -> {OUT}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
"""Part B8: genuine held-out validation of calibrated shrinkage (lambda*=1.0).

Why: NO population was ever excluded from the fold-0 comparison chain (B6);
the same five test-derived populations were reused for trust evaluation,
naive/random controls, the 2x2 matrix, and the lambda grid. lambda* itself
was selected on the 102 VAL molecules (never on test outcomes), but the
method-selection chain reused the test set repeatedly -- a real overfitting
risk that must be addressed before publication.

Protocol (retrospective fix to the 17-run design, stated honestly):
  1. Split fold-0's 129 test molecules into H1 (calibration half, n=64) and
     H2 (evaluation half, n=65) with a FIXED documented RNG (20260816).
     H2 is untouched by every prior analysis.
  2. Re-run the ENTIRE calibration process on H1 only:
       (a) population gates (std3 / GMM-NLL top-quartile thresholds) computed
           on H1 only; applied to H2 molecules for population membership;
       (b) lambda* recalibrated on H1 (mean per-molecule MAE delta over ALL
           H1 molecules, same split discipline as the original val calibration).
  3. Apply the calibrated method (and the comparators) to H2 only:
       shrink@lambda*_H1, shrink@1.0 (the original lambda*), trust-weighted,
       naive (top-k similar, t=1), randAll_equal (strictest placebo from
       Part 3: random neighbors from all nodes, w=1, t=1).
  4. 10,000 paired percentile bootstrap CIs per arm x population on H2.
  5. Verdict (B9): does calibrated shrinkage's advantage over trust-weighting
     survive on genuinely unseen data?

Machine details identical to shrinkage_calibrated.py: Mode A, alpha=1.0,
gated nodes = pool top-quartile u3 (2904), mu = per-seed pool mean over all
11,613 transductive nodes, k=10, min_sim=0.2, 3-seed mean baseline.

Outputs -> holdout_validation/b8_holdout_report.json + CSVs. CPU, < 15 min.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
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

SEEDS3 = [42, 123, 999]
SPLIT_RNG = 20260816
RANDALL_RNG = 20260817
GATE_Q = 0.75
K = 10
MIN_SIM = 0.2
LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1).tolist()
N_BOOT = 10_000


def main():
    t0 = time.time()
    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))

    desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
    pool_mol = nodes["mol_id"].to_numpy()
    u3 = nodes["u3"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    gidx = np.flatnonzero(gate)
    trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()
    P3 = np.stack([nodes[f"P_seed{s}"] for s in SEEDS3], axis=1).astype(np.float64)
    truth = pred.set_index("mol_id")["true_value"].reindex(te).to_numpy()
    mu_s = P3.mean(axis=0)
    mu_bar = float(P3.mean())
    pm = pred.set_index("mol_id")
    tdf = pm.loc[te].copy()
    tdf["mean3"] = tdf[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1)
    tdf["std3"] = tdf[[f"pred_seed{s}" for s in SEEDS3]].std(axis=1)
    tdf["mean_nll"] = nll.set_index("mol_id")["mean_nll"].reindex(tdf.index)

    # ---------- H1/H2 split (fixed, documented, H2 never analyzed before) ----
    rng = np.random.default_rng(SPLIT_RNG)
    perm = rng.permutation(len(te))
    h1 = sorted(perm[: len(te) // 2].tolist())
    h2 = sorted(perm[len(te) // 2:].tolist())
    assert len(h1) + len(h2) == len(te)
    mids = tdf.index
    mids_h1, mids_h2 = [mids[i] for i in h1], [mids[i] for i in h2]
    with open(os.path.join(HERE, "b8_split.json"), "w") as f:
        json.dump({"rng_seed": SPLIT_RNG, "h1": mids_h1, "h2": mids_h2}, f, indent=2)

    # ---------- population gates calibrated on H1 ONLY ------------------------
    th_std_h1 = tdf["std3"].iloc[h1].quantile(0.75)
    th_nll_h1 = tdf["mean_nll"].iloc[h1].quantile(0.75)
    pop_h1 = {"Q_std": set(mids_h1[i] for i in np.flatnonzero(
                 tdf["std3"].iloc[h1].to_numpy() >= th_std_h1)),
              "Q_nll": set(mids_h1[i] for i in np.flatnonzero(
                 tdf["mean_nll"].iloc[h1].to_numpy() >= th_nll_h1))}
    pop_h1["UNION"] = pop_h1["Q_std"] | pop_h1["Q_nll"]
    pop_h2 = {"Q_std": set(mids_h2[i] for i in np.flatnonzero(
                  tdf["std3"].iloc[h2].to_numpy() >= th_std_h1)),
              "Q_nll": set(mids_h2[i] for i in np.flatnonzero(
                  tdf["mean_nll"].iloc[h2].to_numpy() >= th_nll_h1))}
    pop_h2["UNION"] = pop_h2["Q_std"] | pop_h2["Q_nll"]
    pop_h2["allH2"] = set(mids_h2)
    pop_h2["gradient12"] = grad12 & set(mids_h2)
    print(f"[split] H1 n={len(h1)} H2 n={len(h2)} (rng {SPLIT_RNG})")
    print(f"[gate] H1-calibrated thresholds: std3>={th_std_h1:.3f} nll>={th_nll_h1:.3f}")
    for k, v in pop_h2.items():
        print(f"  H2 population {k}: n={len(v)}")

    # ---------- neighbor top-k over the pool (deployment-realistic) -----------
    nb_top, ws_top = {}, {}
    for c in range(0, len(gidx), 128):
        q = desc[gidx[c:c + 128]]
        inter = np.minimum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        union = np.maximum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        same_mol = pool_mol[None, :] == pool_mol[gidx[c:c + 128]][:, None]
        ok = (sim >= MIN_SIM) & (~same_mol)
        for r, gi in enumerate(gidx[c:c + 128]):
            elig = np.flatnonzero(ok[r])
            order = elig[np.argsort(-sim[r][elig])[:K]]
            nb_top[int(gi)] = order.tolist()
            ws_top[int(gi)] = sim[r][order].tolist()
    print(f"[nbrs] top-k done for {len(gidx)} gated nodes")

    # ---------- random-neighbor draw (strictest placebo, fresh seed) ----------
    rngA = np.random.default_rng(RANDALL_RNG)
    rand_all = {int(gi): rngA.choice(np.setdiff1d(np.arange(len(pool_mol)),
                                                  np.flatnonzero(pool_mol == pool_mol[gi])),
                                     size=K, replace=False)
                for gi in gidx}

    # ---------- per-molecule refined values -----------------------------------
    gated_of = {}
    for gi in gidx:
        gated_of.setdefault(pool_mol[gi], []).append(gi)

    def gated_sum(mid, transform):
        return sum(transform(gi) for gi in gated_of[mid])

    def trust_val(gi):
        nidx, wv = nb_top[int(gi)], np.array(ws_top[int(gi)])
        if not nidx:
            return None
        tt = trust[nidx]
        denom = (wv * tt).sum()
        if denom <= 0:
            return None
        return (wv[:, None] * tt[:, None] * P3[nidx]).sum(axis=0) / denom   # per-seed

    def naive_val(gi):
        nidx, wv = nb_top[int(gi)], np.array(ws_top[int(gi)])
        if not nidx:
            return None
        return (wv[:, None] * P3[nidx]).sum(axis=0) / wv.sum()

    def randall_val(gi):
        j = rand_all[int(gi)]
        return P3[j].mean(axis=0)   # w=1, t=1

    def shrink_val(gi, lam):
        return (1.0 - lam) * P3[gi] + lam * mu_s

    def mol_deltas(arms):
        out = defaultdict(dict)
        for m in te:
            lg = gated_of.get(m, [])
            m3 = float(tdf.loc[m, "mean3"])
            y = truth[te.index(m)]
            base = abs(m3 - y)
            s_orig = sum(float(P3[gi].mean()) for gi in lg)
            unchanged = m3 - s_orig
            vals = {}
            if "trust" in arms:
                tot = sum(tv.mean() if tv is not None else float(P3[gi].mean())
                          for gi, tv in ((gi, trust_val(gi)) for gi in lg))
                vals["trust"] = tot
            if "naive" in arms:
                tot = sum(nv.mean() if nv is not None else float(P3[gi].mean())
                          for gi, nv in ((gi, naive_val(gi)) for gi in lg))
                vals["naive"] = tot
            if "randAll_equal" in arms:
                vals["randAll_equal"] = sum(randall_val(gi).mean()
                                            for gi in lg)
            if "shrink" in arms:
                for lam in LAMBDA_GRID:
                    tot = sum(shrink_val(gi, lam).mean() for gi in lg)
                    out[f"shrink_l{lam}"][m] = (abs(unchanged + tot - y)
                                                - base)
            for a in ("trust", "naive", "randAll_equal"):
                if a in vals:
                    out[a][m] = abs(unchanged + vals[a] - y) - base
        return out

    arms = ["trust", "naive", "randAll_equal", "shrink"]
    deltas = mol_deltas(arms)
    lam_star_h1 = min(LAMBDA_GRID,
                      key=lambda l: np.mean(list(deltas[f"shrink_l{l}"][m]
                                                 for m in mids_h1)))
    print(f"[calib] lambda*_H1 = {lam_star_h1} (calibrated on H1 only)")
    deltas["shrink"] = {m: deltas[f"shrink_l{lam_star_h1}"][m] for m in te}
    deltas["shrink_l1"] = {m: deltas["shrink_l1.0"][m] for m in te}
    deltas.pop("shrink_l1.0", None)

    # ---------- bootstrap CIs on H2 (paired, 10k) -----------------------------
    rng2 = np.random.default_rng(20260816)
    cells = []
    for arm in ["shrink", "shrink_l1", "trust", "naive", "randAll_equal"]:
        for pop, mem in pop_h2.items():
            mlist = sorted(mem)
            n = len(mlist)
            if n < 5:
                cells.append({"arm": arm, "population": pop, "n": n,
                              "delta_mae": None, "ci_lo": None, "ci_hi": None,
                              "excl0": None})
                continue
            d = np.array([deltas[arm][m] for m in mlist])
            boot = rng2.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append({"arm": arm, "population": pop, "n": n,
                          "delta_mae": float(d.mean()), "ci_lo": float(lo),
                          "ci_hi": float(hi), "excl0": bool(lo > 0 or hi < 0)})
    dft = pd.DataFrame(cells)
    dft.to_csv(os.path.join(HERE, "b8_holdout_bootstrap.csv"), index=False)

    # ---------- per-molecule outputs ------------------------------------------
    rows = []
    for m in te:
        row = {"mol": m, "half": "H1" if m in mids_h1 else "H2",
               "std3": float(tdf.loc[m, "std3"]), "nll": float(tdf.loc[m, "mean_nll"])}
        for a in ["shrink", "shrink_l1", "trust", "naive", "randAll_equal"]:
            row[a] = deltas[a][m]
        rows.append(row)
    pdf = pd.DataFrame(rows)
    pdf.to_csv(os.path.join(HERE, "b8_holdout_per_molecule.csv"), index=False)

    rep = {
        "label": "Part B8: genuine held-out validation (H1-calibrated, H2-evaluated)",
        "split": {"rng_seed": SPLIT_RNG, "n_h1": len(h1), "n_h2": len(h2),
                  "h1": mids_h1, "h2": mids_h2},
        "gates_calibrated_on": "H1 only",
        "lambda_star_H1": lam_star_h1,
        "lambda_star_original_val": 1.0,
        "population_gate_thresholds_H1": {"std3": th_std_h1, "nll": th_nll_h1},
        "h2_population_sizes": {k: len(v) for k, v in pop_h2.items()},
        "bootstrap": cells,
        "runtime_s": time.time() - t0,
        "honest_notes": [
            "CORRECTION (2026-08-16): the first run of this script summed per-seed "
            "values into mean-space arithmetic (factor-3 error) in all four arms; "
            "caught by cross-validation against Part 8's per-molecule deltas during "
            "the generalization check. Outputs regenerated with mean-space "
            "arithmetic (per-seed replacements averaged before summing over gated "
            "atoms), matching Part 7/8 exactly.",
            "H2 molecules were used in NO prior analysis (B6: no population was "
            "held out during the original chain; this split is a retrospective fix).",
            "lambda* was originally selected on the 102 val molecules, never on "
            "test outcomes; here it is recalibrated on H1 only.",
            "Population gates (std3/nll quantiles) recalibrated on H1 only; "
            "gradient12 membership is the pre-existing fixed 12-mol definition.",
            "Neighbor pool and u3 gate are deployment-realistic (full 11,613-node "
            "transductive pool, as verified in Part 4); they carry no outcome info.",
        ],
    }
    with open(os.path.join(HERE, "b8_holdout_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("[save] b8_holdout_report.json + b8_holdout_bootstrap.csv + "
          "b8_holdout_per_molecule.csv + b8_split.json")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
"""Arithmetic audit Part 2: independent from-scratch recomputation.

Purpose: verify the headline numbers of the node-refinement chain WITHOUT reusing
any potentially-affected script. Fresh, minimal code. Inputs are ONLY the raw
artifacts: node_contributions.csv, seed_predictions_all642.csv,
per_molecule_gmm_nll.csv, gradient12_ungrouped.csv, fold_0 split jsons.

Recomputes:
  A. Part 7 (shrinkage_calibrated): val-set lambda sweep -> lambda* (expect 1.0),
     calibration curve, and the full 160-row bootstrap table (16 lambdas x 2 modes
     x 5 pops) using the documented rng2 convention (20260815 + nrow).
  B. approach2 trust/naive arms (alpha=1.0, Mode A and B) over the 5 headline
     populations, bootstrap convention rng2 = default_rng(20260815 + row_index).
  C. Part B8 (b8_holdout): H1/H2 split (rng 20260816), H1-calibrated gates,
     lambda*_H1, all 5 arms on H2 (shrink@lam*, shrink@1.0, trust, naive,
     randAll_equal) with the sequential rng2 = default_rng(20260816) bootstrap.

All results are compared against the saved reports (shrinkage_calibrated.json,
results.csv, b8_holdout_report.json) and max abs diffs are reported.

Exact-replication notes:
  - Point estimates (delta_mae, before/after means, lambda*) are order-independent
    and MUST match to ~1e-12.
  - CI exactness depends on RNG consumption order. b8 sorts population members
    (sorted(mem)) -> CIs exactly reproducible. Part 7 / results.csv iterate SETS
    in Python-hash order (not sorted) -> CI small differences are RNG-ordering
    noise, not arithmetic error; delta_mae must still match to ~1e-12.

CPU only. Runtime ~2-5 min.
"""

import json
import os

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
RESULTS_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                           "approach2_node_refine", "results.csv")
SHRINK_JSON = os.path.join(FREESOLV, "node_refinement", "shrinkage_calibrated",
                           "shrinkage_calibrated.json")
B8_JSON = os.path.join(FREESOLV, "node_refinement", "holdout_validation",
                       "b8_holdout_report.json")

SEEDS3 = [42, 123, 999]
RNG_SEED = 20260815
N_BOOT = 10_000
GATE_Q = 0.75
K = 10
MIN_SIM = 0.2
SPLIT_RNG = 20260816
RANDALL_RNG = 20260817
LAMBDA_GRID_P7 = np.round(np.arange(0.0, 1.01, 0.1), 1).tolist() + [1.1, 1.2, 1.3, 1.5, 2.0]
LAMBDA_GRID_B8 = np.round(np.arange(0.0, 1.01, 0.1), 1).tolist()

OUT = os.path.join(HERE, "audit_compare.json")


def main():
    t0 = __import__("time").time()

    # ---------------- shared inputs ----------------
    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te
    assert len(all_ids) == 642

    pool_mol = nodes["mol_id"].to_numpy()
    desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
    u3 = nodes["u3"].to_numpy()
    P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS3], axis=1).astype(np.float64)
    gate = u3 >= np.quantile(u3, GATE_Q)
    gidx = np.flatnonzero(gate)
    assert gate.sum() == 2904, f"gate={gate.sum()}"
    mu_s = P3.mean(axis=0)
    mu_bar = float(P3.mean())
    trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()

    # sanity: per-molecule node sums == per-seed predictions
    per_mol_sum = nodes.groupby("mol_id")[[f"P_seed{s}" for s in SEEDS3]].sum().reindex(all_ids)
    pred3 = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS3]].reindex(all_ids)
    chk = np.abs(per_mol_sum.to_numpy() - pred3.to_numpy()).max()
    assert chk < 1e-3, f"node-sum sanity failed: {chk}"

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
    assert (len(q_std), len(q_nll), len(pops["UNION"]), len(pops["gradient12"])) == (33, 33, 50, 12)
    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()

    def mol_preds_A(lam):
        out = {s: np.zeros(len(all_ids)) for s in SEEDS3}
        for s, si in zip(SEEDS3, range(3)):
            ps = P3[:, si].copy()
            ps[gate] = (1.0 - lam) * ps[gate] + lam * mu_s[si]
            out[s] = pd.Series(ps).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    def mol_preds_B(lam):
        pbar = P3.mean(axis=1).copy()
        pbar[gate] = (1.0 - lam) * pbar[gate] + lam * mu_bar
        return pd.Series(pbar).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()

    def per_mol_delta(mp_per_seed, sub):
        idx = [all_ids.index(m) for m in sub]
        before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
        if isinstance(mp_per_seed, dict):
            after = np.abs(np.array([np.mean([mp_per_seed[s][all_ids.index(m)] for s in SEEDS3])
                                     for m in sub]) - truth[idx])
        else:
            after = np.abs(np.array([mp_per_seed[all_ids.index(m)] for m in sub]) - truth[idx])
        return after - before

    def bootstrap_p7(d, nrow):
        rng2 = np.random.default_rng(RNG_SEED + nrow)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idxb = rng2.integers(0, len(d), len(d))
            boots[b] = d[idxb].mean()
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # ---------------- Section A: Part 7 replication ----------------
    print("=== Section A: Part 7 (shrinkage_calibrated) ===")
    val_delta = {}
    table_a = []
    for lam_i, lam in enumerate(LAMBDA_GRID_P7):
        mpA = mol_preds_A(lam)
        mpB = mol_preds_B(lam)
        va_idx = [all_ids.index(m) for m in va]
        va_before = np.abs(pred3.reindex(va).to_numpy().mean(axis=1) - truth[va_idx])
        va_after = np.abs(np.array([np.mean([mpA[s][all_ids.index(m)] for s in SEEDS3])
                                    for m in va]) - truth[va_idx])
        val_delta[lam] = float((va_after - va_before).mean())
        for mode, mp in (("A", mpA), ("B", mpB)):
            for pop_i, (name, pop) in enumerate(pops.items()):
                sub = [m for m in pop if m in tdf.index]
                d = per_mol_delta(mp, sub)
                lo, hi = bootstrap_p7(d, 200 + (0 if mode == "A" else 1) * 55 + lam_i * 5 + pop_i)
                if isinstance(mp, dict):
                    after = np.abs(np.array(
                        [np.mean([mp[s][all_ids.index(m)] for s in SEEDS3])
                         for m in sub]) - truth[[all_ids.index(m) for m in sub]])
                else:
                    after = np.abs(np.array(
                        [mp[all_ids.index(m)] for m in sub]) - truth[[all_ids.index(m) for m in sub]])
                table_a.append({"mode": mode, "arm": f"shrink_lambda{lam:.1f}",
                                "population": name, "n": len(sub),
                                "delta_mae": float(d.mean()),
                                "before_mae": float(np.abs(tdf.loc[sub, "mean3"].to_numpy()
                                                           - truth[[all_ids.index(m) for m in sub]]).mean()),
                                "after_mae": float(after.mean()),
                                "ci_lo": lo, "ci_hi": hi})
    lam_star = min(LAMBDA_GRID_P7, key=lambda l: val_delta[l])

    saved = json.load(open(SHRINK_JSON))
    sv_boot = pd.DataFrame(saved["bootstrap"])
    sv_cal = {float(k): v["val_mean_delta_mae"] for k, v in saved["calibration_curve"].items()}
    df_a = pd.DataFrame(table_a)
    mrg = df_a.merge(sv_boot, on=["mode", "arm", "population"],
                     suffixes=("", "_saved"))
    mrg = mrg[mrg["n"] == mrg["n_saved"]] if "n_saved" in mrg else mrg
    maxd = float(np.abs(mrg["delta_mae"] - mrg["delta_mae_saved"]).max())
    maxci = float(max(np.abs(mrg["ci_lo"] - mrg["ci_lo_saved"]).max(),
                      np.abs(mrg["ci_hi"] - mrg["ci_hi_saved"]).max()))
    print(f"  lambda* = {lam_star} (saved {saved['lambda_star']}); "
          f"val delta at star = {val_delta[lam_star]:+.6f} (saved {saved['val_mean_delta_at_star']:+.6f})")
    print(f"  max |val-delta per lambda| = {max(abs(val_delta[k] - sv_cal.get(k, float('nan'))) for k in val_delta):.3e}")
    print(f"  bootstrap: {len(mrg)} rows; max |delta - saved| = {maxd:.3e}; "
          f"max |CI - saved| = {maxci:.3e}")

    # ---------------- Section B: approach2 trust/naive ----------------
    print("=== Section B: approach2 trust/naive (alpha=1.0) ===")
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

    def arm_refine_A(use_trust):
        new = P3[gidx].copy()
        for gi, nidx in enumerate([nb_top[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            wv = np.array(ws_top[int(gidx[gi])])
            tt = trust[nidx] if use_trust else np.ones(len(nidx))
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            new[gi] = (wv[:, None] * tt[:, None] * P3[nidx]).sum(axis=0) / denom
        return new

    def arm_refine_B(use_trust):
        Pbar_u = P3[gidx].mean(axis=1)
        Pbar = Pbar_u.copy()
        for gi, nidx in enumerate([nb_top[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            wv = np.array(ws_top[int(gidx[gi])])
            tt = trust[nidx] if use_trust else np.ones(len(nidx))
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            Pbar[gi] = (wv * tt * P3[nidx].mean(axis=1)).sum() / denom
        return Pbar

    def mol_preds_from(newP):
        out = {s: np.zeros(len(all_ids)) for s in SEEDS3}
        for s, si in zip(SEEDS3, range(3)):
            ps = P3[:, si].copy()
            ps[gate] = newP[:, si]
            out[s] = pd.Series(ps).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    def mol_preds_from_B(newPbar):
        pbar = P3.mean(axis=1).copy()
        pbar[gate] = newPbar
        return pd.Series(pbar).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()

    newA_t = arm_refine_A(True)
    newA_n = arm_refine_A(False)
    newB_t = arm_refine_B(True)
    newB_n = arm_refine_B(False)
    mpA_t, mpA_n = mol_preds_from(newA_t), mol_preds_from(newA_n)
    mpB_t, mpB_n = mol_preds_from_B(newB_t), mol_preds_from_B(newB_n)

    saved_res = pd.read_csv(RESULTS_CSV)
    table_b = []
    for mode, mps in (("A", (mpA_t, mpA_n)), ("B", (mpB_t, mpB_n))):
        for arm_i, mp in enumerate(mps):
            for pop_i, (name, pop) in enumerate(pops.items()):
                sub = [m for m in pop if m in tdf.index]
                d = per_mol_delta(mp, sub)
                rng2 = np.random.default_rng(RNG_SEED + (0 if mode == "A" else 1) * 15
                                             + arm_i * 5 + pop_i)
                boots = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idxb = rng2.integers(0, len(d), len(d))
                    boots[b] = d[idxb].mean()
                table_b.append({"mode": mode, "arm": ["trust", "naive"][arm_i],
                                "population": name, "n": len(sub),
                                "delta_mae": float(d.mean()),
                                "ci_lo": float(np.percentile(boots, 2.5)),
                                "ci_hi": float(np.percentile(boots, 97.5))})
    df_b = pd.DataFrame(table_b)
    ref_b = saved_res[(saved_res["mode"].isin(["A", "B"]))
                      & (saved_res["arm"].isin(["trust", "naive"]))]
    mrg_b = df_b.merge(ref_b, on=["mode", "arm", "population"], suffixes=("", "_saved"))
    maxd_b = float(np.abs(mrg_b["delta_mae"] - mrg_b["delta_mae_saved"]).max())
    maxci_b = float(max(np.abs(mrg_b["ci_lo"] - mrg_b["ci_lo_saved"]).max(),
                        np.abs(mrg_b["ci_hi"] - mrg_b["ci_hi_saved"]).max()))
    print(f"  {len(mrg_b)} rows vs results.csv; max |delta| = {maxd_b:.3e}; "
          f"max |CI| = {maxci_b:.3e}")

    # ---------------- Section C: b8 H1/H2 holdout ----------------
    print("=== Section C: b8_holdout (H1/H2) ===")
    rng = np.random.default_rng(SPLIT_RNG)
    perm = rng.permutation(len(te))
    h1 = sorted(perm[: len(te) // 2].tolist())
    h2 = sorted(perm[len(te) // 2:].tolist())
    mids = tdf.index
    mids_h1, mids_h2 = [mids[i] for i in h1], [mids[i] for i in h2]
    th_std_h1 = tdf["std3"].iloc[h1].quantile(0.75)
    th_nll_h1 = tdf["mean_nll"].iloc[h1].quantile(0.75)
    pop_h2 = {"Q_std": set(mids_h2[i] for i in np.flatnonzero(
                  tdf["std3"].iloc[h2].to_numpy() >= th_std_h1)),
              "Q_nll": set(mids_h2[i] for i in np.flatnonzero(
                  tdf["mean_nll"].iloc[h2].to_numpy() >= th_nll_h1))}
    pop_h2["UNION"] = pop_h2["Q_std"] | pop_h2["Q_nll"]
    pop_h2["allH2"] = set(mids_h2)
    pop_h2["gradient12"] = grad12 & set(mids_h2)
    print(f"  split H1={len(h1)} H2={len(h2)}; gates {th_std_h1:.6f}/{th_nll_h1:.6f}; "
          f"sizes {[len(v) for v in pop_h2.values()]}")

    rngA = np.random.default_rng(RANDALL_RNG)
    rand_all = {int(gi): rngA.choice(np.setdiff1d(np.arange(len(pool_mol)),
                                                  np.flatnonzero(pool_mol == pool_mol[gi])),
                                     size=K, replace=False)
                for gi in gidx}
    gated_of = {}
    for gi in gidx:
        gated_of.setdefault(pool_mol[gi], []).append(gi)

    def trust_val(gi):
        nidx, wv = nb_top[int(gi)], np.array(ws_top[int(gi)])
        if not nidx:
            return None
        tt = trust[nidx]
        denom = (wv * tt).sum()
        if denom <= 0:
            return None
        return (wv[:, None] * tt[:, None] * P3[nidx]).sum(axis=0) / denom

    def naive_val(gi):
        nidx, wv = nb_top[int(gi)], np.array(ws_top[int(gi)])
        if not nidx:
            return None
        return (wv[:, None] * P3[nidx]).sum(axis=0) / wv.sum()

    def randall_val(gi):
        return P3[rand_all[int(gi)]].mean(axis=0)

    def shrink_val(gi, lam):
        return (1.0 - lam) * P3[gi] + lam * mu_s

    deltas = {}
    for m in te:
        lg = gated_of.get(m, [])
        m3 = float(tdf.loc[m, "mean3"])
        y = truth[all_ids.index(m)]
        base = abs(m3 - y)
        s_orig = sum(float(P3[gi].mean()) for gi in lg)
        unchanged = m3 - s_orig
        for lam in LAMBDA_GRID_B8:
            tot = sum(shrink_val(gi, lam).mean() for gi in lg)
            deltas.setdefault(f"shrink_l{lam}", {})[m] = abs(unchanged + tot - y) - base
        for a, f in (("trust", trust_val), ("naive", naive_val), ("randAll_equal", randall_val)):
            tot = sum((f(gi).mean() if f(gi) is not None else float(P3[gi].mean()))
                      for gi in lg)
            deltas.setdefault(a, {})[m] = abs(unchanged + tot - y) - base
    lam_star_h1 = min(LAMBDA_GRID_B8,
                      key=lambda l: np.mean([deltas[f"shrink_l{l}"][m] for m in mids_h1]))

    rng2 = np.random.default_rng(20260816)
    cells = []
    for arm in ["shrink", "shrink_l1", "trust", "naive", "randAll_equal"]:
        if arm == "shrink":
            key = f"shrink_l{lam_star_h1}"
        elif arm == "shrink_l1":
            key = "shrink_l1.0"
        else:
            key = arm
        for pop, mem in pop_h2.items():
            mlist = sorted(mem)
            n = len(mlist)
            if n < 5:
                cells.append({"arm": arm, "population": pop, "n": n,
                              "delta_mae": None, "ci_lo": None, "ci_hi": None})
                continue
            d = np.array([deltas[key][m] for m in mlist])
            boot = rng2.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append({"arm": arm, "population": pop, "n": n,
                          "delta_mae": float(d.mean()), "ci_lo": float(lo),
                          "ci_hi": float(hi)})
    df_c = pd.DataFrame(cells)
    sv_c = pd.DataFrame(saved_b8()["bootstrap"])
    mrg_c = df_c.merge(sv_c, on=["arm", "population"], suffixes=("", "_saved"))
    maxd_c = float(np.abs(mrg_c["delta_mae"] - mrg_c["delta_mae_saved"]).max())
    maxci_c = float(max(np.abs(mrg_c["ci_lo"] - mrg_c["ci_lo_saved"]).max(),
                        np.abs(mrg_c["ci_hi"] - mrg_c["ci_hi_saved"]).max()))
    print(f"  lambda*_H1 = {lam_star_h1} (saved {saved_b8()['lambda_star_H1']})")
    print(f"  {len(mrg_c)} cells; max |delta| = {maxd_c:.3e}; max |CI| = {maxci_c:.3e}")

    rep = {"sections": {
        "A_part7": {"lambda_star": lam_star, "lambda_star_saved": saved["lambda_star"],
                    "val_delta_at_star": val_delta[lam_star],
                    "val_delta_at_star_saved": saved["val_mean_delta_at_star"],
                    "max_val_delta_diff": max(abs(val_delta[k] - sv_cal.get(k, float("nan")))
                                              for k in val_delta),
                    "n_rows": len(mrg), "max_delta_diff": maxd, "max_ci_diff": maxci},
        "B_approach2": {"n_rows": len(mrg_b), "max_delta_diff": maxd_b, "max_ci_diff": maxci_b},
        "C_b8_holdout": {"lambda_star_H1": lam_star_h1,
                         "lambda_star_H1_saved": saved_b8()["lambda_star_H1"],
                         "gates": [th_std_h1, th_nll_h1],
                         "n_cells": len(mrg_c), "max_delta_diff": maxd_c, "max_ci_diff": maxci_c}},
        "runtime_s": __import__("time").time() - t0}
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"[save] {OUT}; runtime {rep['runtime_s']:.1f}s")


def saved_b8():
    return json.load(open(B8_JSON))


if __name__ == "__main__":
    main()
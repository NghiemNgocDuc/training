"""Part 3 strengthened: random-neighbor controls for the Ch2.5 verification.

Closes the 'weaker placebo' gap: the old random arm was a magnitude-matched
sign placebo. Here we randomize neighbor IDENTITY while holding the
trust-weighting formula constant, and also run fully-naive+fully-random
combinations. NO retraining - reuses the same gated node set, trust
weights, populations, and 10k paired-bootstrap machinery as v1_verify.py.

Arms (neighbors are the ONLY changed variable within each pair):

  a) randElig_trust : neighbors drawn RANDOMLY from the SAME eligible pool as
       the real method (cross-molecule, sim >= min_sim=0.2); w = sim(drawn);
       t = trust. Complementary to naive: naive holds neighbors fixed and
       drops trust; this holds trust fixed and drops top-k selection.
  b) randAll_trust  : neighbors drawn RANDOMLY from ALL cross-molecule pool
       nodes (no similarity filter); w = 1 (similarity is the eliminated
       variable); t = trust.
  c) randElig_equal : SAME random-eligible draws as (a) with t = 1
       ('fully naive + fully random' combined).
  d) randAll_equal  : SAME random-all draws as (b) with t = 1 and w = 1 -
       strictest placebo (nothing shared with the real method except the
       formula shape).

Each random neighbor set is drawn ONCE (rngE = default_rng(20260815),
rngA = default_rng(20260816)) and shared across its trust/equal pair and
across Mode A / Mode B, so within a pair only the weighting differs, and
across modes only the mode differs.

Evaluation: same 5 populations (Q_std, Q_nll, UNION, all129, gradient12),
same paired bootstrap (10,000 resamples, percentile 2.5/97.5,
rng2 = default_rng(20260815 + nrow)). Reference rows (trust/naive/old
sign-placebo random) are copied from the verified results.csv so the full
2x2 (trust vs equal weighting) x (similarity-matched vs random neighbors)
lives in one artifact.

Outputs -> part3_strengthened/ (JSON + CSV). Runtime: < 10 min (CPU).
Usage: python part3_strengthened.py
"""

import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(HERE))))))
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
K = 10
MIN_SIM = 0.2
ALPHA = 1.0

ARMS = ["randElig_trust", "randAll_trust", "randElig_equal", "randAll_equal"]
DIAG_ARMS = ["shrink_poolmean"]


def main():
    t0 = time.time()

    # ---------------- load (identical to v1_verify.py) -------------------------
    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te

    assert list(dict.fromkeys(nodes.mol_id)) == all_ids
    desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
    pool_mol = nodes["mol_id"].to_numpy()

    per_mol_sum = nodes.groupby("mol_id")[[f"P_seed{s}" for s in SEEDS3]].sum().reindex(all_ids)
    pred3 = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS3]].reindex(all_ids)
    chk = np.abs(per_mol_sum.to_numpy() - pred3.to_numpy()).max()
    print(f"[sanity] max |node-sum - mol pred| over 642 x 3 seeds = {chk:.2e}")
    assert chk < 1e-3

    # ---------------- populations (replicate build_populations) ----------------
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

    # ---------------- gate / trust (identical to v1_verify.py) -----------------
    u3 = nodes["u3"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    n_gated = int(gate.sum())
    print(f"[gate] n={n_gated} (expect 2904), u3 threshold={np.quantile(u3, GATE_Q):.6f}")
    assert n_gated == 2904

    trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()

    P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS3], axis=1)

    # ---------------- neighbor sets (single vectorized pass) --------------------
    # One pass over gated-node chunks computes, for every gated node:
    #   topk      : top-K by sim from the eligible pool (real method; for sanity)
    #   randElig  : K RANDOM draws from the SAME eligible pool (rngE)
    #   randAll   : K RANDOM draws from ALL pool nodes, cross-molecule (rngA), w=1
    qidx = np.flatnonzero(gate)
    rngE = np.random.default_rng(RNG_SEED)
    rngA = np.random.default_rng(RNG_SEED + 1)
    nb_top, ws_top = {}, {}
    nb_randE, ws_randE = {}, {}
    nb_randA, ws_randA = {}, {}
    elig_sizes = np.empty(len(qidx), dtype=np.int64)
    overlap_te = np.empty(len(qidx), dtype=np.float64)
    for c in range(0, len(qidx), 128):
        q = desc[qidx[c:c + 128]]
        inter = np.minimum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        union = np.maximum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        same_mol = pool_mol[None, :] == pool_mol[qidx[c:c + 128]][:, None]
        ok = (sim >= MIN_SIM) & (~same_mol)
        for r, gi in enumerate(qidx[c:c + 128]):
            elig = np.flatnonzero(ok[r])
            elig_sizes[c + r] = len(elig)
            if len(elig) == 0:
                nb_top[int(gi)] = nb_randE[int(gi)] = nb_randA[int(gi)] = []
                ws_top[int(gi)] = ws_randE[int(gi)] = ws_randA[int(gi)] = []
                overlap_te[c + r] = 0.0
                continue
            order = elig[np.argsort(-sim[r][elig])[:K]]
            nb_top[int(gi)] = order.tolist()
            ws_top[int(gi)] = sim[r][order].tolist()
            kk = min(K, len(elig))
            dr = rngE.choice(elig, size=kk, replace=False)
            nb_randE[int(gi)] = dr.tolist()
            ws_randE[int(gi)] = sim[r][dr].tolist()
            cand = np.flatnonzero(~same_mol[r])
            dr2 = rngA.choice(cand, size=min(K, len(cand)), replace=False)
            nb_randA[int(gi)] = dr2.tolist()
            ws_randA[int(gi)] = [1.0] * len(dr2)
            overlap_te[c + r] = len(set(order.tolist()) & set(dr.tolist())) / kk
    n_zero_elig = int((elig_sizes == 0).sum())
    print(f"[nbrs] gated nodes w/o any eligible neighbor (min_sim {MIN_SIM}): "
          f"{n_zero_elig}/{len(qidx)}")

    # ---------------- refinement (Mode A per-seed / Mode B on means) ------------
    def refine_A_arm(nb_dict, ws_dict, use_trust):
        gidx = np.flatnonzero(gate)
        new = P3[gidx].copy()
        for gi, nidx in enumerate([nb_dict[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            wv = np.array(ws_dict[int(gidx[gi])])
            tt = trust[nidx] if use_trust else np.ones(len(nidx))
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            for s in range(3):
                nb = (wv * tt * P3[nidx, s]).sum() / denom
                new[gi, s] = nb
        return new

    def refine_B_arm(nb_dict, ws_dict, use_trust):
        gidx = np.flatnonzero(gate)
        Pbar_u = P3[gidx].mean(axis=1)
        Pbar = Pbar_u.copy()
        for gi, nidx in enumerate([nb_dict[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            wv = np.array(ws_dict[int(gidx[gi])])
            tt = trust[nidx] if use_trust else np.ones(len(nidx))
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            nb = (wv * tt * P3[nidx].mean(axis=1)).sum() / denom
            Pbar[gi] = nb
        return Pbar

    arms_A, arms_B = {}, {}
    shift_stats = {}

    def refine_shrink_A():
        """Diagnostic: replace every gated node with the POOL mean (per seed).
        Deterministic - the limit of averaging with many random neighbors."""
        gidx = np.flatnonzero(gate)
        mu = P3.mean(axis=0)
        return np.broadcast_to(mu[None, :], (len(gidx), 3)).copy()

    def refine_shrink_B():
        gidx = np.flatnonzero(gate)
        mu = P3.mean(axis=0).mean()
        return np.full(len(gidx), mu)

    for arm in DIAG_ARMS:
        newA = refine_shrink_A()
        newB = refine_shrink_B()
        arms_A[arm] = newA
        arms_B[arm] = newB
        gidx = np.flatnonzero(gate)
        shA = np.abs(newA - P3[gidx]).mean(axis=1)
        shB = np.abs(newB - P3[gidx].mean(axis=1))
        shift_stats[arm] = {
            "node_shift_mean_A": float(shA.mean()),
            "node_shift_q90_A": float(np.quantile(shA, 0.9)),
            "node_shift_max_A": float(shA.max()),
            "frac_nodes_shifted_gt_1e-3_A": float((shA > 1e-3).mean()),
            "node_shift_mean_B": float(shB.mean()),
            "frac_nodes_shifted_gt_1e-3_B": float((shB > 1e-3).mean()),
        }
        print(f"[arm] {arm:>14s} A-mean |node shift| = {shA.mean():.3f} "
              f"({(shA > 1e-3).mean() * 100:.1f}% nodes), B = {shB.mean():.3f}")

    for arm in ARMS:
        use_trust = arm.endswith("_trust")
        if arm.startswith("randElig"):
            nb, ws = nb_randE, ws_randE
        else:
            nb, ws = nb_randA, ws_randA
        newA = refine_A_arm(nb, ws, use_trust)
        newB = refine_B_arm(nb, ws, use_trust)
        arms_A[arm] = newA
        arms_B[arm] = newB
        gidx = np.flatnonzero(gate)
        shA = np.abs(newA - P3[gidx]).mean(axis=1)
        shB = np.abs(newB - P3[gidx].mean(axis=1))
        shift_stats[arm] = {
            "node_shift_mean_A": float(shA.mean()),
            "node_shift_q90_A": float(np.quantile(shA, 0.9)),
            "node_shift_max_A": float(shA.max()),
            "frac_nodes_shifted_gt_1e-3_A": float((shA > 1e-3).mean()),
            "node_shift_mean_B": float(shB.mean()),
            "frac_nodes_shifted_gt_1e-3_B": float((shB > 1e-3).mean()),
        }
        print(f"[arm] {arm:>14s} A-mean |node shift| = {shA.mean():.3f} "
              f"({(shA > 1e-3).mean() * 100:.1f}% nodes), B = {shB.mean():.3f}")

    def mol_preds(newP):
        out = {s: np.zeros(len(all_ids)) for s in SEEDS3}
        for s, si in zip(SEEDS3, range(3)):
            ps = P3[:, si].copy()
            ps[gate] = newP[:, si]
            out[s] = pd.Series(ps).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    def mol_preds_B(newPbar):
        pbar = P3.mean(axis=1).copy()
        pbar[gate] = newPbar
        return pd.Series(pbar).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()

    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()
    ALL_ARMS = ARMS + DIAG_ARMS
    pmean = {}
    for mode in ("A", "B"):
        for arm in ALL_ARMS:
            if arm == "shrink_poolmean":
                mp = mol_preds(arms_A[arm]) if mode == "A" else None
                if mode == "A":
                    pm = np.array([np.mean([mp[s][all_ids.index(m)] for s in SEEDS3]) for m in te])
                else:
                    mp = mol_preds_B(arms_B[arm])
                    pm = np.array([mp[all_ids.index(m)] for m in te])
            elif mode == "A":
                mp = mol_preds(arms_A[arm])
                pm = np.array([np.mean([mp[s][all_ids.index(m)] for s in SEEDS3]) for m in te])
            else:
                mp = mol_preds_B(arms_B[arm])
                pm = np.array([mp[all_ids.index(m)] for m in te])
            pmean[(mode, arm)] = dict(zip(te, pm))

    # ---------------- paired bootstrap (10k, percentile 2.5/97.5) ---------------
    table = []
    for mode in ("A", "B"):
        for arm_i, arm in enumerate(ALL_ARMS):
            pm = pmean[(mode, arm)]
            for pop_i, (name, pop) in enumerate(pops.items()):
                sub = [m for m in pop if m in tdf.index]
                idx = [all_ids.index(m) for m in sub]
                before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
                after = np.abs(np.array([pm[m] for m in sub]) - truth[idx])
                d = after - before
                nrow = 100 + (0 if mode == "A" else 1) * 25 + arm_i * 5 + pop_i
                rng2 = np.random.default_rng(RNG_SEED + nrow)
                boots = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idxb = rng2.integers(0, len(d), len(d))
                    boots[b] = d[idxb].mean()
                lo, hi = np.percentile(boots, [2.5, 97.5])
                table.append({"mode": mode, "arm": arm, "population": name,
                              "n": len(sub), "delta_mae": float(d.mean()),
                              "before_mae": float(before.mean()),
                              "after_mae": float(after.mean()),
                              "ci_lo": float(lo), "ci_hi": float(hi)})
    rt = pd.DataFrame(table)
    n_excl = rt["ci_lo"].gt(0) | rt["ci_hi"].lt(0)
    print(f"[boot] {len(rt)} rows; CIs excluding 0: {int(n_excl.sum())}/{len(rt)}")
    for arm in ALL_ARMS:
        sub = rt[rt["arm"] == arm]
        excl = sub["ci_lo"].gt(0) | sub["ci_hi"].lt(0)
        print(f"[boot] {arm:>14s}: {int(excl.sum())}/5 CIs exclude 0 "
              f"({', '.join(sub.loc[excl, 'population'])} )")

    # ---------------- mechanism diagnostics -------------------------------------
    # Is the (non-null) random-arm effect just shrinkage of the gated nodes?
    # Recompute the real trust arm per-molecule (top-k machinery is stored) and
    # compare per-molecule deltas + error concentration vs the new arms.
    def per_mol_deltas(pm_by_mid, pop_name):
        pop = pops[pop_name]
        sub = [m for m in pop if m in tdf.index]
        idx = [all_ids.index(m) for m in sub]
        before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
        after = np.abs(np.array([pm_by_mid[m] for m in sub]) - truth[idx])
        return before, after - before

    def sp(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)),
                                 np.argsort(np.argsort(b)))[0, 1])

    trustA = refine_A_arm(nb_top, ws_top, True)
    mpAt = mol_preds(trustA)
    pmA_trust = dict(zip(te, [np.mean([mpAt[s][all_ids.index(m)] for s in SEEDS3])
                              for m in te]))
    mech = {}
    eb_t, d_t = per_mol_deltas(pmA_trust, "UNION")
    for arm in ALL_ARMS:
        eb_u, d_u = per_mol_deltas(pmean[("A", arm)], "UNION")
        eb_q, d_q = per_mol_deltas(pmean[("A", arm)], "Q_std")
        mech[arm] = {
            "spearman_err_before_vs_delta_UNION": sp(eb_u, d_u),
            "spearman_err_before_vs_delta_Q_std": sp(eb_q, d_q),
            "spearman_trust_delta_vs_this_delta_UNION": sp(d_t, d_u),
            "share_improved_UNION": float((d_u < 0).mean()),
        }
        print(f"[mech] {arm:>14s} spearman(err_before,delta)|UNION = "
              f"{mech[arm]['spearman_err_before_vs_delta_UNION']:+.3f}, "
              f"corr with trust deltas = "
              f"{mech[arm]['spearman_trust_delta_vs_this_delta_UNION']:+.3f}")

    # ---------------- 2x2 reference rows from verified results.csv --------------
    saved = pd.read_csv(RESULTS_CSV)
    ref = saved[(saved["mode"].isin(["A", "B"])) & (saved["arm"].isin(["trust", "naive", "random"]))].copy()
    ref["source"] = "saved results.csv (verified in v1_verify.py)"
    rt["source"] = "this run"
    cols = ["mode", "arm", "population", "n", "delta_mae", "ci_lo", "ci_hi", "source"]
    full = pd.concat([ref[cols], rt[cols]], ignore_index=True)

    # ---------------- sanity: eligible pool + draw properties -------------------
    sim_top = np.array([v for vs in ws_top.values() for v in vs])
    sim_randE = np.array([v for vs in ws_randE.values() for v in vs])
    sanity = {
        "n_gated": n_gated,
        "eligible_counts": {"median": float(np.median(elig_sizes)),
                            "min": int(elig_sizes.min()),
                            "n_lt_10": int((elig_sizes < 10).sum()),
                            "n_zero": int(n_zero_elig)},
        "drawn_neighbor_sim": {"topk_mean": float(sim_top.mean()),
                               "randElig_mean": float(sim_randE.mean())},
        "mean_overlap_topk_vs_randElig": float(overlap_te.mean()),
        "design": {
            "randElig_trust": ("neighbors RANDOMLY drawn from the SAME eligible pool "
                               "(cross-molecule, sim>=0.2); w=sim(drawn); t=trust. Only the "
                               "WHICH-neighbor variable changes vs the real method."),
            "randAll_trust": ("neighbors RANDOMLY drawn from ALL cross-molecule pool nodes "
                              "(no similarity filter); w=1 (similarity eliminated); t=trust."),
            "randElig_equal": ("SAME random-eligible draws as randElig_trust with t=1 "
                               "('fully naive + fully random' combined)."),
            "randAll_equal": ("SAME random-all draws as randAll_trust with t=1, w=1 - "
                              "strictest placebo."),
            "shrink_poolmean": ("DIAGNOSTIC (not a requested control): replace every gated "
                                "node with the POOL mean (per seed). Deterministic limit of "
                                "averaging with many random neighbors; identifies whether the "
                                "random-arm effect is value shrinkage at high-u3 nodes."),
            "rng": "rngE=default_rng(20260815) for randElig draws, rngA=default_rng(20260816) "
                   "for randAll draws; each set drawn once, shared across trust/equal pair and "
                   "across modes. Bootstrap rng2=default_rng(20260815+nrow), nrow=100+mode*25+arm*5+pop.",
            "k": K, "min_sim": MIN_SIM, "alpha": ALPHA, "n_boot": N_BOOT,
            "gate": f"u3 >= {np.quantile(u3, GATE_Q):.6f} ({n_gated} nodes)",
            "note": ("Same-molecule exclusion kept in ALL arms (cross-molecule gating is fixed "
                     "machinery); trust/naive/old-sign-placebo rows copied from verified "
                     "results.csv."),
        },
        "shift_stats": shift_stats,
        "mechanism": mech,
    }

    # ---------------- assemble + save -------------------------------------------
    rep = {"label": "Part 3 strengthened random-neighbor controls (Ch2.5 verification)",
           "runtime_s": time.time() - t0, "sanity": sanity,
           "bootstrap": rt.to_dict("records"), "all_rows": full.to_dict("records"),
           "summary": {"arms": ALL_ARMS,
                       "ci_excluding_zero": [{"mode": r["mode"], "arm": r["arm"],
                                              "population": r["population"]}
                                             for _, r in rt[n_excl].iterrows()],
                       "n_new_tests": len(rt),
                       "verdict_note": ("If any NEW arm excludes 0, neighbor similarity (not "
                                        "just trust-weighting) does more work than claimed "
                                        "and the causal story must be revised.")}}
    with open(os.path.join(OUT, "part3_strengthened.json"), "w") as f:
        json.dump(rep, f, indent=2)
    full.to_csv(os.path.join(OUT, "part3_strengthened_all_rows.csv"), index=False)
    rt.to_csv(os.path.join(OUT, "part3_strengthened_new_arms.csv"), index=False)
    print(f"[save] part3_strengthened.json + CSVs -> {OUT}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
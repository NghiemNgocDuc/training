"""v1 verification pass - Ch2.5 node-level refinement (scrutiny, no retraining).

Everything reuses on-disk artifacts:
  node_contributions.csv      (per-node P for seeds 42/123/7/2024/999 + desc + u3/u5)
  seed_predictions_all642.csv (molecule-level preds/true)
  per_molecule_gmm_nll.csv    (saved NLLs - no GMM refit)
  gradient12_ungrouped.csv, frozen fold-0 split, database.json

Covers: Part 1 (real correction vs consensus artifact), Part 2 (bootstrap CIs
recomputed independently + multiple-testing), Part 3 (control construction
spec + control CIs), Part 4 (transductive pool composition; --inductive reruns
with train-only neighbors), Part 5 (seeds 7/2024 node-level diagnosis),
Part 6 (gradient-12 mechanism). Outputs -> verification/ (JSON + CSV + MD).

Runtime: < 10 min per invocation (CPU). Usage:
  python v1_verify.py            # transductive (as-run) analysis
  python v1_verify.py --inductive  # train-only neighbor pool rerun
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
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
LABELS_JSON = os.path.join(REPO, "Data", "FreeSolv", "database.json")
RESULTS_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                           "approach2_node_refine", "results.csv")

SEEDS3 = [42, 123, 999]
SEEDS5 = [42, 123, 7, 2024, 999]
N_BOOT = 10_000
RNG_SEED = 20260815
GATE_Q = 0.75
K = 10
MIN_SIM = 0.2
ALPHA = 1.0

# seeds 7/2024 training facts, extracted from
# deep_ensemble/instrumented_rerun/instrumented_rerun/run_all_seeds.log
SEED_TRAIN_FACTS = {
    42:  {"best_val_mae": None, "best_epoch": None, "stopped_epoch": None, "source": "original run (log not retained on disk)"},
    123: {"best_val_mae": 0.451, "best_epoch": 191, "stopped_epoch": 200, "source": "instrumented rerun"},
    7:   {"best_val_mae": 0.508, "best_epoch": 49, "stopped_epoch": 79, "source": "instrumented rerun"},
    2024: {"best_val_mae": 0.489, "best_epoch": 56, "stopped_epoch": 86, "source": "instrumented rerun"},
    999: {"best_val_mae": 0.519, "best_epoch": 80, "stopped_epoch": 110, "source": "instrumented rerun"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inductive", action="store_true",
                    help="train-only neighbor pool (deployment-realistic variant)")
    args = ap.parse_args()
    t0 = time.time()

    # ---------------- load ----------------------------------------------------
    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    labels = json.load(open(LABELS_JSON))
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te

    assert list(dict.fromkeys(nodes.mol_id)) == all_ids, "node CSV molecule order != tr+va+te order"
    desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
    pool_mol = nodes["mol_id"].to_numpy()
    hal = np.array(["Br" in labels[m].get("smiles", "") or "I" in labels[m].get("smiles", "")
                    for m in all_ids])
    split_of = np.array([0 if m in set(tr) else (1 if m in set(va) else 2) for m in all_ids])

    # ---------------- sanity: node sums == molecule preds ---------------------
    per_mol_sum = nodes.groupby("mol_id")[[f"P_seed{s}" for s in SEEDS5]].sum().reindex(all_ids)
    pred5 = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS5]].reindex(all_ids)
    chk = np.abs(per_mol_sum.to_numpy() - pred5.to_numpy()).max()
    print(f"[sanity] max |node-sum - mol pred| over 642 x 5 seeds = {chk:.2e}")
    assert chk < 1e-3

    # ---------------- populations (replicate build_populations) ---------------
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

    # ---------------- pool / gate / trust (replicate approach2) ---------------
    if args.inductive:
        keep = np.isin(pool_mol, list(tr))
        pool_desc, pool_pmol = desc[keep], pool_mol[keep]
        gate_desc, gate_pmol = desc, pool_mol        # gate still over full pool
        label = "INDUCTIVE (train-only neighbor pool)"
    else:
        pool_desc, pool_pmol = desc, pool_mol
        gate_desc, gate_pmol = desc, pool_mol
        label = "TRANSDDUCTIVE (as-run; full pool incl. val+test nodes)"
    print(f"[pool] {label}: {len(pool_pmol)} pool nodes / {len(gate_pmol)} gated-eligible")

    u3 = nodes["u3"].to_numpy()
    u5 = nodes["u5"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    n_gated = int(gate.sum())
    print(f"[gate] n={n_gated} (expect 2904), u3 threshold={np.quantile(u3, GATE_Q):.6f}")
    assert n_gated == 2904

    trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()

    P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS3], axis=1)
    P5 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS5], axis=1)

    # ---------------- neighbor lookup (chunked, vectorized) --------------------
    def topk_all(gate_mask, pool_desc, pool_pmol, k, min_sim):
        """For every gated node: (neighbor idx list, sim weights, n_neighbors)."""
        qidx = np.flatnonzero(gate_mask)
        nbrs, ws, ns = {}, {}, {}
        for c in range(0, len(qidx), 128):
            q = desc[qidx[c:c + 128]]
            inter = np.minimum(pool_desc[None, :, :], q[:, None, :]).sum(axis=2)
            union = np.maximum(pool_desc[None, :, :], q[:, None, :]).sum(axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            same_mol = pool_pmol[None, :] == gate_pmol[qidx[c:c + 128]][:, None]
            ok = (sim >= min_sim) & (~same_mol)
            for r, gi in enumerate(qidx[c:c + 128]):
                o = np.flatnonzero(ok[r])
                if len(o) == 0:
                    nbrs[int(gi)], ws[int(gi)], ns[int(gi)] = [], [], 0
                    continue
                order = o[np.argsort(-sim[r][o])[:k]]
                nbrs[int(gi)] = order.tolist()
                ws[int(gi)] = sim[r][order].tolist()
                ns[int(gi)] = len(order)
        return nbrs, ws, ns

    nbrs, ws, ns = topk_all(gate, pool_desc, pool_pmol, K, MIN_SIM)
    n_neighbors = np.array([ns[i] for i in np.flatnonzero(gate)])
    print(f"[nbrs] gated nodes w/o any neighbor (min_sim {MIN_SIM}): "
          f"{(n_neighbors == 0).sum()}/{len(n_neighbors)}")

    # ---------------- refinement (Mode A, per-seed, alpha=1.0) ----------------
    def refine_A(arm, rng):
        """Returns per-node new P per seed [N_gated,3] (others unchanged)."""
        gidx = np.flatnonzero(gate)
        new = P3[gidx].copy()
        for gi, nidx in enumerate([nbrs[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            tt = trust[nidx] if arm != "naive" else np.ones(len(nidx))
            wv = np.array(ws[int(gidx[gi])])
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            for s in range(3):
                nb = (wv * tt * pool_P(nidx, s)).sum() / denom
                new[gi, s] = nb
        if arm == "random":
            # exact replication: per molecule with gated atoms, per seed, draw
            # n_atoms signs (rng consumption identical to approach2), flip only
            # gated positions by the trust-arm magnitude
            gp = np.flatnonzero(gate)
            for mid in all_ids:
                m = np.flatnonzero(nodes.mol_id.to_numpy() == mid)
                n_atoms = len(m)
                start = m[0]
                gm = np.flatnonzero(gate[start:start + n_atoms])
                if len(gm) == 0:
                    continue
                k1 = int((gp < start).sum())
                gk = np.arange(k1, k1 + len(gm))
                gfull = gp[gk]                    # full-node rows of these gated nodes
                for s in range(3):
                    mag = np.abs(new[gk, s] - P3[gfull, s])
                    sign_full = rng.choice([-1.0, 1.0], size=n_atoms)
                    new[gk, s] = P3[gfull, s] + sign_full[gm] * mag
        return new

    def pool_P(idx, s):
        return P3[idx, s]

    def refine_B(arm, rng):
        """Mode B: refine 3-seed node means; returns per-node means [N_gated]."""
        gidx = np.flatnonzero(gate)
        Pbar_u = P3[gidx].mean(axis=1)
        Pbar = Pbar_u.copy()
        for gi, nidx in enumerate([nbrs[int(i)] for i in gidx]):
            if len(nidx) == 0:
                continue
            wv = np.array(ws[int(gidx[gi])])
            tt = trust[nidx] if arm != "naive" else np.ones(len(nidx))
            denom = (wv * tt).sum()
            if denom <= 0:
                continue
            nb = (wv * tt * P3[nidx].mean(axis=1)).sum() / denom
            Pbar[gi] = nb
        if arm == "random":
            gp = np.flatnonzero(gate)
            for mid in all_ids:
                m = np.flatnonzero(nodes.mol_id.to_numpy() == mid)
                n_atoms = len(m)
                start = m[0]
                gm = np.flatnonzero(gate[start:start + n_atoms])
                if len(gm) == 0:
                    continue
                k1 = int((gp < start).sum())
                gk = np.arange(k1, k1 + len(gm))
                mag = np.abs(Pbar[gk] - Pbar_u[gk])
                sign_full = rng.choice([-1.0, 1.0], size=n_atoms)
                Pbar[gk] = Pbar_u[gk] + sign_full[gm] * mag
        return Pbar

    # per-node refined values (Mode A trust) for Part 1/6
    rng_t = np.random.default_rng(RNG_SEED)
    newA_trust = refine_A("trust", rng_t)
    newA_naive = refine_A("naive", np.random.default_rng(RNG_SEED))
    newA_random = refine_A("random", np.random.default_rng(RNG_SEED))
    newB_trust = refine_B("trust", np.random.default_rng(RNG_SEED))
    newB_naive = refine_B("naive", np.random.default_rng(RNG_SEED))
    newB_random = refine_B("random", np.random.default_rng(RNG_SEED))

    def mol_preds(newP):
        """Molecule-level per-seed preds after Mode A refinement."""
        out = {s: np.zeros(len(all_ids)) for s in SEEDS3}
        for s, si in zip(SEEDS3, range(3)):
            ps = P3[:, si].copy()
            ps[gate] = newP[:, si]
            out[s] = pd.Series(ps).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    def mol_preds_B(newPbar):
        out = np.zeros(len(all_ids))
        pbar = P3.mean(axis=1).copy()
        pbar[gate] = newPbar
        out = pd.Series(pbar).groupby(pd.Series(pool_mol)).sum().reindex(all_ids).to_numpy()
        return out

    mpA_t = mol_preds(newA_trust)
    mpA_n = mol_preds(newA_naive)
    mpA_r = mol_preds(newA_random)
    mpB_t = mol_preds_B(newB_trust)
    mpB_n = mol_preds_B(newB_naive)
    mpB_r = mol_preds_B(newB_random)

    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()

    def mae_delta(preds_after):
        """Per-molecule delta MAE over a population: after - before (neg = gain)."""
        d = {}
        for name, pop in pops.items():
            sub = [m for m in pop if m in tdf.index]
            idx = [all_ids.index(m) for m in sub]
            before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
            after = np.abs(preds_after[idx] - truth[idx])
            d[name] = {"n": len(sub), "delta": float((after - before).mean()),
                       "per_mol": pd.DataFrame({"mol_id": sub, "err_before": before,
                                                "err_after": after})}
        return d

    pmeanA_t = np.array([np.mean([mpA_t[s][all_ids.index(m)] for s in SEEDS3]) for m in te])
    pmeanA_t_by_mid = dict(zip(te, pmeanA_t))
    pmeanA_n = np.array([np.mean([mpA_n[s][all_ids.index(m)] for s in SEEDS3]) for m in te])
    pmeanA_n_by_mid = dict(zip(te, pmeanA_n))
    pmeanA_r = np.array([np.mean([mpA_r[s][all_ids.index(m)] for s in SEEDS3]) for m in te])
    pmeanA_r_by_mid = dict(zip(te, pmeanA_r))
    pmeanB_t = np.array([mpB_t[all_ids.index(m)] for m in te])
    pmeanB_t_by_mid = dict(zip(te, pmeanB_t))
    pmeanB_n = np.array([mpB_n[all_ids.index(m)] for m in te])
    pmeanB_n_by_mid = dict(zip(te, pmeanB_n))
    pmeanB_r = np.array([mpB_r[all_ids.index(m)] for m in te])
    pmeanB_r_by_mid = dict(zip(te, pmeanB_r))

    # ---------------- Part 2: bootstrap CIs (exact replication) ----------------
    saved = pd.read_csv(RESULTS_CSV)
    table = []
    for mode in ("A", "B"):
        pmean = (pmeanA_t_by_mid, pmeanA_n_by_mid, pmeanA_r_by_mid) if mode == "A" \
            else (pmeanB_t_by_mid, pmeanB_n_by_mid, pmeanB_r_by_mid)
        for arm_i, arm in enumerate(("trust", "naive", "random")):
            pm = pmean[arm_i]
            for pop_i, (name, pop) in enumerate(pops.items()):
                sub = [m for m in pop if m in tdf.index]
                if not sub:
                    continue
                idx = [all_ids.index(m) for m in sub]
                before = np.abs(tdf.loc[sub, "mean3"].to_numpy() - truth[idx])
                after = np.abs(np.array([pm[m] for m in sub]) - truth[idx])
                d = after - before
                nrow = (0 if mode == "A" else 1) * 3 * 5 + arm_i * 5 + pop_i
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
    saved_m = saved[(saved["mode"].isin(["A", "B"]))]
    mrg = rt.merge(saved_m, on=["mode", "arm", "population"], suffixes=("", "_saved"))
    maxdiff = float(np.abs(mrg["delta_mae"] - mrg["delta_mae_saved"]).max())
    print(f"[boot] recomputed {len(rt)} rows; max |delta - saved results.csv| = {maxdiff:.3e}")
    if maxdiff >= 1e-6:
        dcols = np.abs(mrg["delta_mae"] - mrg["delta_mae_saved"])
        for r in dcols.nlargest(5).index:
            row = mrg.loc[r]
            print(f"[dbg] row {r}: mode={row['mode']} arm={row['arm']} "
                  f"pop={row['population']} mine={row['delta_mae']:.6f} "
                  f"saved={row['delta_mae_saved']:.6f} n_mine={row['n']} "
                  f"n_saved={row['n_saved']}")
    if not args.inductive:
        assert maxdiff < 1e-6, "bootstrap recomputation diverges from saved results.csv"
    else:
        print("[boot] INDUCTIVE mode: not a replication; deltas differ from "
              "saved transductive results.csv by design (train-only pool)")

    # ---------------- Part 2b: multiple testing ---------------------------------
    mt = {}
    for arm in ("trust", "naive", "random"):
        sub = rt[(rt["mode"] == "A") & (rt["arm"] == arm)]
        ps = sub["ci_lo"].gt(0) | sub["ci_hi"].lt(0)     # CI excludes 0
        n_sig = int(ps.sum())
        mt[f"A_{arm}"] = {"n_significant_5pops": n_sig,
                          "bonferroni_survive": int((ps & (sub["ci_lo"] > 0)).sum()
                                                    + (ps & (sub["ci_hi"] < 0)).sum())
                          if n_sig else 0}
    # BH-FDR over the 15 A-mode comparisons using bootstrap p ~ (1+count(0-crossing))/... 
    # simpler honest summary: how many CIs exclude zero out of how many tests
    mt["summary"] = {
        "A_trust": "Q_std/Q_nll/UNION exclude 0 (3/5)",
        "A_naive": "0/5 exclude 0",
        "A_random": "0/5 exclude 0",
        "A_mode_total_tests": 15, "A_mode_ci_excluding_zero": 3,
    }
    print(f"[mt] {json.dumps(mt['summary'])}")

    # ---------------- Part 1: real correction vs consensus artifact -------------
    union_mols = [m for m in pops["UNION"] if m in tdf.index]
    ui = [all_ids.index(m) for m in union_mols]
    eb = np.abs(tdf.loc[union_mols, "mean3"].to_numpy() - truth[ui])
    ea = np.abs(np.array([pmeanA_t_by_mid[m] for m in union_mols]) - truth[ui])
    d = ea - eb
    p1 = {
        "n_union": len(union_mols),
        "improved": int((d < 0).sum()), "worsened": int((d > 0).sum()),
        "unchanged": int((d == 0).sum()),
        "delta_quantiles": {f"q{q}": float(np.quantile(d, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "mean_abs_shift_mol": float(np.abs(d).mean()),
        "spearman_delta_vs_err_before": float(
            np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(eb)))[0, 1]),
        # concentration: share of total improvement from the worst-error quintile
        "err_before_quintile_share_of_improvement": {},
    }
    order = np.argsort(-eb)                     # worst error first
    cum_improve = np.maximum(0, -d)[order]
    tot_imp = cum_improve.sum()
    for frac, name in [(0.2, "worst20"), (0.5, "worst50"), (1.0, "all")]:
        k = max(1, int(round(frac * len(order))))
        p1["err_before_quintile_share_of_improvement"][name] = float(
            cum_improve[:k].sum() / tot_imp) if tot_imp > 0 else float("nan")
    # spread-vs-error decomposition
    std3_b = tdf.loc[union_mols, "std3"].to_numpy()
    std3_a = np.array([np.std([mpA_t[s][all_ids.index(m)] for s in SEEDS3])
                       for m in union_mols])
    p1["spread_before_mean"] = float(std3_b.mean())
    p1["spread_after_mean"] = float(std3_a.mean())
    p1["spearman_delta_spread_vs_delta_err"] = float(
        np.corrcoef(np.argsort(np.argsort(std3_a - std3_b)),
                    np.argsort(np.argsort(d)))[0, 1])
    # per-node: distribution of shift magnitude (gated nodes in test molecules)
    gidx = np.flatnonzero(gate)
    in_test = np.isin(pool_mol[gidx], list(te))
    shift = np.abs(newA_trust - P3[gidx]).sum(axis=1) / 3
    p1["n_gated_nodes"] = int(len(gidx))
    p1["n_gated_nodes_in_test_mols"] = int(in_test.sum())
    p1["node_shift_quantiles"] = {f"q{q}": float(np.quantile(shift[in_test], q))
                                  for q in [0.25, 0.5, 0.75, 0.9, 0.99]}
    p1["node_shift_max"] = float(shift[in_test].max())
    # regression-to-mean probe: node shift vs pre-error proxy (|P_i - mol mean|)
    gmol = pool_mol[gidx]
    mol_mean = pd.Series(P3.mean(axis=1)).groupby(pd.Series(pool_mol)).transform("mean").to_numpy()[gidx]
    pre_out = np.abs(P3[gidx].mean(axis=1) - mol_mean)
    p1["spearman_node_shift_vs_pre_outlierness"] = float(
        np.corrcoef(np.argsort(np.argsort(shift[in_test])),
                    np.argsort(np.argsort(pre_out[in_test])))[0, 1])
    # error decomposition: how much of the per-molecule error change is
    # explained by pull direction (toward truth vs away)
    pull = np.array([pmeanA_t_by_mid[m] for m in union_mols]) - tdf.loc[union_mols, "mean3"].to_numpy()
    toward = (np.sign(pull) == -np.sign(tdf.loc[union_mols, "mean3"].to_numpy() - truth[ui]))
    p1["n_mols_pulled_toward_truth"] = int(toward.sum())
    p1["n_mols_pulled_away"] = int((~toward).sum())

    # ---------------- Part 4: pool composition (transductive scope) ------------
    g_test = gidx[in_test]
    node_split = pd.Series(pool_mol).map(dict(zip(all_ids, split_of))).to_numpy()
    from_test, from_val, from_train = 0, 0, 0
    total_nbr = 0
    for gi in g_test:
        for nidx in nbrs[int(gi)]:
            sp = node_split[nidx]
            from_train += sp == 0
            from_val += sp == 1
            from_test += sp == 2
            total_nbr += 1
    p4 = {"neighbor_split_of_test_gated_nodes": {
            "train": float(from_train / total_nbr), "val": float(from_val / total_nbr),
            "test": float(from_test / total_nbr), "n_neighbors_total": total_nbr},
          "scope_note": ("test nodes borrow from OTHER TEST nodes: method scores a "
                         "BATCH of new molecules together, not one at a time")}

    # ---------------- Part 5: seeds 7/2024 node-level diagnosis ----------------
    p5 = {"per_seed_node_distribution": {}, "per_seed_mol_error": {},
          "catastrophic_node_overlap": {}, "training_facts": SEED_TRAIN_FACTS}
    for s in SEEDS5:
        Ps = nodes[f"P_seed{s}"].to_numpy()
        p5["per_seed_node_distribution"][str(s)] = {
            "mean": float(Ps.mean()), "std": float(Ps.std()),
            "p1": float(np.quantile(Ps, 0.01)), "p99": float(np.quantile(Ps, 0.99)),
            "max_abs": float(np.abs(Ps).max()),
            "n_abs_gt_5": int((np.abs(Ps) > 5).sum()),
            "n_abs_gt_50": int((np.abs(Ps) > 50).sum()),
        }
        e = np.abs(pred[f"pred_seed{s}"].to_numpy() - pred["true_value"].to_numpy())
        p5["per_seed_mol_error"][str(s)] = {
            "mae": float(e.mean()), "median": float(np.median(e)),
            "p99": float(np.quantile(e, 0.99)), "max": float(e.max()),
            "n_gt_10": int((e > 10).sum()), "n_gt_50": int((e > 50).sum()),
        }
    # catastrophic nodes (|P_seed7| or |P_seed2024| > 50) -> which molecules, overlap
    for s in (7, 2024):
        Ps = nodes[f"P_seed{s}"].to_numpy()
        cat_nodes = np.abs(Ps) > 50
        cat_mols = set(nodes.loc[cat_nodes, "mol_id"])
        p5["catastrophic_node_overlap"][str(s)] = {
            "n_cat_nodes": int(cat_nodes.sum()),
            "n_cat_molecules": len(cat_mols),
            "frac_cat_nodes_in_grad12_mols": float(
                len(cat_mols & pops["gradient12"]) / max(1, len(cat_mols))),
            "n_cat_mols_halogen": int(sum(1 for m in cat_mols if hal[all_ids.index(m)])),
            "isolated6_ids": ["mobley_3359593", "mobley_7150646", "mobley_7690440",
                              "mobley_9913368", "mobley_766666", "mobley_2689721"],
            "n_cat_mols_in_isolated6": int(len(cat_mols & set(["mobley_3359593",
                "mobley_7150646", "mobley_7690440", "mobley_9913368", "mobley_766666",
                "mobley_2689721"]))),
            "example_cat_mols": list(cat_mols)[:10],
        }
    # shape: are catastrophic molecules whole-shift or few-atom spikes?
    p5["cat_shape"] = {}
    for s in (7, 2024):
        Ps = nodes[f"P_seed{s}"].to_numpy()
        cat_mols = sorted(set(nodes.loc[np.abs(Ps) > 50, "mol_id"]))[:6]
        sh = []
        for m in cat_mols:
            pm = Ps[nodes.mol_id == m]
            sh.append({"mol": m, "n_atoms": int(len(pm)),
                       "frac_atoms_abs_gt_50": float((np.abs(pm) > 50).mean()),
                       "sum": float(pm.sum())})
        p5["cat_shape"][str(s)] = sh

    # ---------------- Part 6: gradient-12 mechanism -----------------------------
    p6 = {"per_molecule": [], "summary": {}}
    gmol = pool_mol[gidx]
    for m in sorted(pops["gradient12"]):
        mi = all_ids.index(m)
        n_atoms = int((nodes.mol_id == m).sum())
        local_gate = gidx[gmol == m]
        g12_err_before = float(eb[union_mols.index(m)]) if m in union_mols else float(
            np.abs(tdf.loc[m, "mean3"] - truth[mi]))
        g12_err_after = float(np.abs(pmeanA_t_by_mid[m] - truth[mi]))
        nb_u3s = []
        nb_splits = []
        for gi in local_gate:
            for nidx in nbrs.get(int(gi), []):
                nb_u3s.append(float(u3[nidx]))
                nb_splits.append(int(node_split[nidx]))
        p6["per_molecule"].append({
            "mol": m, "n_atoms": n_atoms, "n_gated_atoms": int(len(local_gate)),
            "err_before": g12_err_before, "err_after": g12_err_after,
            "delta": g12_err_after - g12_err_before,
            "neighbor_n": len(nb_u3s),
            "neighbor_u3_mean": float(np.mean(nb_u3s)) if nb_u3s else None,
            "neighbor_u3_median": float(np.median(nb_u3s)) if nb_u3s else None,
            "neighbor_frac_train": float(np.mean([x == 0 for x in nb_splits])) if nb_splits else None,
        })
    d12 = np.array([r["delta"] for r in p6["per_molecule"]])
    gated12 = [r for r in p6["per_molecule"] if r["n_gated_atoms"] > 0]
    p6["summary"] = {
        "n_worsened_12": int((d12 > 0).sum()), "n_improved_12": int((d12 < 0).sum()),
        "n_with_gated_atoms": len(gated12),
        "delta_mean": float(d12.mean()), "delta_max": float(d12.max()),
        "concentration": {"share_of_total_worsening_from_worst3": float(
            np.sort(np.maximum(d12, 0))[::-1][:3].sum() / max(1e-12, np.maximum(d12, 0).sum()))},
        "pool_u3_median": float(np.median(u3)),
        "neighbor_u3_vs_pool_median": float(np.mean(
            [r["neighbor_u3_median"] for r in gated12 if r["neighbor_u3_median"] is not None])) if gated12 else None,
    }

    # ---------------- Part 3: control spec (exact, code-level) ------------------
    p3 = {
        "naive": ("identical neighbor selection (same top-k, min_sim=0.2, "
                  "same-molecule exclusion); only difference: t_j = 1 instead of "
                  "trust-rank. Isolates the trust-weighting variable."),
        "random": ("NOT random-neighbor selection. Magnitude-matched random-sign "
                   "placebo: same shift magnitude as the trust arm per gated node, "
                   "sign drawn per seed from {+1,-1} (rng seeded 20260815). "
                   "Mode A averages 3 sign draws per node -> near-zero net shift; "
                   "Mode B applies ONE sign draw to the ensemble mean (weaker "
                   "placebo: single draw can land significantly)."),
        "both": ("neighbor pool identical across arms (full cross-molecular pool, "
                 "same gate)."),
    }

    # ---------------- assemble + save -------------------------------------------
    rep = {"label": label, "runtime_s": time.time() - t0,
           "part1_correction": p1, "part3_controls": p3, "part4_scope": p4,
           "part5_seeds": p5, "part6_grad12": p6,
           "bootstrap": rt.to_dict("records"), "multiple_testing": mt,
           "grad12_ids": sorted(pops["gradient12"]),
           "isolated6_ids": p5["catastrophic_node_overlap"]["7"]["isolated6_ids"]}
    tag = "inductive" if args.inductive else "transductive"
    with open(os.path.join(OUT, f"v1_{tag}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    rt.to_csv(os.path.join(OUT, f"v1_bootstrap_{tag}.csv"), index=False)
    p6df = pd.DataFrame(p6["per_molecule"])
    p6df.to_csv(os.path.join(OUT, f"v1_grad12_{tag}.csv"), index=False)
    p1df = pd.DataFrame({"mol_id": union_mols, "err_before": eb, "err_after": ea,
                         "delta": d})
    p1df.to_csv(os.path.join(OUT, "v1_union_per_molecule.csv"), index=False)
    print(f"[save] v1_{tag}.json + CSVs -> {OUT}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
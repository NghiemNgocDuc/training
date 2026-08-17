"""Part 2 (follow-up): gradient-12 mechanism — atom-level mechanism analysis.

Question: for the 12 'confidently wrong' molecules, WHAT does the trust
mechanism produce at their gated atoms, and WHY does it harm them while
calibrated shrinkage (lambda* from shrinkage_calibrated/, = 1.0) is neutral?
Specifically: are their 'similar' neighbors themselves unreliable (high u3),
or reliable but chemically misleading (confidently wrong in a correlated
way)?

For every gated atom of the 12 molecules (and, as the comparison group, of
the Q_std population): report the value each mechanism produces (original
3-seed mean, trust-weighted neighbor combination, pool-mean shrinkage), the
neighbors' u3 (reliability), neighbor similarity, and whether the neighbor
values are offset from the pool mean ALONG the molecule's error direction
(correlated-error index). Ties back to the existing gradient-12 account
(gradient12_investigation/SUMMARY.md: confidently-wrong cluster, not
coverage-driven, training dynamics never logged).

Outputs -> gradient12_mechanism/ (JSON + per-atom + per-molecule CSVs).
Runtime: < 10 min (CPU). Usage: python gradient12_mechanism.py
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
SHRINK_JSON = os.path.join(os.path.dirname(HERE),
                           "shrinkage_calibrated", "shrinkage_calibrated.json")

SEEDS3 = [42, 123, 999]
RNG_SEED = 20260815
GATE_Q = 0.75
K = 10
MIN_SIM = 0.2


def main():
    t0 = time.time()

    shrink = json.load(open(SHRINK_JSON))
    lam_star = float(shrink["lambda_star"])

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

    u3 = nodes["u3"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    gidx = np.flatnonzero(gate)
    trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()
    P3 = np.stack([nodes[f"P_seed{s}"] for s in SEEDS3], axis=1).astype(np.float64)
    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()
    mu_per_seed = P3.mean(axis=0)
    mu_bar = float(P3.mean())
    pool_u3_median = float(np.median(u3))

    # ---------------- neighbor lookup (top-k, same as method) ------------------
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
    print(f"[nbrs] top-k lookup done for {len(gidx)} gated nodes")

    # ---------------- trust refinement (per-seed values at gated nodes) --------
    newA_trust = P3[gidx].copy()
    for gi, nidx in enumerate([nb_top[int(i)] for i in gidx]):
        if len(nidx) == 0:
            continue
        wv = np.array(ws_top[int(gidx[gi])])
        tt = trust[nidx]
        denom = (wv * tt).sum()
        if denom <= 0:
            continue
        for s in range(3):
            newA_trust[gi, s] = (wv * tt * P3[nidx, s]).sum() / denom
    print("[refine] trust-arm per-node values computed")

    # ---------------- per-atom analysis ----------------------------------------
    def atom_rows(mid):
        return np.flatnonzero(pool_mol == mid)

    def local_gate(mid):
        m = atom_rows(mid)
        return m[np.isin(m, gidx)]

    def trust_val_3mean(gi):
        r = int(np.flatnonzero(gidx == gi)[0])
        return float(newA_trust[r].mean())

    def shrink_val_3mean(gi, lam):
        i3 = float(P3[gi].mean())
        return (1.0 - lam) * i3 + lam * mu_bar

    def gated_sum(mid, transform):
        tot = 0.0
        for gi in local_gate(mid):
            tot += transform(gi)
        return tot

    rows_mol, rows_atom = [], []
    for mid in sorted(pops["gradient12"]):
        mi = all_ids.index(mid)
        m3 = float(tdf.loc[mid, "mean3"])
        err_before = abs(m3 - truth[mi])
        lg = local_gate(mid)
        n_ga = len(lg)
        s_orig = gated_sum(mid, lambda gi: float(P3[gi].mean()))
        s_trust = gated_sum(mid, trust_val_3mean)
        s_shrink = gated_sum(mid, lambda gi: shrink_val_3mean(gi, lam_star))
        unchanged = m3 - s_orig
        gap = truth[mi] - unchanged - s_orig          # gated-atoms' contribution error
        dT = s_trust - s_orig
        dS = s_shrink - s_orig
        err_after_trust = abs(unchanged + s_trust - truth[mi])
        err_after_shrink = abs(unchanged + s_shrink - truth[mi])
        rows_mol.append({
            "mol": mid, "n_atoms": int(len(atom_rows(mid))), "n_gated_atoms": n_ga,
            "err_before": err_before, "err_after_trust": err_after_trust,
            "err_after_shrink": err_after_shrink,
            "delta_trust": err_after_trust - err_before,
            "delta_shrink": err_after_shrink - err_before,
            "gated_sum_orig": s_orig, "gated_sum_trust": s_trust,
            "gated_sum_shrink": s_shrink,
            "gated_gap_vs_truth": gap,
            "trust_pull_dT": dT, "shrink_pull_dS": dS,
            "trust_pull_helps": bool(abs(gap - dT) < abs(gap)),
            "shrink_pull_helps": bool(abs(gap - dS) < abs(gap)),
        })
        for gi in lg:
            nbrs = nb_top[int(gi)]
            nb_u3 = np.array([u3[j] for j in nbrs])
            nb_val = np.array([float(P3[j].mean()) for j in nbrs])
            wv = np.array(ws_top[int(gi)])
            tt = trust[nbrs]
            tv = trust_val_3mean(gi)
            rows_atom.append({
                "mol": mid, "atom_idx": int(gi), "P_orig": float(P3[gi].mean()),
                "P_trust": tv, "P_shrink_lambda_star": shrink_val_3mean(gi, lam_star),
                "P_shrink_lambda1": shrink_val_3mean(gi, 1.0),
                "trust_minus_orig": tv - float(P3[gi].mean()),
                "trust_minus_poolmean": tv - mu_bar,
                "orig_minus_poolmean": float(P3[gi].mean()) - mu_bar,
                "n_neighbors": len(nbrs),
                "neighbor_u3_median": float(np.median(nb_u3)) if len(nb_u3) else None,
                "neighbor_frac_u3_lt_pool_median": float((nb_u3 < pool_u3_median).mean()) if len(nb_u3) else None,
                "neighbor_mean_sim": float(np.mean(wv)) if len(wv) else None,
                "neighbor_value_mean": float(nb_val.mean()) if len(nb_val) else None,
                "mol_error_dir": int(np.sign(m3 - truth[mi])),
            })
    df_mol = pd.DataFrame(rows_mol)
    df_atom = pd.DataFrame(rows_atom)

    # ---------------- comparison group: Q_std gated atoms -----------------------
    rows_atom_q = []
    for mid in sorted(pops["Q_std"]):
        mi = all_ids.index(mid)
        m3 = float(tdf.loc[mid, "mean3"])
        for gi in local_gate(mid):
            nbrs = nb_top[int(gi)]
            nb_u3 = np.array([u3[j] for j in nbrs])
            nb_val = np.array([float(P3[j].mean()) for j in nbrs])
            wv = np.array(ws_top[int(gi)])
            tt = trust[nbrs]
            tv = trust_val_3mean(gi)
            rows_atom_q.append({
                "mol": mid, "atom_idx": int(gi), "P_orig": float(P3[gi].mean()),
                "P_trust": tv, "trust_minus_poolmean": tv - mu_bar,
                "orig_minus_poolmean": float(P3[gi].mean()) - mu_bar,
                "neighbor_u3_median": float(np.median(nb_u3)) if len(nb_u3) else None,
                "neighbor_frac_u3_lt_pool_median": float((nb_u3 < pool_u3_median).mean()) if len(nb_u3) else None,
                "neighbor_mean_sim": float(np.mean(wv)) if len(wv) else None,
                "neighbor_value_mean": float(nb_val.mean()) if len(nb_val) else None,
                "mol_error_dir": int(np.sign(m3 - truth[mi])),
            })
    df_atom_q = pd.DataFrame(rows_atom_q)

    # ---------------- correlated-error index ------------------------------------
    # For each gated atom, does the trust replacement deviate from the pool mean
    # ALONG the molecule's error direction? +1 = fully aligned (neighborhood is
    # confidently wrong in the same direction), -1 = opposed (pulls toward truth).
    def alignment(df):
        s = df["trust_minus_poolmean"] * df["mol_error_dir"]
        return float(s.sum() / max(1e-12, df["trust_minus_poolmean"].abs().sum()))

    def orig_alignment(df):
        s = df["orig_minus_poolmean"] * df["mol_error_dir"]
        return float(s.sum() / max(1e-12, df["orig_minus_poolmean"].abs().sum()))

    g12_align = alignment(df_atom)
    q_align = alignment(df_atom_q)
    g12_align_orig = orig_alignment(df_atom)
    q_align_orig = orig_alignment(df_atom_q)

    def pull_helps_count(df_atom):
        n_help = 0
        for mid, grp in df_atom.groupby("mol"):
            err_dir = int(grp["mol_error_dir"].iloc[0])
            dT = grp["P_trust"].sum() - grp["P_orig"].sum()
            if np.sign(dT) == -err_dir:
                n_help += 1
        return n_help

    def n_mol_with_gated(df_atom):
        return len(df_atom.groupby("mol"))

    def max_abs_gated_sum(df_atom):
        return float(max((grp["P_orig"].sum() for _, grp in df_atom.groupby("mol")),
                         key=abs))

    g12_helps = pull_helps_count(df_atom)
    q_helps = pull_helps_count(df_atom_q)

    summary = {
        "gradient12": {
            "n_molecules": len(df_mol), "n_gated_atoms": len(df_atom),
            "n_gated_molecules": int((df_mol["n_gated_atoms"] > 0).sum()),
            "trust_harm_count": int((df_mol["delta_trust"] > 0).sum()),
            "shrink_harm_count": int((df_mol["delta_shrink"] > 0).sum()),
            "mean_delta_trust": float(df_mol["delta_trust"].mean()),
            "mean_delta_shrink": float(df_mol["delta_shrink"].mean()),
            "neighbor_reliability": {
                "median_neighbor_u3": float(df_atom["neighbor_u3_median"].median()),
                "mean_frac_neighbors_u3_lt_pool_median": float(
                    df_atom["neighbor_frac_u3_lt_pool_median"].mean()),
                "pool_u3_median": pool_u3_median,
            },
            "neighbor_similarity": {"mean_sim": float(df_atom["neighbor_mean_sim"].mean())},
            "correlated_error_index_trust_replacement": g12_align,
            "correlated_error_index_original_values": g12_align_orig,
            "mean_trust_minus_orig": float(df_atom["trust_minus_orig"].mean()),
            "mean_abs_trust_minus_orig": float(df_atom["trust_minus_orig"].abs().mean()),
            "mean_abs_shrink_minus_orig": float(
                (df_atom["P_shrink_lambda_star"] - df_atom["P_orig"]).abs().mean()),
            "molecules_where_trust_pull_opposes_gap": f"{g12_helps}/{n_mol_with_gated(df_atom)}",
            "max_abs_gated_sum_orig": max_abs_gated_sum(df_atom),
        },
        "Q_std_comparison": {
            "n_molecules": len(pops["Q_std"]),
            "n_gated_atoms": len(df_atom_q),
            "neighbor_reliability": {
                "median_neighbor_u3": float(df_atom_q["neighbor_u3_median"].median()),
                "mean_frac_neighbors_u3_lt_pool_median": float(
                    df_atom_q["neighbor_frac_u3_lt_pool_median"].mean()),
            },
            "neighbor_similarity": {"mean_sim": float(df_atom_q["neighbor_mean_sim"].mean())},
            "correlated_error_index_trust_replacement": q_align,
            "correlated_error_index_original_values": q_align_orig,
            "molecules_where_trust_pull_opposes_gap": f"{q_helps}/{n_mol_with_gated(df_atom_q)}",
            "max_abs_gated_sum_orig": max_abs_gated_sum(df_atom_q),
        },
        "design": {
            "lambda_star": lam_star,
            "mechanisms": {
                "trust": "P'_i = sum_j w_ij*t_j*P_j / sum_j w_ij*t_j (top-10 similar "
                         "neighbors, w=sim, t=trust-rank)",
                "shrink": "P''_i = (1-lambda*)*P_i + lambda**mu_bar with lambda*="
                          f"{lam_star} (val-calibrated), mu_bar = {mu_bar:.4f}",
            },
            "correlated_error_index": ("sum((P'_i - mu_bar)*sign(err_mol)) / sum|P'_i - mu_bar| "
                                       "over gated atoms; +1 = neighbor values offset from pool "
                                       "mean ALONG the molecule's error (confidently wrong in a "
                                       "correlated way), -1 = offset opposed to the error"),
            "tie_back": ("existing account (gradient12_investigation/SUMMARY.md): confidently-"
                         "wrong cluster, NOT Tanimoto-isolated, NOT GMM-density outliers, NOT "
                         "explained by experimental uncertainty (labels lack error bars); "
                         "training dynamics never logged (untestable)."),
        },
    }

    rep = {"label": "Part 2 follow-up: gradient-12 mechanism (atom-level)",
           "runtime_s": time.time() - t0, "summary": summary,
           "per_molecule": rows_mol, "per_atom_gradient12": rows_atom,
           "per_atom_Q_std": rows_atom_q}
    with open(os.path.join(OUT, "gradient12_mechanism.json"), "w") as f:
        json.dump(rep, f, indent=2)
    df_mol.to_csv(os.path.join(OUT, "gradient12_molecules.csv"), index=False)
    df_atom.to_csv(os.path.join(OUT, "gradient12_atoms.csv"), index=False)
    df_atom_q.to_csv(os.path.join(OUT, "q_std_atoms_comparison.csv"), index=False)
    print(f"[save] gradient12_mechanism.json + CSVs -> {OUT}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
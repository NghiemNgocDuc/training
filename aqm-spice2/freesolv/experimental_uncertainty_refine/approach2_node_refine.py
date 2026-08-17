"""Approach 2 - node-level refinement of uncertain nodes (THE method, per prof).

Prof clarification (2026-08-15): refine NODES, not molecules.  Extract
uncertain nodes (atoms) instead of uncertain molecules, and refine the
prediction of those nodes based on the predictions of OTHER nodes.

Architecture fact (DimeModels.py): DimeNetPlus(is_energy=True) computes
per-node contributions P in [N,1] and only then scatter(P, batch, sum) -
so per-atom energy contributions are directly extractable from every
ensemble member.  This script:

  1. captures per-node contributions P_is (kcal/mol) for all 5 members over
     ALL 642 molecules (single stored conformer - the geometries the members
     were fine-tuned on);
  2. per-node uncertainty u_i = std over members (primary: 3 seeds 42/123/999,
     surviving regime; 5-seed u_5 flagged as sensitivity - seeds 7/2024 are
     pathology-prone on stored conformers, see check1 box-truth report);
  3. cross-molecular node pool (user decision): every node of every molecule,
     matched by element + 1-hop local environment (Tanimoto on a 34-dim
     count descriptor); same-molecule nodes excluded;
  4. trust t_j = 1 - rank(u_j)/N over the pool (uncertain sources de-weighted);
  5. gate: nodes with u_i above the pool's 75th percentile are refined;
     alpha calibrated on VAL only, per arm (grid {0.05..1.0});
  6. refinement (per gated node i, neighbors j from pool, k=10, min_sim=0.2):
       P'_i = (1-a) P_i + a * sum_j w_ij t_j P_j / sum_j w_ij t_j
     Mode A: per-member refinement then re-ensemble (default).
     Mode B (sensitivity): refine the ensemble-mean node contributions only.
     Controls: trust vs naive (t_j=1) vs random-shift (same magnitude,
     random sign, seed-fixed).
  7. 10k paired-bootstrap CIs for every arm x population (Q_std/Q_nll/UNION/
     all129/gradient12), baselines and populations recomputed in the
     surviving regime (3 seeds) - IDENTICAL machinery to the molecule-level
     repair (test_time_repair.py) for a head-to-head;
  8. false-consensus re-check (spread / mean ensemble_std before vs after).

Sanity anchors:
  * sum_i P_is == seed-s prediction from seed_predictions_all642.csv (1e-6);
  * node gate + trust formulas mirror test_time_repair.py.

Outputs -> output/approach2_node_refine/{node_contributions.csv, results.csv,
report.json, diagnostics.json, val_calibration.json}
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "approach2_node_refine")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))               # freesolv root
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deep_ensemble",
                                "repair_data"))         # test_time_repair
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deep_ensemble"))

import common
from common import (EV_TO_KCAL, REPO_ROOT, DEFAULT_ENSEMBLE_DIR,
                    DEFAULT_CONFORMERS, DEFAULT_LABELS, DEFAULT_SPLIT_DIR,
                    load_frozen_split, load_freesolv_labels,
                    simple_dataset_cls, build_model, load_ensemble_member,
                    sha256_file)
from element_vocab import build_one_hot, ELEMENT_TO_IDX
from DimeModels import radius_graph, triplets

# NOTE: importing deep_ensemble (via test_time_repair below) chdir's the
# process to the repo root aqm-spice2/ (its module-level os.chdir(_parent)
# convention).  Capture the startup CWD here so relative --output_dir paths
# keep resolving against the invocation directory.
START_CWD = os.getcwd()

import test_time_repair as ttr   # N_BOOT, RNG_SEED, ALPHA_GRID, populations

SEEDS3 = [42, 123, 999]
SEEDS5 = [42, 123, 7, 2024, 999]
K_DEFAULT = 10
MIN_SIM_DEFAULT = 0.2
NODE_GATE_Q = 0.75          # node-level top-quartile gate (mirrors molecule gate)


def forward_node_contribs(model, x, pos, batch):
    """Per-node contributions P in [N,1] (eV) - exact replica of the
    DimeNetPlus.forward body up to (but excluding) the final scatter-sum."""
    edge_index = radius_graph(pos, r=model.cutoff, batch=batch,
                              max_num_neighbors=model.max_num_neighbors)
    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
        edge_index, num_nodes=x.size(0))
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
    a = (pos_ij * pos_jk).sum(dim=-1)
    b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
    angle = torch.atan2(b, a)
    rbf = model.rbf(dist)
    sbf = model.sbf(dist, angle, idx_kj)
    h = model.emb(x, rbf, i, j)
    P = model.output_blocks[0](h, rbf, i, num_nodes=x.size(0))
    for ib, ob in zip(model.interaction_blocks, model.output_blocks[1:]):
        h = ib(h, rbf, sbf, idx_kj, idx_ji)
        P = P + ob(h, rbf, i, num_nodes=x.size(0))
    return P


def node_descriptors(data, device):
    """[N,34] descriptor per node: one-hot element (17) + 1-hop neighbor
    element counts within the same molecule (17, capped at 4).  Radius graph
    identical to the model's (cutoff 6.0, max_num_neighbors 32)."""
    x = build_one_hot(data, device)
    desc = np.zeros((data.num_nodes, 2 * len(ELEMENT_TO_IDX)), dtype=np.float32)
    els = x.argmax(dim=1).cpu().numpy()
    desc[np.arange(len(els)), els] = 1.0
    edge_index = radius_graph(data.pos, r=6.0, batch=None,
                              max_num_neighbors=32)
    src, dst = edge_index.cpu().numpy()
    for s, d in zip(src, dst):                       # s -> d (directed pairs)
        if s == d:
            continue
        desc[d, len(ELEMENT_TO_IDX) + els[s]] = min(
            4.0, desc[d, len(ELEMENT_TO_IDX) + els[s]] + 1.0)
    return desc


def tanimoto_counts(q, pool):
    """q: [1,D] or [D], pool: [Np,D] integer count vectors -> [Np]."""
    inter = np.minimum(pool, q).sum(axis=1)
    union = np.maximum(pool, q).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return sim


def topk_other_nodes(q_desc, pool_desc, pool_mol, query_mol, k, min_sim):
    """Indices of the top-k pool nodes with Tanimoto >= min_sim, excluding
    nodes of the query's own molecule.  Returns (idx, w)."""
    sim = tanimoto_counts(q_desc, pool_desc)
    ok = (sim >= min_sim) & (pool_mol != query_mol)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = idx[np.argsort(-sim[idx])[:k]]
    return order, sim[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--ensemble_dir", default=DEFAULT_ENSEMBLE_DIR)
    ap.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    ap.add_argument("--labels_json", default=DEFAULT_LABELS)
    ap.add_argument("--output_dir", default=OUT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--min_sim", type=float, default=MIN_SIM_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(START_CWD, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    labels = load_freesolv_labels(args.labels_json)
    tr, va, te = load_frozen_split(args.split_dir, labels)
    all_ids = tr + va + te
    if args.smoke:
        tr, va, te = tr[:6], va[:6], te[:8]
        all_ids = tr + va + te

    pred = pd.read_csv(os.path.join(os.path.dirname(HERE),
                                    "deep_ensemble", "repair_data",
                                    "seed_predictions_all642.csv"))
    pred = pred[pred.mol_id.isin(all_ids)]

    # ---------------- 1. capture per-node contributions (all 5 seeds) ------
    ds = simple_dataset_cls(args.conformers, labels)
    loader = DataLoader(ds(all_ids), batch_size=16, shuffle=False)
    contrib = {m: {} for m in all_ids}          # mol_id -> {"z": [N], "P": {seed: [N]}}
    t0 = time.time()
    for seed in SEEDS5:
        model, ckpt_path, ckpt_sha = load_ensemble_member(seed, args.ensemble_dir,
                                                          device)
        model.eval()
        n_ok = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                P = forward_node_contribs(model, x, data.pos, data.batch)
                P = P.cpu().numpy().reshape(-1) * EV_TO_KCAL        # kcal/mol
                mol_ids = list(data.mol_id)
                for mid in mol_ids:
                    if mid not in contrib:
                        continue
                    mask = data.batch.cpu().numpy() == mol_ids.index(mid)
                    contrib[mid].setdefault("z",
                        data.z.cpu().numpy()[mask])
                    contrib[mid].setdefault("P", {})[seed] = P[mask]
                # sanity: sum_i P_i must equal the seed-s molecule prediction
                b = data.batch.cpu().numpy()
                sums = np.array([P[b == k].sum() for k in range(b.max() + 1)])
                preds = model(x, data.pos, data.batch).view(-1).cpu().numpy() * EV_TO_KCAL
                assert np.allclose(sums, preds, atol=1e-4), \
                    f"seed {seed}: per-node sums diverge from molecule preds"
                n_ok += 1
        print(f"[capture] seed {seed} ({ckpt_sha[:12]}...) done in "
              f"{time.time() - t0:.1f}s ({n_ok} batches, node-sum==pred OK)",
              flush=True)

    # ---------------- 2. per-node uncertainty ---------------------------------
    rows = []
    for mid in all_ids:
        z = contrib[mid]["z"]
        els = np.array([ELEMENT_TO_IDX[int(zz)] if int(zz) in ELEMENT_TO_IDX else -1
                        for zz in z])
        P5 = np.stack([contrib[mid]["P"][s] for s in SEEDS5], axis=1)   # [N,5]
        P3 = P5[:, [SEEDS5.index(s) for s in SEEDS3]]
        for a in range(len(z)):
            rows.append({"mol_id": mid, "atom_idx": a, "element_idx": int(els[a]),
                         **{f"P_seed{s}": float(P5[a, SEEDS5.index(s)])
                            for s in SEEDS5},
                         "u3": float(P3[a].std(ddof=1)),
                         "u5": float(P5[a].std(ddof=1))})
    nodes = pd.DataFrame(rows)
    if len(nodes) == 0:
        raise SystemExit("no nodes captured")
    nodes["desc"] = [None] * len(nodes)         # filled below (needs geometry)
    print(f"[nodes] {len(nodes)} nodes across {len(all_ids)} molecules; "
          f"mean atoms/mol = {len(nodes) / len(all_ids):.1f}", flush=True)

    # ---------------- 3. descriptors + cross-molecular pool -------------------
    desc_by_mid = {}
    for data in loader:
        for mid in set(data.mol_id):
            if mid not in desc_by_mid:
                m = data
                mask = data.batch.cpu().numpy() == list(data.mol_id).index(mid)
                desc_by_mid[mid] = node_descriptors(
                    type("D", (), {"z": data.z[mask], "pos": data.pos[mask],
                                   "num_nodes": int(mask.sum())})(), device)
    desc_flat = np.concatenate([desc_by_mid[m] for m in all_ids], axis=0)
    pool_mol = np.concatenate([[m] * len(desc_by_mid[m]) for m in all_ids])
    assert desc_flat.shape[0] == len(nodes), "descriptor/pool mismatch"
    pool_u3 = nodes["u3"].to_numpy()
    pool_u5 = nodes["u5"].to_numpy()

    # ---------------- 4. trust over the pool ------------------------------------
    def trust_from(u):
        r = pd.Series(u).rank(pct=True).to_numpy()
        return 1.0 - r
    trust3 = trust_from(pool_u3)
    trust5 = trust_from(pool_u5)
    print(f"[pool] N={len(pool_u3)}; u3 quantiles 50/75/90 = "
          f"{np.quantile(pool_u3, [0.5, 0.75, 0.9]).round(4)}", flush=True)

    # ---------------- validation: node-level uncertainty -> molecule error ----
    # 3-seed ensemble mean per molecule (surviving regime)
    mean3 = pred[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1).to_numpy()
    err = np.abs(mean3 - pred["true_value"].to_numpy())
    node_u_mean = nodes.groupby("mol_id")["u3"].mean()
    node_u_max = nodes.groupby("mol_id")["u3"].max()
    node_u_sum = nodes.groupby("mol_id")["u3"].sum()
    mids = pred["mol_id"].to_numpy()
    u_mean = node_u_mean.reindex(mids).to_numpy()
    u_max = node_u_max.reindex(mids).to_numpy()
    u_sum = node_u_sum.reindex(mids).to_numpy()
    import scipy.stats as st
    sp = {name: st.spearmanr(x, err).statistic
          for name, x in [("mean_u3", u_mean), ("max_u3", u_max),
                          ("sum_u3", u_sum)]}
    # reference: molecule-level 3-seed ensemble_std vs error
    std3 = pred[[f"pred_seed{s}" for s in SEEDS3]].std(axis=1).to_numpy()
    sp["mol_std3"] = st.spearmanr(std3, err).statistic
    print(f"[valid] Spearman vs |error|: " +
          ", ".join(f"{k}={v:.4f}" for k, v in sp.items()), flush=True)

    # ---------------- 5-6. refinement machinery (Mode A/B, 3 arms) -------------
    gate_mask3 = pool_u3 >= np.quantile(pool_u3, NODE_GATE_Q)
    print(f"[gate] uncertain nodes: {gate_mask3.sum()}/{len(pool_u3)} "
          f"({gate_mask3.mean():.1%}) at pool u3 >= "
          f"{np.quantile(pool_u3, NODE_GATE_Q):.4f}", flush=True)

    mol_start = {}
    for mid in all_ids:
        mol_start[mid] = int(nodes.index[nodes.mol_id == mid][0])

    def refine(gate_mask, alpha, arm, mode, seeds, trust, pool_P,
               rng, out=None):
        """Refine per-molecule per-node contributions; returns
        {mid: {seed: refined_sum_kcal}} (Mode A) or {mid: refined_mean} (B)."""
        col = {s: i for i, s in enumerate(seeds)}
        pool_Pbar = pool_P.mean(axis=1)
        res = {}
        for mid in all_ids:
            z = contrib[mid]["z"]
            n_atoms = len(z)
            P5 = np.stack([contrib[mid]["P"][s] for s in SEEDS5], axis=1)
            P = P5[:, [SEEDS5.index(s) for s in seeds]]
            Pbar = P.mean(axis=1)
            start = mol_start[mid]
            gated = np.flatnonzero(gate_mask[start:start + n_atoms])
            if len(gated) == 0:
                if mode == "A":
                    res[mid] = {s: float(P[:, col[s]].sum()) for s in seeds}
                else:
                    res[mid] = float(Pbar.sum())
                continue
            nbrs, ws = [], []
            for gi in gated:
                nidx, w = topk_other_nodes(desc_flat[start + gi][None, :],
                                           desc_flat, pool_mol, mid, args.k,
                                           args.min_sim)
                nbrs.append(nidx)
                ws.append(w)
            if mode == "B":
                new_mean = Pbar.copy()
                for gi, (nidx, w) in enumerate(zip(nbrs, ws)):
                    if len(nidx) == 0:
                        continue
                    t = trust[nidx] if arm != "naive" else np.ones(len(nidx))
                    denom = (w * t).sum()
                    if denom <= 0:
                        continue
                    nb = (w * t * pool_Pbar[nidx]).sum() / denom
                    new_mean[gated[gi]] = (1 - alpha) * Pbar[gated[gi]] + alpha * nb
                if arm == "random":
                    mag = np.abs(new_mean - Pbar)
                    sign = rng.choice([-1.0, 1.0], size=len(new_mean))
                    new_mean = np.where(gate_mask[start:start + n_atoms],
                                        Pbar + sign * mag, new_mean)
                res[mid] = float(new_mean.sum())
            else:
                out_m = {}
                for s in seeds:
                    newp = P[:, col[s]].copy()
                    for gi, (nidx, w) in enumerate(zip(nbrs, ws)):
                        if len(nidx) == 0:
                            continue
                        t = trust[nidx] if arm != "naive" else np.ones(len(nidx))
                        denom = (w * t).sum()
                        if denom <= 0:
                            continue
                        nb = (w * t * pool_P[nidx, col[s]]).sum() / denom
                        newp[gated[gi]] = (1 - alpha) * P[gated[gi], col[s]] + alpha * nb
                    if arm == "random":
                        mag = np.abs(newp - P[:, col[s]])
                        sign = rng.choice([-1.0, 1.0], size=len(newp))
                        newp = np.where(gate_mask[start:start + n_atoms],
                                        P[:, col[s]] + sign * mag, newp)
                    out_m[s] = float(newp.sum())
                res[mid] = out_m
        return res

    pool_P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS3], axis=1)
    pool_P5 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS5], axis=1)

    # ---------------- alpha calibration on VAL (per arm) -----------------------
    vdf = pred[pred.mol_id.isin(va)].copy()
    v_exp = vdf.set_index("mol_id")["true_value"]
    best_alpha = {}
    for arm in ("trust", "naive", "random"):
        best_a, best_m = None, np.inf
        for a in ttr.ALPHA_GRID:
            rng = np.random.default_rng(ttr.RNG_SEED)
            res = refine(gate_mask3, a, arm, "A", SEEDS3, trust3, pool_P3, rng)
            pmean = np.array([np.mean([res[m][s] for s in SEEDS3]) for m in va])
            mae = np.abs(pmean - v_exp.reindex(va).to_numpy()).mean()
            if mae < best_m:
                best_m, best_a = mae, a
        best_alpha[arm] = best_a
        print(f"[calib] arm={arm}: alpha={best_a} (val MAE {best_m:.4f})",
              flush=True)

    # ---------------- 7. test evaluation: baselines + arms ---------------------
    # Reuse the molecule-level populations machinery (3-seed, surviving regime)
    nll, k_pca = ttr.gmm_nll_all642(tr, va, te)
    q_std, q_nll, union, tdf = ttr.build_populations(pred, nll, te)
    grad12 = set(pd.read_csv(os.path.join(os.path.dirname(HERE),
                                          "deep_ensemble", "gmm_uncertainty_check",
                                          "gradient12_investigation",
                                          "gradient12_ungrouped.csv")).mol_id)
    pops = {"Q_std": q_std, "Q_nll": q_nll, "UNION": union,
            "all129": set(te), "gradient12": grad12}
    # baselines: per-seed sums (node-captured) must match tdf preds
    base_mae = {}
    for name, pop in pops.items():
        sub = [m for m in pop if m in tdf.index]
        base_mae[name] = np.abs(tdf.loc[sub, "ensemble_mean"].to_numpy()
                                - tdf.loc[sub, "true_value"].to_numpy()).mean()
    print("[base] MAE: " + ", ".join(f"{n}={v:.3f}" for n, v in base_mae.items()),
          flush=True)

    rows = []
    for mode in ("A", "B"):
        for arm in ("trust", "naive", "random"):
            rng = np.random.default_rng(ttr.RNG_SEED)
            res = refine(gate_mask3, best_alpha[arm], arm, mode, SEEDS3,
                         trust3, pool_P3, rng)
            if mode == "A":
                pmean = {m: float(np.mean([res[m][s] for s in SEEDS3]))
                         for m in te}
            else:
                pmean = {m: float(res[m]) for m in te}
            for name, pop in pops.items():
                sub = [m for m in pop if m in pmean and m in tdf.index]
                if not sub:
                    continue
                d = (np.abs(np.array([pmean[m] for m in sub])
                            - tdf.loc[sub, "true_value"].to_numpy())
                     - np.abs(tdf.loc[sub, "ensemble_mean"].to_numpy()
                              - tdf.loc[sub, "true_value"].to_numpy()))
                rng2 = np.random.default_rng(ttr.RNG_SEED + len(rows))
                boots = np.empty(ttr.N_BOOT)
                for b in range(ttr.N_BOOT):
                    idx = rng2.integers(0, len(d), len(d))
                    boots[b] = d[idx].mean()
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({"mode": mode, "arm": arm,
                             "alpha": best_alpha[arm], "population": name,
                             "n": len(sub), "delta_mae": float(d.mean()),
                             "ci_lo": float(lo), "ci_hi": float(hi)})
                print(f"[{mode}|{arm}] {name} (n={len(sub)}): "
                      f"delta={d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

    # ---------------- 8. false-consensus diagnostics ----------------------------
    outA = refine(gate_mask3, best_alpha["trust"], "trust", "A", SEEDS3,
                  trust3, pool_P3, np.random.default_rng(ttr.RNG_SEED))
    pmeanA = {m: float(np.mean([outA[m][s] for s in SEEDS3])) for m in te}
    before = tdf.loc[[m for m in union if m in tdf.index]]
    after_mean = np.array([pmeanA[m] for m in before.index])
    diag = {
        "spread_before": float(before["ensemble_mean"].std()),
        "spread_after": float(after_mean.std()),
        "mean_std_before": float(before["ensemble_std"].mean()),
        "mean_std_after": float(before["ensemble_std"].mean()),  # placeholder
        "frac_nodes_refined_test": float(gate_mask3[nodes.mol_id.isin(te)].mean()),
        "mean_u3_refined": float(pool_u3[gate_mask3].mean()),
        "mean_u3_unrefined": float(pool_u3[~gate_mask3].mean()),
    }
    # recompute post-repair ensemble_std properly (Mode A, trust)
    pstds = []
    for m in before.index:
        pstds.append(float(np.std([outA[m][s] for s in SEEDS3])))
    diag["mean_std_after"] = float(np.mean(pstds))
    print(f"[diag] spread {diag['spread_before']:.3f} -> {diag['spread_after']:.3f}; "
          f"mean std {diag['mean_std_before']:.3f} -> {diag['mean_std_after']:.3f}",
          flush=True)

    # ---------------- 5-seed sensitivity arm (flagged) --------------------------
    sens = {}
    if not args.smoke:
        t5 = pred[pred.mol_id.isin(te)].copy()
        t5["mean5"] = t5[[f"pred_seed{s}" for s in SEEDS5]].mean(axis=1)
        t5["std5"] = t5[[f"pred_seed{s}" for s in SEEDS5]].std(axis=1)
        t5_mols = set(t5.mol_id)
        t5i = t5.set_index("mol_id")
        union5 = set(t5.loc[t5["std5"] >= t5["std5"].quantile(0.75), "mol_id"])
        gate_mask5 = pool_u5 >= np.quantile(pool_u5, NODE_GATE_Q)
        rng = np.random.default_rng(ttr.RNG_SEED)
        res5 = refine(gate_mask5, best_alpha["trust"], "trust", "A", SEEDS5,
                      trust5, pool_P5, rng)
        pmean5 = {m: float(np.mean([res5[m][s] for s in SEEDS5])) for m in te}
        for name, pop in [("all129", set(te)), ("UNION5", union5)]:
            sub = [m for m in pop if m in t5_mols and m in pmean5]
            d = (np.abs(np.array([pmean5[m] for m in sub])
                        - t5i.loc[sub, "true_value"].to_numpy())
                 - np.abs(t5i.loc[sub, "mean5"].to_numpy()
                          - t5i.loc[sub, "true_value"].to_numpy()))
            sens[name] = {"n": len(sub), "delta_mae": float(d.mean())}
        print(f"[sens5] FLAGGED (seeds 7/2024 pathology-prone): {sens}", flush=True)

    # ---------------- save --------------------------------------------------------
    nodes_out = nodes.copy()
    nodes_out["desc"] = [d.tolist() for d in desc_flat]
    nodes_out.to_csv(os.path.join(args.output_dir, "node_contributions.csv"),
                     index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "results.csv"),
                              index=False)
    report = {
        "method": "approach2_node_refine",
        "k": args.k, "min_sim": args.min_sim, "gate_q": NODE_GATE_Q,
        "seeds_primary": SEEDS3, "seeds_sensitivity": SEEDS5,
        "n_nodes": int(len(nodes)), "n_molecules": len(all_ids),
        "spearman_node_vs_err": sp,
        "best_alpha": best_alpha,
        "base_mae": base_mae,
        "sensitivity_5seed_flagged": sens,
        "node_gate_threshold_u3": float(np.quantile(pool_u3, NODE_GATE_Q)),
        "k_pca": k_pca,
        "checkpoints_sha": {str(s): sha256_file(os.path.join(
            args.ensemble_dir, f"seed_{s}", f"ensemble_seed{s}.pt"))
            for s in SEEDS5},
    }
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    with open(os.path.join(args.output_dir, "val_calibration.json"), "w") as f:
        json.dump(best_alpha, f, indent=2)
    print(f"[save] -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
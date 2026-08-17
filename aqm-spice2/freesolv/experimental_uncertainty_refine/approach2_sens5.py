"""5-seed sensitivity arm for approach2_node_refine (FLAGGED).

Seeds 7/2024 are pathology-prone on stored conformers (see check1 box-truth
report): the per-seed predictions contain catastrophic outliers.  This arm
repeats the node-level trust refinement with all 5 seeds and reports the
delta vs the 5-seed ensemble mean on all129 and a 5-seed top-quartile-union.
Results are for sensitivity only - the paper's primary analysis stays on
seeds 42/123/999 (surviving regime).

Reuses output/approach2_node_refine/node_contributions.csv (per-node P and
u3/u5); recomputes node descriptors from the stored conformers (no model
forward - fast).

Outputs -> output/approach2_node_refine/sens5.json
"""

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
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deep_ensemble",
                                "repair_data"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deep_ensemble"))

import common
from common import (EV_TO_KCAL, DEFAULT_ENSEMBLE_DIR, DEFAULT_CONFORMERS,
                    DEFAULT_LABELS, DEFAULT_SPLIT_DIR, load_frozen_split,
                    load_freesolv_labels, simple_dataset_cls)
from element_vocab import build_one_hot, ELEMENT_TO_IDX
from DimeModels import radius_graph
import approach2_node_refine as a2

SEEDS5 = a2.SEEDS5
NODE_GATE_Q = a2.NODE_GATE_Q


def main():
    nodes = pd.read_csv(os.path.join(OUT, "node_contributions.csv"))
    with open(os.path.join(OUT, "val_calibration.json")) as f:
        alpha = json.load(f)["trust"]
    labels = load_freesolv_labels(DEFAULT_LABELS)
    tr, va, te = load_frozen_split(DEFAULT_SPLIT_DIR, labels)
    all_ids = tr + va + te
    pred = pd.read_csv(os.path.join(os.path.dirname(HERE), "deep_ensemble",
                                    "repair_data",
                                    "seed_predictions_all642.csv"))

    device = torch.device("cpu")
    ds = simple_dataset_cls(DEFAULT_CONFORMERS, labels)
    loader = DataLoader(ds(all_ids), batch_size=16, shuffle=False)

    desc_by_mid = {}
    for data in loader:
        for mid in set(data.mol_id):
            if mid not in desc_by_mid:
                mask = data.batch.cpu().numpy() == list(data.mol_id).index(mid)
                desc_by_mid[mid] = a2.node_descriptors(
                    type("D", (), {"z": data.z[mask], "pos": data.pos[mask],
                                   "num_nodes": int(mask.sum())})(), device)
    desc_flat = np.concatenate([desc_by_mid[m] for m in all_ids], axis=0)
    pool_mol = np.concatenate([[m] * len(desc_by_mid[m]) for m in all_ids])
    assert desc_flat.shape[0] == len(nodes)

    pool_u5 = nodes["u5"].to_numpy()
    pool_P5 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS5], axis=1)
    trust5 = 1.0 - pd.Series(pool_u5).rank(pct=True).to_numpy()
    gate5 = pool_u5 >= np.quantile(pool_u5, NODE_GATE_Q)
    print(f"[sens5] gate: {gate5.sum()}/{len(pool_u5)} nodes "
          f"(u5 >= {np.quantile(pool_u5, NODE_GATE_Q):.4f}); "
          f"alpha={alpha} (val-calibrated trust arm)", flush=True)

    # per-molecule P matrices (5 seeds)
    contrib = {m: {} for m in all_ids}
    start_row = {}
    idx = 0
    for mid in all_ids:
        n = int((nodes.mol_id == mid).sum())
        start_row[mid] = idx
        idx += n
        P5 = nodes.loc[nodes.mol_id == mid,
                       [f"P_seed{s}" for s in SEEDS5]].to_numpy()
        contrib[mid] = {"n": n, "P5": P5}

    t5 = pred[pred.mol_id.isin(te)].copy()
    t5["mean5"] = t5[[f"pred_seed{s}" for s in SEEDS5]].mean(axis=1)
    t5["std5"] = t5[[f"pred_seed{s}" for s in SEEDS5]].std(axis=1)
    t5_mols = set(t5.mol_id)
    t5i = t5.set_index("mol_id")
    union5 = set(t5.loc[t5["std5"] >= t5["std5"].quantile(0.75), "mol_id"])

    res = {}
    for mid in all_ids:
        n, P5 = contrib[mid]["n"], contrib[mid]["P5"]
        g = gate5[start_row[mid]:start_row[mid] + n]
        if not g.any():
            res[mid] = P5.sum(axis=0)
            continue
        newp = P5.copy()
        for gi in np.flatnonzero(g):
            nidx, w = a2.topk_other_nodes(desc_flat[start_row[mid] + gi][None, :],
                                          desc_flat, pool_mol, mid, a2.K_DEFAULT,
                                          a2.MIN_SIM_DEFAULT)
            if len(nidx) == 0:
                continue
            t = trust5[nidx]
            denom = (w * t).sum()
            if denom <= 0:
                continue
            nb = ((w * t)[:, None] * pool_P5[nidx]).sum(axis=0) / denom
            newp[gi] = (1 - alpha) * P5[gi] + alpha * nb
        res[mid] = newp.sum(axis=0)

    pmean5 = {m: float(np.mean(res[m])) for m in te}
    out = {"alpha": alpha, "n_gated": int(gate5.sum()),
           "threshold_u5": float(np.quantile(pool_u5, NODE_GATE_Q))}
    for name, pop in [("all129", set(te)), ("UNION5", union5)]:
        sub = [m for m in pop if m in t5_mols and m in pmean5]
        d = (np.abs(np.array([pmean5[m] for m in sub])
                    - t5i.loc[sub, "true_value"].to_numpy())
             - np.abs(t5i.loc[sub, "mean5"].to_numpy()
                      - t5i.loc[sub, "true_value"].to_numpy()))
        out[name] = {"n": len(sub), "delta_mae": float(d.mean())}
        print(f"[sens5] {name} (n={len(sub)}): delta={d.mean():+.3f}", flush=True)

    with open(os.path.join(OUT, "sens5.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sens5] saved -> {os.path.join(OUT, 'sens5.json')}", flush=True)


if __name__ == "__main__":
    main()
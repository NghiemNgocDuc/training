"""Cross-fold isolation check: per fold, how many test molecules are structurally
isolated from that fold's training pool (Tanimoto best_sim <= 0.22)?

This is a lightweight analysis-only script (no training). It answers: would pooling
across folds meaningfully increase n for the neighbor-regularization v2 study of
isolated molecules?

Outputs -> neighbor_isolation_check/crossfold_isolation/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                    # rmse_analysis/
REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
LABELS = json.load(open(os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json"), "r"))

# Reuse fingerprinting from graph.py (same Morgan r=2, 2048-bit Tanimoto protocol)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "neighbor_regularization"))
from graph import parse_smiles, morgan_fp
from rdkit.Chem import DataStructs

ALL_MIDS = [m for m, d in LABELS.items() if isinstance(d.get("expt"), (int, float))]
ALL_FPS = {m: morgan_fp(parse_smiles(LABELS[m]["smiles"])) for m in ALL_MIDS}
print(f"Computed fingerprints for {len(ALL_FPS)} molecules")

def best_sim_vs_pool(query_fp, pool_mids):
    worst = -1.0
    for t in pool_mids:
        s = DataStructs.TanimotoSimilarity(query_fp, ALL_FPS[t])
        if s > worst:
            worst = s
    return worst

def load_fold_split(fold_dir):
    return [json.load(open(os.path.join(fold_dir, f"{n}_ids.json")))
            for n in ("train", "val", "test")]

CV_DIR = os.path.join(REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full")
OUT = os.path.join(HERE, "neighbor_isolation_check", "crossfold_isolation")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd

rows = []
fold_stats = {}
for fi in range(5):
    train, val, test = load_fold_split(os.path.join(CV_DIR, f"fold_{fi}"))
    train_set = set(train)
    print(f"\nFold {fi}: {len(train)} train, {len(val)} val, {len(test)} test")
    iso_lt_022 = iso_lt_030 = 0
    for mid in test:
        bs = best_sim_vs_pool(ALL_FPS[mid], train_set)
        rows.append({"fold": fi, "mol_id": mid, "train_pool_size": len(train_set),
                     "best_sim_vs_train": round(bs, 4),
                     "isolated_lt_022": int(bs <= 0.22),
                     "isolated_lt_030": int(bs <= 0.30)})
        iso_lt_022 += bs <= 0.22
        iso_lt_030 += bs <= 0.30
    print(f"  isolated vs train: best_sim <= 0.22: {iso_lt_022}/{len(test)}   <= 0.30: {iso_lt_030}/{len(test)}")
    fold_stats[f"fold_{fi}"] = {"n_test": len(test), "n_isolated_lt_022": int(iso_lt_022),
                                "n_isolated_lt_030": int(iso_lt_030)}

# Structural isolation from the existing graph cache (k=5, min_sim=0.1, 642 universe)
NBR_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "neighbor_regularization")
graph = json.load(open(os.path.join(NBR_DIR, "graph_cache", "graph_k5_sim0.1.json")))
graph_best_sim = {mid: (max(w for w, _ in nbrs) if nbrs else 0.0)
                  for mid, nbrs in graph.items()}
for row in rows:
    row["best_sim_vs_642_universe"] = round(graph_best_sim.get(row["mol_id"], 0.0), 4)
    row["structurally_isolated_lt_022"] = int(row["best_sim_vs_642_universe"] <= 0.22)
    row["structurally_isolated_lt_030"] = int(row["best_sim_vs_642_universe"] <= 0.30)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "crossfold_isolation.csv"), index=False)

summary = {
    "fold_config": fold_stats,
    "pooled_across_5_folds": {
        "n_total_test": int(len(df)),
        "n_isolated_vs_train_lt_022": int(df["isolated_lt_022"].sum()),
        "n_isolated_vs_train_lt_030": int(df["isolated_lt_030"].sum()),
        "n_structurally_isolated_lt_022": int(df["structurally_isolated_lt_022"].sum()),
        "n_structurally_isolated_lt_030": int(df["structurally_isolated_lt_030"].sum()),
    },
    "isolated_molecules_lt_022": df[df["isolated_lt_022"] == 1][["fold", "mol_id", "best_sim_vs_train"]].to_dict("records"),
    "structurally_isolated_lt_022": df[df["structurally_isolated_lt_022"] == 1][["fold", "mol_id", "best_sim_vs_642_universe"]].to_dict("records"),
}
json.dump(summary, open(os.path.join(OUT, "crossfold_summary.json"), "w"), indent=2)
print(f"\nPooled across 5 folds: {summary['pooled_across_5_folds']}")
print(f"Saved to {OUT}/")
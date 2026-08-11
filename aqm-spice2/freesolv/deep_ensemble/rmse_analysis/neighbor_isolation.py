"""Neighbor-isolation check: are the 18 confidently-wrong molecules chemically
isolated from the 'certain' low_std_low_rmse molecules?

For each of the 18 low_std_high_rmse molecules: k=5 nearest neighbors by Morgan
(r=2, 2048-bit) Tanimoto among the 47 low_std_low_rmse molecules.
Control: same k=5 procedure for the same 18 seed-42 controls used in
stage2_bias_check (neighbors looked up among the 47 pool, self excluded).
Outputs -> neighbor_isolation_check/
"""

import sys
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_CSV = os.path.join(HERE, "output", "per_molecule_rmse.csv")
LABELS_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE)))), "Data", "FreeSolv", "database.json")
STAGE2_CSV = os.path.join(HERE, "stage2_bias_check", "stage2_predictions.csv")
OUT = os.path.join(HERE, "neighbor_isolation_check")
K = 5

# Reuse the fingerprinting implementation from neighbor_regularization/graph.py
# (same protocol: Morgan r=2, 2048 bits) so both analyses share one code path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "neighbor_regularization"))
from graph import parse_smiles, morgan_fp

df = pd.read_csv(ANALYSIS_CSV)
labels = json.load(open(LABELS_JSON, "r"))

def fp(smiles):
    return morgan_fp(parse_smiles(smiles))

from rdkit.Chem import DataStructs

quads = dict(zip(df.mol_id, df.quadrant_label))
pool_ids = [m for m, q in quads.items() if q == "low_std_low_rmse"]
wrong_ids = [m for m, q in quads.items() if q == "low_std_high_rmse"]
pool_fps = {m: fp(labels[m]["smiles"]) for m in pool_ids}
wrong_fps = {m: fp(labels[m]["smiles"]) for m in wrong_ids}

ctrl = pd.read_csv(STAGE2_CSV)
ctrl_ids = ctrl[ctrl["group"] == "control"]["mol_id"].tolist()
assert len(ctrl_ids) == 18 and all(c in pool_ids for c in ctrl_ids), "control ids broken"

def topk_sims(query_fp, target_ids, exclude=()):
    sims = [(DataStructs.TanimotoSimilarity(query_fp, pool_fps[t]), t)
            for t in target_ids if t not in exclude]
    sims.sort(key=lambda x: -x[0])
    return sims[:K]

rows = []
for mid in wrong_ids:
    top = topk_sims(wrong_fps[mid], pool_ids)
    rows.append({"mol_id": mid, "group": "confidently_wrong",
                 "top5_mean": float(np.mean([s for s, _ in top])),
                 "top5_median": float(np.median([s for s, _ in top])),
                 "best_sim": float(top[0][0]),
                 "best_neighbor": top[0][1],
                 "neighbors": ";".join(f"{t}:{s:.3f}" for s, t in top)})
for mid in ctrl_ids:
    top = topk_sims(pool_fps[mid], pool_ids, exclude={mid})
    rows.append({"mol_id": mid, "group": "control_certain",
                 "top5_mean": float(np.mean([s for s, _ in top])),
                 "top5_median": float(np.median([s for s, _ in top])),
                 "best_sim": float(top[0][0]),
                 "best_neighbor": top[0][1],
                 "neighbors": ";".join(f"{t}:{s:.3f}" for s, t in top)})
res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "neighbor_similarity_results.csv"), index=False)

from scipy.stats import mannwhitneyu
g_wrong = res[res.group == "confidently_wrong"]["top5_mean"]
g_ctrl = res[res.group == "control_certain"]["top5_mean"]
stats = {
    "k": K,
    "n_pool": len(pool_ids),
    "confidently_wrong": {
        "top5_mean_mean": float(g_wrong.mean()), "top5_mean_median": float(g_wrong.median()),
        "best_sim_mean": float(res[res.group == "confidently_wrong"]["best_sim"].mean()),
        "best_sim_median": float(res[res.group == "confidently_wrong"]["best_sim"].median()),
        "n_best_lt_0.5": int((res[res.group == "confidently_wrong"]["best_sim"] < 0.5).sum()),
        "n_best_lt_0.6": int((res[res.group == "confidently_wrong"]["best_sim"] < 0.6).sum()),
        "n_best_lt_0.7": int((res[res.group == "confidently_wrong"]["best_sim"] < 0.7).sum()),
    },
    "control_certain": {
        "top5_mean_mean": float(g_ctrl.mean()), "top5_mean_median": float(g_ctrl.median()),
        "best_sim_mean": float(res[res.group == "control_certain"]["best_sim"].mean()),
        "best_sim_median": float(res[res.group == "control_certain"]["best_sim"].median()),
        "n_best_lt_0.5": int((res[res.group == "control_certain"]["best_sim"] < 0.5).sum()),
        "n_best_lt_0.6": int((res[res.group == "control_certain"]["best_sim"] < 0.6).sum()),
        "n_best_lt_0.7": int((res[res.group == "control_certain"]["best_sim"] < 0.7).sum()),
    },
}
u, p = mannwhitneyu(g_wrong, g_ctrl, alternative="two-sided")
stats["mannwhitney_top5_mean"] = {"U": float(u), "p": float(p)}
with open(os.path.join(OUT, "neighbor_isolation_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("=" * 74)
print(f"Neighbor-isolation check: k={K} Morgan r=2 / 2048-bit Tanimoto vs "
      f"{len(pool_ids)} 'certain' molecules")
print("=" * 74)
for grp in ("confidently_wrong", "control_certain"):
    s = stats[grp]
    print(f"\n{grp.upper().replace('_', ' ')} (n={len(res[res.group == grp])}):")
    print(f"  top-5 mean similarity : mean={s['top5_mean_mean']:.3f}  median={s['top5_mean_median']:.3f}")
    print(f"  best similarity       : mean={s['best_sim_mean']:.3f}  median={s['best_sim_median']:.3f}")
    print(f"  molecules with best <0.5: {s['n_best_lt_0.5']}   <0.6: {s['n_best_lt_0.6']}   <0.7: {s['n_best_lt_0.7']}")
print(f"\nMann-Whitney U (top-5 mean sim, wrong vs control): U={stats['mannwhitney_top5_mean']['U']:.0f}, p={stats['mannwhitney_top5_mean']['p']:.4f}")

print("\nConfidently-wrong 18 by best similarity to any certain molecule (ascending):")
sub = res[res.group == "confidently_wrong"].sort_values("best_sim")
for _, r in sub.iterrows():
    print(f"  {r['mol_id']:<16} best={r['best_sim']:.3f}  neighbor={r['best_neighbor']:<16} top5mean={r['top5_mean']:.3f}")

print("\nPlain-English verdict:")
med_w, med_c = stats["confidently_wrong"]["top5_mean_median"], stats["control_certain"]["top5_mean_median"]
bmed_w, bmed_c = stats["confidently_wrong"]["best_sim_median"], stats["control_certain"]["best_sim_median"]
if p < 0.05 and med_w < med_c:
    verdict = (f"The 18 confidently-wrong molecules are SIGNIFICANTLY more isolated: top-5 mean Tanimoto "
               f"median {med_w:.3f} vs {med_c:.3f} for well-behaved molecules (Mann-Whitney p={p:.4f}). "
               f"Median best-neighbor similarity {bmed_w:.3f} vs {bmed_c:.3f}. "
               f"{stats['confidently_wrong']['n_best_lt_0.7']}/18 have no certain neighbor above 0.7.")
else:
    verdict = (f"No significant isolation: top-5 mean Tanimoto median {med_w:.3f} vs {med_c:.3f} "
               f"(Mann-Whitney p={p:.4f}). The confidently-wrong molecules are NOT systematically "
               f"further from the certain pool than typical certain molecules are from each other.")
print(verdict)
print(f"\nArtifacts -> {os.path.relpath(OUT)}/ : neighbor_similarity_results.csv, "
      f"neighbor_isolation_stats.json")
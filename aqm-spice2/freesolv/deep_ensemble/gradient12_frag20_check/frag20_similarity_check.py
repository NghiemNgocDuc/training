"""Frag20 coverage check for gradient-12 (and certain-47 control).

For each FreeSolv fold-0 test molecule in the two groups, computes the max
Morgan (r=2, 2048-bit) Tanimoto similarity against:

  * FULL Frag20-Aqsol-100K population (100,000 molecules, QM_SMILES from the
    split CSVs) - answers "is gradient-12 structurally covered by Frag20?"
  * the Br/P supplement actually prepared for fine-tuning
    (frag20_brp.hdf5, 9,260 molecules with geometry + labels) - answers
    "does the trainable supplement contain real neighbors?"

Mirrors the earlier neighbor-isolation-check format
(neighbor_similarity_results.csv): best_sim + top-5 mean/median per
molecule, plus group summaries.
"""

import collections
import csv
import json
import os

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "frag20_similarity_results.csv")
OUT_JSON = os.path.join(HERE, "frag20_group_summary.json")
REPORT_MD = os.path.join(HERE, "report.md")

DESCRIPTORS = os.path.join(HERE, "..", "gradient12_descriptor_check",
                           "descriptors_all_129.csv")
SPLIT_CSV = os.path.join(HERE, "..", "..", "experimental_frag20", "data", "split")
H5_PATH = os.path.join(HERE, "..", "..", "experimental_frag20", "data",
                       "frag20_brp.hdf5")

GROUPS = ("gradient12", "certain47")


def load_queries():
    with open(DESCRIPTORS, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["group"] in GROUPS:
            out.append((r["mol_id"], r["group"], r["smiles"]))
    return out


def load_full_frag20():
    ids, smis = [], []
    for split in ("train", "valid", "test"):
        p = os.path.join(SPLIT_CSV, f"frag20_{split}.csv")
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                ids.append(f"{r['SourceFile']}_{int(float(r['ID']))}")
                smis.append(r["QM_SMILES"])
    return ids, smis


def load_brp():
    ids, smis = [], []
    with h5py.File(H5_PATH, "r") as h5:
        for k in h5:
            ids.append(k.replace("frag20_", ""))
            smis.append(h5[k].attrs["smiles"])
    return ids, smis


def fps_from_smiles(smiles_list):
    fps, bad = [], 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return fps, bad


def summarize_sims(sims):
    a = np.asarray(sims, dtype=float)
    a = a[~np.isnan(a)]
    return a


def main():
    queries = load_queries()
    print(f"queries: {len(queries)} ({collections.Counter(g for _, g, _ in queries)})")

    qid, qgrp, qsmiles = zip(*queries)
    qfps, qbad = fps_from_smiles(list(qsmiles))
    print(f"query parse failures: {qbad}")

    full_ids, full_smis = load_full_frag20()
    print(f"Frag20 full population: {len(full_ids)}")
    full_fps, full_bad = fps_from_smiles(full_smis)
    print(f"full parse failures: {full_bad}")

    brp_ids, brp_smis = load_brp()
    print(f"Br/P supplement: {len(brp_ids)}")
    brp_fps, brp_bad = fps_from_smiles(brp_smis)
    print(f"brp parse failures: {brp_bad}")

    rows = []
    for i, (mid, grp, smi) in enumerate(queries):
        qfp = qfps[i]
        full_sims = np.full(len(full_ids), np.nan)
        brp_sims = np.full(len(brp_ids), np.nan)
        if qfp is not None:
            full_sims = np.asarray(BulkTanimotoSimilarity(qfp, full_fps), float)
            brp_sims = np.asarray(BulkTanimotoSimilarity(qfp, brp_fps), float)
        full_sims[np.isnan(full_sims)] = -1.0
        brp_sims[np.isnan(brp_sims)] = -1.0

        f_best = int(full_sims.argmax())
        b_best = int(brp_sims.argmax())
        f_sorted = np.sort(full_sims)[::-1][:5]
        b_sorted = np.sort(brp_sims)[::-1][:5]

        rows.append({
            "mol_id": mid, "group": grp,
            "full_best_sim": float(full_sims[f_best]),
            "full_best_neighbor": full_ids[f_best] if full_sims[f_best] >= 0 else "",
            "full_top5_mean": float(f_sorted.mean()),
            "full_top5_median": float(np.median(f_sorted)),
            "full_n_gt0p3": int((full_sims > 0.3).sum()),
            "brp_best_sim": float(brp_sims[b_best]),
            "brp_best_neighbor": brp_ids[b_best] if brp_sims[b_best] >= 0 else "",
            "brp_top5_mean": float(b_sorted.mean()),
            "brp_top5_median": float(np.median(b_sorted)),
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for g in GROUPS:
        gs = [r for r in rows if r["group"] == g]
        for key in ("full_best_sim", "full_top5_mean", "brp_best_sim", "brp_top5_mean"):
            vals = [r[key] for r in gs]
            summary[f"{g}_{key}_median"] = float(np.median(vals))
            summary[f"{g}_{key}_mean"] = float(np.mean(vals))
        summary[f"{g}_n"] = len(gs)
        summary[f"{g}_full_n_gt0p3"] = int(np.mean([r["full_n_gt0p3"] for r in gs]))
        summary[f"{g}_full_best_sim_min"] = float(min(r["full_best_sim"] for r in gs))
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1)

    print("\nper-molecule rows ->", OUT_CSV)
    for g in GROUPS:
        print(f"\n== {g} ==")
        for r in [x for x in rows if x["group"] == g]:
            print(f"  {r['mol_id']:>14s} full_best={r['full_best_sim']:.3f} "
                  f"({r['full_best_neighbor']}) top5m={r['full_top5_mean']:.3f} "
                  f"brp_best={r['brp_best_sim']:.3f} ({r['brp_best_neighbor']})")
    print("\nsummary ->", OUT_JSON)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
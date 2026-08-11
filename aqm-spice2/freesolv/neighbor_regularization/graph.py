"""Similarity graph over all FreeSolv molecules for neighbor-consistency
regularization.

Reuses the exact Morgan fingerprinting protocol from
../deep_ensemble/rmse_analysis/neighbor_isolation.py (radius=2, 2048 bits,
Tanimoto). Edge (i, j) exists when j is among i's top-k by Tanimoto with
w_ij = Tanimoto(i, j) >= min_sim; self-edges excluded.

Graph is STATIC (molecule structures never change during training) and built
once, cached to graph_cache/graph_k{k}_sim{min_sim}.json.

Node set = the fold-0 universe: all molecules in freesolv_conformers.hdf5 that
also have numeric expt labels in database.json (642 nodes; the same universe
cv_finetune.py builds folds from). Includes train+val+test STRUCTURES; labels
are never used here.
"""

import json
import os
import time

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def parse_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES parse failed: {smiles}")
    return mol


def morgan_fp(mol):
    """Morgan fingerprint, radius=2, 2048 bits - identical to the
    neighbor-isolation check (neighbor_isolation.py)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(a, b):
    return float(DataStructs.TanimotoSimilarity(a, b))


def find_topk(fp_i, other_fps, k, min_sim):
    """Top-k (fp, mid) pairs by Tanimoto, similarity >= min_sim, desc order."""
    sims = [(tanimoto(fp_i, fp_other), mid) for mid, fp_other in other_fps.items()]
    sims.sort(key=lambda x: -x[0])
    out = []
    for s, mid in sims:
        if s < min_sim:
            break
        out.append((s, mid))
        if len(out) >= k:
            break
    return out


def build_graph(mids, smiles_by_mid, k=5, min_sim=0.1):
    """Returns {mid: [[neighbor_mid, w_ij], ...]} sorted desc by w."""
    t0 = time.time()
    fps = {mid: morgan_fp(parse_smiles(smiles_by_mid[mid])) for mid in mids}
    # dedupe identical SMILES -> identical fingerprints (identical molecules)
    seen_fp = {}
    for mid, fp in fps.items():
        seen_fp.setdefault(fp.ToBinary(), []).append(mid)
    for fp_bin, dup_mids in seen_fp.items():
        if len(dup_mids) > 1:
            s = smiles_by_mid[dup_mids[0]]
            print(f"  [graph] NOTE: {len(dup_mids)} identical molecules share SMILES "
                  f"'{s}': {dup_mids}")

    graph = {}
    for i, mid in enumerate(mids):
        others = {m: f for m, f in fps.items() if m != mid}
        graph[mid] = find_topk(fps[mid], others, k, min_sim)
    n_zero = sum(1 for nbrs in graph.values() if not nbrs)
    print(f"  [graph] {len(mids)} nodes, k={k}, min_sim={min_sim}, "
          f"nodes with NO eligible neighbor: {n_zero}, built in "
          f"{time.time() - t0:.1f}s")
    return graph


def load_or_build_graph(graph_dir, mids, smiles_by_mid, k=5, min_sim=0.1):
    os.makedirs(graph_dir, exist_ok=True)
    cache_path = os.path.join(graph_dir, f"graph_k{k}_sim{min_sim}.json")
    meta_path = cache_path + ".meta.json"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(cache_path) as f:
            graph = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        if set(meta["mids"]) == set(mids):
            print(f"  [graph] loaded cache {os.path.relpath(cache_path)} "
                  f"({len(mids)} nodes, {meta['n_zero_neighbor']} zero-neighbor)")
            return graph, meta
        print(f"  [graph] cache stale (node set changed), rebuilding")
    graph = build_graph(mids, smiles_by_mid, k=k, min_sim=min_sim)
    with open(cache_path, "w") as f:
        json.dump(graph, f)
    meta = {
        "mids": mids, "k": k, "min_sim": min_sim,
        "n_mols": len(mids), "n_zero_neighbor": sum(1 for v in graph.values() if not v),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return graph, meta


def graph_to_tensor(graph, mids, device):
    """Static edge lists -> torch tensors (sparse-coo built as triples)."""
    import torch
    idx_i, idx_j, w = [], [], []
    pos = {mid: i for i, mid in enumerate(mids)}
    for mid in mids:
        i = pos[mid]
        for w_ij, nbr in graph[mid]:
            idx_i.append(i)
            idx_j.append(pos[nbr])
            w.append(w_ij)
    return (torch.tensor(idx_i, dtype=torch.long, device=device),
            torch.tensor(idx_j, dtype=torch.long, device=device),
            torch.tensor(w, dtype=torch.float, device=device))
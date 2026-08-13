"""Gradient-12 architecture-level checks (hypotheses 9-11), all inference/geometry-only.

Check 1 - graph diameter / over-squashing (RDKit heavy-atom bond graph from SMILES):
  diameter, mean shortest-path length (unordered pairs), n atoms whose shortest
  path to the graph center exceeds 4 hops (model has 4 interaction blocks).
Check 2 - energy-cancellation magnitude (Stage-1 vacuum + Stage-2 correction ckpts,
  aqm-spice2/aqm-spice2/pipeline/results_full/, stored-hdf5 conformer geometry):
  |E_gas|, |E_solv| = |E_gas + dG_stage2|, cancellation ratio = max(|E_gas|,|E_solv|)/|dG|.
  NOTE: Option-B correction output IS dG_solv (per predict_freesolv / bias-check doc),
  so E_solvated is reconstructed as E_gas + dG.
Check 3 - interaction-cutoff sensitivity (stored hdf5 conformers; cutoff 6.0 A,
  1.5x proximity = 9.0 A; RDKit SMILES+AddHs bond graph, index-aligned with hdf5
  - verified element order matches):
  n_through_space   = pairs 3D-close (< 9.0 A) but > 4 bonds apart in graph
  n_invisible_bonded= pairs <= 4 bonds apart but 3D distance > 6.0 A (cutoff)
Each: Mann-Whitney gradient12 vs certain47 + Spearman vs |error| and signed error
across all 129 (UNCORRECTED p-values; several metrics tested).

Outputs -> gradient12_architecture_check/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, ".."))
sys.path.insert(0, os.path.join(_script_dir, "..", ".."))
OUT_DIR = os.path.join(_script_dir, "gradient12_architecture_check")
AGG_CSV = os.path.join(_script_dir, "aggregate", "per_molecule.csv")
RMSE_CSV = os.path.join(_script_dir, "rmse_analysis", "output", "per_molecule_rmse.csv")
NEIGH_CSV = os.path.join(_script_dir, "rmse_analysis", "neighbor_isolation_check",
                         "neighbor_similarity_results.csv")

EV_TO_KCAL = 23.0605
CUTOFF_A = 6.0
PROX_A = 1.5 * CUTOFF_A
MAX_BOND_HOPS = 4


def find_repo_root():
    d = _script_dir
    while True:
        if os.path.exists(os.path.join(d, "freesolv_conformers.hdf5")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("repo root (freesolv_conformers.hdf5) not found above script")
        d = parent


DB_CANDIDATES = ("Data/FreeSolv/database.json", "aqm-spice2/Data/FreeSolv/database.json",
                 "aqm-spice2/aqm-spice2/Data/FreeSolv/database.json")
REPO_ROOT = find_repo_root()
DB_JSON = next((os.path.join(REPO_ROOT, rel) for rel in DB_CANDIDATES
                if os.path.exists(os.path.join(REPO_ROOT, rel))), None)
if DB_JSON is None:
    raise SystemExit(f"database.json not found under {REPO_ROOT} (tried {DB_CANDIDATES})")
STORED_H5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
STAGE1_CKPT = os.path.join(REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline",
                           "results_full", "stage1_fold_1.pt")
STAGE2_CKPT = os.path.join(REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline",
                           "results_full", "stage2_correction.pt")


def load_groups():
    rmse = pd.read_csv(RMSE_CSV)
    neigh = pd.read_csv(NEIGH_CSV)
    isolated6 = set(neigh[neigh["group"] == "confidently_wrong"]
                    .sort_values("best_sim").head(6)["mol_id"])
    wrong18 = set(rmse.loc[rmse["quadrant_label"] == "low_std_high_rmse", "mol_id"])
    certain47 = set(rmse.loc[rmse["quadrant_label"] == "low_std_low_rmse", "mol_id"])
    return sorted(wrong18 - isolated6), sorted(certain47)


def graph_metrics_heavy(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    n = len(adj)
    from collections import deque
    all_dists = []
    ecc = []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if adj[u, v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        all_dists.extend(dist[i] for i in range(s + 1, n))
        ecc.append(max(dist))
    diameter = max(ecc)
    center = int(np.argmin(ecc))
    center_dists = [-1] * n
    center_dists[center] = 0
    q = deque([center])
    while q:
        u = q.popleft()
        for v in range(n):
            if adj[u, v] and center_dists[v] == -1:
                center_dists[v] = center_dists[u] + 1
                q.append(v)
    n_beyond_4 = sum(1 for d in center_dists if d > MAX_BOND_HOPS)
    return {"diameter": int(diameter),
            "mean_spl": float(np.mean(all_dists)) if all_dists else 0.0,
            "n_atoms_beyond_4hops": int(n_beyond_4)}


def graph_metrics_full_h(mol_with_h):
    n = mol_with_h.GetNumAtoms()
    adj = rdmolops.GetAdjacencyMatrix(mol_with_h)
    from collections import deque
    dists = {}
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if adj[u, v] and d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(v)
        dists[s] = d
    return dists, n


def cutoff_counts(dists, n, pos, z):
    th_space = invisible = 0
    for i in range(n):
        di = pos[i]
        for j in range(i + 1, n):
            dij = float(np.linalg.norm(di - pos[j]))
            gd = dists[i][j]
            if dij < PROX_A and gd > MAX_BOND_HOPS:
                th_space += 1
            elif gd <= MAX_BOND_HOPS and dij > CUTOFF_A:
                invisible += 1
    return {"n_through_space": int(th_space), "n_invisible_bonded": int(invisible)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stage-models", action="store_true",
                        help="skip Check 2 model inference (if checkpoints missing)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    grad12, c47 = load_groups()
    mid_set = set(grad12) | set(c47)
    with open(DB_JSON) as f:
        db = json.load(f)
    agg = pd.read_csv(AGG_CSV)
    agg_by_id = {r["mol_id"]: r for r in agg.to_dict("records")}

    import h5py
    fh = h5py.File(STORED_H5, "r")

    ranks_ok = 0
    rows = []
    for mid in sorted(mid_set):
        smi = db[mid]["smiles"]
        g1 = graph_metrics_heavy(smi)
        if g1 is None:
            print(f"[warn] SMILES parse failed: {mid}")
            continue
        g = fh[mid]
        z = np.asarray(g["atNUM"][...], dtype=int)
        pos = np.asarray(g["atXYZ"][...], dtype=float)
        molH = Chem.AddHs(Chem.MolFromSmiles(smi))
        hz = [a.GetAtomicNum() for a in molH.GetAtoms()]
        if list(z) != hz:
            print(f"[warn] hdf5 atom order mismatch vs SMILES+AddHs: {mid}")
            continue
        ranks_ok += 1
        dists, n = graph_metrics_full_h(molH)
        g3 = cutoff_counts(dists, n, pos, z)
        rec = agg_by_id.get(mid)
        row = {"mol_id": mid,
               "group": "gradient12" if mid in grad12 else "certain47",
               "abs_err_kcal": rec["abs_error"] if rec else np.nan,
               "signed_err_kcal": (rec["ensemble_mean"] - rec["true_value"]) if rec else np.nan}
        row.update({f"c1_{k}": v for k, v in g1.items()})
        row.update({f"c3_{k}": v for k, v in g3.items()})
        rows.append(row)
    fh.close()

    if not args.skip_stage_models and os.path.exists(STAGE1_CKPT) and os.path.exists(STAGE2_CKPT):
        from DimeModels import DimeNetPlus
        from element_vocab import build_one_hot, NUM_ELEMENTS
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Data

        def make_model(num_blocks):
            return DimeNetPlus(
                in_channels=NUM_ELEMENTS, hidden_channels=128, out_channels=1,
                num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
                out_emb_channels=256, num_spherical=7, num_radial=6,
                cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3,
                is_energy=True)

        dev = torch.device("cpu")
        vac = make_model(4)
        vac.load_state_dict(torch.load(STAGE1_CKPT, map_location="cpu", weights_only=True))
        vac.eval()
        corr = make_model(3)
        corr.load_state_dict(torch.load(STAGE2_CKPT, map_location="cpu", weights_only=True))
        corr.eval()
        fh = h5py.File(STORED_H5, "r")
        graphs = []
        mid_list = [r["mol_id"] for r in rows]
        for mid in mid_list:
            g = fh[mid]
            graphs.append(Data(z=torch.tensor(np.asarray(g["atNUM"][...], dtype=np.int32),
                                              dtype=torch.long),
                               pos=torch.tensor(np.asarray(g["atXYZ"][...], dtype=float),
                                                dtype=torch.float)))
        fh.close()
        with torch.no_grad():
            e_gas = []
            e_corr = []
            for data in DataLoader(graphs, batch_size=64, shuffle=False):
                x = build_one_hot(data, dev)
                e_gas.append(vac(x, data.pos, data.batch).view(-1).cpu().numpy() * EV_TO_KCAL)
                e_corr.append(corr(x, data.pos, data.batch).view(-1).cpu().numpy() * EV_TO_KCAL)
        e_gas = np.concatenate(e_gas)
        e_corr = np.concatenate(e_corr)
        for r, eg, ec in zip(rows, e_gas, e_corr):
            dg = abs(ec)
            esolv = abs(eg + ec)
            r["c2_E_gas_kcal"] = float(abs(eg))
            r["c2_E_solv_kcal"] = float(esolv)
            r["c2_dG_kcal"] = float(dg)
            r["c2_cancel_ratio"] = float(max(abs(eg), esolv) / dg) if dg > 1e-6 else float("nan")
        print(f"[arch] Stage models: E_gas/E_solv computed for {len(rows)} molecules")
    else:
        print(f"[arch] Check 2 skipped (stage1={os.path.exists(STAGE1_CKPT)}, "
              f"stage2={os.path.exists(STAGE2_CKPT)}, flag={args.skip_stage_models})")

    print(f"[arch] {ranks_ok}/{len(rows)} molecules with hdf5 order verified")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_molecule_metrics.csv"), index=False)

    g = df[df["group"] == "gradient12"]
    c = df[df["group"] == "certain47"]
    metric_cols = [col for col in df.columns if col.startswith(("c1_", "c2_", "c3_"))]
    mwu, spearman = {}, {}
    for col in metric_cols:
        gv = g[col].dropna().values
        cv = c[col].dropna().values
        if len(gv) < 5 or len(cv) < 5:
            continue
        u, p = stats.mannwhitneyu(gv, cv, alternative="two-sided")
        mwu[col] = {"median_g12": float(np.median(gv)), "median_c47": float(np.median(cv)),
                    "mean_g12": float(np.mean(gv)), "mean_c47": float(np.mean(cv)),
                    "u": float(u), "p": float(p)}
        print(f"[arch] MWU {col:24s} g12 med {np.median(gv):9.4g} vs c47 med "
              f"{np.median(cv):9.4g}  p={p:.4f}")
    for col in metric_cols:
        sub = df.dropna(subset=[col, "abs_err_kcal", "signed_err_kcal"])
        if len(sub) < 20:
            continue
        r_abs = stats.spearmanr(sub[col], sub["abs_err_kcal"])
        r_sgn = stats.spearmanr(sub[col], sub["signed_err_kcal"])
        spearman[col] = {"rho_abs": float(r_abs.statistic), "p_abs": float(r_abs.pvalue),
                         "rho_signed": float(r_sgn.statistic), "p_signed": float(r_sgn.pvalue)}
        print(f"[arch] spearman {col:24s} vs |err| rho={r_abs.statistic:+.3f} p={r_abs.pvalue:.4f} "
              f"| vs signed rho={r_sgn.statistic:+.3f} p={r_sgn.pvalue:.4f}")

    with open(os.path.join(OUT_DIR, "mwu_results.json"), "w") as f:
        json.dump(mwu, f, indent=2)
    with open(os.path.join(OUT_DIR, "spearman_results.json"), "w") as f:
        json.dump(spearman, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gradient12": "#d62728", "certain47": "#1f77b4", "other": "#bbbbbb"}
    check_prefix = {"c1": "check1_graph", "c2": "check2_energy", "c3": "check3_cutoff"}
    for prefix, name in check_prefix.items():
        cols = [col for col in metric_cols if col.startswith(prefix + "_")]
        if not cols:
            continue
        best = max(cols, key=lambda col: abs(spearman.get(col, {}).get("rho_abs", 0)))
        fig, ax = plt.subplots(figsize=(7, 5))
        for grp_name, grp_df in df.groupby("group"):
            ax.scatter(grp_df[best], grp_df["abs_err_kcal"], s=38,
                       color=colors[grp_name], label=grp_name, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
        rho = spearman.get(best, {}).get("rho_abs")
        pv = spearman.get(best, {}).get("p_abs")
        ax.set_xlabel(f"{best} (group medians: g12={np.median(g[best]):.4g}, "
                      f"c47={np.median(c[best]):.4g})")
        ax.set_ylabel("|ensemble error| (kcal/mol)")
        ax.set_title(f"{name}: {best} vs |error|  (spearman rho={rho:+.3f} p={pv:.4f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"scatter_{name}_vs_abserr.png"), dpi=150)
        plt.close(fig)
    print(f"[arch] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
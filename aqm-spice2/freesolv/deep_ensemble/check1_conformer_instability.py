"""Check 1: conformer ensemble instability for gradient-12.

Reuses the exact training-time TTA protocol from deep_ensemble.conformer_average:
RDKit ETKDGv3 (randomSeed 42, pruneRmsThresh 0.5) + MMFF optimization, mean over
conformers = prediction, DimeNetPlus seed-42 checkpoint (deep_ensemble/seed_42).

Per molecule (ALL 129 fold-0 test molecules):
  * n_conformers_kept (50 requested, pruned by ETKDGv3)
  * per-conformer MMFF energy + spread (min-max, kcal/mol) -> genuine multi-minima?
  * per-conformer model prediction -> prediction-across-conformers std
  * 50-conformer TTA prediction (mean) vs the ORIGINAL 5-conformer TTA prediction
    (seed_42/predictions.csv) vs true value

Tests:
  1. Mann-Whitney gradient-12 vs certain-47: pred std, MMFF energy spread, n confs
  2. Spearman: pred-std vs original 5-conf-TTA abs error across all 129
     (+ robustness on the 64 "other" molecules only)
  3. Does 50-conf TTA improve gradient-12 accuracy vs 5-conf TTA?

Outputs -> deep_ensemble/gradient12_conformer_provenance_check/
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, ".."))
OUT_DIR = os.path.join(_script_dir, "gradient12_conformer_provenance_check")


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


def find_db():
    for rel in DB_CANDIDATES:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.exists(p):
            return p
    raise SystemExit(f"database.json not found under {REPO_ROOT} (tried {DB_CANDIDATES})")


REPO_ROOT = find_repo_root()
DB_JSON = find_db()
STORED_H5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")


def find_checkpoint():
    cand = os.path.join(_script_dir, "seed_42", "ensemble_seed42.pt")
    if os.path.exists(cand):
        return cand
    for levels in (1, 2, 3, 4):
        base = _script_dir
        for _ in range(levels):
            base = os.path.dirname(base)
        cand = os.path.join(base, "deep_ensemble", "seed_42", "ensemble_seed42.pt")
        if os.path.exists(cand):
            return cand
    raise SystemExit("ensemble_seed42.pt not found (script_dir/seed_42 or above)")


CKPT = find_checkpoint()
AGG_CSV = os.path.join(_script_dir, "aggregate", "per_molecule.csv")
RMSE_CSV = os.path.join(_script_dir, "rmse_analysis", "output", "per_molecule_rmse.csv")
NEIGH_CSV = os.path.join(_script_dir, "rmse_analysis", "neighbor_isolation_check",
                         "neighbor_similarity_results.csv")
PRED5_CSV = os.path.join(_script_dir, "seed_42", "predictions.csv")
TEST_IDS = os.path.join(_script_dir, "seed_42", "test_ids.json")

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

import deep_ensemble as de
from element_vocab import build_one_hot

N_REQUESTED = 50
BATCH = 32


def load_groups():
    rmse = pd.read_csv(RMSE_CSV)
    neigh = pd.read_csv(NEIGH_CSV)
    isolated6 = set(neigh[neigh["group"] == "confidently_wrong"]
                    .sort_values("best_sim").head(6)["mol_id"])
    wrong18 = set(rmse.loc[rmse["quadrant_label"] == "low_std_high_rmse", "mol_id"])
    certain47 = set(rmse.loc[rmse["quadrant_label"] == "low_std_low_rmse", "mol_id"])
    return sorted(wrong18 - isolated6), sorted(isolated6), sorted(wrong18), sorted(certain47)


def gen_confs(smiles, n):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
                     dtype=torch.long)
    energies = []
    graphs = []
    for i in conf_ids:
        energies.append(rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, props, confId=i).CalcEnergy())
        graphs.append(Data(z=z.clone(), pos=torch.tensor(
            np.array(mol.GetConformer(i).GetPositions(), dtype=np.float64),
            dtype=torch.float)))
    return graphs, energies


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grad12, iso6, wrong18, c47 = load_groups()
    with open(DB_JSON) as f:
        db = json.load(f)
    test_ids = json.load(open(TEST_IDS))
    agg = pd.read_csv(AGG_CSV)
    pred5 = pd.read_csv(PRED5_CSV)
    group_map = {m: ("gradient12" if m in grad12 else "isolated6" if m in iso6 else
                     "certain47" if m in c47 else "wrong18" if m in wrong18 else "other")
                 for m in test_ids}

    device = torch.device("cpu")
    model = de.build_model(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    agg_err = dict(zip(agg["mol_id"], agg["abs_error"]))
    tta5 = dict(zip(pred5["mol_id"], pred5["dG_pred_kcal"]))
    true = dict(zip(pred5["mol_id"], pred5["dG_exp_kcal"]))

    calib = []
    import h5py
    with h5py.File(STORED_H5, "r") as fh:
        for mid in test_ids:
            g = fh[mid]
            d = Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                     pos=torch.tensor(g["atXYZ"][...], dtype=torch.float))
            with torch.no_grad():
                loader = DataLoader([d], batch_size=8, shuffle=False)
                for data in loader:
                    x = build_one_hot(data, device)
                    p = (model(x, data.pos, data.batch).view(-1) * de.EV_TO_KCAL).item()
            calib.append({"mol_id": mid, "stored_single_conf_pred_kcal": float(p),
                          "true_kcal": true[mid]})
    calib = pd.DataFrame(calib)
    calib["stored_single_conf_abs_err"] = (calib["stored_single_conf_pred_kcal"]
                                           - calib["true_kcal"]).abs()
    calib.to_csv(os.path.join(OUT_DIR, "calibration_stored_conformer.csv"), index=False)
    print(f"[ck1] calibration: stored-box-conformer single-conf MAE = "
          f"{calib['stored_single_conf_abs_err'].mean():.4f} "
          f"(training-time single-conf MAE 0.5313, per metrics.json)")

    rows = []
    conf_rows = []
    t0 = time.time()
    for mid in test_ids:
        graphs, energies = gen_confs(db[mid]["smiles"], N_REQUESTED)
        preds = []
        with torch.no_grad():
            loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            for data in loader:
                x = build_one_hot(data, device)
                preds.append((model(x, data.pos, data.batch).view(-1) * de.EV_TO_KCAL).cpu())
        preds = torch.cat(preds).numpy()
        graphs5, _ = gen_confs(db[mid]["smiles"], 5)
        preds5 = []
        with torch.no_grad():
            loader5 = DataLoader(graphs5, batch_size=BATCH, shuffle=False)
            for data in loader5:
                x5 = build_one_hot(data, device)
                preds5.append((model(x5, data.pos, data.batch).view(-1) * de.EV_TO_KCAL).cpu())
        preds5 = torch.cat(preds5).numpy()
        for i, (p, e) in enumerate(zip(preds, energies)):
            conf_rows.append({"mol_id": mid, "conf_idx": i, "mmff_energy_kcal": float(e),
                              "pred_kcal": float(p)})
        std50 = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
        rows.append({
            "mol_id": mid, "group": group_map[mid],
            "n_conformers_kept_50": len(preds),
            "pred_std_across_conformers": std50,
            "pred_range_kcal": float(np.ptp(preds)),
            "mmff_energy_spread_kcal": float(np.ptp(energies)) if len(energies) > 1 else 0.0,
            "tta50_pred_kcal": float(np.mean(preds)),
            "tta5_pred_kcal_local": float(np.mean(preds5)),
            "tta5_pred_kcal_box": tta5[mid], "true_kcal": true[mid],
            "abs_err_tta5_box": agg_err[mid],
            "abs_err_tta5_local": float(abs(np.mean(preds5) - true[mid])),
            "abs_err_tta50": float(abs(np.mean(preds) - true[mid])),
        })
    print(f"[ck1] conformer gen + inference done in {(time.time()-t0)/60:.1f} min "
          f"({len(conf_rows)} conformers)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "per_molecule_conformer_stats.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(
        os.path.join(OUT_DIR, "per_conformer_predictions.csv"), index=False)

    g = df[df["group"] == "gradient12"]
    c = df[df["group"] == "certain47"]
    other = df[df["group"] == "other"]

    stats_out = {"n_gradient12": len(g), "n_certain47": len(c)}
    for col in ("pred_std_across_conformers", "mmff_energy_spread_kcal",
                "n_conformers_kept_50"):
        u, p = stats.mannwhitneyu(g[col].values, c[col].values, alternative="two-sided")
        stats_out[f"mwu_{col}"] = {"median_g12": float(np.median(g[col])),
                                   "median_c47": float(np.median(c[col])),
                                   "mean_g12": float(np.mean(g[col])),
                                   "mean_c47": float(np.mean(c[col])),
                                   "u": float(u), "p": float(p)}
        print(f"[ck1] MWU {col:30s} g12 med {np.median(g[col]):.4f} vs c47 med "
              f"{np.median(c[col]):.4f}  p={p:.4f}")

    for col, label in (("pred_std_across_conformers", "conformer-pred-std"),
                       ("mmff_energy_spread_kcal", "MMFF energy spread")):
        for sub, subname in ((df, "all 129"), (other, "other-64 only")):
            rho, p = stats.spearmanr(sub[col], sub["abs_err_tta5_box"])
            stats_out[f"spearman_{col}_{subname}"] = {"rho": float(rho), "p": float(p),
                                                      "n": int(len(sub))}
            print(f"[ck1] spearman {label:26s} vs 5-conf-TTA err ({subname}): "
                  f"rho={rho:+.3f} p={p:.4f}")

    stats_out["tta_improvement"] = {
        "g12_mae5_local": float(np.mean(g["abs_err_tta5_local"])),
        "g12_mae50": float(np.mean(g["abs_err_tta50"])),
        "c47_mae5_local": float(np.mean(c["abs_err_tta5_local"])),
        "c47_mae50": float(np.mean(c["abs_err_tta50"])),
        "all_mae5_local": float(np.mean(df["abs_err_tta5_local"])),
        "all_mae50": float(np.mean(df["abs_err_tta50"])),
        "g12_n_improved_50conf": int(np.sum(
            (g["abs_err_tta5_local"] - g["abs_err_tta50"]) > 0)),
    }
    print(f"[ck1] MAE5(local) vs MAE50: g12 {stats_out['tta_improvement']['g12_mae5_local']:.3f} -> "
          f"{stats_out['tta_improvement']['g12_mae50']:.3f} | all "
          f"{stats_out['tta_improvement']['all_mae5_local']:.3f} -> "
          f"{stats_out['tta_improvement']['all_mae50']:.3f} | "
          f"g12 improved {stats_out['tta_improvement']['g12_n_improved_50conf']}/12")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gradient12": "tab:red", "isolated6": "tab:orange",
              "certain47": "tab:blue", "other": "lightgray"}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for gn, colr in colors.items():
        sub = df[df["group"] == gn]
        ax.scatter(sub["pred_std_across_conformers"], sub["abs_err_tta5_box"], c=colr, s=28,
                   alpha=0.85, edgecolor="none", label=gn,
                   zorder=3 if gn != "other" else 1)
    ax.set_xlabel("prediction std across 50 conformers (kcal/mol)")
    ax.set_ylabel("original 5-conf TTA |error| (kcal/mol)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scatter_conformer_std_vs_error.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "check1_report.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"[ck1] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

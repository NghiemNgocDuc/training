"""Check 2: conformer-draw sensitivity (box environment) for gradient-12 vs certain-47.

Question 8 in the investigation: would gradient-12's "wrong" predictions change if
a DIFFERENT, equally valid conformer draw had been frozen into the pipeline?

Protocol (identical machinery to check1/deep_ensemble.conformer_average, but with
DIFFERENT embedding seeds):
  * For each gradient-12 + certain-47 molecule: generate 5 conformers (ETKDGv3,
    pruneRmsThresh 0.5, +MMFF) with embedding seeds 7, 123, 2024, 999 (NOT the
    training seed 42).
  * Predict each conformer with the seed_42 checkpoint; TTA-5 mean per draw.
  * Sensitivity baselines for each molecule:
      (a) stored-conformer single prediction (hdf5, recomputed here),
      (b) recorded training-time 5-conf TTA prediction (deep_ensemble/seed_42/
          predictions.csv -- the values behind the recorded MAEs).
      (c) --local-baseline-seed N: also draw seed N fresh (default protocol's
          "original" draw) and use its TTA as the baseline instead -- the
          internally-consistent mode for environments whose stored conformers /
          recorded predictions are not trustworthy (e.g. local env-drift case).
    Per-molecule sensitivity = mean over the new draws of |new_draw_pred - baseline|.
  * Mann-Whitney gradient-12 vs certain-47 on sensitivity (baselines, plus
    per-seed breakdown).
  * Pooling test (actionable fix candidate): pooled prediction = mean of
    baseline draw + all new draws (25 conformers, 5 seeds) vs true value;
    compare gradient-12 error change vs baseline-draw TTA-5 (Wilcoxon signed-rank).

Outputs -> gradient12_conformer_provenance_check/check2_*
"""

import argparse
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
AGG_CSV = os.path.join(_script_dir, "aggregate", "per_molecule.csv")
RMSE_CSV = os.path.join(_script_dir, "rmse_analysis", "output", "per_molecule_rmse.csv")
NEIGH_CSV = os.path.join(_script_dir, "rmse_analysis", "neighbor_isolation_check",
                         "neighbor_similarity_results.csv")
PRED5_CSV = os.path.join(_script_dir, "seed_42", "predictions.csv")

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

import deep_ensemble as de
from element_vocab import build_one_hot

N_CONFORMERS_PER_DRAW = 5
BATCH = 32


def find_repo_root():
    d = _script_dir
    while True:
        if (os.path.exists(os.path.join(d, "Data", "FreeSolv", "database.json"))
                and os.path.exists(os.path.join(d, "freesolv_conformers.hdf5"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("repo root (Data/FreeSolv + freesolv_conformers.hdf5) not found above script")
        d = parent


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


REPO_ROOT = find_repo_root()
DB_JSON = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")
STORED_H5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
CKPT = find_checkpoint()


def load_groups():
    rmse = pd.read_csv(RMSE_CSV)
    neigh = pd.read_csv(NEIGH_CSV)
    isolated6 = set(neigh[neigh["group"] == "confidently_wrong"]
                    .sort_values("best_sim").head(6)["mol_id"])
    wrong18 = set(rmse.loc[rmse["quadrant_label"] == "low_std_high_rmse", "mol_id"])
    certain47 = set(rmse.loc[rmse["quadrant_label"] == "low_std_low_rmse", "mol_id"])
    return sorted(wrong18 - isolated6), sorted(certain47)


def gen_confs(smiles, n, seed):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
                     dtype=torch.long)
    return [Data(z=z.clone(), pos=torch.tensor(
        np.array(mol.GetConformer(i).GetPositions(), dtype=np.float64),
        dtype=torch.float)) for i in conf_ids]


def predict_graphs(model, device, graphs):
    preds = []
    with torch.no_grad():
        loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
        for data in loader:
            x = build_one_hot(data, device)
            preds.append((model(x, data.pos, data.batch).view(-1) * de.EV_TO_KCAL).cpu())
    return torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*",
                        default=[7, 123, 2024, 999], help="embedding seeds != 42")
    parser.add_argument("--local-baseline-seed", type=int, default=None,
                        help="also draw this seed fresh and use it as the baseline "
                             "draw (internally-consistent mode; box default = stored "
                             "conformer + recorded TTA-5 baselines)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    grad12, c47 = load_groups()
    mids = sorted(set(grad12) | set(c47))
    with open(DB_JSON) as f:
        db = json.load(f)
    pred5 = pd.read_csv(PRED5_CSV)
    tta5 = dict(zip(pred5["mol_id"], pred5["dG_pred_kcal"]))
    true = dict(zip(pred5["mol_id"], pred5["dG_exp_kcal"]))

    seeds_all = list(args.seeds)
    if args.local_baseline_seed is not None:
        seeds_all = sorted(set(seeds_all + [args.local_baseline_seed]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ck2] device: {device}")
    model = de.build_model(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    import h5py
    stored_pred, calib_err = {}, []
    with h5py.File(STORED_H5, "r") as fh:
        for mid in mids:
            g = fh[mid]
            d = Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                     pos=torch.tensor(g["atXYZ"][...], dtype=torch.float))
            p = predict_graphs(model, device, [d])[0]
            stored_pred[mid] = float(p)
            calib_err.append(abs(p - true[mid]))
    print(f"[ck2] calibration on {len(mids)} mols: stored-conformer single-conf MAE = "
          f"{np.mean(calib_err):.4f} (recorded 0.5313 per metrics.json) - "
          f"{'BOX ENVIRONMENT INTACT' if abs(np.mean(calib_err) - 0.5313) < 0.05 else 'DISCREPANCY'}")

    t0 = time.time()
    per_seed_tta = {s: {} for s in seeds_all}
    for s in seeds_all:
        for mid in mids:
            graphs = gen_confs(db[mid]["smiles"], N_CONFORMERS_PER_DRAW, seed=s)
            tta = float(np.mean(predict_graphs(model, device, graphs)))
            per_seed_tta[s][mid] = tta
    print(f"[ck2] fresh draws (seeds {seeds_all}) done in {(time.time()-t0)/60:.1f} min")

    local_mode = args.local_baseline_seed is not None
    orig_seed = args.local_baseline_seed
    rows = []
    for mid in mids:
        row = {"mol_id": mid, "group": "gradient12" if mid in grad12 else "certain47",
               "true_kcal": true[mid], "stored_single_conf_pred_kcal": stored_pred[mid],
               "tta5_recorded_kcal": tta5[mid],
               "abs_err_tta5_recorded": abs(tta5[mid] - true[mid])}
        if local_mode:
            row[f"orig_draw_seed{orig_seed}_tta_kcal"] = per_seed_tta[orig_seed][mid]
            row["abs_err_orig_draw"] = abs(per_seed_tta[orig_seed][mid] - true[mid])
        for s in seeds_all:
            row[f"draw_seed{s}_tta_kcal"] = per_seed_tta[s][mid]
        fresh = [s for s in seeds_all] if local_mode else list(args.seeds)
        deltas_vs_stored = [abs(per_seed_tta[s][mid] - stored_pred[mid]) for s in fresh]
        deltas_vs_tta5 = [abs(per_seed_tta[s][mid] - tta5[mid]) for s in fresh]
        if local_mode:
            deltas_vs_orig = [abs(per_seed_tta[s][mid] - per_seed_tta[orig_seed][mid])
                              for s in fresh if s != orig_seed]
        row["mean_abs_delta_vs_stored"] = float(np.mean(deltas_vs_stored))
        row["max_abs_delta_vs_stored"] = float(np.max(deltas_vs_stored))
        row["mean_abs_delta_vs_tta5"] = float(np.mean(deltas_vs_tta5))
        row["max_abs_delta_vs_tta5"] = float(np.max(deltas_vs_tta5))
        if local_mode:
            row["mean_abs_delta_vs_orig_draw"] = float(np.mean(deltas_vs_orig))
            row["max_abs_delta_vs_orig_draw"] = float(np.max(deltas_vs_orig))
        pooled_parts = ([per_seed_tta[orig_seed][mid]] if local_mode else [tta5[mid]]) \
            + [per_seed_tta[s][mid] for s in fresh]
        row["ttpool_allseeds_kcal"] = float(np.mean(pooled_parts))
        row["abs_err_pooled_allseeds"] = abs(row["ttpool_allseeds_kcal"] - true[mid])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "check2_sensitivity.csv"), index=False)
    pd.DataFrame([{**{"seed": s}, **{mid: per_seed_tta[s][mid] for mid in mids}}
                  for s in seeds_all]).to_csv(
        os.path.join(OUT_DIR, "check2_fresh_draw_predictions.csv"), index=False)

    g = df[df["group"] == "gradient12"]
    c = df[df["group"] == "certain47"]
    rep = {"n_gradient12": len(g), "n_certain47": len(c), "seeds": args.seeds,
           "local_baseline_seed": args.local_baseline_seed}
    cols = ["mean_abs_delta_vs_stored", "max_abs_delta_vs_stored",
            "mean_abs_delta_vs_tta5", "max_abs_delta_vs_tta5"]
    if local_mode:
        cols += ["mean_abs_delta_vs_orig_draw", "max_abs_delta_vs_orig_draw"]
    for col in cols:
        u, p = stats.mannwhitneyu(g[col].values, c[col].values, alternative="two-sided")
        rep[f"mwu_{col}"] = {"median_g12": float(np.median(g[col])),
                             "median_c47": float(np.median(c[col])),
                             "mean_g12": float(np.mean(g[col])),
                             "mean_c47": float(np.mean(c[col])),
                             "u": float(u), "p": float(p)}
        print(f"[ck2] MWU {col:32s} g12 med {np.median(g[col]):.4f} vs c47 med "
              f"{np.median(c[col]):.4f}  p={p:.4f}")
    for s in seeds_all:
        gd = np.array([abs(per_seed_tta[s][m] - tta5[m]) for m in g["mol_id"]])
        cd = np.array([abs(per_seed_tta[s][m] - tta5[m]) for m in c["mol_id"]])
        u, p = stats.mannwhitneyu(gd, cd, alternative="two-sided")
        rep[f"mwu_seed{s}_delta_vs_tta5"] = {"median_g12": float(np.median(gd)),
                                             "median_c47": float(np.median(cd)),
                                             "p": float(p)}
        print(f"[ck2] MWU seed {s:>4} delta-vs-tta5: g12 med {np.median(gd):.4f} vs "
              f"c47 med {np.median(cd):.4f}  p={p:.4f}")

    base_err_col = "abs_err_orig_draw" if local_mode else "abs_err_tta5_recorded"
    pool_err_col = "abs_err_pooled_allseeds"
    g_imp = (g[base_err_col] - g[pool_err_col]).values
    c_imp = (c[base_err_col] - c[pool_err_col]).values
    w_p = stats.wilcoxon(g_imp, alternative="two-sided").pvalue if len(g_imp) >= 5 else None
    rep["pooling"] = {
        "g12_mae_baseline": float(np.mean(g[base_err_col])),
        "g12_mae_pooled_allseeds": float(np.mean(g[pool_err_col])),
        "c47_mae_baseline": float(np.mean(c[base_err_col])),
        "c47_mae_pooled_allseeds": float(np.mean(c[pool_err_col])),
        "g12_n_improved_by_pooling": int(np.sum(g_imp > 0)),
        "g12_wilcoxon_p": float(w_p) if w_p else None,
        "baseline": "orig_draw" if local_mode else "tta5_recorded",
    }
    print(f"[ck2] pooling (all seeds x 5 confs, baseline={rep['pooling']['baseline']}): "
          f"g12 MAE {rep['pooling']['g12_mae_baseline']:.3f} -> "
          f"{rep['pooling']['g12_mae_pooled_allseeds']:.3f} | c47 "
          f"{rep['pooling']['c47_mae_baseline']:.3f} -> "
          f"{rep['pooling']['c47_mae_pooled_allseeds']:.3f} | g12 improved "
          f"{rep['pooling']['g12_n_improved_by_pooling']}/{len(g)} (wilcoxon p={w_p:.4f})")

    with open(os.path.join(OUT_DIR, "check2_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[ck2] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
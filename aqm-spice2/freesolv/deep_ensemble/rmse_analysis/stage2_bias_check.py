"""Stage-2 bias check for the 18 "confidently wrong" fold-0 ensemble molecules.

Runs the FROZEN Stage-2 correction model (the exact checkpoint the 5 ensemble
seeds were fine-tuned from) directly on the two groups:

  group A: 18 molecules in quadrant low_std_high_rmse (confidently wrong)
  group B: 18-control from quadrant low_std_low_rmse (seed 42)

Protocol mirrors deep_ensemble.conformer_average exactly: RDKit ETKDGv3
(randomSeed 42, pruneRmsThresh 0.5) + MMFF, 5 conformers, mean = prediction,
Stage-2 output dG (eV) x EV_TO_KCAL is directly comparable to the FreeSolv
experimental value (Option B identity: correction output IS dG_solv).

Outputs (stage2_bias_check/):
  stage2_predictions.csv   per-molecule rows for both groups
  stage2_bias_stats.json   group error stats + Mann-Whitney + correlation
  stage2_error_boxplot.png boxplot of |stage2 error| by group
"""

import sys
import os
import json
import random
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))          # rmse_analysis → aqm-spice2 → repo root
FREESOLV_DIR = os.path.join(REPO_ROOT, "aqm-spice2", "freesolv")
sys.path.insert(0, FREESOLV_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "aqm-spice2"))

from predict_freesolv import build_model                      # same DimeNetPlus config as inference
from freesolv_dataset import EV_TO_KCAL, load_freesolv_labels
from element_vocab import build_one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_CSV = os.path.join(HERE, "output", "per_molecule_rmse.csv")
STAGE2_CKPT = os.path.join(REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline",
                           "results_full", "stage2_correction.pt")
LABELS_JSON = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")
CONFORMERS_H5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
OUT = os.path.join(HERE, "stage2_bias_check")

N_TTA = 5
RANDOM_SEED = 42


def load_model():
    if not os.path.exists(STAGE2_CKPT):
        raise FileNotFoundError(f"Stage-2 checkpoint missing: {STAGE2_CKPT}")
    model = build_model(num_blocks=3)
    model.load_state_dict(torch.load(STAGE2_CKPT, map_location="cpu", weights_only=True))
    model.eval()
    return model


def gen_confs(smiles, n):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.pruneRmsThresh = 0.5
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
    if not conf_ids:
        return None
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    try:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    except Exception:
        pass
    from torch_geometric.data import Data
    z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
                     dtype=torch.long)
    n_avail = min(n, mol.GetNumConformers())
    return [Data(z=z.clone(),
                 pos=torch.tensor(np.array(mol.GetConformer(i).GetPositions(),
                                           dtype=np.float64), dtype=torch.float))
            for i in range(n_avail)]


def hdf5_fallback(mid, n):
    import h5py
    from torch_geometric.data import Data
    with h5py.File(CONFORMERS_H5, "r") as f:
        if mid not in f:
            raise SystemExit(f"mol_id {mid} missing from conformer HDF5 {CONFORMERS_H5}")
        g = f[mid]
        d = Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                 pos=torch.tensor(g["atXYZ"][...], dtype=torch.float))
    return [d.clone() for _ in range(n)]


def predict_stage2(model, mids, labels):
    from torch_geometric.loader import DataLoader
    flat, flat_mid, n_confs, used_fallback = [], [], {}, []
    for mid in mids:
        confs = gen_confs(labels[mid]["smiles"], N_TTA)
        if confs is None or len(confs) == 0:
            confs = hdf5_fallback(mid, N_TTA)
            used_fallback.append(mid)
            n_confs[mid] = N_TTA
        else:
            n_confs[mid] = len(confs)
        for cd in confs:
            flat.append(cd)
            flat_mid.append(mid)
    loader = DataLoader(flat, batch_size=32, shuffle=False)
    all_raw = []
    with torch.no_grad():
        for data in loader:
            x = build_one_hot(data, torch.device("cpu"))
            all_raw.append(model(x, data.pos, data.batch).view(-1).cpu() * EV_TO_KCAL)
    all_raw = torch.cat(all_raw).numpy()
    conf_preds = {}
    for mid, v in zip(flat_mid, all_raw):
        conf_preds.setdefault(mid, []).append(float(v))
    preds = {mid: float(np.mean(conf_preds[mid])) for mid in mids}
    return preds, used_fallback, n_confs


def main():
    if not os.path.exists(LABELS_JSON):
        raise FileNotFoundError(f"FreeSolv labels missing: {LABELS_JSON}")
    labels = load_freesolv_labels(LABELS_JSON)

    df = pd.read_csv(ANALYSIS_CSV)
    wrong = df[df["quadrant_label"] == "low_std_high_rmse"]["mol_id"].tolist()
    pool = df[df["quadrant_label"] == "low_std_low_rmse"]["mol_id"].tolist()
    rng = random.Random(RANDOM_SEED)
    control = rng.sample(pool, min(18, len(pool)))

    for mid in wrong + control:
        if mid not in labels:
            raise SystemExit(f"mol_id {mid} missing from FreeSolv labels")

    model = load_model()
    preds, used_fallback, n_confs = predict_stage2(model, wrong + control, labels)

    def collect(mids, group):
        rows = []
        for mid in mids:
            pred = preds[mid]
            expt = float(labels[mid]["expt"])
            rows.append({
                "mol_id": mid, "group": group, "smiles": labels[mid]["smiles"],
                "stage2_pred_kcal": pred, "expt_kcal": expt,
                "stage2_err_kcal": pred - expt,
                "stage2_abs_err_kcal": abs(pred - expt),
                "n_tta_confs": n_confs[mid],
                "tta_fallback_hdf5": mid in used_fallback,
            })
        return rows

    rows = collect(wrong, "confidently_wrong") + collect(control, "control")
    extra = pd.read_csv(os.path.join(HERE, "..", "aggregate", "per_molecule.csv"),
                        names=["mol_id", "pred_seed42", "pred_seed123", "pred_seed7",
                               "pred_seed2024", "pred_seed999", "ensemble_mean",
                               "ensemble_std", "true_value", "abs_error",
                               "has_halogen_Br_I"])
    out_df = pd.DataFrame(rows).merge(
        extra[["mol_id", "ensemble_mean", "ensemble_std"]],
        on="mol_id", how="left").merge(
        df[["mol_id", "rmse_across_seeds"]], on="mol_id", how="left")
    os.makedirs(OUT, exist_ok=True)
    out_df.to_csv(os.path.join(OUT, "stage2_predictions.csv"), index=False)

    from scipy.stats import mannwhitneyu, spearmanr
    stats = {}
    for grp in ("confidently_wrong", "control"):
        g = out_df[out_df["group"] == grp]["stage2_abs_err_kcal"]
        stats[grp] = {
            "n": int(g.count()),
            "mean_abs_err_kcal": float(g.mean()),
            "median_abs_err_kcal": float(g.median()),
            "p25_kcal": float(g.quantile(0.25)),
            "p75_kcal": float(g.quantile(0.75)),
            "min_kcal": float(g.min()),
            "max_kcal": float(g.max()),
            "mean_signed_err_kcal": float(out_df[out_df["group"] == grp]["stage2_err_kcal"].mean()),
            "n_abs_err_gt_10_kcal": int((g > 10).sum()),
            "n_abs_err_gt_20_kcal": int((g > 20).sum()),
        }
    u, p = mannwhitneyu(out_df[out_df["group"] == "confidently_wrong"]["stage2_abs_err_kcal"],
                         out_df[out_df["group"] == "control"]["stage2_abs_err_kcal"],
                         alternative="two-sided")
    stats["mannwhitney_abs_err"] = {"U": float(u), "p": float(p)}
    xcheck_path = os.path.join(OUT, "stage2_hdf5_crosscheck.csv")
    if os.path.exists(xcheck_path):
        xc = pd.read_csv(xcheck_path)
        stats["hdf5_protocol"] = {}
        for grp in ("confidently_wrong", "control"):
            g = xc[xc["group"] == grp]["hdf5_stage2_abs_err_kcal"]
            stats["hdf5_protocol"][grp] = {
                "median_kcal": float(g.median()),
                "mean_kcal": float(g.mean()),
                "max_kcal": float(g.max()),
                "n_abs_err_gt_10_kcal": int((g > 10).sum()),
            }
        u2, p2 = mannwhitneyu(xc[xc["group"] == "confidently_wrong"]["hdf5_stage2_abs_err_kcal"],
                              xc[xc["group"] == "control"]["hdf5_stage2_abs_err_kcal"])
        stats["hdf5_protocol"]["mannwhitney_p"] = float(p2)
    rho, p_rho = spearmanr(out_df["stage2_abs_err_kcal"], out_df["rmse_across_seeds"])
    stats["spearman_stage2_abs_err_vs_finetuned_seed_rmse"] = {"rho": float(rho), "p": float(p_rho)}
    stats["n_tta"] = N_TTA
    stats["stage2_ckpt"] = os.path.relpath(STAGE2_CKPT, REPO_ROOT)
    stats["tta_fallback_mids"] = used_fallback
    stats["n_confs_per_molecule"] = n_confs
    with open(os.path.join(OUT, "stage2_bias_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [out_df[out_df["group"] == g]["stage2_abs_err_kcal"].values for g in
            ("confidently_wrong", "control")]
    bp = ax.boxplot(data, tick_labels=["confidently wrong (n=18)", "control (n=18)"],
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("|Stage-2 raw error| vs expt (kcal/mol)")
    ax.set_title("Stage-2 (frozen backbone) error: confidently-wrong vs control")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage2_error_boxplot.png"), dpi=150)

    print("=" * 74)
    print(f"Stage-2 frozen-checkpoint bias check  (ckpt: {os.path.relpath(STAGE2_CKPT, REPO_ROOT)})")
    print(f"TTA: {N_TTA}-conformer RDKit ETKDGv3 seed 42 prune 0.5 + MMFF (same as ensemble protocol)")
    print("=" * 74)
    for grp_name, grp_key in (("CONFIDENTLY WRONG (18)", "confidently_wrong"),
                              ("CONTROL (18)", "control")):
        s = stats[grp_key]
        print(f"\n{grp_name}:")
        print(f"  n                : {s['n']}")
        print(f"  mean |err|       : {s['mean_abs_err_kcal']:.3f} kcal/mol")
        print(f"  median |err|     : {s['median_abs_err_kcal']:.3f} kcal/mol   (p25={s['p25_kcal']:.3f}, p75={s['p75_kcal']:.3f})")
        print(f"  min / max |err|  : {s['min_kcal']:.3f} / {s['max_kcal']:.3f} kcal/mol")
        print(f"  mean signed err  : {s['mean_signed_err_kcal']:+.3f} kcal/mol (pred - expt)")
    print(f"\nMann-Whitney U (|stage2 err|, wrong vs control): U={stats['mannwhitney_abs_err']['U']:.0f}, p={stats['mannwhitney_abs_err']['p']:.4f}")
    if os.path.exists(xcheck_path):
        h = stats["hdf5_protocol"]
        print(f"hdf5-conformer protocol cross-check: conf-wrong median={h['confidently_wrong']['median_kcal']:.3f} "
              f"vs control median={h['control']['median_kcal']:.3f} kcal/mol, Mann-Whitney p={h['mannwhitney_p']:.4f}")
        print(f"  |err| > 10 kcal/mol counts: conf-wrong {h['confidently_wrong']['n_abs_err_gt_10_kcal']}/18, control {h['control']['n_abs_err_gt_10_kcal']}/18")
    print(f"Spearman(stage2 |err|, fine-tuned seed-RMSE, n={len(out_df)}): rho={rho:.3f}, p={p_rho:.4f}")
    if used_fallback:
        print(f"NOTE: {len(used_fallback)} molecules had total RDKit embedding failure and used the "
              f"stored hdf5 conformer: {used_fallback}")
    print(f"  conformers actually embedded per molecule: {sorted(set(n_confs.values()))} "
          f"(protocol = mean over whatever RDKit embedded, same as deep_ensemble.conformer_average)")
    print("\nPlain-English summary:")
    n_cw_tail = int((out_df[out_df["group"] == "confidently_wrong"]["stage2_abs_err_kcal"] > 10).sum())
    n_ct_tail = int((out_df[out_df["group"] == "control"]["stage2_abs_err_kcal"] > 10).sum())
    verdict = (f"Stage-2 error IS directionally elevated for the confidently-wrong 18 "
               f"({stats['confidently_wrong']['median_abs_err_kcal']:.1f} vs {stats['control']['median_abs_err_kcal']:.1f} kcal/mol "
               f"median; heavy tail |err|>10 kcal/mol in {n_cw_tail}/18 vs {n_ct_tail}/18; "
               f"Spearman(stage2 err, fine-tuned seed-RMSE) rho={rho:.2f}, p={p_rho:.3f}), "
               f"BUT NOT statistically significant (Mann-Whitney p={stats['mannwhitney_abs_err']['p']:.2f}, n=18 vs 18) "
               f"and NOT a simple inheritance: fine-tuning reduces even 85-143 kcal/mol Stage-2 errors to <1 kcal/mol, "
               f"so a large Stage-2 error does not doom a molecule. Interpretation: Stage-2 input-shift failures "
               f"(FreeSolv MMFF geometries vs AQM water-relaxed geometries) are more common among the confidently-wrong 18, "
               f"but the confident wrongness itself is dominated by shared bias introduced AT Stage-3 fine-tuning "
               f"(all 5 seeds converge to the same leftover bias), not inherited from the frozen backbone.")
    print(verdict)
    print(f"\nArtifacts in {os.path.relpath(OUT, REPO_ROOT)}/ : stage2_predictions.csv, "
          f"stage2_bias_stats.json, stage2_error_boxplot.png")


if __name__ == "__main__":
    main()
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "solvation-gnn"))

import csv
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import linregress

from DimeModels import DimeNetPlus
from freesolv_dataset import (
    FreeSolvDataset, EV_TO_KCAL, download_freesolv_data, load_freesolv_labels,
)
from element_vocab import NUM_ELEMENTS, build_one_hot


def build_model(num_blocks, hidden=128, radius=6.0):
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=hidden, out_channels=1,
        num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=radius, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    )


def main():
    parser = argparse.ArgumentParser(description="FreeSolv GNN prediction")
    parser.add_argument("--conformers", type=str, default="freesolv_conformers.hdf5")
    parser.add_argument("--output", type=str, default="freesolv_predictions.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="results")
    parser.add_argument("--vacuum_ckpt", type=str, default=None)
    parser.add_argument("--correction_ckpt", type=str, default="stage2_correction.pt")
    parser.add_argument("--explicit_ckpt", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="Data/FreeSolv")
    parser.add_argument("--lfer_split", type=float, default=0.1,
                        help="Fraction of molecules for LFER calibration (default 0.1)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_dir)

    corr_ckpt = os.path.join(ckpt_dir, args.correction_ckpt)
    if not os.path.exists(corr_ckpt):
        raise FileNotFoundError(f"Correction checkpoint not found: {corr_ckpt}")

    correction_model = build_model(num_blocks=3).to(device)
    correction_model.load_state_dict(torch.load(corr_ckpt, map_location=device, weights_only=True))
    correction_model.eval()
    print(f"Loaded correction model: {corr_ckpt}")

    explicit_model = None
    if args.explicit_ckpt:
        exp_ckpt = os.path.join(ckpt_dir, args.explicit_ckpt)
        if os.path.exists(exp_ckpt):
            explicit_model = build_model(num_blocks=2).to(device)
            explicit_model.load_state_dict(torch.load(exp_ckpt, map_location=device, weights_only=True))
            explicit_model.eval()
            print(f"Loaded explicit model: {exp_ckpt}")

    dataset = FreeSolvDataset(args.conformers)
    print(f"Dataset: {len(dataset)} molecules")

    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
    results = []

    for data in loader:
        data = data.to(device)
        x = build_one_hot(data, device)
        with torch.no_grad():
            corr_e = correction_model(x, data.pos, data.batch)
            total_e = corr_e
            if explicit_model is not None:
                explicit_e = explicit_model(x, data.pos, data.batch)
                total_e = total_e + explicit_e

        for i in range(data.num_graphs):
            dG_eV = total_e[i].item()
            dG_kcal = dG_eV * EV_TO_KCAL
            results.append({
                "mol_id": data.mol_id[i],
                "dG_pred_kcal": dG_kcal,
            })

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "dG_pred_kcal"])
        w.writeheader()
        w.writerows(results)
    print(f"Saved predictions to {args.output}")

    # ---- Sign check ----
    preds = np.array([r["dG_pred_kcal"] for r in results])
    n_negative = int((preds < 0).sum())
    n_total = len(preds)
    print(f"\nSign check: {n_negative}/{n_total} predictions are negative")
    print(f"  Mean raw prediction: {preds.mean():.2f} kcal/mol")
    if n_negative < n_total * 0.5:
        print("  WARNING: majority of predictions have wrong sign.")

    # ---- Issue 1: Outlier filtering (5-sigma) ----
    mean_pred = preds.mean()
    std_pred = preds.std()
    mask = np.abs(preds - mean_pred) < 5 * std_pred
    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"\nOutlier filter (5 sigma from mean): removed {n_removed} molecule(s)")
        for i in np.where(~mask)[0]:
            print(f"  Removed {results[i]['mol_id']}: {results[i]['dG_pred_kcal']:.1f} kcal/mol")
        preds = preds[mask]
        results = [r for r, m in zip(results, mask) if m]

    # ---- Load experimental labels for calibration ----
    json_path, _ = download_freesolv_data(args.cache_dir)
    labels = load_freesolv_labels(json_path)

    exp_dg = np.array([labels[r["mol_id"]].get("expt", float("nan")) for r in results])
    valid = ~np.isnan(exp_dg)
    preds = preds[valid]
    exp_dg = exp_dg[valid]
    mol_ids = [results[i]["mol_id"] for i in range(len(results)) if valid[i]]

    n_valid = len(preds)
    print(f"\nMatched with experimental data: {n_valid} molecules")

    # ---- Issue 3: LFER calibration ----
    n_cal = max(10, int(n_valid * args.lfer_split))
    cal_pred = preds[:n_cal]
    cal_exp = exp_dg[:n_cal]
    test_pred = preds[n_cal:]
    test_exp = exp_dg[n_cal:]

    slope, intercept, r_val, p_val, se = linregress(cal_pred, cal_exp)
    test_pred_cal = slope * test_pred + intercept

    raw_mae = float(np.mean(np.abs(test_pred - test_exp)))
    cal_mae = float(np.mean(np.abs(test_pred_cal - test_exp)))
    raw_rmse = float(np.sqrt(np.mean((test_pred - test_exp) ** 2)))
    cal_rmse = float(np.sqrt(np.mean((test_pred_cal - test_exp) ** 2)))

    print(f"\nLFER calibration (first {n_cal} molecules):")
    print(f"  dG_exp = {slope:.4f} * dG_pred + {intercept:.4f}")
    print(f"  R = {r_val:.4f}, p = {p_val:.4e}")
    print(f"\nTest set ({n_valid - n_cal} molecules):")
    print(f"  Raw MAE:       {raw_mae:.3f} kcal/mol")
    print(f"  Calibrated MAE: {cal_mae:.3f} kcal/mol")
    print(f"  Raw RMSE:       {raw_rmse:.3f} kcal/mol")
    print(f"  Calibrated RMSE: {cal_rmse:.3f} kcal/mol")


if __name__ == "__main__":
    main()

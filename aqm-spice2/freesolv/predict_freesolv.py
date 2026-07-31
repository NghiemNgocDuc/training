import sys
import os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_parent)

import csv
import json
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import linregress, kendalltau

from DimeModels import DimeNetPlus
from freesolv_dataset import (
    FreeSolvDataset, EV_TO_KCAL, download_freesolv_data, load_freesolv_labels,
)
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import load_reference_energies, compute_molecular_reference


def build_model(num_blocks, hidden=128, radius=6.0):
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=hidden, out_channels=1,
        num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=radius, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    )


def compute_metrics(y_true, y_pred, label):
    residuals = y_true - y_pred
    n = len(y_true)
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    tau, p_val = kendalltau(y_true, y_pred)
    return {
        "method": label,
        "n": n,
        "MAE_kcal": mae,
        "RMSE_kcal": rmse,
        "R2": r2,
        "Kendall_tau": float(tau),
        "Kendall_p": float(p_val),
    }


def main():
    parser = argparse.ArgumentParser(description="FreeSolv GNN prediction")
    parser.add_argument("--conformers", type=str, default="freesolv_conformers.hdf5",
                        help="Path to FreeSolv conformer HDF5")
    parser.add_argument("--output", type=str, default="freesolv_predictions.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="results")
    parser.add_argument("--vacuum_ckpt", type=str, default=None,
                        help="Stage 1 vacuum checkpoint (required for Option A or explicit)")
    parser.add_argument("--correction_ckpt", type=str, default="stage2_correction.pt",
                        help="Stage 2a correction checkpoint (Option B)")
    parser.add_argument("--option_a_ckpt", type=str, default=None,
                        help="Option A scratch model checkpoint (for Option A comparison)")
    parser.add_argument("--explicit_ckpt", type=str, default=None,
                        help="Stage 2b explicit model (experimental, needs --vacuum_ckpt)")
    parser.add_argument("--method", type=str, default="B",
                        choices=["A", "B", "all"],
                        help="Prediction method: B=correction-only, A=vacuum+OptionA, all=both")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="Data/FreeSolv")
    parser.add_argument("--lfer_split", type=float, default=0.1,
                        help="Fraction of molecules for LFER calibration (0 to disable)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_dir)
    print(f"Checkpoint directory: {ckpt_dir}")

    # ── Load atomic references (needed for Option A) ──
    ref_path = os.path.join(ckpt_dir, "atomic_references.json")
    ref_energies = None
    if os.path.exists(ref_path):
        ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
        print(f"Loaded atomic reference energies from {ref_path}")
    else:
        print("No atomic_references.json found")

    # ── Load models ──
    vacuum_model = None
    if args.vacuum_ckpt:
        vac_path = os.path.join(ckpt_dir, args.vacuum_ckpt)
        if os.path.exists(vac_path):
            vacuum_model = build_model(num_blocks=4).to(device)
            vacuum_model.load_state_dict(torch.load(vac_path, map_location=device, weights_only=True))
            for p in vacuum_model.parameters():
                p.requires_grad_(False)
            vacuum_model.eval()
            print(f"Loaded vacuum model: {vac_path}")
        else:
            print(f"Warning: vacuum checkpoint not found: {vac_path}")

    option_a_model = None
    if args.option_a_ckpt:
        oa_path = os.path.join(ckpt_dir, args.option_a_ckpt)
        if os.path.exists(oa_path):
            option_a_model = build_model(num_blocks=3).to(device)
            option_a_model.load_state_dict(torch.load(oa_path, map_location=device, weights_only=True))
            option_a_model.eval()
            print(f"Loaded Option A model: {oa_path}")
        else:
            print(f"Warning: Option A checkpoint not found: {oa_path}")

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
            if vacuum_model is None:
                print("  WARNING: explicit model used without vacuum backbone. "
                      "Predictions will be unreliable.")
            print("  WARNING: Stage 2b was trained on SPICE2 solute-water clusters,"
                  " not single molecules.")
            print("  Results on FreeSolv will be physically meaningless"
                  " (~1000 kcal/mol error).")
        else:
            print(f"Warning: explicit checkpoint not found: {exp_ckpt}")

    # ── Load dataset and labels ──
    json_path, _ = download_freesolv_data(args.cache_dir)
    labels = load_freesolv_labels(json_path)

    dataset = FreeSolvDataset(args.conformers)
    print(f"Dataset: {len(dataset)} molecules")

    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    # ── Run inference ──
    all_results = []
    for data in loader:
        data = data.to(device)
        x = build_one_hot(data, device)
        with torch.no_grad():
            corr_e = correction_model(x, data.pos, data.batch)
            dG_B = corr_e * EV_TO_KCAL  # Option B: correction = ΔG

            dG_A = None
            if option_a_model is not None and vacuum_model is not None:
                vac_e = vacuum_model(x, data.pos, data.batch)
                oa_e = option_a_model(x, data.pos, data.batch)
                dG_A = (oa_e - vac_e) * EV_TO_KCAL

            dG_exp = None
            if explicit_model is not None and vacuum_model is not None:
                vac_e = vacuum_model(x, data.pos, data.batch) if vacuum_model else 0
                impl_e = correction_model(x, data.pos, data.batch)
                expl_e = explicit_model(x, data.pos, data.batch)
                total_e = vac_e + impl_e + expl_e
                dG_exp = (total_e - vac_e) * EV_TO_KCAL
            elif explicit_model is not None and vacuum_model is None:
                expl_e = explicit_model(x, data.pos, data.batch)
                dG_exp = (corr_e + expl_e) * EV_TO_KCAL

        for i in range(data.num_graphs):
            row = {
                "mol_id": data.mol_id[i],
                "dG_B_kcal": dG_B[i].item(),
            }
            if dG_A is not None:
                row["dG_A_kcal"] = dG_A[i].item()
            if dG_exp is not None:
                row["dG_explicit_kcal"] = dG_exp[i].item()
            expt = labels.get(data.mol_id[i], {}).get("expt", None)
            if expt is not None:
                row["dG_exp_kcal"] = expt
            all_results.append(row)

    print(f"\nInference complete: {len(all_results)} molecules")

    # ── Save CSV ──
    fieldnames = list(all_results[0].keys()) if all_results else []
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved predictions to {args.output}")

    # ── Evaluate each method ──
    methods_to_eval = []
    if args.method in ("B", "all"):
        methods_to_eval.append(("Option B (delta-learning)", "dG_B_kcal"))
    if args.method in ("A", "all") and "dG_A_kcal" in fieldnames:
        methods_to_eval.append(("Option A (scratch)", "dG_A_kcal"))
    if args.method == "all" and "dG_explicit_kcal" in fieldnames:
        methods_to_eval.append(("+ Explicit (Stage 2b)", "dG_explicit_kcal"))
    if "dG_A_kcal" not in fieldnames and "dG_B_kcal" in fieldnames:
        methods_to_eval.append(("Option B (recommended)", "dG_B_kcal"))

    for label, key in methods_to_eval:
        preds = np.array([r[key] for r in all_results])
        expts = np.array([r.get("dG_exp_kcal", np.nan) for r in all_results])
        valid = ~np.isnan(expts)
        preds = preds[valid]
        expts = expts[valid]
        n = len(preds)

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Molecules: {n}")
        print(f"  Mean pred: {preds.mean():.3f} kcal/mol")
        print(f"  Mean expt: {expts.mean():.3f} kcal/mol")
        print(f"  Pred std:  {preds.std():.3f} kcal/mol")
        print(f"  Expt std:  {expts.std():.3f} kcal/mol")

        n_neg = int((preds < 0).sum())
        print(f"  Sign agreement: {n_neg}/{n} negative ({n_neg/n*100:.1f}%)")

        metrics = compute_metrics(expts, preds, label)
        print(f"  MAE:  {metrics['MAE_kcal']:.3f} kcal/mol")
        print(f"  RMSE: {metrics['RMSE_kcal']:.3f} kcal/mol")
        print(f"  R2:   {metrics['R2']:.4f}")
        print(f"  Kendall tau: {metrics['Kendall_tau']:.4f} (p={metrics['Kendall_p']:.4e})")

        # LFER calibration
        if args.lfer_split > 0 and n >= 20:
            n_cal = max(10, int(n * args.lfer_split))
            n_test = n - n_cal
            slope, intercept, r_val, p_val, _ = linregress(preds[:n_cal], expts[:n_cal])
            test_pred_cal = slope * preds[n_cal:] + intercept
            test_exp = expts[n_cal:]
            raw_mae = float(np.mean(np.abs(preds[n_cal:] - test_exp)))
            cal_mae = float(np.mean(np.abs(test_pred_cal - test_exp)))
            print(f"  LFER (first {n_cal}/test {n_test}):")
            print(f"    dG_exp = {slope:.4f} * dG_pred + {intercept:.4f}  (R={r_val:.4f})")
            print(f"    Raw MAE:       {raw_mae:.3f} kcal/mol")
            print(f"    Calibrated MAE: {cal_mae:.3f} kcal/mol")

    # ── Comparison table if both methods available ──
    if args.method == "all" and len(methods_to_eval) >= 2:
        print(f"\n{'='*60}")
        print(f"  METHOD COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Method':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Kendall_tau':>10}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for label, key in methods_to_eval:
            preds = np.array([r[key] for r in all_results])
            expts = np.array([r.get("dG_exp_kcal", np.nan) for r in all_results])
            valid = ~np.isnan(expts)
            m = compute_metrics(expts[valid], preds[valid], label)
            print(f"  {m['method']:<30} {m['MAE_kcal']:>8.3f} {m['RMSE_kcal']:>8.3f} "
                  f"{m['R2']:>8.4f} {m['Kendall_tau']:>10.4f}")
        print()


if __name__ == "__main__":
    main()
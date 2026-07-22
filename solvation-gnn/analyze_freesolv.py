import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)

import argparse
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, linregress
from rdkit import Chem

from freesolv_dataset import download_freesolv_data, load_freesolv_labels


def parse_groups_from_labels(labels):
    groups = {}
    for mol_id, entry in labels.items():
        g = entry.get("groups")
        if g and len(g) > 0:
            groups[mol_id] = g[0]
    return groups


def evaluate_metrics(y_true, y_pred, label):
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
    parser = argparse.ArgumentParser(description="Analyze FreeSolv results")
    parser.add_argument("--predictions", type=str, default="freesolv_predictions.csv")
    parser.add_argument("--gbn2", type=str, default="freesolv_gbn2.csv")
    parser.add_argument("--conformers", type=str, default="freesolv_conformers.hdf5")
    parser.add_argument("--labels", type=str, default="freesolv_labels.csv")
    parser.add_argument("--cache_dir", type=str, default="Data/FreeSolv")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_summary", type=str, default="evaluation_summary.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(pred_df)} predictions from {args.predictions}")

    # Load GBNeck2 baseline
    gbn2_df = pd.read_csv(args.gbn2)
    print(f"Loaded {len(gbn2_df)} GBNeck2 baselines from {args.gbn2}")

    # Load FreeSolv labels + groups
    json_path, groups_path = download_freesolv_data(args.cache_dir)
    labels = load_freesolv_labels(json_path)

    groups = parse_groups_from_labels(labels)
    print(f"Loaded {len(groups)} functional groups from database")

    # Load label CSV for in_model_vocab flag
    label_df = pd.read_csv(args.labels)

    # Merge everything on mol_id
    merged = pred_df.merge(gbn2_df, on="mol_id", how="inner")
    merged = merged.merge(label_df[["mol_id", "in_model_vocab"]], on="mol_id", how="left")

    # Add experimental values from labels
    expt_list = []
    for mol_id in merged["mol_id"]:
        entry = labels.get(mol_id, {})
        expt_list.append(entry.get("expt", float("nan")))
    merged["dG_exp_kcal"] = expt_list
    merged = merged.dropna(subset=["dG_exp_kcal"])

    print(f"Evaluating on {len(merged)} molecules with experimental data")

    # Only evaluate molecules in model vocabulary
    merged_model = merged[merged["in_model_vocab"] == True].copy()
    print(f"  Model-vocab subset: {len(merged_model)} molecules")
    merged_other = merged[merged["in_model_vocab"] != True].copy()
    if len(merged_other) > 0:
        print(f"  Excluded from analysis: {len(merged_other)} (not in model vocab)")

    y_exp = merged_model["dG_exp_kcal"].values
    y_option_b = merged_model["dG_pred_kcal"].values
    y_gbn2 = merged_model["dG_GBn2_kcal"].values

    # ---- Issue 1: 5-sigma outlier removal ----
    mean_b = y_option_b.mean()
    std_b = y_option_b.std()
    mask = np.abs(y_option_b - mean_b) < 5 * std_b
    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"\nOutlier filter (5 sigma): removed {n_removed} molecule(s)")
        y_option_b = y_option_b[mask]
        y_exp = y_exp[mask]
        y_gbn2 = y_gbn2[mask]

    # ---- Issue 3: LFER calibration for GNN prediction ----
    n_cal = max(10, int(len(y_option_b) * 0.1))
    slope, intercept, r_lfer, p_lfer, _ = linregress(y_option_b[:n_cal], y_exp[:n_cal])
    test_b_raw = y_option_b[n_cal:]
    test_b_cal = slope * test_b_raw + intercept
    test_exp = y_exp[n_cal:]
    raw_mae = float(np.mean(np.abs(test_b_raw - test_exp)))
    cal_mae = float(np.mean(np.abs(test_b_cal - test_exp)))
    print(f"\nLFER calibration (first {n_cal} molecules):")
    print(f"  dG_exp = {slope:.4f} * dG_pred + {intercept:.4f}  (R={r_lfer:.4f})")
    print(f"  Test set raw MAE:       {raw_mae:.3f} kcal/mol")
    print(f"  Test set calibrated MAE: {cal_mae:.3f} kcal/mol")

    # Use full arrays for remaining metrics (report raw GNN)
    metrics_b = evaluate_metrics(y_exp, y_option_b, "Option B (delta-learning)")
    metrics_gbn2 = evaluate_metrics(y_exp, y_gbn2, "GBNeck2 (classical)")

    # Scatter plot: Predicted vs Experimental
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (y_pred, label, color) in zip(
        axes,
        [
            (y_option_b, "Option B (delta-learning)", "#1f77b4"),
            (y_gbn2, "GBNeck2", "#ff7f0e"),
        ],
    ):
        ax.scatter(y_exp, y_pred, alpha=0.5, s=15, color=color)
        lims = [
            min(min(y_exp), min(y_pred)) - 1,
            max(max(y_exp), max(y_pred)) + 1,
        ]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental ΔG (kcal/mol)")
        ax.set_ylabel("Predicted ΔG (kcal/mol)")
        ax.set_title(label)
        ax.axis("equal")

    plt.tight_layout()
    scatter_path = os.path.join(args.output_dir, "freesolv_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    # Functional group analysis
    merged_model["functional_group"] = merged_model["mol_id"].map(groups)
    fg_data = merged_model.dropna(subset=["functional_group"])

    if len(fg_data) > 0:
        group_stats = (
            fg_data.groupby("functional_group")
            .apply(
                lambda g: pd.Series({
                    "n_molecules": len(g),
                    "MAE_OptionB": (g["dG_pred_kcal"] - g["dG_exp_kcal"]).abs().mean(),
                    "MAE_GBNeck2": (g["dG_GBn2_kcal"] - g["dG_exp_kcal"]).abs().mean(),
                })
            )
            .sort_values("MAE_OptionB")
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(group_stats) * 0.3)))
        y_pos = range(len(group_stats))
        ax.barh(y_pos, group_stats["MAE_OptionB"].values, height=0.4,
                label="Option B", color="#1f77b4", alpha=0.8)
        ax.barh([p + 0.4 for p in y_pos], group_stats["MAE_GBNeck2"].values,
                height=0.4, label="GBNeck2", color="#ff7f0e", alpha=0.8)
        ax.set_yticks([p + 0.2 for p in y_pos])
        ax.set_yticklabels(group_stats.index, fontsize=8)
        ax.set_xlabel("MAE (kcal/mol)")
        ax.set_title("MAE by Functional Group (FreeSolv)")
        ax.legend()
        plt.tight_layout()
        fg_path = os.path.join(args.output_dir, "freesolv_functional_groups.png")
        plt.savefig(fg_path, dpi=150)
        plt.close()
        print(f"Saved functional group plot: {fg_path}")

        print("\nTop 5 best functional groups (Option B):")
        for name, row in group_stats.head(5).iterrows():
            print(f"  {name:<25} MAE={row['MAE_OptionB']:.3f}  n={int(row['n_molecules'])}")
        print("\nTop 5 worst functional groups (Option B):")
        for name, row in group_stats.tail(5).iterrows():
            print(f"  {name:<25} MAE={row['MAE_OptionB']:.3f}  n={int(row['n_molecules'])}")
    else:
        print("No functional group data available (groups.txt may not be found)")

    # Molecular size bias check
    smiles_list = []
    for mol_id in merged_model["mol_id"]:
        entry = labels.get(mol_id, {})
        smiles_list.append(entry.get("smiles", ""))
    merged_model["smiles"] = smiles_list

    n_heavy = []
    for smi in merged_model["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        n_heavy.append(mol.GetNumHeavyAtoms() if mol else 0)
    merged_model["n_heavy_atoms"] = n_heavy
    merged_model["error"] = merged_model["dG_pred_kcal"] - merged_model["dG_exp_kcal"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged_model["n_heavy_atoms"], merged_model["error"],
               alpha=0.5, s=15, color="#1f77b4")
    ax.axhline(0, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("Number of heavy atoms")
    ax.set_ylabel("Predicted - Experimental ΔG (kcal/mol)")
    ax.set_title("Prediction error vs molecular size")
    plt.tight_layout()
    size_path = os.path.join(args.output_dir, "freesolv_size_bias.png")
    plt.savefig(size_path, dpi=150)
    plt.close()
    print(f"Saved size bias plot: {size_path}")

    r, p = pearsonr(merged_model["n_heavy_atoms"], merged_model["error"])
    print(f"Size-error correlation: r={r:.3f}, p={p:.4f}")
    if abs(r) > 0.3 and p < 0.05:
        print("  WARNING: significant size bias detected in predictions")

    # Save joined results CSV
    results_path = os.path.join(args.output_dir, "freesolv_results.csv")
    merged_model.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")

    # Sign convention
    n_negative = int((y_option_b < 0).sum())
    n_total_b = len(y_option_b)
    sign_agreement = n_negative / n_total_b if n_total_b > 0 else 0.0

    # Write evaluation summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("FreeSolv Benchmark Evaluation Summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"")
    summary_lines.append(f"Dataset: {len(merged_model)} molecules evaluated")
    summary_lines.append(f"  Total compatible: {len(merged_model)}")
    summary_lines.append(f"  Total FreeSolv:   {len(labels)}")
    summary_lines.append(f"")
    summary_lines.append(f"Mean experimental DeltaG: {y_exp.mean():.2f} kcal/mol")
    summary_lines.append(f"Sign agreement (Option B): {sign_agreement:.1%}")
    summary_lines.append(f"")

    all_metrics = [metrics_b, metrics_gbn2]
    summary_lines.append(f"{'Method':<35} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Kendall_tau':>10} {'n':>6}")
    summary_lines.append(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
    for m in all_metrics:
        summary_lines.append(
            f"{m['method']:<35} {m['MAE_kcal']:>8.3f} {m['RMSE_kcal']:>8.3f} "
            f"{m['R2']:>8.4f} {m['Kendall_tau']:>10.4f} {m['n']:>6}"
        )
    summary_lines.append(f"")
    summary_lines.append("Target benchmarks for zero-shot QM-trained model:")
    summary_lines.append("  MAE < 2.0 kcal/mol: competitive with classical implicit solvent")
    summary_lines.append("  MAE < 1.5 kcal/mol: strong result")
    summary_lines.append("  MAE < 1.0 kcal/mol: exceptional, better than classical GB")
    summary_lines.append(f"")

    # GAFF reference from FreeSolv
    gaff_preds = []
    gaff_expts = []
    for mol_id in merged_model["mol_id"]:
        entry = labels.get(mol_id, {})
        calc = entry.get("calc")
        expt = entry.get("expt")
        if calc is not None and expt is not None:
            gaff_preds.append(calc)
            gaff_expts.append(expt)
    if gaff_preds:
        gaff_preds = np.array(gaff_preds)
        gaff_expts = np.array(gaff_expts)
        gaff_residuals = gaff_expts - gaff_preds
        gaff_mae = np.mean(np.abs(gaff_residuals))
        gaff_rmse = np.sqrt(np.mean(gaff_residuals ** 2))
        summary_lines.append(f"GAFF reference (from FreeSolv DB):")
        summary_lines.append(f"  MAE  = {gaff_mae:.3f} kcal/mol")
        summary_lines.append(f"  RMSE = {gaff_rmse:.3f} kcal/mol")
        summary_lines.append(f"")

    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, args.output_summary)
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSaved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()

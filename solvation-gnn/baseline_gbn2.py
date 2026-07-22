import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from tqdm import tqdm

from freesolv_dataset import FreeSolvDataset, download_freesolv_data, load_freesolv_labels

SOLUTE_DIELECTRIC = 1.0
SOLVENT_DIELECTRIC = 78.5

# Amber mbondi3 radii (Angstrom) and scale factors
MBONDI3_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.50,
    9: 1.50,
    15: 1.85,
    16: 1.80,
    17: 1.70,
    35: 1.85,
}

SCALE_FACTORS = {
    1: 0.85, 6: 0.72, 7: 0.79, 8: 0.85,
    9: 0.88, 15: 0.86, 16: 0.96, 17: 0.80, 35: 0.80,
}

# Standard Coulomb constant: e^2/(4πε0) in kcal·Å·mol⁻¹·e⁻²
_KCAL_ANG_PER_E2 = 332.0636


def compute_gb_energy(
    charges: torch.Tensor,
    born_radii: torch.Tensor,
    positions: torch.Tensor,
    eps_solute: float = SOLUTE_DIELECTRIC,
    eps_solvent: float = SOLVENT_DIELECTRIC,
) -> torch.Tensor:
    """Still GB energy with Born radii (GBNeck2-style).

    dG = -0.5 * (1/eps_solute - 1/eps_solvent)
         * [sum_i q_i^2/B_i + sum_{i<j} q_i q_j / f_ij]

    where f_ij = sqrt(r_ij^2 + B_i B_j exp(-r_ij^2 / 4 B_i B_j)).
    """
    dist = torch.cdist(positions, positions).clamp(min=1e-4)

    born_self = torch.sum(charges**2 / born_radii)

    B_prod = (born_radii.unsqueeze(1) * born_radii.unsqueeze(0)).clamp(min=1e-8)
    f_ij = torch.sqrt(dist**2 + B_prod * torch.exp(-dist**2 / (4.0 * B_prod)))
    pair_energy = (charges.unsqueeze(1) * charges.unsqueeze(0) / f_ij).triu(1).sum()

    prefactor = -0.5 * (1.0 / eps_solute - 1.0 / eps_solvent) * _KCAL_ANG_PER_E2
    return prefactor * (born_self + pair_energy)





def assign_gasteiger_charges(mol: Chem.Mol) -> List[float]:
    mol = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return [
        a.GetDoubleProp("_GasteigerCharge") if a.HasProp("_GasteigerCharge") else 0.0
        for a in mol.GetAtoms()
    ]


def evaluate(preds: np.ndarray, expts: np.ndarray) -> dict:
    """Raw metrics + linearly-scaled metrics."""
    mae_raw = np.mean(np.abs(preds - expts))
    rmse_raw = np.sqrt(np.mean((preds - expts)**2))

    A = np.vstack([preds, np.ones_like(preds)]).T
    a, b, *_ = np.linalg.lstsq(A, expts, rcond=None)[0]
    pred_scaled = a * preds + b
    mae_scaled = np.mean(np.abs(pred_scaled - expts))
    rmse_scaled = np.sqrt(np.mean((pred_scaled - expts)**2))
    r2_scaled = 1 - np.sum((expts - pred_scaled)**2) / np.sum((expts - expts.mean())**2)

    # Kendall tau rank correlation
    from scipy.stats import kendalltau
    tau, p_kendall = kendalltau(preds, expts)

    return {
        "N": len(preds),
        "mean_pred_raw": float(preds.mean()),
        "mean_expt": float(expts.mean()),
        "MAE_raw": float(mae_raw),
        "RMSE_raw": float(rmse_raw),
        "linear_slope": float(a),
        "linear_intercept": float(b),
        "MAE_scaled": float(mae_scaled),
        "RMSE_scaled": float(rmse_scaled),
        "R2_scaled": float(r2_scaled),
        "Kendall_tau": float(tau),
        "Kendall_p": float(p_kendall),
    }


def predict_freesolv_gb(
    conformer_hdf5: str = "freesolv_conformers.hdf5",
    output_csv: str = "freesolv_gbn2.csv",
):
    """Compute GB baseline for all FreeSolv molecules, evaluate against experiment."""

    json_path, _ = download_freesolv_data("Data/FreeSolv")
    labels = load_freesolv_labels(json_path)
    ds = FreeSolvDataset(conformer_hdf5)
    mol_ids = ds.get_mol_ids()

    preds, expts_arr = [], []
    rows = []

    for idx, mol_id in enumerate(tqdm(mol_ids, desc="GBNeck baseline")):
        data = ds[idx]
        smiles = labels[mol_id]["smiles"]
        expt = float(labels[mol_id].get("expt", labels[mol_id].get("dG_expt", labels[mol_id].get("dG", 0))))
        z_hdf5 = data.z.numpy()
        pos = data.pos.numpy()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        z_smiles = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        if len(z_smiles) != len(z_hdf5) or not np.array_equal(z_smiles, z_hdf5):
            continue

        charges = assign_gasteiger_charges(mol)

        charges_t = torch.tensor(charges, dtype=torch.float)
        intrinsic_radii = torch.tensor(
            [MBONDI3_RADII.get(int(ai), 1.70) for ai in z_hdf5], dtype=torch.float)
        scale = torch.tensor(
            [SCALE_FACTORS.get(int(ai), 0.80) for ai in z_hdf5], dtype=torch.float)
        born_radii = intrinsic_radii * scale
        pos_t = torch.tensor(pos, dtype=torch.float)

        dG = compute_gb_energy(charges_t, born_radii, pos_t).item()
        preds.append(dG)
        expts_arr.append(expt)
        rows.append({"mol_id": mol_id, "dG_GBn2_kcal": dG, "dG_expt_kcal": expt})

    preds = np.array(preds)
    expts_arr = np.array(expts_arr)

    # Sign convention check: experimental dG < 0 = favorable solvation
    n_bad_sign = int((preds > 1e-6).sum())
    if n_bad_sign > 0:
        print(f"  WARNING: {n_bad_sign}/{len(preds)} predictions have wrong sign (positive, destabilizing)")
    else:
        print(f"  Sign convention OK (all {len(preds)} predictions <= 0, negative = favorable)")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "dG_GBn2_kcal", "dG_expt_kcal"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved GB baseline to {output_csv}")

    metrics = evaluate(preds, expts_arr)
    print(f"\nGB baseline vs FreeSolv experiment ({metrics['N']} molecules):")
    print(f"  Mean pred (raw): {metrics['mean_pred_raw']:.2f} kcal/mol")
    print(f"  Mean expt:       {metrics['mean_expt']:.2f} kcal/mol")
    print(f"  RMSE raw:   {metrics['RMSE_raw']:.3f} kcal/mol")
    print(f"  MAE raw:    {metrics['MAE_raw']:.3f} kcal/mol")
    print(f"  Linear fit: dG_expt = {metrics['linear_slope']:.4f} * dG_pred + {metrics['linear_intercept']:.4f}")
    print(f"  RMSE scaled:  {metrics['RMSE_scaled']:.3f} kcal/mol")
    print(f"  MAE scaled:   {metrics['MAE_scaled']:.3f} kcal/mol")
    print(f"  R^2 scaled:   {metrics['R2_scaled']:.3f}")
    print(f"  Kendall tau:  {metrics['Kendall_tau']:.4f} (p={metrics['Kendall_p']:.2e})")

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", type=str, default="freesolv_conformers.hdf5")
    parser.add_argument("--output", type=str, default="freesolv_gbn2.csv")
    args = parser.parse_args()
    predict_freesolv_gb(args.conformers, output_csv=args.output)

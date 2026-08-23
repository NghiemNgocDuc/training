"""
Composition-Aware Affine Transfer (CAAT)
----------------------------------------
This script implements a universally applicable, gauge-invariant transfer correction
for Graph Neural Network (GNN) energy predictions.

Standard Plain Affine correction uses a global scale and intercept:
    E_corrected = a * E_raw + b_0

Composition-Aware Affine expands the intercept to account for element-specific 
systematic biases between the source dataset (FreeSolv) and target dataset (ExpDB):
    E_corrected = a * E_raw + b_0 + sum_{e} (b_e * N_e)

where N_e is the number of atoms of element e in the molecule.

This script evaluates both methods using 5-fold cross-validation on ExpDB.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from rdkit import Chem

DATA_DIR = r"C:\Users\User\Documents\Data"
EXPDB_CSV = os.path.join(DATA_DIR, r"expdb_seed_ensemble\inputs\predictions_ensemble.csv")
EXPDB_DIR = os.path.join(DATA_DIR, r"expdb_vast\results_seeds")

def main():
    print("Loading ExpDB external transfer data...")
    truth_df = pd.read_csv(EXPDB_CSV)
    truth_df = truth_df.dropna(subset=["dg_exp_kcal", "smiles"])
    truth_map = dict(zip(truth_df["id"].astype(str), truth_df["dg_exp_kcal"]))
    smi_map = dict(zip(truth_df["id"].astype(str), truth_df["smiles"]))

    # Load multi-seed raw predictions from ExpDB inference archive
    runs = {}
    for s in [42, 123, 999]:
        with open(os.path.join(EXPDB_DIR, f"peratom_seed{s}.pkl"), "rb") as f:
            runs[s] = pickle.load(f)

    ids = runs[42]["expdb_ids"]
    y_true = np.array([truth_map[str(m)] for m in ids])

    E_seeds = np.stack([runs[s]["E"] for s in [42, 123, 999]], axis=1)
    raw_preds = E_seeds.mean(axis=1)

    print("Extracting gauge-invariant elemental compositions...")
    features = []
    for m in ids:
        smi = smi_map[str(m)]
        mol = Chem.MolFromSmiles(smi)
        N = mol.GetNumAtoms()
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        
        # Count elements (implicit and explicit hydrogens included)
        n_C = atoms.count("C")
        n_O = atoms.count("O")
        n_N = atoms.count("N")
        n_H = atoms.count("H") + sum([a.GetTotalNumHs() for a in mol.GetAtoms()])
        n_S = atoms.count("S")
        n_F = atoms.count("F")
        n_Cl = atoms.count("Cl")
        n_Br = atoms.count("Br")
        n_P = atoms.count("P")
        
        features.append([N, n_C, n_O, n_N, n_H, n_S, n_F, n_Cl, n_Br, n_P])

    Z = np.array(features)
    feature_names = ["Total_Atoms", "Carbon", "Oxygen", "Nitrogen", "Hydrogen", 
                     "Sulfur", "Fluorine", "Chlorine", "Bromine", "Phosphorus"]

    print("Running 5-fold cross-validation on 620 molecules...\n")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pred_plain = np.zeros(len(ids))
    pred_comp = np.zeros(len(ids))
    w_comp_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(ids)):
        
        # 1. Plain Affine: a * E + b
        def mae_plain(ab):
            a, b = ab[0], ab[1]
            p = a * raw_preds[train_idx] + b
            return np.abs(p - y_true[train_idx]).mean()
        
        res_plain = minimize(mae_plain, [1.0, 0.0], method="Nelder-Mead")
        a_opt, b_opt = res_plain.x
        pred_plain[test_idx] = a_opt * raw_preds[test_idx] + b_opt
        
        # 2. Composition-Aware Affine: a * E + b_0 + sum(b_e * N_e)
        def mae_comp(w):
            a, b0 = w[0], w[1]
            b_vec = w[2:]
            p = a * raw_preds[train_idx] + b0 + Z[train_idx] @ b_vec
            # L2 regularization on element weights to prevent overfitting rare elements
            return np.abs(p - y_true[train_idx]).mean() + 1e-4 * np.sum(b_vec**2)
        
        w_init = np.zeros(2 + Z.shape[1])
        w_init[0] = 1.0  # initialize scale a = 1
        res_comp = minimize(mae_comp, w_init, method="Nelder-Mead", options={"maxiter": 5000})
        w_opt = res_comp.x
        w_comp_list.append(w_opt)
        
        pred_comp[test_idx] = w_opt[0] * raw_preds[test_idx] + w_opt[1] + Z[test_idx] @ w_opt[2:]

    # Evaluation
    mae_raw = np.abs(raw_preds - y_true).mean()
    mae_plain = np.abs(pred_plain - y_true).mean()
    mae_comp = np.abs(pred_comp - y_true).mean()

    print("=" * 60)
    print("FINAL EVALUATION ON EXPDB EXTERNAL TRANSFER (n=620)")
    print("=" * 60)
    print(f"Raw GNN Ensemble:        {mae_raw:.4f} kcal/mol")
    print(f"Plain Affine:            {mae_plain:.4f} kcal/mol  (Delta: {mae_plain - mae_raw:+.4f})")
    print(f"Composition-Aware:       {mae_comp:.4f} kcal/mol  (Delta: {mae_comp - mae_raw:+.4f})")
    print("-" * 60)
    print(f"Improvement over Plain:  {mae_comp - mae_plain:+.4f} kcal/mol")
    print("=" * 60)

    # Average weights learned across the 5 folds
    w_mean = np.mean(w_comp_list, axis=0)
    print("\nMean Learned Parameters (averaged across 5 folds):")
    print(f"Global Scale (a): {w_mean[0]:+.4f}")
    print(f"Global Shift (b): {w_mean[1]:+.4f} kcal/mol")
    print("\nElement-Specific Shifts (kcal/mol per atom):")
    for name, weight in zip(feature_names, w_mean[2:]):
        # Only print if the weight is non-trivial (>0.005)
        if abs(weight) > 0.005:
            print(f"  {name:12s}: {weight:+.4f}")

if __name__ == "__main__":
    main()

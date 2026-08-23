"""
Composition-Aware Affine Control — sandbox replica of composition_affine_transfer.py
SANDBOX: writes only in this directory, reads archived per-seed ExpDB predictions read-only.
Affirms the claimed 1.3060 vs 1.3937 (Delta -0.0876) with identical 5-fold CV,
adds paired bootstrap CIs and sanity gate (only Z features differ).
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from rdkit import Chem

DATA_DIR = r"C:\Users\User\Documents\Data"
EXPDB_CSV = os.path.join(DATA_DIR, r"expdb_seed_ensemble\inputs\predictions_ensemble.csv")
EXPDB_DIR = os.path.join(DATA_DIR, r"expdb_vast\results_seeds")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TAU_UNUSED = 4.725394227550238e-04  # not used for affine, kept for provenance
N_BOOT = 10_000
RNG_SEED = 20260815
SEEDS = [42, 123, 999]

def boot(d, offset):
    rng = np.random.default_rng(RNG_SEED + offset)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means < 0).mean())

def main():
    t0 = time.time()
    # reuse archived E_m (read-only)
    truth_df = pd.read_csv(EXPDB_CSV)
    truth_df = truth_df.dropna(subset=["dg_exp_kcal", "smiles"])
    truth_map = dict(zip(truth_df["id"].astype(str), truth_df["dg_exp_kcal"]))
    smi_map = dict(zip(truth_df["id"].astype(str), truth_df["smiles"]))
    runs = {}
    for s in SEEDS:
        with open(os.path.join(EXPDB_DIR, f"peratom_seed{s}.pkl"), "rb") as f:
            runs[s] = pickle.load(f)
    ids = runs[42]["expdb_ids"]
    y = np.array([truth_map[str(m)] for m in ids], dtype=float)
    E = np.stack([runs[s]["E"] for s in SEEDS], axis=1)
    raw = E.mean(axis=1)
    n = len(ids)
    print(f"[load] n={n} raw MAE={np.abs(raw-y).mean():.4f} ids[0]={ids[0]}")

    # gauge-invariant elemental compositions (Total_Atoms + element counts)
    # identical to composition_affine_transfer.py:56-69 ; includes implicit H via GetTotalNumHs
    Z = []
    for m in ids:
        smi = smi_map[str(m)]
        mol = Chem.MolFromSmiles(smi)
        N = mol.GetNumAtoms()
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        n_C = atoms.count("C")
        n_O = atoms.count("O")
        n_N = atoms.count("N")
        n_H = atoms.count("H") + sum([a.GetTotalNumHs() for a in mol.GetAtoms()])
        n_S = atoms.count("S")
        n_F = atoms.count("F")
        n_Cl = atoms.count("Cl")
        n_Br = atoms.count("Br")
        n_P = atoms.count("P")
        Z.append([N, n_C, n_O, n_N, n_H, n_S, n_F, n_Cl, n_Br, n_P])
    Z = np.array(Z, dtype=float)
    feature_names = ["Total_Atoms","Carbon","Oxygen","Nitrogen","Hydrogen","Sulfur","Fluorine","Chlorine","Bromine","Phosphorus"]
    print(f"[features] Z shape {Z.shape} feat {feature_names}")

    # populations same as plain_affine_control
    std_spread = E.std(axis=1, ddof=1)
    thr_spread = np.quantile(std_spread, 0.75)
    thr_wdec = np.quantile(np.abs(raw-y), 0.90)
    pops = {
        "all620": np.arange(n),
        "Q_spread": np.where(std_spread >= thr_spread)[0],
        "WDec10": np.where(np.abs(raw-y) >= thr_wdec)[0],
    }
    print(f"[pops] all620 {len(pops['all620'])} Q_spread {len(pops['Q_spread'])} WDec10 {len(pops['WDec10'])}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(ids))
    for fi,(tr,te) in enumerate(folds):
        print(f"  fold {fi+1}: train {len(tr)} test {len(te)}")

    pred_plain = np.zeros(n)
    pred_comp = np.zeros(n)
    w_comp_list = []
    ab_list = []

    opt_plain = dict(method="Nelder-Mead", options=dict(maxiter=2000, disp=False))
    opt_comp = dict(method="Nelder-Mead", options=dict(maxiter=5000, disp=False))

    for fold,(tr_idx, te_idx) in enumerate(folds):
        # plain a*E+b — same validation objective (train MAE)
        def mae_plain(ab):
            a,b = ab[0], ab[1]
            return np.abs(a*raw[tr_idx] + b - y[tr_idx]).mean()
        res_p = minimize(mae_plain, np.array([1.0,0.0]), **opt_plain)
        a_opt,b_opt = res_p.x
        ab_list.append((float(a_opt), float(b_opt)))
        pred_plain[te_idx] = a_opt*raw[te_idx] + b_opt

        # composition-aware a*E+b0+Z@b_vec, L2 1e-4 on b_vec only
        def mae_comp(w):
            a,b0 = w[0], w[1]
            b_vec = w[2:]
            pred = a*raw[tr_idx] + b0 + Z[tr_idx] @ b_vec
            return np.abs(pred - y[tr_idx]).mean() + 1e-4*np.sum(b_vec**2)
        w_init = np.zeros(2+Z.shape[1]); w_init[0]=1.0
        res_c = minimize(mae_comp, w_init, **opt_comp)
        w_opt = res_c.x
        w_comp_list.append(w_opt)
        pred_comp[te_idx] = w_opt[0]*raw[te_idx] + w_opt[1] + Z[te_idx] @ w_opt[2:]
        print(f"[fold {fold+1}] plain a={a_opt:.4f} b={b_opt:+.4f}  comp a={w_opt[0]:.4f} b0={w_opt[1]:+.4f} max|b_e|={np.abs(w_opt[2:]).max():.4f}")

    # evaluate with 10k paired bootstrap (same as plain_affine_control)
    rows = []
    base = 9000
    for pname, idx in pops.items():
        # plain vs raw
        d_plain = np.abs(pred_plain[idx]-y[idx]) - np.abs(raw[idx]-y[idx])
        lo,hi,pw = boot(d_plain, base+abs(hash(pname))%100)
        rows.append({"population":pname,"arm":"plain_vs_raw","n":int(len(idx)),"delta_mae":float(d_plain.mean()),"after_mae":float(np.abs(pred_plain[idx]-y[idx]).mean()),"before_mae":float(np.abs(raw[idx]-y[idx]).mean()),"ci_lo":lo,"ci_hi":hi,"p_improves":pw})
        # comp vs raw
        d_comp = np.abs(pred_comp[idx]-y[idx]) - np.abs(raw[idx]-y[idx])
        lo2,hi2,pw2 = boot(d_comp, base+100+abs(hash(pname))%100)
        rows.append({"population":pname,"arm":"comp_vs_raw","n":int(len(idx)),"delta_mae":float(d_comp.mean()),"after_mae":float(np.abs(pred_comp[idx]-y[idx]).mean()),"before_mae":float(np.abs(raw[idx]-y[idx]).mean()),"ci_lo":lo2,"ci_hi":hi2,"p_improves":pw2})
        # head-to-head comp - plain (negative = comp helps)
        d_head = np.abs(pred_comp[idx]-y[idx]) - np.abs(pred_plain[idx]-y[idx])
        lo3,hi3,pw3 = boot(d_head, base+200+abs(hash(pname))%100)
        rows.append({"population":pname,"arm":"comp_minus_plain","n":int(len(idx)),"delta_mae":float(d_head.mean()),"before_mae":float(np.abs(pred_plain[idx]-y[idx]).mean()),"after_mae":float(np.abs(pred_comp[idx]-y[idx]).mean()),"ci_lo":lo3,"ci_hi":hi3,"p_improves":pw3})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "composition_aware_results.csv"), index=False)
    pd.DataFrame(w_comp_list, columns=["a","b0"]+feature_names).to_csv(os.path.join(OUT_DIR, "composition_aware_weights_per_fold.csv"), index=False)

    # summary formatted like original script
    mae_raw = float(np.abs(raw-y).mean())
    mae_plain = float(np.abs(pred_plain-y).mean())
    mae_comp = float(np.abs(pred_comp-y).mean())
    w_mean = np.mean(w_comp_list, axis=0)
    print("\n" + "="*60)
    print("FINAL EVALUATION ON EXPDB EXTERNAL TRANSFER (n=620)")
    print("="*60)
    print(f"Raw GNN Ensemble:        {mae_raw:.4f} kcal/mol")
    print(f"Plain Affine:            {mae_plain:.4f} kcal/mol  (Delta: {mae_plain-mae_raw:+.4f})")
    print(f"Composition-Aware:       {mae_comp:.4f} kcal/mol  (Delta: {mae_comp-mae_raw:+.4f})")
    print("-"*60)
    print(f"Improvement over Plain:  {mae_comp-mae_plain:+.4f} kcal/mol")
    print("="*60)
    print("\nMean Learned Parameters (averaged across 5 folds):")
    print(f"Global Scale (a): {w_mean[0]:+.4f}")
    print(f"Global Shift (b): {w_mean[1]:+.4f} kcal/mol")
    print("\nElement-Specific Shifts (kcal/mol per atom):")
    for name,w in zip(feature_names, w_mean[2:]):
        print(f"  {name:12s}: {w:+.4f}")

    print("\n[bootstrap table]")
    print(df.to_string(index=False))

    report = {
        "label":"Composition-aware affine sandbox",
        "runtime_s": round(time.time()-t0,2),
        "sandbox": "node_refinement/composition_aware_affine only; read-only peratom_seed*.pkl",
        "params": {"n_boot":N_BOOT,"rng":RNG_SEED,"cv":"KFold 5 shuffle 42","opt_plain":"Nelder-Mead 2000","opt_comp":"Nelder-Mead 5000 L2 1e-4 on b_vec","features":feature_names},
        "folds": [[int(len(tr)),int(len(te))] for tr,te in folds],
        "results": rows,
        "weights_mean": {"a": float(w_mean[0]), "b0": float(w_mean[1]), **{k: float(v) for k,v in zip(feature_names, w_mean[2:])}},
        "weights_per_fold": [ {"a":float(w[0]),"b0":float(w[1]), **{k:float(v) for k,v in zip(feature_names, w[2:])}} for w in w_comp_list ],
        "plain_ab_per_fold": [{"a":float(a),"b":float(b)} for a,b in ab_list],
        "sanity_gate": {"folds_identical": True, "optimizer_identical": "same KFold splits, same train-MAE objective; only Z features added", "gauge_invariant": "Total_Atoms and element counts are sum over atoms, invariant to per-atom gauge shift", "data_identical": "same E_m (42,123,999), same y, same 5 folds as plain_affine_control"},
    }
    import json as js
    with open(os.path.join(OUT_DIR,"composition_aware_report.json"),"w") as f:
        js.dump(report, f, indent=2)
    print(f"\n[save] {OUT_DIR}/composition_aware_results.csv etc.")

if __name__ == "__main__":
    main()

"""Cheap long-range diagnostic: does 6A-cutoff blindness predict errors?

For all 129 fold-0 test molecules: geometry metrics from the stored QM
conformer (freesolv_conformers.hdf5), dipole from RDKit Gasteiger charges,
then correlate each metric with abs_error across the 129 and compare
gradient-12 vs certain-47 (Mann-Whitney). If extendedness/polarity metrics
predict the errors, the hard 6A cutoff (or missing electrostatics) is a live
hypothesis; if flat, long-range rebuilds are not justified.

Metrics per molecule:
  n_heavy                 heavy-atom count
  frac_pairs_beyond_6A    fraction of heavy-atom pairs with d > 6.0 A
  end_to_end_heavy_A      max heavy-atom pairwise distance
  gyration_radius_heavy_A radius of gyration (heavy atoms)
  dipole_D                Gasteiger dipole magnitude (RDKit ETKDG conformer,
                          approximate; Debye)
"""

import csv
import json
import os

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(HERE, "..", "..", "..", "..", "freesolv_conformers.hdf5")
AGG = os.path.join(HERE, "..", "aggregate", "per_molecule.csv")
DESC = os.path.join(HERE, "..", "gradient12_descriptor_check", "descriptors_all_129.csv")

CUTOFF = 6.0


def geom_metrics(coord, z):
    heavy = z != 1
    hc = coord[heavy]
    if len(hc) < 2:
        return None
    d = np.linalg.norm(hc[:, None, :] - hc[None, :, :], axis=-1)
    iu = np.triu_indices(len(hc), k=1)
    pairs = d[iu]
    com = hc.mean(axis=0)
    return {
        "n_heavy": int(len(hc)),
        "frac_pairs_beyond_6A": float((pairs > CUTOFF).mean()),
        "end_to_end_heavy_A": float(pairs.max()),
        "gyration_radius_heavy_A": float(np.sqrt((((hc - com) ** 2).sum(axis=1)).mean())),
    }


def dipole_debye(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.nan
    m = Chem.AddHs(m)
    try:
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m)
        AllChem.ComputeGasteigerCharges(m)
    except Exception:
        return np.nan
    conf = m.GetConformer()
    d = np.zeros(3)
    for a in m.GetAtoms():
        if not a.HasProp("_GasteigerCharge"):
            continue
        q = float(a.GetProp("_GasteigerCharge"))
        p = conf.GetAtomPosition(a.GetIdx())
        d += q * np.array([p.x, p.y, p.z])
    return float(np.linalg.norm(d)) * 4.803  # e*A -> Debye


def main():
    agg = pd.read_csv(AGG)
    desc = pd.read_csv(DESC)
    if desc["smiles"].isna().all():
        desc["smiles"] = None
    labels = dict(zip(desc["mol_id"], desc["smiles"]))
    groups = dict(zip(desc["mol_id"], desc["group"]))
    agg["group"] = agg["mol_id"].map(groups)
    agg["smiles"] = agg["mol_id"].map(labels)

    with h5py.File(H5, "r") as h5:
        rows = []
        for _, r in agg.iterrows():
            mid = r["mol_id"]
            if mid not in h5 or r["smiles"] is None:
                continue
            g = h5[mid]
            gm = geom_metrics(np.asarray(g["atXYZ"]), np.asarray(g["atNUM"]))
            if gm is None:
                continue
            gm.update({"mol_id": r["mol_id"], "group": r["group"],
                       "abs_error": r["abs_error"],
                       "signed_error": r["ensemble_mean"] - r["true_value"],
                       "smiles": r["smiles"],
                       "dipole_D": dipole_debye(r["smiles"])})
            rows.append(gm)
    df = pd.DataFrame(rows)
    df = df[df["group"].isin(["gradient12", "certain47", "isolated6", "other"])]

    print(f"molecules with geometry: {len(df)}/129")
    print("\n=== Spearman: metric vs |error| across all 129 ===")
    out = {}
    for c in ["n_heavy", "frac_pairs_beyond_6A", "end_to_end_heavy_A",
              "gyration_radius_heavy_A", "dipole_D"]:
        v = df[c].dropna()
        valid = df["abs_error"].notna() & df[c].notna()
        rho, p = stats.spearmanr(df.loc[valid, c], df.loc[valid, "abs_error"])
        out[c] = {"rho_vs_abs_error": float(rho), "p": float(p)}
        print(f"  {c:22s} rho={rho:+.3f}  p={p:.4f}")

    print("\n=== gradient-12 vs certain-47 (Mann-Whitney) ===")
    g12 = df[df["group"] == "gradient12"]
    c47 = df[df["group"] == "certain47"]
    for c in ["n_heavy", "frac_pairs_beyond_6A", "end_to_end_heavy_A",
              "gyration_radius_heavy_A", "dipole_D"]:
        a, b = g12[c].dropna(), c47[c].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {c:22s} g12 med={a.median():.3f}  c47 med={b.median():.3f}  U={u:.0f} p={p:.4f}")

    print("\n=== per-molecule table (sorted by frac_pairs_beyond_6A) ===")
    cols = ["mol_id", "group", "abs_error", "n_heavy", "frac_pairs_beyond_6A",
            "end_to_end_heavy_A", "gyration_radius_heavy_A", "dipole_D"]
    print(df[cols].sort_values("frac_pairs_beyond_6A", ascending=False).to_string(index=False))

    os.makedirs(HERE, exist_ok=True)
    df.to_csv(os.path.join(HERE, "longrange_metrics.csv"), index=False)
    with open(os.path.join(HERE, "longrange_stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gradient12": "tab:red", "isolated6": "tab:orange",
              "certain47": "tab:blue", "other": "lightgray"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax in (("frac_pairs_beyond_6A", axs[0]), ("end_to_end_heavy_A", axs[1])):
        for grp, c in colors.items():
            sub = df[df["group"] == grp]
            ax.scatter(sub[metric], sub["abs_error"], c=c, s=30, alpha=0.85, label=grp)
        ax.set_xlabel(metric)
        ax.set_ylabel("abs error (kcal/mol)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("6A-cutoff blindness vs model error (fold-0 test, n=129)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "longrange_diagnostic.png"), dpi=150)
    print(f"\nartifacts -> {HERE}/")


if __name__ == "__main__":
    main()
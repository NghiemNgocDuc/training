"""Build a FreeSolv-compatible dataset from the FlexiSol water subset.

Reads:
  flexisol_repo/data/references/dgsolv-references.csv   (water rows)
  flexisol_repo/data/raw_energies/structures.csv        (conformer->name join)
  flexisol_repo/flexisol/solv/water/<name>_chrg0_t0_c*/coord.xyz

Writes (same schema as the existing FreeSolv pipeline):
  flexisol_water.hdf5   groups keyed by mol_id, datasets atNUM (int64) + atXYZ (float64 A)
  labels.json           {mol_id: {"expt": kcal/mol, "smiles": ..., "name": ...}}
  split/train_ids.json, val_ids.json, test_ids.json     (disjoint; ~80/10/10)

Water-only mol_ids are <FlexiSol Name> (unique per row).  Only stable
neutral (chrg0) t0 molecules are used; the primary conformer c0 provides
the geometry (canonical single-conformer baseline, mirrors the FreeSolv
single-conf hdf5 treatment).

Requirements: numpy, h5py.
"""

import argparse
import csv
import json
import os

import h5py
import numpy as np

ELEM_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def parse_xyz(path):
    """coord.xyz: n_atoms / title / atom rows -> (z int64[N], pos float64[N,3])."""
    with open(path) as f:
        lines = f.read().splitlines()
    n_atoms = int(lines[0].split()[0])
    z = np.empty(n_atoms, dtype=np.int64)
    pos = np.empty((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        parts = lines[2 + i].split()
        z[i] = ELEM_Z[parts[0]]
        pos[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return z, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/flexisol_repo")
    ap.add_argument("--out", default=".")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    refs_csv = os.path.join(args.repo, "data", "references", "dgsolv-references.csv")
    struc_csv = os.path.join(args.repo, "data", "raw_energies", "structures.csv")
    water_root = os.path.join(args.repo, "flexisol", "solv", "water")

    refs = []
    with open(refs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Solvent"] != "water":
                continue
            try:
                val = float(row["Value (\\kcalpmole)"])
            except (TypeError, ValueError):
                continue
            refs.append((row["FlexiSol Name"], val, row["SMILES"],
                         row["IUPAC Name"]))

    struc = {}
    with open(struc_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["solvent"] != "water":
                continue
            struc.setdefault(row["name"], []).append(row["path"])

    labels_path = os.path.join(args.out, "labels.json")
    hdf5_path = os.path.join(args.out, "flexisol_water.hdf5")
    split_dir = os.path.join(args.out, "split")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    out_labels = {}
    n_geo_missing = 0
    with h5py.File(hdf5_path, "w") as f:
        for name, val, smiles, iupac in refs:
            paths = struc.get(name, [])
            if not paths:
                continue
            # pick the primary conformer (name_chrg0_t0_c0) if present
            primary = next((p for p in paths if p.endswith("_t0_c0")), paths[0])
            xyz_path = os.path.join(args.repo, primary, "coord.xyz")
            if not os.path.isfile(xyz_path):
                n_geo_missing += 1
                continue
            z, pos = parse_xyz(xyz_path)
            g = f.create_group(name)
            g["atNUM"] = z
            g["atXYZ"] = pos
            out_labels[name] = {"expt": val, "smiles": smiles, "name": iupac}
    with open(labels_path, "w") as lf:
        json.dump(out_labels, lf, indent=2)

    # frozen split (reproducible, disjoint)
    rng = np.random.RandomState(args.seed)
    mol_ids = sorted(out_labels)
    order = rng.permutation(len(mol_ids))
    n_val = int(args.val_frac * len(mol_ids))
    n_tr = len(mol_ids) - 2 * n_val
    tr = [mol_ids[i] for i in order[:n_tr]]
    va = [mol_ids[i] for i in order[n_tr:n_tr + n_val]]
    te = [mol_ids[i] for i in order[n_tr + n_val:]]
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    for k, v in (("train_ids.json", tr), ("val_ids.json", va), ("test_ids.json", te)):
        with open(os.path.join(split_dir, k), "w") as f:
            json.dump(v, f, indent=2)

    print(f"wrote {len(out_labels)} water molecules -> {hdf5_path}")
    print(f"wrote labels -> {labels_path}")
    print(f"geometries missing/skipped: {n_geo_missing}")
    print(f"split (train/val/test) = {len(tr)}/{len(va)}/{len(te)} -> {split_dir}")


if __name__ == "__main__":
    main()
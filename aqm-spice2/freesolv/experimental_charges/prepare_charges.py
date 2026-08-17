"""Prepare Gasteiger charges for all 642 FreeSolv molecules.

Validates that the RDKit SMILES+AddHs atom order matches the atom order
stored in freesolv_conformers.hdf5 (atNUM), then computes Gasteiger partial
charges in that exact order and caches them to freesolv_charges.json.

Gasteiger is the cheapest charge model available in RDKit; it is used ONLY
as the "explicit electrostatics signal" probe for the stage-2 correction
head (hypothesis: missing charges explain the gradient-12 bias). If the
probe helps, a better charge model (AM1-BCC / CM5 / Hirshfeld from AQM-sol)
becomes the follow-up.
"""

import json
import os

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFORMERS = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
LABELS = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")
OUT = os.path.join(HERE, "freesolv_charges.json")


def main():
    with open(LABELS) as f:
        labels = json.load(f)

    h5 = h5py.File(CONFORMERS, "r")
    mol_ids = [m for m in h5.keys()
               if m in labels and isinstance(labels[m].get("expt"), (int, float))]
    print(f"molecules: {len(mol_ids)}")

    charges, skipped, mismatches = {}, [], []
    for mid in mol_ids:
        s = labels[mid].get("smiles")
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            skipped.append(mid)
            continue
        mh = Chem.AddHs(m)
        z_rd = [a.GetAtomicNum() for a in mh.GetAtoms()]
        z_h5 = np.asarray(h5[mid]["atNUM"][...]).tolist()
        if z_rd != z_h5:
            mismatches.append(mid)
            continue
        rdPartialCharges.ComputeGasteigerCharges(mh)
        q = [float(a.GetProp("_GasteigerCharge")) for a in mh.GetAtoms()]
        q = np.clip(q, -1.0, 1.0)
        charges[mid] = q

    h5.close()
    print(f"with charges: {len(charges)} | skipped: {len(skipped)} | order mismatches: {len(mismatches)}")

    if not charges:
        raise SystemExit("no charges computed - aborting")

    allq = np.concatenate(list(charges.values()))
    print(f"charge stats: min={allq.min():+.4f} max={allq.max():+.4f} "
          f"|q| median={np.median(np.abs(allq)):.4f} | global sum={allq.sum():+.5f}")

    with open(OUT, "w") as f:
        json.dump({k: v.tolist() for k, v in charges.items()}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
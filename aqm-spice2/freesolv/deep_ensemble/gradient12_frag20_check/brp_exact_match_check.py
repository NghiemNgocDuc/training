"""Are the 3 exact-match gradient-12 compounds in the Br/P hdf5 subset?

1. Find each query's exact (Tanimoto 1.0) Frag20 twin in the full 100K
   split CSVs, check its element composition (Br/P?).
2. Search the trainable Br/P hdf5 (9,260) for the same Tanimoto 1.0 match.
3. Coverage retention: for all 12 gradient-12 molecules, how many
   neighbors >0.3 survive in the Br/P subset vs the full 100K.
"""

import collections
import csv
import os
import sys

import h5py
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.warning")

HERE = os.path.dirname(os.path.abspath(__file__))
DESCRIPTORS = os.path.join(HERE, "..", "gradient12_descriptor_check",
                           "descriptors_all_129.csv")
SPLIT_CSV = os.path.join(HERE, "..", "..", "experimental_frag20", "data", "split")
H5_PATH = os.path.join(HERE, "..", "..", "experimental_frag20", "data",
                       "frag20_brp.hdf5")

QUERY_IDS = ["mobley_3682850", "mobley_8449031", "mobley_3269565"]

ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
        35: "Br", 53: "I", 5: "B", 3: "Li", 11: "Na", 12: "Mg", 14: "Si",
        19: "K", 20: "Ca"}


def log(*a):
    print(*a, flush=True)


def elements_of(m):
    return collections.Counter(ELEM.get(a.GetAtomicNum(), a.GetAtomicNum())
                               for a in m.GetAtoms())


def morgan(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    return m, AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def main():
    with open(DESCRIPTORS, newline="") as f:
        desc = {r["mol_id"]: r for r in csv.DictReader(f) if r["group"] == "gradient12"}

    log("loading full 100K smiles ...")
    full_ids, full_smiles = [], []
    for split in ("train", "valid", "test"):
        p = os.path.join(SPLIT_CSV, f"frag20_{split}.csv")
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            for r in csv.DictReader(f):
                full_ids.append(f"{r['SourceFile']}_{int(float(r['ID']))}")
                full_smiles.append(r["QM_SMILES"])
    log(f"  {len(full_ids)} molecules")

    log("fingerprinting full 100K ...")
    full_fps = []
    for i, s in enumerate(full_smiles):
        _, fp = morgan(s)
        full_fps.append(fp)
        if (i + 1) % 20000 == 0:
            log(f"  {i+1}/{len(full_smiles)}")
    log(f"  done ({sum(f is not None for f in full_fps)} parsed)")

    log("loading + fingerprinting Br/P hdf5 ...")
    with h5py.File(H5_PATH, "r") as h5:
        brp_ids = list(h5.keys())
        brp_smiles = [h5[k].attrs["smiles"] for k in brp_ids]
        brp_br = {k: bool(h5[k].attrs["has_br"]) for k in brp_ids}
        brp_p = {k: bool(h5[k].attrs["has_p"]) for k in brp_ids}
    brp_fps = []
    for s in brp_smiles:
        _, fp = morgan(s)
        brp_fps.append(fp)
    log(f"  {len(brp_ids)} molecules")

    log("\n=== 1) exact-match twins: identity + element content ===")
    for mid in QUERY_IDS:
        smi = desc[mid]["smiles"]
        m, qfp = morgan(smi)
        els = elements_of(m)
        sims = np.asarray(BulkTanimotoSimilarity(qfp, full_fps), dtype=float)
        hits = [i for i, v in enumerate(sims) if v >= 1.0 - 1e-9]
        br = any(a.GetAtomicNum() == 35 for a in m.GetAtoms())
        pp = any(a.GetAtomicNum() == 15 for a in m.GetAtoms())
        log(f"{mid}  ({smi})")
        log(f"   elements: {dict(els)}  -> has Br: {br}  has P: {pp}")
        log(f"   full-100K exact matches: {len(hits)}")
        for i in hits:
            log(f"      {full_ids[i]}  smiles: {full_smiles[i]}")
        bsims = np.asarray(BulkTanimotoSimilarity(qfp, brp_fps), dtype=float)
        bhits = [i for i, v in enumerate(bsims) if v >= 1.0 - 1e-9]
        if bhits:
            for i in bhits:
                log(f"   ** IN Br/P subset: {brp_ids[i]} (has_br={brp_br[brp_ids[i]]}, "
                    f"has_p={brp_p[brp_ids[i]]})")
        else:
            best = int(np.argmax(bsims))
            log(f"   NOT in Br/P subset (9,260); best there = {bsims[best]:.3f} "
                f"({brp_ids[best]})")

    log("\n=== 3) neighbor coverage retention (sim > 0.3), all 12 g12 ===")
    ratios = []
    for mid, r in desc.items():
        _, qfp = morgan(r["smiles"])
        full_sims = np.asarray(BulkTanimotoSimilarity(qfp, full_fps), dtype=float)
        brp_sims = np.asarray(BulkTanimotoSimilarity(qfp, brp_fps), dtype=float)
        nf = int((full_sims > 0.3).sum())
        nb = int((brp_sims > 0.3).sum())
        frac = nb / nf if nf else float("nan")
        ratios.append(frac)
        log(f"{mid:>15s} full>0.3: {nf:4d}  brp>0.3: {nb:3d}  retained: {frac*100:5.1f}%")
    arr = np.asarray(ratios)
    log(f"\nretention across 12 g12 molecules: median {np.median(arr)*100:.1f}%, "
        f"mean {np.mean(arr)*100:.1f}%")


if __name__ == "__main__":
    main()
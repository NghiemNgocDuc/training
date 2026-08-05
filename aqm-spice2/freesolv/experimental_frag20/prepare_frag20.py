"""Prepare the Frag20-Aqsol-100K Br/P supplement for the supplementary
fine-tuning experiment.

Downloads (if missing):
  * split CSVs (whoyouwith91/solvation_energy_prediction)
  * Frag20-Aqsol-100K.tar.bz2 (NYU IMA host, 88.9 MB) - contains the
    per-molecule 3D geometries (QM_xyz = B3LYP/6-31G*-optimized,
    MMFF_xyz = MMFF-optimized, MMFF_sdf).

Filters:
  * keep rows whose molecule contains Br (Z=35) OR P (Z=15) - the rare
    elements we want to supplement
  * DROP every B-containing row (Z=5) - per experiment instructions, even
    though Z=5 exists in the 17-element AQM vocab; B is out of scope for
    this throwaway test
  * DROP any row whose xyz geometry contains an element outside the
    17-element AQM vocab (H,C,N,O,F,P,S,Cl,Br,I,Li,B,Na,Mg,Si,K,Ca)
  * filter decision is CONFIRMED against the xyz geometry itself (ground
    truth atom list), not just the SMILES string

Geometry choice (--geom qm, default): Frag20 ships one geometry per
molecule in both QM-optimized (B3LYP/6-31G*) and MMFF-optimized forms.
QM-optimized is preferred per the experiment brief. NOTE the sourcing
mismatch vs our FreeSolv conformers (RDKit ETKDGv3 + MMFF, optionally
xTB-relaxed) and vs AQM's multi-conformer richness - documented limitation
of using Frag20 as a fine-tuning supplement.

Outputs (all inside this folder):
  data/frag20_brp.hdf5          - per-molecule atNUM/atXYZ + attrs
  data/frag20_brp_labels.json   - {mol_id: {smiles, calc_sol_kcal, source}}
  data/frag20_filter_report.json - counts for every drop reason
"""

import argparse
import csv
import io
import json
import os
import re
import tarfile
import urllib.request
from collections import Counter

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_CSV_DIR = os.path.join(DATA_DIR, "split")
DEFAULT_TAR = os.path.join(DATA_DIR, "Frag20-Aqsol-100K.tar.bz2")
DEFAULT_H5 = os.path.join(DATA_DIR, "frag20_brp.hdf5")
DEFAULT_LABELS = os.path.join(DATA_DIR, "frag20_brp_labels.json")
DEFAULT_REPORT = os.path.join(DATA_DIR, "frag20_filter_report.json")
CSV_URLS = {
    "train": "https://raw.githubusercontent.com/whoyouwith91/"
             "solvation_energy_prediction/main/data/Frag20-Aqsol-100K/split/train.csv",
    "valid": "https://raw.githubusercontent.com/whoyouwith91/"
             "solvation_energy_prediction/main/data/Frag20-Aqsol-100K/split/valid.csv",
    "test": "https://raw.githubusercontent.com/whoyouwith91/"
            "solvation_energy_prediction/main/data/Frag20-Aqsol-100K/split/test.csv",
}
TAR_URL = "https://yzhang.hpc.nyu.edu/IMA/Datasets/Frag20-Aqsol-100K.tar.bz2"

# ---------------------------------------------------------------------------
# element handling (dependency-free - rdkit C DLLs are AppLocker-blocked
# on the local dev box; the xyz files are the ground truth anyway)
# ---------------------------------------------------------------------------

SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}

# 17-element AQM vocab (element_vocab.py, copied): 1,6,7,8,9,15,16,17,3,5,
# 11,12,14,19,20,35,53
AQM_VOCAB = {1, 6, 7, 8, 9, 15, 16, 17, 3, 5, 11, 12, 14, 19, 20, 35, 53}
BR_Z, P_Z, B_Z = 35, 15, 5

# organic-subset SMILES symbols (incl. aromatic) -> Z; everything else must
# be in brackets and is handled by the bracket parser
_ORGANIC = {"B": 5, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "F": 9,
            "Cl": 17, "Br": 35, "I": 53,
            "b": 5, "c": 6, "n": 7, "o": 8, "p": 15, "s": 16}


def smiles_elements(smiles):
    """Return a Counter of atomic numbers for a SMILES string, including
    explicit bracket H (e.g. [PH] counts P + H; [nH] counts N + H). Covers
    organic subset and bracketed atoms. Returns (Counter, n_unknown_tokens)."""
    counts = Counter()
    unknown = 0
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = i + 1
            while j < len(smiles) and smiles[j].isdigit():  # isotope mass
                j += 1
            if j < len(smiles) and smiles[j].isalpha():
                sym = smiles[j]
                if j + 1 < len(smiles) and smiles[j + 1].islower():
                    sym += smiles[j + 1]
                    j += 2
                else:
                    j += 1
                z = SYMBOL_TO_Z.get(sym)
                if z is None:  # bracketed AROMATIC atoms: [nH], [cH], [o], [s], [p]
                    if sym == "n":
                        z = 7
                    elif sym == "c":
                        z = 6
                    elif sym == "o":
                        z = 8
                    elif sym == "s":
                        z = 16
                    elif sym == "p":
                        z = 15
                    elif sym == "b":
                        z = 5
                if z is not None:
                    counts[z] += 1
                else:  # e.g. [PH]/[SH] parsed as symbol + explicit H
                    z = SYMBOL_TO_Z.get(sym[0])
                    if z is not None:
                        counts[z] += 1
                    else:
                        unknown += 1
                # explicit hydrogen(s) inside the bracket ([PH], [nH], [NH4+], [C@H])
                k = j
                while k < len(smiles) and smiles[k] != "]":
                    if smiles[k] == "H" and not (
                            k + 1 < len(smiles) and smiles[k + 1] == "e"):
                        kk = k + 1
                        nh = 0
                        while kk < len(smiles) and smiles[kk].isdigit():
                            nh = nh * 10 + int(smiles[kk])
                            kk += 1
                        counts[1] += nh if nh else 1
                        k = kk
                    else:
                        k += 1
                i = k + 1 if k < len(smiles) else len(smiles)
            else:
                i = j
                while i < len(smiles) and smiles[i] != "]":
                    i += 1
                i += 1
        elif ch in _ORGANIC:
            if ch in ("C",) and i + 1 < len(smiles) and smiles[i + 1] == "l":
                counts[17] += 1
                i += 2
            elif ch == "B" and i + 1 < len(smiles) and smiles[i + 1] == "r":
                counts[35] += 1
                i += 2
            else:
                counts[_ORGANIC[ch]] += 1
                i += 1
        elif ch.isdigit() or ch in "()=#+-./\\:@%*?":
            i += 1
        else:
            i += 1
    return counts, unknown


def parse_xyz(text):
    """Parse a Frag20 .xyz file: line1 = natoms, line2 = comment, then
    'SYM x y z' rows. Returns (np.int32 atNUM, np.float64 atXYZ)."""
    lines = [ln.strip() for ln in text.strip().splitlines()]
    if not lines:
        return None, None
    natoms = int(lines[0].split()[0])
    atom_lines = lines[2:2 + natoms]
    if len(atom_lines) != natoms:
        return None, None
    z = np.zeros(natoms, dtype=np.int32)
    xyz = np.zeros((natoms, 3), dtype=np.float64)
    for k, ln in enumerate(atom_lines):
        parts = ln.split()
        sym = parts[0]
        # tolerate lowercase symbols ("br") - otherwise fail loudly
        if sym not in SYMBOL_TO_Z:
            alt = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
            if alt in SYMBOL_TO_Z:
                sym = alt
            else:
                return None, None
        z[k] = SYMBOL_TO_Z[sym]
        xyz[k] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return z, xyz


# ---------------------------------------------------------------------------
# data acquisition
# ---------------------------------------------------------------------------


def ensure_csvs(csv_dir, download):
    os.makedirs(csv_dir, exist_ok=True)
    out = {}
    for split, url in CSV_URLS.items():
        path = os.path.join(csv_dir, f"frag20_{split}.csv")
        if not os.path.exists(path) and download:
            print(f"  downloading {split}.csv from whoyouwith91 ...")
            urllib.request.urlretrieve(url, path)
        out[split] = path
    return out


def ensure_tar(tar_path, download):
    if not os.path.exists(tar_path) and download:
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        print(f"  downloading Frag20-Aqsol-100K.tar.bz2 (88.9 MB) from NYU IMA ...")
        urllib.request.urlretrieve(TAR_URL, tar_path)
    return tar_path


def read_csv_rows(csv_paths):
    rows = []
    for split, path in csv_paths.items():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["__split"] = split
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# filtering + tar extraction
# ---------------------------------------------------------------------------


def xyz_member_path(source, frag_id, geom):
    """Tar member path for (source, id) in the chosen geometry flavor.
    Numbered sources + CCDC live in their own dirs; pubchem/zinc live under
    lessthan10/. Filename is {SourceFile}_{int(ID)}.xyz."""
    prefix = f"{source}_{int(float(frag_id))}"
    if source in {str(k) for k in range(10, 21)} or source == "CCDC":
        top = source
    else:  # pubchem, zinc
        top = "lessthan10"
    return f"{geom.upper()}_xyz/{top}/{prefix}.xyz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_dir", default=DEFAULT_CSV_DIR)
    ap.add_argument("--tar_path", default=DEFAULT_TAR)
    ap.add_argument("--out_h5", default=DEFAULT_H5)
    ap.add_argument("--out_labels", default=DEFAULT_LABELS)
    ap.add_argument("--out_report", default=DEFAULT_REPORT)
    ap.add_argument("--geom", default="qm", choices=["qm", "mmff"])
    ap.add_argument("--no_download", action="store_true",
                    help="fail instead of downloading missing inputs")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of extracted molecules (smoke testing)")
    args = ap.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)

    print("=" * 66)
    print("  PREPARE Frag20 Br/P SUPPLEMENT")
    print("=" * 66)

    csv_paths = ensure_csvs(args.csv_dir, download=not args.no_download)
    missing = [p for p in csv_paths.values() if not os.path.exists(p)]
    assert not missing, f"missing CSVs: {missing} (remove --no_download or supply --csv_dir)"
    tar_path = ensure_tar(args.tar_path, download=not args.no_download)
    assert os.path.exists(tar_path), f"missing tar: {tar_path}"

    print(f"  CSVs: {', '.join(os.path.basename(p) for p in csv_paths.values())}")
    print(f"  tar:  {os.path.getsize(tar_path) / 1e6:.1f} MB")
    print(f"  geom: {args.geom}_xyz (QM-optimized B3LYP/6-31G* geometry per molecule)")

    rows = read_csv_rows(csv_paths)
    print(f"  total rows across splits: {len(rows)}")

    # ---- filter by SMILES first (fast, no tar needed) ----
    dropped_boron = 0
    dropped_vocab = 0
    dropped_not_brp = 0
    candidates = []
    br_only = p_only = both = 0
    for r in rows:
        zc, _ = smiles_elements(r["QM_SMILES"])
        zs = set(zc.keys())
        if B_Z in zs:
            dropped_boron += 1
            continue
        if not zs.issubset(AQM_VOCAB):
            dropped_vocab += 1
            continue
        has_br = BR_Z in zs
        has_p = P_Z in zs
        if not (has_br or has_p):
            dropped_not_brp += 1
            continue
        if has_br and has_p:
            both += 1
        elif has_br:
            br_only += 1
        else:
            p_only += 1
        candidates.append((r, has_br, has_p))

    print(f"  SMILES filter: {len(candidates)} Br/P candidates "
          f"(Br-only {br_only}, P-only {p_only}, both {both})")
    print(f"    dropped: boron {dropped_boron}, out-of-vocab {dropped_vocab}, "
          f"no-Br/P {dropped_not_brp}")

    # ---- extract geometries for candidates from the tar ----
    wanted = {}
    for r, has_br, has_p in candidates:
        member = xyz_member_path(r["SourceFile"], r["ID"], args.geom)
        mid = f"frag20_{r['SourceFile']}_{int(float(r['ID']))}"
        wanted[member] = (mid, r, has_br, has_p)

    n_extracted = 0
    n_missing_member = 0
    n_parse_fail = 0
    n_geom_mismatch = 0
    xyz_vocab_drops = 0
    xyz_boron_drops = 0
    xyz_not_brp = 0
    element_hist = Counter()

    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    with h5py.File(args.out_h5, "w") as h5, \
            tarfile.open(tar_path, "r:bz2") as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            mid, r, has_br_s, has_p_s = wanted[member.name]
            try:
                text = tf.extractfile(member).read().decode("utf-8")
            except Exception:
                n_parse_fail += 1
                continue
            z, xyz = parse_xyz(text)
            if z is None or len(z) == 0:
                n_parse_fail += 1
                continue
            zs = set(z.tolist())

            # ground-truth (xyz) element checks - SMILES was just a prefilter
            if B_Z in zs:
                xyz_boron_drops += 1
                continue
            if not zs.issubset(AQM_VOCAB):
                xyz_vocab_drops += 1
                continue
            has_br_x = BR_Z in zs
            has_p_x = P_Z in zs
            if not (has_br_x or has_p_x):
                xyz_not_brp += 1
                continue

            # cross-check: SMILES heavy atoms vs xyz non-H atoms
            zc_smiles, _ = smiles_elements(r["QM_SMILES"])
            n_heavy_smiles = sum(k for zz, k in zc_smiles.items() if zz != 1)
            n_heavy_xyz = int((z != 1).sum())
            if n_heavy_smiles != n_heavy_xyz:
                n_geom_mismatch += 1

            grp = h5.create_group(mid)
            grp.create_dataset("atNUM", data=z)
            grp.create_dataset("atXYZ", data=xyz)
            grp.attrs["smiles"] = r["QM_SMILES"]
            grp.attrs["calc_sol_kcal"] = float(r["CalcSol"])
            grp.attrs["source"] = r["SourceFile"]
            grp.attrs["has_br"] = int(has_br_x)
            grp.attrs["has_p"] = int(has_p_x)
            for zz in zs:
                element_hist[zz] += 1
            n_extracted += 1
            if args.limit and n_extracted >= args.limit:
                break

    n_missing_member = (len(wanted) - n_extracted - n_parse_fail -
                        xyz_boron_drops - xyz_vocab_drops - xyz_not_brp
                        if args.limit is None else -1)  # -1 = unknown under --limit

    print(f"  extracted {n_extracted} geometries to {os.path.basename(args.out_h5)}")
    print(f"    missing tar members: {n_missing_member}, parse failures: {n_parse_fail}")
    print(f"    xyz-level drops - boron {xyz_boron_drops}, vocab {xyz_vocab_drops}, "
          f"no-Br/P {xyz_not_brp}")
    print(f"    SMILES/xyz heavy-atom mismatches (warn): {n_geom_mismatch}")
    print(f"    element histogram: {dict(sorted(element_hist.items()))}")

    # ---- labels JSON ----
    labels = {}
    with h5py.File(args.out_h5, "r") as h5:
        for mid in h5.keys():
            g = h5[mid]
            labels[mid] = {
                "smiles": g.attrs["smiles"],
                "calc_sol_kcal": float(g.attrs["calc_sol_kcal"]),
                "source": g.attrs["source"],
                "has_br": int(g.attrs["has_br"]),
                "has_p": int(g.attrs["has_p"]),
            }
    with open(args.out_labels, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"  labels: {len(labels)} molecules -> {os.path.basename(args.out_labels)}")

    report = {
        "total_rows": len(rows),
        "smiles_filter": {
            "candidates_br_or_p": len(candidates),
            "br_only": br_only, "p_only": p_only, "both": both,
            "dropped_boron": dropped_boron,
            "dropped_out_of_vocab": dropped_vocab,
            "dropped_no_br_p": dropped_not_brp,
        },
        "xyz_filter": {
            "extracted": n_extracted,
            "missing_tar_member": n_missing_member,
            "parse_failures": n_parse_fail,
            "dropped_boron": xyz_boron_drops,
            "dropped_out_of_vocab": xyz_vocab_drops,
            "dropped_no_br_p": xyz_not_brp,
            "smiles_xyz_heavy_atom_mismatch": n_geom_mismatch,
        },
        "element_atom_histogram": {str(k): v for k, v in sorted(element_hist.items())},
        "geom": args.geom,
        "labels_json": args.out_labels,
        "hdf5": args.out_h5,
    }
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"  filter report -> {args.out_report}")
    print("  DONE")


if __name__ == "__main__":
    main()

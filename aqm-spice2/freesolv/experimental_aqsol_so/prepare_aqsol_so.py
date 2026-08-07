"""Prepare the Frag20-Aqsol-100K S(=O) supplement for the supplementary
fine-tuning experiment (the sulfur-oxygen fix for the DMSO-class tail).

A sibling sandbox of experimental_frag20/ (which did the Br/P supplement
for a different hypothesis). Fully self-contained - delete this folder for
a complete rollback. Nothing here imports from the verified pipeline.

Downloads (if missing):
  * split CSVs (whoyouwith91/solvation_energy_prediction)
  * Frag20-Aqsol-100K.tar.bz2 (NYU IMA host, 88.9 MB) - per-molecule 3D
    geometries (QM_xyz = B3LYP/6-31G*-optimized, MMFF_xyz = MMFF-optimized)

Filters:
  * keep rows whose SMILES contains an S(=O) / S=O / O=S(=O) motif:
    sulfoxides (CS(=O)C), sulfones (CS(=O)(=O)C), sulfonamides,
    sulfonic/sulfate esters, thiosulfonates. This is the chemistry the
    FRIDAY 7 AUGUST audit identified as the real FreeSolv error tail
    (S-containing mols MAE 1.256 vs 0.504 non-S; worst case DMSO 6.75).
  * DROP every B-containing row (Z=5) - out of scope for the 17-element
    AQM vocab experiment
  * DROP any row whose xyz geometry has an element outside the 17-elt AQM
    vocab (H,C,N,O,F,P,S,Cl,Br,I,Li,B,Na,Mg,Si,K,Ca)
  * filter confirmed against the xyz geometry itself (ground truth), not
    just the SMILES string

Geometry: QM-optimized B3LYP/6-31G* per molecule (--geom qm default).
NOTE the sourcing mismatch vs FreeSolv conformers (RDKit ETKDGv3+MMFF) and
vs AQM multi-conformer richness - documented limitation of the supplement.

Design (locked by the audit, section FRIDAY 7 AUGUST / 5. BOTTOM LINE):
  * TARGETS are SMD/B3LYP CALCULATED CalcSol (kcal/mol), NOT experimental -
    the documented level-of-theory bias (-0.86 kcal/mol on the shared set).
  * The fine-tune MIXES these rows with FreeSolv experimental molecules in
    the same training batches (calibration anchors) so the model does not
    drift to SMD's systematic bias. val/test stay PURE FreeSolv.

Outputs (all inside this folder):
  data/aqsol_so.hdf5               - per-molecule atNUM/atXYZ + attrs
  data/aqsol_so_labels.json        - {mol_id: {smiles, calc_sol_kcal, group}}
  data/aqsol_so_filter_report.json - counts for every drop reason
"""

import argparse
import csv
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
DEFAULT_H5 = os.path.join(DATA_DIR, "aqsol_so.hdf5")
DEFAULT_LABELS = os.path.join(DATA_DIR, "aqsol_so_labels.json")
DEFAULT_REPORT = os.path.join(DATA_DIR, "aqsol_so_filter_report.json")
# Reuse the sibling frag20 split CSVs / tar if present (avoid re-download):
SIBLING_DIR = os.path.join(os.path.dirname(HERE), "experimental_frag20", "data")
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
# element handling (dependency-free - rdkit C DLLs are AppLocker-blocked on
# the local dev box; the xyz files are the ground truth anyway)
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
B_Z = 5
S_Z = 16

# organic-subset SMILES symbols (incl. aromatic) -> Z; everything else must
# be in brackets and is handled by the bracket parser
_ORGANIC = {"B": 5, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "F": 9,
            "Cl": 17, "Br": 35, "I": 53,
            "b": 5, "c": 6, "n": 7, "o": 8, "p": 15, "s": 16}

# SMILES S=O motif tests. Sulfoxide/sulfone/sulfonamide sulfur is always
# written uppercase 'S' in RDKit canonical SMILES (lower-case 's' = aromatic
# thiophene, NOT =O). Same counting rule as the FRIDAY 7 AUGUST audit
# (Acsol-100K S=O coverage section). No RDKit needed.
_RE_SULFONE = re.compile(r"S\(=O\)\(=O\)", re.I)
_RE_SULFONAMIDE = re.compile(r"S\(=O\)\(=O\)N\b|NS\(=O\)", re.I)
_RE_SULFONATE = re.compile(r"OS\(=O\)", re.I)
_SO_STR = "S(=O)"


def has_so_bond(smiles):
    """True if the SMILES contains an S(=O) motif (sulfoxide, sulfone,
    sulfonamide, sulfonic/sulfuric ester). Aromatic sulfur 's' is never
    detected - it has no =O bond."""
    return _SO_STR in smiles


def so_subtype(smiles):
    """Short chemistry tag for a molecule that has S=O (best-effort)."""
    has_sulfone = bool(_RE_SULFONE.search(smiles))
    if _RE_SULFONAMIDE.search(smiles):
        return "sulfonamide_sulfone" if has_sulfone else "sulfonamide"
    if _RE_SULFONATE.search(smiles):
        return "sulfonic_ester" if has_sulfone else "sulfate_ester"
    if has_sulfone:
        return "sulfone"
    return "sulfoxide"


def smiles_elements(smiles):
    """Counter of atomic numbers for a SMILES string, including explicit
    bracket H. Returns (Counter, n_unknown_tokens)."""
    counts = Counter()
    unknown = 0
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = i + 1
            while j < len(smiles) and smiles[j].isdigit():
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
            if ch == "C" and i + 1 < len(smiles) and smiles[i + 1] == "l":
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
    """Parse a Frag20 .xyz file. Returns (np.int32 atNUM, np.float64 atXYZ)."""
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
    # prefer the sibling (experimental_frag20) copies; else this folder
    candidates = [os.path.join(SIBLING_DIR, "split", f"frag20_{s}.csv")
                  for s in ("train", "valid", "test")]
    for s in ("train", "valid", "test"):
        p = os.path.join(csv_dir, f"frag20_{s}.csv")
        existing = [os.path.join(SIBLING_DIR, "split", f"frag20_{s}.csv")]
        if any(os.path.exists(e) for e in existing):
            import shutil
            os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(p):
                shutil.copy(existing[0], p)
            continue
        if not os.path.exists(p) and download:
            os.makedirs(csv_dir, exist_ok=True)
            print(f"  downloading {s}.csv from whoyouwith91 ...")
            urllib.request.urlretrieve(CSV_URLS[s], p)
    return {s: os.path.join(csv_dir, f"frag20_{s}.csv")
            for s in ("train", "valid", "test")}


def ensure_tar(csv_dir, tar_path, download):
    if not os.path.exists(tar_path):
        sibling_tar = os.path.join(SIBLING_DIR, "Frag20-Aqsol-100K.tar.bz2")
        if os.path.exists(sibling_tar):
            os.makedirs(os.path.dirname(tar_path), exist_ok=True)
            print(f"  reusing sibling tar {sibling_tar}")
            import shutil
            shutil.copy2(sibling_tar, tar_path)
        elif download:
            os.makedirs(os.path.dirname(tar_path), exist_ok=True)
            print("  downloading Frag20-Aqsol-100K.tar.bz2 (88.9 MB) from NYU IMA ...")
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
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of extracted molecules (smoke testing)")
    args = ap.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)

    print("=" * 66)
    print("  PREPARE AQSOL-100K S(=O) SUPPLEMENT  (sulfur-oxygen fix)")
    print("=" * 66)

    csv_paths = ensure_csvs(args.csv_dir, download=not args.no_download)
    missing = [p for p in csv_paths.values() if not os.path.exists(p)]
    assert not missing, f"missing CSVs: {missing} (remove --no_download or supply --csv_dir)"
    tar_path = ensure_tar(args.csv_dir, args.tar_path, download=not args.no_download)
    assert os.path.exists(tar_path), f"missing tar: {tar_path}"

    print(f"  CSVs: {', '.join(os.path.basename(p) for p in csv_paths.values())}")
    print(f"  tar:  {os.path.getsize(tar_path) / 1e6:.1f} MB")
    print(f"  geom: {args.geom}_xyz (QM-optimized B3LYP/6-31G* per molecule)")

    rows = read_csv_rows(csv_paths)
    print(f"  total rows across splits: {len(rows)}")

    # ---- filter by SMILES first (fast, no tar needed) ----
    dropped_boron = 0
    dropped_vocab = 0
    dropped_no_so = 0
    candidates = []
    subtype_hist = Counter()
    for r in rows:
        smi = r["QM_SMILES"]
        zc, _ = smiles_elements(smi)
        zs = set(zc.keys())
        if B_Z in zs:
            dropped_boron += 1
            continue
        if not zs.issubset(AQM_VOCAB):
            dropped_vocab += 1
            continue
        if not has_so_bond(smi):
            dropped_no_so += 1
            continue
        st = so_subtype(smi)
        subtype_hist[st] += 1
        candidates.append((r, st))

    print(f"  SMILES filter: {len(candidates)} S=O candidates "
          f"(subtype hist {dict(subtype_hist)})")
    print(f"    dropped: boron {dropped_boron}, out-of-vocab {dropped_vocab}, "
          f"no-S=O {dropped_no_so}")

    # ---- extract geometries for candidates from the tar ----
    wanted = {}
    for r, st in candidates:
        member = xyz_member_path(r["SourceFile"], r["ID"], args.geom)
        mid = f"frag20_{r['SourceFile']}_{int(float(r['ID']))}"
        wanted[member] = (mid, r, st)

    n_extracted = 0
    n_missing_member = 0
    n_parse_fail = 0
    n_geom_mismatch = 0
    xyz_vocab_drops = 0
    xyz_boron_drops = 0
    xyz_no_so = 0
    element_hist = Counter()

    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    with h5py.File(args.out_h5, "w") as h5, \
            tarfile.open(tar_path, "r:bz2") as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            mid, r, st = wanted[member.name]
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

            if B_Z in zs:
                xyz_boron_drops += 1
                continue
            if not zs.issubset(AQM_VOCAB):
                xyz_vocab_drops += 1
                continue
            if S_Z not in zs:
                xyz_no_so += 1  # SMILES said S=O, xyz lacks S entirely (warn)

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
            grp.attrs["so_subtype"] = st
            for zz in zs:
                element_hist[zz] += 1
            n_extracted += 1
            if args.limit and n_extracted >= args.limit:
                break

    n_missing_member = -1 if args.limit is not None else (
        len(wanted) - n_extracted - n_parse_fail - xyz_boron_drops - xyz_vocab_drops)

    print(f"  extracted {n_extracted} geometries to {os.path.basename(args.out_h5)}")
    print(f"    missing tar members: {n_missing_member}, parse failures: {n_parse_fail}")
    print(f"    xyz drops - boron {xyz_boron_drops}, vocab {xyz_vocab_drops}, "
          f"no-S-in-xyz {xyz_no_so}")
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
                "so_subtype": g.attrs["so_subtype"],
            }
    with open(args.out_labels, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"  labels: {len(labels)} molecules -> {os.path.basename(args.out_labels)}")

    report = {
        "total_rows": len(rows),
        "smiles_filter": {
            "candidates_so": len(candidates),
            "so_subtype_hist": dict(subtype_hist),
            "dropped_boron": dropped_boron,
            "dropped_out_of_vocab": dropped_vocab,
            "dropped_no_so": dropped_no_so,
        },
        "xyz_filter": {
            "extracted": n_extracted,
            "missing_tar_member": n_missing_member,
            "parse_failures": n_parse_fail,
            "dropped_boron": xyz_boron_drops,
            "dropped_out_of_vocab": xyz_vocab_drops,
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
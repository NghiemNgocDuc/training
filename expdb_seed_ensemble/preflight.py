"""Preflight: verify all required inputs exist; optionally gather them into
./inputs/ so the bundle is self-contained for a git clone on Vast.

Usage:
  python preflight.py            # check only
  python preflight.py --gather   # copy inputs from ../expdb_vast into ./inputs/
"""

import argparse
import os
import shutil
import sys

import common_io as cio

FILES = [
    ("stage2_correction.pt", cio.path_stage2),
    ("freesolv_conformers.hdf5", cio.path_freesolv_h5),
    ("expdb_conformers.hdf5", cio.path_expdb_h5),
    ("split_check/train_ids.json", cio.path_split.__call__ if False else
     (lambda: cio.path_split("train"))),
    ("split_check/val_ids.json", lambda: cio.path_split("val")),
    ("split_check/test_ids.json", lambda: cio.path_split("test")),
    ("predictions_ensemble.csv", cio.path_truth_csv),
    ("database.json", cio.path_labels_json),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gather", action="store_true",
                    help="copy inputs from ../expdb_vast into ./inputs/")
    args = ap.parse_args()

    ok = True
    for name, fn in FILES:
        try:
            p = fn()
            print(f"  OK   {name:35s} -> {p}")
        except FileNotFoundError as e:
            ok = False
            print(f"  MISS {name}")
    if ok:
        print("\n[preflight] ALL INPUTS PRESENT")
        return 0

    if args.gather:
        src_root = cio.VAST
        dst = cio.BUNDLE_INPUTS
        copies = [
            ("stage2_correction.pt", os.path.join(src_root, "stage2_correction.pt")),
            ("freesolv_conformers.hdf5", os.path.join(src_root, "freesolv_conformers.hdf5")),
            ("expdb_conformers.hdf5", os.path.join(src_root, "results", "expdb_conformers.hdf5")),
            ("split_check/train_ids.json", os.path.join(src_root, "results", "split_check", "train_ids.json")),
            ("split_check/val_ids.json", os.path.join(src_root, "results", "split_check", "val_ids.json")),
            ("split_check/test_ids.json", os.path.join(src_root, "results", "split_check", "test_ids.json")),
            ("predictions_ensemble.csv", os.path.join(src_root, "results", "predictions_ensemble.csv")),
            ("database.json", os.path.join(src_root, "freesolv_cache", "database.json")),
        ]
        for rel, src in copies:
            if not os.path.exists(src):
                print(f"  [gather] source missing: {src}")
                continue
            d = os.path.join(dst, os.path.dirname(rel))
            os.makedirs(d, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, rel))
            print(f"  [gather] {rel}  ({os.path.getsize(os.path.join(dst, rel))} B)")
        print("\n[gather] done — re-run: python preflight.py")
        return 0

    print("\n[preflight] MISSING FILES. Fix options:")
    print("  a) run on a machine that has expdb_vast/:  python preflight.py --gather")
    print("     then commit expdb_seed_ensemble/inputs/ to git (use git add -f)")
    print("  b) or place the files manually into expdb_seed_ensemble/inputs/")
    return 1


if __name__ == "__main__":
    sys.exit(main())

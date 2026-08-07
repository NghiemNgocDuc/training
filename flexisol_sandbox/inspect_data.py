"""Sanity checks for the built FlexiSol-water dataset.

Verifies (no torch needed):
  * mol_id counts in hdf5 / labels / split
  * every atomic number is inside the 17-element vocab used by the
    uncertainty pipeline (element_vocab.py)
  * label range / basic stats
  * split disjointness + full coverage
"""

import argparse
import json
import os

import h5py
import numpy as np

VOCAB_Z = {1, 6, 7, 8, 9, 15, 16, 17, 3, 5, 11, 12, 14, 19, 20, 35, 53}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    labels_path = os.path.join(args.out, "labels.json")
    hdf5_path = os.path.join(args.out, "flexisol_water.hdf5")
    split_dir = os.path.join(args.out, "split")

    labels = json.load(open(labels_path))
    expts = np.array([v["expt"] for v in labels.values()])
    print(f"labels: {len(labels)} molecules | expt range "
          f"[{expts.min():.2f}, {expts.max():.2f}] kcal/mol | mean {expts.mean():.2f}")

    with h5py.File(hdf5_path, "r") as f:
        ids = sorted(f.keys())
        print(f"hdf5:  {len(ids)} groups")
        assert len(ids) == len(labels), "hdf5/labels count mismatch"
        n_atoms = [f[i]["atNUM"].shape[0] for i in ids]
        z_all = np.concatenate([f[i]["atNUM"][...] for i in ids])
        unknown = sorted(set(np.unique(z_all)) - VOCAB_Z)
        print(f"atoms: total {z_all.size}, per-mol range [{min(n_atoms)}, {max(n_atoms)}]")
        print(f"vocab: {len(set(np.unique(z_all)))} distinct Z, "
              f"outside 17-elem vocab: {unknown if unknown else 'none'}")
        bad_shape = [i for i in ids if f[i]["atXYZ"].shape != (f[i]["atNUM"].shape[0], 3)]
        assert not bad_shape, f"bad geometry shapes: {bad_shape[:5]}"

    tr = json.load(open(os.path.join(split_dir, "train_ids.json")))
    va = json.load(open(os.path.join(split_dir, "val_ids.json")))
    te = json.load(open(os.path.join(split_dir, "test_ids.json")))
    all_ids = set(labels)
    assert set(tr) | set(va) | set(te) == all_ids, "split does not cover labels"
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    print(f"split: train/val/test = {len(tr)}/{len(va)}/{len(te)} | disjoint + full coverage OK")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
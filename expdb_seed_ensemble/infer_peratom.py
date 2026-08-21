"""Per-atom, per-seed inference for the Exp-DB seed ensemble.

For each seed checkpoint (results_seeds/finetuned_seed{S}.pt):
  1. Exp-DB: every molecule, ALL stored TTA conformers -> per-atom P averaged
     over conformers; molecular total E = sum(P). Cross-check gate on the
     first molecule of each seed: |sum(P_conf0) - E_direct(conf0)| < 1e-6.
  2. FreeSolv fold-0 TRAIN molecules (411, single stored conformer): per-atom
     P -> mu_T^(k) and train-lambda stats (deployment-safe reference).

Outputs: results_seeds/peratom_seed{S}.pkl  (one per seed)

Usage: python infer_peratom.py [--seeds 42,123,7,2024,999]
"""

import argparse
import os
import pickle
import time

import numpy as np

import common_io as cio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,7,2024,999")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}", flush=True)

    import h5py
    import pandas as pd

    truth = cio.load_truth()
    expdb_ids = [str(i) for i in truth.keys()]
    h5_expdb = cio.path_expdb_h5()
    with h5py.File(h5_expdb, "r") as f:
        expdb_ids = [m for m in expdb_ids if m in f]
    print(f"[infer] Exp-DB molecules with geometry+label: {len(expdb_ids)}",
          flush=True)

    train_ids = json_load(cio.path_split("train"))
    h5_free = cio.path_freesolv_h5()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(out_dir, "results_seeds")
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        t0 = time.time()
        model = cio.load_seed_model(device, seed, out_dir)
        E = np.zeros(len(expdb_ids))
        n_conf = np.zeros(len(expdb_ids), dtype=int)
        P_store = {}
        gate_done = False

        with h5py.File(h5_expdb, "r") as f:
            for i, mid in enumerate(expdb_ids):
                g = f[mid]
                z = g["atNUM"][0]
                xyz = g["atXYZ"][...]
                Ps = [cio.per_atom_predict(model, z, xyz[c], device)
                      for c in range(xyz.shape[0])]
                P_avg = np.mean(Ps, axis=0)
                E[i] = float(P_avg.sum())
                n_conf[i] = len(Ps)
                P_store[mid] = P_avg.astype(np.float64)

                if not gate_done:
                    e_direct = cio.energy_predict(model, z, xyz[0], device)
                    diff = abs(float(Ps[0].sum()) - e_direct)
                    print(f"[infer seed {seed}] gate first-mol "
                          f"|sum(P)-E_direct| = {diff:.2e}", flush=True)
                    assert diff < 1e-6, "per-atom sum != direct energy"
                    gate_done = True
                if (i + 1) % 50 == 0 or (i + 1) == len(expdb_ids):
                    print(f"[infer seed {seed}] expdb {i+1}/{len(expdb_ids)}",
                          flush=True)

        # FreeSolv train molecules (single stored conformer)
        P_train = {}
        with h5py.File(h5_free, "r") as f:
            miss = [m for m in train_ids if m not in f]
            assert not miss, f"train mols missing from freesolv hdf5: {miss[:3]}"
            for j, mid in enumerate(train_ids):
                g = f[mid]
                P_train[mid] = cio.per_atom_predict(
                    model, g["atNUM"][...], g["atXYZ"][...], device).astype(np.float64)
                if (j + 1) % 100 == 0 or (j + 1) == len(train_ids):
                    print(f"[infer seed {seed}] freesolv-train "
                          f"{j+1}/{len(train_ids)}", flush=True)

        mu_T = float(np.concatenate(list(P_train.values())).mean())
        out = {
            "seed": seed,
            "expdb_ids": expdb_ids,
            "E": E,
            "n_conf": n_conf,
            "P_expdb": P_store,
            "P_train": P_train,
            "mu_T_kcal": mu_T,
        }
        pk = os.path.join(out_dir, f"peratom_seed{seed}.pkl")
        with open(pk, "wb") as fh:
            pickle.dump(out, fh)
        print(f"[infer seed {seed}] DONE {time.time()-t0:.0f}s  "
              f"E mean={E.mean():+.2f}  mu_T={mu_T:+.4f}  -> {pk}", flush=True)


def json_load(p):
    import json
    with open(p) as f:
        return json.load(f)


if __name__ == "__main__":
    main()

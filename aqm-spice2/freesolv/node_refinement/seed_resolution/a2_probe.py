"""Part A2 probe: original checkpoints x conformer protocol.

Question: are seeds 7/2024's catastrophic stored-hdf5 predictions (MAE ~38/58
on test) a CONFORMER-SOURCE artifact fixable by re-scoring with fresh
conformers (the training-time protocol: ETKDGv3 seed 42, prune 0.5, +MMFF),
restoring a genuine 5-seed ensemble?

Protocols (129 test molecules, fold-0, ORIGINAL checkpoints only):
  (a) stored-hdf5 single conf  -- reproduces seed_predictions_all642.csv regime
  (b) fresh ETKDGv3+MMFF single conf (conformer[0] of the 5-draw protocol)
  (c) fresh ETKDGv3+MMFF TTA-5 (mean over kept confs, same as
      deep_ensemble.conformer_average / the original eval protocol)

Outputs -> seed_resolution/a2_probe_report.json + a2_probe_predictions.csv.
CPU only, < 10 min.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE)))))
FREESOLV = os.path.join(REPO, "aqm-spice2", "freesolv")
sys.path.insert(0, FREESOLV)

import deep_ensemble as de
from element_vocab import build_one_hot
from freesolv_dataset import load_freesolv_labels

SEEDS = [42, 123, 7, 2024, 999]
CKPTS = {s: os.path.join(FREESOLV, "deep_ensemble", f"seed_{s}",
                         f"ensemble_seed{s}.pt") for s in SEEDS}
H5 = os.path.join(REPO, "freesolv_conformers.hdf5")
LABELS = os.path.join(REPO, "Data", "FreeSolv", "database.json")
SPLIT = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                     "cv_results_full", "fold_0")


def fresh_confs(smiles, n):
    """ETKDGv3 (randomSeed 42, pruneRmsThresh 0.5) + MMFF; returns (z, [pos...])
    or None on failure (mirrors deep_ensemble.conformer_average)."""
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
    if not conf_ids:
        return None
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    try:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    except Exception:
        pass
    z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()],
                              dtype=np.int32), dtype=torch.long)
    n_avail = min(n, mol.GetNumConformers())
    pos = [torch.tensor(np.array(mol.GetConformer(i).GetPositions(),
                                 dtype=np.float64), dtype=torch.float)
           for i in range(n_avail)]
    return z, pos


def hdf5_geom(mid):
    import h5py
    with h5py.File(H5, "r") as f:
        g = f[mid]
        return (torch.tensor(g["atNUM"][...], dtype=torch.long),
                torch.tensor(g["atXYZ"][...], dtype=torch.float))


def predict(model, device, mids, geoms):
    """geoms: dict mid -> (z, pos_tensor) single geometry."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    out = {}
    with torch.no_grad():
        for m in mids:
            z, pos = geoms[m]
            d = Data(z=z, pos=pos).to(device)
            x = build_one_hot(d, device)
            p = model(x, d.pos, d.batch).view(-1) * de.EV_TO_KCAL
            out[m] = float(p.cpu())
    return out


def main():
    t0 = time.time()
    labels = load_freesolv_labels(LABELS)
    tr, va, te = de.load_frozen_split(SPLIT, labels)
    mids = te
    device = torch.device("cpu")
    y = np.array([labels[m]["expt"] for m in mids])
    ref = pd.read_csv(os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                                   "seed_predictions_all642.csv")).set_index("mol_id")

    h5_geoms = {m: hdf5_geom(m) for m in mids}
    fresh = {m: fresh_confs(labels[m]["smiles"], 5) for m in mids}
    n_fresh_fail = sum(1 for m in mids if fresh[m] is None)
    print(f"[conf] fresh conformers: {len(mids)-n_fresh_fail}/{len(mids)} ok, "
          f"{n_fresh_fail} hdf5-fallback", flush=True)

    results, rows = {}, []
    for s in SEEDS:
        model = de.build_model(device)
        model.load_state_dict(torch.load(CKPTS[s], map_location=device,
                                         weights_only=True))
        model.eval()
        p_h5 = predict(model, device, mids, h5_geoms)
        p_single = {}
        for m in mids:
            if fresh[m] is not None:
                p_single[m] = predict(model, device, [m],
                                      {m: (fresh[m][0], fresh[m][1][0])})[m]
            else:
                p_single[m] = p_h5[m]
        p_tta = {}
        for m in mids:
            if fresh[m] is not None:
                vals = [predict(model, device, [m],
                                {m: (fresh[m][0], pos)})[m]
                        for pos in fresh[m][1]]
                p_tta[m] = float(np.mean(vals))
            else:
                p_tta[m] = p_h5[m]
        a_h5 = np.array([p_h5[m] for m in mids]) - y
        a_s1 = np.array([p_single[m] for m in mids]) - y
        a_t5 = np.array([p_tta[m] for m in mids]) - y
        r = ref.loc[mids, f"pred_seed{s}"].to_numpy()
        results[s] = {
            "mae_hdf5": float(np.abs(a_h5).mean()),
            "mae_fresh_single": float(np.abs(a_s1).mean()),
            "mae_fresh_tta5": float(np.abs(a_t5).mean()),
            "mae_repair_csv": float(np.abs(r - y).mean()),
            "max_abs_hdf5": float(np.abs(a_h5).max()),
            "max_abs_fresh_single": float(np.abs(a_s1).max()),
            "max_abs_fresh_tta5": float(np.abs(a_t5).max()),
            "n_hdf5_gt20": int((np.abs(a_h5) > 20).sum()),
            "n_fresh_single_gt20": int((np.abs(a_s1) > 20).sum()),
            "n_fresh_tta5_gt20": int((np.abs(a_t5) > 20).sum()),
        }
        print(f"[seed {s}] hdf5 MAE={results[s]['mae_hdf5']:.3f} | "
              f"fresh-single MAE={results[s]['mae_fresh_single']:.3f} | "
              f"fresh-TTA5 MAE={results[s]['mae_fresh_tta5']:.3f} | "
              f"repair_csv MAE={results[s]['mae_repair_csv']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for m in mids:
            rows.append({"seed": s, "mol": m,
                         "p_hdf5": p_h5[m], "p_fresh_single": p_single[m],
                         "p_fresh_tta5": p_tta[m], "true": float(labels[m]["expt"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "a2_probe_predictions.csv"), index=False)
    with open(os.path.join(HERE, "a2_probe_report.json"), "w") as f:
        json.dump({"summary": results, "runtime_s": time.time() - t0}, f, indent=2)
    print(f"[save] a2_probe_report.json + a2_probe_predictions.csv -> {HERE}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
"""Per-seed inference over ALL 642 FreeSolv molecules (single stored conformer,
identical protocol to deep_ensemble.py predictions.csv).

Outputs seed_predictions_all642.csv: mol_id, pred_seed42..pred_seed999,
ensemble_mean, ensemble_std, true_value (kcal/mol).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deep_ensemble as de
from element_vocab import build_one_hot
from freesolv_dataset import load_freesolv_labels

REPO = de.REPO_ROOT
SEED_DIRS = {
    s: os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed", f"seed_{s}", f"ensemble_seed{s}.pt")
    for s in (42, 123, 7, 2024, 999)
}
SEED_DIRS = {
    s: os.path.join(os.path.dirname(os.path.abspath(__file__)), f"seed_{s}", f"ensemble_seed{s}.pt")
    for s in (42, 123, 7, 2024, 999)
}
CONFORMERS = os.path.join(REPO, "freesolv_conformers.hdf5")
LABELS = os.path.join(REPO, "Data", "FreeSolv", "database.json")
SPLIT = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repair_data")


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")
    all_labels = load_freesolv_labels(LABELS)
    train_ids, val_ids, test_ids = de.load_frozen_split(SPLIT, all_labels)
    all_ids = train_ids + val_ids + test_ids

    SimpleDataset = de.simple_dataset_cls(CONFORMERS, all_labels)
    ds = SimpleDataset(all_ids)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    rows = {m: {} for m in all_ids}
    for s, ckpt in SEED_DIRS.items():
        assert os.path.exists(ckpt), f"missing {ckpt}"
        model = de.build_model(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * de.EV_TO_KCAL
                for mid, p in zip(data.mol_id, pred.cpu().tolist()):
                    rows[mid][f"pred_seed{s}"] = p
        print(f"seed {s}: done in {time.time()-t0:.1f}s", flush=True)

    import csv
    with open(os.path.join(OUT, "seed_predictions_all642.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id"] + [f"pred_seed{s}" for s in SEED_DIRS] +
                   ["ensemble_mean", "ensemble_std", "true_value"])
        for m in all_ids:
            preds = np.array([rows[m][f"pred_seed{s}"] for s in SEED_DIRS])
            w.writerow([m] + [f"{v:.6f}" for v in preds] +
                       [f"{preds.mean():.6f}", f"{preds.std():.6f}",
                        f"{all_labels[m]['expt']:.6f}"])
    print(f"saved -> {os.path.join(OUT, 'seed_predictions_all642.csv')} "
          f"({len(all_ids)} molecules)")


if __name__ == "__main__":
    main()
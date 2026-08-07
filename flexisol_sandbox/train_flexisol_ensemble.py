"""Train a 5-member DimeNet+ deep ensemble on FlexiSol-water (sandbox).

Mirrors the verified deep_ensemble.py recipe as closely as the sandbox
allows:
  * DimeNet+ arch via the experiment package's common.build_model
  * per-member train on the FlexiSol train split, MSE-in-eV, Adam lr=1e-4,
    wd=1e-5, batch 8, ReduceLROnPlateau, early stop on val MAE, best-ckpt
  * optional --init transfer: warm-start each member from the FreeSolv
    stage2_correction.pt (tests the transfer hypothesis); default scratch.

Artifacts (same layout the approach-1 script expects):
  <out>/seed_<s>/ensemble_seed<s>.pt          (5 members)
  <out>/aggregate/per_molecule.csv            (test-side: seeds' preds,
                                               mean, std, true, abs_err,
                                               has_halogen_Br_I)

No existing experiment code is modified; this file imports read-only.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

SANDBOX_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.abspath(os.path.join(
    SANDBOX_ROOT, "..", "aqm-spice2", "freesolv", "experimental_uncertainty_refine"))
sys.path.insert(0, EXPERIMENT_DIR)

from common import (  # noqa: E402  (read-only import of the frozen package)
    EV_TO_KCAL, SEEDS, set_seed, load_frozen_split, load_freesolv_labels,
    simple_dataset_cls, build_model, evaluate,
)
from element_vocab import build_one_hot  # noqa: E402

FREESOLV_CORRECTION = os.path.join(
    EXPERIMENT_DIR, "..", "..", "..", "aqm-spice2", "pipeline",
    "results_full", "stage2_correction.pt")


def train_member(seed, init_ckpt, train_ds, val_ds, device, epochs, patience,
                 lr, batch_size):
    set_seed(seed)
    model = build_model(device)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_mae, best_state, stale = float("inf"), None, 0
    for ep in tqdm(range(1, epochs + 1), desc=f"seed{seed} train",
                   leave=False, unit="epoch"):
        model.train()
        for data in tr:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            dG = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            loss = torch.nn.functional.mse_loss(pred, dG)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        p, e = [], []
        with torch.no_grad():
            for data in va:
                data = data.to(device)
                x = build_one_hot(data, device)
                p.append((model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL).cpu())
                e.append(data.y_dG.view(-1).cpu())
        p = torch.cat(p).numpy(); e = torch.cat(e).numpy()
        mae = float(np.mean(np.abs(p - e)))
        sched.step(mae)
        if mae < best_val_mae:
            best_val_mae, best_state, stale = mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return model, best_state, best_val_mae


def write_aggregate(seeds, ensemble_dir, test_ids, labels, conformers, device, out_csv):
    ds = simple_dataset_cls(conformers, labels)
    loader = DataLoader(ds(test_ids), batch_size=8, shuffle=False)
    per_seed = {s: {} for s in seeds}
    for seed in seeds:
        model = build_model(device)
        model.load_state_dict(torch.load(
            os.path.join(ensemble_dir, f"seed_{seed}", f"ensemble_seed{seed}.pt"),
            map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                for mid, v in zip(data.mol_id, pred.tolist()):
                    per_seed[seed][mid] = v

    import h5py
    from common import HALOGEN_Z
    has_hu = {}
    with h5py.File(conformers, "r") as f:
        for mid in test_ids:
            z = set(f[mid]["atNUM"][...].tolist())
            has_hu[mid] = 1 if z & HALOGEN_Z else 0

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id"] + [f"pred_seed{s}" for s in seeds] +
                   ["ensemble_mean", "ensemble_std", "true_value",
                    "abs_error", "has_halogen_Br_I"])
        for mid in test_ids:
            vals = [per_seed[s][mid] for s in seeds]
            mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1))
            expt = labels[mid]["expt"]
            w.writerow([mid] + [f"{v:.6f}" for v in vals] +
                       [f"{mean:.6f}", f"{std:.6f}", f"{expt:.6f}",
                        f"{abs(mean - expt):.6f}", f"{has_hu[mid]}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ensemble")
    ap.add_argument("--data", default="out")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--init", choices=["scratch", "transfer"], default="scratch")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = load_freesolv_labels(os.path.join(args.data, "labels.json"))
    train_ids, val_ids, test_ids = load_frozen_split(
        os.path.join(args.data, "split"), labels)
    if args.smoke:
        train_ids, val_ids, test_ids = train_ids[:8], val_ids[:8], test_ids[:10]
        args.epochs = min(args.epochs, 6)
        args.patience = min(args.patience, 2)
    conformers = os.path.join(args.data, "flexisol_water.hdf5")
    ds = simple_dataset_cls(conformers, labels)
    init_ckpt = FREESOLV_CORRECTION if args.init == "transfer" else None
    if init_ckpt and not os.path.isfile(init_ckpt):
        print(f"[warn] transfer init not found: {init_ckpt} -> scratch")
        init_ckpt = None

    for seed in SEEDS:
        model, best_state, best_val = train_member(
            seed, init_ckpt, ds(train_ids), ds(val_ids), device,
            args.epochs, args.patience, args.lr, args.batch_size)
        seed_dir = os.path.join(args.out, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        torch.save(best_state, os.path.join(seed_dir, f"ensemble_seed{seed}.pt"))
        print(f"  seed {seed}: best val MAE = {best_val:.4f} kcal/mol -> saved")

    test_ids = [m for m in test_ids if m in labels]
    per_molecule = os.path.join(args.out, "aggregate", "per_molecule.csv")
    write_aggregate(SEEDS, args.out, test_ids, labels, conformers, device,
                    per_molecule)
    print(f"  aggregate -> {per_molecule}")


if __name__ == "__main__":
    main()
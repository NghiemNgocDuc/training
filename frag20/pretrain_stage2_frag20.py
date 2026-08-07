"""Stage 2: pretrain a DimeNet+ CORRECTION model from scratch on Frag20,
anchored by the frozen stage-1 vacuum model.

EXPERIMENTAL sandbox. Mirrors the verified AQM
pipeline/train_stage2_correction.py, adapted for Frag20:
  * frozen vacuum model (from stage 1, same architecture as AQM stage 2:
    vacuum = 4 blocks, correction = 3 blocks)
  * primary target = dG_solv in eV = watEnergy - gasEnergy (the exact paired
    electronic-energy difference Frag20 ships; CalcSol = dG*627.509 kcal/mol)
  * secondary regularizer lambda_total on the total-energy residual
    (vacuum + correction) vs watEnergy - atomic refs (refs fit on the TRAIN
    split of stage-2's own data, leak-free)
  * NO forces (Frag20 ships none) -> lambda_force=0
  * dataset's OWN fixed split used for train/valid; test held out
  * optimizer: lr 1e-3, batch 16, epochs 200, patience 10,
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6), grad clip 10

Usage:
  # GPU (Vast):
  python pretrain_stage2_frag20.py --device cuda \
      --stage1_ckpt output/stage1_scratch.pt \
      --stage1_refs output/stage1_scratch_refs.json --output_dir output
  # local smoke test (CPU, tiny):
  python pretrain_stage2_frag20.py --device cpu --max_structures 64 \
      --epochs 3 --output_dir output_smoke
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from common_scratch import (Frag20Dataset, CachedListDataset, DEFAULT_FRAG20_H5,
                            DEFAULT_FRAG20_LABELS, DEFAULT_SEED,
                            build_model, fit_atomic_references,
                            compute_molecular_reference, set_seed)
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")

mse = torch.nn.MSELoss()


def combined_loss(corr_pred, dG_true, total_pred, y_shifted, n_atoms,
                  lambda_total):
    """Mirrors the AQM stage-2 loss: primary dG MSE (eV) + lambda_total *
    total-energy residual MSE (per-atom normalized)."""
    loss = mse(corr_pred, dG_true)
    loss += lambda_total * mse(total_pred / n_atoms, y_shifted / n_atoms)
    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=DEFAULT_FRAG20_H5)
    ap.add_argument("--labels", default=DEFAULT_FRAG20_LABELS)
    ap.add_argument("--stage1_ckpt", default=os.path.join(HERE, "output", "stage1_scratch.pt"))
    ap.add_argument("--stage1_refs", default=os.path.join(HERE, "output", "stage1_scratch_refs.json"))
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lambda_total", type=float, default=0.05)
    ap.add_argument("--max_structures", type=int, default=None,
                    help="cap on TRAIN rows (smoke tests); valid/test unaffected")
    ap.add_argument("--ckpt_tag", default="stage2_scratch")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable - falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)

    labels = json.load(open(args.labels))
    all_ids = sorted(labels.keys())
    train_ids = [m for m in all_ids if labels[m]["split"] == "train"]
    valid_ids = [m for m in all_ids if labels[m]["split"] == "valid"]
    test_ids = [m for m in all_ids if labels[m]["split"] == "test"]
    print(f"Frag20 labels: {len(train_ids)} train / {len(valid_ids)} valid / "
          f"{len(test_ids)} test (dataset's own fixed split)")
    if args.max_structures is not None:
        rng = np.random.RandomState(args.seed)
        kept = set(rng.choice(train_ids, size=min(args.max_structures, len(train_ids)),
                              replace=False).tolist())
        print(f"  [smoke] capping TRAIN to {len(kept)} molecules")
        train_ids = [m for m in train_ids if m in kept]

    def build(items):
        return CachedListDataset([Frag20Dataset(args.h5, [m], labels)[0] for m in items])

    train_ds = build(train_ids)
    valid_ds = build(valid_ids)
    test_ds = build(test_ids)
    print(f"train {len(train_ds)} | valid {len(valid_ds)} | test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- stage-2 atomic refs: fit on stage-2's OWN train split (watEnergy),
    # mirroring the verified pipeline's 2026-08-01 decision ----
    ref_energies = fit_atomic_references(train_ds, ELEMENT_TO_IDX, NUM_ELEMENTS,
                                         energy_attr="y_wat_eV").to(device)

    vacuum_model = build_model(device, num_blocks=4)
    vacuum_model.load_state_dict(torch.load(args.stage1_ckpt, map_location=device,
                                            weights_only=True))
    for p in vacuum_model.parameters():
        p.requires_grad_(False)
    vacuum_model.eval()
    print(f"  loaded frozen vacuum: {args.stage1_ckpt}")

    correction_model = build_model(device, num_blocks=3)
    print(f"correction params: {sum(p.numel() for p in correction_model.parameters()):,} "
          f"(DimeNet+, 3 blocks)")
    optimizer = optim.Adam(correction_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.ckpt_tag}.pt")

    def run_epoch(loader, train=True):
        correction_model.train() if train else correction_model.eval()
        total = 0.0
        dG_sum = 0.0
        dG_n = 0
        with torch.set_grad_enabled(train):
            for data in loader:
                data = data.to(device)
                if train:
                    optimizer.zero_grad()
                x = build_one_hot(data, device)
                mol_ref = compute_molecular_reference(
                    x, data.batch, ref_energies, data.num_graphs)
                y_shifted = data.y_wat_eV - mol_ref
                with torch.no_grad():
                    vacuum_energy = vacuum_model(x, data.pos, data.batch)
                corr_energy = correction_model(x, data.pos, data.batch)
                total_energy = vacuum_energy + corr_energy
                n_atoms = torch.bincount(data.batch).float()
                dG_true = data.y_dG_eV
                loss = combined_loss(corr_energy.view(-1), dG_true.view(-1),
                                     total_energy.view(-1), y_shifted.view(-1),
                                     n_atoms, args.lambda_total)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(correction_model.parameters(), 10.0)
                    optimizer.step()
                total += loss.item() * data.num_graphs
                dG_sum += mse(corr_energy.view(-1), dG_true.view(-1)).item() * data.num_graphs
                dG_n += data.num_graphs
        return total / len(loader.dataset), dG_sum / dG_n

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = args.epochs
    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dG = run_epoch(train_loader, train=True)
        val_loss, val_dG = run_epoch(valid_loader, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(correction_model.state_dict(), ckpt_path)
        else:
            stale += 1
        print(f"  epoch {epoch:3d}/{args.epochs} | train {tr_loss:.6f} "
              f"(dG {np.sqrt(tr_dG)*23.0605:.3f} kcal) | val {val_loss:.6f} "
              f"(dG {np.sqrt(val_dG)*23.0605:.3f} kcal) | best {best_val:.6f} "
              f"(ep {best_epoch}) | {dt:5.1f}s", flush=True)
        if stale >= args.patience:
            stop_epoch = epoch
            print(f"  early stopped at epoch {epoch} (patience {args.patience})")
            break

    correction_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))
    _, test_dG = run_epoch(test_loader, train=False)
    print(f"\n  DONE in {(time.time() - t0_all) / 60:.1f} min | best val {best_val:.6f} "
          f"(ep {best_epoch}) | early stop {stop_epoch} | "
          f"held-out test dG RMSE {np.sqrt(test_dG)*23.0605:.3f} kcal/mol")
    print(f"  checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()

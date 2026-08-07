"""Stage 1: pretrain a DimeNet+ VACUUM model from scratch on Frag20 gasEnergy.

EXPERIMENTAL sandbox. Self-contained; mirrors the verified AQM
pipeline/train_stage1_vacuum.py but for the Frag20-Aqsol-100K dataset:
  * target = gas_eV (QM electronic gas-phase energy, eV; shifted by atomic
    references fit on the TRAIN split only, leak-free)
  * energy-only loss (Frag20 ships NO forces) -> lambda_force=0, the AQM
    force term is dropped by construction
  * dataset's OWN fixed split (80K train / 10K valid / 10K test) is used for
    train/valid; the test split is held out for a final honest check
  * molecule-level split (one molecule = one row in Frag20, so no conformer
    grouping is needed)
  * architecture + optimizer defaults mirror the verified pipeline:
    hidden 128, num_blocks 4, lr 1e-3, batch 32, epochs 200, patience 10,
    ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6), grad clip 10

Usage:
  # GPU (Vast):
  python pretrain_stage1_frag20.py --device cuda --output_dir output
  # local smoke test (CPU, tiny):
  python pretrain_stage1_frag20.py --device cpu --max_structures 64 \
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


def energy_loss(pred, true, n_atoms):
    return mse(pred / n_atoms, true / n_atoms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=DEFAULT_FRAG20_H5)
    ap.add_argument("--labels", default=DEFAULT_FRAG20_LABELS)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_structures", type=int, default=None,
                    help="cap on TRAIN rows (smoke tests); valid/test unaffected")
    ap.add_argument("--energy_attr", default="y_gas_eV")
    ap.add_argument("--ckpt_tag", default="stage1_scratch",
                    help="checkpoint stem, saved as {output_dir}/{ckpt_tag}.pt")
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

    ref_energies = fit_atomic_references(train_ds, ELEMENT_TO_IDX, NUM_ELEMENTS,
                                         energy_attr=args.energy_attr).to(device)
    print(f"  refs: {ref_energies.cpu().tolist()}")

    model = build_model(device, num_blocks=4)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,} "
          f"(DimeNet+, vacuum, 4 blocks)")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.ckpt_tag}.pt")
    ref_path = os.path.join(args.output_dir, f"{args.ckpt_tag}_refs.json")

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total = 0.0
        with torch.set_grad_enabled(train):
            for data in loader:
                data = data.to(device)
                if train:
                    optimizer.zero_grad()
                x = build_one_hot(data, device)
                mol_ref = compute_molecular_reference(
                    x, data.batch, ref_energies, data.num_graphs)
                y_shifted = getattr(data, args.energy_attr) - mol_ref
                pred = model(x, data.pos, data.batch)
                n_atoms = torch.bincount(data.batch).float()
                loss = energy_loss(pred.view(-1), y_shifted.view(-1), n_atoms)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                total += loss.item() * data.num_graphs
        return total / len(loader.dataset)

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = args.epochs
    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(valid_loader, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), ckpt_path)
            with open(ref_path, "w") as f:
                json.dump({str(z): ref_energies.cpu()[i].item()
                           for z, i in ELEMENT_TO_IDX.items()}, f, indent=1)
        else:
            stale += 1
        print(f"  epoch {epoch:3d}/{args.epochs} | train {tr_loss:.6f} | "
              f"val {val_loss:.6f} | best {best_val:.6f} (ep {best_epoch}) | "
              f"{dt:5.1f}s", flush=True)
        if stale >= args.patience:
            stop_epoch = epoch
            print(f"  early stopped at epoch {epoch} (patience {args.patience})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    test_loss = run_epoch(test_loader, train=False)
    print(f"\n  DONE in {(time.time() - t0_all) / 60:.1f} min | best val {best_val:.6f} "
          f"(ep {best_epoch}) | early stop {stop_epoch} | held-out test loss {test_loss:.6f}")
    print(f"  checkpoint -> {ckpt_path}\n  refs        -> {ref_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aqm_data import AQMMACEDataset, target_stats_kcal
from config import EV_TO_KCAL
from data import collate_mace
from scratch_model import MACEFreeSolvScratch
from train import WarmupWrapper, train_epoch, validate

# ---------------------------------------------------------------------------
# FROM-SCRATCH variant of train_stage_a.py.
#
# Same training loop, same data path, same verified infra
# (AQMMACEDataset, fit_atomic_references, fit_dataset=train split only,
# WarmupWrapper, ReduceLROnPlateau, save/load reshape handling, meta json).
# The ONLY difference: the model is built with random weights
# (MACEFreeSolvScratch) instead of load_mace_foundation().
#
# Part-5 hyperparameter rationale (from-scratch, vs the pretrained-path
# defaults in train_stage_a.py):
#   lr 1e-2        : MACE's own from-scratch CLI default (0.01); the
#                    pretrained fine-tune lr (1e-4) is wrong for random init.
#   warmup 20      : Adam at lr 1e-2 needs warmup (mace default warmup_steps
#                    is small; 20 epochs on ~2000 batch/epoch is ~40k steps).
#   weight_decay 1e-5 : repo standard; mace from-scratch default is 5e-7,
#                    but 1e-5 was validated on this repo's pretrained runs.
#   epochs 300 / patience 40 : energy-only target; mace's 2048 max epochs is
#                    calibrated for force training at batch 10. 300 epochs
#                    with early stopping is a full run here.
#   batch 32       : repo default (mace from-scratch default is 10).
#   energy-only    : single scalar (dG) per molecule -> forces_weight N/A;
#                    loss is MSE on energy (huber option retained).
# ---------------------------------------------------------------------------


class ScaledTargets(torch.utils.data.Dataset):
    """Wraps a dataset and multiplies its 'y' target by a constant (Run B)."""

    def __init__(self, base, mult):
        self.base = base
        self.mult = mult

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item["y"] = item["y"] * self.mult
        return item


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Stage A (from-scratch): train a RANDOM-INIT MACE-OFF23 'medium' clone "
                    "on AQM solvation (dG = E_sol - E_gas). Architecture matches the "
                    "checkpoint exactly (see mace_off23_medium_arch_config.py); "
                    "weights start random. No pretrained weights anywhere in this path.")
    parser.add_argument("--hdf5_sol", type=str, required=True, help="AQM sol HDF5")
    parser.add_argument("--hdf5_gas", type=str, required=True, help="AQM gas HDF5")
    parser.add_argument("--model_size", type=str, default="medium", choices=["small", "medium", "large"],
                        help="Ignored (scratch arch is always the MACE-OFF23 medium shape); kept for CLI parity")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--r_max", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of MOLECULES held out (conformer-level splits would leak)")
    parser.add_argument("--max_samples", type=int, default=None, help="Subsample N conformers (quick test)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: mace_freesolv/results_stage_a_scratch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_atomic_energies", dest="freeze_atomic_energies", action="store_true", default=True, help="Freeze atomic reference energies (default)")
    parser.add_argument("--no_freeze_atomic_energies", dest="freeze_atomic_energies", action="store_false", help="Train atomic reference energies")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--quick_test", action="store_true", help="2 epochs, 300 samples")
    parser.add_argument("--setup_only", action="store_true",
                        help="Fit atomic refs + calibrate output on random init, then exit before training")
    parser.add_argument("--normalize_targets", action="store_true",
                        help="Run B: pin scale_shift.scale=1.0 and train on y/original_scale "
                             "(isolates gradient magnitude); metrics + saved checkpoint rescaled back")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_stage_a_scratch")

    set_seed(args.seed)
    device = torch.device(args.device if args.device else "cpu")
    print(f"Device: {device}")
    print(f"[scratch] random-init MACE-OFF23-medium clone "
          f"(arch: mace_off23_medium_arch_config.py, weights: random, seed path rng={args.seed})")
    print(f"[scratch] Part-5 defaults: lr={args.lr} warmup={args.warmup_epochs}ep "
          f"wd={args.weight_decay} epochs={args.epochs} patience={args.patience} "
          f"batch={args.batch_size} energy-only MSE loss")

    if args.quick_test:
        args.epochs = 2
        args.max_samples = 300

    split_max = args.max_samples

    full_ds = AQMMACEDataset(
        args.hdf5_sol, args.hdf5_gas,
        r_max=args.r_max, max_neighbors=args.max_neighbors,
        max_samples=args.max_samples,
    )
    if len(full_ds) == 0:
        print("ERROR: no paired conformers found")
        sys.exit(1)

    mol_ids = list(dict.fromkeys(s[0] for s in full_ds.samples))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(mol_ids)
    n_val_mol = max(1, int(len(mol_ids) * args.val_split))
    val_mol_ids = set(mol_ids[:n_val_mol])
    train_mol_ids = set(mol_ids[n_val_mol:])

    train_ds = AQMMACEDataset(
        args.hdf5_sol, args.hdf5_gas,
        r_max=args.r_max, max_neighbors=args.max_neighbors,
        mol_ids=train_mol_ids, max_samples=split_max,
    )
    val_ds = AQMMACEDataset(
        args.hdf5_sol, args.hdf5_gas,
        r_max=args.r_max, max_neighbors=args.max_neighbors,
        mol_ids=val_mol_ids, max_samples=split_max,
    )
    print(f"Split (molecule-level): {len(train_ds)} train / {len(val_ds)} val conformers "
          f"({len(train_mol_ids)} / {len(val_mol_ids)} molecules)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mace, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mace, num_workers=0,
    )

    t_mean_kcal, t_std_kcal = target_stats_kcal(train_ds)
    print(f"Target stats: mean={t_mean_kcal:.2f} std={t_std_kcal:.2f} kcal/mol (dG = E_sol - E_gas)")

    model = MACEFreeSolvScratch(
        model_size=args.model_size,
        device=device,
        freeze_atomic_energies=args.freeze_atomic_energies,
        target_mean=0.0,
        target_std=t_std_kcal,
        fit_dataset=train_ds,
    ).to(device)

    if args.setup_only:
        print("\n[setup_only] refs fitted + calibration done on RANDOM init; exiting before training loop")
        sys.exit(0)

    metric_scale = 1.0
    if args.normalize_targets:
        s = float(model.model.scale_shift.scale.item())
        if s <= 0.0 or s >= 1.0:
            print(f"  [normalize_targets] scale={s:.6f} already >= 1.0; skipping normalization")
            args.normalize_targets = False
        else:
            model.model.scale_shift.scale.requires_grad_(False)
            model.model.scale_shift.shift.requires_grad_(False)
            model.model.scale_shift.scale.data.fill_(1.0)
            model.model.scale_shift.shift.data.fill_(0.0)
            metric_scale = s
            target_mult = 1.0 / s
            print(f"  [normalize_targets] scale {s:.6f} -> 1.0 (scale_shift frozen); "
                  f"train/val targets rescaled by {target_mult:.1f}x; "
                  f"all logged metrics rescaled back by {s:.6f}")

    if args.normalize_targets:
        train_ds_w = ScaledTargets(train_ds, 1.0 / metric_scale)
        val_ds_w = ScaledTargets(val_ds, 1.0 / metric_scale)
        train_loader = DataLoader(
            train_ds_w, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mace, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds_w, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mace, num_workers=0,
        )
        print(f"  [normalize_targets] loaders rebuilt on scaled targets (y x {target_mult:.1f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=args.lr_min,
    )
    warmup = WarmupWrapper(optimizer, args.warmup_epochs, args.lr)

    if args.loss_type == "huber":
        loss_fn = torch.nn.HuberLoss(delta=args.huber_delta)
    else:
        loss_fn = torch.nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "stage_a.pt")
    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    best_epoch = -1
    stale = 0
    epoch_times = []
    loss_scale2 = metric_scale * metric_scale if args.normalize_targets else 1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        warmup.step()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mae, val_rmse, val_r2, _, _ = validate(model, val_loader, device)
        if epoch > args.warmup_epochs:
            scheduler.step(val_mae)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch = float(np.mean(epoch_times[-5:]))
        remaining = args.epochs - epoch
        eta_h = remaining * avg_epoch / 3600.0
        print(f"Epoch {epoch:3d}/{args.epochs} ({100*epoch/args.epochs:4.1f}%) | "
              f"Loss: {train_loss*loss_scale2:.6f} | "
              f"Val MAE: {val_mae*metric_scale*EV_TO_KCAL:.3f} RMSE: {val_rmse*metric_scale*EV_TO_KCAL:.3f} R2: {val_r2:.4f} | "
              f"LR: {warmup.get_lr():.2e} | {elapsed:.1f}s/epoch | ETA ~{eta_h:.1f}h")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_epoch = epoch
            stale = 0
            if args.normalize_targets:
                model.model.scale_shift.scale.data.fill_(metric_scale)
            model.save(checkpoint_path)
            if args.normalize_targets:
                model.model.scale_shift.scale.data.fill_(1.0)
            print(f"  [*] Best checkpoint saved (Val MAE={val_mae*metric_scale*EV_TO_KCAL:.3f} kcal/mol)")
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    meta = {
        "model_size": args.model_size,
        "scratch_init": True,
        "arch_config": "mace_off23_medium_arch_config.py (extracted from loaded MACE-OFF23 medium, 2026-08-01)",
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_mae_kcal": best_val_mae * metric_scale * EV_TO_KCAL,
        "best_val_rmse_kcal": best_val_rmse * metric_scale * EV_TO_KCAL,
        "normalize_targets": args.normalize_targets,
        "original_scale": metric_scale if args.normalize_targets else None,
        "n_train_conformers": len(train_ds),
        "n_val_conformers": len(val_ds),
        "n_train_molecules": len(train_mol_ids),
        "n_val_molecules": len(val_mol_ids),
        "target_mean_kcal": t_mean_kcal,
        "target_std_kcal": t_std_kcal,
        "hdf5_sol": args.hdf5_sol,
        "hdf5_gas": args.hdf5_gas,
        "seed": args.seed,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "note": "FROM-SCRATCH run: random-init MACE-OFF23-medium clone; dG = E_sol - E_gas (ePBE0+MBD), eV via 23.0605 kcal/mol",
    }
    with open(os.path.join(args.output_dir, "stage_a_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  STAGE A (SCRATCH) DONE: checkpoint at {checkpoint_path}")
    print(f"  Val MAE: {meta['best_val_mae_kcal']:.3f} kcal/mol at epoch {best_epoch}")
    print(f"  Use as --init_checkpoint for the FreeSolv fine-tune (Stage B)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

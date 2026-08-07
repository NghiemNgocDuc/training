"""Fine-tune the Frag20-scratch stage-2 checkpoint on the frozen FreeSolv
fold-0 split, then evaluate on the FIXED fold-0 test set (single + 5-conf TTA).

EXPERIMENTAL sandbox. Mirrors the verified freesolv/cv_finetune.py protocol
exactly so results are directly comparable to the verified band
(seed-42 baseline TTA MAE 0.5048 kcal/mol, single-conf 0.5313; every prior
uncertainty experiment plateaued at 0.50-0.55 TTA):
  * init = stage2_scratch.pt  (the Frag20-scratch correction model, 3 blocks)
  * SAME frozen fold-0 split (411/102/129, md5 c0ef293341...)
  * SAME hyperparameters: lr=1e-4, wd=1e-5, batch=8, epochs=200, patience=30,
    MSE in eV, grad clip 10.0, ReduceLROnPlateau(f=0.5, pat=15, min_lr=1e-6)
  * 5-conformer RDKit TTA on the frozen test set (fallback to stored conformer
    if rdkit unavailable)

NOTE: this is the DIMENSION where the experiment is decided - the Frag20-scratch
pretrain only matters insofar as it beats the from-scratch AQM init when both
are fine-tuned identically onto experimental FreeSolv.

Usage:
  # GPU (Vast):
  python finetune_freesolv.py --device cuda \
      --init_ckpt output/stage2_scratch.pt --output_dir output
  # local smoke test (CPU, tiny):
  python finetune_freesolv.py --device cpu --quick_test --n_conformers 1
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from common_scratch import (EV_TO_KCAL, DEFAULT_SPLIT_DIR,
                            DEFAULT_FREESOLV_CONFORMERS, DEFAULT_FREESOLV_LABELS,
                            DEFAULT_SEED, build_model, evaluate, load_frozen_split,
                            load_freesolv_labels, conformer_average, set_seed,
                            md5_bytes)
from element_vocab import NUM_ELEMENTS, build_one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output_finetune")
DEFAULT_INIT_CKPT = os.path.join(HERE, "output", "stage2_scratch.pt")

# Verified baselines (kcal/mol, fold-0 test set) - reported for context.
BASELINE_SEED42_TTA = {"mae": 0.5048, "rmse": 0.7568}
BASELINE_SEED42_SINGLE = {"mae": 0.5313, "rmse": 0.7746}


def train(split_dir, freesolv_h5, freesolv_labels_json, init_ckpt, output_dir,
          epochs, patience, lr, batch_size, n_conformers, device_name, seed,
          quick_test):
    import h5py
    import torch
    from torch_geometric.loader import DataLoader

    freesolv_labels = load_freesolv_labels(freesolv_labels_json)
    train_ids, val_ids, test_ids = load_frozen_split(split_dir, freesolv_labels)
    split_blob = b"".join(
        open(os.path.join(split_dir, name), "rb").read()
        for name in ("train_ids.json", "val_ids.json", "test_ids.json"))
    split_md5 = md5_bytes(split_blob)

    device = torch.device(device_name)
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "finetuned_freesolv.pt")

    print("\n" + "=" * 66)
    print("  EXPERIMENT: Frag20-scratch pretrain -> FreeSolv fold-0 fine-tune")
    print("=" * 66)
    print(f"  split: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test (md5 {split_md5[:10]}...) frozen from {split_dir}")
    print(f"  init: {init_ckpt}")
    print(f"  seed {seed} | lr={lr} wd=1e-5 batch={batch_size} epochs={epochs} "
          f"| MSE in eV, grad-clip 10.0, ReduceLROnPlateau(f=0.5, pat=15, min_lr=1e-6) "
          f"| {n_conformers}-conf TTA")

    set_seed(seed)
    model = build_model(device, num_blocks=3)
    ckpt = torch.load(init_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"  loaded init checkpoint ({sum(p.numel() for p in model.parameters()):,} params)")

    # SimpleDataset over FreeSolv conformers (mirrors cv_finetune.py exactly).
    class SimpleDataset:
        def __init__(self, ids):
            self.ids = ids
            self._cache = {}
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            from torch_geometric.data import Data
            mid = self.ids[idx]
            if mid not in self._cache:
                with h5py.File(freesolv_h5, "r") as f:
                    g = f[mid]
                    d = Data(
                        z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                        pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                    )
                self._cache[mid] = d.clone()
            data = self._cache[mid].clone()
            data.mol_id = mid
            data.y_dG = torch.tensor([freesolv_labels[mid]["expt"]], dtype=torch.float)
            return data

    train_ds = SimpleDataset(train_ids)
    val_ds = SimpleDataset(val_ids)
    test_ds = SimpleDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    mse_loss = torch.nn.MSELoss()

    def evaluate_loader(loader):
        model.eval()
        all_p, all_e = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                ex = data.y_dG.view(-1).to(device)
                valid = ~torch.isnan(ex)
                all_p.append(pred[valid].cpu())
                all_e.append(ex[valid].cpu())
        preds = torch.cat(all_p).numpy()
        expts = torch.cat(all_e).numpy()
        mae = float(np.mean(np.abs(preds - expts)))
        rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
        return mae, rmse, preds, expts

    epochs_run = 2 if quick_test else epochs
    patience_run = 5 if quick_test else 30
    best_val_mae = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = epochs_run
    t0_all = time.time()
    for epoch in range(1, epochs_run + 1):
        t0 = time.time()
        model.train()
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            ex = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            valid = ~torch.isnan(ex)
            if valid.sum() == 0:
                continue
            loss = mse_loss(pred[valid], ex[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        val_mae, val_rmse, _, _ = evaluate_loader(val_loader)
        scheduler.step(val_mae)
        dt = time.time() - t0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            stale += 1
        print(f"  epoch {epoch:3d}/{epochs_run} | best val {best_val_mae:7.3f} "
              f"(ep {best_epoch}) | cur val {val_mae:7.3f} | {dt:5.1f}s", flush=True)
        if stale >= patience_run:
            stop_epoch = epoch
            print(f"  early stopped at epoch {epoch} (patience {patience_run})")
            break
    total_min = (time.time() - t0_all) / 60.0

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device,
                                     weights_only=True))
    model.eval()
    test_mae, test_rmse, test_preds, test_expts = evaluate_loader(test_loader)

    tta_mae = tta_rmse = None
    tta_preds = {}
    if n_conformers > 1:
        tta_mae, tta_rmse, tta_preds = conformer_average(
            model, device, test_ids, freesolv_labels, freesolv_h5,
            n_conformers, batch_size)

    metrics = {
        "kind": "EXPERIMENTAL: Frag20-scratch pretrain -> FreeSolv fold-0 fine-tune",
        "seed": seed,
        "init_ckpt": init_ckpt,
        "split_md5": split_md5,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "hyperparams": {"lr": lr, "weight_decay": 1e-5, "batch_size": batch_size,
                        "epochs": epochs, "patience": patience_run, "device": device_name},
        "best_val_mae_kcal": best_val_mae,
        "best_epoch": best_epoch,
        "early_stop_epoch": stop_epoch,
        "total_min": round(total_min, 1),
        "test_mae_single_conf_kcal": test_mae,
        "test_rmse_single_conf_kcal": test_rmse,
        "test_mae_tta_kcal": tta_mae,
        "test_rmse_tta_kcal": tta_rmse,
        "baselines": {
            "seed42_tta": BASELINE_SEED42_TTA,
            "seed42_single_conf": BASELINE_SEED42_SINGLE,
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # per-molecule predictions (test only)
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal\n")
        ttl = tta_preds if tta_preds else dict(zip(test_ids, test_preds))
        for mid in test_ids:
            f.write(f"{mid},{ttl[mid]:.6f},{freesolv_labels[mid]['expt']:.6f}\n")

    print("\n" + "=" * 66)
    print("  RESULTS (frozen fold-0 test set, kcal/mol)")
    print("=" * 66)
    print(f"  test MAE (single-conf): {test_mae:.3f} | RMSE {test_rmse:.3f}")
    print(f"  baseline seed42 single: MAE {BASELINE_SEED42_SINGLE['mae']:.3f} | "
          f"RMSE {BASELINE_SEED42_SINGLE['rmse']:.3f}")
    if tta_mae is not None:
        print(f"  test MAE (TTA-{n_conformers})      : {tta_mae:.3f} | RMSE {tta_rmse:.3f} "
              f"| baseline seed42 TTA MAE {BASELINE_SEED42_TTA['mae']:.3f}")
    print(f"  -> {output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--freesolv_h5", default=DEFAULT_FREESOLV_CONFORMERS)
    ap.add_argument("--freesolv_labels", default=DEFAULT_FREESOLV_LABELS)
    ap.add_argument("--init_ckpt", default=DEFAULT_INIT_CKPT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quick_test", action="store_true")
    args = ap.parse_args()

    if args.quick_test:
        args.epochs = max(args.epochs, 2)
        print("  [quick_test] running 2 epochs, patience 5")
    train(split_dir=args.split_dir, freesolv_h5=args.freesolv_h5,
          freesolv_labels_json=args.freesolv_labels, init_ckpt=args.init_ckpt,
          output_dir=args.output_dir, epochs=args.epochs,
          patience=(5 if args.quick_test else 30), lr=args.lr,
          batch_size=args.batch_size, n_conformers=args.n_conformers,
          device_name=args.device, seed=args.seed, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
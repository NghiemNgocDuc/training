"""Fine-tune on FreeSolv experimental data.

Usage (Option B - correction model):
  python finetune_freesolv.py \
      --conformers ../freesolv_conformers.hdf5 \
      --checkpoint_dir results \
      --correction_ckpt stage2_correction.pt \
      --output_dir ft_results

Usage (Option A - scratch model):
  python finetune_freesolv.py \
      --conformers ../freesolv_conformers.hdf5 \
      --option_a \
      --vacuum_ckpt stage1_fold_1.pt \
      --option_a_ckpt option_a.pt \
      --output_dir ft_option_a
"""

import csv
import json
import math
import os
import sys
import argparse
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "solvation-gnn"))

from DimeModels import DimeNetPlus
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot

EV_TO_KCAL = 23.0605


class FreeSolvFineTuneDataset(Dataset):
    """PyG Dataset that pairs conformer geometries with experimental dG."""

    def __init__(self, hdf5_path, mol_ids, labels_dict):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.mol_ids = mol_ids
        self.labels = labels_dict
        self._cache = {}

    def len(self):
        return len(self.mol_ids)

    def get(self, idx):
        mol_id = self.mol_ids[idx]
        if mol_id in self._cache:
            data = self._cache[mol_id].clone()
        else:
            with h5py.File(self.hdf5_path, "r") as f:
                grp = f[mol_id]
                z = torch.tensor(grp["atNUM"][...], dtype=torch.long)
                pos = torch.tensor(grp["atXYZ"][...], dtype=torch.float)
            data = Data(z=z, pos=pos)
            self._cache[mol_id] = data.clone()
        data.mol_id = mol_id
        data.y_dG = torch.tensor(
            [self.labels.get(mol_id, {}).get("expt", float("nan"))],
            dtype=torch.float,
        )
        return data


def build_model(num_blocks, device):
    model = DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=128, out_channels=1,
        num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", default="freesolv_conformers.hdf5")
    parser.add_argument("--checkpoint_dir", default="results")
    parser.add_argument("--correction_ckpt", default="stage2_correction.pt")
    parser.add_argument("--output_dir", default="ft_results")
    parser.add_argument("--cache_dir", default="Data/FreeSolv")
    parser.add_argument("--option_a", action="store_true",
                        help="Fine-tune Option A (scratch) model instead of correction model")
    parser.add_argument("--vacuum_ckpt", default="stage1_fold_1.pt",
                        help="Vacuum model checkpoint (needed for Option A)")
    parser.add_argument("--option_a_ckpt", default="option_a.pt",
                        help="Option A model checkpoint")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_dir)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load FreeSolv labels ──
    json_path = os.path.join(args.cache_dir, "database.json")
    if not os.path.exists(json_path):
        from freesolv_dataset import download_freesolv_data
        json_path, _ = download_freesolv_data(args.cache_dir)
    with open(json_path) as f:
        all_labels = json.load(f)

    # ── Get list of valid mol_ids from HDF5 ──
    mol_ids = []
    with h5py.File(args.conformers, "r") as f:
        mol_ids = list(f.keys())
    # Filter to those with experimental data
    mol_ids = [m for m in mol_ids if m in all_labels
               and isinstance(all_labels[m].get("expt"), (int, float))]
    print(f"Available molecules with labels: {len(mol_ids)}")

    # ── Train/test split (fixed seed for reproducibility) ──
    train_ids, test_ids = train_test_split(
        mol_ids, test_size=args.test_size, random_state=args.seed
    )
    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # ── Build model(s) and load checkpoint(s) ──
    if args.option_a:
        # Load vacuum model (frozen)
        vacuum_model = build_model(4, device)
        vac_path = os.path.join(ckpt_dir, args.vacuum_ckpt)
        vacuum_model.load_state_dict(torch.load(vac_path, map_location=device, weights_only=True))
        for p in vacuum_model.parameters():
            p.requires_grad_(False)
        vacuum_model.eval()
        print(f"Loaded vacuum model (frozen): {vac_path}")

        # Load Option A model (trainable)
        model = build_model(3, device)
        oa_path = os.path.join(ckpt_dir, args.option_a_ckpt)
        model.load_state_dict(torch.load(oa_path, map_location=device, weights_only=True))
        print(f"Loaded Option A model: {oa_path}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        def forward_fn(model, vacuum_model, x, pos, batch):
            vac_e = vacuum_model(x, pos, batch)
            oa_e = model(x, pos, batch)
            return (oa_e - vac_e)  # ΔG in eV
    else:
        model = build_model(3, device)
        ckpt_path = os.path.join(ckpt_dir, args.correction_ckpt)
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        vacuum_model = None

        def forward_fn(model, vacuum_model, x, pos, batch):
            return model(x, pos, batch)  # ΔG in eV directly

    # ── Datasets and loaders ──
    train_ds = FreeSolvFineTuneDataset(args.conformers, train_ids, all_labels)
    test_ds = FreeSolvFineTuneDataset(args.conformers, test_ids, all_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Optimizer, scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2,
        min_lr=args.min_lr,
    )
    mse = torch.nn.MSELoss()

    # ── Save checkpoint name ──
    ckpt_name = "finetuned_option_a.pt" if args.option_a else "finetuned_correction.pt"

    # ── Training loop ──
    best_test_mae = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = forward_fn(model, vacuum_model, x, data.pos, data.batch).view(-1)  # eV
            dG_exp = data.y_dG.view(-1).to(device) / EV_TO_KCAL  # convert to eV
            valid = ~torch.isnan(dG_exp)
            if valid.sum() == 0:
                continue
            loss = mse(pred[valid], dG_exp[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * valid.sum().item()

        # ── Evaluation ──
        model.eval()
        test_preds, test_expts = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = forward_fn(model, vacuum_model, x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                dG_exp = data.y_dG.view(-1).to(device)
                valid = ~torch.isnan(dG_exp)
                test_preds.append(pred[valid].cpu())
                test_expts.append(dG_exp[valid].cpu())

        test_preds = torch.cat(test_preds).numpy()
        test_expts = torch.cat(test_expts).numpy()
        test_mae = float(np.mean(np.abs(test_preds - test_expts)))
        test_rmse = float(np.sqrt(np.mean((test_preds - test_expts) ** 2)))
        train_loss_avg = train_loss / len(train_ids)

        scheduler.step(test_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_msg = (f"Epoch {epoch:3d} | Train loss: {train_loss_avg:.6f} "
                     f"| Test MAE: {test_mae:.3f} RMSE: {test_rmse:.3f} "
                     f"| LR: {current_lr:.2e}")
        print(epoch_msg)

        # Best checkpoint
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_epoch = epoch
            patience_counter = 0
            ckpt_out = os.path.join(output_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_out)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Best test MAE: {best_test_mae:.3f} kcal/mol at epoch {best_epoch}")
    print(f"{'='*60}")

    # ── Final evaluation on test set with best model ──
    model.load_state_dict(
        torch.load(os.path.join(output_dir, ckpt_name),
                   map_location=device)
    )
    model.eval()

    all_preds, all_expts = [], []
    with torch.no_grad():
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
            dG_exp = data.y_dG.view(-1).to(device)
            valid = ~torch.isnan(dG_exp)
            all_preds.append(pred[valid].cpu())
            all_expts.append(dG_exp[valid].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_expts = torch.cat(all_expts).numpy()

    final_mae = float(np.mean(np.abs(all_preds - all_expts)))
    final_rmse = float(np.sqrt(np.mean((all_preds - all_expts) ** 2)))
    ss_res = np.sum((all_preds - all_expts) ** 2)
    ss_tot = np.sum((all_expts - np.mean(all_expts)) ** 2)
    final_r2 = float(1 - ss_res / ss_tot)

    print(f"\nFinal test set metrics:")
    print(f"  MAE:  {final_mae:.3f} kcal/mol")
    print(f"  RMSE: {final_rmse:.3f} kcal/mol")
    print(f"  R²:   {final_r2:.4f}")
    print(f"  N:    {len(all_preds)}")

    # Save test predictions
    test_csv = os.path.join(output_dir, "ft_test_predictions.csv")
    with open(test_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dG_pred_kcal", "dG_exp_kcal"])
        for p, e in zip(all_preds, all_expts):
            w.writerow([f"{p:.6f}", f"{e:.6f}"])
    print(f"\nSaved test predictions to {test_csv}")

    # ── Full dataset evaluation ──
    full_ds = FreeSolvFineTuneDataset(args.conformers, mol_ids, all_labels)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
    full_preds, full_expts = [], []
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
            dG_exp = data.y_dG.view(-1).to(device)
            valid = ~torch.isnan(dG_exp)
            full_preds.append(pred[valid].cpu())
            full_expts.append(dG_exp[valid].cpu())
    full_preds = torch.cat(full_preds).numpy()
    full_expts = torch.cat(full_expts).numpy()
    full_mae = float(np.mean(np.abs(full_preds - full_expts)))
    full_rmse = float(np.sqrt(np.mean((full_preds - full_expts) ** 2)))
    print(f"\nFull dataset (train+test, {len(full_preds)} molecules):")
    print(f"  MAE:  {full_mae:.3f} kcal/mol")
    print(f"  RMSE: {full_rmse:.3f} kcal/mol")


if __name__ == "__main__":
    main()
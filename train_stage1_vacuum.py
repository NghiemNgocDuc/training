import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

import torch
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
import argparse
import time
import numpy as np

from aqm_dataset import AQMDataset
from aqm_config import VACUUM_ENERGY_TARGET, VACUUM_FORCES_TARGET

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description="Stage 1: Train Vacuum DimeNetPlus on AQM-gas (5-fold CV)")
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-gas.hdf5",
                    help="Path to AQM-gas.hdf5")
parser.add_argument("--batchsize", "-b", type=int, default=16)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=5.0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks", type=int, default=4)
parser.add_argument("--int_emb_size", type=int, default=64)
parser.add_argument("--basis_emb_size", type=int, default=8)
parser.add_argument("--out_emb_channels", type=int, default=256)
parser.add_argument("--num_spherical", type=int, default=7)
parser.add_argument("--num_radial", type=int, default=6)
parser.add_argument("--envelope_exponent", type=int, default=5)
parser.add_argument("--num_before_skip", type=int, default=1)
parser.add_argument("--num_after_skip", type=int, default=2)
parser.add_argument("--num_output_layers", type=int, default=3)
parser.add_argument("--max_neighbors", type=int, default=32)
parser.add_argument("--lambda_force", type=float, default=1000.0)
parser.add_argument("--k_folds", type=int, default=5)
parser.add_argument("--max_structures", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

if args.device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)
print(f"Using device: {device}")

dataset = AQMDataset(
    root="../Data/AQM-gas",
    hdf5_path=args.hdf5,
    energy_key=VACUUM_ENERGY_TARGET,
    forces_key=VACUUM_FORCES_TARGET,
    max_structures=args.max_structures,
)
print(f"Dataset: {len(dataset)} samples")

os.makedirs(args.output_dir, exist_ok=True)

kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n{'='*60}\nFold {fold + 1}/{args.k_folds}\n{'='*60}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False)

    model = DimeNetPlus(
        in_channels=1,
        hidden_channels=args.hidden,
        out_channels=1,
        num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size,
        basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical,
        num_radial=args.num_radial,
        cutoff=args.radius,
        max_num_neighbors=args.max_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip,
        num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers,
        is_energy=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    def combined_loss(energy_pred, energy_true, forces_pred, forces_true, lambda_force):
        loss_e = mse(energy_pred, energy_true)
        loss_f = mse(forces_pred, forces_true)
        return loss_e + lambda_force * loss_f

    best_val_loss = float("inf")
    patience = 20
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        t0 = time.time()
        for data in train_loader:
            data = data.to(device)
            data.pos.requires_grad_()
            optimizer.zero_grad()

            x = data.z.float().view(-1, 1)
            energy_pred = model(x, data.pos, data.batch)
            forces_pred = -torch.autograd.grad(
                outputs=energy_pred,
                inputs=data.pos,
                grad_outputs=torch.ones_like(energy_pred),
                create_graph=True,
            )[0]

            loss = combined_loss(
                energy_pred.view(-1), data.y_energy,
                forces_pred, data.y_forces,
                args.lambda_force,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.enable_grad():
            for data in val_loader:
                data = data.to(device)
                data.pos.requires_grad_()
                x = data.z.float().view(-1, 1)
                energy_pred = model(x, data.pos, data.batch)
                forces_pred = -torch.autograd.grad(
                    outputs=energy_pred,
                    inputs=data.pos,
                    grad_outputs=torch.ones_like(energy_pred),
                    create_graph=False,
                )[0]
                loss = combined_loss(
                    energy_pred.view(-1), data.y_energy,
                    forces_pred, data.y_forces,
                    args.lambda_force,
                )
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_loader.dataset)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.output_dir, f"stage1_fold_{fold + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> Saved best model to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping after {epoch} epochs")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    fold_results.append(best_val_loss)
    print(f"Fold {fold + 1} best val loss: {best_val_loss:.6f}")

print(f"\n{'='*60}")
print(f"5-Fold CV complete. Best val losses: {[f'{l:.6f}' for l in fold_results]}")
print(f"Mean val loss: {np.mean(fold_results):.6f} +/- {np.std(fold_results):.6f}")

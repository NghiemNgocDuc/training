import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

import json
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
import argparse
import time
import numpy as np

from aqm_dataset import AQMDataset
from aqm_config import SOLVATED_ENERGY_TARGET, SOLVATED_FORCES_TARGET

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description="Option A: Train single DimeNetPlus from scratch on AQM-sol to predict total solvated energy"
)
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-sol.hdf5",
                    help="Path to AQM-sol.hdf5")
parser.add_argument("--gas_hdf5", type=str, default=None,
                    help="Path to AQM-gas.hdf5 for eSOLV computation (optional)")
parser.add_argument("--batchsize", "-b", type=int, default=16)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=5.0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks", type=int, default=3)
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
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--max_structures", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)

# Option B paths for comparison
parser.add_argument("--option_b_checkpoint", type=str, default=None,
                    help="Path to Option B correction checkpoint for comparison")
parser.add_argument("--option_b_vacuum_ckpt", type=str, default=None,
                    help="Path to Option B vacuum checkpoint for comparison")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

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

def build_model():
    return DimeNetPlus(
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

# ---- Dataset ----
dataset = AQMDataset(
    root="../Data/AQM-sol-optionA",
    hdf5_path=args.hdf5,
    gas_hdf5_path=args.gas_hdf5,
    energy_key=SOLVATED_ENERGY_TARGET,
    forces_key=SOLVATED_FORCES_TARGET,
    max_structures=args.max_structures,
)
print(f"Dataset: {len(dataset)} samples")

# Only keep samples that have gas_energy for eSOLV computation
if args.gas_hdf5 is not None:
    valid_indices = [i for i in range(len(dataset)) if hasattr(dataset[i], 'gas_energy')]
    dataset = dataset[valid_indices]
    print(f"  After filtering for gas pairing: {len(dataset)} samples")

n_total = len(dataset)
n_val = max(1, int(n_total * args.val_split))
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(seed),
)
print(f"  Train: {n_train}  Val: {n_val}")

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

# ---- Model ----
model = build_model()
model.train()
print(f"Option A model: {sum(p.numel() for p in model.parameters()):,} params")

optimizer = optim.Adam(model.parameters(), lr=args.lr)
mse = torch.nn.MSELoss()

def combined_loss(energy_pred, energy_true, forces_pred, forces_true, lambda_force):
    return mse(energy_pred, energy_true) + lambda_force * mse(forces_pred, forces_true)

def predict_energy_and_forces(m, data_batch):
    x = data_batch.z.float().view(-1, 1)
    pos = data_batch.pos
    e = m(x, pos, data_batch.batch)
    f = -torch.autograd.grad(e, pos, grad_outputs=torch.ones_like(e),
                             create_graph=True)[0]
    return e, f

@torch.no_grad()
def predict_energy_and_forces_eval(m, data_batch):
    x = data_batch.z.float().view(-1, 1)
    data_batch.pos.requires_grad_(True)
    e = m(x, data_batch.pos, data_batch.batch)
    f = -torch.autograd.grad(e, data_batch.pos, grad_outputs=torch.ones_like(e),
                             create_graph=False)[0]
    return e, f

def compute_esolv_mae(m, loader):
    m.eval()
    total_mae = 0.0
    count = 0
    for data in loader:
        data = data.to(device)
        e_pred, _ = predict_energy_and_forces_eval(m, data)
        for i in range(data.num_graphs):
            mask = data.batch == i
            if hasattr(data, 'gas_energy'):
                gas_e = data.gas_energy[i].item() if data.gas_energy.dim() > 0 else data.gas_energy.item()
                esolv_pred = e_pred[i].item() - gas_e
                esolv_true = data.y_esolv[i].item() if hasattr(data, 'y_esolv') and data.y_esolv is not None else 0.0
                total_mae += abs(esolv_pred - esolv_true)
                count += 1
    return total_mae / count if count > 0 else 0.0

# ---- Training loop ----
best_val_loss = float("inf")
best_esolv_mae = float("inf")
patience = 20
epochs_no_improve = 0

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()
        e_pred, f_pred = predict_energy_and_forces(model, data)
        loss = combined_loss(e_pred.view(-1), data.y_energy,
                             f_pred, data.y_forces, args.lambda_force)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.enable_grad():
        for data in val_loader:
            data = data.to(device)
            data.pos.requires_grad_()
            e_pred, f_pred = predict_energy_and_forces_eval(model, data)
            loss = combined_loss(e_pred.view(-1), data.y_energy,
                                 f_pred, data.y_forces, args.lambda_force)
            val_loss += loss.item() * data.num_graphs
    val_loss /= len(val_loader.dataset)

    val_esolv_mae = compute_esolv_mae(model, val_loader)

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:3d}/{args.epochs} | "
        f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
        f"eSOLV MAE: {val_esolv_mae:.6f} | {elapsed:.2f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_esolv_mae = val_esolv_mae
        epochs_no_improve = 0
        ckpt_path = os.path.join(args.output_dir, "option_a.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping after {epoch} epochs")
            break

# ---- Evaluate Option B on same validation set for comparison ----
option_b_esolv_mae = None
if args.option_b_checkpoint and args.option_b_vacuum_ckpt:
    print("\n--- Evaluating Option B on same validation set ---")
    from DimeModels import DimeNetPlus

    # Build vacuum model (+1 block) and correction model
    vacuum_model = DimeNetPlus(
        in_channels=1, hidden_channels=args.hidden, out_channels=1,
        num_blocks=args.num_blocks + 1,
        int_emb_size=args.int_emb_size, basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical, num_radial=args.num_radial,
        cutoff=args.radius, max_num_neighbors=args.max_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip, num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers, is_energy=True,
    ).to(device)
    vacuum_model.load_state_dict(
        torch.load(args.option_b_vacuum_ckpt, map_location=device, weights_only=True))
    for p in vacuum_model.parameters():
        p.requires_grad_(False)
    vacuum_model.eval()

    correction_model = DimeNetPlus(
        in_channels=1, hidden_channels=args.hidden, out_channels=1,
        num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size, basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical, num_radial=args.num_radial,
        cutoff=args.radius, max_num_neighbors=args.max_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip, num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers, is_energy=True,
    ).to(device)
    correction_model.load_state_dict(
        torch.load(args.option_b_checkpoint, map_location=device, weights_only=True))
    correction_model.eval()

    # Option B: total = vacuum(solvated) + correction(solvated), correction approximates eSOLV
    total_esolv_mae = 0.0
    count = 0
    for data in val_loader:
        data = data.to(device)
        x = data.z.float().view(-1, 1)
        with torch.no_grad():
            vacuum_e = vacuum_model(x, data.pos, data.batch)
            correction_e = correction_model(x, data.pos, data.batch)
        for i in range(data.num_graphs):
            mask = data.batch == i
            if hasattr(data, 'y_esolv') and data.y_esolv is not None:
                esolv_true = data.y_esolv[i].item()
                esolv_pred = correction_e[i].item()
                total_esolv_mae += abs(esolv_pred - esolv_true)
                count += 1
    option_b_esolv_mae = total_esolv_mae / count if count > 0 else None
    if option_b_esolv_mae is not None:
        print(f"  Option B eSOLV MAE: {option_b_esolv_mae:.6f} eV")
    else:
        print("  Could not compute Option B eSOLV MAE (no y_esolv in data)")

# ---- Save results ----
results = {
    "best_val_loss": best_val_loss,
    "best_esolv_mae": best_esolv_mae,
    "option_a_esolv_mae": best_esolv_mae,
    "option_b_esolv_mae": option_b_esolv_mae,
    "config": {
        "model": "DimeNetPlus (single, from scratch)",
        "dataset": args.hdf5,
        "gas_dataset": args.gas_hdf5,
        "num_blocks": args.num_blocks,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "lambda_force": args.lambda_force,
    },
}
out_path = os.path.join(args.output_dir, "option_a_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

# ---- Comparison table ----
print(f"\n{'='*60}")
print("  COMPARISON: Option A vs Option B")
print(f"{'='*60}")
print(f"  {'Metric':<40} {'Option A':>12} {'Option B':>12}")
print(f"  {'-'*40} {'-'*12} {'-'*12}")
print(f"  {'Best val loss (total energy)':<40} {best_val_loss:>12.6f} {'N/A':>12}")
if best_esolv_mae is not None:
    b_str = f"{option_b_esolv_mae:.6f}" if option_b_esolv_mae is not None else "N/A"
    print(f"  {'eSOLV MAE (eV)':<40} {best_esolv_mae:>12.6f} {b_str:>12}")
print(f"  {'Trainable params':<40} {sum(p.numel() for p in model.parameters()):>12,}")
if option_b_esolv_mae is not None and best_esolv_mae > 0:
    ratio = best_esolv_mae / option_b_esolv_mae
    print(f"  {'Option A / Option B ratio':<40} {ratio:>12.3f}")
    if ratio < 1:
        print(f"  >>> Option A (direct) beats Option B (delta-learning)!")
    else:
        print(f"  >>> Option B (delta-learning) beats Option A (direct)!")
print(f"{'='*60}")

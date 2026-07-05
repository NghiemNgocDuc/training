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

from spice2_dataset import SPICE2Dataset
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description="Stage 2b: Explicit-water refinement using SPICE2 solute forces"
)
parser.add_argument("--hdf5", type=str, default="../aqm_data/SPICE-2.0.1.hdf5",
                    help="Path to SPICE-2.0.1.hdf5")
parser.add_argument("--dataset_root", type=str, default="../Data/SPICE2",
                    help="Dataset cache root")
parser.add_argument("--vacuum_ckpt", type=str, required=True,
                    help="Path to Stage 1 vacuum checkpoint")
parser.add_argument("--implicit_ckpt", type=str, required=True,
                    help="Path to Stage 2 implicit correction checkpoint")
parser.add_argument("--batchsize", "-b", type=int, default=8)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=5.0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks_vacuum", type=int, default=4)
parser.add_argument("--num_blocks_implicit", type=int, default=3)
parser.add_argument("--num_blocks_explicit", type=int, default=2,
                    help="Fewer blocks for explicit correction (smallest)")
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
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--max_molecules", type=int, default=None)
parser.add_argument("--max_conformers", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)
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

# ---- Helper to build models ----
def build_model(num_blocks):
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=args.hidden, out_channels=1,
        num_blocks=num_blocks,
        int_emb_size=args.int_emb_size, basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical, num_radial=args.num_radial,
        cutoff=args.radius, max_num_neighbors=args.max_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip, num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers, is_energy=True,
    ).to(device)

# ---- Load frozen models ----
print("Loading frozen models...")

vacuum_model = build_model(args.num_blocks_vacuum)
vacuum_model.load_state_dict(
    torch.load(args.vacuum_ckpt, map_location=device, weights_only=True))
for p in vacuum_model.parameters():
    p.requires_grad_(False)
vacuum_model.eval()
print(f"  Vacuum model ({args.num_blocks_vacuum} blocks): "
      f"{sum(p.numel() for p in vacuum_model.parameters()):,} params (frozen)")

implicit_model = build_model(args.num_blocks_implicit)
implicit_model.load_state_dict(
    torch.load(args.implicit_ckpt, map_location=device, weights_only=True))
for p in implicit_model.parameters():
    p.requires_grad_(False)
implicit_model.eval()
print(f"  Implicit correction ({args.num_blocks_implicit} blocks): "
      f"{sum(p.numel() for p in implicit_model.parameters()):,} params (frozen)")

# ---- Explicit correction model (trainable) ----
explicit_model = build_model(args.num_blocks_explicit)
explicit_model.train()
print(f"  Explicit correction ({args.num_blocks_explicit} blocks): "
      f"{sum(p.numel() for p in explicit_model.parameters()):,} params (trainable)")

# ---- Dataset ----
dataset = SPICE2Dataset(
    root=args.dataset_root,
    hdf5_path=args.hdf5,
    max_molecules=args.max_molecules,
    max_conformers_per_mol=args.max_conformers,
)
print(f"Dataset: {len(dataset)} samples")

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

# ---- Optimizer (explicit model only) ----
optimizer = optim.Adam(explicit_model.parameters(), lr=args.lr)
mse = torch.nn.MSELoss()

# Sanity snapshots of frozen params
vacuum_init_sum = sum(p.sum().item() for p in vacuum_model.parameters())
implicit_init_sum = sum(p.sum().item() for p in implicit_model.parameters())

# ---- Training loop ----
best_val_loss = float("inf")
patience = 20
epochs_no_improve = 0

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    # --- Train ---
    explicit_model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()

        x = build_one_hot(data, device)
        vacuum_e = vacuum_model(x, data.pos, data.batch)
        implicit_e = implicit_model(x, data.pos, data.batch)
        explicit_e = explicit_model(x, data.pos, data.batch)
        total_e = vacuum_e + implicit_e + explicit_e

        forces_pred = -torch.autograd.grad(
            total_e, data.pos,
            grad_outputs=torch.ones_like(total_e),
            create_graph=True,
        )[0]

        loss = mse(forces_pred, data.y_forces)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(explicit_model.parameters(), 10.0)
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    train_loss /= len(train_loader.dataset)

    # --- Validate ---
    explicit_model.eval()
    val_loss = 0.0
    with torch.enable_grad():
        for data in val_loader:
            data = data.to(device)
            data.pos.requires_grad_()

            x = build_one_hot(data, device)
            vacuum_e = vacuum_model(x, data.pos, data.batch)
            implicit_e = implicit_model(x, data.pos, data.batch)
            explicit_e = explicit_model(x, data.pos, data.batch)
            total_e = vacuum_e + implicit_e + explicit_e

            forces_pred = -torch.autograd.grad(
                total_e, data.pos,
                grad_outputs=torch.ones_like(total_e),
                create_graph=False,
            )[0]

            loss = mse(forces_pred, data.y_forces)
            val_loss += loss.item() * data.num_graphs
    val_loss /= len(val_loader.dataset)

    elapsed = time.time() - t0
    print(f"Epoch {epoch:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.2f}s")

    # Sanity check every 5 epochs
    if epoch % 5 == 0:
        v_sum = sum(p.sum().item() for p in vacuum_model.parameters())
        i_sum = sum(p.sum().item() for p in implicit_model.parameters())
        v_delta = abs(v_sum - vacuum_init_sum)
        i_delta = abs(i_sum - implicit_init_sum)
        print(f"  [Sanity] Vacuum delta: {v_delta:.6e}  | Implicit delta: {i_delta:.6e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        ckpt_path = os.path.join(args.output_dir, "stage2b.pt")
        torch.save(explicit_model.state_dict(), ckpt_path)
        print(f"  -> Saved best explicit model to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping after {epoch} epochs")
            break

# ---- Save results ----
results = {
    "best_val_force_mse": best_val_loss,
    "best_val_force_mae": float(np.sqrt(best_val_loss)),
    "config": {
        "vacuum_ckpt": args.vacuum_ckpt,
        "implicit_ckpt": args.implicit_ckpt,
        "num_blocks_vacuum": args.num_blocks_vacuum,
        "num_blocks_implicit": args.num_blocks_implicit,
        "num_blocks_explicit": args.num_blocks_explicit,
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "lr": args.lr,
        "loss": "force_mse_only",
    },
    "params": {
        "vacuum": sum(p.numel() for p in vacuum_model.parameters()),
        "implicit": sum(p.numel() for p in implicit_model.parameters()),
        "explicit": sum(p.numel() for p in explicit_model.parameters()),
    },
}
out_path = os.path.join(args.output_dir, "stage2b_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNet
import argparse
import time

from aqm_dataset import AQMDataset
from aqm_config import SOLVATED_ENERGY_TARGET, SOLVATED_FORCES_TARGET

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description="Stage 2: Train Correction DimeNet on AQM-sol with frozen vacuum model"
)
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-sol.hdf5",
                    help="Path to AQM-sol.hdf5")
parser.add_argument("--gas_hdf5", type=str, default="../aqm_data/AQM-gas.hdf5",
                    help="Path to AQM-gas.hdf5 for pairing (optional)")
parser.add_argument("--vacuum_ckpt", type=str, default="vacuum_model.pth",
                    help="Path to trained vacuum model checkpoint")
parser.add_argument("--batchsize", "-b", type=int, default=16)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=5.0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks", type=int, default=3,
                    help="Fewer blocks for correction model (smaller than vacuum)")
parser.add_argument("--num_spherical", type=int, default=7)
parser.add_argument("--num_radial", type=int, default=6)
parser.add_argument("--num_bilinear", type=int, default=8)
parser.add_argument("--envelope_exponent", type=int, default=5)
parser.add_argument("--num_before_skip", type=int, default=1)
parser.add_argument("--num_after_skip", type=int, default=2)
parser.add_argument("--num_output_layers", type=int, default=3)
parser.add_argument("--max_neighbors", type=int, default=32)
parser.add_argument("--lambda_force", type=float, default=1000.0,
                    help="Weight for force loss component")
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--max_structures", type=int, default=None,
                    help="Limit dataset size for testing")
parser.add_argument("--output", type=str, default="correction_model.pth")
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

# ------------------------------------------------------------------ #
# 1. Build & freeze the vacuum model
# ------------------------------------------------------------------ #
vacuum_model = DimeNet(
    hidden_channels=args.hidden,
    out_channels=1,
    num_blocks=args.num_blocks + 1,
    num_bilinear=args.num_bilinear,
    num_spherical=args.num_spherical,
    num_radial=args.num_radial,
    cutoff=args.radius,
    max_num_neighbors=args.max_neighbors,
    envelope_exponent=args.envelope_exponent,
    num_before_skip=args.num_before_skip,
    num_after_skip=args.num_after_skip,
    num_output_layers=args.num_output_layers,
).to(device)

vacuum_model.load_state_dict(torch.load(args.vacuum_ckpt, weights_only=True))
for p in vacuum_model.parameters():
    p.requires_grad = False
vacuum_model.eval()
print(f"Loaded frozen vacuum model from {args.vacuum_ckpt}")

# ------------------------------------------------------------------ #
# 2. Build correction model (smaller)
# ------------------------------------------------------------------ #
correction_model = DimeNet(
    hidden_channels=args.hidden,
    out_channels=1,
    num_blocks=args.num_blocks,
    num_bilinear=args.num_bilinear,
    num_spherical=args.num_spherical,
    num_radial=args.num_radial,
    cutoff=args.radius,
    max_num_neighbors=args.max_neighbors,
    envelope_exponent=args.envelope_exponent,
    num_before_skip=args.num_before_skip,
    num_after_skip=args.num_after_skip,
    num_output_layers=args.num_output_layers,
).to(device)

print(f"Correction model: {sum(p.numel() for p in correction_model.parameters())} params")
print(f"Vacuum model:     {sum(p.numel() for p in vacuum_model.parameters())} params")

optimizer = optim.Adam(correction_model.parameters(), lr=args.lr)
mse = torch.nn.MSELoss()

# ------------------------------------------------------------------ #
# 3. Load solvated dataset (optionally paired with gas)
# ------------------------------------------------------------------ #
dataset = AQMDataset(
    root="../Data/AQM-sol",
    hdf5_path=args.hdf5,
    gas_hdf5_path=args.gas_hdf5,
    energy_key=SOLVATED_ENERGY_TARGET,
    forces_key=SOLVATED_FORCES_TARGET,
    max_structures=args.max_structures,
)

n_total = len(dataset)
n_val = int(n_total * args.val_split)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(seed),
)
print(f"Dataset: {n_total} samples ({n_train} train, {n_val} val)")

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)


def combined_loss(energy_pred, energy_true, forces_pred, forces_true, lambda_force):
    loss_energy = mse(energy_pred, energy_true)
    loss_force = mse(forces_pred, forces_true)
    return loss_energy + lambda_force * loss_force


# ------------------------------------------------------------------ #
# 4. Training loop
# ------------------------------------------------------------------ #
def train_epoch(loader):
    correction_model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()

        with torch.no_grad():
            vacuum_energy = vacuum_model(data.z, data.pos, data.batch)

        correction_energy = correction_model(data.z, data.pos, data.batch)
        total_energy = vacuum_energy + correction_energy

        forces_pred = -torch.autograd.grad(
            outputs=total_energy,
            inputs=data.pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=True,
        )[0]

        loss = combined_loss(
            total_energy, data.y_energy,
            forces_pred, data.y_forces,
            args.lambda_force,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.enable_grad()
def validate_epoch(loader):
    correction_model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad_()

        with torch.no_grad():
            vacuum_energy = vacuum_model(data.z, data.pos, data.batch)

        correction_energy = correction_model(data.z, data.pos, data.batch)
        total_energy = vacuum_energy + correction_energy

        forces_pred = -torch.autograd.grad(
            outputs=total_energy,
            inputs=data.pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=True,
        )[0]

        loss = combined_loss(
            total_energy, data.y_energy,
            forces_pred, data.y_forces,
            args.lambda_force,
        )
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


best_val_loss = float("inf")
patience = 20
epochs_no_improve = 0

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_loss = train_epoch(train_loader)
    val_loss = validate_epoch(val_loader)
    elapsed = time.time() - t0

    print(
        f"Epoch {epoch:3d}/{args.epochs} | "
        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
        f"Time: {elapsed:.2f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(correction_model.state_dict(), args.output)
        print(f"  -> Saved best correction model to {args.output}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

correction_model.load_state_dict(
    torch.load(args.output, weights_only=True)
)
print(f"Training complete. Best val loss: {best_val_loss:.6f}")

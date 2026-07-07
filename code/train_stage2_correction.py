import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

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
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import load_reference_energies, compute_molecular_reference

seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser(
    description="Stage 2 (Option B): Train Correction DimeNetPlus on AQM-sol with frozen vacuum model"
)
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-sol.hdf5",
                    help="Path to AQM-sol.hdf5")
parser.add_argument("--vacuum_ckpt", type=str, default="results/stage1_fold_1.pt",
                    help="Path to trained vacuum model checkpoint")
parser.add_argument("--batchsize", "-b", type=int, default=16)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=6.0)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks", type=int, default=3,
                    help="Fewer blocks for correction model (smaller than vacuum)")
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


def build_dimenet(num_blocks=None):
    n_blocks = num_blocks if num_blocks is not None else args.num_blocks
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS,
        hidden_channels=args.hidden,
        out_channels=1,
        num_blocks=n_blocks,
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


# ---- Dataset: ONLY AQM-sol, no gas pairing (Option B) ----
dataset = AQMDataset(
    root="../Data/AQM-sol",
    hdf5_path=args.hdf5,
    gas_hdf5_path=None,
    energy_key=SOLVATED_ENERGY_TARGET,
    forces_key=SOLVATED_FORCES_TARGET,
    max_structures=args.max_structures,
)
print(f"Dataset: {len(dataset)} samples")

ref_path = os.path.join(os.path.dirname(args.vacuum_ckpt), "atomic_references.json")
if not os.path.exists(ref_path):
    raise FileNotFoundError(
        f"Atomic reference file not found at {ref_path}. "
        f"Train Stage 1 first to generate it."
    )
print(f"Loading atomic reference energies from {ref_path}")
ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
print(f"Reference energies: {ref_energies.cpu().tolist()}")

n_total = len(dataset)
n_val = int(n_total * args.val_split)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(seed),
)
print(f"  Train: {n_train}  Val: {n_val}")

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

# ---- Build models ----
# Vacuum: +1 block (larger)
vacuum_model = build_dimenet(num_blocks=args.num_blocks + 1)
vacuum_model.load_state_dict(torch.load(args.vacuum_ckpt, map_location=device, weights_only=True))
# Dual freeze: no gradients AND exclude from optimizer
for p in vacuum_model.parameters():
    p.requires_grad_(False)
vacuum_model.eval()

# Correction: smaller
correction_model = build_dimenet(num_blocks=args.num_blocks)
correction_model.train()

print(f"Vacuum model:     {sum(p.numel() for p in vacuum_model.parameters()):,} params (frozen)")
print(f"Correction model: {sum(p.numel() for p in correction_model.parameters()):,} params (trainable)")

# Optimizer only sees correction model params (second safeguard)
optimizer = optim.Adam(correction_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-6
)
mse = torch.nn.MSELoss()


def combined_loss(energy_pred, energy_true, forces_pred, forces_true, n_atoms, lambda_force=None):
    loss_e = mse(energy_pred / n_atoms, energy_true / n_atoms)
    loss_f = mse(forces_pred, forces_true)
    return loss_e + lambda_force * loss_f


def train_epoch(loader):
    correction_model.train()
    total_loss = 0.0
    total_esolv_loss = 0.0
    esolv_count = 0
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()

        x = build_one_hot(data, device)
        mol_ref = compute_molecular_reference(x, data.batch, ref_energies, data.num_graphs)
        y_energy_shifted = data.y_energy - mol_ref
        vacuum_energy = vacuum_model(x, data.pos, data.batch)
        correction_energy = correction_model(x, data.pos, data.batch)
        total_energy = vacuum_energy + correction_energy

        forces_pred = -torch.autograd.grad(
            outputs=total_energy,
            inputs=data.pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=True,
        )[0]

        n_atoms = torch.bincount(data.batch).float()
        loss = combined_loss(
            total_energy.view(-1), y_energy_shifted,
            forces_pred, data.y_forces,
            n_atoms=n_atoms,
            lambda_force=args.lambda_force,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(correction_model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Secondary eSOLV metric (if available in batch)
        if hasattr(data, 'y_esolv') and data.y_esolv is not None:
            # In Option B: correction_model(R_sol) approximates eSOLV
            esolv_loss = mse(correction_energy.view(-1), data.y_esolv.view(-1))
            total_esolv_loss += esolv_loss.item() * data.num_graphs
            esolv_count += data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    avg_esolv = total_esolv_loss / esolv_count if esolv_count > 0 else None
    return avg_loss, avg_esolv


@torch.enable_grad()
def validate_epoch(loader):
    correction_model.eval()
    total_loss = 0.0
    total_esolv_loss = 0.0
    esolv_count = 0
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad_()

        x = build_one_hot(data, device)
        mol_ref = compute_molecular_reference(x, data.batch, ref_energies, data.num_graphs)
        y_energy_shifted = data.y_energy - mol_ref
        vacuum_energy = vacuum_model(x, data.pos, data.batch)
        correction_energy = correction_model(x, data.pos, data.batch)
        total_energy = vacuum_energy + correction_energy

        forces_pred = -torch.autograd.grad(
            outputs=total_energy,
            inputs=data.pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=False,
        )[0]

        n_atoms = torch.bincount(data.batch).float()
        loss = combined_loss(
            total_energy.view(-1), y_energy_shifted,
            forces_pred, data.y_forces,
            n_atoms=n_atoms,
            lambda_force=args.lambda_force,
        )
        total_loss += loss.item() * data.num_graphs


        if hasattr(data, 'y_esolv') and data.y_esolv is not None:
            esolv_loss = mse(correction_energy.view(-1), data.y_esolv.view(-1))
            total_esolv_loss += esolv_loss.item() * data.num_graphs
            esolv_count += data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    avg_esolv = total_esolv_loss / esolv_count if esolv_count > 0 else None
    return avg_loss, avg_esolv


# ---- Training loop ----
best_val_loss = float("inf")
patience = 20
epochs_no_improve = 0

# Sanity: snapshot of frozen params sum at start
frozen_params_init_sum = sum(p.sum().item() for p in vacuum_model.parameters())

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_loss, train_esolv = train_epoch(train_loader)
    val_loss, val_esolv = validate_epoch(val_loader)
    elapsed = time.time() - t0

    esolv_str = ""
    if val_esolv is not None:
        esolv_str = f" | eSOLV Train: {train_esolv:.6f} Val: {val_esolv:.6f}"

    print(
        f"Epoch {epoch:3d}/{args.epochs} | "
        f"Train: {train_loss:.6f} | Val: {val_loss:.6f}{esolv_str} | "
        f"Time: {elapsed:.2f}s"
    )

    # Every 5 epochs: sanity check frozen params haven't moved
    if epoch % 5 == 0:
        frozen_sum = sum(p.sum().item() for p in vacuum_model.parameters())
        diff = abs(frozen_sum - frozen_params_init_sum)
        print(f"  [Sanity] Frozen params sum: {frozen_sum:.6e}  (delta from init: {diff:.6e})")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        ckpt_path = os.path.join(args.output_dir, "stage2_correction.pt")
        torch.save(correction_model.state_dict(), ckpt_path)
        print(f"  -> Saved best correction model to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping after {epoch} epochs")
            break

print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

import sys
import os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_parent)
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
import argparse
import time
import numpy as np

from aqm_dataset import AQMDataset
from aqm_config import SOLVATED_ENERGY_TARGET, SOLVATED_FORCES_TARGET
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import (
    fit_atomic_references,
    load_reference_energies,
    compute_molecular_reference,
    save_reference_energies,
)
from ddp_utils import init_ddp, is_main, cleanup, sync_barrier

seed = 42


class _CachedListDataset(torch.utils.data.Dataset):
    """Wraps a pre-built list of torch_geometric Data objects so each conformer
    is materialized exactly once (filter pass) instead of being re-sliced and
    re-built by every subsequent pass (grouping, ref fit, DataLoaders)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

parser = argparse.ArgumentParser(
    description="Stage 2 (Option B): Train Correction DimeNetPlus on AQM-sol with frozen vacuum model"
)
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-sol.hdf5",
                    help="Path to AQM-sol.hdf5")
parser.add_argument("--gas_hdf5", type=str, default="../aqm_data/AQM-gas.hdf5",
                    help="Path to AQM-gas.hdf5 for paired dG target (E_sol - E_gas)")
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
parser.add_argument("--lambda_total", type=float, default=0.05,
                    help="Weight for total-energy regularizer (default 0.05)")
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--max_structures", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank (set by torchrun)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

local_rank, world_size, is_ddp, device = init_ddp()
if is_main(local_rank):
    print(f"Using device: {device}  |  GPUs: {world_size}")


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


# ---- Dataset: AQM-sol paired with AQM-gas (Option B) ----
# The correction target is the true solvation free energy dG = E_sol - E_gas,
# computed from the paired gas conformer's energy (see AQMDataset.gas_energy).
dataset = AQMDataset(
    root="../Data/AQM-sol",
    hdf5_path=args.hdf5,
    gas_hdf5_path=args.gas_hdf5,
    energy_key=SOLVATED_ENERGY_TARGET,
    forces_key=SOLVATED_FORCES_TARGET,
    max_structures=args.max_structures,
)
# ---- Gas-pairing filter + in-memory item cache (each conformer is built
# exactly once; the filter, the molecule grouping, the ref fit and the
# DataLoaders all reuse the cached items instead of re-indexing).
paired_items = []
for i in range(len(dataset)):
    item = dataset[i]
    if hasattr(item, "gas_energy"):
        paired_items.append(item)
if len(paired_items) != len(dataset):
    print(f"NOTE: dropping {len(dataset) - len(paired_items)} conformers without gas pairing")
dataset = _CachedListDataset(paired_items)
print(f"Dataset: {len(dataset)} samples (gas-paired, cached in memory)")

n_total = len(dataset)
# Molecule-level split: all conformers of a molecule stay in one split
# (conformer-level random_split would leak molecule IDs into val)
mol_id_to_idx = {}
for i in range(len(dataset)):
    mol_id_to_idx.setdefault(dataset[i].mol_id, []).append(i)
mol_ids = sorted(mol_id_to_idx.keys())
n_val_mol = max(1, int(len(mol_ids) * args.val_split))
rng = np.random.RandomState(seed)
shuffled_mol_ids = list(mol_ids)
rng.shuffle(shuffled_mol_ids)
val_mol_ids = set(shuffled_mol_ids[:n_val_mol])
train_idx = [i for m in shuffled_mol_ids[n_val_mol:] for i in mol_id_to_idx[m]]
val_idx = [i for m in shuffled_mol_ids[:n_val_mol] for i in mol_id_to_idx[m]]
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
if is_main(local_rank):
    train_mol_ids = set(shuffled_mol_ids[n_val_mol:])
    print(f"  Train: {len(train_idx)} conformers / {len(mol_ids) - n_val_mol} molecules  "
          f"Val: {len(val_idx)} conformers / {n_val_mol} molecules  "
          f"train/val molecule overlap: {len(train_mol_ids & val_mol_ids)}")

# ---- Stage-2 atomic references: fit on Stage-2's OWN train split.
# (2026-08-01 decision: no longer aliased from Stage 1 fold 1. These refs
# anchor the lambda_total regularizer on the SOLVATED total energy, so fitting
# them on solvated train-split energies is the physically consistent, leak-free
# choice, and no downstream stage depends on Stage-1 fold ordering.)
if is_main(local_rank):
    print(f"Fitting Stage-2 atomic references on train split "
          f"({len(train_dataset)} conformers)...")
    ref_energies = fit_atomic_references(train_dataset, ELEMENT_TO_IDX, NUM_ELEMENTS)
    ref_path = os.path.join(args.output_dir, "atomic_references.json")
    save_reference_energies(ref_energies, ELEMENT_TO_IDX, ref_path)
    ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
    print(f"Reference energies: {ref_energies.cpu().tolist()}")
else:
    ref_energies = None
sync_barrier(is_ddp)
if ref_energies is None:
    ref_path = os.path.join(args.output_dir, "atomic_references.json")
    ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)

if is_ddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=train_sampler)
else:
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
raw_correction_model = build_dimenet(num_blocks=args.num_blocks)
raw_correction_model.train()
if is_ddp:
    correction_model = torch.nn.parallel.DistributedDataParallel(
        raw_correction_model, device_ids=[local_rank])
else:
    correction_model = raw_correction_model

if is_main(local_rank):
    print(f"Vacuum model:     {sum(p.numel() for p in vacuum_model.parameters()):,} params (frozen)")
    print(f"Correction model: {sum(p.numel() for p in raw_correction_model.parameters()):,} params (trainable)")

# Optimizer only sees correction model params (second safeguard)
optimizer = optim.Adam(raw_correction_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-6
)
mse = torch.nn.MSELoss()


def combined_loss(
    energy_pred, energy_true, forces_pred, forces_true, n_atoms,
    dG_pred=None, dG_true=None,
    lambda_force=None, lambda_total=None,
):
    loss_total = mse(energy_pred / n_atoms, energy_true / n_atoms)
    loss_f = mse(forces_pred, forces_true)
    loss = loss_f * lambda_force
    if dG_pred is not None and dG_true is not None:
        loss += lambda_total * loss_total
        loss += mse(dG_pred, dG_true)  # primary: correction_model → dG_solv directly
    else:
        loss += loss_total
    return loss


def train_epoch(loader):
    correction_model.train()
    total_loss = 0.0
    total_dG_loss = 0.0
    dG_count = 0
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
        # True solvation free energy: E_solvated - E_gas (paired conformer).
        # NOT y_esolv (eSOLV sits on the total-energy scale, ~corr 0.2 with dG).
        dG_true = data.y_energy - data.gas_energy
        loss = combined_loss(
            total_energy.view(-1), y_energy_shifted,
            forces_pred, data.y_forces,
            n_atoms=n_atoms,
            dG_pred=correction_energy.view(-1),
            dG_true=dG_true.view(-1),
            lambda_force=args.lambda_force,
            lambda_total=args.lambda_total,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(correction_model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Track dG MAE
        dG_loss = mse(correction_energy.view(-1), dG_true.view(-1))
        total_dG_loss += dG_loss.item() * data.num_graphs
        dG_count += data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    avg_dG = total_dG_loss / dG_count if dG_count > 0 else None
    return avg_loss, avg_dG


@torch.enable_grad()
def validate_epoch(loader):
    correction_model.eval()
    total_loss = 0.0
    total_dG_loss = 0.0
    dG_count = 0
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
        # True solvation free energy: E_solvated - E_gas (paired conformer).
        dG_true = data.y_energy - data.gas_energy
        loss = combined_loss(
            total_energy.view(-1), y_energy_shifted,
            forces_pred, data.y_forces,
            n_atoms=n_atoms,
            dG_pred=correction_energy.view(-1),
            dG_true=dG_true.view(-1),
            lambda_force=args.lambda_force,
            lambda_total=args.lambda_total,
        )
        total_loss += loss.item() * data.num_graphs

        dG_loss = mse(correction_energy.view(-1), dG_true.view(-1))
        total_dG_loss += dG_loss.item() * data.num_graphs
        dG_count += data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    avg_dG = total_dG_loss / dG_count if dG_count > 0 else None
    return avg_loss, avg_dG


# ---- Training loop ----
best_val_dG_mae = float("inf")
patience = 10
epochs_no_improve = 0

# Sanity: snapshot of frozen params sum at start
frozen_params_init_sum = sum(p.sum().item() for p in vacuum_model.parameters())

for epoch in range(1, args.epochs + 1):
    if is_ddp:
        train_sampler.set_epoch(epoch)
    t0 = time.time()
    train_loss, train_dG = train_epoch(train_loader)
    val_loss, val_dG = validate_epoch(val_loader)
    elapsed = time.time() - t0

    if is_main(local_rank):
        dG_str = ""
        if val_dG is not None:
            val_dG_eV = np.sqrt(val_dG)
            train_dG_eV = np.sqrt(train_dG)
            dG_str = (f"  |  dG MAE: {train_dG_eV:.6f} / {val_dG_eV:.6f} eV  "
                      f"({train_dG_eV*23.0605:.3f} / {val_dG_eV*23.0605:.3f} kcal/mol)")
        lr_now = optimizer.param_groups[0]["lr"]
        finite_ok = (
            np.isfinite(train_loss) and np.isfinite(val_loss)
            and all(torch.isfinite(p).all().item() for p in correction_model.parameters())
        )
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  |  "
            f"Loss: {train_loss:.6f} / {val_loss:.6f}{dG_str}  |  "
            f"LR: {lr_now:.1e}  finite: {'OK' if finite_ok else 'NAN!'}  |  "
            f"{elapsed:.2f}s"
        )
        print()

        if epoch % 5 == 0:
            frozen_sum = sum(p.sum().item() for p in vacuum_model.parameters())
            diff = abs(frozen_sum - frozen_params_init_sum)
            print(f"    [Sanity] Frozen params sum: {frozen_sum:.6e}  (delta: {diff:.6e})")
            print()

    scheduler.step(val_loss)

    if val_dG is not None and val_dG < best_val_dG_mae:
        best_val_dG_mae = val_dG
        epochs_no_improve = 0
        if is_main(local_rank):
            ckpt_path = os.path.join(args.output_dir, "stage2_correction.pt")
            torch.save(raw_correction_model.state_dict(), ckpt_path)
            best_rmse = np.sqrt(best_val_dG_mae)
            print(f"    [OK] Saved best correction model -> {ckpt_path}")
            print(f"      (dG val RMSE = {best_rmse:.6f} eV = {best_rmse*23.0605:.3f} kcal/mol)")
            print()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            if is_main(local_rank):
                print(f"    [x] Early stopping after {epoch} epochs")
                print()
            break
    if is_main(local_rank):
        print(f"    [stale counter] epochs_no_improve = {epochs_no_improve} "
              f"(patience={patience}, best val dG = {np.sqrt(best_val_dG_mae)*23.0605:.3f} kcal/mol)")
    sync_barrier(is_ddp)

cleanup(is_ddp)

if is_main(local_rank):
    best_rmse = np.sqrt(best_val_dG_mae)
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val dG RMSE: {best_rmse:.6f} eV = {best_rmse*23.0605:.3f} kcal/mol")
    print(f"{'='*60}\n")

import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))
sys.stdout.reconfigure(line_buffering=True)

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
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import load_reference_energies, compute_molecular_reference
from ddp_utils import init_ddp, is_main, cleanup, sync_barrier

seed = 42

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
parser.add_argument("--radius", "-ra", type=float, default=6.0)
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
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank (set by torchrun)")

# Option B paths for comparison
parser.add_argument("--option_b_checkpoint", type=str, default=None,
                    help="Path to Option B correction checkpoint for comparison")
parser.add_argument("--option_b_vacuum_ckpt", type=str, default=None,
                    help="Path to Option B vacuum checkpoint for comparison")
parser.add_argument("--reference_ckpt", type=str, default=None,
                    help="Path to atomic_references.json (default: look beside --option_b_vacuum_ckpt)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

local_rank, world_size, is_ddp, device = init_ddp()
if is_main(local_rank):
    print(f"Using device: {device}  |  GPUs: {world_size}")

def build_model():
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS,
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

# Load atomic reference energies
if args.reference_ckpt is not None:
    ref_path = args.reference_ckpt
elif args.option_b_vacuum_ckpt is not None:
    ref_path = os.path.join(os.path.dirname(args.option_b_vacuum_ckpt), "atomic_references.json")
else:
    ref_path = None

if ref_path is not None and os.path.exists(ref_path):
    print(f"Loading atomic reference energies from {ref_path}")
    ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
    print(f"Reference energies: {ref_energies.cpu().tolist()}")
else:
    if ref_path is not None and not os.path.exists(ref_path):
        print(f"Warning: reference file not found at {ref_path}. Training without energy reference shift.")
    else:
        print("No reference checkpoint provided. Training without energy reference shift.")
    ref_energies = None

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
if is_main(local_rank):
    print(f"  Train: {n_train}  Val: {n_val}")

if is_ddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

# ---- Model ----
raw_model = build_model()
raw_model.train()
if is_ddp:
    model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
else:
    model = raw_model
if is_main(local_rank):
    print(f"Option A model: {sum(p.numel() for p in raw_model.parameters()):,} params")

optimizer = optim.Adam(raw_model.parameters(), lr=args.lr)
mse = torch.nn.MSELoss()

def combined_loss(energy_pred, energy_true, forces_pred, forces_true, n_atoms, lambda_force):
    loss_e = mse(energy_pred / n_atoms, energy_true / n_atoms)
    loss_f = mse(forces_pred, forces_true)
    return loss_e + lambda_force * loss_f

def predict_energy_and_forces(m, data_batch):
    x = build_one_hot(data_batch, device)
    pos = data_batch.pos
    e = m(x, pos, data_batch.batch)
    f = -torch.autograd.grad(e, pos, grad_outputs=torch.ones_like(e),
                             create_graph=True)[0]
    if ref_energies is not None:
        mol_ref = compute_molecular_reference(x, data_batch.batch, ref_energies, data_batch.num_graphs)
    else:
        mol_ref = None
    return e, f, mol_ref

def predict_energy_and_forces_eval(m, data_batch):
    x = build_one_hot(data_batch, device)
    data_batch.pos.requires_grad_(True)
    e = m(x, data_batch.pos, data_batch.batch)
    f = -torch.autograd.grad(e, data_batch.pos, grad_outputs=torch.ones_like(e),
                             create_graph=False)[0]
    if ref_energies is not None:
        mol_ref = compute_molecular_reference(x, data_batch.batch, ref_energies, data_batch.num_graphs)
    else:
        mol_ref = None
    return e, f, mol_ref

def compute_esolv_mae(m, loader):
    m.eval()
    total_mae = 0.0
    count = 0
    for data in loader:
        data = data.to(device)
        e_pred, _, mol_ref = predict_energy_and_forces_eval(m, data)
        for i in range(data.num_graphs):
            if hasattr(data, 'gas_energy'):
                gas_e = data.gas_energy[i].item() if data.gas_energy.dim() > 0 else data.gas_energy.item()
                if ref_energies is not None and mol_ref is not None:
                    # Model predicts shifted energy (residual). Add reference back
                    # to get absolute predicted energy before subtracting gas energy.
                    # NOTE: gas_energy is a raw DFT energy from AQM-gas.hdf5 and includes
                    # the same atomic reference. For the same molecule (same composition),
                    # mol_ref for the solvated conformer equals the gas-phase mol_ref,
                    # so adding mol_ref here and subtracting unshifted gas_e is correct.
                    esolv_pred = (e_pred[i].item() + mol_ref[i].item()) - gas_e
                else:
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
    if is_ddp:
        train_sampler.set_epoch(epoch)
    t0 = time.time()
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        data.pos.requires_grad_()
        optimizer.zero_grad()
        e_pred, f_pred, mol_ref = predict_energy_and_forces(model, data)
        n_atoms = torch.bincount(data.batch).float()
        y_energy_shifted = data.y_energy - mol_ref if mol_ref is not None else data.y_energy
        loss = combined_loss(e_pred.view(-1), y_energy_shifted,
                             f_pred, data.y_forces,
                             n_atoms=n_atoms, lambda_force=args.lambda_force)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 10.0)
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.enable_grad():
        for data in val_loader:
            data = data.to(device)
            data.pos.requires_grad_()
            e_pred, f_pred, mol_ref = predict_energy_and_forces_eval(model, data)
            n_atoms = torch.bincount(data.batch).float()
            y_energy_shifted = data.y_energy - mol_ref if mol_ref is not None else data.y_energy
            loss = combined_loss(e_pred.view(-1), y_energy_shifted,
                                 f_pred, data.y_forces,
                                 n_atoms=n_atoms, lambda_force=args.lambda_force)
            val_loss += loss.item() * data.num_graphs
    val_loss /= len(val_loader.dataset)

    val_esolv_mae = compute_esolv_mae(model, val_loader)

    if is_main(local_rank):
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
        if is_main(local_rank):
            ckpt_path = os.path.join(args.output_dir, "option_a.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  -> Saved best model to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            if is_main(local_rank):
                print(f"  Early stopping after {epoch} epochs")
            break
    sync_barrier(is_ddp)

cleanup(is_ddp)

# ---- Evaluate Option B on same validation set for comparison ----
option_b_esolv_mae = None
if args.option_b_checkpoint and args.option_b_vacuum_ckpt:
    if is_main(local_rank):
        print("\n--- Evaluating Option B on same validation set ---")
    from DimeModels import DimeNetPlus

    # Build vacuum model (+1 block) and correction model
    vacuum_model = DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=args.hidden, out_channels=1,
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
        in_channels=NUM_ELEMENTS, hidden_channels=args.hidden, out_channels=1,
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
        x = build_one_hot(data, device)
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
    if is_main(local_rank):
        if option_b_esolv_mae is not None:
            print(f"  Option B eSOLV MAE: {option_b_esolv_mae:.6f} eV")
        else:
            print("  Could not compute Option B eSOLV MAE (no y_esolv in data)")

if is_main(local_rank):
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
    print(f"  {'Trainable params':<40} {sum(p.numel() for p in raw_model.parameters()):>12,}")
    if option_b_esolv_mae is not None and best_esolv_mae > 0:
        ratio = best_esolv_mae / option_b_esolv_mae
        print(f"  {'Option A / Option B ratio':<40} {ratio:>12.3f}")
        if ratio < 1:
            print(f"  >>> Option A (direct) beats Option B (delta-learning)!")
        else:
            print(f"  >>> Option B (delta-learning) beats Option A (direct)!")
    print(f"{'='*60}")

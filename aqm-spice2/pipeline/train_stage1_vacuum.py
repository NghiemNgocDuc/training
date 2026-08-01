import sys
import os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_parent)
sys.stdout.reconfigure(line_buffering=True)

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
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import fit_atomic_references, save_reference_energies, compute_molecular_reference
from ddp_utils import init_ddp, is_main, cleanup, sync_barrier

seed = 42

parser = argparse.ArgumentParser(description="Stage 1: Train Vacuum DimeNetPlus on AQM-gas")
parser.add_argument("--hdf5", type=str, default="../aqm_data/AQM-gas.hdf5",
                    help="Path to AQM-gas.hdf5")
parser.add_argument("--batchsize", "-b", type=int, default=32)
parser.add_argument("--lr", "-l", type=float, default=0.001)
parser.add_argument("--epochs", "-e", type=int, default=200)
parser.add_argument("--radius", "-ra", type=float, default=6.0)
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
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--max_structures", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1,
                    help="Local rank (set by torchrun)")
args = parser.parse_args()

local_rank, world_size, is_ddp, device = init_ddp()
if is_main(local_rank):
    print(f"Using device: {device}  |  GPUs: {world_size}")

dataset = AQMDataset(
    root="../Data/AQM-gas",
    hdf5_path=args.hdf5,
    energy_key=VACUUM_ENERGY_TARGET,
    forces_key=VACUUM_FORCES_TARGET,
    max_structures=args.max_structures,
)
print(f"Dataset: {len(dataset)} samples")

os.makedirs(args.output_dir, exist_ok=True)


def fit_refs_on_train(train_ds, ref_tag, val_ids=None):
    """Fit atomic references on the TRAIN split only (never val/test).

    Mirrors the MACE pipeline's fit_dataset=train_ds pattern: the reference
    energies for a fold/stage are computed exclusively from that split's
    training molecules.
    """
    ref_energies = fit_atomic_references(train_ds, ELEMENT_TO_IDX, NUM_ELEMENTS)
    ref_energies = ref_energies.to(device)

    train_ids = set()
    for d in train_ds:
        train_ids.add(d.mol_id)
    print(f"  [refs] fit dataset: {type(train_ds).__name__} "
          f"({len(train_ds)} samples, {len(train_ids)} molecules)")
    if val_ids is not None:
        overlap = train_ids & val_ids
        print(f"  [refs] train/val molecule overlap: {len(overlap)} (must be 0)")

    if ref_tag is None:
        ref_path = os.path.join(args.output_dir, "atomic_references.json")
    else:
        ref_path = os.path.join(args.output_dir, f"atomic_references_{ref_tag}.json")
    if is_main(local_rank):
        save_reference_energies(ref_energies.cpu(), ELEMENT_TO_IDX, ref_path)
        if ref_tag == "fold_1":
            # Back-compat alias: stage 2 / evaluate default to this filename
            save_reference_energies(ref_energies.cpu(), ELEMENT_TO_IDX,
                                    os.path.join(args.output_dir, "atomic_references.json"))
    print(f"Reference energies tensor ({NUM_ELEMENTS} elements): {ref_energies.cpu().tolist()}")
    return ref_energies

mse = torch.nn.MSELoss()


def combined_loss(energy_pred, energy_true, forces_pred, forces_true, n_atoms, lambda_force):
    loss_e = mse(energy_pred / n_atoms, energy_true / n_atoms)
    loss_f = mse(forces_pred, forces_true)
    return loss_e + lambda_force * loss_f


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


def train_one_fold(train_loader, val_loader, fold_idx, ref_energies, sampler=None):
    raw_model = build_model()
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    else:
        model = raw_model
    optimizer = optim.Adam(raw_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0
    ckpt_path = os.path.join(args.output_dir, f"stage1_fold_{fold_idx}.pt")

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        t0 = time.time()
        for data in train_loader:
            data = data.to(device)
            data.pos.requires_grad_()
            optimizer.zero_grad()

            x = build_one_hot(data, device)
            mol_ref = compute_molecular_reference(x, data.batch, ref_energies, data.num_graphs)
            y_energy_shifted = data.y_energy - mol_ref
            energy_pred = model(x, data.pos, data.batch)
            forces_pred = -torch.autograd.grad(
                outputs=energy_pred,
                inputs=data.pos,
                grad_outputs=torch.ones_like(energy_pred),
                create_graph=True,
            )[0]

            n_atoms = torch.bincount(data.batch).float()
            loss = combined_loss(
                energy_pred.view(-1), y_energy_shifted,
                forces_pred, data.y_forces,
                n_atoms=n_atoms,
                lambda_force=args.lambda_force,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.enable_grad():
            for data in val_loader:
                data = data.to(device)
                data.pos.requires_grad_()
                x = build_one_hot(data, device)
                mol_ref = compute_molecular_reference(x, data.batch, ref_energies, data.num_graphs)
                y_energy_shifted = data.y_energy - mol_ref
                energy_pred = model(x, data.pos, data.batch)
                forces_pred = -torch.autograd.grad(
                    outputs=energy_pred,
                    inputs=data.pos,
                    grad_outputs=torch.ones_like(energy_pred),
                    create_graph=False,
                )[0]
                n_atoms = torch.bincount(data.batch).float()
                loss = combined_loss(
                    energy_pred.view(-1), y_energy_shifted,
                    forces_pred, data.y_forces,
                    n_atoms=n_atoms,
                    lambda_force=args.lambda_force,
                )
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_loader.dataset)

        torch.cuda.empty_cache()

        if is_main(local_rank):
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{args.epochs}  |  Train: {train_loss:.6f}  |  Val: {val_loss:.6f}  |  LR: {current_lr:.2e}  |  {elapsed:.2f}s")
            print()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if is_main(local_rank):
                torch.save(raw_model.state_dict(), ckpt_path)
                print(f"    [OK] Saved best model -> {ckpt_path}")
                print()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if is_main(local_rank):
                    print(f"    [x] Early stopping after {epoch} epochs")
                    print()
                break
        sync_barrier(is_ddp)

    return best_val_loss


fold_results = []

# Molecule-level splits: all conformers of a molecule stay in the same split.
# Conformer-level KFold would leak molecule IDs into the val set, which would
# then leak into the per-fold atomic-reference fit (see fit_refs_on_train).
mol_id_to_idx = {}
for i in range(len(dataset)):
    mol_id_to_idx.setdefault(dataset[i].mol_id, []).append(i)
mol_ids = sorted(mol_id_to_idx.keys())
print(f"Dataset: {len(dataset)} conformers from {len(mol_ids)} molecules")

if args.k_folds <= 1:
    n_val_mol = max(1, int(len(mol_ids) * args.val_split))
    rng = np.random.RandomState(seed)
    shuffled_mol_ids = list(mol_ids)
    rng.shuffle(shuffled_mol_ids)
    val_mol_ids = set(shuffled_mol_ids[:n_val_mol])
    train_idx = [i for m in shuffled_mol_ids[n_val_mol:] for i in mol_id_to_idx[m]]
    val_idx = [i for m in shuffled_mol_ids[:n_val_mol] for i in mol_id_to_idx[m]]
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batchsize, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batchsize, shuffle=False)
    if is_main(local_rank):
        print(f"\n{'='*60}\nSingle train/val split "
              f"({len(train_idx)} train / {len(val_idx)} val conformers, "
              f"{len(mol_ids) - n_val_mol} train / {n_val_mol} val molecules)\n{'='*60}\n")
    val_ids = set(val_mol_ids)
    ref_energies = fit_refs_on_train(train_ds, ref_tag=None, val_ids=val_ids)
    best_loss = train_one_fold(train_loader, val_loader, 1, ref_energies,
                                sampler=train_sampler if is_ddp else None)
    fold_results.append(best_loss)
else:
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
    for fold, (train_mol_pos, val_mol_pos) in enumerate(kf.split(mol_ids)):
        train_mol_ids = [mol_ids[j] for j in train_mol_pos]
        val_mol_ids = [mol_ids[j] for j in val_mol_pos]
        train_idx = [i for m in train_mol_ids for i in mol_id_to_idx[m]]
        val_idx = [i for m in val_mol_ids for i in mol_id_to_idx[m]]
        if is_main(local_rank):
            print(f"\n{'='*60}\nFold {fold + 1}/{args.k_folds} "
                  f"({len(train_idx)} train / {len(val_idx)} val conformers, "
                  f"{len(train_mol_ids)} train / {len(val_mol_ids)} val molecules)\n{'='*60}\n")
        train_subset = Subset(dataset, train_idx)
        ref_energies = fit_refs_on_train(train_subset, ref_tag=f"fold_{fold + 1}",
                                         val_ids=set(val_mol_ids))
        if is_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_subset, shuffle=True)
            train_loader = DataLoader(train_subset, batch_size=args.batchsize, sampler=train_sampler)
        else:
            train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batchsize, shuffle=False)
        best_loss = train_one_fold(train_loader, val_loader, fold + 1, ref_energies,
                                    sampler=(train_sampler if is_ddp else None))
        fold_results.append(best_loss)
        if is_main(local_rank):
            print(f"Fold {fold + 1} best val loss: {best_loss:.6f}\n")

cleanup(is_ddp)

if is_main(local_rank):
    print(f"\n{'='*60}")
    if len(fold_results) > 1:
        print(f"\nCV complete. Best val losses: {[f'{l:.6f}' for l in fold_results]}")
        print(f"Mean val loss: {np.mean(fold_results):.6f} +/- {np.std(fold_results):.6f}")
    else:
        print(f"\nTraining complete. Best val loss: {fold_results[0]:.6f}")
    print(f"{'='*60}\n")

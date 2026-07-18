import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

import json
import torch
import numpy as np
import math
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
from aqm_dataset import AQMDataset
from aqm_config import VACUUM_ENERGY_TARGET, VACUUM_FORCES_TARGET
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import load_reference_energies, compute_molecular_reference
import argparse

# Standard atomic masses (amu) keyed by atomic number
ATOMIC_MASSES = {
    1: 1.008, 3: 6.941, 5: 10.811, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 11: 22.990, 12: 24.305, 14: 28.086, 15: 30.974,
    16: 32.065, 17: 35.453, 19: 39.098, 20: 40.078, 35: 79.904, 53: 126.904,
}

CONV = 0.0096485  # (eV/A) / amu -> A/fs^2

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate trained DimeNetPlus model")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
parser.add_argument("--hdf5", type=str, required=True,
                    help="Path to HDF5 dataset")
parser.add_argument("--dataset_root", type=str, default="../Data/AQM-gas-eval",
                    help="Dataset cache root")
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--max_structures", type=int, default=None)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--md_steps", type=int, default=100,
                    help="Number of MD steps for stability test")
parser.add_argument("--md_dt", type=float, default=0.5,
                    help="MD timestep in fs")
parser.add_argument("--force_threshold", type=float, default=50.0,
                    help="Force threshold for stability (eV/A)")
parser.add_argument("--md_temp_K", type=float, default=300.0,
                    help="Initial temperature for MD (K)")
parser.add_argument("--md_gamma", type=float, default=0.5,
                    help="Langevin friction coefficient (fs^-1); 0.5 fs^-1 = 500 ps^-1 (strong)")
parser.add_argument("--md_clip_force", type=float, default=30.0,
                    help="Clip per-atom forces to this magnitude during MD (eV/A); 0 = no clip")
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--num_blocks", type=int, default=4)
parser.add_argument("--int_emb_size", type=int, default=64)
parser.add_argument("--basis_emb_size", type=int, default=8)
parser.add_argument("--out_emb_channels", type=int, default=256)
parser.add_argument("--num_spherical", type=int, default=7)
parser.add_argument("--num_radial", type=int, default=6)
parser.add_argument("--radius", type=float, default=5.0)
parser.add_argument("--max_neighbors", type=int, default=32)
parser.add_argument("--envelope_exponent", type=int, default=5)
parser.add_argument("--num_before_skip", type=int, default=1)
parser.add_argument("--num_after_skip", type=int, default=2)
parser.add_argument("--num_output_layers", type=int, default=3)
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print(f"Using device: {device}")

os.makedirs(args.output_dir, exist_ok=True)

# ---- Build model with same architecture as stage 1 ----
def build_vacuum_model():
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS,
        hidden_channels=args.hidden, out_channels=1, num_blocks=args.num_blocks,
        int_emb_size=args.int_emb_size, basis_emb_size=args.basis_emb_size,
        out_emb_channels=args.out_emb_channels,
        num_spherical=args.num_spherical, num_radial=args.num_radial,
        cutoff=args.radius, max_num_neighbors=args.max_neighbors,
        envelope_exponent=args.envelope_exponent,
        num_before_skip=args.num_before_skip, num_after_skip=args.num_after_skip,
        num_output_layers=args.num_output_layers, is_energy=True,
    ).to(device)

model = build_vacuum_model()
state = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {args.checkpoint}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- Load atomic reference energies ----
ref_path = os.path.join(os.path.dirname(args.checkpoint), "atomic_references.json")
if os.path.exists(ref_path):
    print(f"Loading atomic reference energies from {ref_path}")
    ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
    print(f"Reference energies: {ref_energies.cpu().tolist()}")
else:
    print(f"Note: no atomic_references.json found at {ref_path}. "
          f"Energy predictions used as-is (no reference shift correction).")
    ref_energies = None

# ---- Load dataset ----
dataset = AQMDataset(
    root=args.dataset_root,
    hdf5_path=args.hdf5,
    energy_key=VACUUM_ENERGY_TARGET,
    forces_key=VACUUM_FORCES_TARGET,
    max_structures=args.max_structures,
)
print(f"Dataset: {len(dataset)} samples")

from torch.utils.data import random_split
n_val = max(1, int(len(dataset) * 0.1))
n_test = max(1, len(dataset) - n_val)
_, test_dataset = random_split(dataset, [n_val, n_test],
                               generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
print(f"Test samples: {len(test_dataset)}")


# ---- 1. Standard force MAE ----
print("\n--- Computing force MAE ---")
all_force_mae = []
all_energies = []

for data in test_loader:
    data = data.to(device)
    x = build_one_hot(data, device)
    data.pos.requires_grad_(True)
    energy_pred = model(x, data.pos, data.batch)
    forces_pred = -torch.autograd.grad(
        energy_pred, data.pos,
        grad_outputs=torch.ones_like(energy_pred),
        create_graph=False,
    )[0]

    for i in range(data.num_graphs):
        mask = data.batch == i
        f_pred = forces_pred[mask]
        f_true = data.y_forces[mask]
        e_true = data.y_energy[i].item()

        all_force_mae.append((f_pred - f_true).abs().mean().item())
        all_energies.append(e_true)

all_force_mae = np.array(all_force_mae)
all_energies = np.array(all_energies)

overall_force_mae = float(np.mean(all_force_mae))
print(f"  Overall force MAE: {overall_force_mae:.6f} eV/A")

# ---- 2. Force MAE breakdown by energy ----
print("\n--- Force MAE breakdown by energy ---")
# Low energy: bottom 80%, High energy: top 20%
threshold = np.percentile(all_energies, 80)
low_mask = all_energies <= threshold
high_mask = all_energies > threshold

low_force_mae = float(np.mean(all_force_mae[low_mask])) if low_mask.sum() > 0 else 0.0
high_force_mae = float(np.mean(all_force_mae[high_mask])) if high_mask.sum() > 0 else 0.0

print(f"  Low-energy structures  (n={low_mask.sum()}): {low_force_mae:.6f} eV/A")
print(f"  High-energy structures (n={high_mask.sum()}): {high_force_mae:.6f} eV/A")
print(f"  Ratio (high/low): {high_force_mae / low_force_mae:.3f}" if low_force_mae > 0 else "")


# ---- 3. Simulation stability + 4. Energy conservation ----
# Run MD on individual molecules (single-graph batches)
print(f"\n--- MD Stability Test ({args.md_steps} steps, threshold={args.force_threshold} eV/A) ---")
print(f"--- Energy Conservation Check (dt={args.md_dt} fs) ---")

boltzmann_k = 8.617333262e-5  # eV/K

def get_mass(z):
    masses = []
    for zn in z:
        zn_int = int(zn)
        if zn_int not in ATOMIC_MASSES:
            raise ValueError(f"No atomic mass defined for element Z={zn_int}")
        masses.append(ATOMIC_MASSES[zn_int])
    return torch.tensor(masses, dtype=torch.float, device=device)

stable_count = 0
total_md = 0
energy_drifts = []
total_clip_count = 0

for idx in range(len(test_dataset)):
    data = test_dataset[idx]
    data = data.to(device)
    n_atoms = data.z.size(0)
    masses = get_mass(data.z)

    x = build_one_hot(data, device)
    pos = data.pos.clone().to(device).requires_grad_(True)

    # Initialize velocities from Maxwell-Boltzmann at args.md_temp_K
    # <v^2> = CONV * kT / m  (from equipartition: 0.5 * kT = 0.5 * (m/CONV) * <v^2>)
    vel_std = torch.sqrt(CONV * boltzmann_k * args.md_temp_K / masses.view(-1, 1))
    vel = torch.randn(n_atoms, 3, device=device) * vel_std

    def compute_energy_and_forces(p):
        p = p.requires_grad_(True)
        e = model(x, p, None)
        if ref_energies is not None:
            mol_ref = compute_molecular_reference(x, None, ref_energies, 1)
            e = e + mol_ref  # restore absolute energy from shifted prediction
        f = -torch.autograd.grad(e, p, grad_outputs=torch.ones_like(e),
                                 create_graph=False)[0]
        return e, f

    # Initial state
    energy0, forces0 = compute_energy_and_forces(pos)
    initial_potential = energy0.item()

    # Langevin OABAO integrator (Leimkuhler & Matthews, 2015)
    # O: Ornstein-Uhlenbeck (friction + noise)
    # A: Half-step velocity kick from forces
    # B: Position update
    gamma = args.md_gamma  # fs^-1
    friction_half = math.exp(-gamma * args.md_dt / 2)  # scalar (same for all atoms)

    step = 0
    unstable = False
    clip_count = 0
    clipped_warned = False
    trajectory = {"potential": [initial_potential],
                  "kinetic": [], "total": []}

    pos = pos.detach().requires_grad_(True)
    forces = forces0.detach()
    if args.md_clip_force > 0:
        forces = torch.clamp(forces, -args.md_clip_force, args.md_clip_force)
    accel = forces * CONV / masses.view(-1, 1)

    for step in range(args.md_steps):
        # Check stability threshold
        max_force = forces.abs().max().item()
        if max_force > args.force_threshold:
            if args.md_clip_force > 0:
                if not clipped_warned:
                    print(f"  Molecule {idx}: force >{args.force_threshold:.0f} at step {step} ({max_force:.1f} eV/A), clipped")
                    clipped_warned = True
            else:
                unstable = True
                print(f"  Molecule {idx}: unstable at step {step} (max force={max_force:.2f} eV/A)")
                break

        # O half-step: friction + noise (fluctuation-dissipation)
        noise_std = torch.sqrt(
            boltzmann_k * args.md_temp_K * CONV * (1.0 - friction_half**2) / masses.view(-1, 1)
        )
        vel = vel * friction_half + noise_std * torch.randn_like(vel)

        # A half-step: velocity kick from forces
        vel = vel + 0.5 * accel * args.md_dt

        # B: position update
        pos = pos + vel * args.md_dt
        pos = pos.detach().requires_grad_(True)

        # Compute new forces
        energy, forces = compute_energy_and_forces(pos)
        forces = forces.detach()
        if args.md_clip_force > 0:
            c = forces.abs().max().item()
            if c > args.md_clip_force:
                clip_count += 1
                forces = torch.clamp(forces, -args.md_clip_force, args.md_clip_force)
        accel = forces * CONV / masses.view(-1, 1)

        # A half-step: velocity kick from forces
        vel = vel + 0.5 * accel * args.md_dt

        # O half-step: friction + noise
        vel = vel * friction_half + noise_std * torch.randn_like(vel)

        # Track energy
        kinetic = 0.5 * (masses * (vel ** 2).sum(dim=1)).sum() / CONV
        potential = energy.item()
        total_e = potential + kinetic.item()

        trajectory["potential"].append(potential)
        trajectory["kinetic"].append(kinetic.item())
        trajectory["total"].append(total_e)

    total_md += 1

    if not unstable:
        stable_count += 1
        # Energy drift: linear fit of total energy over steps
        total_arr = np.array(trajectory["total"])
        steps_arr = np.arange(len(total_arr))
        if len(total_arr) > 1:
            coeffs = np.polyfit(steps_arr, total_arr, 1)
            drift = coeffs[0]  # eV per step
            energy_drifts.append(drift)

    total_clip_count += clip_count

stability_fraction = stable_count / total_md if total_md > 0 else 0.0
clip_fraction = total_clip_count / (total_md * args.md_steps) if total_md > 0 and args.md_steps > 0 else 0.0
mean_drift = float(np.mean(energy_drifts)) if energy_drifts else 0.0
mean_abs_drift = float(np.mean(np.abs(energy_drifts))) if energy_drifts else 0.0

print(f"\n--- Results ---")
print(f"  Stable molecules: {stable_count}/{total_md} ({stability_fraction:.2%})")
if energy_drifts:
    print(f"  Mean energy drift: {mean_drift:.6f} eV/step")
    print(f"  Mean |drift|:      {mean_abs_drift:.6f} eV/step")
if args.md_clip_force > 0:
    print(f"  Force clipping: {clip_fraction:.4%} of steps clipped at {args.md_clip_force} eV/A")

# ---- Save results ----
results = {
    "force_mae": {
        "overall": overall_force_mae,
        "low_energy": low_force_mae,
        "high_energy": high_force_mae,
        "n_low": int(low_mask.sum()),
        "n_high": int(high_mask.sum()),
        "energy_percentile_threshold": float(threshold),
    },
    "md_stability": {
        "stable_fraction": stability_fraction,
        "stable_count": stable_count,
        "total_molecules": total_md,
        "steps": args.md_steps,
        "force_threshold_eV_per_A": args.force_threshold,
        "clip_enabled": args.md_clip_force > 0,
        "clip_max_eV_per_A": args.md_clip_force,
        "clip_fraction": clip_fraction,
        "clip_total_steps": total_clip_count,
    },
    "energy_conservation": {
        "mean_drift_eV_per_step": mean_drift,
        "mean_abs_drift_eV_per_step": mean_abs_drift,
        "n_molecules": len(energy_drifts),
    },
    "config": {
        "checkpoint": args.checkpoint,
        "dataset": args.hdf5,
        "md_dt_fs": args.md_dt,
        "md_temp_K": args.md_temp_K,
        "md_gamma_fs_inv": args.md_gamma,
    },
}

out_path = os.path.join(args.output_dir, "evaluation_metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved evaluation metrics to {out_path}")

# Print JSON
print(json.dumps(results, indent=2))

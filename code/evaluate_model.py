import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))

import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from DimeModels import DimeNetPlus
from aqm_dataset import AQMDataset
from aqm_config import VACUUM_ENERGY_TARGET, VACUUM_FORCES_TARGET
import argparse

# Standard atomic masses (amu) keyed by atomic number
ATOMIC_MASSES = {
    1: 1.008,   6: 12.011,  7: 14.007,  8: 15.999,
    9: 18.998,  16: 32.065, 17: 35.453, 35: 79.904,
}
DEFAULT_MASS = 12.0  # fallback

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
        in_channels=1,
        hidden_channels=128, out_channels=1, num_blocks=4,
        int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=5.0,
        max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2,
        num_output_layers=3, is_energy=True,
    ).to(device)

model = build_vacuum_model()
state = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {args.checkpoint}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    x = data.z.float().view(-1, 1)
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
    return torch.tensor([ATOMIC_MASSES.get(int(zn), DEFAULT_MASS) for zn in z],
                        dtype=torch.float, device=device)

stable_count = 0
total_md = 0
energy_drifts = []

for idx in range(len(test_dataset)):
    data = test_dataset[idx]
    data = data.to(device)
    n_atoms = data.z.size(0)
    masses = get_mass(data.z)

    x = data.z.float().view(-1, 1)
    pos = data.pos.clone().to(device).requires_grad_(True)

    # Initialize velocities from Maxwell-Boltzmann at args.md_temp_K
    # <v^2> = CONV * kT / m  (from equipartition: 0.5 * kT = 0.5 * (m/CONV) * <v^2>)
    vel_std = torch.sqrt(CONV * boltzmann_k * args.md_temp_K / masses.view(-1, 1))
    vel = torch.randn(n_atoms, 3, device=device) * vel_std

    def compute_energy_and_forces(p):
        p = p.requires_grad_(True)
        e = model(x, p, None)
        f = -torch.autograd.grad(e, p, grad_outputs=torch.ones_like(e),
                                 create_graph=False)[0]
        return e, f

    # Initial state
    energy0, forces0 = compute_energy_and_forces(pos)
    initial_potential = energy0.item()

    # Velocity Verlet with force threshold check
    step = 0
    unstable = False
    trajectory = {"potential": [initial_potential],
                  "kinetic": [], "total": []}

    pos = pos.detach().requires_grad_(True)
    forces = forces0.detach()
    accel = forces * CONV / masses.view(-1, 1)

    for step in range(args.md_steps):
        # Check force threshold
        max_force = forces.abs().max().item()
        if max_force > args.force_threshold:
            unstable = True
            print(f"  Molecule {idx}: unstable at step {step} (max force={max_force:.2f} eV/A)")
            break

        # Half-step velocity
        vel = vel + 0.5 * accel * args.md_dt

        # Full-step position
        pos = pos + vel * args.md_dt
        pos = pos.detach().requires_grad_(True)

        # Compute new forces
        energy, forces = compute_energy_and_forces(pos)
        forces = forces.detach()
        accel = forces * CONV / masses.view(-1, 1)

        # Full-step velocity
        vel = vel + 0.5 * accel * args.md_dt

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

stability_fraction = stable_count / total_md if total_md > 0 else 0.0
mean_drift = float(np.mean(energy_drifts)) if energy_drifts else 0.0
mean_abs_drift = float(np.mean(np.abs(energy_drifts))) if energy_drifts else 0.0

print(f"\n--- Results ---")
print(f"  Stable molecules: {stable_count}/{total_md} ({stability_fraction:.2%})")
if energy_drifts:
    print(f"  Mean energy drift: {mean_drift:.6f} eV/step")
    print(f"  Mean |drift|:      {mean_abs_drift:.6f} eV/step")

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
    },
}

out_path = os.path.join(args.output_dir, "evaluation_metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved evaluation metrics to {out_path}")

# Print JSON
print(json.dumps(results, indent=2))

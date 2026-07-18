import sys, os, json, math, time, datetime
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "aqm_data"))
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset

from DimeModels import DimeNetPlus
from aqm_dataset import AQMDataset
from aqm_config import VACUUM_ENERGY_TARGET, VACUUM_FORCES_TARGET, SOLVATED_ENERGY_TARGET, SOLVATED_FORCES_TARGET
from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS, build_one_hot
from energy_reference import load_reference_energies, compute_molecular_reference

# ---- Config ----
HDF5_DIR = _root
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device = torch.device("cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Constants ----
ATOMIC_MASSES = {
    1: 1.008, 3: 6.941, 5: 10.811, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 11: 22.990, 12: 24.305, 14: 28.086, 15: 30.974,
    16: 32.065, 17: 35.453, 19: 39.098, 20: 40.078, 35: 79.904, 53: 126.904,
}
CONV = 0.0096485
boltzmann_k = 8.617333262e-5
BOHR_TO_ANG = 0.529177
HARTREE_TO_EV = 27.2114
HARTREE_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANG

# ---- Build model ----
def build_model(num_blocks, hidden=128, radius=6.0):
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=hidden, out_channels=1,
        num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=radius, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    ).to(device)

def get_mass(z):
    masses = [ATOMIC_MASSES[int(zn)] for zn in z]
    return torch.tensor(masses, dtype=torch.float, device=device)

# ---- ZBL ----
_ZBL_COEFFS = [(0.18175, 3.19980), (0.50986, 0.94229), (0.28022, 0.40290), (0.02817, 0.20162)]

def zbl_energy(pos, z, cutoff_low=0.5, cutoff_high=1.0):
    n = pos.size(0)
    dist = torch.cdist(pos, pos).clamp(min=1e-6)
    Zi = z.view(-1, 1).expand(n, n)
    Zj = z.view(1, -1).expand(n, n)
    a = 0.529 * 0.46850 / (Zi.pow(0.23) + Zj.pow(0.23))
    x = dist / a
    phi = torch.zeros_like(dist)
    for c, d in _ZBL_COEFFS:
        phi = phi + c * torch.exp(-d * x)
    e_pair = 14.399645 * Zi * Zj / dist * phi
    switch = torch.where(dist < cutoff_low, torch.ones_like(dist),
                torch.where(dist > cutoff_high, torch.zeros_like(dist),
                    0.5 * (1.0 + torch.cos(math.pi * (dist - cutoff_low) / (cutoff_high - cutoff_low)))))
    self_mask = torch.eye(n, device=pos.device, dtype=torch.bool)
    return 0.5 * (e_pair * (~self_mask).float() * switch * (~self_mask).float()).sum()

def min_pairwise_distance(pos):
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float('inf'))
    return dist.min().item()

# ============================================================
# PART 1: Force MAE — Baseline comparison
# ============================================================
print("=" * 70)
print("PART 1: Force MAE comparison across approaches")
print("=" * 70)

gas_hdf5 = os.path.join(HDF5_DIR, "AQM-gas.hdf5")
sol_hdf5 = os.path.join(HDF5_DIR, "AQM-sol.hdf5")
ref_path = os.path.join(RESULTS_DIR, "atomic_references.json")

# Load ref energies
ref_energies = load_reference_energies(ref_path, ELEMENT_TO_IDX, NUM_ELEMENTS, device)
print(f"Reference energies loaded from {ref_path}")

# Load AQM-sol dataset for evaluation
print("\nLoading AQM-sol dataset...")
t0 = time.time()
sol_dataset = AQMDataset(
    root=os.path.join(HDF5_DIR, "Data", "AQM-sol-eval"),
    hdf5_path=sol_hdf5,
    gas_hdf5_path=gas_hdf5,
    energy_key=SOLVATED_ENERGY_TARGET,
    forces_key=SOLVATED_FORCES_TARGET,
    max_structures=20,
)
sol_loader = DataLoader(sol_dataset, batch_size=8, shuffle=False)
print(f"  {len(sol_dataset)} samples ({time.time()-t0:.1f}s)")

# Build models
vacuum_model = build_model(num_blocks=4)
vacuum_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "stage1_fold_1.pt"), map_location=device, weights_only=True))
vacuum_model.eval()
for p in vacuum_model.parameters(): p.requires_grad_(False)

correction_model = build_model(num_blocks=3)
correction_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "stage2_correction.pt"), map_location=device, weights_only=True))
correction_model.eval()
for p in correction_model.parameters(): p.requires_grad_(False)

option_a_model = build_model(num_blocks=3)
option_a_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "option_a.pt"), map_location=device, weights_only=True))
option_a_model.eval()

def compute_force_mae(model, loader, name, use_vacuum_correction=False):
    total_mae = 0.0
    total_rmse = 0.0
    count = 0
    all_mae = []
    all_energies = []
    for data in loader:
        data = data.to(device)
        x = build_one_hot(data, device)
        data.pos.requires_grad_(True)

        if use_vacuum_correction:
            with torch.no_grad():
                vac_e = vacuum_model(x, data.pos, data.batch)
            corr_e = model(x, data.pos, data.batch)
            total_e = vac_e + corr_e
        else:
            total_e = model(x, data.pos, data.batch)

        forces_pred = -torch.autograd.grad(total_e, data.pos,
            grad_outputs=torch.ones_like(total_e), create_graph=False)[0]

        for i in range(data.num_graphs):
            mask = data.batch == i
            f_pred = forces_pred[mask]
            f_true = data.y_forces[mask]
            mae = (f_pred - f_true).abs().mean().item()
            total_mae += mae
            total_rmse += ((f_pred - f_true) ** 2).mean().item()
            all_mae.append(mae)
            all_energies.append(data.y_energy[i].item())
            count += 1

    overall_mae = total_mae / count
    overall_rmse = math.sqrt(total_rmse / count)
    all_mae = np.array(all_mae)
    all_energies = np.array(all_energies)
    threshold = np.percentile(all_energies, 80)
    low_mae = np.mean(all_mae[all_energies <= threshold]) if (all_energies <= threshold).sum() > 0 else 0
    high_mae = np.mean(all_mae[all_energies > threshold]) if (all_energies > threshold).sum() > 0 else 0
    print(f"  {name}:")
    print(f"    Force MAE:  {overall_mae:.6f} eV/A")
    print(f"    Force RMSE: {overall_rmse:.6f} eV/A")
    print(f"    Low-energy: {low_mae:.6f} eV/A  High-energy: {high_mae:.6f} eV/A")
    return {"mae": overall_mae, "rmse": overall_rmse, "low_mae": low_mae, "high_mae": high_mae}

results_part1 = {}

print("\n--- 1. Option A (scratch on AQM-sol) ---")
results_part1["option_a"] = compute_force_mae(option_a_model, sol_loader, "Option A (scratch)")

print("\n--- 2. Option B (delta-learning: vacuum + correction) ---")
results_part1["option_b"] = compute_force_mae(correction_model, sol_loader, "Option B (delta)", use_vacuum_correction=True)

print("\n--- 3. Vacuum only (baseline) ---")
results_part1["vacuum_only"] = compute_force_mae(vacuum_model, sol_loader, "Vacuum only")

# ============================================================
# PART 2: A/B — MD with ZBL on vs off
# ============================================================
print("\n" + "=" * 70)
print("PART 2: MD Stability — ZBL ON vs ZBL OFF")
print("=" * 70)

gas_eval_dataset = AQMDataset(
    root=os.path.join(HDF5_DIR, "AQM-gas-eval"),
    hdf5_path=gas_hdf5,
    energy_key=VACUUM_ENERGY_TARGET,
    forces_key=VACUUM_FORCES_TARGET,
    max_structures=10,
)
gas_eval_loader = DataLoader(gas_eval_dataset, batch_size=1, shuffle=False)
print(f"Dataset: {len(gas_eval_dataset)} samples")

md_steps = 50
md_dt = 0.1
force_threshold = 50.0
md_temp_K = 300.0
md_gamma = 0.5

def run_md(model, dataset, use_zbl, name):
    stable_count = 0
    total = 0
    energy_drifts = []
    results_list = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        data = data.to(device)
        n_atoms = data.z.size(0)
        masses = get_mass(data.z)
        x = build_one_hot(data, device)
        pos = data.pos.clone().to(device)

        vel_std = torch.sqrt(CONV * boltzmann_k * md_temp_K / masses.view(-1, 1))
        vel = torch.randn(n_atoms, 3, device=device) * vel_std

        def compute_energy_and_forces(p):
            p = p.requires_grad_(True)
            e = model(x, p, None)
            mol_ref = compute_molecular_reference(x, None, ref_energies, 1)
            e = e + mol_ref
            if use_zbl:
                e = e + zbl_energy(p, data.z, 0.5, 1.0)
            f = -torch.autograd.grad(e, p, grad_outputs=torch.ones_like(e), create_graph=False)[0]
            return e, f

        energy0, forces0 = compute_energy_and_forces(pos)
        initial_potential = energy0.item()
        gamma = md_gamma
        friction_half = math.exp(-gamma * md_dt / 2)

        pos = pos.detach().requires_grad_(True)
        forces = forces0.detach()
        accel = forces * CONV / masses.view(-1, 1)
        unstable = False
        trajectory_pot = [initial_potential]
        trajectory_kin = []

        min_dist_initial = min_pairwise_distance(pos)
        trace_diag = []

        for step in range(md_steps):
            noise_std = torch.sqrt(boltzmann_k * md_temp_K * CONV * (1.0 - friction_half**2) / masses.view(-1, 1))
            vel = vel * friction_half + noise_std * torch.randn_like(vel)
            vel = vel + 0.5 * accel * md_dt
            pos = pos + vel * md_dt
            pos = pos.detach().requires_grad_(True)
            energy, forces = compute_energy_and_forces(pos)
            forces = forces.detach()
            max_force = forces.abs().max().item()
            min_dist = min_pairwise_distance(pos)
            trace_diag.append((step, min_dist, max_force))

            if max_force > force_threshold:
                unstable = True
                break

            accel = forces * CONV / masses.view(-1, 1)
            vel = vel + 0.5 * accel * md_dt
            vel = vel * friction_half + noise_std * torch.randn_like(vel)
            kinetic = 0.5 * (masses * (vel ** 2).sum(dim=1)).sum() / CONV
            trajectory_pot.append(energy.item())
            trajectory_kin.append(kinetic.item())

        total += 1
        if not unstable:
            stable_count += 1
            total_arr = np.array(trajectory_pot) + np.array([0.0] + trajectory_kin)
            if len(total_arr) > 1:
                coeffs = np.polyfit(np.arange(len(total_arr)), total_arr, 1)
                energy_drifts.append(coeffs[0])

        results_list.append({
            "mol_idx": idx, "stable": not unstable, "n_atoms": n_atoms,
            "min_dist_initial": min_dist_initial, "steps_completed": step+1 if unstable else md_steps,
        })

    stable_frac = stable_count / total if total > 0 else 0
    mean_drift = float(np.mean(energy_drifts)) if energy_drifts else 0
    mean_abs_drift = float(np.mean(np.abs(energy_drifts))) if energy_drifts else 0

    print(f"  {name}:")
    print(f"    Stable: {stable_count}/{total} ({stable_frac:.1%})")
    print(f"    Mean drift: {mean_drift:.6f} eV/step")
    print(f"    Mean |drift|: {mean_abs_drift:.6f} eV/step")

    return {
        "stable_count": stable_count, "total": total,
        "stable_fraction": stable_frac,
        "mean_drift": mean_drift, "mean_abs_drift": mean_abs_drift,
        "details": results_list,
    }

results_part2 = {}

print("\n--- A: ZBL ON ---")
results_part2["zbl_on"] = run_md(vacuum_model, gas_eval_dataset, use_zbl=True, name="ZBL ON")

print("\n--- B: ZBL OFF ---")
results_part2["zbl_off"] = run_md(vacuum_model, gas_eval_dataset, use_zbl=False, name="ZBL OFF")

# ============================================================
# PART 3: Combined Results Table
# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)

# Baseline comparison
print("\n--- BASELINE: Option A vs Option B vs Vacuum (AQM-sol force MAE) ---")
print(f"  {'Approach':<40} {'Force MAE':>12} {'Force RMSE':>12} {'Low-E MAE':>12} {'High-E MAE':>12}")
print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for key, label in [("option_a", "Option A (scratch, direct)"),
                    ("option_b", "Option B (delta: vacuum + correction)"),
                    ("vacuum_only", "Vacuum only (no solvation)")]:
    r = results_part1[key]
    print(f"  {label:<40} {r['mae']:>12.6f} {r['rmse']:>12.6f} {r['low_mae']:>12.6f} {r['high_mae']:>12.6f}")

# Training loss comparison
print("\n--- TRAINING VAL LOSS (from training logs) ---")
print(f"  {'Approach':<40} {'Val Loss':>12}")
print(f"  {'-'*40} {'-'*12}")
print(f"  {'Option A (scratch, 3 blocks)':<40} {1.945:>12.3f}")
print(f"  {'Option B (vacuum 4B + correction 3B)':<40} {1.282:>12.3f}")
print(f"  {'Stage 1 (vacuum 4B, on AQM-gas)':<40} {1.599:>12.3f}")
print(f"  {'Stage 2b (explicit water)':<40} {27.536:>12.3f}")
print(f"    → Delta-learning (Opt B) beats scratch (Opt A) by {((1.945-1.282)/1.282)*100:.1f}%")

# A/B comparison
print("\n--- A/B: ZBL ON vs ZBL OFF (vacuum MD on AQM-gas) ---")
print(f"  {'Setting':<40} {'Stable':>12} {'Frac':>10} {'Drift':>14} {'|Drift|':>14}")
print(f"  {'-'*40} {'-'*12} {'-'*10} {'-'*14} {'-'*14}")
for key, label in [("zbl_on", "ZBL ON (0.5/1.0 A cutoff)"),
                    ("zbl_off", "ZBL OFF")]:
    r = results_part2[key]
    print(f"  {label:<40} {r['stable_count']:>4}/{r['total']:<5} {r['stable_fraction']:>9.1%} {r['mean_drift']:>+14.6f} {r['mean_abs_drift']:>14.6f}")

# Parameter counts
print("\n--- MODEL PARAMETERS ---")
print(f"  {'Model':<40} {'Params':>12}")
print(f"  {'-'*40} {'-'*12}")
for name, ckpt in [("Vacuum (stage1, 4 blocks)", "stage1_fold_1.pt"),
                    ("Correction (stage2a, 3 blocks)", "stage2_correction.pt"),
                    ("Option A (scratch, 3 blocks)", "option_a.pt"),
                    ("Explicit (stage2b, 2 blocks)", "stage2b.pt")]:
    path = os.path.join(RESULTS_DIR, ckpt)
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        n = sum(p.numel() for p in sd.values())
        print(f"  {name:<40} {n:>12,}")

# Save all results
all_results = {"force_mae_comparison": results_part1, "zbl_ab_comparison": results_part2}
class CompactEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

out_path = os.path.join(OUTPUT_DIR, "ab_comparison_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, cls=CompactEncoder)
print(f"\nSaved results to {out_path}")
print("=" * 70)

import json
import torch
from torch_geometric.utils import scatter


def fit_atomic_references(dataset, element_to_idx, num_elements):
    n = len(dataset)
    A = torch.zeros(n, num_elements)
    b = torch.zeros(n)

    for i in range(n):
        d = dataset[i]
        for z in d.z:
            A[i, element_to_idx[z.item()]] += 1
        b[i] = d.y_energy.item()

    # Only fit elements that actually appear — avoids rank deficiency from all-zero columns
    present_mask = (A.sum(dim=0) > 0)
    A_present = A[:, present_mask]

    # Ridge regression for numerical stability: (A^T A + λI) x = A^T b
    lambda_reg = 1e-6
    AtA = A_present.T @ A_present + lambda_reg * torch.eye(A_present.shape[1])
    Atb = A_present.T @ b
    ref_present = torch.linalg.solve(AtA, Atb)

    ref_energies = torch.zeros(num_elements)
    ref_energies[present_mask] = ref_present

    residuals = b - (A @ ref_energies)
    rmse = residuals.pow(2).mean().sqrt().item()
    raw_std = b.std().item()
    residual_std = residuals.std().item()
    print(f"  Atomic reference fit: RMSE={rmse:.4f}, raw E std={raw_std:.4f}, residual std={residual_std:.4f}")
    for z, idx in sorted(element_to_idx.items(), key=lambda kv: kv[1]):
        print(f"    Z={z:3d} (idx={idx}): ref_energy = {ref_energies[idx]:.4f}")

    return ref_energies


def save_reference_energies(ref_energies, element_to_idx, path):
    ref_dict = {
        str(atomic_num): float(ref_energies[idx].item())
        for atomic_num, idx in element_to_idx.items()
    }
    payload = {
        "element_to_idx": {str(k): v for k, v in element_to_idx.items()},
        "reference_energies": ref_dict,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved atomic reference energies to {path}")


def load_reference_energies(path, element_to_idx, num_elements, device):
    with open(path, "r") as f:
        payload = json.load(f)
    ref_energies = torch.zeros(num_elements, device=device)
    for atomic_num_str, energy in payload["reference_energies"].items():
        atomic_num = int(atomic_num_str)
        idx = element_to_idx.get(atomic_num, None)
        if idx is not None:
            ref_energies[idx] = energy
    return ref_energies


def compute_molecular_reference(x_one_hot, batch, ref_energies, num_graphs):
    atom_ref = x_one_hot @ ref_energies  # [n_atoms]
    if batch is None:
        return atom_ref.sum().unsqueeze(0)  # [1]
    return scatter(atom_ref, batch, dim=0, dim_size=num_graphs)  # [num_graphs]

import numpy as np
import torch
from mace.calculators.foundations_models import mace_off
from torch.utils.data import DataLoader

from data import collate_mace


def load_mace_foundation(model_size="medium", device="cpu"):
    return mace_off(model_size, return_raw_model=True, device=device)


def fit_atomic_references(dataset):
    from data import MACE_NUM_ELEMENTS
    n = len(dataset)
    A = torch.zeros(n, MACE_NUM_ELEMENTS)
    b = torch.zeros(n)
    for i in range(n):
        node_attrs = dataset._get_node_attrs(i)
        A[i] = node_attrs.sum(dim=0)
        b[i] = dataset.samples[i][1]
    present = A.sum(dim=0) > 0
    A_p = A[:, present]
    lam = 1e-6
    AtA = A_p.T @ A_p + lam * torch.eye(A_p.shape[1])
    Atb = A_p.T @ b
    ref = torch.linalg.solve(AtA, Atb)
    ref_energies = torch.zeros(MACE_NUM_ELEMENTS)
    ref_energies[present] = ref
    residual_std = (b - A @ ref_energies).std().item()
    print(f"Atomic reference fit: residual std = {residual_std:.4f} kcal/mol")
    return ref_energies


def calibrate_output(model, dataset, batch_size=64, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mace, num_workers=0)
    model.eval()
    all_energies = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch, compute_force=False)
            all_energies.append(out["energy"].cpu())
    all_energies = torch.cat(all_energies)
    mean_e = all_energies.mean().item()
    std_e = all_energies.std().item()
    print(f"Model output: mean={mean_e:.4f} eV, std={std_e:.4f} eV")
    print(f"  = {mean_e*23.0605:.1f} kcal/mol mean, {std_e*23.0605:.1f} kcal/mol std")
    return mean_e, std_e


class MACEFreeSolv(torch.nn.Module):
    def __init__(self, model_size="medium", device="cpu", fit_refs=True,
                 freeze_atomic_energies=False, target_mean=0.0, target_std=None):
        super().__init__()
        self.device = device
        self.model_size = model_size
        self.freeze_atomic_energies = freeze_atomic_energies
        self.target_mean = target_mean
        self.target_std = target_std

        base = load_mace_foundation(model_size, device)
        base = base.float()
        for p in base.parameters():
            p.requires_grad_(True)

        if fit_refs:
            from data import MACEFreeSolvDataset
            ds = MACEFreeSolvDataset(targets_in_ev=False)
            ref_kcal = fit_atomic_references(ds)
            ref_ev = ref_kcal / 23.0605
            base.atomic_energies_fn.atomic_energies.data = ref_ev.unsqueeze(0).float()
            base.atomic_energies_fn.atomic_energies.requires_grad_(not self.freeze_atomic_energies)

            mean_e, std_e = calibrate_output(base, ds, batch_size=64, device=device)

            if target_std is None:
                target_std_ev = 1.0 / 23.0605
            else:
                target_std_ev = target_std / 23.0605
            scale_analytic = target_std_ev / max(std_e, 1e-8)
            scale = min(0.001, scale_analytic)
            shift = 0.0
            base.scale_shift.scale.data.fill_(scale)
            base.scale_shift.shift.data.fill_(shift)
            print(f"  Calibration: target_std={target_std_ev:.4f} eV, model_std={std_e:.4f} eV")
            print(f"    analytic_scale={scale_analytic:.6f}, capped_scale={scale:.6f}")
            print(f"    shift=0.0 (per-node shift must be 0 — summed over atoms)")

        self.model = base
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"MACEFreeSolv: {n_trainable:,}/{n_total:,} trainable params")

    def forward(self, data, compute_force=False):
        out = self.model(data, compute_force=compute_force, training=self.training)
        return out["energy"]

    def freeze_interactions(self):
        for name, p in self.model.named_parameters():
            if "interactions" in name or "products" in name or "readouts" in name:
                p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Frozen interactions: {n_trainable:,} trainable params remaining")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, strict=True):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=strict)
        print(f"Loaded checkpoint: {path}")

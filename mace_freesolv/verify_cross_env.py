"""Cross-environment checkpoint equivalence check (Stage-A producer vs Stage-B consumer).

The remote crash: a Stage-A checkpoint produced by MACEFreeSolvScratch carries 9
`weights_*_zeroed` buffers (registered by the mace 0.3.16 ScaleShiftMACE constructor),
but Stage-B loads into MACEFreeSolv (plain mace_off / checkpoint-era serialization,
0 such buffers) -> strict load_state_dict fails on "Unexpected key(s)".

This script reproduces BOTH environments locally (same mace version, 0.3.16) and
proves the tolerant loader reproduces, exactly, the predictions of the
producing environment on the same inputs:

  env A (producer): MACEFreeSolvScratch  constructed-from-constructor = has the buffers
  env B (consumer): MACEFreeSolv          plain loader = lacks the buffers

Run (repo root):
  python mace_freesolv/verify_cross_env.py <stage_a.pt> [n_batches]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from aqm_data import AQMMACEDataset
from data import collate_mace
from model import MACEFreeSolv
from scratch_model import MACEFreeSolvScratch

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"scratch\smoke_scratch_run\stage_a.pt"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SOL = r"scratch\aqm_sol_slice.hdf5"
GAS = r"scratch\aqm_gas_slice.hdf5"

val_ds = AQMMACEDataset(SOL, GAS, r_max=5.0, max_neighbors=32)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_mace, num_workers=0)


def preds(model, loader, n_batches=N_BATCHES):
    model.eval()
    out = []
    with torch.no_grad():
        for i, (batch) in enumerate(loader):
            if i >= n_batches:
                break
            batch = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in batch.items()}
            out.append(model(batch, compute_force=False))
    return torch.cat(out)


state = torch.load(CKPT, map_location="cpu", weights_only=True)
n_zeroed = sum(1 for k in state if k.endswith("_zeroed"))
values = {k: bool(v.item()) for k, v in state.items() if k.endswith("_zeroed")}
print(f"checkpoint: {CKPT}")
print(f"  total keys={len(state)}   *_zeroed buffers={n_zeroed}")
if values:
    print(f"  zeroed-flag values: True x{sum(values.values())} / False x{len(values) - sum(values.values())}")

print("\n--- env A (producer: MACEFreeSolvScratch, constructor-registered buffers) ---")
model_a = MACEFreeSolvScratch(init_checkpoint=CKPT, device="cpu").to("cpu")
p_a = preds(model_a, val_loader)

print("\n--- env B (consumer: MACEFreeSolv, plain loader + tolerant strip) ---")
model_b = MACEFreeSolv(init_checkpoint=CKPT, device="cpu").to("cpu")
p_b = preds(model_b, val_loader)

diff = (p_a - p_b).abs().max().item()
mean = (p_a - p_b).abs().mean().item()
print(f"\nCROSS-ENV max |Î”pred| = {diff:.6e} eV | mean |Î”pred| = {mean:.6e} eV")
print(f"  env A pred mean/std: {p_a.mean().item():.4f}/{p_a.std().item():.4f} eV")
print(f"  env B pred mean/std: {p_b.mean().item():.4f}/{p_b.std().item():.4f} eV")
print(f"  in kcal/mol: max |Î”pred| = {diff * 23.0605:.6e}")
assert diff < 1e-4, f"CROSS-ENV MISMATCH: max |Î”pred| = {diff:.6e} eV exceeds 1e-4"
print("\nCROSS-ENV OK: tolerant loader reproduces producer predictions")
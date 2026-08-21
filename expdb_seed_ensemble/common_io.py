"""Shared IO + model builders for the Exp-DB seed-ensemble GIMS bundle.

Read-only reuse of the existing expdb_vast assets. Nothing here modifies any
archived file. All paths resolve in order:
  1. ./inputs/            (self-contained copies, gathered by preflight.py)
  2. ../expdb_vast/...    (original bundle, if running from a full repo clone)
"""

import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_INPUTS = os.path.join(HERE, "inputs")
VAST = os.path.abspath(os.path.join(HERE, "..", "expdb_vast"))

SEEDS_ALL = [42, 123, 7, 2024, 999]
SEEDS_PRIMARY = [42, 123, 999]
EV_TO_KCAL = 23.0605
TAU_STAR = 4.725394227550238e-04   # archived FreeSolv-calibrated tau2*
N_BOOT = 10_000
RNG_SEED = 20260815


def _find(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def require(name, *candidates):
    p = _find(*candidates)
    if p is None:
        raise FileNotFoundError(
            f"[preflight] MISSING required file: {name}\n"
            f"  looked in:\n" + "\n".join(f"    {c}" for c in candidates if c) +
            "\n  Fix: python preflight.py --gather   (on a machine that has "
            "expdb_vast/), commit inputs/, and pull on Vast.")
    return p


def path_stage2():
    return require("stage2_correction.pt",
                   os.path.join(BUNDLE_INPUTS, "stage2_correction.pt"),
                   os.path.join(VAST, "stage2_correction.pt"))

def path_freesolv_h5():
    return require("freesolv_conformers.hdf5",
                   os.path.join(BUNDLE_INPUTS, "freesolv_conformers.hdf5"),
                   os.path.join(VAST, "freesolv_conformers.hdf5"))

def path_expdb_h5():
    return require("expdb_conformers.hdf5",
                   os.path.join(BUNDLE_INPUTS, "expdb_conformers.hdf5"),
                   os.path.join(VAST, "results", "expdb_conformers.hdf5"))

def path_split(which):
    return require(f"split_check/{which}_ids.json",
                   os.path.join(BUNDLE_INPUTS, "split_check", f"{which}_ids.json"),
                   os.path.join(VAST, "results", "split_check", f"{which}_ids.json"))

def path_truth_csv():
    return require("predictions_ensemble.csv (truth labels)",
                   os.path.join(BUNDLE_INPUTS, "predictions_ensemble.csv"),
                   os.path.join(VAST, "results", "predictions_ensemble.csv"))

def path_labels_json():
    return require("freesolv_cache/database.json",
                   os.path.join(BUNDLE_INPUTS, "database.json"),
                   os.path.join(VAST, "freesolv_cache", "database.json"))


def add_vast_to_path():
    """Import DimeModels/element_vocab/freesolv_dataset from expdb_vast (read-only)."""
    src = _find(os.path.join(VAST), HERE)
    for cand in (VAST, BUNDLE_INPUTS):
        if os.path.exists(os.path.join(cand, "DimeModels.py")):
            src = cand
            break
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def one_hot_x(z_tensor, device):
    """One-hot atom features via ELEMENT_TO_IDX (raw Z values are NOT indices;
    e.g. Br Z=35, I Z=53 map to columns 15/16 of a 17-column vocab)."""
    import torch
    from element_vocab import ELEMENT_TO_IDX, NUM_ELEMENTS
    x = torch.zeros(z_tensor.shape[0], NUM_ELEMENTS, device=device)
    idx = torch.tensor([ELEMENT_TO_IDX[int(zz)] for zz in z_tensor],
                       dtype=torch.long, device=device)
    x[torch.arange(z_tensor.shape[0], device=device), idx] = 1.0
    return x


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_labels():
    sys.path.insert(0, add_vast_to_path())
    from freesolv_dataset import load_freesolv_labels
    return load_freesolv_labels(path_labels_json())


def load_truth():
    """Exp-DB id -> expt ΔG (kcal/mol) from the archived ensemble csv."""
    import pandas as pd
    df = pd.read_csv(path_truth_csv())
    return dict(zip(df["id"], df["dg_exp_kcal"].astype(float)))


def build_model(device, hidden=128, blocks=3):
    sys.path.insert(0, add_vast_to_path())
    from DimeModels import DimeNetPlusSE
    from element_vocab import NUM_ELEMENTS
    return DimeNetPlusSE(
        in_channels=NUM_ELEMENTS,
        hidden_channels=hidden,
        out_channels=1,
        num_blocks=blocks,
        int_emb_size=min(64, hidden // 2),
        basis_emb_size=8,
        out_emb_channels=min(256, hidden * 2),
        num_spherical=7,
        num_radial=6,
        cutoff=6.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        is_energy=True,
        use_multi_aggregate=False,
        use_se=False,
    ).to(device)


def load_seed_model(device, seed, weights_dir):
    import torch
    m = build_model(device)
    ck = os.path.join(weights_dir, f"finetuned_seed{seed}.pt")
    state = torch.load(ck, map_location=device, weights_only=True)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"    [warn] seed{seed}: {len(missing)} randomly-init params")
    m.eval()
    return m


def per_atom_predict(model, z, pos, device):
    """Per-atom contributions (kcal/mol) for ONE conformer.

    Trick: DimeNetPlusSE.forward returns scatter(P, batch) when is_energy=True
    and raw P when is_energy=False. Flip the attribute only (no weight change),
    predict, flip back.
    """
    import torch
    from torch_geometric.data import Data
    with torch.no_grad():
        data = Data(z=torch.tensor(z, dtype=torch.long),
                    pos=torch.tensor(pos, dtype=torch.float)).to(device)
        x = one_hot_x(data.z, device)
        was = model.is_energy
        model.is_energy = False
        P = model(x, data.pos, None)            # (n_atoms,) in eV
        model.is_energy = was
    return P.detach().cpu().numpy() * EV_TO_KCAL


def energy_predict(model, z, pos, device):
    """Scalar molecular energy (kcal/mol), the pipeline's normal path."""
    import torch
    from torch_geometric.data import Data
    with torch.no_grad():
        data = Data(z=torch.tensor(z, dtype=torch.long),
                    pos=torch.tensor(pos, dtype=torch.float)).to(device)
        x = one_hot_x(data.z, device)
        e = model(x, data.pos, None)
    return float(e.item()) * EV_TO_KCAL

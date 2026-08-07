"""Shared helpers for the Frag20 FROM-SCRATCH pretraining experiment.

EVERYTHING here is a COPY of code from the verified pipeline
(train_stage1_vacuum.py / train_stage2_correction.py / cv_finetune.py /
element_vocab.py / energy_reference.py), adapted only where the experiment
demands it (Frag20 hdf5 layout, energy-only targets, no forces, dataset's own
fixed 80K/10K/10K split). Nothing here imports from the original files - this
folder must remain fully self-contained (delete the folder = complete rollback).
"""

import hashlib
import json
import os
import random

import numpy as np

EV_TO_KCAL = 23.0605
DEFAULT_SEED = 42

# Frozen fold-0 FreeSolv split from the VERIFIED full run. This sandbox lives
# at .../aqm-spice2/freesolv/experimental_frag20_scratch/common_scratch.py so
# 4x dirname lands on the repo root (.../Data).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_SPLIT_DIR = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")

DEFAULT_FREESOLV_CONFORMERS = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
DEFAULT_FREESOLV_LABELS = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FRAG20_H5 = os.path.join(HERE, "data", "frag20_full.hdf5")
DEFAULT_FRAG20_LABELS = os.path.join(HERE, "data", "frag20_full_labels.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(blob):
    return hashlib.md5(blob).hexdigest()


def evaluate(preds, expts):
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    r2 = float(1 - np.sum((preds - expts) ** 2) / np.sum((expts - expts.mean()) ** 2))
    return mae, rmse, r2


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_split(split_dir, all_labels):
    """Load the frozen fold-0 *_ids.json files. Verifies disjointness and that
    every id has a numeric label. Returns (train_ids, val_ids, test_ids)."""
    with open(os.path.join(split_dir, "train_ids.json")) as f:
        train_ids = json.load(f)
    with open(os.path.join(split_dir, "val_ids.json")) as f:
        val_ids = json.load(f)
    with open(os.path.join(split_dir, "test_ids.json")) as f:
        test_ids = json.load(f)

    assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap in frozen split"
    assert len(set(train_ids) & set(test_ids)) == 0, "train/test overlap in frozen split"
    assert len(set(val_ids) & set(test_ids)) == 0, "val/test overlap in frozen split"

    missing = [m for m in train_ids + val_ids + test_ids
               if m not in all_labels or not isinstance(all_labels[m].get("expt"), (int, float))]
    assert not missing, f"{len(missing)} split ids lack numeric expt labels: {missing[:5]}"

    return train_ids, val_ids, test_ids


def load_freesolv_labels(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_frag20_labels(path):
    """Read the frag20_full_labels.json -> {mol_id: {split, calc_sol_kcal,
    smiles, gas_eV, wat_eV}}."""
    with open(path) as f:
        return json.load(f)


class Frag20Dataset:
    """Dataset over the frag20_full.hdf5 built by prepare_frag20_scratch.py.

    Each item is a torch_geometric Data with:
      z, pos, mol_id, y_gas_eV (gas electronic energy), y_wat_eV (SMD water
      electronic energy), y_dG_eV = wat_eV - gas_eV, y_calc_sol_kcal.

    Energy targets are stored as tensors so the same dataset serves both
    stage 1 (gas) and stage 2 (dG) pretraining.
    """

    def __init__(self, h5_path, ids, labels):
        self.h5_path = h5_path
        self.ids = ids
        self.labels = labels
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def _load(self, mid):
        import h5py
        import torch
        from torch_geometric.data import Data
        with h5py.File(self.h5_path, "r") as f:
            g = f[mid]
            return Data(
                z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                mol_id=mid,
                y_gas_eV=torch.tensor([self.labels[mid]["gas_eV"]], dtype=torch.float),
                y_wat_eV=torch.tensor([self.labels[mid]["wat_eV"]], dtype=torch.float),
            )

    def __getitem__(self, idx):
        import torch
        mid = self.ids[idx]
        if mid not in self._cache:
            self._cache[mid] = self._load(mid).clone()
        data = self._cache[mid].clone()
        data.y_dG_eV = data.y_wat_eV - data.y_gas_eV
        data.y_calc_sol_kcal = torch.tensor(
            [self.labels[mid]["calc_sol_kcal"]], dtype=torch.float)
        return data


class CachedListDataset:
    """Wraps a pre-built list of torch_geometric Data objects so each molecule
    is materialized exactly once (same pattern as the verified pipeline's
    _CachedListDataset)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_model(device, num_blocks=4):
    from DimeModels import DimeNetPlus
    from element_vocab import NUM_ELEMENTS
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=128, out_channels=1,
        num_blocks=num_blocks, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    ).to(device)


def fit_atomic_references(dataset, element_to_idx, num_elements, energy_attr="y_gas_eV"):
    """Ridge-fit atomic reference energies on the TRAIN split only (mirrors
    energy_reference.fit_atomic_references, adapted for Frag20 target attrs)."""
    import torch
    n = len(dataset)
    A = torch.zeros(n, num_elements)
    b = torch.zeros(n)

    for i in range(n):
        d = dataset[i]
        for z in d.z:
            A[i, element_to_idx[z.item()]] += 1
        b[i] = getattr(d, energy_attr).item()

    present_mask = (A.sum(dim=0) > 0)
    A_present = A[:, present_mask]
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
    print(f"  Atomic reference fit ({energy_attr}): RMSE={rmse:.4f} eV, "
          f"raw E std={raw_std:.4f}, residual std={residual_std:.4f}")
    return ref_energies


def compute_molecular_reference(x_one_hot, batch, ref_energies, num_graphs):
    import torch
    from torch_geometric.utils import scatter
    atom_ref = x_one_hot @ ref_energies
    if batch is None:
        return atom_ref.sum().unsqueeze(0)
    return scatter(atom_ref, batch, dim=0, dim_size=num_graphs)


def conformer_average(model, device, test_ids, all_labels, conformers,
                      n_conformers, batch_size):
    """RDKit ETKDGv3 conformer test-time averaging - identical protocol to
    cv_finetune.py / deep_ensemble.py. Falls back to the stored hdf5 conformer
    when rdkit is unavailable."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot

    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
        rdkit_ok = True
    except ImportError:
        Chem = rdDistGeom = rdForceFieldHelpers = None
        rdkit_ok = False

    def _gen_confs(smiles, n):
        if not rdkit_ok:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.5
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
        if not conf_ids:
            return None
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        try:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
        except Exception:
            pass
        z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
                         dtype=torch.long)
        n_avail = min(n, mol.GetNumConformers())
        return [Data(z=z.clone(),
                     pos=torch.tensor(np.array(mol.GetConformer(i).GetPositions(),
                                               dtype=np.float64), dtype=torch.float))
                for i in range(n_avail)]

    flat_data, flat_mid = [], []
    hdf5_cache = {}
    import h5py
    for mid in test_ids:
        confs = _gen_confs(all_labels[mid]["smiles"], n_conformers)
        if confs is None:
            if mid not in hdf5_cache:
                with h5py.File(conformers, "r") as f:
                    g = f[mid]
                    hdf5_cache[mid] = Data(
                        z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                        pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                    )
            confs = [hdf5_cache[mid].clone() for _ in range(n_conformers)]
        for cd in confs:
            flat_data.append(cd)
            flat_mid.append(mid)

    loader = DataLoader(flat_data, batch_size=batch_size * 4, shuffle=False)
    all_raw = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            preds = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
            all_raw.append(preds.cpu())
    all_raw = torch.cat(all_raw).numpy()

    conf_preds = {}
    for mid, val in zip(flat_mid, all_raw):
        conf_preds.setdefault(mid, []).append(float(val))

    tta_preds_by_mid = {mid: float(np.mean(conf_preds[mid])) for mid in test_ids}
    preds = np.array([tta_preds_by_mid[mid] for mid in test_ids])
    expts = np.array([all_labels[mid]["expt"] for mid in test_ids])
    mae, rmse, _ = evaluate(preds, expts)
    if not rdkit_ok:
        print(f"  WARNING: rdkit unavailable - TTA fell back to the stored conformer "
              f"({n_conformers} clones of it; mean unchanged)")
    return mae, rmse, tta_preds_by_mid

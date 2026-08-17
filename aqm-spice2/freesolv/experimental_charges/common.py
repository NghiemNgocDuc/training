"""Shared helpers for the partial-charges fine-tuning experiment.

EVERYTHING in this file is a COPY of code from the verified pipeline
(experimental_frag20/common.py / finetune_freesolv.py / element_vocab.py),
verbatim where possible, plus the additions needed for the charges variant.
Nothing here imports from the original files - this folder must remain fully
self-contained (delete the folder = complete rollback).
"""

import hashlib
import json
import os
import random

import numpy as np

EV_TO_KCAL = 23.0605
DEFAULT_SEED = 42

# common.py lives at .../aqm-spice2/freesolv/experimental_charges/common.py
# so 4x dirname lands on the repo root (.../Data).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DEFAULT_SPLIT_DIR = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")

# Stage-2 correction checkpoint the verified fold-0 run started from.
DEFAULT_CORRECTION_CKPT = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline", "results_full", "stage2_correction.pt")

DEFAULT_FREESOLV_CONFORMERS = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
DEFAULT_FREESOLV_LABELS = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")
DEFAULT_CHARGES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "freesolv_charges.json")


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


def load_charges(json_path):
    """{mol_id: [q per atom]} cache produced by prepare_charges.py."""
    with open(json_path, "r") as f:
        return json.load(f)


def dataset_cls(hdf5_path, labels, charges):
    """Dataset over a single hdf5 file whose groups are keyed by mol_id, with
    numeric labels from a dict {mol_id: {...}} (the deep_ensemble.py pattern).
    If `charges` is given, attaches data.charges (Gasteiger, cached)."""
    import torch
    from torch_geometric.data import Data

    class SimpleDataset:
        def __init__(self, ids):
            self.ids = ids
            self.hdf5_path = hdf5_path
            self.labels = labels
            self.charges = charges
            self._cache = {}

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            mid = self.ids[idx]
            if mid not in self._cache:
                import h5py
                with h5py.File(self.hdf5_path, "r") as f:
                    g = f[mid]
                    d = Data(
                        z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                        pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                    )
                if self.charges is not None and mid in self.charges:
                    d.charges = torch.tensor(self.charges[mid], dtype=torch.float)
                self._cache[mid] = d.clone()
            data = self._cache[mid].clone()
            data.mol_id = mid
            data.y_dG = torch.tensor([self.labels[mid]["expt"]], dtype=torch.float)
            return data

    return SimpleDataset


def build_model(device, use_charges):
    from DimeModels import DimeNetPlus
    from element_vocab import NUM_ELEMENTS
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS + (1 if use_charges else 0),
        hidden_channels=128, out_channels=1,
        num_blocks=3, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    ).to(device)


def conformer_average(model, device, test_ids, all_labels, conformers, n_conformers,
                      batch_size, charges=None):
    """RDKit ETKDGv3 conformer test-time averaging - identical to the protocol
    in cv_finetune.py / deep_ensemble.py. Falls back to the stored hdf5
    conformer when rdkit is unavailable (e.g. AppLocker-blocked locally).
    `charges` (Gasteiger cache) are attached per molecule when provided -
    order matches SMILES+AddHs, verified by prepare_charges.py."""
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
            if charges is not None and mid in charges:
                cd.charges = torch.tensor(charges[mid], dtype=torch.float)
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
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data import ELEMENT_TO_IDX, MACE_NUM_ELEMENTS, radius_graph

ENERGY_KEY = "ePBE0+MBD"
EV_TO_KCAL = 23.0605


class AQMMACEDataset(Dataset):
    """MACE-ready dataset over AQM sol/gas pairs.

    Target: dG = E_sol - E_gas (eV) per paired conformer, read from the
    'ePBE0+MBD' fields of matching mol_id/conf keys in the sol and gas files.

    NOTE: the AQM 'eSOLV' field is NOT the solvation free energy — it sits on
    the total-energy scale (eSOLV ~= E_sol + ~0.08 eV). The old Stage-2a
    correction model was trained on (eSOLV - E_gas) residuals, which is why the
    zero-shot FreeSolv MAE was ~27 kcal/mol. Use this dataset's target instead.

    Items are built lazily with a bounded LRU cache (~60k conformers is too
    many to precompute like MACEFreeSolvDataset). Use num_workers=0.
    """

    def __init__(self, hdf5_sol, hdf5_gas, r_max=5.0, max_neighbors=32,
                 max_samples=None, cache_size=4096, element_subset=None,
                 mol_ids=None):
        self.hdf5_sol = hdf5_sol
        self.hdf5_gas = hdf5_gas
        self.r_max = r_max
        self.max_neighbors = max_neighbors
        self.cache_size = cache_size
        self.label_units = "eV"  # samples[] targets (dG = E_sol - E_gas in eV)
        if element_subset is None:
            element_subset = set(ELEMENT_TO_IDX.keys())
        self.element_subset = set(element_subset)
        self.samples = []

        self._fsol = None
        self._fgas = None

        with h5py.File(hdf5_sol, "r") as fsol, h5py.File(hdf5_gas, "r") as fgas:
            gas_ids = set(fgas.keys())
            sol_ids = list(fsol.keys())
            if mol_ids is not None:
                sol_ids = [m for m in sol_ids if m in mol_ids]
            for mol_id in sol_ids:
                if mol_id not in gas_ids:
                    continue
                gas_keys = set(fgas[mol_id].keys())
                for conf_id in fsol[mol_id].keys():
                    if conf_id not in gas_keys:
                        continue
                    if ENERGY_KEY not in fsol[mol_id][conf_id] or ENERGY_KEY not in fgas[mol_id][conf_id]:
                        continue
                    e_sol = float(np.asarray(fsol[mol_id][conf_id][ENERGY_KEY][...]).reshape(-1)[0])
                    e_gas = float(np.asarray(fgas[mol_id][conf_id][ENERGY_KEY][...]).reshape(-1)[0])
                    if not (np.isfinite(e_sol) and np.isfinite(e_gas)):
                        continue
                    dG_ev = e_sol - e_gas
                    if dG_ev < -50.0 or dG_ev > 50.0:
                        continue
                    z = np.asarray(fsol[mol_id][conf_id]["atNUM"][...]).astype(int)
                    if not set(np.unique(z)).issubset(self.element_subset):
                        continue
                    self.samples.append((mol_id, conf_id, dG_ev))

        if max_samples is not None and 0 < max_samples < len(self.samples):
            step = len(self.samples) // max_samples
            self.samples = self.samples[::step][:max_samples]

        self._cache = OrderedDict()
        print(f"AQMMACEDataset: {len(self.samples)} paired conformers "
              f"(sol={os.path.basename(hdf5_sol)}, gas={os.path.basename(hdf5_gas)})")

    def _get_file(self, path):
        if path == self.hdf5_sol:
            if self._fsol is None:
                self._fsol = h5py.File(path, "r")
            return self._fsol
        if self._fgas is None:
            self._fgas = h5py.File(path, "r")
        return self._fgas

    def _read(self, path, mol_id, conf_id, field):
        f = self._get_file(path)
        return torch.tensor(f[mol_id][conf_id][field][...])

    def _read_z(self, mol_id, conf_id):
        z = self._read(self.hdf5_sol, mol_id, conf_id, "atNUM")
        return z.long()

    def _build_item(self, idx):
        mol_id, conf_id, dG_ev = self.samples[idx]
        z = self._read_z(mol_id, conf_id)
        pos = self._read(self.hdf5_sol, mol_id, conf_id, "atXYZ").float()

        n_atoms = z.size(0)
        node_attrs = torch.zeros(n_atoms, MACE_NUM_ELEMENTS, dtype=torch.float32)
        for i, zi in enumerate(z):
            idx_e = ELEMENT_TO_IDX.get(zi.item())
            if idx_e is None:
                raise ValueError(
                    f"Unknown element Z={zi.item()} in {mol_id}/{conf_id} — not in "
                    f"MACE vocab {sorted(ELEMENT_TO_IDX.keys())}")
            node_attrs[i, idx_e] = 1.0

        edge_index = radius_graph(pos, r=self.r_max, max_num_neighbors=self.max_neighbors)
        n_edges = edge_index.size(1)

        return {
            "positions": pos,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "num_nodes": n_atoms,
            "num_edges": n_edges,
            "y": torch.tensor([dG_ev], dtype=torch.float32),
        }

    def _get_node_attrs(self, idx):
        item = self._build_item(idx)
        self._cache_store(idx, item)
        return item["node_attrs"]

    def _cache_store(self, idx, item):
        self._cache[idx] = item
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self._cache.get(idx)
        if item is None:
            item = self._build_item(idx)
            self._cache_store(idx, item)
        return item


def target_stats_kcal(dataset):
    """Mean/std of the dG target in kcal/mol (from eV samples)."""
    targets_ev = torch.tensor([s[2] for s in dataset.samples], dtype=torch.float32)
    return targets_ev.mean().item() * EV_TO_KCAL, targets_ev.std().item() * EV_TO_KCAL

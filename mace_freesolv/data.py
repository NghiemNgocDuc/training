import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import HDF5_PATH, EV_TO_KCAL, ROOT

ELEMENT_TO_IDX = {
    1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9,
}
ELEMENT_ATOMIC_NUMBERS = list(ELEMENT_TO_IDX.keys())
MACE_NUM_ELEMENTS = 10  # MACE-OFF23 was trained with 10 elements


def radius_graph(pos, r, batch=None, max_num_neighbors=32):
    if batch is None:
        batch = pos.new_zeros(pos.size(0), dtype=torch.long)
    rows, cols = [], []
    for b in range(batch.max().item() + 1):
        mask = batch == b
        p = pos[mask]
        dist = torch.cdist(p, p)
        adj = dist <= r
        adj.fill_diagonal_(False)
        i, j = adj.nonzero(as_tuple=True)
        if i.numel() > 0 and max_num_neighbors > 0:
            dist_vals = dist[i, j]
            order = torch.argsort(dist_vals, stable=True)
            i_sorted, j_sorted = i[order], j[order]
            uniq, counts = torch.unique(i_sorted, return_counts=True)
            keep = []
            start = 0
            for c in counts.tolist():
                end = start + c
                k = min(c, max_num_neighbors)
                keep.append(torch.ones(k, dtype=torch.bool))
                if c > k:
                    keep.append(torch.zeros(c - k, dtype=torch.bool))
                start = end
            keep = torch.cat(keep)
            i, j = i_sorted[keep], j_sorted[keep]
        rows.append(i)
        cols.append(j)
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(cols), torch.cat(rows)], dim=0)


def get_labels(cache_dir="Data/FreeSolv"):
    json_path = os.path.join(ROOT, cache_dir, "database.json")
    if not os.path.exists(json_path):
        _download_freesolv_data(json_path, os.path.join(ROOT, cache_dir))
    with open(json_path) as f:
        return json.load(f)


def _download_freesolv_data(json_path, cache_dir):
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json"
    print(f"Downloading FreeSolv database from {url}")
    urllib.request.urlretrieve(url, json_path)


class MACEFreeSolvDataset(Dataset):
    def __init__(self, hdf5_path=HDF5_PATH, r_max=5.0, max_neighbors=32,
                 targets_in_ev=True, mol_ids=None):
        self.r_max = r_max
        self.max_neighbors = max_neighbors
        self.targets_in_ev = targets_in_ev
        self.all_labels = get_labels()
        self.samples = []

        with h5py.File(hdf5_path, "r") as f:
            all_mol_ids = list(f.keys())

        if mol_ids is not None:
            all_mol_ids = [m for m in all_mol_ids if m in mol_ids]

        for mol_id in all_mol_ids:
            label = self.all_labels.get(mol_id, {})
            expt = label.get("expt")
            if not isinstance(expt, (int, float)):
                continue
            self.samples.append((mol_id, float(expt)))

        self._cached = [self._build_item(i) for i in range(len(self.samples))]
        print(f"MACEFreeSolvDataset: {len(self.samples)} samples (precomputed)")

    def _build_item(self, idx):
        mol_id, dG_kcal = self.samples[idx]
        with h5py.File(HDF5_PATH, "r") as f:
            grp = f[mol_id]
            z = torch.tensor(grp["atNUM"][...], dtype=torch.long)
            pos = torch.tensor(grp["atXYZ"][...], dtype=torch.float32)

        n_atoms = z.size(0)
        node_attrs = torch.zeros(n_atoms, MACE_NUM_ELEMENTS, dtype=torch.float32)
        for i, zi in enumerate(z):
            idx_e = ELEMENT_TO_IDX.get(zi.item())
            if idx_e is not None:
                node_attrs[i, idx_e] = 1.0

        edge_index = radius_graph(pos, r=self.r_max, max_num_neighbors=self.max_neighbors)
        n_edges = edge_index.size(1)

        target = dG_kcal / EV_TO_KCAL if self.targets_in_ev else dG_kcal

        return {
            "mol_id": mol_id,
            "positions": pos,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "num_nodes": n_atoms,
            "num_edges": n_edges,
            "y": torch.tensor([target], dtype=torch.float32),
        }

    def _get_node_attrs(self, idx):
        return self._cached[idx]["node_attrs"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._cached[idx]


def collate_mace(batch):
    batch_size = len(batch)
    positions_list = []
    node_attrs_list = []
    edge_index_list = []
    y_list = []
    batch_assignment = []
    ptr = [0]
    n_edges_cum = 0

    for i, item in enumerate(batch):
        n = item["num_nodes"]
        e = item["num_edges"]
        positions_list.append(item["positions"])
        node_attrs_list.append(item["node_attrs"])
        edge_index_list.append(item["edge_index"] + ptr[-1])
        y_list.append(item["y"])
        batch_assignment.append(torch.full((n,), i, dtype=torch.long))
        ptr.append(ptr[-1] + n)

    positions = torch.cat(positions_list, dim=0)
    node_attrs = torch.cat(node_attrs_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    batch = torch.cat(batch_assignment, dim=0)
    ptr = torch.tensor(ptr, dtype=torch.long)
    y = torch.cat(y_list, dim=0)
    total_edges = edge_index.size(1)

    cell_diag = positions.max(dim=0).values - positions.min(dim=0).values + 20.0
    cell = torch.diag(cell_diag).unsqueeze(0).expand(batch_size, -1, -1)

    data = {
        "positions": positions,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": cell,
        "edge_index": edge_index,
        "shifts": torch.zeros(total_edges, 3, dtype=torch.float32),
        "unit_shifts": torch.zeros(total_edges, 3, dtype=torch.float32),
        "y": y,
    }
    return data

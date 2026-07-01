import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple
from torch_geometric.data import InMemoryDataset, Data


class AQMDataset(InMemoryDataset):
    ATOM_TYPE_KEY = "atNUM"
    ATOM_COORDS_KEY = "atXYZ"
    ENERGY_KEY = "ePBE0+MBD"
    FORCES_KEY = "totFOR"
    ESOLV_KEY = "eSOLV"

    def __init__(
        self,
        root: str,
        hdf5_path: str,
        gas_hdf5_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        energy_key: str = ENERGY_KEY,
        forces_key: str = FORCES_KEY,
        atom_type_key: str = ATOM_TYPE_KEY,
        coords_key: str = ATOM_COORDS_KEY,
        max_structures: Optional[int] = None,
    ):
        self.hdf5_path = hdf5_path
        self.gas_hdf5_path = gas_hdf5_path
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.atom_type_key = atom_type_key
        self.coords_key = coords_key
        self.max_structures = max_structures
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        name = os.path.splitext(os.path.basename(self.hdf5_path))[0]
        suffix = ""
        if self.max_structures is not None:
            suffix = f"_{self.max_structures}"
        return [f"{name}{suffix}.pt"]

    @staticmethod
    def _read_conformer(src: h5py.Group) -> dict:
        return {
            "z": torch.tensor(src["atNUM"][:], dtype=torch.long).squeeze(),
            "pos": torch.tensor(src["atXYZ"][:], dtype=torch.float),
            "energy": torch.tensor(src["ePBE0+MBD"][:], dtype=torch.float).view(1),
            "forces": torch.tensor(src["totFOR"][:], dtype=torch.float),
        }

    def _load_hdf5(self, path: str) -> Dict[str, dict]:
        result = {}
        with h5py.File(path, "r") as f:
            for mol_id in f.keys():
                mol_grp = f[mol_id]
                for conf_key in mol_grp.keys():
                    if isinstance(mol_grp[conf_key], h5py.Group):
                        src = mol_grp[conf_key]
                        data = self._read_conformer(src)

                        # Ensure shapes
                        if data["z"].dim() == 0:
                            data["z"] = data["z"].unsqueeze(0)
                        n_atoms = data["z"].size(0)

                        if data["pos"].ndim == 1:
                            data["pos"] = data["pos"].view(-1, 3)
                        if data["forces"].ndim == 1:
                            data["forces"] = data["forces"].reshape(n_atoms, 3)

                        # Read solvation energy if present
                        if "eSOLV" in src:
                            data["esolv"] = torch.tensor(src["eSOLV"][:], dtype=torch.float).view(1)

                        result[conf_key] = data
        return result

    def process(self):
        data_list = []

        # Load primary dataset (gas or sol)
        primary = self._load_hdf5(self.hdf5_path)

        # Load paired gas dataset if provided (for AQM-sol)
        gas_data = {}
        if self.gas_hdf5_path is not None:
            gas_data = self._load_hdf5(self.gas_hdf5_path)

        # Build conformer ID map for pairing
        conf_ids = sorted(primary.keys(), key=self._conf_sort_key)
        if self.max_structures is not None:
            conf_ids = conf_ids[:self.max_structures]

        for conf_id in conf_ids:
            src = primary[conf_id]
            z = src["z"].clone()
            pos = src["pos"].clone()
            energy = src["energy"].clone()
            forces = src["forces"].clone()

            # Parse molecule/conformer index from Geom-mr-ct
            parts = conf_id.replace("Geom-", "").split("-c")
            mol_id = parts[0] if len(parts) > 0 else conf_id
            conf_idx = int(parts[1]) if len(parts) > 1 else 0

            data = Data(
                z=z,
                pos=pos,
                y_energy=energy,
                y_forces=forces,
                mol_id=mol_id,
                conf_idx=conf_idx,
                conf_id=conf_id,
            )

            # Add solvation energy if present
            if "esolv" in src:
                data.y_esolv = src["esolv"].clone()

            # Add paired gas data if available
            if conf_id in gas_data:
                g = gas_data[conf_id]
                data.gas_pos = g["pos"].clone()
                data.gas_energy = g["energy"].clone()
                data.gas_forces = g["forces"].clone()

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def _conf_sort_key(conf_id: str):
        parts = conf_id.replace("Geom-", "").split("-c")
        mol = int(parts[0]) if parts[0].isdigit() else 0
        conf = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (mol, conf)

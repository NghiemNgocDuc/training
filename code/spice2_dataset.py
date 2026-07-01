import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from typing import Optional, List
from torch_geometric.data import InMemoryDataset, Data

BOHR_TO_ANG = 0.529177
HARTREE_TO_EV = 27.2114
HARTREE_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANG  # 51.422

WATER_TRIPLE = [8, 1, 1]  # O, H, H


def _find_water_start(atomic_numbers: np.ndarray) -> int:
    n = len(atomic_numbers)
    if n < 3:
        return n
    for i in range(n - 3, -1, -3):
        if atomic_numbers[i:i + 3].tolist() != WATER_TRIPLE:
            return i + 3
    return 0


class SPICE2Dataset(InMemoryDataset):
    SUBSET_NAME = "Solvated PubChem Molecules"

    def __init__(
        self,
        root: str,
        hdf5_path: str,
        transform=None,
        pre_transform=None,
        max_molecules: Optional[int] = None,
        max_conformers_per_mol: Optional[int] = None,
    ):
        self.hdf5_path = hdf5_path
        self.max_molecules = max_molecules
        self.max_conformers_per_mol = max_conformers_per_mol
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        name = os.path.splitext(os.path.basename(self.hdf5_path))[0]
        suffix = ""
        if self.max_molecules is not None:
            suffix += f"_mol{self.max_molecules}"
        if self.max_conformers_per_mol is not None:
            suffix += f"_conf{self.max_conformers_per_mol}"
        return [f"{name}{suffix}.pt"]

    def process(self):
        data_list = []
        mol_count = 0

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            for grp_name in f.keys():
                grp = f[grp_name]
                grp_subset = grp.attrs.get("subset", "")
                if grp_subset != self.SUBSET_NAME:
                    continue

                atomic_numbers = grp["atomic_numbers"][:]
                conformations = grp["conformations"][:]
                formation_energy = grp["formation_energy"][:]
                gradients = grp["dft_total_gradient"][:]

                has_charges = "mbis_charges" in grp
                if has_charges:
                    mbis_all = grp["mbis_charges"][:]

                n_atoms_total = len(atomic_numbers)
                water_start = _find_water_start(atomic_numbers)

                solute_z = torch.tensor(atomic_numbers[:water_start], dtype=torch.long)
                n_solute = len(solute_z)

                if has_charges:
                    solute_charges = torch.tensor(mbis_all[:water_start], dtype=torch.float)

                n_conf = conformations.shape[0]
                if self.max_conformers_per_mol is not None:
                    n_conf = min(n_conf, self.max_conformers_per_mol)

                for c in range(n_conf):
                    pos_bohr = conformations[c]
                    solute_pos = torch.tensor(pos_bohr[:water_start] * BOHR_TO_ANG, dtype=torch.float)

                    energy_hartree = formation_energy[c]
                    energy_ev = float(torch.tensor(energy_hartree * HARTREE_TO_EV))

                    grad = gradients[c]
                    solute_forces = torch.tensor(
                        -grad[:water_start] * HARTREE_PER_BOHR_TO_EV_PER_ANG,
                        dtype=torch.float,
                    )

                    data = Data(
                        z=solute_z.clone(),
                        pos=solute_pos,
                        y_energy=torch.tensor([energy_ev], dtype=torch.float),
                        y_forces=solute_forces,
                        mol_id=grp_name,
                        conf_idx=c,
                    )
                    if has_charges:
                        data.charges = solute_charges.clone()

                    data_list.append(data)

                mol_count += 1
                if self.max_molecules is not None and mol_count >= self.max_molecules:
                    break

        if len(data_list) == 0:
            diag = [f"WARNING: No data loaded from {self.hdf5_path}"]
            diag.append(f"  Subset filter: '{self.SUBSET_NAME}'")
            with h5py.File(self.hdf5_path, "r") as f2:
                grps = list(f2.keys())
                diag.append(f"  Groups in file ({len(grps)}):")
                for gn in grps:
                    g = f2[gn]
                    diag.append(f"    {gn}: subset='{g.attrs.get('subset', '')}'")
            for line in diag:
                print(line)
            raise RuntimeError(
                f"No molecules matched subset '{self.SUBSET_NAME}'. "
                f"Check the 'subset' attribute in the HDF5 groups printed above."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

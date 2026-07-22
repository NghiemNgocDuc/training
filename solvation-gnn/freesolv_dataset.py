import csv
import json
import os
import sys
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from tqdm import tqdm

FREESOLV_URL = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json"
FREESOLV_GROUPS_URL = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/groups.txt"

MODEL_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17}

ELEMENT_TO_IDX = {
    1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7,
    3: 8, 5: 9, 11: 10, 12: 11, 14: 12, 19: 13, 20: 14, 35: 15, 53: 16,
}
HARTREE_TO_EV = 27.2114
EV_TO_KCAL = 23.0605


def download_freesolv_data(cache_dir: str = "Data/FreeSolv") -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    json_path = os.path.join(cache_dir, "database.json")
    groups_path = os.path.join(cache_dir, "groups.txt")

    if not os.path.exists(json_path):
        print(f"Downloading FreeSolv database from {FREESOLV_URL}")
        urllib.request.urlretrieve(FREESOLV_URL, json_path)
    if not os.path.exists(groups_path):
        print(f"Downloading FreeSolv groups from {FREESOLV_GROUPS_URL}")
        urllib.request.urlretrieve(FREESOLV_GROUPS_URL, groups_path)

    return json_path, groups_path


def load_freesolv_labels(json_path: str) -> Dict[str, dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def _relax_xtb(mol: Chem.Mol, conf_id: int) -> Optional[np.ndarray]:
    try:
        from xtb.ase.calculator import XTB
        from ase import Atoms
        from ase.optimize import BFGS
    except ImportError:
        return None

    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = conf.GetPositions()
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = XTB(method="GFN2-xTB", solvent="water")
    opt = BFGS(atoms, logfile=None)
    try:
        opt.run(fmax=0.05, steps=200)
        return np.array(atoms.get_positions(), dtype=np.float64)
    except Exception:
        return None


def generate_conformers(
    smiles: str,
    num_confs: int = 10,
    random_seed: int = 42,
    use_xtb: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = 0.5

    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if not conf_ids:
        conf_id = rdDistGeom.EmbedMolecule(mol, randomSeed=random_seed)
        if conf_id < 0:
            return None
        print(f"    Warning: EmbedMultipleConfs failed for {smiles}, using single conformer")

    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    energies = []
    for conf_id in range(mol.GetNumConformers()):
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energies.append(ff.CalcEnergy() if ff else float("inf"))
    best_conf_id = int(np.argmin(energies))
    if energies[best_conf_id] == float("inf"):
        return None

    atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)

    if use_xtb:
        relaxed = _relax_xtb(mol, best_conf_id)
        if relaxed is not None:
            return atomic_numbers, relaxed

    conf = mol.GetConformer(best_conf_id)
    positions = np.array(conf.GetPositions(), dtype=np.float64)
    return atomic_numbers, positions


def create_freesolv_hdf5(
    labels: Dict[str, dict],
    output_path: str = "freesolv_conformers.hdf5",
    labels_csv: str = "freesolv_labels.csv",
    num_confs: int = 10,
    use_xtb: bool = True,
):
    n_total = len(labels)
    n_compatible = 0
    n_generated = 0
    n_failed_smiles = 0
    n_failed_elements = 0
    n_failed_conformer = 0
    excluded_element_counts: Counter = Counter()

    if use_xtb:
        try:
            from xtb.ase.calculator import XTB
            print("xTB relaxation: enabled (GFN2-xTB + GBSA water)")
        except ImportError:
            print("xTB relaxation: requested but not available (install xtb-python + ase)")
            print("Falling back to MMFF geometries")
            use_xtb = False
    else:
        print("xTB relaxation: disabled (using MMFF geometries)")

    csv_rows = []

    with h5py.File(output_path, "w") as f:
        for mol_id, entry in tqdm(labels.items(), desc="Generating conformers"):
            smiles = entry["smiles"]

            mol_check = Chem.MolFromSmiles(smiles)
            if mol_check is None:
                n_failed_smiles += 1
                csv_rows.append({"mol_id": mol_id, "smiles": smiles,
                                 "in_model_vocab": False, "reason": "smiles_parse_failed"})
                continue

            mol_elements = set(a.GetAtomicNum() for a in mol_check.GetAtoms())
            if not mol_elements.issubset(MODEL_ELEMENTS):
                n_failed_elements += 1
                for z in mol_elements - MODEL_ELEMENTS:
                    excluded_element_counts[z] += 1
                csv_rows.append({"mol_id": mol_id, "smiles": smiles,
                                 "in_model_vocab": False, "reason": "excluded_elements"})
                continue
            n_compatible += 1

            result = generate_conformers(smiles, num_confs=num_confs, use_xtb=use_xtb)
            if result is None:
                n_failed_conformer += 1
                csv_rows.append({"mol_id": mol_id, "smiles": smiles,
                                 "in_model_vocab": True, "reason": "conformer_failed"})
                continue

            atomic_numbers, positions = result
            grp = f.create_group(mol_id)
            grp.create_dataset("atNUM", data=atomic_numbers)
            grp.create_dataset("atXYZ", data=positions)
            grp.attrs["smiles"] = smiles
            grp.attrs["expt_dg"] = entry.get("expt", float("nan"))
            grp.attrs["expt_uncertainty"] = entry.get("d_expt", float("nan"))
            grp.attrs["calc_dg"] = entry.get("calc", float("nan"))
            grp.attrs["iupac"] = entry.get("iupac", "")
            n_generated += 1
            csv_rows.append({"mol_id": mol_id, "smiles": smiles,
                             "in_model_vocab": True, "reason": "ok"})

    with open(labels_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "smiles", "in_model_vocab", "reason"])
        w.writeheader()
        w.writerows(csv_rows)

    n_failed = n_total - n_generated
    print(f"\nSummary:")
    print(f"  Total FreeSolv entries:                  {n_total}")
    print(f"  Element-compatible (model vocab):        {n_compatible}")
    print(f"  Conformers generated:                    {n_generated}")
    print(f"  Failed:                                  {n_failed}")
    print(f"    - SMILES parse failures:               {n_failed_smiles}")
    print(f"    - Excluded elements:                   {n_failed_elements}")
    print(f"    - Conformer generation failures:       {n_failed_conformer}")
    if excluded_element_counts:
        print(f"  Excluded elements (atomic numbers):")
        for z, cnt in sorted(excluded_element_counts.items()):
            print(f"    Z={z:3d}: {cnt} molecules")
    print(f"  FreeSolv coverage: {n_generated}/{n_total} molecules compatible with model vocabulary")
    print(f"  HDF5 output:       {output_path}")
    print(f"  Labels CSV:        {labels_csv}")


class FreeSolvDataset:
    def __init__(self, hdf5_path: str):
        import torch
        from torch_geometric.data import Data
        self.samples = []
        with h5py.File(hdf5_path, "r") as f:
            for mol_id in f.keys():
                grp = f[mol_id]
                z = grp["atNUM"][...]
                pos = grp["atXYZ"][...]
                data = Data(
                    z=torch.tensor(z, dtype=torch.long),
                    pos=torch.tensor(pos, dtype=torch.float),
                )
                data.mol_id = mol_id
                self.samples.append(data)
        print(f"FreeSolvDataset: {len(self.samples)} molecules loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def get_mol_ids(self) -> List[str]:
        return [s.mol_id for s in self.samples]


def predict_freesolv(
    dataset: FreeSolvDataset,
    correction_model,
    device,
    batch_size: int = 32,
    vacuum_model=None,
    explicit_model=None,
):
    """Run GNN inference on FreeSolv conformers.

    The correction model was trained to predict the residual solvated energy
    (E_solv - E_vacuum), which directly equals the solvation free energy ΔG.
    Model outputs are in eV; conversion: 1 eV = 23.0605 kcal/mol.

    Args:
        dataset: FreeSolvDataset with generated conformers.
        correction_model: Frozen Stage-2a implicit correction model.
        explicit_model: Optional frozen Stage-2b explicit correction.

    Returns:
        List of dicts with mol_id, dG_pred_kcal.
    """
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []

    for data in loader:
        data = data.to(device)
        x = build_one_hot(data, device)
        with torch.no_grad():
            corr_e = correction_model(x, data.pos, data.batch)
            total_e = corr_e
            if explicit_model is not None:
                explicit_e = explicit_model(x, data.pos, data.batch)
                total_e = total_e + explicit_e

        for i in range(data.num_graphs):
            dG_eV = total_e[i].item()
            dG_kcal = dG_eV * EV_TO_KCAL
            results.append({
                "mol_id": data.mol_id[i],
                "dG_pred_kcal": dG_kcal,
            })

    # Sign convention check
    preds = np.array([r["dG_pred_kcal"] for r in results])
    n_negative = int((preds < 0).sum())
    n_total = len(preds)
    print(f"\nSign check: {n_negative}/{n_total} predictions are negative")
    print(f"  Mean predicted DeltaG: {preds.mean():.2f} kcal/mol")
    print(f"  Unit: 1 eV = {EV_TO_KCAL:.4f} kcal/mol")
    if n_negative < n_total * 0.5:
        print("  WARNING: majority of predictions have wrong sign.")
        print("  Check whether correction model output needs sign flip")
        print("  or whether atomic references were applied correctly.")

    return results


def evaluate_freesolv(
    predictions: List[dict],
    labels: Dict[str, dict],
) -> dict:
    """Compute RMSE, MAE, R^2, Kendall tau against experimental FreeSolv data."""
    import numpy as np
    from scipy.stats import kendalltau

    pred_map = {p["mol_id"]: p["dG_pred_kcal"] for p in predictions}

    y_true, y_pred = [], []
    for mol_id, entry in labels.items():
        expt = entry.get("expt")
        if mol_id in pred_map and isinstance(expt, (int, float)):
            y_true.append(expt)
            y_pred.append(pred_map[mol_id])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    n = len(y_true)

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    tau, p_value = kendalltau(y_true, y_pred)

    return {
        "n_molecules": n,
        "MAE_kcal": mae,
        "RMSE_kcal": rmse,
        "R2": r2,
        "Kendall_tau": float(tau),
        "Kendall_p": float(p_value),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FreeSolv conformers for GNN inference")
    parser.add_argument("--output", type=str, default="freesolv_conformers.hdf5")
    parser.add_argument("--num_confs", type=int, default=10, help="Number of conformers to generate per molecule")
    parser.add_argument("--cache_dir", type=str, default="Data/FreeSolv")
    parser.add_argument("--check_elements", action="store_true", help="Only check element compatibility, don't generate")
    parser.add_argument("--no_xtb", action="store_true", help="Skip GFN2-xTB relaxation (use MMFF geometries)")
    args = parser.parse_args()

    json_path, groups_path = download_freesolv_data(args.cache_dir)
    labels = load_freesolv_labels(json_path)

    if args.check_elements:
        vocab_set = MODEL_ELEMENTS
        n_compatible = 0
        all_elements = set()
        excluded_counts = Counter()
        for mol_id, entry in labels.items():
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            zs = set(a.GetAtomicNum() for a in mol.GetAtoms())
            all_elements.update(zs)
            if zs.issubset(vocab_set):
                n_compatible += 1
            else:
                for z in zs - vocab_set:
                    excluded_counts[z] += 1
        print(f"Elements in FreeSolv: {sorted(all_elements)}")
        print(f"Model vocabulary: {sorted(vocab_set)}")
        print(f"Compatible: {n_compatible}/{len(labels)}")
        print(f"Incompatible: {len(labels) - n_compatible}")
        print(f"Excluded elements:")
        for z, cnt in sorted(excluded_counts.items()):
            print(f"  Z={z:3d}: {cnt} molecules")
        sys.exit(0)

    create_freesolv_hdf5(
        labels,
        output_path=args.output,
        num_confs=args.num_confs,
        use_xtb=not args.no_xtb,
    )

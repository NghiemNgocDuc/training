"""Deep Ensemble over the verified DimeNet+ FreeSolv fine-tune pipeline.

Fully ADDITIVE companion to cv_finetune.py - it never modifies that script,
the fold-splitting logic, or any part of the verified 5-fold CV pipeline.

A real Deep Ensemble requires the SAME train/val/test split with ONLY the
random seed changing across runs (the 5-fold CV models differ in BOTH fold
membership AND seed, which mixes data-sensitivity with training randomness
and is therefore not a valid ensemble). This script:

  PHASE 1 - BUILD
    * Freezes fold 0's split by LOADING the saved *_ids.json files from
      cv_results_full/fold_0 (already verified disjoint and leak-free; never
      regenerated).
    * Fine-tunes the SAME Stage-2 correction checkpoint (the one the verified
      fold-0 run started from) N times with N different seeds; every other
      knob is byte-for-byte the fold-0 cv_finetune setup: architecture,
      optimizer, scheduler, early stopping on val_mae, MSE-in-eV loss, grad
      clip, batch size, and the same 5-conformer RDKit test-time averaging
      that produced the published fold-0 numbers (MAE 0.515 / RMSE 0.773).

  PHASE 2 - VALIDATE
    * Per-molecule ensemble mean + std across seeds.
    * Ensemble-mean MAE/RMSE vs each individual seed (point-estimate effect).
    * Spearman rank correlation between ensemble_std and |error| - the UQ
      calibration check - plus a scatter plot you can actually look at.
    * Halogen (Br/I) vs non-halogen disagreement test (Mann-Whitney U) tied
      to the known missing-reference-energy gap for those elements.

Usage (all paths repo-root relative or absolute; run from anywhere):
  # train one member:
  python deep_ensemble.py --mode train --seed 42
  # train all five members:
  python deep_ensemble.py --mode train --seeds 42 123 7 2024 999
  # sanity check + full UQ analysis (after >=2 members exist):
  python deep_ensemble.py --mode analyze
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # live logs when stdout is a file (nohup)

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)          # aqm-spice2/
REPO_ROOT = os.path.dirname(_parent)
sys.path.append(_parent)
sys.path.append(_script_dir)

os.chdir(_parent)  # same convention as cv_finetune.py: pipeline imports resolve from aqm-spice2/

EV_TO_KCAL = 23.0605
DEFAULT_SEEDS = [42, 123, 7, 2024, 999]

# Frozen fold-0 split from the VERIFIED full run (disjoint, leak-free).
DEFAULT_SPLIT_DIR = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")

# Stage-2 correction checkpoint the verified fold-0 run started from.
DEFAULT_CORRECTION_CKPT = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline", "results_full", "stage2_correction.pt")

DEFAULT_CONFORMERS = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
DEFAULT_OUTPUT_DIR = os.path.join(_script_dir, "deep_ensemble")

HALOGEN_Z = {35, 53}  # Br, I

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


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
    import numpy as np
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


def simple_dataset_cls(hdf5_path, labels):
    import torch
    from torch_geometric.data import Data

    class SimpleDataset:
        def __init__(self, ids):
            self.ids = ids
            self.hdf5_path = hdf5_path
            self.labels = labels
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
                self._cache[mid] = d.clone()
            data = self._cache[mid].clone()
            data.mol_id = mid
            data.y_dG = torch.tensor([self.labels[mid]["expt"]], dtype=torch.float)
            return data

    return SimpleDataset


def build_model(device):
    from DimeModels import DimeNetPlus
    from element_vocab import NUM_ELEMENTS
    return DimeNetPlus(
        in_channels=NUM_ELEMENTS, hidden_channels=128, out_channels=1,
        num_blocks=3, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
    ).to(device)


# ---------------------------------------------------------------------------
# PHASE 1 - train one ensemble member
# ---------------------------------------------------------------------------


def train_member(seed, conformers, split_dir, correction_ckpt, output_dir,
                 epochs, patience, lr, batch_size, n_conformers, device_name):
    import h5py
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels

    json_path, _ = download_freesolv_data(os.path.join(REPO_ROOT, "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(json_path)

    train_ids, val_ids, test_ids = load_frozen_split(split_dir, all_labels)

    # Record the hash of the EXACT bytes we loaded, so the analyze step can
    # prove every member loaded the identical frozen split.
    split_blob = b"".join(
        open(os.path.join(split_dir, name), "rb").read()
        for name in ("train_ids.json", "val_ids.json", "test_ids.json"))
    split_md5 = md5_bytes(split_blob)

    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable - falling back to cpu to match "
              "the verified fold-0 pipeline (which ran on cpu).")
        device_name = "cpu"
    device = torch.device(device_name)

    print(f"\n{'='*66}")
    print(f"  ENSEMBLE MEMBER seed={seed}  (identical split + arch + hyperparams; "
          f"only the seed changes)")
    print(f"{'='*66}")
    print(f"  split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test "
          f"({split_md5[:10]}...) frozen from {split_dir}")
    print(f"  init: {correction_ckpt}")
    print(f"  hyperparams: lr={lr} wd=1e-5 batch={batch_size} epochs={epochs} "
          f"patience={patience} | MSE in eV, grad-clip 10.0, ReduceLROnPlateau "
          f"(f=0.5, pat={patience // 2}, min_lr=1e-6) | {n_conformers}-conf TTA")

    set_seed(seed)

    model = build_model(device)
    ckpt = torch.load(correction_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)  # strict; the arch is identical
    print(f"  loaded Stage-2 correction checkpoint ({sum(p.numel() for p in model.parameters()):,} params)")

    SimpleDataset = simple_dataset_cls(conformers, all_labels)
    train_ds = SimpleDataset(train_ids)
    val_ds = SimpleDataset(val_ids)
    test_ds = SimpleDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)
    mse = torch.nn.MSELoss()

    def evaluate_loader(loader):
        model.eval()
        all_p, all_e = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                dG_exp = data.y_dG.view(-1).to(device)
                valid = ~torch.isnan(dG_exp)
                all_p.append(pred[valid].cpu())
                all_e.append(dG_exp[valid].cpu())
        preds = torch.cat(all_p).numpy()
        expts = torch.cat(all_e).numpy()
        mae = float(np.mean(np.abs(preds - expts)))
        rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
        return mae, rmse, preds, expts

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    best_ckpt_path = os.path.join(seed_dir, f"ensemble_seed{seed}.pt")

    best_val_mae = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = epochs
    t0_all = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            dG_exp = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            valid = ~torch.isnan(dG_exp)
            if valid.sum() == 0:
                continue
            loss = mse(pred[valid], dG_exp[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        val_mae, val_rmse, _, _ = evaluate_loader(val_loader)
        scheduler.step(val_mae)
        dt = time.time() - t0

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            stale += 1

        print(f"    seed {seed:>4} | epoch {epoch:3d}/{epochs} | best val {best_val_mae:7.3f} "
              f"(ep {best_epoch}) | cur val {val_mae:7.3f} | {dt:5.1f}s/ep", flush=True)

        if stale >= patience:
            stop_epoch = epoch
            print(f"    seed {seed} early stopped at epoch {epoch} (patience {patience})")
            break

    total_min = (time.time() - t0_all) / 60.0

    # ---- final test pass with best-val checkpoint, single conformer ----
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    model.eval()
    test_mae, test_rmse, test_preds, test_expts = evaluate_loader(test_loader)

    # ---- conformer test-time averaging (same protocol as the published run) ----
    tta_mae, tta_rmse, tta_preds_by_mid = conformer_average(
        model, device, test_ids, all_labels, conformers, n_conformers, batch_size)

    preds_path = os.path.join(seed_dir, "predictions.csv")
    with open(preds_path, "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal\n")
        for mid in test_ids:
            f.write(f"{mid},{tta_preds_by_mid[mid]:.6f},{all_labels[mid]['expt']:.6f}\n")

    # ---- copy the frozen split in so analyze() can diff per-member copies ----
    for name in ("train_ids.json", "val_ids.json", "test_ids.json"):
        with open(os.path.join(split_dir, name), "rb") as src, \
                open(os.path.join(seed_dir, name), "wb") as dst:
            dst.write(src.read())
    with open(os.path.join(seed_dir, "split.md5"), "w") as f:
        f.write(split_md5 + "\n")

    metrics = {
        "seed": seed,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "split_md5": split_md5,
        "split_source": split_dir,
        "correction_ckpt": correction_ckpt,
        "hyperparams": {"lr": lr, "weight_decay": 1e-5, "batch_size": batch_size,
                        "epochs": epochs, "patience": patience, "device": device_name},
        "best_val_mae_kcal": best_val_mae,
        "best_val_epoch": best_epoch,
        "early_stop_epoch": stop_epoch,
        "total_min": round(total_min, 1),
        "test_mae_single_conf_kcal": test_mae,
        "test_rmse_single_conf_kcal": test_rmse,
        "test_mae_tta_kcal": tta_mae,
        "test_rmse_tta_kcal": tta_rmse,
        "n_conformers_tta": n_conformers,
        "checkpoint_sha256": sha256_file(best_ckpt_path),
    }
    with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  seed {seed} DONE | best val MAE {best_val_mae:.3f} (ep {best_epoch}, "
          f"stopped ep {stop_epoch}) | test {test_mae:.3f}/{test_rmse:.3f} (single conf) | "
          f"test {tta_mae:.3f}/{tta_rmse:.3f} ({n_conformers}-conf TTA) | "
          f"ckpt sha256 {metrics['checkpoint_sha256'][:12]}... | {total_min:.1f} min")
    return metrics


def conformer_average(model, device, test_ids, all_labels, conformers, n_conformers, batch_size):
    """RDKit ETKDGv3 conformer test-time averaging - identical to the protocol
    in cv_finetune.py that produced the published fold-0 numbers."""
    import numpy as np
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


# ---------------------------------------------------------------------------
# PHASE 2 - validate the uncertainty is meaningful
# ---------------------------------------------------------------------------


def analyze(output_dir, conformers, seeds=DEFAULT_SEEDS, n_expected=5):
    import h5py
    import numpy as np
    from scipy.stats import spearmanr, mannwhitneyu

    # ---- collect members ----
    members = {}
    for seed in seeds:
        d = os.path.join(output_dir, f"seed_{seed}")
        preds_path = os.path.join(d, "predictions.csv")
        if not os.path.exists(preds_path):
            print(f"  WARNING: seed {seed} has no predictions.csv - skipping")
            continue
        with open(preds_path) as f:
            next(f)
            rows = [line.rstrip("\n").split(",") for line in f]
        members[seed] = {r[0]: (float(r[1]), float(r[2])) for r in rows}
    if len(members) < 2:
        print("ERROR: need at least 2 trained members before analyze(). "
              "Train first: python deep_ensemble.py --mode train --seeds 42 123 7 2024 999")
        sys.exit(1)

    seed_list = list(members.keys())

    # ---- sanity 1: identical frozen split across all members ----
    split_md5s, ckpt_shas, ckpt_times = {}, {}, {}
    for seed in seed_list:
        d = os.path.join(output_dir, f"seed_{seed}")
        with open(os.path.join(d, "split.md5")) as f:
            split_md5s[seed] = f.read().strip()
        ckpt = os.path.join(d, f"ensemble_seed{seed}.pt")
        ckpt_shas[seed] = sha256_file(ckpt)
        ckpt_times[seed] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt)))

    same_split = len(set(split_md5s.values())) == 1
    print("\n" + "=" * 66)
    print("  Sanity checks")
    print("=" * 66)
    print(f"  split md5 across seeds: {sorted(set(split_md5s.values()))} -> "
          f"{'IDENTICAL' if same_split else 'MISMATCH!'}")
    distinct_ckpt = len(set(ckpt_shas.values())) == len(seed_list)
    print(f"  checkpoint sha256: {len(set(ckpt_shas.values()))} distinct across "
          f"{len(seed_list)} seeds -> {'DISTINCT (good)' if distinct_ckpt else 'DUPLICATES!'}")
    for seed in seed_list:
        print(f"      seed {seed:>4}: {ckpt_shas[seed][:12]}... written {ckpt_times[seed]}")

    # ---- per-member split-file byte check (diff of saved copies) ----
    per_member_split_blob = []
    for seed in seed_list:
        d = os.path.join(output_dir, f"seed_{seed}")
        blob = b"".join(open(os.path.join(d, name), "rb").read()
                        for name in ("train_ids.json", "val_ids.json", "test_ids.json"))
        per_member_split_blob.append(md5_bytes(blob))
    same_copies = len(set(per_member_split_blob)) == 1
    print(f"  saved split copies (train/val/test) diff across seeds -> "
          f"{'IDENTICAL' if same_copies else 'MISMATCH!'}")

    # ---- build per-molecule table ----
    test_ids = [r[0] for r in rows]  # same order for every member (split frozen)
    per_mol = {}
    with h5py.File(conformers, "r") as f:
        for mid in test_ids:
            z = f[mid]["atNUM"][...].tolist()
            per_mol[mid] = {
                "preds": [members[seed][mid][0] for seed in seed_list],
                "exp": members[seed][mid][1],
                "has_halogen": int(any(int(a) in HALOGEN_Z for a in z)),
            }
    for mid in per_mol:
        arr = np.array(per_mol[mid]["preds"])
        per_mol[mid]["std"] = float(np.std(arr, ddof=1))
        per_mol[mid]["mean"] = float(np.mean(arr))
        per_mol[mid]["abs_error"] = float(abs(per_mol[mid]["mean"] - per_mol[mid]["exp"]))

    # ---- per-molecule CSV (Step 4) ----
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    csv_path = os.path.join(agg_dir, "per_molecule.csv")
    cols = ["mol_id"] + [f"pred_seed{s}" for s in seed_list] + \
           ["ensemble_mean", "ensemble_std", "true_value", "abs_error", "has_halogen_Br_I"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for mid in test_ids:
            p = per_mol[mid]
            row = [mid] + [f"{v:.6f}" for v in p["preds"]] + \
                  [f"{p['mean']:.6f}", f"{p['std']:.6f}",
                   f"{p['exp']:.6f}", f"{p['abs_error']:.6f}", str(p["has_halogen"])]
            f.write(",".join(row) + "\n")

    # ---- Step 5: individual vs ensemble point estimates ----
    ind_mae, ind_rmse = {}, {}
    for seed in seed_list:
        preds = np.array([per_mol[mid]["preds"][seed_list.index(seed)] for mid in test_ids])
        expts = np.array([per_mol[mid]["exp"] for mid in test_ids])
        mae, rmse, _ = evaluate(preds, expts)
        ind_mae[seed] = mae
        ind_rmse[seed] = rmse
    ens_preds = np.array([per_mol[mid]["mean"] for mid in test_ids])
    ens_expts = np.array([per_mol[mid]["exp"] for mid in test_ids])
    ens_mae, ens_rmse, ens_r2 = evaluate(ens_preds, ens_expts)

    # ---- Step 6: disagreement vs error ----
    stds = np.array([per_mol[mid]["std"] for mid in test_ids])
    abserr = np.array([per_mol[mid]["abs_error"] for mid in test_ids])
    rho, pval = spearmanr(stds, abserr)
    from scipy.stats import pearsonr
    pear_r, pear_p = pearsonr(stds, abserr)

    # ---- scatter plot (Step 6) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([per_mol[mid]["std"] for mid in test_ids])
    y = np.array([per_mol[mid]["abs_error"] for mid in test_ids])
    halogen = np.array([per_mol[mid]["has_halogen"] == 1 for mid in test_ids])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x[~halogen], y[~halogen], alpha=0.55, s=32, edgecolors="k", linewidths=0.3, label="non-halogen")
    ax.scatter(x[halogen], y[halogen], alpha=0.85, s=48, facecolors="crimson", edgecolors="k", linewidths=0.4, label="Br/I")
    ax.set_xlabel("Ensemble std (kcal/mol) - model disagreement")
    ax.set_ylabel("|ensemble mean - experiment| (kcal/mol)")
    ax.set_title(f"Uncertainty vs error (N={len(x)})\nSpearman $\\rho$={rho:.3f} (p={pval:.3f})")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(agg_dir, "scatter_uncertainty.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # ---- Step 7: halogen vs non-halogen disagreement ----
    g_hal = stds[halogen]
    g_non = stds[~halogen]
    e_hal = abserr[halogen]
    e_non = abserr[~halogen]
    n_hal = int(halogen.sum())
    if n_hal > 0 and len(g_non) > 0:
        stat_u, p_u = mannwhitneyu(g_hal, g_non, alternative="two-sided")
        stat_u_err, p_u_err = mannwhitneyu(e_hal, e_non, alternative="two-sided")
    else:
        stat_u, p_u = float("nan"), float("nan")
        stat_u_err, p_u_err = float("nan"), float("nan")
        print(f"  WARNING: no halogen molecules (n_hal={n_hal}); "
              f"skipping halogen vs non-halogen Mann-Whitney test")

    # ---- honest verdict ----
    verdict = compose_verdict(rho, pval, n_hal, g_hal, g_non, e_hal, e_non, stat_u, p_u)

    # ---- write report ----
    report = {
        "seeds": seed_list,
        "sanity": {"split_identical": same_split, "split_copies_identical": same_copies,
                   "checkpoints_distinct": distinct_ckpt},
        "individual_test": {str(s): {"mae_kcal": ind_mae[s], "rmse_kcal": ind_rmse[s]} for s in seed_list},
        "ensemble_mean": {"mae_kcal": ens_mae, "rmse_kcal": ens_rmse, "r2": ens_r2},
        "spearman_std_vs_abs_error": {"rho": float(rho), "p": float(pval), "N": int(len(x))},
        "pearson_std_vs_abs_error": {"r": float(pear_r), "p": float(pear_p), "N": int(len(x))},
        "halogen": {
            "halogen_count": n_hal, "non_halogen_count": len(g_non),
            "mean_std_halogen": float(np.mean(g_hal)), "mean_std_non_halogen": float(np.mean(g_non)),
            "median_std_halogen": float(np.median(g_hal)), "median_std_non_halogen": float(np.median(g_non)),
            "mean_abs_error_halogen": float(np.mean(e_hal)), "mean_abs_error_non_halogen": float(np.mean(e_non)),
            "mannwhitney_std_U": float(stat_u), "mannwhitney_std_p": float(p_u),
            "mannwhitney_abs_error_U": float(stat_u_err), "mannwhitney_abs_error_p": float(p_u_err),
        },
        "artifacts": {"per_molecule_csv": csv_path, "scatter_png": plot_path},
        "verdict": verdict,
    }
    with open(os.path.join(agg_dir, "ensemble_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    txt_path = os.path.join(agg_dir, "ensemble_report.txt")
    with open(txt_path, "w") as f:
        f.write(report_block(report))

    # ---- console report ----
    print("\n" + "=" * 66)
    print("  STEP 5  individual members vs ensemble mean (test, TTA preds, kcal/mol)")
    print("=" * 66)
    print(f"  {'seed':<8} {'MAE':<10} {'RMSE':<10}")
    for s in seed_list:
        print(f"  {s:<8} {ind_mae[s]:<10.3f} {ind_rmse[s]:<10.3f}")
    print(f"  {'ensemble':<8} {ens_mae:<10.3f} {ens_rmse:<10.3f}  (R2={ens_r2:.4f})")

    print("\n" + "=" * 66)
    print("  STEP 6  does disagreement predict error?")
    print("=" * 66)
    print(f"  Spearman rho = {rho:.4f}   p = {pval:.4e}   N = {len(x)}")
    print(f"  Pearson r    = {pear_r:.4f}   p = {pear_p:.4e}   N = {len(x)}")
    print(f"  scatter -> {plot_path}")

    print("\n" + "=" * 66)
    print("  STEP 7  halogen (Br/I) vs non-halogen")
    print("=" * 66)
    print(f"  Br/I molecules: {n_hal}  |  others: {len(g_non)}")
    print(f"  mean ensemble_std: halogen {np.mean(g_hal):.4f} | non {np.mean(g_non):.4f}")
    print(f"  median " + f"         : halogen {np.median(g_hal):.4f} | non {np.median(g_non):.4f}")
    print(f"  mean abs_error   : halogen {np.mean(e_hal):.4f} | non {np.mean(e_non):.4f}")
    print(f"  Mann-Whitney U on std: U={stat_u:.1f}, p={p_u:.4f}")
    print(f"  Mann-Whitney U on |err|: U={stat_u_err:.1f}, p={p_u_err:.4f}")

    print("\n" + "=" * 66)
    print("  VERDICT")
    print("=" * 66)
    print(verdict)
    print(f"\n  saved: {csv_path}")
    print(f"         {plot_path}")
    print(f"         {txt_path}")


def compose_verdict(rho, pval, n_hal, g_hal, g_non, e_hal, e_non, stat_u, p_u):
    lines = []
    meaningful = rho > 0.25 and pval < 0.05
    lines.append(f"  Spearman(ensemble_std, |error|) = {rho:.3f} (p={pval:.3f}).")
    if meaningful:
        lines.append(f"  Statistically meaningful: molecules the 5 models disagree on are")
        lines.append(f"  disproportionately wrong. A user told to trust/disregard predictions")
        lines.append(f"  by std would beat chance.")
    else:
        lines.append(f"  Weak/insignificant: ensemble disagreement barely tracks where the")
        lines.append(f"  model is wrong. Trusting std as a confidence flag would not beat chance.")
    lines.append("")
    if n_hal < 5:
        lines.append(f"  Halogen group too small (n={n_hal}) for a firm conclusion.")
    elif p_u < 0.05 and np.mean(g_hal) > np.mean(g_non):
        lines.append(f"  Halogen disagreement is significantly higher (std {np.mean(g_hal):.3f} vs")
        lines.append(f"  {np.mean(g_non):.3f}, U={stat_u:.1f}, p={p_u:.3f}) - the model 'knows' it is")
        lines.append(f"  less sure about Br/I, tying to the missing reference-energy baseline.")
    elif p_u < 0.05 and np.mean(g_hal) < np.mean(g_non):
        lines.append(f"  Halogen disagreement is significantly LOWER than non-halogen, despite")
        lines.append(f"  worse errors (|err| {np.mean(e_hal):.3f} vs {np.mean(e_non):.3f}) - a case of")
        lines.append(f"  confident-wrong, an important limitation to flag.")
    else:
        lines.append(f"  Halogen vs non-halogen disagreement does not differ significantly")
        lines.append(f"  (p={p_u:.3f}) - no clean confirmation either way; if halogen errors are")
        lines.append(f"  also not elevated here, this check is simply underpowered or moot.")
    return "\n".join(lines)


def report_block(report):
    r = report
    out = []
    out.append("DEEP ENSEMBLE UNCERTAINTY REPORT")
    out.append("=" * 60)
    out.append(f"seeds: {r['seeds']}")
    out.append(f"sanity: split_identical={r['sanity']['split_identical']}, "
               f"checkpoints_distinct={r['sanity']['checkpoints_distinct']}")
    out.append("")
    out.append("individual test (kcal/mol):")
    out.append(f"  {'seed':<8} {'MAE':<10} {'RMSE':<10}")
    for s, m in r["individual_test"].items():
        out.append(f"  {s:<8} {m['mae_kcal']:<10.3f} {m['rmse_kcal']:<10.3f}")
    e = r["ensemble_mean"]
    out.append(f"  ensemble mean MAE {e['mae_kcal']:.3f} RMSE {e['rmse_kcal']:.3f} R2 {e['r2']:.4f}")
    out.append("")
    sp = r["spearman_std_vs_abs_error"]
    pe = r["pearson_std_vs_abs_error"]
    out.append(f"Spearman(std, |error|): rho={sp['rho']:.4f} p={sp['p']:.4e} N={sp['N']}")
    out.append(f"Pearson (std, |error|):  r={pe['r']:.4f} p={pe['p']:.4e} N={pe['N']}")
    out.append("")
    h = r["halogen"]
    out.append(f"halogen Br/I: n={h['halogen_count']}  non: n={h['non_halogen_count']}")
    out.append(f"  mean std: halogen {h['mean_std_halogen']:.4f} | non {h['mean_std_non_halogen']:.4f}")
    out.append(f"  median std: halogen {h['median_std_halogen']:.4f} | non {h['median_std_non_halogen']:.4f}")
    out.append(f"  mean |err|: halogen {h['mean_abs_error_halogen']:.4f} | non {h['mean_abs_error_non_halogen']:.4f}")
    out.append(f"  Mann-Whitney std: U={h['mannwhitney_std_U']:.1f} p={h['mannwhitney_std_p']:.4f}")
    out.append(f"  Mann-Whitney |err|: U={h['mannwhitney_abs_error_U']:.1f} p={h['mannwhitney_abs_error_p']:.4f}")
    out.append("")
    out.append("VERDICT")
    out.append(r["verdict"])
    out.append("")
    out.append(f"artifacts: {r['artifacts']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import numpy as np
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description="Deep ensemble (UQ) for the DimeNet+ FreeSolv pipeline")
    parser.add_argument("--mode", choices=["train", "analyze"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    parser.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--correction_ckpt", default=DEFAULT_CORRECTION_CKPT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_conformers", type=int, default=5,
                        help="test-time conformer averaging (5 = the published fold-0 protocol)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.mode == "train":
        if args.seeds:
            seed_list = args.seeds
        elif args.seed is not None:
            seed_list = [args.seed]
        else:
            parser.error("train mode needs --seed N or --seeds N [N ...]")
        for s in seed_list:
            train_member(seed=s, conformers=args.conformers, split_dir=args.split_dir,
                         correction_ckpt=args.correction_ckpt, output_dir=args.output_dir,
                         epochs=args.epochs, patience=args.patience, lr=args.lr,
                         batch_size=args.batch_size, n_conformers=args.n_conformers,
                         device_name=args.device)
    else:
        analyze(args.output_dir, args.conformers)


if __name__ == "__main__":
    main()
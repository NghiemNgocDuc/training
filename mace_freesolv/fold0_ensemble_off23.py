"""MACE fold-0 ensemble FROM THE RELEASED OFF23 FOUNDATION — train + per-atom dump.

Differs from fold0_ensemble.py (from-scratch Stage-A init) in ONE thing:
each seed fine-tunes the official MACE-OFF23 foundation weights
(ACEsuit/mace-off, SPICE wB97M-D3(BJ)/def2-TZVPPD, ASL license) on the SAME
frozen fold-0 split — only the seed changes. Atomic refs are refit on the
fold-0 train split per seed (standard foundation fine-tune path).
Everything else (frozen split, hyperparams, per-atom dump, gates) identical,
so GIMS/VW/uniform numbers are directly comparable to the scratch run.

Why this exists:
  `mace_freesolv/results/fold_*/model.pt` are SINGLE-seed per-fold models with
  total-energy only. GIMS / VW / matched-uniform need a same-split ENSEMBLE
  (seeds differ, data identical) plus PER-ATOM contributions P_mi^k with
  sum_i P_mi^k == E_m^k. This script produces exactly that for fold 0.

What it does (every step has a tqdm progress bar):
  1. Load FROZEN fold-0 split (411 train / 102 val / 129 test). Never reshuffles.
  2. For each seed in --seeds (default 42,123,7,2024,999):
     a. Train from the SAME released OFF23 foundation (only seed changes).
     b. Save best-val checkpoint + metrics.json + test predictions.
  3. Probe the MACE output dict once to find node energies (logs all keys).
  4. Dump per-atom P for ALL 642 FreeSolv molecules (stored HDF5 geometry,
     single conformer — same regime as the DimeNet shrinkage arms):
       peratom_mace_seed{S}.pkl  (P_all / P_train / P_val / P_test / E_all / mu_T)
       mace_node_contributions.csv  (mol_id, atom_idx, P_seed*, for GIMS reuse)
       mace_seed_predictions_all642.csv  (mol_id, pred_seed*, true_value)

Usage (Vast GPU, see run_mace_ensemble_off23.sh):
  python mace_freesolv/fold0_ensemble_off23.py --seeds 42,123,7,2024,999 --device cuda
  python mace_freesolv/fold0_ensemble_off23.py --seeds 42 --quick_test --device cuda

Outputs -> mace_freesolv/fold0_ensemble/ (seed_{S}/ + per-atom files).
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # repo root (contains freesolv_conformers.hdf5)
sys.path.insert(0, SCRIPT_DIR)

from config import EV_TO_KCAL  # noqa: E402
from data import MACEFreeSolvDataset, collate_mace, get_labels  # noqa: E402
from model import MACEFreeSolv  # noqa: E402
from train import WarmupWrapper, validate  # noqa: E402

FROZEN_SPLIT_DIR = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")
DEFAULT_HDF5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
DEFAULT_STAGE_A = os.path.join(
    REPO_ROOT, "mace_freesolv", "results_stage_a_scratch", "stage_a.pt")

CANDIDATE_NODE_KEYS = [
    "node_energy", "node_energies", "atomic_energies",
    "site_energy", "site_energies", "contributions", "atomic_energy",
]


# ---------------------------------------------------------------- helpers
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_split():
    """Load frozen fold-0 ids. Falls back to round-robin reconstruction."""
    tr_p = os.path.join(FROZEN_SPLIT_DIR, "train_ids.json")
    va_p = os.path.join(FROZEN_SPLIT_DIR, "val_ids.json")
    te_p = os.path.join(FROZEN_SPLIT_DIR, "test_ids.json")
    if os.path.exists(tr_p) and os.path.exists(va_p) and os.path.exists(te_p):
        with open(tr_p) as f:
            tr = json.load(f)
        with open(va_p) as f:
            va = json.load(f)
        with open(te_p) as f:
            te = json.load(f)
        print(f"[split] FROZEN fold-0: train={len(tr)} val={len(va)} test={len(te)}",
              flush=True)
        assert len(tr) == 411 and len(va) == 102 and len(te) == 129, \
            f"unexpected frozen sizes {len(tr)}/{len(va)}/{len(te)}"
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        return tr, va, te, "frozen"
    print("[split] WARNING: frozen split not found, reconstructing round-robin",
          flush=True)
    labels = get_labels()
    with h5py.File(DEFAULT_HDF5, "r") as f:
        mids = [m for m in f.keys()
                if m in labels and isinstance(labels[m].get("expt"), (int, float))]
    ex = np.array([labels[m]["expt"] for m in mids])
    order = [mids[i] for i in np.argsort(ex)]
    folds = [[] for _ in range(5)]
    for i, m in enumerate(order):
        folds[i % 5].append(m)
    te = folds[0]
    tv = [m for fi in range(1, 5) for m in folds[fi]]
    rng = np.random.RandomState(42)
    idx = np.arange(len(tv))
    rng.shuffle(idx)
    n_val = max(1, int(len(tv) * 0.2))
    va = [tv[i] for i in idx[:n_val]]
    tr = [tv[i] for i in idx[n_val:]]
    print(f"[split] reconstructed: train={len(tr)} val={len(va)} test={len(te)}",
          flush=True)
    return tr, va, te, "reconstructed"


def build_mace(device, train_ds, init_ckpt, cfg):
    m = MACEFreeSolv(
        model_size=cfg["model_size"], device=str(device),
        freeze_atomic_energies=cfg["freeze_atomic_energies"],
        target_mean=0.0, target_std=cfg.get("target_std"),
        use_lora=cfg.get("use_lora", False),
        fit_dataset=train_ds, init_checkpoint=init_ckpt,
    ).to(device)
    if cfg.get("freeze_interactions"):
        m.freeze_interactions()
    return m


def train_one_seed(seed, tr, va, te, device, out_root, cfg, init_ckpt, quick):
    """Train fold-0 only for one seed. Returns metrics dict."""
    seed_dir = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    set_seed(seed)
    tag = "foundation:OFF23" if init_ckpt is None else os.path.basename(init_ckpt)
    print(f"\n{'='*64}\n  SEED {seed}  (fold-0 only, init={tag})"
          f"\n{'='*64}", flush=True)

    # ---- datasets (precompute is the slow part; wrap with progress log)
    print(f"[seed {seed}] building datasets ...", flush=True)
    t0d = time.time()
    train_ds = MACEFreeSolvDataset(
        mol_ids=tr, r_max=cfg["r_max"], max_neighbors=cfg["max_neighbors"],
        targets_in_ev=True)
    val_ds = MACEFreeSolvDataset(
        mol_ids=va, r_max=cfg["r_max"], max_neighbors=cfg["max_neighbors"],
        targets_in_ev=True)
    test_ds = MACEFreeSolvDataset(
        mol_ids=te, r_max=cfg["r_max"], max_neighbors=cfg["max_neighbors"],
        targets_in_ev=True)
    print(f"[seed {seed}] datasets ready "
          f"({len(train_ds)}/{len(val_ds)}/{len(test_ds)}) in {time.time()-t0d:.0f}s",
          flush=True)

    from data import collate_mace as _coll
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=_coll, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            collate_fn=_coll, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=_coll, num_workers=0, pin_memory=pin)

    t_mean = float(np.mean([s[1] for s in train_ds.samples]))
    t_std = float(np.std([s[1] for s in train_ds.samples]))
    print(f"[seed {seed}] train target: mean={t_mean:.3f} std={t_std:.3f} kcal/mol",
          flush=True)
    cfg = dict(cfg)
    cfg["target_std"] = t_std

    model = build_mace(device, train_ds, init_ckpt, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg["patience"] // 2,
        min_lr=cfg["lr_min"])
    warm = WarmupWrapper(opt, cfg["warmup_epochs"], cfg["lr"])
    loss_fn = torch.nn.MSELoss()

    epochs = 2 if quick else cfg["epochs"]
    best_val, best_ep, stale = float("inf"), -1, 0
    ckpt = os.path.join(seed_dir, "model.pt")
    pbar_ep = tqdm(range(1, epochs + 1), desc=f"seed{seed} epochs", unit="ep")
    for epoch in pbar_ep:
        warm.step()
        # ---- train batches with progress bar
        model.train()
        tot, n = 0.0, 0
        pbar_b = tqdm(train_loader, desc=f"seed{seed} ep{epoch} train",
                      leave=False, unit="batch")
        for batch in pbar_b:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            y = batch.pop("y").view(-1)
            opt.zero_grad()
            pred = model(batch, compute_force=False).view(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
            pbar_b.set_postfix(loss=f"{tot/max(n,1):.4f}")
        if epoch > cfg["warmup_epochs"]:
            # need a val number for scheduler; compute below
            pass
        # ---- validate (batched, with bar)
        model.eval()
        vp, ve = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"seed{seed} ep{epoch} val",
                              leave=False, unit="batch"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                y = batch.pop("y").view(-1)
                pr = model(batch, compute_force=False).view(-1)
                vp.append(pr.cpu())
                ve.append(y.cpu())
        vp = torch.cat(vp).numpy()
        ve = torch.cat(ve).numpy()
        val_mae = float(np.mean(np.abs(vp - ve)))  # eV
        if epoch > cfg["warmup_epochs"]:
            sched.step(val_mae)
        val_kcal = val_mae * EV_TO_KCAL
        pbar_ep.set_postfix(best=f"{best_val*EV_TO_KCAL:.3f}", cur=f"{val_kcal:.3f}")
        print(f"  seed {seed} ep {epoch:3d}/{epochs} "
              f"loss={tot/max(n,1):.5f} valMAE={val_kcal:.3f} kcal "
              f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if val_mae < best_val:
            best_val, best_ep, stale = val_mae, epoch, 0
            model.save(ckpt)
            print(f"  [seed {seed}] * best checkpoint (val {val_kcal:.3f})", flush=True)
        else:
            stale += 1
        if stale >= cfg["patience"]:
            print(f"  [seed {seed}] early stop at ep {epoch}", flush=True)
            break

    # ---- test with best checkpoint
    model.load(ckpt)
    model.eval()
    tp, te_ = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"seed{seed} test", unit="batch"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            y = batch.pop("y").view(-1)
            pr = model(batch, compute_force=False).view(-1)
            tp.append(pr.cpu())
            te_.append(y.cpu())
    tp = torch.cat(tp).numpy() * EV_TO_KCAL
    te_ = torch.cat(te_).numpy() * EV_TO_KCAL
    mae = float(np.mean(np.abs(tp - te_)))
    rmse = float(np.sqrt(np.mean((tp - te_) ** 2)))
    print(f"[seed {seed}] TEST MAE={mae:.3f} RMSE={rmse:.3f} (best ep {best_ep})",
          flush=True)
    with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
        json.dump({"seed": seed, "n_train": len(tr), "n_val": len(va),
                   "n_test": len(te), "best_epoch": best_ep,
                   "best_val_mae_kcal": float(best_val * EV_TO_KCAL),
                   "test_mae_kcal": mae, "test_rmse_kcal": rmse,
                   "init_checkpoint": (init_ckpt if init_ckpt is not None
                                       else "foundation:OFF23")}, f, indent=2)
    with open(os.path.join(seed_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "pred_kcal", "true_kcal"])
        for m, p, t in zip(te, tp, te_):
            w.writerow([m, f"{p:.6f}", f"{t:.6f}"])
    return {"seed": seed, "test_mae": mae, "test_rmse": rmse}


# ------------------------------------------------- per-atom dump
@torch.no_grad()
def probe_node_key(model, device, hdf5_path):
    """Run one molecule through the RAW mace model and log every output key."""
    model.eval()
    with h5py.File(hdf5_path, "r") as f:
        mid = sorted(f.keys())[0]
        g = f[mid]
        z = torch.tensor(np.asarray(g["atNUM"]).reshape(-1), dtype=torch.long)
        pos = torch.tensor(np.asarray(g["atXYZ"], dtype=np.float32).reshape(-1, 3))
    n = z.size(0)
    from data import ELEMENT_TO_IDX, MACE_NUM_ELEMENTS, radius_graph
    na = torch.zeros(n, MACE_NUM_ELEMENTS)
    for i, zi in enumerate(z.tolist()):
        na[i, ELEMENT_TO_IDX[int(zi)]] = 1.0
    ei = radius_graph(pos, r=5.0, max_num_neighbors=32)
    batch = torch.zeros(n, dtype=torch.long)
    ptr = torch.tensor([0, n], dtype=torch.long)
    cell = (pos.max(0).values - pos.min(0).values + 20.0).diag().unsqueeze(0)
    data = {"positions": pos.to(device), "node_attrs": na.to(device),
            "edge_index": ei.to(device), "batch": batch.to(device),
            "ptr": ptr.to(device), "cell": cell.to(device),
            "shifts": torch.zeros(ei.size(1), 3).to(device),
            "unit_shifts": torch.zeros(ei.size(1), 3).to(device)}
    raw = model.model(data, compute_force=False, training=False)
    print(f"[probe] raw MACE output keys: {list(raw.keys())}", flush=True)
    for k, v in raw.items():
        if torch.is_tensor(v):
            print(f"[probe]   {k}: shape={tuple(v.shape)} dtype={v.dtype}",
                  flush=True)
    total = raw["energy"]
    print(f"[probe] energy shape={tuple(total.shape)} "
          f"value_eV={float(total.view(-1)[0]):.4f}", flush=True)
    for k in CANDIDATE_NODE_KEYS:
        if k in raw and torch.is_tensor(raw[k]):
            v = raw[k].view(-1)
            print(f"[probe] CANDIDATE {k}: n={v.numel()} sum_eV={float(v.sum()):.4f} "
                  f"vs total_eV={float(total.view(-1)[0]):.4f} "
                  f"diff={abs(float(v.sum())-float(total.view(-1)[0])):.2e}",
                  flush=True)
    return raw


@torch.no_grad()
def per_atom_single(model, device, z_np, pos_np, node_key, r_max=5.0):
    """Per-atom energies (kcal/mol) + total (kcal/mol) for ONE conformer."""
    from data import ELEMENT_TO_IDX, MACE_NUM_ELEMENTS, radius_graph
    z = torch.tensor(np.asarray(z_np).reshape(-1), dtype=torch.long)
    pos = torch.tensor(np.asarray(pos_np, dtype=np.float32).reshape(-1, 3))
    n = z.size(0)
    na = torch.zeros(n, MACE_NUM_ELEMENTS)
    for i, zi in enumerate(z.tolist()):
        j = ELEMENT_TO_IDX.get(int(zi))
        if j is None:
            return None, None  # unsupported element -> caller skips
        na[i, j] = 1.0
    ei = radius_graph(pos, r=r_max, max_num_neighbors=32)
    batch = torch.zeros(n, dtype=torch.long)
    ptr = torch.tensor([0, n], dtype=torch.long)
    cell = (pos.max(0).values - pos.min(0).values + 20.0).diag().unsqueeze(0)
    data = {"positions": pos.to(device), "node_attrs": na.to(device),
            "edge_index": ei.to(device), "batch": batch.to(device),
            "ptr": ptr.to(device), "cell": cell.to(device),
            "shifts": torch.zeros(ei.size(1), 3).to(device),
            "unit_shifts": torch.zeros(ei.size(1), 3).to(device)}
    raw = model.model(data, compute_force=False, training=False)
    if node_key not in raw:
        raise KeyError(f"node key '{node_key}' not in {list(raw.keys())}")
    P = raw[node_key].view(-1).detach().cpu().numpy() * EV_TO_KCAL
    E = float(raw["energy"].view(-1).detach().cpu().numpy()[0] * EV_TO_KCAL)
    return P.astype(np.float64), E


def dump_per_atom(seeds, tr, va, te, device, out_root, hdf5_path, r_max, quick):
    """Dump per-atom P for every seed. Returns paths of written files."""
    from data import collate_mace  # noqa: F401 (kept for parity)
    labels = get_labels()
    with h5py.File(hdf5_path, "r") as f:
        all_mols = [m for m in f.keys()
                    if m in labels and isinstance(labels[m].get("expt"), (int, float))]
    if quick:
        all_mols = all_mols[:20]
        print(f"[dump] QUICK mode: first 20 molecules only", flush=True)
    print(f"[dump] molecules: {len(all_mols)} | seeds: {seeds}", flush=True)

    # ---- load models once
    models = {}
    for s in tqdm(seeds, desc="load models", unit="seed"):
        m = MACEFreeSolv(model_size="medium", device=str(device), fit_refs=False)
        m.load(os.path.join(out_root, f"seed_{s}", "model.pt"))
        m.to(device)
        m.eval()
        models[s] = m

    # ---- probe node key on seed[0]
    raw_probe = probe_node_key(models[seeds[0]], device, hdf5_path)
    node_key = None
    for k in CANDIDATE_NODE_KEYS:
        if k in raw_probe and torch.is_tensor(raw_probe[k]):
            v = raw_probe[k].view(-1)
            tot = raw_probe["energy"].view(-1)
            if v.numel() > 1 and abs(float(v.sum()) - float(tot[0])) < 1e-3:
                node_key = k
                break
    if node_key is None:
        # fall back to first tensor key with n_atoms length
        raise RuntimeError(
            f"Could not identify node-energy key. Keys={list(raw_probe.keys())}. "
            "Paste the [probe] lines above and stop.")
    print(f"[dump] using node key '{node_key}'", flush=True)

    # ---- per-seed dump (molecule loop with progress bar + gate)
    per_seed = {}
    for s in seeds:
        m = models[s]
        P_all, E_all, N_all, skip = {}, {}, {}, []
        max_gate = 0.0
        with h5py.File(hdf5_path, "r") as f:
            for mid in tqdm(all_mols, desc=f"dump seed{s}", unit="mol"):
                if mid not in f:
                    skip.append(mid)
                    continue
                g = f[mid]
                z = np.asarray(g["atNUM"]).reshape(-1)
                xyz = np.asarray(g["atXYZ"], dtype=np.float32)
                # stored file is single-conformer (N,3) or (1,N,3)-like; take first
                if xyz.ndim == 3:
                    xyz = xyz[0]
                P, E = per_atom_single(m, device, z, xyz, node_key, r_max)
                if P is None:
                    skip.append(mid)
                    continue
                gate = abs(float(P.sum()) - E)
                max_gate = max(max_gate, gate)
                P_all[mid] = P
                E_all[mid] = E
                N_all[mid] = len(P)
        print(f"[dump seed {s}] kept={len(P_all)} skipped={len(skip)} "
              f"max|sum(P)-E|={max_gate:.2e} kcal/mol", flush=True)
        assert max_gate < 1e-3, f"node-sum gate failed ({max_gate:.2e})"
        if skip:
            print(f"[dump seed {s}] skipped e.g. {skip[:5]} (MACE 10-elem vocab)",
                  flush=True)
        mu_T = float(np.concatenate(
            [P_all[m] for m in tr if m in P_all]).mean())
        out = {"seed": s, "node_key": node_key, "P_all": P_all, "E_all": E_all,
               "N_all": N_all, "skipped": skip, "mu_T_kcal": mu_T,
               "P_train": {m: P_all[m] for m in tr if m in P_all},
               "P_val": {m: P_all[m] for m in va if m in P_all},
               "P_test": {m: P_all[m] for m in te if m in P_all}}
        pk = os.path.join(out_root, f"peratom_mace_seed{s}.pkl")
        with open(pk, "wb") as fh:
            pickle.dump(out, fh)
        print(f"[dump seed {s}] mu_T={mu_T:+.4f} -> {pk}", flush=True)
        per_seed[s] = out

    # ---- joint CSVs for direct GIMS reuse
    common = [m for m in all_mols if all(m in per_seed[s]["P_all"] for s in seeds)]
    print(f"[dump] common molecules across seeds: {len(common)}", flush=True)
    node_csv = os.path.join(out_root, "mace_node_contributions.csv")
    with open(node_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "atom_idx"] + [f"P_seed{s}" for s in seeds])
        for mid in tqdm(common, desc="write node csv", unit="mol"):
            Pstack = np.stack([per_seed[s]["P_all"][mid] for s in seeds], axis=1)
            for a in range(Pstack.shape[0]):
                w.writerow([mid, a] + [f"{v:.6f}" for v in Pstack[a]])
    pred_csv = os.path.join(out_root, "mace_seed_predictions_all642.csv")
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "true_value"] + [f"pred_seed{s}" for s in seeds])
        for mid in tqdm(common, desc="write pred csv", unit="mol"):
            w.writerow([mid, f"{labels[mid]['expt']:.6f}"] +
                       [f"{per_seed[s]['E_all'][mid]:.6f}" for s in seeds])
    print(f"[dump] wrote {node_csv}\n[dump] wrote {pred_csv}", flush=True)
    return node_csv, pred_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,7,2024,999")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--r_max", type=float, default=5.0)
    ap.add_argument("--max_neighbors", type=int, default=32)
    ap.add_argument("--model_size", default="medium")
    ap.add_argument("--freeze_atomic_energies", action="store_true", default=True)
    ap.add_argument("--no_freeze_atomic_energies", action="store_false",
                    dest="freeze_atomic_energies")
    ap.add_argument("--freeze_interactions", action="store_true", default=False)
    ap.add_argument("--foundation", default="medium",
                    choices=["none", "small", "medium", "large"],
                    help="Released OFF23 size to fine-tune ('none' uses --init_checkpoint)")
    ap.add_argument("--init_checkpoint", default=None)
    ap.add_argument("--hdf5", default=DEFAULT_HDF5)
    ap.add_argument("--output_dir",
                    default=os.path.join(REPO_ROOT, "mace_freesolv",
                                         "fold0_ensemble_off23"))
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_dump", action="store_true")
    ap.add_argument("--quick_test", action="store_true")
    a = ap.parse_args()

    device = torch.device(
        a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[run] device={device} | torch={torch.__version__} "
          f"| cuda={torch.version.cuda} | gpus={torch.cuda.device_count()}",
          flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[run] seeds={seeds} quick={a.quick_test} "
          f"skip_train={a.skip_train} skip_dump={a.skip_dump}", flush=True)
    init_ckpt = None if a.foundation != "none" else a.init_checkpoint
    init_tag = (f"foundation:OFF23-{a.foundation}" if a.foundation != "none"
                else os.path.basename(a.init_checkpoint))
    print(f"[run] hdf5={a.hdf5}\n[run] init={init_tag}\n"
          f"[run] out={a.output_dir}", flush=True)
    assert os.path.exists(a.hdf5), f"missing HDF5 {a.hdf5}"
    if init_ckpt is not None:
        assert os.path.exists(init_ckpt), \
            f"missing Stage-A checkpoint {init_ckpt}"
    os.makedirs(a.output_dir, exist_ok=True)

    cfg = {"model_size": a.model_size, "r_max": a.r_max,
           "max_neighbors": a.max_neighbors, "batch_size": a.batch_size,
           "lr": a.lr, "lr_min": a.lr_min, "weight_decay": a.weight_decay,
           "epochs": a.epochs, "patience": a.patience,
           "warmup_epochs": a.warmup_epochs,
           "freeze_atomic_energies": a.freeze_atomic_energies,
           "freeze_interactions": a.freeze_interactions,
           "use_lora": False}

    tr, va, te, src = load_frozen_split()
    with open(os.path.join(a.output_dir, "split_used.json"), "w") as f:
        json.dump({"source": src, "n_train": len(tr), "n_val": len(va),
                   "n_test": len(te)}, f, indent=2)

    t0 = time.time()
    if not a.skip_train:
        summary = {}
        for s in tqdm(seeds, desc="OVERALL seeds", unit="seed"):
            r = train_one_seed(s, tr, va, te, device, a.output_dir, cfg,
                               init_ckpt, a.quick_test)
            summary[str(s)] = r
        with open(os.path.join(a.output_dir, "ensemble_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[train] all seeds done in {(time.time()-t0)/60:.1f} min", flush=True)
    if not a.skip_dump:
        dump_per_atom(seeds, tr, va, te, device, a.output_dir, a.hdf5,
                      a.r_max, a.quick_test)
    print(f"[done] total {(time.time()-t0)/3600:.2f} h -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()

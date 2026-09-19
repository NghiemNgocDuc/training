"""Retry the failed FreeSolv Stage-3 seeds (7, 2024) + energy-spike recheck.

Same recipe as train_seed.py (fold-0 archived splits, Stage-1+2 ckpt,
Adam lr=1e-4/wd=1e-5/batch=8/<=200ep/patience=30, MSE-in-eV, clip 10,
all RNG pinned to --seed) with two differences:
  * outputs go to results_seeds_retry/ (finetuned_seed{S}_retry.pt);
    results_seeds/ originals are NEVER overwritten (they are the control).
  * after training, each retry seed gets the spike check: stored-conformer
    prediction vs mean-of-5-fresh-conformer prediction per test molecule.
    PASS = max abs gap <= 5.0 kcal/mol (retry-chosen threshold, reported
    explicitly; counts + worst molecules saved).

Honesty note: same seed + same code + same data is deterministic, so a
retry is expected to reproduce the original weights nearly exactly. The
determinism check below quantifies that. A retry can only "work this
time" via environment nondeterminism (different GPU/driver/cuDNN) --
which is itself a finding worth logging.

Usage (vast.ai GPU, detached):
  nohup python retry_freesolv_seeds.py --seeds 7 2024 > retry.log 2>&1 &
  tail -f retry.log
Smoke (CPU, ~2 min):
  python retry_freesolv_seeds.py --seeds 7 --quick --max_mols 12
Every loop has a tqdm progress bar.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# common_io does "from freesolv_dataset import ..." at call time; make the
# self-contained inputs/ copy (plus repo fallbacks) importable everywhere,
# not just on machines with PYTHONPATH set.
for _p in (os.path.join(HERE, "inputs"),
           os.path.join(REPO_ROOT, "expdb_vast"),
           os.path.join(REPO_ROOT, "aqm-spice2", "freesolv")):
    if _p not in sys.path:
        sys.path.append(_p)
import common_io as cio

OUT_SUBDIR = "results_seeds_retry"
SPIKE_THR = 5.0  # kcal/mol, retry-chosen; reported, not paper-recorded


def train_one(seed, epochs, patience, batch_size, lr, quick, max_mols, device, out_dir):
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    cio.set_all_seeds(seed)
    labels = cio.load_labels()
    train_ids = json.load(open(cio.path_split("train")))
    val_ids = json.load(open(cio.path_split("val")))
    if max_mols is not None:
        rng = np.random.RandomState(0)
        train_ids = [train_ids[i] for i in sorted(
            rng.choice(len(train_ids), min(max_mols, len(train_ids)), replace=False))]
        val_ids = val_ids[:max(1, min(len(val_ids), 4))]
        print(f"[retry {seed}] max_mols cap: train={len(train_ids)} val={len(val_ids)} (smoke only)",
              flush=True)
    h5_free = cio.path_freesolv_h5()
    print(f"[retry {seed}] device={device} train={len(train_ids)} val={len(val_ids)}", flush=True)

    ck_path = os.path.join(out_dir, f"finetuned_seed{seed}_retry.pt")

    class SimpleDS:
        def __init__(self, ids):
            self.ids = ids
            self._c = {}

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            import h5py
            mid = self.ids[i]
            if mid not in self._c:
                with h5py.File(h5_free, "r") as f:
                    g = f[mid]
                    d = Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                             pos=torch.tensor(g["atXYZ"][...], dtype=torch.float))
                self._c[mid] = d.clone()
            d = self._c[mid].clone()
            d.y_dG = torch.tensor([labels[mid]["expt"]], dtype=torch.float)
            return d

    train_loader = DataLoader(SimpleDS(train_ids), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SimpleDS(val_ids), batch_size=batch_size, shuffle=False)

    model = cio.build_model(device)
    state = torch.load(cio.path_stage2(), map_location=device, weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    print(f"[retry {seed}] init from stage2_correction.pt (random-init {len(missing)} params)",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, patience // 2), min_lr=1e-6)
    mse = torch.nn.MSELoss()

    def eval_loader(loader):
        model.eval()
        P, E = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = cio.one_hot_x(data.z, device)
                pred = model(x, data.pos, data.batch).view(-1) * cio.EV_TO_KCAL
                y = data.y_dG.view(-1).to(device)
                ok = ~torch.isnan(y)
                P.append(pred[ok].cpu())
                E.append(y[ok].cpu())
        p = torch.cat(P).numpy()
        e = torch.cat(E).numpy()
        return float(np.mean(np.abs(p - e))), float(np.sqrt(np.mean((p - e) ** 2)))

    if quick:
        epochs = min(epochs, 2)
    best_val, best_epoch, stale = float("inf"), -1, 0
    t0 = time.time()
    ep_bar = tqdm(range(1, epochs + 1), desc=f"[retry {seed}] epochs", unit="epoch")
    for epoch in ep_bar:
        model.train()
        for data in tqdm(train_loader, desc=f"[retry {seed}] ep{epoch} train",
                         leave=False, unit="batch"):
            data = data.to(device)
            x = cio.one_hot_x(data.z, device)
            pred = model(x, data.pos, data.batch).view(-1)
            y = data.y_dG.view(-1).to(device) / cio.EV_TO_KCAL
            ok = ~torch.isnan(y)
            if ok.sum() == 0:
                continue
            loss = mse(pred[ok], y[ok])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        val_mae, val_rmse = eval_loader(val_loader)
        sched.step(val_mae)
        ep_bar.set_postfix(val_mae=f"{val_mae:.3f}", best=f"{best_val:.3f}")
        if val_mae < best_val:
            best_val, best_epoch, stale = val_mae, epoch, 0
            torch.save(model.state_dict(), ck_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"[retry {seed}] early stop at epoch {epoch}", flush=True)
                break
    meta = {"seed": seed, "best_val_mae": best_val, "best_epoch": best_epoch,
            "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, f"train_meta_seed{seed}_retry.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[retry {seed}] DONE best val MAE {best_val:.3f} @ epoch {best_epoch} -> {ck_path}",
          flush=True)
    return ck_path


def fresh_confs(smiles, n=5):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    if not rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params):
        return None
    try:
        if rdForceFieldHelpers.MMFFGetMoleculeProperties(mol) is not None:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    except Exception:
        pass
    import torch
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return [(z.clone(),
             torch.tensor(mol.GetConformer(i).GetPositions(), dtype=torch.float))
            for i in range(mol.GetNumConformers())]


def spike_check(seed, ck_path, orig_ck_path, max_mols, device):
    """Stored-vs-fresh per-seed gap + retry-vs-original determinism gap."""
    import h5py
    import torch
    labels = cio.load_labels()
    test_ids = json.load(open(cio.path_split("test")))
    if max_mols is not None:
        test_ids = test_ids[:max_mols]

    def load(ckpt):
        m = cio.build_model(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True),
                          strict=False)
        m.eval()
        return m

    m_new = load(ck_path)
    m_old = load(orig_ck_path) if os.path.exists(orig_ck_path) else None

    def mol_E(model, z, pos):
        with torch.no_grad():
            x = cio.one_hot_x(z.to(device), device)
            return float(model(x, pos.to(device),
                               torch.zeros(len(z), dtype=torch.long).to(device)
                               ).view(-1).cpu() * cio.EV_TO_KCAL)

    gaps, det_gaps, fails = [], [], []
    with h5py.File(cio.path_freesolv_h5(), "r") as f:
        for mid in tqdm(test_ids, desc=f"[spike {seed}]", unit="mol"):
            g = f[mid]
            z0 = torch.tensor(g["atNUM"][...], dtype=torch.long)
            p0 = torch.tensor(g["atXYZ"][...], dtype=torch.float)
            e_stored = mol_E(m_new, z0, p0)
            confs = fresh_confs(labels[mid]["smiles"])
            if not confs:
                continue
            e_fresh = float(np.mean([mol_E(m_new, z, p) for z, p in confs]))
            gap = abs(e_fresh - e_stored)
            gaps.append(gap)
            if gap > SPIKE_THR:
                fails.append({"mol": mid, "gap": round(gap, 3),
                              "stored": round(e_stored, 3), "fresh": round(e_fresh, 3)})
            if m_old is not None:
                det_gaps.append(abs(mol_E(m_new, z0, p0) - mol_E(m_old, z0, p0)))
    gaps = np.array(gaps)
    rep = {"seed": seed, "n_test": len(gaps),
           "max_gap": round(float(gaps.max()), 3),
           "mean_gap": round(float(gaps.mean()), 3),
           "n_over_thr": len(fails), "thr": SPIKE_THR,
           "verdict": "PASS" if len(fails) == 0 else "FAIL",
           "worst": sorted(fails, key=lambda r: -r["gap"])[:10]}
    if det_gaps:
        rep["determinism_max_gap_vs_orig"] = round(float(np.max(det_gaps)), 6)
    print(f"[spike {seed}] max_gap={rep['max_gap']:.3f} mean={rep['mean_gap']:.3f} "
          f"over_thr={rep['n_over_thr']}/{rep['n_test']} -> {rep['verdict']}", flush=True)
    if det_gaps:
        print(f"[spike {seed}] retry-vs-orig max_gap={rep['determinism_max_gap_vs_orig']:.6f} "
              f"(~0 = deterministic rerun)", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 2024])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max_mols", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--orig_dir", default=None,
                    help="dir with original finetuned_seed{S}.pt (default: auto-detect)")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} seeds={args.seeds} config={vars(args)}", flush=True)
    out_dir = os.path.join(HERE, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    orig_dir = args.orig_dir
    if orig_dir is None:
        for cand in (os.path.join(HERE, "results_seeds"),
                     os.path.join(REPO_ROOT, "expdb_vast", "results_seeds")):
            if os.path.exists(cand):
                orig_dir = cand
                break
        else:
            orig_dir = os.path.join(HERE, "results_seeds")
    print(f"orig control dir: {orig_dir}", flush=True)
    summary = {"seeds": args.seeds, "spike_thr": SPIKE_THR, "results": {}}
    for seed in tqdm(args.seeds, desc="retry seeds", unit="seed"):
        ck = os.path.join(out_dir, f"finetuned_seed{seed}_retry.pt")
        if os.path.exists(ck):
            print(f"[retry {seed}] ckpt exists, skipping train (--retrain N/A; delete to redo)",
                  flush=True)
        else:
            train_one(seed, args.epochs, args.patience, args.batch_size,
                      args.lr, args.quick, args.max_mols, device, out_dir)
        rep = spike_check(seed, ck,
                          os.path.join(orig_dir, f"finetuned_seed{seed}.pt"),
                          args.max_mols, device)
        with open(os.path.join(out_dir, f"spike_report_seed{seed}.json"), "w") as f:
            json.dump(rep, f, indent=2)
        summary["results"][str(seed)] = rep
    with open(os.path.join(out_dir, "retry_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for s, r in summary["results"].items():
        print(f"seed {s}: {r['verdict']} (max_gap {r['max_gap']:.3f}, "
              f"det_gap {r.get('determinism_max_gap_vs_orig', float('nan')):.6f})", flush=True)
    print("STOP: originals untouched; see results_seeds_retry/retry_summary.json", flush=True)


if __name__ == "__main__":
    main()

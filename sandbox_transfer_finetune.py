"""Sandboxed transfer fine-tune: FlexiSol-water (297) + Guthrie-subset (445).

Mirrors FreeSolv Stage-3 exactly, per paper:
  arch: DimeNetPlusSE (128h, 3 blocks, 7 sph / 6 rad, cutoff 6.0,
        32 nbrs, envelope 5, use_multi_aggregate=False, use_se=False)
  Stage-3: Adam lr=1e-4, wd=1e-5, batch=8, <=200 epochs,
        early-stop patience=30 on VAL MAE, MSE-in-eV, clip 10.0
  seeds: 42, 123, 999 from Stage-1+2 correction ckpt (no retrain of 1+2)
  splits: sort by exp value, round-robin 5 folds, test=fold0 (20%),
        val=20% of train_val (=16% overall, seed 42) -> ~64/16/20
  GIMS: tau^2 grid log-spaced 1e-8*var(s2) .. 100*var(s2) on VAL,
        mu_T from TRAIN atoms only, uniform matched as mean-lambda on VAL
  eval: MAE/RMSE/R2/Kendall + dMAE vs raw, 10k paired bootstrap 95% CI
  conformers: RDKit ETKDGv3 seed 42 + MMFF94, 1 stored train / TTA-5 test

Frozen Table-3 numbers are restated, never recomputed here.
This script ADDS the fine-tuned column next to them.

Usage (sandbox smoke, CPU, ~2 min):
  python sandbox_transfer_finetune.py --dataset both --sandbox --make_splits_only
  python sandbox_transfer_finetune.py --dataset flexisol --sandbox
Usage (vast.ai full run, GPU, detached):
  nohup python sandbox_transfer_finetune.py --dataset both --device cuda \
    --outdir out_transfer > transfer.log 2>&1 &
  tail -f transfer.log
Every loop has a tqdm progress bar.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(REPO, "expdb_vast"),
          os.path.join(REPO, "aqm-spice2", "freesolv"),
          os.path.join(REPO, "aqm-spice2", "freesolv", "experimental_uncertainty_refine")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from DimeModels import DimeNetPlusSE
    from element_vocab import NUM_ELEMENTS, build_one_hot
except ImportError as e:
    print(f"FATAL: cannot import model/vocab: {e}", file=sys.stderr)
    sys.exit(1)

EV_TO_KCAL = 23.0605
SEEDS_DEFAULT = [42, 123, 999]
# Frozen Table-3 restatement (paper, DimeNet++ frozen, never touched here)
FROZEN = {
    "flexisol": {"n": 297, "raw": 4.80, "gims": 3.98, "vw": 3.99, "uniform": 4.00},
    "guthrie": {"n": 445, "raw": 3.99, "gims": 3.13, "vw": 3.12, "uniform": 3.15},
}


def build_model(device):
    m = DimeNetPlusSE(
        hidden_channels=128, in_channels=NUM_ELEMENTS, out_channels=1,
        num_blocks=3, int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=6.0, max_num_neighbors=32,
        envelope_exponent=5, num_before_skip=1, num_after_skip=2,
        num_output_layers=3, is_energy=True,
        use_multi_aggregate=False, use_se=False,
    ).to(device)
    return m


def gen_one_conformer(smiles, seed=42):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    if not rdDistGeom.EmbedMolecule(mol, params):
        # fallback single attempt
        if mol.GetNumConformers() == 0:
            return None
    try:
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if props is not None:
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                   dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return z, pos


def load_flexisol():
    import pandas as pd
    ref = os.path.join(REPO, "flexisol", "data", "references", "dgsolv-references.csv")
    if not os.path.exists(ref):
        print(f"FATAL: FlexiSol refs missing: {ref}\n  run: python flexisol_sandbox/fetch_flexisol.py --dest flexisol_sandbox/data/flexisol_repo",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ref)
    w = df[df["Solvent"] == "water"].reset_index(drop=True)
    recs = []
    for _, r in w.iterrows():
        recs.append({"id": str(r["FlexiSol Name"]),
                     "smiles": str(r["SMILES"]),
                     "exp": float(r["Value (\\kcalpmole)"])})
    print(f"FlexiSol-water loaded: n={len(recs)} (expect 297)")
    return recs


def load_guthrie(csv_path):
    import pandas as pd
    if csv_path is None:
        csv_path = os.path.join(REPO, "guthrie_subset_445.csv")
    if not os.path.exists(csv_path):
        print(f"FATAL: Guthrie-subset CSV missing: {csv_path}\n"
              f"  Expected columns: id,smiles,exp[,pubchem_cid]\n"
              f"  Build it as final-kcal slice, CID-deduped vs FreeSolv (paper Sec. 5.1),\n"
              f"  then rerun with --guthrie_csv <path>. Stopping, not guessing.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    for c in ("smiles", "exp"):
        if c not in df.columns:
            print(f"FATAL: {csv_path} needs columns id,smiles,exp (got {list(df.columns)})",
                  file=sys.stderr)
            sys.exit(1)
    ids = df["id"].astype(str) if "id" in df.columns else [f"g{i}" for i in range(len(df))]
    recs = [{"id": str(i), "smiles": str(s), "exp": float(e)}
            for i, s, e in zip(ids, df["smiles"], df["exp"])]
    print(f"Guthrie-subset loaded: n={len(recs)} (expect 445)")
    if len(recs) != 445:
        print(f"WARNING: n={len(recs)} != 445 paper count. Continue but flag in summary.")
    # CID dedup check vs FreeSolv if cid column present
    if "pubchem_cid" in df.columns:
        try:
            from freesolv_dataset import download_freesolv_data, load_freesolv_labels
            jp, _ = download_freesolv_data(os.path.join(REPO, "Data", "FreeSolv"))
            fl = load_freesolv_labels(jp)
            print("  (FreeSolv labels loaded for CID cross-check; enforce zero overlap offline)")
        except Exception as e:
            print(f"  CID cross-check skipped: {e}")
    return recs


def make_split(recs, outdir, tag):
    """Sorted-by-exp round-robin 5 folds; test=fold0; val=20% of rest (seed42)."""
    exps = np.array([r["exp"] for r in recs])
    order = np.argsort(exps)
    srt = [recs[i] for i in order]
    folds = [[] for _ in range(5)]
    for i, r in enumerate(srt):
        folds[i % 5].append(r)
    test = folds[0]
    train_val = [r for f in folds[1:] for r in f]
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_val))
    rng.shuffle(idx)
    n_val = max(1, int(len(train_val) * 0.2))
    val = [train_val[i] for i in idx[:n_val]]
    train = [train_val[i] for i in idx[n_val:]]
    for name, split in (("train", train), ("val", val), ("test", test)):
        e = np.array([r["exp"] for r in split])
        print(f"  [{tag}/{name}] n={len(split)} mean={e.mean():+.3f} std={e.std():.3f}")
    os.makedirs(outdir, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        p = os.path.join(outdir, f"{tag}_{name}_ids.json")
        with open(p, "w") as f:
            json.dump([r["id"] for r in split], f)
        print(f"  saved {p}")
    # full label map for inspect
    with open(os.path.join(outdir, f"{tag}_labels.json"), "w") as f:
        json.dump({r["id"]: {"smiles": r["smiles"], "exp": r["exp"]} for r in recs}, f)
    return train, val, test


def build_hdf5(recs, h5_path, n_confs_train=1, n_confs_test=5):
    """Cache RDKit conformers once. Train uses conf0; test averages TTA-5."""
    if os.path.exists(h5_path):
        print(f"  conformers cached: {h5_path}, skipping build")
        return
    with h5py.File(h5_path, "w") as f:
        for r in tqdm(recs, desc="gen conformers", unit="mol"):
            from rdkit import Chem
            from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            params.pruneRmsThresh = 0.5
            n_try = max(n_confs_train, n_confs_test)
            cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_try, params=params)
            if not cids:
                continue
            try:
                if rdForceFieldHelpers.MMFFGetMoleculeProperties(mol) is not None:
                    rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
            except Exception:
                pass
            z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
            xyz = np.array([list(mol.GetConformer(i).GetPositions())
                            for i in range(mol.GetNumConformers())], dtype=np.float32)
            g = f.create_group(r["id"])
            g.create_dataset("atNUM", data=z)
            g.create_dataset("atXYZ", data=xyz)
    print(f"  saved {h5_path}")


class ConfDataset(torch.utils.data.Dataset):
    def __init__(self, ids, labels, h5_path, conf_idx=0, tta=False, n_tta=5):
        self.ids = ids
        self.labels = labels
        self.h5 = h5_path
        self.conf_idx = conf_idx
        self.tta = tta
        self.n_tta = n_tta
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mid = self.ids[i]
        if mid not in self._cache:
            with h5py.File(self.h5, "r") as f:
                g = f[mid]
                self._cache[mid] = (g["atNUM"][...], g["atXYZ"][...])
        z, xyz = self._cache[mid]
        if self.tta:
            n = min(self.n_tta, xyz.shape[0])
            # TTA entries expanded at eval time; here return conf0, expansion done in eval_tta
            c = 0
        else:
            c = min(self.conf_idx, xyz.shape[0] - 1)
        d = Data(z=torch.tensor(z, dtype=torch.long),
                 pos=torch.tensor(xyz[c], dtype=torch.float))
        d.mol_id = mid
        d.y_dG = torch.tensor([self.labels[mid]], dtype=torch.float)
        return d


def train_seed(seed, ckpt_init, train_ids, val_ids, labels, h5, device, outdir,
               epochs, patience, lr, wd, batch_size, sandbox):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(device)
    st = torch.load(ckpt_init, map_location=device)
    model.load_state_dict(st, strict=False)
    print(f"seed {seed}: warm-started from {ckpt_init}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, patience // 2), min_lr=1e-6)
    mse = torch.nn.MSELoss()
    tr_ds = ConfDataset(train_ids, labels, h5)
    va_ds = ConfDataset(val_ids, labels, h5)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    if sandbox:
        epochs = min(epochs, 2)
    best, best_state, stale, best_ep = float("inf"), None, 0, -1
    ep_bar = tqdm(range(1, epochs + 1), desc=f"seed{seed} epochs", unit="epoch")
    for ep in ep_bar:
        model.train()
        for data in tqdm(tr_ld, desc=f"seed{seed} ep{ep} train", leave=False, unit="batch"):
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            tgt = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            loss = mse(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        p, e = [], []
        with torch.no_grad():
            for data in tqdm(va_ld, desc=f"seed{seed} ep{ep} val", leave=False, unit="batch"):
                data = data.to(device)
                x = build_one_hot(data, device)
                p.append((model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL).cpu())
                e.append(data.y_dG.view(-1).cpu())
        p = torch.cat(p).numpy()
        e = torch.cat(e).numpy()
        mae = float(np.mean(np.abs(p - e)))
        sched.step(mae)
        ep_bar.set_postfix(val_mae=f"{mae:.3f}", best=f"{best:.3f}")
        if mae < best:
            best, best_ep, stale = mae, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(outdir, f"finetuned_seed{seed}.pt"))
        else:
            stale += 1
            if stale >= patience:
                print(f"seed {seed}: early stop at epoch {ep} (best {best:.3f} @ {best_ep})")
                break
    if stale >= patience and best_state is None:
        print(f"WARNING seed {seed}: never improved (possible divergence like 7/2024). Flagged.")
    return best, best_ep


def per_atom_predict(model, z, xyz_list, device):
    """Return per-seed per-atom P (kcal/mol) stacked over TTA confs then averaged.

    Returns P_mean_atoms (N,), E_raw (TTA-averaged molecular sum).
    """
    from element_vocab import ELEMENT_TO_IDX
    P_confs, E_confs = [], []
    orig = model.is_energy
    for xyz in xyz_list:
        x = np.zeros((len(z), NUM_ELEMENTS), dtype=np.float32)
        for i, zz in enumerate(z):
            if int(zz) in ELEMENT_TO_IDX:
                x[i, ELEMENT_TO_IDX[int(zz)]] = 1.0
        xt = torch.tensor(x).to(device)
        pt = torch.tensor(xyz, dtype=torch.float).to(device)
        b = torch.zeros(len(z), dtype=torch.long).to(device)
        model.is_energy = False
        with torch.no_grad():
            P = model(xt, pt, b).detach().cpu().numpy().flatten() * EV_TO_KCAL
        P_confs.append(P)
        E_confs.append(float(P.sum()))
    model.is_energy = orig
    return np.mean(P_confs, axis=0), float(np.mean(E_confs)), len(z)


def run_dataset(tag, recs, args, device):
    print(f"\n{'=' * 60}\n DATASET {tag}: n={len(recs)}\n{'=' * 60}")
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    labels = {r["id"]: r["exp"] for r in recs}
    # 1) splits first (inspectable, before any training)
    id2rec = {r["id"]: r for r in recs}
    split_ids = {}
    for name in ("train", "val", "test"):
        p = os.path.join(outdir, f"{tag}_{name}_ids.json")
        if os.path.exists(p):
            split_ids[name] = json.load(open(p))
            print(f"  reusing split {p} n={len(split_ids[name])}")
    if len(split_ids) < 3:
        train_r, val_r, test_r = make_split(recs, outdir, tag)
        split_ids = {"train": [r["id"] for r in train_r],
                     "val": [r["id"] for r in val_r],
                     "test": [r["id"] for r in test_r]}
    if args.make_splits_only:
        print("  --make_splits_only: stopping before training. Inspect IDs first.")
        return None
    # conformers
    h5 = os.path.join(outdir, f"{tag}_conformers.hdf5")
    build_hdf5(recs, h5)
    # 2) fine-tune ensemble
    for seed in tqdm(args.seeds, desc=f"{tag} seeds", unit="seed"):
        ckpt = os.path.join(outdir, f"finetuned_seed{seed}.pt")
        if os.path.exists(ckpt) and not args.retrain:
            print(f"  seed {seed}: ckpt exists, skipping (--retrain to force)")
            continue
        train_seed(seed, args.correction_ckpt, split_ids["train"], split_ids["val"],
                   labels, h5, device, outdir, args.epochs, args.patience,
                   args.lr, args.weight_decay, args.batch_size, args.sandbox)
    # 3) per-atom inference on val+test for mu_T + tau calibration
    models = {}
    for seed in args.seeds:
        m = build_model(device)
        m.load_state_dict(torch.load(os.path.join(outdir, f"finetuned_seed{seed}.pt"),
                                     map_location=device), strict=False)
        m.eval()
        models[seed] = m
    with h5py.File(h5, "r") as f:
        h5_ids = set(f.keys())
    missing = [i for i in split_ids["train"] + split_ids["val"] + split_ids["test"]
               if i not in h5_ids]
    if missing:
        print(f"WARNING: {len(missing)} split molecules missing conformers (MMFF fail?), e.g. {missing[:5]}")

    def infer_ids(ids, tta):
        P, E, N, Z = {}, {}, {}, {}
        for mid in tqdm(ids, desc="per-atom infer", unit="mol"):
            with h5py.File(h5, "r") as f:
                g = f[mid]
                z = g["atNUM"][...]
                xyz = g["atXYZ"][...]
            n = min(5, xyz.shape[0]) if tta else 1
            per_seed_P, per_seed_E = [], []
            for seed, m in models.items():
                Pm, Em, Nm = per_atom_predict(m, z, [xyz[i] for i in range(n)], device)
                per_seed_P.append(Pm)
                per_seed_E.append(Em)
            P[mid] = np.stack(per_seed_P, axis=1)  # N x K
            E[mid] = float(np.mean(per_seed_E))
            N[mid] = int(len(z))
            Z[mid] = z
        return P, E, N

    P_val, E_val, N_val = infer_ids(split_ids["val"], tta=False)
    P_test, E_test, N_test = infer_ids(split_ids["test"], tta=True)
    # mu_T from TRAIN atoms only
    P_tr, _, _ = infer_ids(split_ids["train"], tta=False)
    mu_per_seed = []
    for k, seed in enumerate(args.seeds):
        mu_per_seed.append(float(np.mean([P_tr[m][:, k].mean() for m in P_tr])))
    mu_mean = float(np.mean(mu_per_seed))
    print(f"  mu_T per-seed {[f'{v:.4f}' for v in mu_per_seed]} mean {mu_mean:.4f} (train atoms only)")
    # tau grid on VAL
    s2_val = np.concatenate([P_val[m].var(axis=1, ddof=1) for m in P_val])
    v = float(s2_val.var())
    grid = np.logspace(np.log10(1e-8 * v), np.log10(100 * v), 25)
    exp_val = {m: labels[m] for m in P_val}

    def val_mae_for_tau(tau2):
        errs = []
        for m in P_val:
            Pv = P_val[m]
            lam = Pv.var(axis=1, ddof=1) / (Pv.var(axis=1, ddof=1) + tau2)
            Lam = float(lam.mean())
            raw = float(np.mean([Pv[:, k].sum() for k in range(Pv.shape[1])]))
            g = (1 - Lam) * raw + Lam * N_val[m] * mu_mean
            errs.append(abs(g - exp_val[m]))
        return float(np.mean(errs))

    maes = [val_mae_for_tau(t) for t in tqdm(grid, desc="tau grid", unit="tau")]
    bi = int(np.argmin(maes))
    tau_star = float(grid[bi])
    print(f"  tau2* = {tau_star:.3e} valMAE {maes[bi]:.3f} "
          f"(grid {grid[0]:.2e}..{grid[-1]:.2e})")
    if bi in (0, len(grid) - 1):
        print("  WARNING: tau* on grid EDGE. Widen grid before trusting.")
    # uniform matched = mean lambda on VAL at tau*
    lam_vals = []
    for m in P_val:
        Pv = P_val[m]
        lam_vals.append((Pv.var(axis=1, ddof=1) / (Pv.var(axis=1, ddof=1) + tau_star)).mean())
    lam_bar = float(np.mean(lam_vals))
    print(f"  lambda_bar (uniform matched) = {lam_bar:.4f}")
    # 4) test eval
    rows = []
    for m in tqdm(split_ids["test"], desc="test eval", unit="mol"):
        Pt = P_test[m]
        s2 = Pt.var(axis=1, ddof=1)
        lam = s2 / (s2 + tau_star)
        Lam = float(lam.mean())
        raw = float(np.mean([Pt[:, k].sum() for k in range(Pt.shape[1])]))
        gims = (1 - Lam) * raw + Lam * N_test[m] * mu_mean
        uni = (1 - lam_bar) * raw + lam_bar * N_test[m] * mu_mean
        vw_s = [float(((1 - lam) * Pt[:, k] + lam * mu_per_seed[k]).sum())
                for k in range(Pt.shape[1])]
        vw = float(np.mean(vw_s))
        rows.append((m, labels[m], raw, gims, vw, uni))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["id", "exp", "raw", "gims", "vw", "uniform"])
    df.to_csv(os.path.join(outdir, f"{tag}_finetuned_test.csv"), index=False)

    def mets(a, b):
        a, b = np.asarray(a), np.asarray(b)
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        r2 = float(1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2))
        try:
            from scipy.stats import kendalltau
            tau = float(kendalltau(a, b)[0])
        except Exception:
            tau = float("nan")
        return mae, rmse, r2, tau

    def boot_d(a, b, ref_a, ref_b, n=10000):
        d = np.abs(np.asarray(a) - np.asarray(b)) - np.abs(np.asarray(ref_a) - np.asarray(ref_b))
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(d), (n, len(d)))
        ms = d[idx].mean(axis=1)
        return float(d.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))

    print(f"\n  [{tag} TEST n={len(df)}] tau*={tau_star:.3e} mu={mu_mean:.4f} lam_bar={lam_bar:.4f}")
    summary = {"tag": tag, "n_test": len(df), "tau_star": tau_star,
               "mu_T": mu_mean, "lambda_bar": lam_bar, "seeds": args.seeds}
    for arm in ("raw", "gims", "vw", "uniform"):
        mae, rmse, r2, tau = mets(df[arm], df["exp"])
        print(f"    {arm:8s} MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.4f} tau {tau:.3f}")
        summary[arm] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
                        "R2": round(r2, 4), "tau": round(float(tau), 4)}
    for arm in ("gims", "vw", "uniform"):
        dm, lo, hi = boot_d(df[arm], df["exp"], df["raw"], df["exp"], n=args.n_boot)
        print(f"    {arm} - raw dMAE {dm:+.3f} [{lo:+.3f},{hi:+.3f}] (n_boot={args.n_boot})")
        summary[f"d_{arm}_vs_raw"] = {"dMAE": round(dm, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    # 5) frozen vs fine-tuned
    fr = FROZEN[tag]
    print(f"\n  [{tag} FROZEN (Table-3 restated, untouched)]: "
          f"raw {fr['raw']:.2f} gims {fr['gims']:.2f} vw {fr['vw']:.2f} uniform {fr['uniform']:.2f} (n={fr['n']})")
    print(f"  [{tag} FINE-TUNED (new)]: raw {summary['raw']['MAE']:.3f} "
          f"gims {summary['gims']['MAE']:.3f} vw {summary['vw']['MAE']:.3f} "
          f"uniform {summary['uniform']['MAE']:.3f} (n={summary['n_test']})")
    json.dump(summary, open(os.path.join(outdir, f"{tag}_summary.json"), "w"), indent=2)
    print(f"  saved {outdir}/{tag}_finetuned_test.csv + {tag}_summary.json")
    print("  STOP: numbers + CIs + split files ready. LaTeX untouched by design.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["flexisol", "guthrie", "both"])
    ap.add_argument("--guthrie_csv", default=None)
    ap.add_argument("--outdir", default="out_transfer")
    ap.add_argument("--correction_ckpt", default=os.path.join(REPO, "expdb_vast", "stage2_correction.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--sandbox", action="store_true", help="2 epochs smoke test")
    ap.add_argument("--make_splits_only", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.correction_ckpt):
        print(f"FATAL: Stage-1+2 ckpt missing: {args.correction_ckpt}", file=sys.stderr)
        sys.exit(1)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} seeds={args.seeds} epochs={args.epochs} patience={args.patience} "
          f"lr={args.lr} wd={args.weight_decay} batch={args.batch_size} sandbox={args.sandbox}")
    print(f"config logged: {vars(args)}")
    tags = ["flexisol", "guthrie"] if args.dataset == "both" else [args.dataset]
    loaders = {"flexisol": load_flexisol,
               "guthrie": lambda: load_guthrie(args.guthrie_csv)}
    for tag in tqdm(tags, desc="datasets", unit="dataset"):
        run_dataset(tag, loaders[tag](), args, device)


if __name__ == "__main__":
    main()

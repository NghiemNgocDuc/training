"""Exp-DB external generalization - Vast GPU bundle driver.

Subcommands:
  md5check   Gate: regenerate the recorded fold-0 split from database.json +
             freesolv_conformers.hdf5 and verify the raw-bytes md5 of
             train+val+test ids json against the recorded c0ef2933...
             (must MATCH before training).
  generate   Build expdb_conformers.hdf5 for all 623 Exp-DB molecules
             (all conformers + MMFF energies) - verbatim _gen_confs protocol.
  infer      TTA-5 x 5 fold models (from --model_dir) -> per-fold + ensemble
             predictions on the Exp-DB set, metrics, worst-10 outliers.
  anchor     FreeSolv anchor from the retrained fold_*/ft_test_predictions.csv
             (TTA-5): per-fold MAE/RMSE/R2 + ensemble, mean-of-fold-MAE
             (headline-comparable) and pooled MAE.

Conformer protocol is a verbatim copy of cv_finetune_se.py _gen_confs
(ETKDGv3 randomSeed=42, pruneRmsThresh=0.5, MMFF94 optimize, numThreads=1).
"""

import os
import sys
import json
import argparse
import hashlib

BUNDLE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    p = os.path.abspath(os.path.join(BUNDLE, "..", "..", "..", ".."))
    return p if os.path.exists(os.path.join(p, "freesolv_conformers.hdf5")) else None


def _assets(conf):
    repo = _repo_root()
    hdf5 = conf.get("conformers") or (os.path.join(repo, "freesolv_conformers.hdf5")
                                      if repo else os.path.join(BUNDLE, "freesolv_conformers.hdf5"))
    cache = conf.get("cache_dir") or (os.path.join(repo, "aqm-spice2", "Data", "FreeSolv")
                                      if repo else os.path.join(BUNDLE, "freesolv_cache"))
    csv = conf.get("csv") or os.path.join(BUNDLE, "expdb_external_test.csv")
    outdir = conf.get("outdir") or os.path.join(BUNDLE, "results")
    return hdf5, cache, csv, outdir


RECORDED_MD5 = "c0ef2933417413f7d9837987421e7ceb"
RECORDED_FOLD_MEANS = [-3.872, -3.821, -3.819, -3.767, -3.736]

sys.path.insert(0, BUNDLE)

import numpy as np
import torch
import h5py
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from DimeModels import DimeNetPlusSE
from element_vocab import NUM_ELEMENTS, build_one_hot

EV_TO_KCAL = 23.0605
N_CONFS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_freesolv_labels(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def build_model():
    return DimeNetPlusSE(
        in_channels=NUM_ELEMENTS,
        hidden_channels=128,
        out_channels=1,
        num_blocks=3,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7, num_radial=6,
        cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        is_energy=True,
        use_multi_aggregate=False,
        use_se=False,
    )


def load_fold_models(model_dir):
    models = {}
    for fold in range(5):
        m = build_model().to(DEVICE)
        ck = os.path.join(model_dir, f"fold_{fold}", "finetuned.pt")
        state = torch.load(ck, map_location=DEVICE, weights_only=True)
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, (fold, missing, unexpected)
        m.eval()
        models[fold] = m
    return models


def gen_confs(smiles, n=N_CONFS):
    """Verbatim copy of cv_finetune_se.py _gen_confs."""
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
        opt = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    except Exception:
        opt = None
    z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32), dtype=torch.long)
    n_avail = min(n, mol.GetNumConformers())
    cds = [Data(z=z.clone(),
                pos=torch.tensor(np.array(mol.GetConformer(i).GetPositions(), dtype=np.float64),
                                 dtype=torch.float))
           for i in range(n_avail)]
    e = ([float(r[1]) for r in opt][:n_avail]
         if opt is not None and len(opt) >= n_avail
         else [float("nan")] * n_avail)
    return mol, cds, e


def predict(model, confs, batch_size=32):
    model.eval()
    loader = DataLoader(confs, batch_size=batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            x = build_one_hot(data, DEVICE)
            pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
            outs.append(pred.cpu())
    return torch.cat(outs).numpy()


def metrics(preds, expts):
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    r2 = float(1 - np.sum((preds - expts) ** 2) / np.sum((expts - expts.mean()) ** 2))
    if len(preds) >= 5:
        from scipy.stats import kendalltau
        tau, tau_p = kendalltau(preds, expts)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "kendall_tau": float(tau), "kendall_p": float(tau_p)}
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "kendall_tau": float("nan"), "kendall_p": float("nan")}


def load_expdb_csv(csv_path=None):
    import pandas as pd
    p = csv_path or os.path.join(BUNDLE, "expdb_external_test.csv")
    return pd.read_csv(p)


# ---------------- md5check ----------------

def md5check(conf):
    hdf5, cache, _, _ = _assets(conf)
    labels = load_freesolv_labels(os.path.join(cache, "database.json"))
    with h5py.File(hdf5, "r") as f:
        mol_ids = [m for m in f.keys()
                   if m in labels and isinstance(labels[m].get("expt"), (int, float))]
    expts = np.array([labels[m]["expt"] for m in mol_ids])
    mol_ids_sorted = [mol_ids[i] for i in np.argsort(expts)]
    n_folds = 5
    folds = [[] for _ in range(n_folds)]
    for i, mid in enumerate(mol_ids_sorted):
        folds[i % n_folds].append(mid)

    fold_means = [np.mean([labels[m]["expt"] for m in folds[f]]) for f in range(n_folds)]
    print(f"n={len(mol_ids)} fold means={[f'{v:.3f}' for v in fold_means]}")
    print(f"recorded fold means = {RECORDED_FOLD_MEANS}")

    test_ids = folds[0]
    train_val = []
    for f in range(1, 5):
        train_val.extend(folds[f])
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_val))
    rng.shuffle(idx)
    n_val = max(1, int(len(train_val) * 0.2))
    val_ids = [train_val[i] for i in idx[:n_val]]
    train_ids = [train_val[i] for i in idx[n_val:]]

    outdir = os.path.join(_assets(conf)[3], "split_check")
    os.makedirs(outdir, exist_ok=True)
    blob = b""
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        p = os.path.join(outdir, f"{name}_ids.json")
        with open(p, "w") as f:
            json.dump(ids, f)
        with open(p, "rb") as f:
            blob += f.read()
    md5 = hashlib.md5(blob).hexdigest()
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"recomputed md5: {md5}")
    print(f"recorded md5:   {RECORDED_MD5}")
    ok = md5 == RECORDED_MD5
    print("SPLIT MATCH" if ok else "SPLIT MISMATCH - DO NOT TRAIN")
    return 0 if ok else 1


# ---------------- generate ----------------

def generate(conf):
    import pandas as pd
    from tqdm import tqdm
    _, _, csv, outdir = _assets(conf)
    os.makedirs(outdir, exist_ok=True)
    df = load_expdb_csv(csv)
    out = os.path.join(outdir, "expdb_conformers.hdf5")
    fails = []
    n_conf_total = 0
    with h5py.File(out, "w") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="conformers"):
            r = gen_confs(row["smiles"])
            if r is None:
                fails.append(str(row["id"]))
                continue
            mol, cds, e = r
            g = f.create_group(str(row["id"]))
            g.attrs["smiles"] = row["smiles"]
            g.attrs["name"] = str(row["name"])
            g.attrs["dg_solv_kcal_mol"] = float(row["dg_solv_kcal_mol"])
            g.attrs["source"] = str(row["comments"])
            g.attrs["doi"] = str(row["doi"])
            g.create_dataset("atNUM", data=np.array([c.z.numpy() for c in cds], dtype=np.int32))
            g.create_dataset("atXYZ", data=np.array([c.pos.numpy() for c in cds], dtype=np.float32))
            g.create_dataset("mmff_energies", data=np.array(e, dtype=np.float64))
            n_conf_total += len(cds)
    print(f"saved {out}")
    print(f"conformers archived: {n_conf_total} across {len(df) - len(fails)} molecules")
    print("generation failures:", fails if fails else "none")


# ---------------- infer ----------------

def infer(model_dir, conf):
    import pandas as pd
    from tqdm import tqdm
    _, _, csv, outdir = _assets(conf)
    os.makedirs(outdir, exist_ok=True)
    df = load_expdb_csv(csv)
    h5 = os.path.join(outdir, "expdb_conformers.hdf5")
    models = load_fold_models(model_dir)

    mols = [str(r["id"]) for _, r in df.iterrows()]
    with h5py.File(h5, "r") as f:
        archived = set(f.keys())
    skipped = [m for m in mols if m not in archived]
    if skipped:
        print(f"WARNING: {len(skipped)} molecules not in conformer archive (MMFF-unparameterizable, "
              f"excluded): {skipped}")
    with h5py.File(h5, "r") as f:
        entries = []
        for mid in tqdm(mols, desc="load conformers"):
            if mid not in archived:
                continue
            g = f[mid]
            z = torch.tensor(g["atNUM"][0], dtype=torch.long)
            xyz = g["atXYZ"][...]
            confs = [Data(z=z.clone(), pos=torch.tensor(xyz[i], dtype=torch.float))
                     for i in range(xyz.shape[0])]
            entries.append((mid, confs))

    per_fold_preds = {}
    for fold in range(5):
        flat, flat_mid = [], []
        for mid, confs in entries:
            for cd in confs:
                flat.append(cd)
                flat_mid.append(mid)
        raw = predict(models[fold], flat, batch_size=64)
        means = {}
        for mid, v in zip(flat_mid, raw):
            means.setdefault(mid, []).append(float(v))
        per_fold_preds[fold] = {mid: float(np.mean(vs)) for mid, vs in means.items()}

    all_rows = []
    for mid, confs in entries:
        row = df.loc[df["id"].astype(str) == mid].iloc[0]
        per_fold = {f"pred_fold{fold}": per_fold_preds[fold][mid] for fold in range(5)}
        ens = float(np.mean(list(per_fold.values())))
        all_rows.append({
            "id": mid, "name": row["name"], "smiles": row["smiles"],
            "dg_exp_kcal": float(row["dg_solv_kcal_mol"]),
            "n_conf": len(confs), "source": str(row["comments"]), "doi": str(row["doi"]),
            **per_fold, "dg_pred_ensemble_kcal": ens,
            "abs_err": abs(ens - float(row["dg_solv_kcal_mol"])),
        })
    pdf = pd.DataFrame(all_rows)
    pdf.to_csv(os.path.join(outdir, "predictions_ensemble.csv"), index=False)

    summary = {"n": len(pdf)}
    for fold in range(5):
        m = metrics(pdf[f"pred_fold{fold}"].values, pdf["dg_exp_kcal"].values)
        summary[f"fold_{fold}"] = {k: round(v, 4) for k, v in m.items()}
        print(f"fold {fold}: MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} tau={m['kendall_tau']:.4f}")
    em = metrics(pdf["dg_pred_ensemble_kcal"].values, pdf["dg_exp_kcal"].values)
    summary["ensemble"] = {k: round(v, 4) for k, v in em.items()}
    print(f"ENSEMBLE: MAE={em['MAE']:.4f} RMSE={em['RMSE']:.4f} R2={em['R2']:.4f} tau={em['kendall_tau']:.4f}")

    src = pdf["source"].astype(str)
    for s in sorted(pdf["source"].astype(str).unique()):
        m = metrics(pdf.loc[src == s, "dg_pred_ensemble_kcal"].values,
                    pdf.loc[src == s, "dg_exp_kcal"].values)
        summary.setdefault("by_source", {})[s] = {k: round(v, 4) for k, v in m.items()}
        print(f"  source {s}: n={int((src == s).sum())} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")

    worst = pdf.nlargest(10, "abs_err")[["id", "name", "smiles", "dg_exp_kcal", "dg_pred_ensemble_kcal", "abs_err", "source"]]
    print("\nworst-10 outliers:")
    print(worst.to_string(index=False))
    worst.to_csv(os.path.join(outdir, "worst_10_outliers.csv"), index=False)
    json.dump(summary, open(os.path.join(outdir, "inference_metrics.json"), "w"), indent=2)
    print("saved predictions_ensemble.csv / worst_10_outliers.csv / inference_metrics.json")


# ---------------- anchor ----------------

def anchor(model_dir, conf):
    import pandas as pd
    _, _, _, outdir = _assets(conf)
    os.makedirs(outdir, exist_ok=True)
    pdf = pd.DataFrame()
    missing = []
    for fold in range(5):
        p = os.path.join(model_dir, f"fold_{fold}", "ft_test_predictions.csv")
        if not os.path.exists(p):
            missing.append(fold)
            continue
        fdf = pd.read_csv(p)
        fdf["fold"] = fold
        pdf = pd.concat([pdf, fdf], ignore_index=True)
    if missing:
        print(f"WARNING: missing folds {missing}; anchor over available folds only")
    if len(pdf) == 0:
        print("no predictions found; run training first")
        return
    per_fold = []
    pooled = metrics(pdf["dG_pred_kcal"].values, pdf["dG_exp_kcal"].values)
    for fold in range(5):
        sub = pdf.loc[pdf["fold"] == fold]
        if len(sub) == 0:
            continue
        m = metrics(sub["dG_pred_kcal"].values, sub["dG_exp_kcal"].values)
        per_fold.append(m)
        print(f"fold {fold} (n={len(sub)}): "
              f"MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")
    mean_fold_mae = float(np.mean([m["MAE"] for m in per_fold]))
    print(f"\nmean-of-fold MAE (headline-comparable): {mean_fold_mae:.4f}")
    print(f"pooled MAE over n={len(pdf)}: {pooled['MAE']:.4f} RMSE={pooled['RMSE']:.4f} R2={pooled['R2']:.4f}")
    out = {
        "n": int(len(pdf)),
        "per_fold": [{"MAE": round(m["MAE"], 4), "RMSE": round(m["RMSE"], 4), "R2": round(m["R2"], 4)} for m in per_fold],
        "mean_of_fold_mae": round(mean_fold_mae, 4),
        "pooled": {k: round(v, 4) for k, v in pooled.items()},
    }
    json.dump(out, open(os.path.join(outdir, "anchor_metrics.json"), "w"), indent=2)
    print("saved anchor_metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["md5check", "generate", "infer", "anchor"])
    ap.add_argument("--model_dir", default=os.path.join(BUNDLE, "results", "cv_results_retrain"))
    ap.add_argument("--conformers", default=None, help="freesolv_conformers.hdf5 path (auto-detects repo layout)")
    ap.add_argument("--cache_dir", default=None, help="FreeSolv labels dir (database.json)")
    ap.add_argument("--csv", default=None, help="expdb_external_test.csv path")
    ap.add_argument("--outdir", default=None, help="results output dir")
    args = ap.parse_args()
    conf = {"conformers": args.conformers, "cache_dir": args.cache_dir,
            "csv": args.csv, "outdir": args.outdir}
    if args.cmd == "md5check":
        sys.exit(md5check(conf))
    elif args.cmd == "generate":
        generate(conf)
    elif args.cmd == "infer":
        infer(args.model_dir, conf)
    elif args.cmd == "anchor":
        anchor(args.model_dir, conf)

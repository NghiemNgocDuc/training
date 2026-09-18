"""Step 3: DimeNet 3-seed inference on Guthrie-novel 3D structures + GIMS/VW/uniform.

Frozen transfer settings (same as FlexiSol paper arm):
  TAU_STAR=4.725394227550238e-04, mu_T per-seed from FreeSolv train pool,
  uniform lambda_bar=0.8014. No retraining, no recalibration.
TQDM everywhere; incremental CSV checkpoint + resume.
"""
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

RUN = r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run"
DATA = r"C:\Users\User\Documents\Data"
SDF_PKL = os.path.join(RUN, "guthrie_3d.pkl")
OUT_CSV = os.path.join(RUN, "guthrie_all_methods.csv")
CKPT_DIR = os.path.join(DATA, "expdb_vast", "results_seeds")
SEEDS = [42, 123, 999]
EV_TO_KCAL = 23.0605
TAU_STAR = 4.725394227550238e-04
LAMBDA_BAR = 0.8014

sys.path.insert(0, os.path.join(DATA, "aqm-spice2", "freesolv"))
sys.path.insert(0, os.path.join(DATA, "aqm-spice2", "freesolv",
                                "experimental_uncertainty_refine"))
from DimeModels import DimeNetPlusSE
from element_vocab import NUM_ELEMENTS, ELEMENT_TO_IDX

SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
         "Cl": 17, "Br": 35, "Si": 14, "I": 53, "B": 5, "Li": 3, "Na": 11,
         "Mg": 12, "K": 19, "Ca": 20}


def load_mu():
    mu = {}
    for s in SEEDS:
        with open(os.path.join(CKPT_DIR, f"peratom_seed{s}.pkl"), "rb") as f:
            d = pickle.load(f)
        pt = np.concatenate([d["P_train"][m] for m in d["P_train"]])
        mu[s] = float(pt.mean())
        print(f"seed {s}: mu_T={mu[s]:+.6f} ({len(pt)} train atoms)", flush=True)
    return mu


def build_models(device):
    models = {}
    for s in tqdm(SEEDS, desc="load models", unit="seed"):
        m = DimeNetPlusSE(
            hidden_channels=128, in_channels=NUM_ELEMENTS, out_channels=1,
            num_blocks=3, int_emb_size=64, basis_emb_size=8,
            out_emb_channels=256, num_spherical=7, num_radial=6, cutoff=6.0,
            max_num_neighbors=32, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_output_layers=3, is_energy=True).to(device)
        state = torch.load(os.path.join(CKPT_DIR, f"finetuned_seed{s}.pt"),
                           map_location=device)
        m.load_state_dict(state, strict=False)
        m.eval()
        models[s] = m
    return models


def infer_mol(syms, xyz, models, device):
    z = np.array([SYM2Z[e] for e in syms], dtype=np.int64)
    if any(int(v) not in ELEMENT_TO_IDX for v in z):
        return None
    x = np.zeros((len(z), NUM_ELEMENTS), dtype=np.float32)
    for i, v in enumerate(z):
        x[i, ELEMENT_TO_IDX[int(v)]] = 1.0
    xt = torch.tensor(x).to(device)
    pt = torch.tensor(np.asarray(xyz, dtype=np.float32)).to(device)
    bt = torch.zeros(len(z), dtype=torch.long).to(device)
    per = {}
    with torch.no_grad():
        for s, m in models.items():
            orig = m.is_energy
            m.is_energy = False
            try:
                P = m(xt, pt, bt).detach().cpu().numpy().flatten() * EV_TO_KCAL
            finally:
                m.is_energy = orig
            per[s] = P.astype(np.float64)
    return per


def main():
    device = torch.device("cpu")
    print(f"torch {torch.__version__}", flush=True)
    mu = load_mu()
    mu_mean = float(np.mean(list(mu.values())))
    models = build_models(device)
    blob = pickle.load(open(SDF_PKL, "rb"))
    mols, values = blob["mols"], blob["values"]
    ids = sorted(mols, key=int)
    print(f"molecules: {len(ids)}", flush=True)

    done = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                done[r["cid"]] = r
        print(f"resume: {len(done)} done", flush=True)

    t0 = time.time()
    with open(OUT_CSV, "a" if done else "w", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["cid", "exp", "raw", "gims", "uniform", "vw",
                        "Lambda", "N"])
        todo = [c for c in ids if c not in done]
        for cid in tqdm(todo, desc="infer", unit="mol"):
            syms, xyz = mols[cid]
            per = infer_mol(syms, xyz, models, device)
            if per is None:
                continue
            P = np.stack([per[s] for s in SEEDS], axis=1)
            s2 = P.var(axis=1, ddof=1)
            lam = s2 / (s2 + TAU_STAR)
            Lm = float(lam.mean())
            N = len(syms)
            Eraw = float(np.mean([per[s].sum() for s in SEEDS]))
            Eg = (1 - Lm) * Eraw + Lm * N * mu_mean
            Eu = (1 - LAMBDA_BAR) * Eraw + LAMBDA_BAR * N * mu_mean
            Ev = float(np.mean([float(((1 - lam) * per[s]
                                       + lam * mu[s]).sum()) for s in SEEDS]))
            w.writerow([cid, f"{values[cid]:.4f}", f"{Eraw:.4f}",
                        f"{Eg:.4f}", f"{Eu:.4f}", f"{Ev:.4f}",
                        f"{Lm:.6f}", N])
            if (len(done) + 1) % 25 == 0:
                f.flush()
    print(f"infer done in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}", flush=True)

    # ---- evaluate with paired bootstrap
    rows = list(csv.DictReader(open(OUT_CSV)))
    print(f"scored: {len(rows)}", flush=True)
    exp = np.array([float(r["exp"]) for r in rows])
    out = {}
    for arm in ("raw", "gims", "uniform", "vw"):
        pred = np.array([float(r[arm]) for r in rows])
        out[arm] = float(np.abs(pred - exp).mean())
    print(f"Raw MAE {out['raw']:.3f}", flush=True)
    rng = np.random.default_rng(7)
    for arm in ("gims", "uniform", "vw"):
        d = (np.abs(np.array([float(r[arm]) for r in rows]) - exp)
             - np.abs(np.array([float(r["raw"]) for r in rows]) - exp))
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{arm:8s} MAE {out[arm]:.3f}  Delta {out[arm]-out['raw']:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]", flush=True)
    d = (np.abs(np.array([float(r["gims"]) for r in rows]) - exp)
         - np.abs(np.array([float(r["vw"]) for r in rows]) - exp))
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"GIMS-VW {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    d = (np.abs(np.array([float(r["gims"]) for r in rows]) - exp)
         - np.abs(np.array([float(r["uniform"]) for r in rows]) - exp))
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"GIMS-Uni {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()

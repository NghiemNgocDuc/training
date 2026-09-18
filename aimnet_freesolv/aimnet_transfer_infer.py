"""MACE transfer inference: FlexiSol-water (shipped gas c0) + Guthrie-novel (stored 3D).

Frozen MACE 3-seed fold-0 models (CPU). Per-atom node_energy + molecular E.
TQDM everywhere; per-set incremental CSV checkpoint + resume.
Skips molecules with elements outside the MACE 10-elem vocab (recorded).

Outputs -> mace_freesolv/fold0_ensemble/transfer/:
  mace_flexisol_permol.csv  (name, exp, E_seed*, P files ref)
  mace_flexisol_peratom.pkl {name: P_stack}
  mace_guthrie_permol.csv / mace_guthrie_peratom.pkl
Usage: py -V:3.12 mace_freesolv/mace_transfer_infer.py [--sets flexisol,guthrie]
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "aimnet_freesolv"))
OUT = os.path.join(REPO, "aimnet_freesolv", "fold0_ensemble", "transfer")
os.makedirs(OUT, exist_ok=True)

from aimnet_ensemble import build_model, prep, attach_hook, decompose  # noqa: E402

EV_TO_KCAL = 23.0605
SEEDS = [42, 123, 7, 2024, 999]
AIM_DIR = os.path.join(REPO, "aimnet_freesolv", "fold0_ensemble")
FLEXI_GAS = os.path.join(REPO, "flexisol", "flexisol", "gas", "water")
FLEXI_REF = os.path.join(REPO, "flexisol", "data", "references",
                         "dgsolv-references.csv")
GUTH_PKL_REPO = os.path.join(REPO, "guthrie_novel", "guthrie_3d.pkl")
GUTH_PKL = (GUTH_PKL_REPO if os.path.exists(GUTH_PKL_REPO)
            else r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run\guthrie_3d.pkl")

SYM2Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14,
         "P": 15, "S": 16, "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53}


def load_models(device):
    models = {}
    for s in tqdm(SEEDS, desc="load AIMNet2 models", unit="seed"):
        calc, model = build_model(device)
        model.load_state_dict(torch.load(
            os.path.join(AIM_DIR, f"seed_{s}", "model.pt"),
            map_location=device, weights_only=True), strict=False)
        model.to(device)
        model.eval()
        models[s] = (calc, model)
    return models


@torch.no_grad()
def per_atom(pair, device, syms, xyz):
    calc, model = pair
    try:
        z = np.array([SYM2Z[e] for e in syms], dtype=np.int64)
    except KeyError:
        return None
    n = len(z)
    store = {}
    handle = attach_hook(model, store)
    try:
        P, E = decompose(model, store,
                         prep(calc, z,
                              np.asarray(xyz, dtype=np.float32),
                              device), n)
    except Exception:
        handle.remove()
        return None
    handle.remove()
    gate = abs(float(P.sum()) - E)
    if gate > 1e-3 or len(P) != n:
        return None
    return P.astype(np.float64), E


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].strip())
    syms, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, xyz


def flexi_items():
    refs = list(csv.DictReader(open(FLEXI_REF, encoding="utf-8")))
    water = [r for r in refs if r["Solvent"] == "water"]
    dirs = os.listdir(FLEXI_GAS)
    by_mol = {}
    for d in dirs:
        if "_chrg0_" not in d:
            continue
        name = d.split("_chrg0")[0]
        by_mol.setdefault(name, []).append(d)
    items, missing = [], []
    for r in water:
        name = r["FlexiSol Name"]
        cands = sorted(by_mol.get(name, []))
        c0 = [d for d in cands if d.endswith("_c0")]
        if not c0:
            missing.append(name)
            continue
        items.append((name, os.path.join(FLEXI_GAS, c0[0], "coord.xyz"),
                      float(r["Value (\\kcalpmole)"])))
    print(f"[flexisol] water={len(water)} mapped={len(items)} "
          f"missing={len(missing)} {missing[:5]}", flush=True)
    return items


def guthrie_items():
    blob = pickle.load(open(GUTH_PKL, "rb"))
    mols, values = blob["mols"], blob["values"]
    items = [(cid, None, float(values[cid])) for cid in sorted(mols, key=int)]
    print(f"[guthrie] molecules={len(items)}", flush=True)
    return items, mols


def run_set(set_name, items, mols_store, models, device, center_ev=0.0):
    csv_path = os.path.join(OUT, f"aimnet_{set_name}_permol.csv")
    pkl_path = os.path.join(OUT, f"aimnet_{set_name}_peratom.pkl")
    done, peratom, skipped = {}, {}, []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done[r["id"]] = r
    if os.path.exists(pkl_path):
        peratom = pickle.load(open(pkl_path, "rb"))
    print(f"[{set_name}] resume: {len(done)} done", flush=True)
    t0 = time.time()
    with open(csv_path, "a" if done else "w", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["id", "exp"] + [f"E_{s}" for s in SEEDS] + ["N"])
        for _id, path, exp in tqdm([t for t in items if str(t[0]) not in done],
                                   desc=f"AIMNet2 {set_name}", unit="mol"):
            if path is not None:
                syms, xyz = read_xyz(path)
            else:
                syms, xyz = mols_store[str(_id)][0], mols_store[str(_id)][1]
            Ps, Es, ok = {}, {}, True
            for s in SEEDS:
                r = per_atom(models[s], device, syms, xyz)
                if r is None:
                    ok = False
                    break
                Ps[s], Es[s] = r
            if not ok:
                skipped.append(str(_id))
                continue
            # Uncenter to final space (constant shift; variances unchanged).
            c_kcal = center_ev * EV_TO_KCAL
            for s in SEEDS:
                Es[s] = Es[s] + c_kcal
                Ps[s] = Ps[s] + c_kcal / len(syms)
            P = np.stack([Ps[s] for s in SEEDS], axis=1)
            peratom[str(_id)] = P
            w.writerow([_id, f"{exp:.4f}"] + [f"{Es[s]:.4f}" for s in SEEDS]
                       + [len(syms)])
            if (len(done) + len(peratom)) % 25 == 0:
                f.flush()
                pickle.dump(peratom, open(pkl_path, "wb"))
    pickle.dump(peratom, open(pkl_path, "wb"))
    print(f"[{set_name}] done {len(peratom)} kept, {len(skipped)} skipped "
          f"{skipped[:5]} in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    global AIM_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="flexisol,guthrie")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,7,2024,999")
    ap.add_argument("--base_dir", default=AIM_DIR,
                    help="Ensemble dir (aimnet_freesolv/fold0_ensemble)")
    a = ap.parse_args()
    global SEEDS
    SEEDS = [int(s) for s in a.seeds.split(",")]
    print(f"[seeds] {SEEDS} (K={len(SEEDS)})", flush=True)
    AIM_DIR = a.base_dir
    OUT = os.path.join(AIM_DIR, "transfer")
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(a.device)
    print(f"device={device}", flush=True)
    models = load_models(device)
    try:
        center_ev = float(json.load(open(os.path.join(
            AIM_DIR, "ensemble_summary.json")))["_target_center_ev"])
    except (KeyError, FileNotFoundError):
        center_ev = 0.0
        print("[warn] no target center found, using 0.0", flush=True)
    print(f"[center] {center_ev:.6f} eV", flush=True)
    if "flexisol" in a.sets.split(","):
        run_set("flexisol", flexi_items(), None, models, device, center_ev)
    if "guthrie" in a.sets.split(","):
        items, store = guthrie_items()
        run_set("guthrie", items, store, models, device, center_ev)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

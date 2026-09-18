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
sys.path.insert(0, os.path.join(REPO, "mace_freesolv"))
OUT = os.path.join(REPO, "mace_freesolv", "fold0_ensemble", "transfer")
os.makedirs(OUT, exist_ok=True)

from data import radius_graph, ELEMENT_TO_IDX, MACE_NUM_ELEMENTS  # noqa: E402
from model import MACEFreeSolv  # noqa: E402

EV_TO_KCAL = 23.0605
SEEDS = [42, 123, 999]
MACE_DIR = os.path.join(REPO, "mace_freesolv", "fold0_ensemble")
FLEXI_GAS = os.path.join(REPO, "flexisol", "flexisol", "gas", "water")
FLEXI_REF = os.path.join(REPO, "flexisol", "data", "references",
                         "dgsolv-references.csv")
GUTH_PKL_REPO = os.path.join(REPO, "guthrie_novel", "guthrie_3d.pkl")
GUTH_PKL = (GUTH_PKL_REPO if os.path.exists(GUTH_PKL_REPO)
            else r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run\guthrie_3d.pkl")

SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
         "Cl": 17, "Br": 35, "I": 53, "Si": 14}


def load_models(device):
    models = {}
    for s in tqdm(SEEDS, desc="load MACE models", unit="seed"):
        m = MACEFreeSolv(model_size="medium", device=str(device),
                         fit_refs=False)
        m.load(os.path.join(MACE_DIR, f"seed_{s}", "model.pt"))
        m.to(device)
        m.eval()
        models[s] = m
    return models


@torch.no_grad()
def per_atom(model, device, syms, xyz, r_max=5.0):
    try:
        z = [SYM2Z[e] for e in syms]
    except KeyError:
        return None
    if any(ELEMENT_TO_IDX.get(int(v)) is None for v in z):
        return None
    n = len(z)
    na = torch.zeros(n, MACE_NUM_ELEMENTS)
    for i, v in enumerate(z):
        na[i, ELEMENT_TO_IDX[int(v)]] = 1.0
    pos = torch.tensor(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
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
    P = raw["node_energy"].view(-1).detach().cpu().numpy() * EV_TO_KCAL
    E = float(raw["energy"].view(-1).detach().cpu().numpy()[0] * EV_TO_KCAL)
    if abs(float(P.sum()) - E) > 1e-3:
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


def run_set(set_name, items, mols_store, models, device):
    csv_path = os.path.join(OUT, f"mace_{set_name}_permol.csv")
    pkl_path = os.path.join(OUT, f"mace_{set_name}_peratom.pkl")
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
                                   desc=f"MACE {set_name}", unit="mol"):
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
    global MACE_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="flexisol,guthrie")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,7,2024,999")
    ap.add_argument("--base_dir", default=MACE_DIR,
                    help="Ensemble dir (fold0_ensemble or fold0_ensemble_off23)")
    a = ap.parse_args()
    global SEEDS
    SEEDS = [int(s) for s in a.seeds.split(",")]
    print(f"[seeds] {SEEDS} (K={len(SEEDS)})", flush=True)
    MACE_DIR = a.base_dir
    OUT = os.path.join(MACE_DIR, "transfer")
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(a.device)
    print(f"device={device}", flush=True)
    models = load_models(device)
    if "flexisol" in a.sets.split(","):
        run_set("flexisol", flexi_items(), None, models, device)
    if "guthrie" in a.sets.split(","):
        items, store = guthrie_items()
        run_set("guthrie", items, store, models, device)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

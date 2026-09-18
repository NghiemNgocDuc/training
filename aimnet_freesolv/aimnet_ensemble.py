"""AIMNet2-2025 fold-0 ensemble — 5-seed fine-tune + per-atom dump.

Protocol (mirrors fold0_ensemble_off23.py, adapted to aimnet API):
  - Init: official aimnet2-2025 member-0 weights for ALL seeds (only seed changes:
    data shuffling + optimizer stochasticity). HF download cached automatically.
  - Data: frozen fold-0 split (411/102/129), stored HDF5 single conformers.
  - Loss: energy-only MSE on hydration dG in eV (no forces/charges labels).
    Adam lr=1e-4, wd=1e-5, batch=1, <=200 epochs, ReduceLROnPlateau (f=0.5,
    patience 15, min 1e-6), early stop patience 30 on val MAE (kcal).
  - Decomposition: P_mi = shifted atomic energies (hook on outputs.atomic_shift)
    plus a UNIFORM split of the short-range Coulomb residual, so sum(P) == E
    exactly (<1e-9 check). External D3/long-range Coulomb excluded throughout
    (train and infer consistently on short-range model energy).
  - Per-seed best-val state_dict + metrics + test CSV + per-atom pkls.

Every step has a tqdm progress bar.
Usage (Vast GPU, see run_aimnet_ensemble.sh):
  python aimnet_freesolv/aimnet_ensemble.py --seeds 42,123,7,2024,999 --device cuda
  python aimnet_freesolv/aimnet_ensemble.py --seeds 42,123,7,2024,999 --quick_test --device cuda
Outputs -> aimnet_freesolv/fold0_ensemble/ (seed_{S}/ + per-atom files).
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
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

EV_TO_KCAL = 23.0605
TRAINER_VERSION = "norefit-center-v1"  # printed first line of every run
FROZEN_SPLIT_DIR = os.path.join(
    REPO_ROOT, "aqm-spice2", "aqm-spice2", "freesolv", "cv_results_full", "fold_0")
DEFAULT_HDF5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
DEFAULT_LABELS = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")
MODEL_ID = "isayevlab/aimnet2-2025"


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split():
    tr = json.load(open(os.path.join(FROZEN_SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(FROZEN_SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(FROZEN_SPLIT_DIR, "test_ids.json")))
    assert (len(tr), len(va), len(te)) == (411, 102, 129), "frozen split changed!"
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    print(f"[split] FROZEN fold-0: train={len(tr)} val={len(va)} test={len(te)}",
          flush=True)
    return tr, va, te


def load_labels():
    if not os.path.exists(DEFAULT_LABELS):
        import urllib.request
        os.makedirs(os.path.dirname(DEFAULT_LABELS), exist_ok=True)
        print("[labels] database.json missing -> downloading from MobleyLab/FreeSolv",
              flush=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json",
            DEFAULT_LABELS)
        print("[labels] downloaded", flush=True)
    with open(DEFAULT_LABELS) as f:
        lab = json.load(f)
    return {m: float(v["expt"]) / EV_TO_KCAL for m, v in lab.items()
            if isinstance(v.get("expt"), (int, float))}


def fit_refs(model, device, train_mols, labels_ev, hdf5):
    """Least-squares per-Z reference energies on train labels (MACE-style).

    Overwrites outputs.atomic_shift.shifts.weight so initial predictions land
    at hydration scale instead of total-energy scale. Train-only (no leakage).
    """
    import h5py as _h5
    h5 = _h5.File(hdf5, "r")
    A = np.zeros((len(train_mols), 64))
    b = np.zeros(len(train_mols))
    for i, m in enumerate(train_mols):
        z = np.asarray(h5[m]["atNUM"]).reshape(-1).astype(int)
        for v in z:
            if 0 <= v < 64:
                A[i, v] += 1.0
        b[i] = labels_ev[m]
    h5.close()
    ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    w = model.outputs.atomic_shift.shifts.weight
    with torch.no_grad():
        w.copy_(torch.tensor(ref, dtype=w.dtype, device=device).reshape(-1, 1))
    pred = A @ ref
    mae = float(np.mean(np.abs(pred - b))) * EV_TO_KCAL
    print(f"[refit] per-Z refs set; train MAE at init = {mae:.3f} kcal/mol",
          flush=True)
    return ref
def build_model(device, member=0):
    from aimnet.calculators import AIMNet2Calculator
    calc = AIMNet2Calculator(MODEL_ID, ensemble_member=member,
                             device=str(device))
    # AEV CUDA kernels support float32 backward only (double fails in
    # aev.py; mixed fails on the double atomic shifts). Train fully in
    # float32: cast the double-loaded shifts down (fine-tune adapts them).
    model = calc.model.to(device).float()
    for p in model.parameters():
        p.requires_grad_(True)
    return calc, model


def prep(calc, z, xyz, device):
    return calc.prepare_input({
        "coord": torch.tensor(np.asarray(xyz, dtype=np.float32),
                              dtype=torch.float32, device=device),
        "numbers": torch.tensor(np.asarray(z).reshape(-1), dtype=torch.int64,
                                device=device),
        "charge": torch.tensor([0.0], dtype=torch.float32,
                               device=device)})


def decompose(model, store, data, n_real):
    """Forward + per-atom decomposition. Returns (P_kcal (N,), E_kcal)."""
    out = model(data)
    E_model = float(out["energy"].view(-1)[0].detach()) * EV_TO_KCAL
    P = np.asarray(store["P_shift"], dtype=np.float64)
    assert len(P) == n_real, f"unmasked {len(P)} != {n_real} atoms"
    R = E_model - float(P.sum())
    P = P + R / n_real  # uniform split of short-range Coulomb residual
    return P, E_model


def attach_hook(model, store):
    def hook(mod, inp, out):
        mask = out["mask_i"].detach().reshape(-1)
        P = out["energy"].detach().reshape(-1) * EV_TO_KCAL
        real = (mask < 0.5).cpu().numpy()
        store["P_shift"] = P.cpu().numpy()[real]
    return model.outputs.atomic_shift.register_forward_hook(hook)


def train_one_seed(seed, tr, va, te, labels, center, hdf5, device, out_root,
                   cfg, quick):
    seed_dir = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    set_seed(seed)
    print(f"\n{'='*64}\n  SEED {seed}  (fold-0 only, init=aimnet2-2025 member-0)"
          f"\n{'='*64}", flush=True)

    calc, model = build_model(device)
    ckpt = os.path.join(seed_dir, "model.pt")
    state_path = os.path.join(seed_dir, "train_state.json")
    done_path = os.path.join(seed_dir, "metrics.json")
    if os.path.exists(done_path) and os.path.exists(ckpt) and not quick:
        print(f"[seed {seed}] DONE already, skipping train", flush=True)
        return json.load(open(done_path))
    resumed = False
    start_ep, best, best_ep, stale = 1, float("inf"), -1, 0
    if os.path.exists(ckpt) and os.path.exists(state_path) and not quick:
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        st = json.load(open(state_path))
        best, best_ep, stale = st["best"], st["best_ep"], st["stale"]
        start_ep = st["epoch"] + 1
        resumed = True
        print(f"[seed {seed}] RESUME from ep {start_ep} "
              f"(best {best:.3f} @ ep {best_ep}, stale {stale})", flush=True)
    else:
        # Centered targets only (NO refit copy_: in-place weight writes
        # trigger a CUDA-only autograd dtype failure at first backward;
        # proven by elimination on Vast GPU 2026-09-18). Center-only matches
        # the Phase-1 probe graph exactly (40/40 mols OK on Vast CUDA).
        print("[refit] SKIPPED (center-only mode for CUDA safety)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg["patience"] // 2,
        min_lr=cfg["lr_min"])
    loss_fn = torch.nn.MSELoss()
    h5 = h5py.File(hdf5, "r")

    def batch_E(mols, train_mode):
        # Returns FINAL-space kcal/mol (model predicts centered; add back here).
        model.train(train_mode)
        Es = {}
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            it = (tqdm(mols, desc="val", leave=False, unit="mol")
                  if not train_mode else mols)
            for m in it:
                g = h5[m]
                z = np.asarray(g["atNUM"]).reshape(-1)
                xyz = np.asarray(g["atXYZ"], dtype=np.float32).reshape(-1, 3)
                out = model(prep(calc, z, xyz, device))
                Es[m] = (float(out["energy"].view(-1)[0].detach()) + center) \
                    * EV_TO_KCAL
        return Es

    def mae(mols, Es):
        return float(np.mean([abs(Es[m] - labels[m] * EV_TO_KCAL)
                              for m in mols]))

    epochs = 2 if quick else cfg["epochs"]
    rng = np.random.RandomState(seed)
    for _ in range(1, start_ep):
        rng.permutation(len(tr))  # fast-forward RNG past resumed epochs
    pbar = tqdm(range(start_ep, epochs + 1), desc=f"seed{seed} epochs",
                unit="ep", initial=start_ep - 1, total=epochs)
    for epoch in pbar:
        model.train()
        order = np.array(tr, dtype=object)[rng.permutation(len(tr))]
        tot = 0.0
        for m in tqdm(order, desc=f"seed{seed} ep{epoch} train", leave=False,
                      unit="mol"):
            g = h5[m]
            z = np.asarray(g["atNUM"]).reshape(-1)
            xyz = np.asarray(g["atXYZ"], dtype=np.float32).reshape(-1, 3)
            out = model(prep(calc, z, xyz, device))
            loss = loss_fn(out["energy"].view(-1),
                           torch.tensor([labels[m] - center],
                                        dtype=torch.float32,
                                        device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item()
        Es_v = batch_E(va, False)
        vm = mae(va, Es_v)
        sched.step(vm)
        pbar.set_postfix(best=f"{best:.3f}", cur=f"{vm:.3f}")
        print(f"  seed {seed} ep {epoch:3d}/{epochs} loss={tot/len(tr):.5f} "
              f"valMAE={vm:.3f} lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if vm < best:
            best, best_ep, stale = vm, epoch, 0
            torch.save(model.state_dict(), ckpt)
            print(f"  [seed {seed}] * best checkpoint (val {vm:.3f})", flush=True)
        else:
            stale += 1
        json.dump({"best": best, "best_ep": best_ep, "stale": stale,
                   "epoch": epoch}, open(state_path, "w"))
        if stale >= cfg["patience"]:
            print(f"  [seed {seed}] early stop at ep {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    model.eval()
    Es_t = batch_E(te, False)
    tm = mae(te, Es_t)
    metrics = {"seed": seed, "n_train": len(tr), "n_val": len(va),
               "n_test": len(te), "best_epoch": best_ep,
               "best_val_mae_kcal": best, "test_mae_kcal": tm,
               "init": "aimnet2-2025 member-0"}
    json.dump(metrics, open(os.path.join(seed_dir, "metrics.json"), "w"),
              indent=2)
    with open(os.path.join(seed_dir, "test_predictions.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "pred_kcal", "true_kcal"])
        for m in te:
            w.writerow([m, f"{Es_t[m]:.6f}", f"{labels[m]*EV_TO_KCAL:.6f}"])
    print(f"[seed {seed}] TEST MAE={tm:.3f} (best ep {best_ep})", flush=True)
    h5.close()
    return metrics


def dump_per_atom(seeds, tr, va, te, labels, center, hdf5, device, out_root,
                  quick):
    all_ids = tr + va + te
    if quick:
        all_ids = all_ids[:20]
        print("[dump] QUICK mode: first 20 molecules only", flush=True)
    per_seed = {}
    for s in tqdm(seeds, desc="load models", unit="seed"):
        calc, model = build_model(device)
        model.load_state_dict(torch.load(
            os.path.join(out_root, f"seed_{s}", "model.pt"),
            map_location=device, weights_only=True))
        model.eval()
        per_seed[s] = (calc, model)
    h5 = h5py.File(hdf5, "r")
    max_gate, skipped = 0.0, []
    P_all = {s: {} for s in seeds}
    E_all = {s: {} for s in seeds}
    for m in tqdm(all_ids, desc="dump per-atom", unit="mol"):
        g = h5[m]
        z = np.asarray(g["atNUM"]).reshape(-1)
        xyz = np.asarray(g["atXYZ"], dtype=np.float32).reshape(-1, 3)
        ok = True
        with torch.no_grad():
            for s in seeds:
                calc, model = per_seed[s]
                store = {}
                handle = attach_hook(model, store)
                try:
                    P, E = decompose(model, store,
                                     prep(calc, z, xyz, device), len(z))
                except Exception:
                    ok = False
                    handle.remove()
                    break
                handle.remove()
                # Uncenter to final space (constant shift: gate unaffected).
                P = P + center * EV_TO_KCAL / len(z)
                E = E + center * EV_TO_KCAL
                gate = abs(float(P.sum()) - E)
                max_gate = max(max_gate, gate)
                assert gate < 1e-3, f"gate failed {m} seed {s}: {gate:.2e}"
                P_all[s][m] = P
                E_all[s][m] = E
        if not ok:
            skipped.append(m)
            for s in seeds:
                P_all[s].pop(m, None)
                E_all[s].pop(m, None)
    print(f"[dump] kept={len(P_all[seeds[0]])} skipped={len(skipped)} "
          f"{skipped[:5]} max|sum(P)-E|={max_gate:.2e}", flush=True)
    for s in seeds:
        P_train = {m: P_all[s][m] for m in tr if m in P_all[s]}
        mu_T = float(np.concatenate(list(P_train.values())).mean())
        out = {"seed": s, "P_all": P_all[s], "E_all": E_all[s],
               "P_train": P_train, "mu_T_kcal": mu_T,
               "node_key": "atomic_shift energy + uniform Coulomb-residual split"}
        with open(os.path.join(out_root, f"peratom_aimnet_seed{s}.pkl"),
                  "wb") as fh:
            pickle.dump(out, fh)
        print(f"[dump seed {s}] mu_T={mu_T:+.4f}", flush=True)

    common = [m for m in all_ids if all(m in P_all[s] for s in seeds)]
    with open(os.path.join(out_root, "aimnet_node_contributions.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "atom_idx"] + [f"P_seed{s}" for s in seeds])
        for m in tqdm(common, desc="write node csv", unit="mol"):
            Ps = np.stack([P_all[s][m] for s in seeds], axis=1)
            for a in range(Ps.shape[0]):
                w.writerow([m, a] + [f"{v:.6f}" for v in Ps[a]])
    with open(os.path.join(out_root, "aimnet_seed_predictions_all642.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "true_value"] + [f"pred_seed{s}" for s in seeds])
        for m in tqdm(common, desc="write pred csv", unit="mol"):
            w.writerow([m, f"{labels[m]*EV_TO_KCAL:.6f}"] +
                       [f"{E_all[s][m]:.6f}" for s in seeds])
    h5.close()
    return skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,7,2024,999")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hdf5", default=DEFAULT_HDF5)
    ap.add_argument("--output_dir",
                    default=os.path.join(REPO_ROOT, "aimnet_freesolv",
                                         "fold0_ensemble"))
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_dump", action="store_true")
    ap.add_argument("--quick_test", action="store_true")
    a = ap.parse_args()
    device = torch.device(
        a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[version] {TRAINER_VERSION}", flush=True)
    print(f"[run] device={device} torch={torch.__version__}", flush=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[seeds] {seeds} (K={len(seeds)})", flush=True)
    os.makedirs(a.output_dir, exist_ok=True)
    cfg = {"lr": a.lr, "lr_min": a.lr_min, "wd": a.weight_decay,
           "epochs": a.epochs, "patience": a.patience}
    tr, va, te = load_split()
    labels = load_labels()
    # Built-in micro-probe: proves forward+backward+step on THIS box before
    # the 5-seed run. Any later failure is therefore in training, not setup.
    try:
        _calc, _model = build_model(device)
        _model.train()
        _opt = torch.optim.Adam(_model.parameters(), lr=1e-4)
        _h5 = h5py.File(a.hdf5, "r")
        _g = _h5["mobley_2310185"]
        _out = _model(prep(
            _calc, np.asarray(_g["atNUM"]).reshape(-1),
            np.asarray(_g["atXYZ"], dtype=np.float32).reshape(-1, 3), device))
        _loss = torch.nn.MSELoss()(
            _out["energy"].view(-1).float(),
            torch.tensor([-0.25], dtype=torch.float32, device=device))
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        del _calc, _model, _opt
        _h5.close()
        print("[microprobe] forward+backward+step OK", flush=True)
    except Exception as e:
        print(f"[microprobe] FAIL {type(e).__name__}: {e}", flush=True)
        raise
    # Center targets on the train mean (MACE's atomic-ref fit equivalent).
    # Model learns hydration-scale variations; outputs are uncentered back at
    # every boundary (metrics, dump, transfer). Gauge math is unaffected:
    # the shift is seed- and atom-independent, so sigma2/Lambda are unchanged.
    CENTER = float(np.mean([labels[m] for m in tr]))
    print(f"[center] train-mean target = {CENTER:.6f} eV "
          f"({CENTER*EV_TO_KCAL:.3f} kcal/mol)", flush=True)
    json.dump({"source": "frozen", "n_train": len(tr), "n_val": len(va),
               "n_test": len(te), "target_center_ev": CENTER},
              open(os.path.join(a.output_dir, "split_used.json"), "w"), indent=2)
    t0 = time.time()
    if not a.skip_train:
        summary = {}
        for s in tqdm(seeds, desc="OVERALL seeds", unit="seed"):
            summary[str(s)] = train_one_seed(
                s, tr, va, te, labels, CENTER, a.hdf5, device, a.output_dir,
                cfg, a.quick_test)
        summary["_target_center_ev"] = CENTER
        json.dump(summary, open(os.path.join(a.output_dir,
                                             "ensemble_summary.json"), "w"),
                  indent=2)
        print(f"[train] done in {(time.time()-t0)/60:.1f} min", flush=True)
    if not a.skip_dump:
        dump_per_atom(seeds, tr, va, te, labels, CENTER, a.hdf5, device,
                      a.output_dir, a.quick_test)
    print(f"[done] total {(time.time()-t0)/3600:.2f} h -> {a.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()

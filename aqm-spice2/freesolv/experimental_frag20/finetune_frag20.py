"""EXPERIMENTAL: fine-tune DimeNet+ with Frag20-Aqsol-100K Br/P supplement.

Throwaway experiment (advisor not yet approved). Self-contained folder -
delete experimental_frag20/ for a complete rollback. Nothing here imports
from the verified pipeline; all helpers are copied into this folder.

Design (locked by the experiment brief):
  * Same frozen fold-0 split for FreeSolv's own molecules (411 train /
    102 val / 129 test, md5 c0ef293341...) - results directly comparable
    to every previously reported number.
  * Frag20 Br/P molecules (B-rows dropped, elements confirmed from xyz)
    are added to the TRAINING set ONLY. val/test stay pure FreeSolv so the
    comparison to the baseline is not corrupted.
  * Starts from the SAME Stage-2 correction checkpoint as every other
    fine-tuning run in this project.
  * Same hyperparameters as the verified runs: lr=1e-4, wd=1e-5, batch=8,
    epochs=200, patience=30, MSE in eV, grad clip 10.0, ReduceLROnPlateau
    (f=0.5, pat=15, min_lr=1e-6), 5-conformer RDKit TTA on test.
  * DEVIATION FLAG: Frag20 targets are the SMD/B3LYP *calculated* aqueous
    solvation energies (CalcSol, kcal/mol) from the Frag20 CSVs, NOT
    experimental values. This is inherent to the dataset (no expt data) and
    is the documented level-of-theory mismatch of the experiment.

Usage:
  # full run (GPU):
  python finetune_frag20.py --mode train --device cuda
  # re-report from a saved best checkpoint without retraining:
  python finetune_frag20.py --mode eval --device cuda
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from common import (EV_TO_KCAL, DEFAULT_SEED, DEFAULT_SPLIT_DIR,
                    DEFAULT_CORRECTION_CKPT, DEFAULT_FREESOLV_CONFORMERS,
                    DEFAULT_FREESOLV_LABELS, BR_Z, P_Z,
                    evaluate, set_seed, load_frozen_split, load_freesolv_labels,
                    simple_dataset_cls, combined_dataset_cls, build_model,
                    conformer_average, md5_bytes, sha256_file)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_FRAG20_H5 = os.path.join(DATA_DIR, "frag20_brp.hdf5")
DEFAULT_FRAG20_LABELS = os.path.join(DATA_DIR, "frag20_brp_labels.json")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")

# Baseline for comparison (verified earlier in this project, kcal/mol):
# seed 42 single fine-tune on FreeSolv alone, 5-conf TTA, fold-0 test set.
BASELINE_SEED42_TTA = {"mae": 0.5048, "rmse": 0.7568, "r2": 0.9658}
BASELINE_ENSEMBLE_TTA = {"mae": 0.5059, "rmse": 0.7696, "r2": 0.9646}
BASELINE_SEED42_SINGLE_CONF = {"mae": 0.5313, "rmse": 0.7746}


def load_frag20_labels(path, max_frag20=None, rng=None):
    """Read frag20 labels JSON; returns (ids, labels) where labels[id]["expt"]
    = CalcSol kcal/mol (calculated, not experimental - flagged below)."""
    with open(path) as f:
        raw = json.load(f)
    ids = sorted(raw.keys())
    if max_frag20 is not None:
        ids = rng.choice(ids, size=min(max_frag20, len(ids)), replace=False)
        ids = sorted(ids.tolist())
    labels = {}
    for mid in ids:
        labels[mid] = {
            "expt": raw[mid]["calc_sol_kcal"],
            "smiles": raw[mid]["smiles"],
            "has_br": raw[mid]["has_br"],
            "has_p": raw[mid]["has_p"],
            "target_kind": "SMD_calculated_CalcSol_kcal",  # NOT experimental
            "source": "frag20",
        }
    return ids, labels


def train(split_dir, freesolv_h5, freesolv_labels_json, frag20_h5,
          frag20_labels_json, correction_ckpt, output_dir, epochs, patience,
          lr, batch_size, n_conformers, device_name, seed, max_frag20,
          baseline_csv):
    import h5py
    import torch
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot

    freesolv_labels = load_freesolv_labels(freesolv_labels_json)
    for mid in freesolv_labels:
        freesolv_labels[mid]["target_kind"] = "experimental_expt_kcal"

    train_ids, val_ids, test_ids = load_frozen_split(split_dir, freesolv_labels)

    # ---- Frag20 supplement (train only) ----
    rng = np.random.RandomState(seed)
    frag20_ids, frag20_labels = load_frag20_labels(
        frag20_labels_json, max_frag20=max_frag20, rng=rng)
    assert not (set(frag20_ids) & set(train_ids + val_ids + test_ids)), \
        "frag20 id collides with a freesolv id (should be impossible)"

    # combined label dict for the dataset classes
    all_labels = dict(freesolv_labels)
    all_labels.update(frag20_labels)

    combined_train_ids = train_ids + frag20_ids
    n_frag20_br = sum(1 for m in frag20_ids
                      if frag20_labels[m].get("has_br") and not frag20_labels[m].get("has_p"))
    n_frag20_p = sum(1 for m in frag20_ids
                     if frag20_labels[m].get("has_p") and not frag20_labels[m].get("has_br"))
    n_frag20_both = sum(1 for m in frag20_ids
                        if frag20_labels[m].get("has_br") and frag20_labels[m].get("has_p"))

    split_blob = b"".join(
        open(os.path.join(split_dir, name), "rb").read()
        for name in ("train_ids.json", "val_ids.json", "test_ids.json"))
    split_md5 = md5_bytes(split_blob)

    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable - falling back to cpu")
        device_name = "cpu"
    device = torch.device(device_name)

    print("\n" + "=" * 66)
    print("  EXPERIMENT: fine-tune + Frag20 Br/P supplement (seed %d)" % seed)
    print("=" * 66)
    print(f"  split: {len(train_ids)} freesolv train / {len(val_ids)} val / "
          f"{len(test_ids)} test (md5 {split_md5[:10]}...) frozen from {split_dir}")
    print(f"  FRAG20 SUPPLEMENT (TRAIN ONLY): +{len(frag20_ids)} molecules "
          f"(Br-only {n_frag20_br}, P-only {n_frag20_p}, both {n_frag20_both})")
    print(f"  combined train set: {len(combined_train_ids)}")
    print(f"  init: {correction_ckpt}")
    print(f"  hyperparams: lr={lr} wd=1e-5 batch={batch_size} epochs={epochs} "
          f"patience={patience} | MSE in eV, grad-clip 10.0, ReduceLROnPlateau "
          f"(f=0.5, pat={patience // 2}, min_lr=1e-6) | {n_conformers}-conf TTA")
    print("  TARGETS: freesolv = experimental expt (kcal); frag20 = SMD/B3LYP "
          "CalcSol (kcal, CALCULATED - level-of-theory mismatch documented)")

    set_seed(seed)
    model = build_model(device)
    ckpt = torch.load(correction_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"  loaded Stage-2 correction checkpoint "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    Dataset = combined_dataset_cls(freesolv_h5, frag20_h5, all_labels)
    train_ds = Dataset(combined_train_ids)
    FsDataset = simple_dataset_cls(freesolv_h5, all_labels)
    val_ds = FsDataset(val_ids)
    test_ds = FsDataset(test_ids)
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

    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_finetune_frag20.pt")

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
            print(f"    early stopped at epoch {epoch} (patience {patience})")
            break

    total_min = (time.time() - t0_all) / 60.0

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    model.eval()
    test_mae, test_rmse, _, _ = evaluate_loader(test_loader)

    tta_mae, tta_rmse, tta_preds = conformer_average(
        model, device, test_ids, all_labels, freesolv_h5, n_conformers, batch_size)

    metrics = {
        "kind": "EXPERIMENTAL: freesolv fold-0 + frag20 Br/P supplement",
        "seed": seed,
        "n_freesolv_train": len(train_ids), "n_freesolv_val": len(val_ids),
        "n_freesolv_test": len(test_ids),
        "n_frag20_supplement": len(frag20_ids),
        "frag20_br_only": n_frag20_br, "frag20_p_only": n_frag20_p,
        "frag20_both": n_frag20_both,
        "split_md5": split_md5,
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
        "baselines": {
            "seed42_tta": BASELINE_SEED42_TTA,
            "5seed_ensemble_tta": BASELINE_ENSEMBLE_TTA,
            "seed42_single_conf": BASELINE_SEED42_SINGLE_CONF,
        },
        "checkpoint_sha256": sha256_file(best_ckpt_path),
        "frag20_targets_note": "frag20 targets are SMD/B3LYP CalcSol (calculated, not experimental)",
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- per-molecule predictions (test only, TTA) ----
    with h5py.File(freesolv_h5, "r") as h5:
        brp_mask = {
            mid: int(any(int(z) in (BR_Z, P_Z) for z in h5[mid]["atNUM"][...]))
            for mid in test_ids
        }
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal,has_br_or_p\n")
        for mid in test_ids:
            f.write(f"{mid},{tta_preds[mid]:.6f},{all_labels[mid]['expt']:.6f},"
                    f"{brp_mask[mid]}\n")

    write_report(metrics, brp_mask, tta_preds, all_labels, test_ids,
                 output_dir, baseline_csv=baseline_csv)
    return metrics


def write_report(metrics, brp_mask, tta_preds, all_labels, test_ids,
                 output_dir, baseline_csv=None):
    """Overall + Br/P subgroup report, compared to the verified baseline."""
    expts = np.array([all_labels[m]["expt"] for m in test_ids])
    preds = np.array([tta_preds[m] for m in test_ids])
    brp = np.array([brp_mask[m] for m in test_ids], dtype=bool)

    overall_mae, overall_rmse, overall_r2 = evaluate(preds, expts)
    sub = {}
    for name, mask in (("br_p", brp), ("rest", ~brp)):
        if mask.sum() == 0:
            sub[name] = {"n": 0, "mae": None, "rmse": None, "r2": None}
            continue
        mae, rmse, r2 = evaluate(preds[mask], expts[mask])
        sub[name] = {"n": int(mask.sum()), "mae": mae, "rmse": rmse, "r2": r2}

    # optional read-only baseline comparison (deep_ensemble seed_42 predictions)
    baseline_sub = None
    if baseline_csv and os.path.exists(baseline_csv):
        with open(baseline_csv) as f:
            next(f)
            rows = [ln.rstrip("\n").split(",") for ln in f]
        base_preds = {r[0]: float(r[1]) for r in rows}
        b_preds = np.array([base_preds[m] for m in test_ids])
        b_mae, b_rmse, b_r2 = evaluate(b_preds, expts)
        b_sub = {}
        for name, mask in (("br_p", brp), ("rest", ~brp)):
            if mask.sum() == 0:
                b_sub[name] = {"n": 0, "mae": None}
                continue
            mae, _, _ = evaluate(b_preds[mask], expts[mask])
            b_sub[name] = {"n": int(mask.sum()), "mae": mae}
        baseline_sub = {"source": baseline_csv, "overall": {
            "mae": b_mae, "rmse": b_rmse, "r2": b_r2}, "subgroup": b_sub}

    delta = overall_mae - BASELINE_SEED42_TTA["mae"]
    if delta < 0:
        verdict = (f"IMPROVED over seed-42 baseline by {-delta:.3f} kcal/mol "
                   f"({BASELINE_SEED42_TTA['mae']:.3f} -> {overall_mae:.3f}). "
                   f"Frag20 Br/P supplement helped overall fold-0 test MAE.")
    elif delta > 0:
        verdict = (f"HURT vs seed-42 baseline by {delta:.3f} kcal/mol "
                   f"({BASELINE_SEED42_TTA['mae']:.3f} -> {overall_mae:.3f}). "
                   f"The supplement did NOT help overall fold-0 test MAE.")
    else:
        verdict = f"NO meaningful change vs seed-42 baseline ({overall_mae:.3f})."

    if sub["br_p"]["n"] >= 3 and baseline_sub:
        b = baseline_sub["subgroup"]["br_p"]["mae"]
        d = sub["br_p"]["mae"] - b
        verdict += (f"\n  Br/P subgroup (n={sub['br_p']['n']}): "
                    f"MAE {sub['br_p']['mae']:.3f} vs baseline {b:.3f} "
                    f"({'improved by' if d < 0 else 'worse by'} {abs(d):.3f}).")
    elif sub["br_p"]["n"] < 3:
        verdict += (f"\n  Br/P subgroup too small (n={sub['br_p']['n']}) for a firm "
                    f"conclusion - reported for completeness only.")

    report = {
        "overall": {"mae_kcal": overall_mae, "rmse_kcal": overall_rmse, "r2": overall_r2},
        "subgroup_br_p_vs_rest": sub,
        "baseline_seed42_tta": BASELINE_SEED42_TTA,
        "baseline_5seed_ensemble_tta": BASELINE_ENSEMBLE_TTA,
        "delta_vs_seed42_tta_kcal": delta,
        "baseline_subgroup_readonly": baseline_sub,
        "verdict": verdict,
        "metrics_json": "metrics.json",
        "predictions_csv": "predictions.csv",
    }
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(report_block(metrics, report))

    print("\n" + "=" * 66)
    print("  RESULTS (fold-0 fixed test set, kcal/mol)")
    print("=" * 66)
    print(f"  overall MAE {overall_mae:.3f} | RMSE {overall_rmse:.3f} | R2 {overall_r2:.4f}")
    print(f"  baseline seed42 TTA: MAE {BASELINE_SEED42_TTA['mae']:.3f} | "
          f"RMSE {BASELINE_SEED42_TTA['rmse']:.3f}")
    print(f"  baseline 5-seed ensemble TTA: MAE {BASELINE_ENSEMBLE_TTA['mae']:.3f}")
    for name in ("br_p", "rest"):
        s = sub[name]
        print(f"  {name:<6} n={s['n']:<4} MAE {s['mae'] if s['mae'] is not None else 'n/a':>7}"
              f"  RMSE {s['rmse'] if s['rmse'] is not None else 'n/a':>7}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {output_dir}")


def report_block(metrics, report):
    out = []
    out.append("EXPERIMENTAL FRAG20 SUPPLEMENT REPORT")
    out.append("=" * 60)
    out.append(f"kind: {metrics['kind']}")
    out.append(f"seed: {metrics['seed']} | split md5: {metrics['split_md5'][:10]}...")
    out.append(f"train: {metrics['n_freesolv_train']} freesolv + "
               f"{metrics['n_frag20_supplement']} frag20 (br {metrics['frag20_br_only']}, "
               f"p {metrics['frag20_p_only']}, both {metrics['frag20_both']})")
    out.append(f"val/test: {metrics['n_freesolv_val']}/{metrics['n_freesolv_test']} freesolv only")
    o = report["overall"]
    out.append(f"overall: MAE {o['mae_kcal']:.3f} RMSE {o['rmse_kcal']:.3f} R2 {o['r2']:.4f}")
    out.append(f"baseline seed42: MAE {report['baseline_seed42_tta']['mae']:.3f} | "
               f"ensemble: {report['baseline_5seed_ensemble_tta']['mae']:.3f}")
    for name in ("br_p", "rest"):
        s = report["subgroup_br_p_vs_rest"][name]
        out.append(f"subgroup {name}: n={s['n']} MAE={s['mae']}")
    out.append("")
    out.append(report["verdict"])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--freesolv_h5", default=DEFAULT_FREESOLV_CONFORMERS)
    ap.add_argument("--freesolv_labels", default=DEFAULT_FREESOLV_LABELS)
    ap.add_argument("--frag20_h5", default=DEFAULT_FRAG20_H5)
    ap.add_argument("--frag20_labels", default=DEFAULT_FRAG20_LABELS)
    ap.add_argument("--correction_ckpt", default=DEFAULT_CORRECTION_CKPT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max_frag20", type=int, default=None,
                    help="cap the number of frag20 supplement molecules (smoke tests)")
    ap.add_argument("--baseline_csv", default=None,
                    help="read-only baseline predictions.csv for subgroup comparison")
    args = ap.parse_args()

    if args.mode == "train":
        train(split_dir=args.split_dir, freesolv_h5=args.freesolv_h5,
              freesolv_labels_json=args.freesolv_labels, frag20_h5=args.frag20_h5,
              frag20_labels_json=args.frag20_labels,
              correction_ckpt=args.correction_ckpt, output_dir=args.output_dir,
              epochs=args.epochs, patience=args.patience, lr=args.lr,
              batch_size=args.batch_size, n_conformers=args.n_conformers,
              device_name=args.device, seed=args.seed, max_frag20=args.max_frag20,
              baseline_csv=args.baseline_csv)


if __name__ == "__main__":
    main()

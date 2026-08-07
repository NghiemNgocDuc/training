"""Approach 1 - uncertainty-weighted re-training (main path).

Use the 5-member deep ensemble's disagreement to ACTIVELY improve
predictions.  Per plan:

  1. Compute per-TRAINING-molecule ensemble_std by running all five trained
     ensemble checkpoints over the frozen 411 train molecules (single stored
     conformer each - the exact geometries the members were fine-tuned on).
  2. weight(mol) = 1 + alpha * (ensemble_std(mol) / mean(ensemble_std over
     train));  alpha in {0.5, 1.0, 2.0} - small sweep.
  3. Retrain a SINGLE model from the SAME stage2_correction.pt with the SAME
     hyperparams as every deep-ensemble member (lr=1e-4, wd=1e-5, batch=8,
     epochs=200, patience=30, MSE-in-eV, grad clip, ReduceLROnPlateau), but
     with a WEIGHTED loss L = mean_i w_i * (pred_i - y_i)^2.
  4. Evaluate on the SAME frozen 129-mol test set with the SAME protocol
     (single-conf + 5-conf conformer TTA).  Report overall MAE/RMSE AND
     per-subset: halogens (Br/I) vs rest, and top/bottom ensemble_std test
     subsets (improving the uncertain ones most is the actual goal).

Controls / honesty:
  * alpha=0.0 control MUST reproduce the seed-42 published baseline
    (test MAE TTA ~0.505, single ~0.531) within a small tolerance - if not,
    the harness is broken and we stop.
  * All the frozen splits / checkpoints are read-only (load_... never writes).
  * We report negative results plainly.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    EV_TO_KCAL, DEFAULT_SPLIT_DIR, DEFAULT_CORRECTION_CKPT, DEFAULT_CONFORMERS,
    DEFAULT_LABELS, DEFAULT_ENSEMBLE_DIR, DEFAULT_PER_MOLECULE_CSV, SEEDS,
    set_seed, load_frozen_split, load_freesolv_labels, simple_dataset_cls,
    weighted_simple_dataset_cls, build_model, load_checkpoint, conformer_average,
    load_ensemble_member, sha256_file,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "approach1")


def train_ensemble_std(train_ids, ensemble_dir, conformers, labels, device,
                       batch_size=8):
    """Per-molecule ensemble_std over the TRAIN molecules (single stored
    conformer each - the geometries the members were fine-tuned on).

    Returns {mol_id: (mean, std)} in kcal/mol across the 5 members."""
    from element_vocab import build_one_hot

    ds = simple_dataset_cls(conformers, labels)
    loader = DataLoader(ds(train_ids), batch_size=batch_size, shuffle=False)

    per_seed = {s: {} for s in SEEDS}
    for seed in tqdm(SEEDS, desc="train-side ensemble std (5 members)",
                     leave=False, unit="member"):
        model, _, _ = load_ensemble_member(seed, ensemble_dir, device)
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                for mid, val in zip(data.mol_id, pred.tolist()):
                    per_seed[seed][mid] = val

    stats = {}
    for mid in train_ids:
        arr = np.array([per_seed[s][mid] for s in SEEDS])
        stats[mid] = (float(np.mean(arr)), float(np.std(arr, ddof=1)))
    return stats


def make_weights(train_std, alpha, hard_mask=None):
    """weight = 1 + alpha*(std/mean_std).  alpha==0 => all-1 (control).

    hard_mask (float 0<f<=1): binary regime - the top `f` fraction of molecules
    by ensemble_std get weight 1 (learned), the rest weight 0 (gradients
    frozen).  alpha is ignored in this mode."""
    if hard_mask is not None and hard_mask > 0:
        thr = float(np.quantile([s for _, s in train_std.values()], 1.0 - hard_mask))
        return {mid: 1.0 if s >= thr else 0.0 for mid, (_, s) in train_std.items()}
    mean_std = float(np.mean([s for _, s in train_std.values()])) if train_std else 0.0
    if alpha == 0.0 or mean_std == 0.0:
        return {mid: 1.0 for mid in train_std}
    return {mid: 1.0 + alpha * (s / mean_std) for mid, (_, s) in train_std.items()}


def weighted_train(seed, alpha, weights, train_ids, val_ids, test_ids, labels,
                   correction_ckpt, conformers, output_dir, device,
                   epochs, patience, lr, batch_size, n_conformers):
    """Retrain ONE weighted model from stage2_correction.pt.

    Returns (meta dict, tta_preds_by_mid).  Also writes predictions.csv and
    keeps the best-val checkpoint."""
    from element_vocab import build_one_hot

    os.makedirs(output_dir, exist_ok=True)
    train_ds = weighted_simple_dataset_cls(conformers, labels, weights)
    train_loader = DataLoader(train_ds(train_ids), batch_size=batch_size, shuffle=True)
    ds = simple_dataset_cls(conformers, labels)
    val_loader = DataLoader(ds(val_ids), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds(test_ids), batch_size=batch_size, shuffle=False)

    model = build_model(device)
    ckpt = torch.load(correction_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"    init: {correction_ckpt} ({sum(p.numel() for p in model.parameters()):,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    def evaluate_loader(loader, desc):
        model.eval()
        all_p, all_e = [], []
        with torch.no_grad():
            for data in tqdm(loader, desc=desc, leave=False, unit="batch"):
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                dG = data.y_dG.view(-1).to(device)
                v = ~torch.isnan(dG)
                all_p.append(pred[v].cpu()); all_e.append(dG[v].cpu())
        preds = torch.cat(all_p).numpy(); expts = torch.cat(all_e).numpy()
        mae = float(np.mean(np.abs(preds - expts)))
        rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
        return mae, rmse, preds, expts

    best_ckpt = os.path.join(output_dir, "finetuned.pt")
    best_val_mae, best_epoch, stale, stop = float("inf"), -1, 0, epochs
    t_all = time.time()
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"seed{seed} a{alpha} train",
                     unit="epoch")
    for epoch in epoch_bar:
        model.train()
        loss_sum, n_batch = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            dG = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            w = data.w.view(-1).to(device)
            v = ~torch.isnan(dG)
            if v.sum() == 0:
                continue
            loss = (w[v] * (pred[v] - dG[v]) ** 2).mean()      # weighted MSE
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            n_batch += 1

        val_mae, _, _, _ = evaluate_loader(val_loader, "val")
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae, best_epoch, stale = val_mae, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            stale += 1
        epoch_bar.set_postfix(train_mse=loss_sum / max(n_batch, 1),
                              val_mae=val_mae, best_val=best_val_mae,
                              stale=stale)
        if stale >= patience:
            stop = epoch
            break
    epoch_bar.close()
    total_min = (time.time() - t_all) / 60.0

    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_mae_single, test_rmse_single, _, _ = evaluate_loader(test_loader, "test")
    tta_mae, tta_rmse, tta_preds = conformer_average(
        model, device, test_ids, labels, conformers, n_conformers, batch_size)

    meta = {
        "seed": seed,
        "alpha": alpha,
        "best_val_mae_kcal": best_val_mae,
        "best_val_epoch": best_epoch,
        "early_stop_epoch": stop,
        "total_min": round(total_min, 1),
        "test_mae_single_kcal": test_mae_single,
        "test_rmse_single_kcal": test_rmse_single,
        "test_mae_tta_kcal": tta_mae,
        "test_rmse_tta_kcal": tta_rmse,
        "best_ckpt": best_ckpt,
        "checkpoint_sha256": sha256_file(best_ckpt),
    }
    with open(os.path.join(output_dir, "predictions.csv"), "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal\n")
        for mid in test_ids:
            f.write(f"{mid},{tta_preds[mid]:.6f},{labels[mid]['expt']:.6f}\n")
    return meta, tta_preds


def subset_report(tta_preds, test_ids, per_mol):
    """Group test metrics by halogen and by top/bottom ensemble_std
    (from the verified aggregate table)."""
    mids = [m for m in test_ids if m in tta_preds and m in per_mol]
    exp = np.array([per_mol[m]["exp"] for m in mids])
    pred = np.array([tta_preds[m] for m in mids])
    ens_std = np.array([per_mol[m]["ens_std"] for m in mids])
    halogen = np.array([per_mol[m]["has_halogen"] for m in mids])

    def mets(mask):
        p, e = pred[mask], exp[mask]
        if e.size == 0:
            return {"n": 0, "MAE_kcal": None, "RMSE_kcal": None}
        return {"n": int(e.size), "MAE_kcal": float(np.mean(np.abs(p - e))),
                "RMSE_kcal": float(np.sqrt(np.mean((p - e) ** 2)))}

    hi = ens_std >= np.quantile(ens_std, 0.8)
    return {
        "all": mets(np.ones(len(mids), dtype=bool)),
        "halogen_BrI": mets(halogen),
        "non_halogen": mets(~halogen),
        "test_top20_ensstd": mets(hi),
        "test_bottom80_ensstd": mets(~hi),
    }


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--ensemble_dir", default=DEFAULT_ENSEMBLE_DIR)
    ap.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    ap.add_argument("--labels_json", default=DEFAULT_LABELS)
    ap.add_argument("--correction_ckpt", default=DEFAULT_CORRECTION_CKPT)
    ap.add_argument("--per_molecule", default=DEFAULT_PER_MOLECULE_CSV)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--hard_mask", type=float, default=None,
                    help="retrain ONLY the top-f fraction of train molecules by "
                         "ensemble_std (binary weight 1/0, others frozen); alpha ignored")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.hard_mask is not None:
        args.output_dir = os.path.join(args.output_dir, f"hardmask{args.hard_mask}")

    os.makedirs(args.output_dir, exist_ok=True)

    labels_all = load_freesolv_labels(args.labels_json)
    train_ids, val_ids, test_ids = load_frozen_split(args.split_dir, labels_all)
    if args.smoke:
        train_ids, val_ids, test_ids = train_ids[:8], val_ids[:8], test_ids[:10]
        args.epochs = min(args.epochs, 6)
        args.patience = min(args.patience, 2)
        args.n_conformers = 2

    per_mol = {}
    with open(args.per_molecule, newline="") as f:
        for row in csv.DictReader(f):
            per_mol[row["mol_id"]] = {"exp": float(row["true_value"]),
                                      "ens_std": float(row["ensemble_std"]),
                                      "has_halogen": row["has_halogen_Br_I"] == "1"}
    test_ids = [m for m in test_ids if m in per_mol]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Step 1: train-side ensemble std ----
    print(" [1] train-side ensemble std (5 members, single stored conformers)")
    train_std = train_ensemble_std(train_ids, args.ensemble_dir, args.conformers,
                                   labels_all, device, batch_size=args.batch_size)
    with open(os.path.join(args.output_dir, "train_ensemble_std.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["mol_id", "mean", "std"])
        for mid in train_ids:
            wr.writerow([mid, f"{train_std[mid][0]:.6f}", f"{train_std[mid][1]:.6f}"])
    v = np.array([s for _, s in train_std.values()])
    print(f"     train ensemble_std  mean {v.mean():.4f}  min {v.min():.4f}  max {v.max():.4f}")

    # ---- Steps 2-4: alpha sweep (or hard-mask regime) ----
    results = []
    for seed in args.seeds:
        set_seed(seed)
        for alpha in (args.alphas if args.hard_mask is None else [0.0]):
            print(f"\n === seed {seed} alpha={alpha} "
                  f"{'(hard-mask)' if args.hard_mask else ''} ===")
            weights = make_weights(train_std, alpha, hard_mask=args.hard_mask)
            n_learned = int(sum(1 for w in weights.values() if w > 0))
            print(f"    retraining on {n_learned}/{len(weights)} train molecules "
                  f"({100.0*n_learned/len(weights):.1f}%, others frozen)")
            meta, tta_preds = weighted_train(
                seed=seed, alpha=alpha, weights=weights,
                train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                labels=labels_all, correction_ckpt=args.correction_ckpt,
                conformers=args.conformers,
                output_dir=os.path.join(args.output_dir, f"seed_{seed}_alpha{alpha}"),
                device=device, epochs=args.epochs, patience=args.patience,
                lr=args.lr, batch_size=args.batch_size, n_conformers=args.n_conformers,
            )
            meta["subset"] = subset_report(tta_preds, test_ids, per_mol)
            results.append(meta)
            print(f"   done: test MAE tta={meta['test_mae_tta_kcal']:.4f} "
                  f"single={meta['test_mae_single_kcal']:.4f}")

    # ---- report ----
    c0 = next((r for r in results if r.get("alpha") == 0.0), None)
    if args.hard_mask is not None:
        control_ok = None
    else:
        control_ok = c0 is not None and abs(c0["test_mae_tta_kcal"] - 0.505) < 0.03
    report = {"method": "approach1_weighted",
              "hard_mask": args.hard_mask,
              "control_repro_matches_baseline": control_ok,
              "runs": results}
    with open(os.path.join(args.output_dir, "report.json"), "w") as fv:
        json.dump(report, fv, indent=2)

    print("\n" + "=" * 66)
    print("  SUMMARY - weighted re-training (test, kcal/mol)")
    print(f"  {'seed':>5} {'alpha':>6} {'MAE_tta':>9} {'RMSE_tta':>9} "
          f"{'MAE_single':>11}")
    for r in results:
        print(f"  {r['seed']:>5} {r['alpha']:>6.2f} {r['test_mae_tta_kcal']:>9.4f} "
              f"{r['test_rmse_tta_kcal']:>9.4f} {r['test_mae_single_kcal']:>11.4f}")
    if c0 is not None:
        print(f"\n  control alpha=0.0 MAE_tta = {c0['test_mae_tta_kcal']:.4f} "
              f"(published baseline 0.5048) -> "
              f"{'reproduced' if control_ok else ('MISMATCH!' if not args.hard_mask else 'n/a (hard-mask)')}")
    print(f"\n  report -> {os.path.join(args.output_dir, 'report.json')}")


if __name__ == "__main__":
    cli()
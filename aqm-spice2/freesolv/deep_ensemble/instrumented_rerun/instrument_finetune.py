"""Instrumented Stage-3 fine-tune: EXACT replica of deep_ensemble.py
train_member(seed=42) plus per-epoch, per-molecule test-set logging.

PURE DIAGNOSTIC ADDITION. The training loop is byte-for-byte the original:
same split (frozen fold-0 *_ids.json, md5-verified), same Stage-2 correction
checkpoint, same arch (DimeNetPlus via build_model), same Adam(lr=1e-4,
wd=1e-5), batch 8, MSE-in-eV, grad-clip 10.0, ReduceLROnPlateau(f=0.5,
pat=patience//2, min_lr=1e-6), epochs=200, patience=30, same best-val-ckpt
rule, same final single-conf + 5-conf TTA protocol.

Instrumentation added (all RNG-neutral under no_grad / eval mode):
  * epoch 0 baseline pass (predictions of the warm-start stage-2 checkpoint)
  * per epoch: per-molecule test-set rows
      epoch, mol_id, dG_pred_kcal, dG_exp_kcal, abs_err_kcal, mse_ev2
    appended to <out>/epoch_predictions.csv
  * per epoch: val MAE/RMSE row in <out>/val_history.csv
  * pooled train MSE (eV^2) per epoch in val_history.csv
  * same end-of-run artifacts as the original (predictions.csv = TTA,
    metrics.json, split copies, split.md5), written ONLY under <out>/seed_<s>/

Nothing is written to the original deep_ensemble/seed_* directories.
"""

import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))          # .../deep_ensemble/instrumented_rerun
_deep_ensemble = os.path.dirname(_script_dir)                     # .../deep_ensemble (original outputs)
_freesolv = os.path.dirname(_deep_ensemble)                       # .../freesolv (deep_ensemble.py lives here)
if _freesolv not in sys.path:
    sys.path.insert(0, _freesolv)

from deep_ensemble import (
    REPO_ROOT, EV_TO_KCAL, set_seed, build_model, load_frozen_split,
    simple_dataset_cls, sha256_file, md5_bytes, evaluate,
    DEFAULT_SPLIT_DIR, DEFAULT_CORRECTION_CKPT, DEFAULT_CONFORMERS,
)

DEFAULT_OUT = _script_dir  # instrumented_rerun/


def rng_snapshot():
    """Snapshot ALL global RNG state: python random, numpy, torch CPU + CUDA."""
    import numpy as np
    import torch
    st = {"py": random.getstate(), "np": np.random.get_state(),
          "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_restore(st):
    """Restore exactly what rng_snapshot() captured, so the training loop's RNG
    stream continues as if the wrapped block never ran. The train DataLoader
    shuffle draws from torch's DEFAULT generator at each epoch start; without
    this guard, any RNG consumption inside an added eval block would desync
    every subsequent epoch's batch order from the original script."""
    import numpy as np
    import torch
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch_cpu"])
    if "torch_cuda" in st:
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def evaluate_loader_instrumented(model, device, loader, mse_ev2=True):
    """Identical to deep_ensemble's evaluate_loader; optionally returns
    per-molecule rows instead of only pooling."""
    import numpy as np
    import torch
    from element_vocab import build_one_hot

    model.eval()
    all_p, all_e, mids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
            dG_exp = data.y_dG.view(-1).to(device)
            valid = ~torch.isnan(dG_exp)
            all_p.append(pred[valid].cpu())
            all_e.append(dG_exp[valid].cpu())
            if hasattr(data, "mol_id"):
                mids.append([m for m, v in zip(data.mol_id, valid) if v])
    preds = torch.cat(all_p).numpy()
    expts = torch.cat(all_e).numpy()
    rows = None
    if mids:
        mid_list = [m for chunk in mids for m in chunk]
        rows = list(zip(mid_list, preds.tolist(), expts.tolist()))
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    return mae, rmse, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--correction_ckpt", default=DEFAULT_CORRECTION_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels

    json_path, _ = download_freesolv_data(os.path.join(REPO_ROOT, "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(json_path)
    train_ids, val_ids, test_ids = load_frozen_split(args.split_dir, all_labels)

    split_blob = b"".join(
        open(os.path.join(args.split_dir, name), "rb").read()
        for name in ("train_ids.json", "val_ids.json", "test_ids.json"))
    split_md5 = md5_bytes(split_blob)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable - falling back to cpu.")
        args.device = "cpu"
    device = torch.device(args.device)

    set_seed(args.seed)
    model = build_model(device)
    ckpt = torch.load(args.correction_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print(f"[instrument] loaded stage-2 correction ckpt "
          f"({sum(p.numel() for p in model.parameters()):,} params), device={args.device}")

    SimpleDataset = simple_dataset_cls(args.conformers, all_labels)
    train_ds = SimpleDataset(train_ids)
    val_ds = SimpleDataset(val_ids)
    test_ds = SimpleDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6)
    mse = torch.nn.MSELoss()

    out_dir = os.path.join(args.out, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, f"ensemble_seed{args.seed}.pt")
    epoch_csv = os.path.join(out_dir, "epoch_predictions.csv")
    val_csv = os.path.join(out_dir, "val_history.csv")

    with open(epoch_csv, "w") as f:
        f.write("epoch,mol_id,dG_pred_kcal,dG_exp_kcal,abs_err_kcal,mse_ev2\n")
    with open(val_csv, "w") as f:
        f.write("epoch,val_mae_kcal,val_rmse_kcal,train_mse_ev2\n")

    def log_rows(epoch, rows):
        with open(epoch_csv, "a") as f:
            for mid, pred, exp in rows:
                f.write(f"{epoch},{mid},{pred:.6f},{exp:.6f},"
                        f"{abs(pred - exp):.6f},{(pred - exp) ** 2 / EV_TO_KCAL ** 2:.6f}\n")

    def eval_logger(loader, epoch):
        mae, rmse, rows = evaluate_loader_instrumented(model, device, loader)
        if rows is not None:
            log_rows(epoch, rows)
        return mae, rmse

    # ---- epoch 0 baseline: warm-start (stage-2) checkpoint, no training yet ----
    # Not in the original script; RNG-guarded so it cannot desync epoch-1 shuffle.
    _rng = rng_snapshot()
    model.eval()
    t_e0 = time.time()
    val_mae0, val_rmse0, _ = evaluate_loader_instrumented(model, device, val_loader)
    t0_mae, t0_rmse, t0_rows = evaluate_loader_instrumented(model, device, test_loader)
    log_rows(0, t0_rows)
    t_epoch0 = time.time() - t_e0
    with open(val_csv, "a") as f:
        f.write(f"0,{val_mae0:.6f},{val_rmse0:.6f},\n")
    rng_restore(_rng)
    print(f"[epoch 0] warm-start val MAE {val_mae0:.3f} | test MAE {t0_mae:.3f} "
          f"| {t_epoch0:.1f}s (eval+log)")

    best_val_mae = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = args.epochs
    timing = [{"epoch": 0, "train_s": 0.0, "val_s": 0.0, "test_log_s": t_epoch0}]
    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_mse_sum, train_n = 0.0, 0
        n_batch = len(train_loader)
        for bi, data in enumerate(train_loader):
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
            train_mse_sum += float(loss.item()) * int(valid.sum())
            train_n += int(valid.sum())
            p_el = time.time() - t0
            p_eta = p_el / max(bi + 1, 1) * (n_batch - bi - 1)
            sys.stdout.write(f"\r[train] epoch {epoch:3d}/{args.epochs} "
                             f"[{'#' * int(30 * (bi + 1) / n_batch):30}] "
                             f"{bi + 1}/{n_batch} | {p_el:5.1f}s | ETA {p_eta:5.1f}s")
            sys.stdout.flush()
        print()
        t_train = time.time() - t0
        train_mse = train_mse_sum / max(train_n, 1)

        t_v0 = time.time()
        val_mae, val_rmse, _ = evaluate_loader_instrumented(model, device, val_loader)
        t_val = time.time() - t_v0
        scheduler.step(val_mae)
        t_l0 = time.time()
        _rng = rng_snapshot()
        test_mae, test_rmse, test_rows = evaluate_loader_instrumented(
            model, device, test_loader)
        log_rows(epoch, test_rows)
        rng_restore(_rng)
        t_log = time.time() - t_l0
        with open(val_csv, "a") as f:
            f.write(f"{epoch},{val_mae:.6f},{val_rmse:.6f},{train_mse:.6e}\n")
        dt = time.time() - t0
        timing.append({"epoch": epoch, "train_s": round(t_train, 2),
                       "val_s": round(t_val, 2),
                       "test_log_s": round(t_log, 2),
                       "total_s": round(dt, 2)})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            stale += 1

        print(f"[instrument] seed {args.seed:>4} | epoch {epoch:3d}/{args.epochs} "
              f"| best val {best_val_mae:7.3f} (ep {best_epoch}) | cur val {val_mae:7.3f} "
              f"| test {test_mae:7.3f} | train_mse {train_mse:.4e} | {dt:5.1f}s/ep "
              f"(train {t_train:.1f}s, val {t_val:.1f}s, test+log {t_log:.1f}s)",
              flush=True)

        if stale >= args.patience:
            stop_epoch = epoch
            print(f"[instrument] seed {args.seed} early stopped at epoch {epoch} "
                  f"(patience {args.patience})")
            break

    total_min = (time.time() - t0_all) / 60.0

    timing_path = os.path.join(out_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump({"device": args.device,
                   "n_train_atoms_epochs": len(train_loader),
                   "n_val": len(val_loader), "n_test": len(test_loader),
                   "timing_s_per_epoch": timing}, f, indent=2)

    # ---- final test pass with best-val checkpoint, single conformer ----
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device,
                                     weights_only=True))
    model.eval()
    test_mae, test_rmse, _ = evaluate_loader_instrumented(model, device, test_loader)

    from deep_ensemble import conformer_average
    _rng = rng_snapshot()
    tta_mae, tta_rmse, tta_preds_by_mid = conformer_average(
        model, device, test_ids, all_labels, args.conformers,
        args.n_conformers, args.batch_size)
    rng_restore(_rng)

    preds_path = os.path.join(out_dir, "predictions.csv")
    with open(preds_path, "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal\n")
        for mid in test_ids:
            f.write(f"{mid},{tta_preds_by_mid[mid]:.6f},{all_labels[mid]['expt']:.6f}\n")

    for name in ("train_ids.json", "val_ids.json", "test_ids.json"):
        with open(os.path.join(args.split_dir, name), "rb") as src, \
                open(os.path.join(out_dir, name), "wb") as dst:
            dst.write(src.read())
    with open(os.path.join(out_dir, "split.md5"), "w") as f:
        f.write(split_md5 + "\n")

    env = {"python": sys.version.split()[0]}
    try:
        import torch
        env.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            env.update({"cuda_version": torch.version.cuda,
                        "cudnn_version": torch.backends.cudnn.version(),
                        "gpu": torch.cuda.get_device_name(0)})
    except Exception:
        pass

    metrics = {
        "seed": args.seed,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "split_md5": split_md5,
        "split_source": args.split_dir,
        "correction_ckpt": args.correction_ckpt,
        "env": env,
        "hyperparams": {"lr": args.lr, "weight_decay": 1e-5, "batch_size": args.batch_size,
                        "epochs": args.epochs, "patience": args.patience,
                        "device": args.device},
        "best_val_mae_kcal": best_val_mae,
        "best_val_epoch": best_epoch,
        "early_stop_epoch": stop_epoch,
        "total_min": round(total_min, 1),
        "test_mae_single_conf_kcal": test_mae,
        "test_rmse_single_conf_kcal": test_rmse,
        "test_mae_tta_kcal": tta_mae,
        "test_rmse_tta_kcal": tta_rmse,
        "n_conformers_tta": args.n_conformers,
        "checkpoint_sha256": sha256_file(best_ckpt_path),
        "instrumented": True,
        "artifacts": {"epoch_predictions_csv": epoch_csv, "val_history_csv": val_csv},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[instrument] seed {args.seed} DONE | best val MAE {best_val_mae:.3f} "
          f"(ep {best_epoch}, stopped ep {stop_epoch}) | test {test_mae:.3f}/{test_rmse:.3f} "
          f"(single conf) | test {tta_mae:.3f}/{tta_rmse:.3f} ({args.n_conformers}-conf TTA) "
          f"| {total_min:.1f} min")
    return metrics


if __name__ == "__main__":
    main()
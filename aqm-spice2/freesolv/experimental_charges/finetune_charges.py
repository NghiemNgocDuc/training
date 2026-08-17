"""Fine-tune on FreeSolv experimental data with optional partial charges.

Charges variant: build_one_hot appends cached Gasteiger charges after the
17-dim one-hot; the copied HybridEmbeddingBlock consumes them as continuous
features. The stage-2 checkpoint's embedding block is re-initialized in the
charges variant (new continuous channel), everything else is loaded.

Usage (plain reproduction, mirrors verified fold-0 run):
  python finetune_charges.py --no_charges --seed 42

Usage (charges variant):
  python finetune_charges.py --charges --seed 42
"""

import argparse
import csv
import json
import os

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import common
from element_vocab import build_one_hot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", default=common.DEFAULT_FREESOLV_CONFORMERS)
    parser.add_argument("--checkpoint_dir", default=os.path.dirname(
        common.DEFAULT_CORRECTION_CKPT))
    parser.add_argument("--correction_ckpt", default=os.path.basename(
        common.DEFAULT_CORRECTION_CKPT))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cache_dir", default=os.path.join(
        common.REPO_ROOT, "Data", "FreeSolv"))
    parser.add_argument("--split_dir", default=common.DEFAULT_SPLIT_DIR)
    parser.add_argument("--charges_json", default=common.DEFAULT_CHARGES_JSON)
    parser.add_argument("--charges", action="store_true",
                        help="attach Gasteiger partial charges as features")
    parser.add_argument("--no_charges", action="store_true",
                        help="plain reproduction (no charges)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta_conformers", type=int, default=4,
                        help="conformer TTA at final evaluation (0 = stored conformer only)")
    args = parser.parse_args()

    if args.charges and args.no_charges:
        raise SystemExit("pick one of --charges / --no_charges")
    use_charges = args.charges and not args.no_charges
    mode = "charges" if use_charges else "plain"

    common.set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device} | mode: {mode} | seed: {args.seed}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", mode, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Labels, conformers, split, charges ──
    json_path = os.path.join(args.cache_dir, "database.json")
    all_labels = common.load_freesolv_labels(json_path)
    train_ids, val_ids, test_ids = common.load_frozen_split(args.split_dir, all_labels)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    charges = common.load_charges(args.charges_json) if use_charges else None
    n_charged = 0 if charges is None else sum(
        1 for m in train_ids + val_ids + test_ids if m in charges)
    print(f"charges available: {n_charged}")

    # ── Model + checkpoint ──
    model = common.build_model(device, use_charges)
    ckpt_path = os.path.join(args.checkpoint_dir, args.correction_ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if use_charges:
        # The embedding block's continuous_lin changes input dim (16 redundant
        # one-hot columns -> 1 charge column), so only that layer is dropped
        # from the checkpoint and re-initialized; atom_embedding / lin_rbf /
        # lin are shared with the pretrained model and are kept.
        state = {k: v for k, v in state.items()
                 if not k.startswith("emb.continuous_lin.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("emb.continuous_lin.")]
        unexpected = [k for k in unexpected if not k.startswith("emb.continuous_lin.")]
        assert not missing, f"unexpected missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        model.emb.continuous_lin.reset_parameters()
        print("charges variant: emb.continuous_lin re-initialized (1->128), rest loaded from stage-2 ckpt")
    else:
        model.load_state_dict(state)
        print("plain variant: full stage-2 ckpt loaded")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Datasets / loaders ──
    DS = common.dataset_cls(args.conformers, all_labels, charges)
    train_ds = DS(train_ids)
    val_ds = DS(val_ids)
    test_ds = DS(test_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2,
        min_lr=args.min_lr)
    mse = torch.nn.MSELoss()

    ckpt_name = f"finetuned_{mode}_seed{args.seed}.pt"

    # ── Training (early stopping on val only) ──
    best_val_mae = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            dG_exp = data.y_dG.view(-1).to(device) / common.EV_TO_KCAL
            valid = ~torch.isnan(dG_exp)
            if valid.sum() == 0:
                continue
            loss = mse(pred[valid], dG_exp[valid])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * valid.sum().item()

        model.eval()
        val_preds, val_expts = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * common.EV_TO_KCAL
                dG_exp = data.y_dG.view(-1).to(device)
                valid = ~torch.isnan(dG_exp)
                val_preds.append(pred[valid].cpu())
                val_expts.append(dG_exp[valid].cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_expts = torch.cat(val_expts).numpy()
        val_mae = float(np.mean(np.abs(val_preds - val_expts)))
        val_rmse = float(np.sqrt(np.mean((val_preds - val_expts) ** 2)))
        train_loss_avg = train_loss / len(train_ids)

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | Train loss: {train_loss_avg:.6f} "
              f"| Val MAE: {val_mae:.3f} RMSE: {val_rmse:.3f} | LR: {current_lr:.2e}",
              flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, ckpt_name))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Best val MAE: {best_val_mae:.3f} kcal/mol at epoch {best_epoch}")
    print(f"{'='*60}")

    # ── Final evaluation ──
    model.load_state_dict(torch.load(os.path.join(out_dir, ckpt_name),
                                     map_location=device))
    model.eval()

    if args.tta_conformers > 0:
        print(f"\nConformer TTA evaluation (n={args.tta_conformers})...")
        tta_mae, tta_rmse, tta_preds = common.conformer_average(
            model, device, test_ids, all_labels, args.conformers,
            args.tta_conformers, args.batch_size, charges=charges)
        print(f"TTA test MAE: {tta_mae:.3f} RMSE: {tta_rmse:.3f} (n={len(test_ids)})")
    else:
        tta_preds, tta_mae, tta_rmse = None, None, None

    # Stored-conformer single-shot on the test set
    test_preds, test_expts = [], []
    with torch.no_grad():
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1) * common.EV_TO_KCAL
            dG_exp = data.y_dG.view(-1).to(device)
            valid = ~torch.isnan(dG_exp)
            test_preds.append(pred[valid].cpu())
            test_expts.append(dG_exp[valid].cpu())
    test_preds = torch.cat(test_preds).numpy()
    test_expts = torch.cat(test_expts).numpy()
    mae = float(np.mean(np.abs(test_preds - test_expts)))
    rmse = float(np.sqrt(np.mean((test_preds - test_expts) ** 2)))
    print(f"\nStored-conformer test MAE: {mae:.3f} RMSE: {rmse:.3f} (n={len(test_preds)})")

    # ── Full 642-molecule prediction (all stored conformers) ──
    full_ds = DS(mol_ids_for_full(train_ids, val_ids, test_ids))
    preds_by_mid = {}
    with torch.no_grad():
        loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
        for data in loader:
            mids = list(data.mol_id)
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1) * common.EV_TO_KCAL
            for mid, p in zip(mids, pred.cpu().tolist()):
                preds_by_mid[mid] = p

    if tta_preds is not None:
        preds_by_mid.update(tta_preds)

    per_mol = os.path.join(out_dir, "per_molecule.csv")
    with open(per_mol, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", f"pred_seed{args.seed}", "ensemble_mean",
                    "ensemble_std", "true_value", "abs_error", "has_halogen_Br_I"])
        for mid in preds_by_mid:
            pred = preds_by_mid[mid]
            expt = all_labels[mid]["expt"]
            smiles = all_labels[mid].get("smiles", "")
            br_i = 1 if ("Br" in smiles or "I" in smiles) else 0
            w.writerow([mid, f"{pred:.6f}", f"{pred:.6f}", "nan",
                        f"{expt:.6f}", f"{abs(pred - expt):.6f}", br_i])
    print(f"Saved per-molecule predictions -> {per_mol}")

    meta = {
        "mode": mode, "seed": args.seed, "best_val_mae": best_val_mae,
        "best_epoch": best_epoch, "test_mae_stored_conf": mae,
        "test_rmse_stored_conf": rmse,
        "test_mae_tta": tta_mae, "test_rmse_tta": tta_rmse,
        "checkpoint": os.path.join(out_dir, ckpt_name),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def mol_ids_for_full(train_ids, val_ids, test_ids):
    seen, out = set(), []
    for ids in (train_ids, val_ids, test_ids):
        for m in ids:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


if __name__ == "__main__":
    main()
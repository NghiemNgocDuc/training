import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, EV_TO_KCAL, MACE_MODEL_SIZE, MACE_R_MAX, MACE_MAX_NEIGHBORS, FREEZE_ATOMIC_ENERGIES, FREEZE_INTERACTIONS, HDF5_PATH
from config import EPOCHS, LR, LR_MIN, WEIGHT_DECAY, BATCH_SIZE, PATIENCE, SEED, N_FOLDS
from config import WARMUP_EPOCHS, LOSS_TYPE, USE_LORA, LORA_RANK, LORA_ALPHA, LORA_UNFREEZE_READOUTS, LORA_UNFREEZE_SKIP_TP
from train import run_cv, evaluate, validate, compute_target_stats
from data import MACEFreeSolvDataset, collate_mace, get_labels
from model import MACEFreeSolv


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="MACE-OFF23 fine-tuning for FreeSolv")
    parser.add_argument("--quick_test", action="store_true", help="2 epochs, 2 folds")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--freeze_interactions", action="store_true", default=FREEZE_INTERACTIONS, help="Freeze interaction blocks")
    parser.add_argument("--freeze_atomic_energies", action="store_true", default=FREEZE_ATOMIC_ENERGIES, help="Freeze atomic reference energies")
    parser.add_argument("--no_freeze_atomic_energies", action="store_false", dest="freeze_atomic_energies", help="Train atomic reference energies")
    parser.add_argument("--model_size", type=str, default=MACE_MODEL_SIZE, choices=["small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr_min", type=float, default=LR_MIN)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--r_max", type=float, default=MACE_R_MAX)
    parser.add_argument("--max_neighbors", type=int, default=MACE_MAX_NEIGHBORS)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_only", type=str, default=None, help="Evaluate a saved checkpoint")
    parser.add_argument("--eval_fold", type=int, default=None, help="Fold index for eval_only (reconstructs the same CV split)")
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS, help="LR warmup epochs")
    parser.add_argument("--loss_type", type=str, default=LOSS_TYPE, choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Huber loss delta (in eV)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0=main process)")
    parser.add_argument("--use_lora", action="store_true", default=USE_LORA, help="Enable LoRA")
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=LORA_ALPHA, help="LoRA alpha scaling")
    parser.add_argument("--lora_unfreeze_readouts", action="store_true", default=LORA_UNFREEZE_READOUTS, help="Unfreeze readout base weights (hybrid)")
    parser.add_argument("--no_lora_unfreeze_readouts", action="store_false", dest="lora_unfreeze_readouts", help="Keep readout base weights frozen")
    parser.add_argument("--lora_unfreeze_skip_tp", action="store_true", default=LORA_UNFREEZE_SKIP_TP, help="Unfreeze skip_tp weights (hybrid)")
    parser.add_argument("--no_lora_unfreeze_skip_tp", action="store_false", dest="lora_unfreeze_skip_tp", help="Keep skip_tp weights frozen")
    parser.add_argument("--no_seed", action="store_true", help="Disable deterministic seeding")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Start from a Stage-A (AQM) checkpoint instead of raw foundation weights")
    args = parser.parse_args()

    if not args.no_seed:
        set_seed(args.seed)

    if args.quick_test:
        args.epochs = 2
        args.n_folds = 2

    if args.eval_only:
        eval_checkpoint(args.eval_only, args)
        return

    fold_maes, fold_rmses = run_cv(args)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: MACE-OFF23 ({args.model_size}) on FreeSolv")
    print(f"{'='*60}")
    maes = [m * EV_TO_KCAL for m in fold_maes]
    rmses = [r * EV_TO_KCAL for r in fold_rmses]
    print(f"  Per-fold MAE: {[f'{m:.3f}' for m in maes]}")
    print(f"  Per-fold RMSE: {[f'{r:.3f}' for r in rmses]}")
    print(f"  Mean MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f} kcal/mol")
    print(f"  Mean RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f} kcal/mol")

    print(f"\n{'='*60}")
    print(f"  COMPARISON WITH PUBLISHED METHODS")
    print(f"{'='*60}")
    print(f"  {'Method':<32} {'MAE':<10} {'RMSE':<10}")
    print(f"  {'-'*32} {'-'*10} {'-'*10}")
    refs = [
        ("Zhang 2022 (A3D-PNAConv-FT)", 0.417, 0.719),
        ("COSMO-RS (Klamt 2015)", 0.52, None),
        ("ReSolv (Röcken 2024)", 0.63, 0.96),
        ("DimeNet++ correction (this repo)", 0.52, 0.84),
    ]
    for name, mae_val, rmse_val in refs:
        m_str = f"{mae_val:.3f}" if mae_val is not None else "—"
        r_str = f"{rmse_val:.3f}" if rmse_val is not None else "—"
        print(f"  {name:<34} {m_str:<10} {r_str:<10}")
    mean_mae = np.mean(maes) if maes else 0.0
    mean_rmse = np.mean(rmses) if rmses else 0.0
    print(f"  {'MACE-OFF23 fine-tuned (this work)':<32} {mean_mae:<10.3f} {mean_rmse:<10.3f}")


def _load_fold_metadata(model_dir):
    path = os.path.join(model_dir, "fold_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _get_sorted_mol_ids(r_max=5.0, max_neighbors=32):
    labels = get_labels()
    with h5py.File(HDF5_PATH, "r") as f:
        all_mol_ids = [m for m in f.keys()
                       if m in labels and isinstance(labels[m].get("expt"), (int, float))]
    all_expts = np.array([labels[m]["expt"] for m in all_mol_ids])
    sort_idx = np.argsort(all_expts)
    return [all_mol_ids[i] for i in sort_idx], all_expts[sort_idx]


def eval_checkpoint(checkpoint_path, args):
    device = torch.device(args.device if args.device else "cpu")
    print(f"Evaluating: {checkpoint_path}")

    model = MACEFreeSolv(
        model_size=args.model_size, device=device, fit_refs=False,
        use_lora=args.use_lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_unfreeze_readouts=args.lora_unfreeze_readouts,
        lora_unfreeze_skip_tp=args.lora_unfreeze_skip_tp,
    ).to(device)
    model.load(checkpoint_path)
    model.eval()

    meta = _load_fold_metadata(os.path.dirname(checkpoint_path)) if os.path.isdir(os.path.dirname(checkpoint_path)) else None
    fold_index = args.eval_fold
    n_folds = args.n_folds
    if meta is not None:
        n_folds = meta["n_folds"]
        if fold_index is None:
            fold_index = meta.get("fold_index")
            print(f"  Read fold index {fold_index} from checkpoint metadata")
        else:
            print(f"  Using --eval_fold {fold_index} (metadata has fold_index={meta.get('fold_index')})")

    if fold_index is None:
        ds = MACEFreeSolvDataset(r_max=args.r_max, max_neighbors=args.max_neighbors, targets_in_ev=True)
        print("  Warning: no --eval_fold specified — evaluating on full dataset (includes training data)")
    else:
        mol_ids_sorted, _ = _get_sorted_mol_ids(r_max=args.r_max, max_neighbors=args.max_neighbors)
        fold_mol_ids = [[] for _ in range(n_folds)]
        for i, mol_id in enumerate(mol_ids_sorted):
            fold_mol_ids[i % n_folds].append(mol_id)
        test_ids = fold_mol_ids[fold_index]
        ds = MACEFreeSolvDataset(
            mol_ids=test_ids, r_max=args.r_max, max_neighbors=args.max_neighbors,
            targets_in_ev=True,
        )
        print(f"  Fold {fold_index} test set: {len(ds)} molecules (round-robin from sorted targets)")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mace, num_workers=args.num_workers)

    _, _, _, all_preds, all_expts = validate(model, loader, device)
    all_preds_kcal = all_preds * EV_TO_KCAL
    all_expts_kcal = all_expts * EV_TO_KCAL

    mae, rmse, r2 = evaluate(all_preds_kcal, all_expts_kcal)
    print(f"\nResults (in kcal/mol):")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  N:    {len(all_preds)}")


if __name__ == "__main__":
    main()

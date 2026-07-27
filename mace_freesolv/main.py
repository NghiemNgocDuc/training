import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, EV_TO_KCAL, MACE_MODEL_SIZE, MACE_R_MAX, MACE_MAX_NEIGHBORS
from config import EPOCHS, LR, LR_MIN, WEIGHT_DECAY, BATCH_SIZE, PATIENCE, VAL_SPLIT, SEED, N_FOLDS
from train import run_cv, evaluate, validate
from data import MACEFreeSolvDataset, collate_mace
from model import MACEFreeSolv


def main():
    parser = argparse.ArgumentParser(description="MACE-OFF23 fine-tuning for FreeSolv")
    parser.add_argument("--quick_test", action="store_true", help="2 epochs, 2 folds")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--freeze_interactions", action="store_true", help="Freeze interaction blocks")
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
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 2
        args.n_folds = 2

    if args.eval_only:
        eval_checkpoint(args.eval_only, args)
        return

    fold_results = run_cv(args)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: MACE-OFF23 ({args.model_size}) on FreeSolv")
    print(f"{'='*60}")
    maes = [m * EV_TO_KCAL for m in fold_results]
    print(f"  Per-fold MAE: {[f'{m:.3f}' for m in maes]}")
    print(f"  Mean MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f} kcal/mol")

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
    mean_str = f"{np.mean(maes):.3f}" if maes else "-"
    print(f"  {'MACE-OFF23 fine-tuned (this work)':<32} {mean_str:<10} {'-':<10}")


def eval_checkpoint(checkpoint_path, args):
    device = torch.device(args.device if args.device else "cpu")
    print(f"Evaluating: {checkpoint_path}")

    model = MACEFreeSolv(model_size=args.model_size, device=device).to(device)
    model.load(checkpoint_path)
    model.eval()

    ds = MACEFreeSolvDataset(targets_in_ev=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mace, num_workers=0)

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

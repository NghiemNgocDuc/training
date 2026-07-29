import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data import MACEFreeSolvDataset, collate_mace
from model import MACEFreeSolv

EV_TO_KCAL = 23.0605


def evaluate(preds, expts):
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    ss_res = np.sum((preds - expts) ** 2)
    ss_tot = np.sum((expts - expts.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mae, rmse, r2


def compute_target_stats(dataset):
    targets = torch.tensor([s[1] for s in dataset.samples], dtype=torch.float32)
    return targets.mean().item(), targets.std().item()


class WarmupWrapper:
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * factor

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        y_true = batch.pop("y").view(-1)
        optimizer.zero_grad()
        y_pred = model(batch, compute_force=False).view(-1)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item() * y_true.size(0)
        n_samples += y_true.size(0)
    return total_loss / n_samples if n_samples > 0 else 0.0


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_expts = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        y_true = batch.pop("y").view(-1)
        y_pred = model(batch, compute_force=False).view(-1)
        all_preds.append(y_pred.cpu())
        all_expts.append(y_true.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_expts = torch.cat(all_expts).numpy()
    mae, rmse, r2 = evaluate(all_preds, all_expts)
    return mae, rmse, r2, all_preds, all_expts


def run_fold(train_ids, test_ids, fold, output_dir, device, cfg):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    train_ds = MACEFreeSolvDataset(
        mol_ids=train_ids,
        r_max=cfg["r_max"],
        max_neighbors=cfg["max_neighbors"],
        targets_in_ev=True,
    )
    test_ds = MACEFreeSolvDataset(
        mol_ids=test_ids,
        r_max=cfg["r_max"],
        max_neighbors=cfg["max_neighbors"],
        targets_in_ev=True,
    )

    num_w = cfg.get("num_workers", 0)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_mace,
        num_workers=num_w, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_mace,
        num_workers=num_w, pin_memory=pin,
    )

    t_mean_kcal, t_std_kcal = compute_target_stats(train_ds)
    print(f"  Target stats: mean={t_mean_kcal:.4f} kcal/mol, std={t_std_kcal:.4f} kcal/mol")

    model = MACEFreeSolv(
        model_size=cfg["model_size"],
        device=device,
        freeze_atomic_energies=cfg.get("freeze_atomic_energies", False),
        target_mean=0.0,
        target_std=t_std_kcal,
    ).to(device)
    if cfg.get("freeze_interactions"):
        model.freeze_interactions()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["patience"] // 2, min_lr=cfg["lr_min"],
    )
    warmup = WarmupWrapper(optimizer, cfg.get("warmup_epochs", 10), cfg["lr"])

    if cfg.get("loss_type") == "huber":
        loss_fn = torch.nn.HuberLoss(delta=cfg.get("huber_delta", 1.0))
    else:
        loss_fn = torch.nn.MSELoss()

    best_mae = float("inf")
    best_epoch = -1
    stale = 0
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        warmup.step()
        val_mae, val_rmse, val_r2, val_preds, val_expts = validate(model, test_loader, device)
        scheduler.step(val_mae)
        current_lr = warmup.get_lr()

        val_mae_kcal = val_mae * EV_TO_KCAL
        val_rmse_kcal = val_rmse * EV_TO_KCAL
        elapsed = time.time() - t0

        print(f"    Epoch {epoch:3d}/{cfg['epochs']} | Loss: {train_loss:.6f} | "
              f"MAE: {val_mae_kcal:.3f} RMSE: {val_rmse_kcal:.3f} R²: {val_r2:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            stale = 0
            ckpt_path = os.path.join(fold_dir, "model.pt")
            model.save(ckpt_path)
            np.savez(os.path.join(fold_dir, "test_preds.npz"), preds=val_preds, expts=val_expts)
            print(f"    [*] Best model saved (MAE={val_mae_kcal:.3f} kcal/mol)")
        else:
            stale += 1

        if stale >= cfg["patience"]:
            print(f"    Early stopping at epoch {epoch}")
            break

    best_mae_kcal = best_mae * EV_TO_KCAL
    print(f"\n  Fold {fold} best: MAE={best_mae_kcal:.3f} kcal/mol at epoch {best_epoch}")
    return best_mae, best_epoch


def run_cv(args):
    device = torch.device(args.device if args.device else "cpu")
    print(f"Device: {device}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = {
        "model_size": args.model_size,
        "r_max": args.r_max,
        "max_neighbors": args.max_neighbors,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_folds": args.n_folds,
        "freeze_interactions": args.freeze_interactions,
        "freeze_atomic_energies": args.freeze_atomic_energies,
        "warmup_epochs": args.warmup_epochs,
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "num_workers": args.num_workers,
    }

    full_ds = MACEFreeSolvDataset(r_max=args.r_max, max_neighbors=args.max_neighbors, targets_in_ev=True)
    all_mol_ids = [s[0] for s in full_ds.samples]
    all_expts_all = np.array([s[1] for s in full_ds.samples])
    sort_idx = np.argsort(all_expts_all)
    mol_ids_sorted = [all_mol_ids[i] for i in sort_idx]

    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=args.seed)
    fold_results = []

    for fold in range(cfg["n_folds"]):
        train_idx, test_idx = list(kf.split(mol_ids_sorted))[fold]
        train_ids = [mol_ids_sorted[i] for i in train_idx]
        test_ids = [mol_ids_sorted[i] for i in test_idx]
        best_mae, best_epoch = run_fold(train_ids, test_ids, fold, args.output_dir, device, cfg)
        fold_results.append(best_mae)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for f, m in enumerate(fold_results):
        print(f"  Fold {f}: MAE = {m*EV_TO_KCAL:.3f} kcal/mol")
    mean_mae = np.mean(fold_results) * EV_TO_KCAL
    std_mae = np.std(fold_results) * EV_TO_KCAL
    print(f"  Mean ± std: {mean_mae:.3f} ± {std_mae:.3f} kcal/mol")
    return fold_results

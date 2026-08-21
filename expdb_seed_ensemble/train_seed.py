"""Train ONE stage-3 seed of the fold-0 model (Exp-DB seed ensemble).

Replicates the expdb_vast recipe exactly (DimeNet+, hidden 128, blocks 3,
cutoff 6.0, Adam lr 1e-4 / wd 1e-5, batch 8, 200 epochs, patience 30,
ReduceLROnPlateau 0.5, grad clip 10, MSE in eV) with two additions:
  * all RNG streams pinned to --seed (torch / numpy / random / cuda)
  * the ARCHIVED fold-0 split jsons are loaded directly (byte-identical
    split; no re-derivation)

Usage: python train_seed.py --seed 42 [--epochs 200] [--quick]
"""

import argparse
import json
import os
import time

import numpy as np

import common_io as cio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    cio.set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_seed {args.seed}] device={device}", flush=True)

    labels = cio.load_labels()
    train_ids = json.load(open(cio.path_split("train")))
    val_ids = json.load(open(cio.path_split("val")))
    h5_free = cio.path_freesolv_h5()
    print(f"[train_seed {args.seed}] fold-0 archived split: "
          f"train={len(train_ids)} val={len(val_ids)}", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_seeds")
    os.makedirs(out_dir, exist_ok=True)
    ck_path = os.path.join(out_dir, f"finetuned_seed{args.seed}.pt")

    class SimpleDS:
        def __init__(self, ids):
            self.ids = ids
            self._c = {}

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            import h5py
            mid = self.ids[i]
            if mid not in self._c:
                with h5py.File(h5_free, "r") as f:
                    g = f[mid]
                    d = Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                             pos=torch.tensor(g["atXYZ"][...], dtype=torch.float))
                self._c[mid] = d.clone()
            d = self._c[mid].clone()
            d.y_dG = torch.tensor([labels[mid]["expt"]], dtype=torch.float)
            return d

    train_loader = DataLoader(SimpleDS(train_ids), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(SimpleDS(val_ids), batch_size=args.batch_size,
                            shuffle=False)

    model = cio.build_model(device)
    state = torch.load(cio.path_stage2(), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[train_seed {args.seed}] init from stage2_correction.pt "
          f"(random-init {len(missing)} params)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, args.patience // 2),
        min_lr=1e-6)
    mse = torch.nn.MSELoss()

    def eval_loader(loader):
        model.eval()
        P, E = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                from element_vocab import NUM_ELEMENTS
                x = torch.zeros(data.z.shape[0], NUM_ELEMENTS, device=device)
                x[torch.arange(data.z.shape[0], device=device), data.z] = 1.0
                pred = model(x, data.pos, data.batch).view(-1) * cio.EV_TO_KCAL
                y = data.y_dG.view(-1).to(device)
                ok = ~torch.isnan(y)
                P.append(pred[ok].cpu())
                E.append(y[ok].cpu())
        p = torch.cat(P).numpy()
        e = torch.cat(E).numpy()
        return float(np.mean(np.abs(p - e))), float(np.sqrt(np.mean((p - e) ** 2)))

    epochs = 2 if args.quick else args.epochs
    patience = 2 if args.quick else args.patience
    best_val, best_epoch, stale = float("inf"), -1, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        n_batches = 0
        for data in train_loader:
            data = data.to(device)
            from element_vocab import NUM_ELEMENTS
            x = torch.zeros(data.z.shape[0], NUM_ELEMENTS, device=device)
            x[torch.arange(data.z.shape[0], device=device), data.z] = 1.0
            pred = model(x, data.pos, data.batch).view(-1)
            y = data.y_dG.view(-1).to(device) / cio.EV_TO_KCAL
            ok = ~torch.isnan(y)
            if ok.sum() == 0:
                continue
            loss = mse(pred[ok], y[ok])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            n_batches += 1
        val_mae, val_rmse = eval_loader(val_loader)
        sched.step(val_mae)
        star = ""
        if val_mae < best_val:
            best_val, best_epoch, stale = val_mae, epoch, 0
            torch.save(model.state_dict(), ck_path)
            star = "  <- best"
        else:
            stale += 1
        print(f"[train_seed {args.seed}] epoch {epoch:3d}/{epochs} "
              f"({n_batches} batches) val MAE {val_mae:.3f} RMSE {val_rmse:.3f}"
              f"{star}", flush=True)
        if stale >= patience:
            print(f"[train_seed {args.seed}] early stop at epoch {epoch}",
                  flush=True)
            break

    model.load_state_dict(torch.load(ck_path, map_location=device,
                                     weights_only=True))
    val_mae, val_rmse = eval_loader(val_loader)
    meta = {"seed": args.seed, "best_val_mae": best_val,
            "best_epoch": best_epoch, "final_val_mae": val_mae,
            "final_val_rmse": val_rmse, "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, f"train_meta_seed{args.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train_seed {args.seed}] DONE best val MAE {best_val:.3f} "
          f"@ epoch {best_epoch} ({meta['runtime_s']}s) -> {ck_path}",
          flush=True)


if __name__ == "__main__":
    main()

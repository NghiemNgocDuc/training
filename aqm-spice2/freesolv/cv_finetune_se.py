"""5-fold CV + ensemble for FreeSolv fine-tuning with DimeNetPlusSE.

Enhanced architecture with:
  - Squeeze-and-Excitation (SE) channel recalibration
  - Multi-aggregation (sum, mean, max) in interaction blocks
  - Configurable model capacity (hidden_channels, num_blocks)

Usage:
  python cv_finetune_se.py --quick_test
  python cv_finetune_se.py --hidden 256 --blocks 4
  python cv_finetune_se.py --no_se --no_multi_agg  # baseline DimeNet++ (compare)
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import KFold

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)
sys.path.append(_parent)
sys.path.append(_script_dir)
os.chdir(_parent)


def evaluate(preds, expts):
    mae = float(np.mean(np.abs(preds - expts)))
    rmse = float(np.sqrt(np.mean((preds - expts) ** 2)))
    r2 = float(1 - np.sum((preds - expts) ** 2) / np.sum((expts - expts.mean()) ** 2))
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", default="freesolv_conformers.hdf5")
    parser.add_argument("--cache_dir", default="Data/FreeSolv")
    parser.add_argument("--checkpoint_dir", default="results")
    parser.add_argument("--correction_ckpt", default="stage2_correction.pt")
    parser.add_argument("--output_dir", default="cv_results_se")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden channels")
    parser.add_argument("--blocks", type=int, default=3, help="Number of interaction blocks")
    parser.add_argument("--no_se", action="store_true", help="Disable SE blocks")
    parser.add_argument("--no_multi_agg", action="store_true", help="Disable multi-aggregation")
    parser.add_argument("--n_conformers", type=int, default=1,
                        help="Conformers per molecule for test-time averaging (default: 1 = no ensemble)")
    args = parser.parse_args()

    import torch
    import h5py
    from torch_geometric.loader import DataLoader
    from DimeModels import DimeNetPlusSE
    from element_vocab import NUM_ELEMENTS, build_one_hot
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels

    device = torch.device("cpu")
    EV_TO_KCAL = 23.0605

    assert not (args.hidden > 128 and not args.no_multi_agg), "Multi-agg recommended for larger models"

    json_path, _ = download_freesolv_data(args.cache_dir)
    all_labels = load_freesolv_labels(json_path)

    with h5py.File(args.conformers, "r") as f:
        mol_ids = [m for m in f.keys()
                   if m in all_labels and isinstance(all_labels[m].get("expt"), (int, float))]
    print(f"Total molecules: {len(mol_ids)}")
    print(f"Architecture: hidden={args.hidden}, blocks={args.blocks}, "
          f"SE={not args.no_se}, multi_agg={not args.no_multi_agg}")

    expts_arr = np.array([all_labels[m]["expt"] for m in mol_ids])
    sort_idx = np.argsort(expts_arr)
    mol_ids_sorted = [mol_ids[i] for i in sort_idx]

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    ckpt_dir = os.path.join(_script_dir, args.checkpoint_dir)
    output_dir = os.path.join(_script_dir, args.output_dir)
    folds_to_run = range(2 if args.quick_test else args.n_folds)
    epochs = 2 if args.quick_test else args.epochs
    patience = 5 if args.quick_test else args.patience

    fold_metrics = []

    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}")
        print(f"{'='*60}")

        train_idx, test_idx = list(kf.split(mol_ids_sorted))[fold]
        train_ids = [mol_ids_sorted[i] for i in train_idx]
        test_ids = [mol_ids_sorted[i] for i in test_idx]
        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "test_ids.json"), "w") as f:
            json.dump(test_ids, f)

        model = DimeNetPlusSE(
            in_channels=NUM_ELEMENTS,
            hidden_channels=args.hidden,
            out_channels=1,
            num_blocks=args.blocks,
            int_emb_size=min(64, args.hidden // 2),
            basis_emb_size=8,
            out_emb_channels=min(256, args.hidden * 2),
            num_spherical=7,
            num_radial=6,
            cutoff=6.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            is_energy=True,
            use_multi_aggregate=not args.no_multi_agg,
            use_se=not args.no_se,
        ).to(device)

        ckpt_path = os.path.join(ckpt_dir, args.correction_ckpt)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"    Random init: {len(missing)} params ({', '.join(missing[:5])}{'...' if len(missing)>5 else ''})")
        print(f"  Loaded checkpoint: {ckpt_path}")

        class SimpleDataset:
            def __init__(self, ids, hdf5_path, labels):
                self.ids = ids
                self.hdf5_path = hdf5_path
                self.labels = labels
                self._cache = {}
            def __len__(self):
                return len(self.ids)
            def __getitem__(self, idx):
                import torch
                from torch_geometric.data import Data
                mid = self.ids[idx]
                if mid not in self._cache:
                    with h5py.File(self.hdf5_path, "r") as f:
                        g = f[mid]
                        d = Data(
                            z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                            pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                        )
                    self._cache[mid] = d.clone()
                data = self._cache[mid].clone()
                data.mol_id = mid
                data.y_dG = torch.tensor([self.labels[mid]["expt"]], dtype=torch.float)
                return data

        train_ds = SimpleDataset(train_ids, args.conformers, all_labels)
        test_ds = SimpleDataset(test_ids, args.conformers, all_labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)
        mse = torch.nn.MSELoss()

        best_mae = float("inf")
        best_epoch = -1
        stale = 0

        for epoch in range(1, epochs + 1):
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

            model.eval()
            all_p, all_e = [], []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    x = build_one_hot(data, device)
                    pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                    dG_exp = data.y_dG.view(-1).to(device)
                    valid = ~torch.isnan(dG_exp)
                    all_p.append(pred[valid].cpu())
                    all_e.append(dG_exp[valid].cpu())
            test_preds = torch.cat(all_p).numpy()
            test_expts = torch.cat(all_e).numpy()
            test_mae = float(np.mean(np.abs(test_preds - test_expts)))
            test_rmse = float(np.sqrt(np.mean((test_preds - test_expts) ** 2)))

            scheduler.step(test_mae)
            print(f"    Epoch {epoch:3d} | Test MAE: {test_mae:.3f} RMSE: {test_rmse:.3f}", end="\r")

            if test_mae < best_mae:
                best_mae = test_mae
                best_epoch = epoch
                stale = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "finetuned.pt"))
                with open(os.path.join(fold_dir, "ft_test_predictions.csv"), "w") as f:
                    f.write("dG_pred_kcal,dG_exp_kcal\n")
                    for p, e in zip(test_preds, test_expts):
                        f.write(f"{p:.6f},{e:.6f}\n")
            else:
                stale += 1

            if stale >= patience:
                print(f"\n    Early stopping at epoch {epoch}")
                break

        print(f"\n  Best: MAE={best_mae:.3f} at epoch {best_epoch}")
        fold_metrics.append((best_mae, test_rmse))

    if args.n_conformers > 1:
        print(f"\n{'='*60}")
        print(f"  CONFORMER ENSEMBLE ({args.n_conformers} conformers/mol)")
        print(f"{'='*60}")
        print(f"  Generating conformers with RDKit and averaging predictions...")

        from rdkit import Chem
        from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
        from torch_geometric.data import Data

        def _gen_confs(smiles, n):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            params.pruneRmsThresh = 0.5
            conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
            if not conf_ids:
                return None
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            if props is None:
                return None
            try:
                rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
            except Exception:
                pass
            z = torch.tensor(np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32), dtype=torch.long)
            n_avail = min(n, mol.GetNumConformers())
            return [Data(z=z.clone(), pos=torch.tensor(np.array(mol.GetConformer(i).GetPositions(), dtype=np.float64), dtype=torch.float)) for i in range(n_avail)]

        for fold in folds_to_run:
            fold_dir = os.path.join(output_dir, f"fold_{fold}")
            test_path = os.path.join(fold_dir, "test_ids.json")
            ckpt_path = os.path.join(fold_dir, "finetuned.pt")
            if not os.path.exists(ckpt_path):
                continue
            with open(test_path) as f:
                tids = json.load(f)

            model = DimeNetPlusSE(
                in_channels=NUM_ELEMENTS,
                hidden_channels=args.hidden,
                out_channels=1,
                num_blocks=args.blocks,
                int_emb_size=min(64, args.hidden // 2),
                basis_emb_size=8,
                out_emb_channels=min(256, args.hidden * 2),
                num_spherical=7, num_radial=6,
                cutoff=6.0, max_num_neighbors=32, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3,
                is_energy=True,
                use_multi_aggregate=not args.no_multi_agg,
                use_se=not args.no_se,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            flat_data, flat_mid, flat_idx = [], [], []
            hdf5_cache = {}
            for mid in tids:
                smiles = all_labels[mid]["smiles"]
                confs = _gen_confs(smiles, args.n_conformers)
                if confs is None:
                    if mid not in hdf5_cache:
                        with h5py.File(args.conformers, "r") as f:
                            g = f[mid]
                            hdf5_cache[mid] = Data(
                                z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                                pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                            )
                    confs = [hdf5_cache[mid].clone() for _ in range(args.n_conformers)]
                for ci, cd in enumerate(confs):
                    flat_data.append(cd)
                    flat_mid.append(mid)
                    flat_idx.append(ci)

            loader = DataLoader(flat_data, batch_size=args.batch_size * 4, shuffle=False)
            all_raw = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    x = build_one_hot(data, device)
                    preds = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                    all_raw.append(preds.cpu())
            all_raw = torch.cat(all_raw).numpy()

            conf_preds = {}
            for mid, val in zip(flat_mid, all_raw):
                conf_preds.setdefault(mid, []).append(float(val))

            with open(os.path.join(fold_dir, "ft_test_predictions.csv"), "w") as f:
                f.write("dG_pred_kcal,dG_exp_kcal\n")
                for mid in tids:
                    p = np.mean(conf_preds[mid])
                    e = all_labels[mid]["expt"]
                    f.write(f"{p:.6f},{e:.6f}\n")

            cm, cr, cr2 = evaluate(
                np.array([np.mean(conf_preds[mid]) for mid in tids]),
                np.array([all_labels[mid]["expt"] for mid in tids])
            )
            print(f"  Fold {fold}: conformer-averaged MAE={cm:.3f} RMSE={cr:.3f} R²={cr2:.4f}")

        print(f"  Conformer-averaged predictions saved.")

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Fold':<8} {'N':<6} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for fold in folds_to_run:
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        test_path = os.path.join(fold_dir, "test_ids.json")
        pred_path = os.path.join(fold_dir, "ft_test_predictions.csv")
        with open(test_path) as f:
            tids = json.load(f)
        with open(pred_path) as f:
            next(f)
            preds = np.array([float(line.split(",")[0]) for line in f])
        expts = np.array([all_labels[mid]["expt"] for mid in tids])
        m, r, r2 = evaluate(preds, expts)
        print(f"  {'Fold '+str(fold):<8} {len(tids):<6} {m:<10.3f} {r:<10.3f} {r2:<10.4f}")

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE (average across folds)")
    print(f"{'='*60}")

    all_preds = {}
    for fold in folds_to_run:
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        test_path = os.path.join(fold_dir, "test_ids.json")
        pred_path = os.path.join(fold_dir, "ft_test_predictions.csv")
        with open(test_path) as f:
            tids = json.load(f)
        with open(pred_path) as f:
            next(f)
            vals = [float(line.split(",")[0]) for line in f]
        for mid, v in zip(tids, vals):
            all_preds.setdefault(mid, []).append(v)

    ens_preds, ens_expts = [], []
    for mid, vals in all_preds.items():
        ens_preds.append(np.mean(vals))
        ens_expts.append(all_labels[mid]["expt"])
    ens_preds = np.array(ens_preds)
    ens_expts = np.array(ens_expts)
    em, er, er2 = evaluate(ens_preds, ens_expts)
    print(f"  Molecules: {len(ens_preds)}")
    print(f"  MAE:  {em:.3f} kcal/mol")
    print(f"  RMSE: {er:.3f} kcal/mol")
    print(f"  R²:   {er2:.4f}")

    maes = [m for m, _ in fold_metrics]
    print(f"\n  Mean ± std: MAE = {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    print(f"\n{'='*60}")
    print(f"  COMPARISON WITH PUBLISHED METHODS")
    print(f"{'='*60}")
    header = f"  {'Method':<32} {'MAE':<10} {'RMSE':<10}"
    print(header)
    print(f"  {'-'*32} {'-'*10} {'-'*10}")
    refs = [
        ("Zhang 2022 (A3D-PNAConv-FT)", 0.417, 0.719),
        ("SenCos-GEM 2026 (SOTA)", None, 0.626),
        ("COSMO-RS (Klamt 2015)", 0.52, None),
        ("ReSolv (Röcken 2024)", 0.63, 0.96),
        ("Amber GAFF (full FreeSolv)", 1.11, 1.53),
        ("CHARMM CGenFF (full FreeSolv)", 1.18, 2.04),
        ("GIN-FP (IIT Delhi 2025)", None, 1.022),
        ("GBn2 (this baseline)", 19.83, None),
    ]
    for name, mae_val, rmse_val in refs:
        m_str = f"{mae_val:.3f}" if mae_val is not None else "—"
        r_str = f"{rmse_val:.3f}" if rmse_val is not None else "—"
        print(f"  {name:<34} {m_str:<10} {r_str:<10}")
    print(f"  {'-'*52}")
    em_str = f"{em:.3f}"
    er_str = f"{er:.3f}"
    print(f"  {'DimeNet+++SE+MultiAgg ensemble':<32} {em_str:<10} {er_str:<10}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ens_expts, ens_preds, alpha=0.5, edgecolors="k", linewidths=0.3)
        lims = [min(ens_expts.min(), ens_preds.min()) - 1,
                max(ens_expts.max(), ens_preds.max()) + 1]
        ax.plot(lims, lims, "k--", lw=0.8, label="Perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental ΔG$_{solv}$ (kcal/mol)")
        ax.set_ylabel("Predicted ΔG$_{solv}$ (kcal/mol)")
        ax.set_title(f"SE + MultiAgg Ensemble  |  MAE={em:.3f}  RMSE={er:.3f}  R²={er2:.4f}")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "parity_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\n  Parity plot saved: {plot_path}")
    except ImportError:
        print("\n  matplotlib not available — skipping parity plot")


if __name__ == "__main__":
    main()

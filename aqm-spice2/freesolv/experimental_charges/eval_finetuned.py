"""Post-hoc final evaluation of the charges-ablation fine-tuned checkpoints.

The 6 detached training runs (plain / Gasteiger-charges x seeds 42/7/2024)
were killed after epoch 14, before their built-in final evaluation block ran
(no meta.json / per_molecule.csv were produced). The saved checkpoints are
the best-val saves from each run and are the protocol's standard artifacts.

This script re-runs the final evaluation for all 6 checkpoints:
  * stored-conformer single-shot test MAE (129) - overall + gradient-12
  * TTA-4 test MAE (129) - overall + gradient-12 (falls back to stored
    conformer clones when rdkit is unavailable; mean unchanged)
  * full-642 stored-conformer predictions saved per molecule, so the same
    agreement-accuracy audit can be re-run on the charges variants later

Usage:
  python eval_finetuned.py
"""

import csv
import json
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import common
from element_vocab import build_one_hot

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7, 2024]
MODES = ["plain", "charges"]
GRAD12_CSV = os.path.join(
    HERE, "..", "deep_ensemble", "gmm_uncertainty_check",
    "gradient12_investigation", "gradient12_ungrouped.csv")


def load_grad12():
    import pandas as pd
    return set(pd.read_csv(GRAD12_CSV).mol_id)


def main():
    device = torch.device("cpu")
    all_labels = common.load_freesolv_labels(
        os.path.join(common.REPO_ROOT, "Data", "FreeSolv", "database.json"))
    train_ids, val_ids, test_ids = common.load_frozen_split(
        common.DEFAULT_SPLIT_DIR, all_labels)
    charges_cache = common.load_charges(common.DEFAULT_CHARGES_JSON)
    grad12 = load_grad12()
    print(f"Test n={len(test_ids)}; gradient-12 n={len(grad12 & set(test_ids))}")
    assert len(grad12 & set(test_ids)) == 12, "gradient-12 list does not match test split"

    summary_rows = []
    for mode in MODES:
        use_charges = mode == "charges"
        for seed in SEEDS:
            model = common.build_model(device, use_charges)
            ckpt = os.path.join(HERE, "results", mode, f"seed{seed}",
                                f"finetuned_{mode}_seed{seed}.pt")
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            print(f"\n=== {mode} seed {seed} ===", flush=True)

            # TTA-4 (falls back to stored-conformer clones locally)
            tta_mae, tta_rmse, tta_by_mid = common.conformer_average(
                model, device, test_ids, all_labels, common.DEFAULT_FREESOLV_CONFORMERS,
                4, 16, charges=charges_cache if use_charges else None)
            print(f"TTA-4 test MAE: {tta_mae:.3f} RMSE: {tta_rmse:.3f}", flush=True)

            # Stored-conformer single-shot
            DS = common.dataset_cls(common.DEFAULT_FREESOLV_CONFORMERS, all_labels,
                                    charges_cache if use_charges else None)
            loader = DataLoader(DS(test_ids), batch_size=16, shuffle=False)
            stored_by_mid = {}
            with torch.no_grad():
                for data in loader:
                    mids = list(data.mol_id)
                    data = data.to(device)
                    x = build_one_hot(data, device)
                    pred = model(x, data.pos, data.batch).view(-1) * common.EV_TO_KCAL
                    for mid, p in zip(mids, pred.cpu().tolist()):
                        stored_by_mid[mid] = p
            expts = {m: all_labels[m]["expt"] for m in test_ids}
            stored_mae = float(np.mean([abs(stored_by_mid[m] - expts[m]) for m in test_ids]))
            stored_rmse = float(np.sqrt(np.mean([(stored_by_mid[m] - expts[m]) ** 2
                                                 for m in test_ids])))
            print(f"Stored-conformer test MAE: {stored_mae:.3f} RMSE: {stored_rmse:.3f}",
                  flush=True)

            g12_stored = float(np.mean([abs(stored_by_mid[m] - expts[m])
                                        for m in grad12]))
            g12_tta = float(np.mean([abs(tta_by_mid[m] - expts[m]) for m in grad12]))
            print(f"gradient-12 MAE: stored {g12_stored:.3f} | TTA-4 {g12_tta:.3f}",
                  flush=True)

            # Full-642 stored-conformer predictions (audit-ready artifact)
            all642 = list(dict.fromkeys(train_ids + val_ids + test_ids))
            loader642 = DataLoader(DS(all642), batch_size=16, shuffle=False)
            preds642 = {}
            with torch.no_grad():
                for data in loader642:
                    mids = list(data.mol_id)
                    data = data.to(device)
                    x = build_one_hot(data, device)
                    pred = model(x, data.pos, data.batch).view(-1) * common.EV_TO_KCAL
                    for mid, p in zip(mids, pred.cpu().tolist()):
                        preds642[mid] = p

            out_dir = os.path.join(HERE, "results", mode, f"seed{seed}")
            per_mol = os.path.join(out_dir, f"per_molecule_{mode}_seed{seed}.csv")
            with open(per_mol, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mol_id", f"pred_seed{seed}", "ensemble_mean",
                            "ensemble_std", "true_value", "abs_error",
                            "has_halogen_Br_I"])
                for mid in all642:
                    p = preds642[mid]
                    expt = all_labels[mid]["expt"]
                    smiles = all_labels[mid].get("smiles", "")
                    br_i = 1 if ("Br" in smiles or "I" in smiles) else 0
                    w.writerow([mid, f"{p:.6f}", f"{p:.6f}", "nan",
                                f"{expt:.6f}", f"{abs(p - expt):.6f}", br_i])
            print(f"Saved -> {per_mol}", flush=True)

            meta = {
                "mode": mode, "seed": seed, "checkpoint": ckpt,
                "test_mae_stored": stored_mae, "test_rmse_stored": stored_rmse,
                "test_mae_tta4": tta_mae, "test_rmse_tta4": tta_rmse,
                "gradient12_mae_stored": g12_stored,
                "gradient12_mae_tta4": g12_tta,
            }
            with open(os.path.join(out_dir, f"eval_{mode}_seed{seed}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            summary_rows.append(meta)

    summary_path = os.path.join(HERE, "results", "eval_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary -> {summary_path}")
    for r in summary_rows:
        print(f"{r['mode']:>7} seed {r['seed']:>4} | stored {r['test_mae_stored']:.3f} "
              f"| tta4 {r['test_mae_tta4']:.3f} | grad12 stored {r['gradient12_mae_stored']:.3f} "
              f"| grad12 tta4 {r['gradient12_mae_tta4']:.3f}")


if __name__ == "__main__":
    main()
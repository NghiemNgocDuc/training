"""Stage-3 fine-tuning with neighbor-consistency regularization (fold-0).

Adds to the VERIFIED deep_ensemble.train_member harness (same split, same
stage-2 init checkpoint, same hyperparameters, same MSE-in-eV task loss, same
scheduler/early-stopping/TTA) one extra per-epoch loss:

    L_neighbor = mean over i of [ sum_j w_ij * (p_i - p_j)^2 / sum_j w_ij ]

over a STATIC similarity graph of all 642 FreeSolv molecules (train+val+test
STRUCTURES only; no labels). Both p_i and p_j are the model's CURRENT
predictions, computed in a single full-graph forward pass, so the loss is the
exact user-specified objective (no stop-gradient approximation), it is fully
transductive (test structures participate with gradients), and zero
test/val LABELS are ever read.

Total loss = task_loss + lambda_nbr * L_neighbor (both in eV^2).
With --normalize_nbr the consistency term is L_neighbor / var(p) (unitless);
the raw eV^2 value is still recorded per epoch in epoch_history.csv so the
effective strength (raw magnitude vs task loss, prediction variance drift)
can be inspected directly.

lambda_nbr=0 skips the graph pass entirely -> byte-identical to
deep_ensemble.train_member -> the baseline reproduction control.

v2 (--neighbor_source latent, see DESIGN_v2.md): the graph is built on cosine
similarity of the model's own mean-pooled latent space (graph_cache/
latent_k{k}_sim{sim}.json, built by latent_graph.py, which also computes
per-molecule GMM-NLL uncertainty u_i, the trust gate t_j on neighbors, and
the static trust-filtered weight sum S_i). Per node:

    L_i = u_i * [ sum_j w_ij*t_j*(p_i - p_j)^2 / S_i ]

with contribution 0 for nodes whose S_i < --coverage_floor (no trusted
neighbors; logged to epoch_fallback.csv + counted per epoch in
epoch_history.csv n_fallback). Graph, u_i, t_j, S_i are all STATIC and
computed once; only p_i are current-model predictions. The v1 Tanimoto path
(--neighbor_source tanimoto, default) is byte-identical to the original.

--track_groups (smoke-test only): per-epoch test-set eval split by the
isolated6 / gradient12 / wrong18 / certain47 groups (definitions from the
rmse_analysis CSVs), written to epoch_test_groups.csv. The added eval pass is
wrapped in the EXISTING rng_snapshot/rng_restore utility from
instrumented_rerun.instrument_finetune (loaded by file path; the
deep_ensemble/ dir has no __init__.py) so the training loop's RNG stream is
untouched.

Usage:
  python finetune_nbr.py --lambda_nbr 0.1 --out results_lambda0.1 --epochs 200
  python finetune_nbr.py --lambda_nbr 0   --out results_lambda0   --epochs 200
  python finetune_nbr.py --neighbor_source latent --k_nbr 5 --min_sim 0.5 \\
      --lambda_nbr 0.001 --out results_v2_lam0.001 --epochs 200
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import hashlib

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)           # freesolv/
_aqm = os.path.dirname(_parent)                  # aqm-spice2/
REPO_ROOT = os.path.dirname(_aqm)
for p in (_aqm, _parent, _script_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_aqm)

import numpy as np

from deep_ensemble import (
    evaluate, set_seed, load_frozen_split, simple_dataset_cls, build_model,
    conformer_average, EV_TO_KCAL,
    DEFAULT_SPLIT_DIR, DEFAULT_CORRECTION_CKPT, DEFAULT_CONFORMERS,
)
from freesolv_dataset import download_freesolv_data, load_freesolv_labels
from graph import load_or_build_graph, graph_to_tensor

# Load the EXISTING rng_snapshot/rng_restore utility from
# deep_ensemble/instrumented_rerun/instrument_finetune.py (by file path:
# the deep_ensemble/ directory has no __init__.py). The added per-epoch
# group eval in this script must NOT perturb the training loop's RNG stream.
_rng_util_path = os.path.join(
    _parent, "deep_ensemble", "instrumented_rerun", "instrument_finetune.py")
_rng_spec = importlib.util.spec_from_file_location("nbr_rng_util", _rng_util_path)
_rng_util = importlib.util.module_from_spec(_rng_spec)
_rng_spec.loader.exec_module(_rng_util)
rng_snapshot, rng_restore = _rng_util.rng_snapshot, _rng_util.rng_restore


def ckpt_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_nbr", type=float, default=0.1,
                    help="weight of L_neighbor; 0 disables the graph pass entirely")
    ap.add_argument("--normalize_nbr", action="store_true",
                    help="use L_neighbor / var(p) (unitless consistency strength; "
                         "raw eV^2 magnitude still recorded in the epoch history)")
    ap.add_argument("--k_nbr", type=int, default=5)
    ap.add_argument("--min_sim", type=float, default=0.1)
    ap.add_argument("--neighbor_source", default="tanimoto",
                    choices=["tanimoto", "latent"],
                    help="tanimoto: v1 graph (Morgan r=2, cached). latent: v2 "
                         "graph (cosine in model latent space, graph_cache/"
                         "latent_k{k}_sim{sim}.json + .meta.json with "
                         "GMM-NLL u_i/trust signals, built by latent_graph.py).")
    ap.add_argument("--coverage_floor", type=float, default=1e-6,
                    help="v2: node with trusted weight sum < floor is skipped "
                         "(contribution 0, logged to epoch_fallback.csv)")
    ap.add_argument("--graph_every", type=int, default=1, help="run graph pass every N epochs")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--correction_ckpt", default=DEFAULT_CORRECTION_CKPT)
    ap.add_argument("--track_groups", action="store_true",
                    help="smoke-test: per-epoch test-set eval split by "
                         "isolated6/gradient12/wrong18/certain47 (RNG-guarded)")
    ap.add_argument("--out", default=os.path.join(_script_dir, "results_lambda0.1"))
    ap.add_argument("--device", default=None, help="cuda or cpu; default auto (cuda if available)")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        print("WARNING: cuda requested but unavailable - falling back to cpu "
              "(same behavior as the verified fold-0 pipeline)")
        args.device = "cpu"
    device = __import__("torch").device(args.device)

    import torch
    import h5py
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot

    set_seed(args.seed)
    labels_path, _ = download_freesolv_data(os.path.join(REPO_ROOT, "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(labels_path)
    train_ids, val_ids, test_ids = load_frozen_split(args.split_dir, all_labels)

    # ---- molecule universe = fold-0 universe (642 = 411+102+129) ----
    with h5py.File(args.conformers, "r") as f:
        mol_ids = [m for m in f.keys()
                   if m in all_labels and isinstance(all_labels[m].get("expt"), (int, float))]
    assert set(mol_ids) == set(train_ids) | set(val_ids) | set(test_ids), (
        "graph universe != fold-0 universe")
    model_mids = [m for m in mol_ids if m in set(train_ids) | set(val_ids) | set(test_ids)]

    # ---- static similarity graph (structures only, no labels) ----
    if args.neighbor_source == "latent":
        # v2: skip the Tanimoto graph entirely; load the latent graph below.
        graph, graph_meta = None, None
    else:
        graph, graph_meta = load_or_build_graph(
            os.path.join(_script_dir, "graph_cache"), model_mids,
            {m: all_labels[m]["smiles"] for m in model_mids},
            k=args.k_nbr, min_sim=args.min_sim)
    edge_i = edge_j = edge_w = None
    if graph is not None:
        edge_i, edge_j, edge_w = graph_to_tensor(graph, model_mids, device)

    # ---- v2: latent-space graph + GMM-NLL uncertainty/trust signals ----
    u_vec = None
    t_vec = None
    s_vec = None
    fallback_mids = []
    latent_meta = None
    if args.neighbor_source == "latent":
        latent_path = os.path.join(
            _script_dir, "graph_cache",
            f"latent_k{args.k_nbr}_sim{args.min_sim}.json")
        meta_path = latent_path + ".meta.json"
        if not os.path.exists(latent_path):
            raise SystemExit(f"latent graph missing: {latent_path}\n"
                             "build it first: python latent_graph.py "
                             f"--k {args.k_nbr} --min-sim {args.min_sim}")
        with open(latent_path) as f:
            latent_graph = json.load(f)
        with open(meta_path) as f:
            latent_meta = json.load(f)
        if set(latent_meta["mids"]) != set(model_mids):
            raise SystemExit("latent graph node set != fold-0 universe; "
                             "rebuild latent_graph.py")
        graph = latent_graph                         # edges = latent cosine
        graph_meta = latent_meta
        edge_i, edge_j, edge_w = graph_to_tensor(graph, model_mids, device)
        sig = latent_meta["signals"]
        u_vec = torch.tensor([sig["u"][m] for m in model_mids],
                             dtype=torch.float, device=device)
        t_vec = torch.tensor([sig["trust"][m] for m in model_mids],
                             dtype=torch.float, device=device)
        s_vec = torch.tensor([sig["S"][m] for m in model_mids],
                             dtype=torch.float, device=device)
        fallback_mids = [m for m in sig["fallback"] if m in set(model_mids)]
        n_fb = len(fallback_mids)
        print(f"  [v2 latent] {len(model_mids)} nodes | u in "
              f"[{u_vec.min().item():.4f},{u_vec.max().item():.4f}] | "
              f"trusted {int(t_vec.sum().item())}/642 | fallback (S<floor): {n_fb}")
        print(f"  [v2 latent] trust threshold {sig['trust_threshold']:.4f} "
              f"({sig['trust_policy']}) | nll sanity Spearman "
              f"{latent_meta['provenance']['cached_corr']['test_mean_nll_spearman']}")

    test_set = set(test_ids)
    test_nodes = [i for i, m in enumerate(model_mids) if m in test_set]
    n_test_active = sum(1 for i in test_nodes if any(
        True for nbr, _ in graph[model_mids[i]]))
    print(f"  [graph] {len(model_mids)} nodes | test-set nodes: {len(test_nodes)} "
          f"(active: {n_test_active})")

    model = build_model(device)
    ckpt = torch.load(args.correction_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)

    SimpleDataset = simple_dataset_cls(args.conformers, all_labels)
    train_ds = SimpleDataset(train_ids)
    val_ds = SimpleDataset(val_ids)
    test_ds = SimpleDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- smoke-test group tracking: reuse the rmse_analysis group definitions ----
    if args.track_groups:
        from report_results import load_groups
        wrong18, certain47, isolated6, gradient12 = load_groups()
        group_sets = {
            "all129": set(test_ids),
            "wrong18": wrong18, "certain47": certain47,
            "isolated6": isolated6, "gradient12": gradient12,
        }
        for gname, mids in group_sets.items():
            n_known = len([m for m in mids if m in test_ids])
            print(f"  [groups] {gname}: {n_known} of {len(mids)} in test set")
    else:
        group_sets = {}

    # Label-free dataset for the graph pass (STRUCTURES only).
    class GraphDataset:
        def __init__(self, mids, hdf5_path):
            self.ids = mids
            self.hdf5_path = hdf5_path
            self._cache = {}
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            from torch_geometric.data import Data
            mid = self.ids[idx]
            if mid not in self._cache:
                with h5py.File(self.hdf5_path, "r") as f:
                    g = f[mid]
                    self._cache[mid] = Data(
                        z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                        pos=torch.tensor(g["atXYZ"][...], dtype=torch.float),
                    ).clone()
            return self._cache[mid].clone()

    graph_ds = GraphDataset(model_mids, args.conformers)
    graph_loader = DataLoader(graph_ds, batch_size=args.batch_size * 4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2, min_lr=1e-6)
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
        return float(np.mean(np.abs(preds - expts))), float(np.sqrt(np.mean((preds - expts) ** 2))), preds, expts

    def eval_test_by_group():
        """Per-molecule test predictions -> per-group (MAE, RMSE, n).
        Only runs when --track_groups; must be called inside the RNG guard."""
        model.eval()
        rows = {}
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                pred = model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
                dG_exp = data.y_dG.view(-1).to(device)
                for batch_pos, mid in enumerate(data.mol_id):
                    p, e = pred[batch_pos].item(), dG_exp[batch_pos].item()
                    if not np.isnan(e):
                        rows[mid] = (p, e)
        out = {}
        for gname, mids in group_sets.items():
            vals = [rows[m] for m in mids if m in rows]
            if not vals:
                out[gname] = (None, None, 0)
                continue
            ps = np.array([v[0] for v in vals]); es = np.array([v[1] for v in vals])
            out[gname] = (float(np.mean(np.abs(ps - es))),
                          float(np.sqrt(np.mean((ps - es) ** 2))), len(vals))
        return out

    def graph_pass():
        """Full-graph forward (all 642 structures, no labels) -> L_neighbor.

        v1 tanimoto: mean_i [ sum_j w_ij (p_i-p_j)^2 / sum_j w_ij ] (eV^2).
        v2 latent:   mean_i [ u_i * sum_j w_ij*t_j*(p_i-p_j)^2 / S_i ],
                     fallback nodes (S_i < floor) contribute 0.

        Returns (L_used, L_raw, var_p, n_active, n_fallback):
          L_raw : raw graph loss (eV^2) — for v1: mean_i [...]; for v2:
                  mean_i [u_i * ...] (with 0 for fallback).
          L_used: L_raw / var(p) if --normalize_nbr else L_raw.
        """
        if args.lambda_nbr == 0:
            return None, None, None, None, None
        model.train()
        preds = []
        with torch.enable_grad():
            for data in graph_loader:
                data = data.to(device)
                x = build_one_hot(data, device)
                preds.append(model(x, data.pos, data.batch).view(-1))
        p = torch.cat(preds)                     # (642,) in eV
        assert p.shape[0] == len(model_mids)
        pi = p[edge_i]
        pj = p[edge_j]

        if args.neighbor_source == "latent" and u_vec is not None:
            # v2: trust-gated edges, u_i uncertainty weight, static S_i
            edge_t = t_vec[edge_j]                # trust gate per neighbor
            trust_weight = edge_w * edge_t
            num = torch.zeros(len(model_mids), device=device)
            num.index_add_(0, edge_i, trust_weight * (pi - pj).pow(2))
            active = s_vec >= args.coverage_floor
            n_active = int(active.sum().item())
            n_fb = len(model_mids) - n_active
            l_full = torch.zeros(len(model_mids), device=device)
            l_full[active] = num[active] / s_vec[active]
            l_raw = (u_vec * l_full).mean()        # 0 contributions for fallback
        else:
            # v1: plain Tanimoto graph (unchanged)
            wsum = torch.zeros(len(model_mids), device=device).index_add_(
                0, edge_i, edge_w)
            active = wsum > 0
            n_active = int(active.sum().item())
            n_fb = 0
            per_node = torch.zeros(len(model_mids), device=device)
            per_node.index_add_(0, edge_i, edge_w * (pi - pj).pow(2))
            l_raw = (per_node[active] / wsum[active]).mean()

        var_p = torch.var(p)
        if args.normalize_nbr and var_p > 1e-12:
            l_used = l_raw / var_p
        else:
            l_used = l_raw
        return l_used, l_raw, var_p, n_active, n_fb

    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_script_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, "finetuned_nbr.pt")
    best_val_mae = float("inf")
    best_epoch = -1
    stale = 0
    stop_epoch = args.epochs
    t0_all = time.time()
    epoch_history = []
    group_history = []
    nan_inf_seen = False

    def check_finite(name, t):
        nonlocal nan_inf_seen
        ok = bool(torch.isfinite(t).all().item())
        if not ok:
            nan_inf_seen = True
            print(f"    [WARN] non-finite value in {name} at epoch {epoch}", flush=True)
        return ok

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        task_mse_acc, n_batches = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            pred = model(x, data.pos, data.batch).view(-1)
            dG_exp = data.y_dG.view(-1).to(device) / EV_TO_KCAL
            valid = ~torch.isnan(dG_exp)
            if valid.sum() == 0:
                continue
            loss = mse(pred[valid], dG_exp[valid])
            check_finite("task_loss", loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            task_mse_acc += float(loss.detach().item())
            n_batches += 1
        task_mse = task_mse_acc / max(n_batches, 1)

        l_used = l_raw_ = var_p = n_active = n_fb = None
        if args.lambda_nbr > 0 and epoch % args.graph_every == 0:
            l_used, l_raw_, var_p, n_active, n_fb = graph_pass()
            check_finite("L_neighbor", l_used)
            epoch_history.append({
                "epoch": epoch, "task_mse_train_eV2": round(task_mse, 6),
                "l_nbr_raw_eV2": round(float(l_raw_.detach().item()), 6),
                "l_nbr_used": round(float(l_used.detach().item()), 6),
                "var_p_eV2": round(float(var_p.detach().item()), 6),
                "n_active": n_active, "n_fallback": n_fb,
                "normalized": args.normalize_nbr,
            })
            optimizer.zero_grad()
            total = l_used * args.lambda_nbr
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        else:
            epoch_history.append({
                "epoch": epoch, "task_mse_train_eV2": round(task_mse, 6),
                "l_nbr_raw_eV2": None, "l_nbr_used": None,
                "var_p_eV2": None, "n_active": None, "n_fallback": None,
                "normalized": args.normalize_nbr,
            })

        check_finite("model_params", torch.cat(
            [p.detach().reshape(-1) for p in model.parameters()]))

        val_mae, val_rmse, _, _ = evaluate_loader(val_loader)
        scheduler.step(val_mae)
        dt = time.time() - t0

        if args.track_groups:
            _rng = rng_snapshot()
            group_maes = eval_test_by_group()
            rng_restore(_rng)
            for gname, (mae, rmse, n) in group_maes.items():
                group_history.append({
                    "epoch": epoch, "group": gname, "n": n,
                    "mae": "" if mae is None else round(mae, 4),
                    "rmse": "" if rmse is None else round(rmse, 4),
                })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            stale += 1

        msg = (f"    seed {args.seed:>4} | ep {epoch:3d}/{args.epochs} | "
               f"best val {best_val_mae:7.3f} (ep {best_epoch}) | cur val {val_mae:7.3f}")
        if l_used is not None:
            msg += f" | task {task_mse:.5f} | L_nbr {float(l_used.detach()):.5f}"
            if args.normalize_nbr:
                msg += f" (raw {float(l_raw_.detach()):.4f}, var_p {float(var_p.detach()):.4f})"
        print(msg + f" | {dt:5.1f}s/ep", flush=True)

        if stale >= args.patience:
            stop_epoch = epoch
            print(f"    early stopped at epoch {epoch} (patience {args.patience})")
            break

    total_min = (time.time() - t0_all) / 60.0

    # ---- best-val checkpoint: single-conf test + 5-conf TTA ----
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    model.eval()
    test_mae, test_rmse, test_preds, test_expts = evaluate_loader(test_loader)
    tta_mae, tta_rmse, tta_preds_by_mid = conformer_average(
        model, device, test_ids, all_labels, args.conformers, 5, args.batch_size)

    with open(os.path.join(out_dir, "augmented_predictions.csv"), "w") as f:
        f.write("mol_id,dG_pred_kcal,dG_exp_kcal\n")
        for mid in test_ids:
            f.write(f"{mid},{tta_preds_by_mid[mid]:.6f},{all_labels[mid]['expt']:.6f}\n")

    with open(os.path.join(out_dir, "epoch_history.csv"), "w") as f:
        f.write("epoch,task_mse_train_eV2,l_nbr_raw_eV2,l_nbr_used,var_p_eV2,n_active,n_fallback,normalized\n")
        for row in epoch_history:
            f.write("{epoch},{task},{raw},{used},{varp},{act},{fb},{norm}\n".format(
                epoch=row["epoch"], task=row["task_mse_train_eV2"],
                raw="" if row["l_nbr_raw_eV2"] is None else row["l_nbr_raw_eV2"],
                used="" if row["l_nbr_used"] is None else row["l_nbr_used"],
                varp="" if row["var_p_eV2"] is None else row["var_p_eV2"],
                act="" if row["n_active"] is None else row["n_active"],
                fb="" if row["n_fallback"] is None else row["n_fallback"],
                norm=int(row["normalized"])))

    if args.neighbor_source == "latent" and fallback_mids:
        with open(os.path.join(out_dir, "epoch_fallback.csv"), "w") as f:
            f.write("mol_id,trusted_weight_sum_S\n")
            sig = latent_meta["signals"]
            for m in fallback_mids:
                f.write(f"{m},{sig['S'].get(m, 0.0)}\n")
        print(f"  [fallback] {len(fallback_mids)} molecules below coverage "
              f"floor logged -> epoch_fallback.csv")

    if args.track_groups:
        with open(os.path.join(out_dir, "epoch_test_groups.csv"), "w") as f:
            f.write("epoch,group,n,mae,rmse\n")
            for row in group_history:
                f.write("{epoch},{group},{n},{mae},{rmse}\n".format(
                    epoch=row["epoch"], group=row["group"], n=row["n"],
                    mae=row["mae"], rmse=row["rmse"]))

    config = vars(args)
    config["device"] = args.device
    config["graph_meta"] = graph_meta
    config["latent_meta"] = latent_meta
    config["epoch_history"] = epoch_history
    config["task_loss_units"] = "eV^2"
    config["nan_inf_seen"] = nan_inf_seen
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    metrics = {
        "seed": args.seed,
        "lambda_nbr": args.lambda_nbr,
        "normalize_nbr": bool(args.normalize_nbr),
        "neighbor_source": args.neighbor_source,
        "k_nbr": args.k_nbr,
        "min_sim": args.min_sim,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "graph_nodes": len(model_mids),
        "hyperparams": {"lr": args.lr, "weight_decay": 1e-5,
                        "batch_size": args.batch_size, "epochs": args.epochs,
                        "patience": args.patience, "device": args.device},
        "best_val_mae_kcal": best_val_mae,
        "best_val_epoch": best_epoch,
        "early_stop_epoch": stop_epoch,
        "total_min": round(total_min, 1),
        "test_mae_single_conf_kcal": test_mae,
        "test_rmse_single_conf_kcal": test_rmse,
        "test_mae_tta_kcal": tta_mae,
        "test_rmse_tta_kcal": tta_rmse,
        "n_conformers_tta": 5,
        "nan_inf_seen": nan_inf_seen,
        "checkpoint_sha256": ckpt_sha256(best_ckpt_path),
        "stage2_init_sha256": ckpt_sha256(args.correction_ckpt),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  DONE lambda={args.lambda_nbr} seed={args.seed} | "
          f"best val {best_val_mae:.3f} (ep {best_epoch}, stopped {stop_epoch}) | "
          f"test {test_mae:.3f}/{test_rmse:.3f} (single conf) | "
          f"test {tta_mae:.3f}/{tta_rmse:.3f} (5-conf TTA) | {total_min:.1f} min")


if __name__ == "__main__":
    main()
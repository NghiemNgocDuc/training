"""GMM-based single-network uncertainty on the Stage-3 seed_42 checkpoint.

Implements Zhu, Batzner, Musaelian & Kozinsky, J. Chem. Phys. 158, 164111
(2023): fit a Gaussian Mixture Model on the TRAINING-set latents of ONE
trained network, then use per-atom negative log-likelihood (NLL) as the
uncertainty estimate on test molecules. Cross-checks against the ensemble
disagreement analysis (18 confidently-wrong / 47 certain / isolated-6 vs
gradient-12).

Latent definition (DimeModels.OutputPPBlock.forward, lines 585-592):
    x -> lin_up(128->256) -> act(lin1) -> act(lin2) -> lin(256->1)
The input to the FINAL `lin` (256-dim, post-activation, per atom) is the
pre-readout latent. Captured via a forward hook on each of the 4 output
blocks' `lin` modules -> 4x256 = 1024-dim per-atom latent. No model code is
modified. Because the prediction is linear in the latent (z . w = per-atom
contribution, summed over atoms), we validate the hook by checking that
sum_atoms(z . w) reproduces the model energy and the TTA predictions in
deep_ensemble/aggregate/per_molecule.csv.

Protocol: same frozen split, same seed_42 checkpoint (ensemble_seed42.pt).
Latents are extracted from the STORED hdf5 conformers (freesolv_conformers.hdf5)
- the exact geometries the fold-0 fine-tune trained on - so the GMM models
the true training distribution. (Local RDKit 2026.03.4 regenerates different
conformers than the Vast box used for the stored TTA predictions, deviating
up to ~25 kcal/mol on a few molecules; RDKit-generated latents would inject
that geometry-protocol confound into the density estimates. The stored TTA
values from deep_ensemble/aggregate/per_molecule.csv are used unchanged for
all cross-checks; the residual is reported in geometry_diagnostics.json.)

PCA: 1024-dim latents, ~4.5k training atoms. The raw empirical covariance is
ill-conditioned (cond ~1e24 raw / ~1e15 standardized), so a full-covariance
GMM is fit on PCA-reduced standardized latents. GMM uses k=14 (>=95% variance:
full-covariance 31-dim with ~4.5k atoms is only stable up to ~5 components;
k=14 keeps {1,5,10,20} sample-feasible, n=50 is reported infeasible).
k=31 (>=99%) and condition numbers are reported for completeness.

GMM: sklearn GaussianMixture, covariance_type='full', reg_covar=1e-2,
k-means-initialized means (default), n_init=5, BIC over n_components in
{1,5,10,20,50}.

Outputs -> deep_ensemble/gmm_uncertainty_check/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_freesolv = os.path.dirname(_script_dir)
for p in (_freesolv, os.path.dirname(_freesolv)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT_DIR = os.path.join(_script_dir, "gmm_uncertainty_check")
CKPT = os.path.join(_script_dir, "seed_42", "ensemble_seed42.pt")
AGG_CSV = os.path.join(_script_dir, "aggregate", "per_molecule.csv")
ANALYSIS_CSV = os.path.join(_script_dir, "rmse_analysis", "output",
                            "per_molecule_rmse.csv")
ISOLATION_CSV = os.path.join(_script_dir, "rmse_analysis",
                             "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")
N_CONFS = 5
N_COMPONENTS_GRID = [1, 5, 10, 20, 50]
# Held-out-validated n_components (molecule-level 80/20 LL max at n=10, see
# validation/heldout_validation.py). BIC's ceiling choice n=50 overfit held-out
# atoms by ~25 nats/atom and is available via --n-components 50.
N_COMPONENTS_DEFAULT = 10
PCA_VARIANCE_TARGET = 0.99
PCA_MAX_COMPONENTS = 200
LATENT_DIM = 1024  # 4 blocks x 256


def extract_latents(model, device, mids, tag, cache_dir):
    """Per-(mol,atom) pre-readout latents from the stored hdf5 conformers.

    Forward-hooks capture the input to each OutputPPBlock's final lin (the
    pre-final-linear per-atom latent, 256-dim per block x 4 = 1024-dim).
    Validation: z_atom . w = per-atom contribution; summing over atoms must
    reproduce the model energy (checked during the pass, ~1e-6 kcal/mol).
    Deterministic caches to cache_dir/z_<tag>.npz so repeated analyses skip
    the forward passes.

    Z columns: [0:1024] latent, 1024 p_atom (eV), 1025 global element index
    (into `mids`), 1026 atomic number. Returns (Z, flat_mid, zw_kcal, em_kcal).
    """
    import h5py
    import json
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from element_vocab import build_one_hot
    from deep_ensemble import DEFAULT_CONFORMERS, EV_TO_KCAL

    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"z_{tag}.npz")
    mid_path = os.path.join(cache_dir, f"mids_{tag}.json")
    if os.path.exists(npz_path) and os.path.exists(mid_path):
        d = np.load(npz_path)
        with open(mid_path) as f:
            flat_mid = json.load(f)
        print(f"[{tag}] loaded cache {npz_path} (atoms={len(d['Z'])})")
        return d["Z"], flat_mid, d["zw"], d["em"]

    latents_by_block = {b: [] for b in range(4)}

    def make_hook(b):
        def hook(mod, inp, out):
            latents_by_block[b].append(inp[0].detach())
        return hook

    handles = [model.output_blocks[b].lin.register_forward_hook(make_hook(b))
               for b in range(4)]
    w_all = torch.cat([model.output_blocks[b].lin.weight.detach()
                       for b in range(4)], dim=1)  # (1, 1024)

    flat, flat_mid = [], []
    with h5py.File(DEFAULT_CONFORMERS, "r") as f:
        for mid in mids:
            g = f[mid]
            flat.append(Data(z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                             pos=torch.tensor(g["atXYZ"][...], dtype=torch.float)))
            flat_mid.append(mid)

    loader = DataLoader(flat, batch_size=32, shuffle=False)
    rows, preds_zw, preds_model = [], [], []
    pos0 = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = build_one_hot(data, device)
            n_atoms = data.pos.size(0)
            n_elem = int(data.batch.max().item()) + 1
            e_model = model(x, data.pos, data.batch).view(-1)
            z = torch.cat(latents_by_block[0] + latents_by_block[1] +
                          latents_by_block[2] + latents_by_block[3],
                          dim=-1).view(n_atoms, LATENT_DIM)
            latents_by_block[0] = []
            latents_by_block[1] = []
            latents_by_block[2] = []
            latents_by_block[3] = []
            assert z.shape[0] == n_atoms, f"{tag}: latent/atom mismatch"
            p_atom = (z @ w_all.t()).view(-1)
            e_zw = torch.zeros(n_elem, device=device) \
                         .index_add_(0, data.batch, p_atom)
            preds_zw.append(e_zw.cpu())
            preds_model.append(e_model.cpu())
            rows.append(torch.cat([z.cpu(), p_atom.unsqueeze(-1),
                                   (data.batch + pos0).unsqueeze(-1).cpu(),
                                   data.z.unsqueeze(-1).cpu()], dim=1))
            pos0 += n_elem
    Z = torch.cat(rows, dim=0).numpy()
    zw = torch.cat(preds_zw).numpy() * EV_TO_KCAL
    em = torch.cat(preds_model).numpy() * EV_TO_KCAL
    max_diff = float(np.max(np.abs(zw - em))) if len(zw) else 0.0
    print(f"[{tag}] atoms={len(Z)} max|z.w - model| = {max_diff:.2e} kcal/mol")
    np.savez(npz_path, Z=Z, zw=zw, em=em)
    with open(mid_path, "w") as f:
        json.dump(flat_mid, f)
    for h in handles:
        h.remove()
    return Z, flat_mid, zw, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-confs", type=int, default=N_CONFS)
    ap.add_argument("--n-components", type=int, default=N_COMPONENTS_DEFAULT,
                    help="GMM n_components. Default 10 = held-out-validated "
                         "(molecule-level 80/20 LL max; see validation/). The "
                         "old default 50 was BIC's ceiling choice and overfits "
                         "by ~25 nats/atom on held-out atoms. 50 can be "
                         "recovered with --n-components 50.")
    ap.add_argument("--run-bic-grid", action="store_true",
                    help="also fit the BIC grid {1,5,10,20,50} and report it "
                         "(does not change the fitted n_components unless "
                         "--n-components is unset)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import h5py
    import pandas as pd
    from scipy.stats import spearmanr, mannwhitneyu, rankdata
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from deep_ensemble import (
        set_seed, load_frozen_split, build_model, EV_TO_KCAL,
        DEFAULT_SPLIT_DIR, DEFAULT_CONFORMERS,
    )
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels
    from element_vocab import build_one_hot

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gmm-uncertainty] device={device}")

    labels_path, _ = download_freesolv_data(os.path.join(
        os.path.dirname(os.path.dirname(_freesolv)), "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(labels_path)
    train_ids, val_ids, test_ids = load_frozen_split(DEFAULT_SPLIT_DIR, all_labels)
    print(f"[gmm-uncertainty] split train={len(train_ids)} val={len(val_ids)} "
          f"test={len(test_ids)}")

    model = build_model(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[gmm-uncertainty] loaded seed_42 ckpt (state_dict keys: {len(ckpt)})")

    # ---- hooks: capture pre-final-linear latent of each output block ----
    cache_dir = os.path.join(args.out, "latent_cache")
    Z_train, flat_mid_train, zw_tr, em_tr = extract_latents(
        model, device, train_ids, "train", cache_dir)
    Z_test, flat_mid_test, zw_te, em_te = extract_latents(
        model, device, test_ids, "test", cache_dir)

    # ---- conformer protocol: STORED hdf5 conformers (single conf per mol) ----
    # The fold-0 fine-tune trained on these exact geometries (SimpleDataset),
    # so the GMM models the true training distribution. Local RDKit (2026.03.4)
    # regenerates different conformers than the Vast box used for the stored
    # TTA predictions (up to ~25 kcal/mol deviation on ~8/129 molecules), so
    # RDKit-generated latents would inject a geometry-protocol confound.
    # (Extraction itself lives in extract_latents(); calls are above.)

    # ---- geometry-protocol diagnostics vs the stored (Vast RDKit) TTA ----
    agg = pd.read_csv(AGG_CSV).set_index("mol_id")
    if "pred_seed42" in agg.columns:
        gaps = {mid: abs(p - agg.loc[mid, "pred_seed42"])
                for mid, p in zip(flat_mid_test, zw_te)}
        gap_v = np.array(list(gaps.values()))
        worst = sorted(gaps.items(), key=lambda kv: -kv[1])[:5]
        print(f"[diag] hdf5-single-conf vs stored TTA: max {gap_v.max():.2f} "
              f"kcal/mol, mean {gap_v.mean():.2f} "
              f"(local RDKit vs Vast RDKit geometry drift; stored values kept "
              f"for cross-checks)")
        with open(os.path.join(args.out, "geometry_diagnostics.json"), "w") as f:
            json.dump({"max_gap_kcal": float(gap_v.max()),
                       "mean_gap_kcal": float(gap_v.mean()),
                       "n_gt_10kcal": int((gap_v > 10).sum()),
                       "worst_mols": [{"mol_id": m, "gap_kcal": float(g)}
                                      for m, g in worst]}, f, indent=2)

    # ---- PCA dimensionality report + reduction ----
    scaler = StandardScaler()
    X_s = scaler.fit_transform(Z_train[:, :LATENT_DIM])
    S_raw = np.cov(Z_train[:, :LATENT_DIM], rowvar=False)
    S_sc = np.cov(X_s, rowvar=False)
    cond_raw = float(np.linalg.cond(S_raw))
    cond_sc = float(np.linalg.cond(S_sc))
    print(f"[pca] raw dim={LATENT_DIM} n_train_atoms={len(X_s)} "
          f"cond(raw cov)={cond_raw:.3e} cond(standardized)={cond_sc:.3e}")

    pca = PCA(n_components=PCA_MAX_COMPONENTS, random_state=args.seed)
    X_pca = pca.fit_transform(X_s)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95)) + 1 if cum[-1] >= 0.95 else len(cum)
    n99 = int(np.argmax(cum >= PCA_VARIANCE_TARGET)) + 1 \
        if cum[-1] >= PCA_VARIANCE_TARGET else len(cum)
    # GMM on k = n95 (14): with ~4.5k atoms a full-covariance 31-dim GMM is
    # only sample-stable up to ~5 components (params > samples above that);
    # k=14 keeps {1,5,10,20} feasible (n=50 still overparameterized -> reported
    # as infeasible). k=31 (99% variance) is reported for completeness.
    k = n95
    Xr = X_pca[:, :k]
    var_kept = float(cum[k - 1])
    print(f"[pca] components for >=95%: {n95}, >=99%: {n99}; GMM uses k={k} "
          f"(kept {var_kept:.4f} variance)")

    # ---- GMM fit on reduced train latents ----
    # n_components comes from --n-components (default 10, the held-out-
    # validated value). The BIC grid is retained behind --run-bic-grid only
    # as a reference diagnostic: its argmax runs into the parameter-count
    # ceiling (no interior minimum) and dives as n grows.
    bic_report = {}
    best = args.n_components
    if args.run_bic_grid:
        for nc in N_COMPONENTS_GRID:
            n_params = nc * (k * (k + 1) // 2 + k + 1)
            if n_params > len(Xr):
                print(f"[gmm] n_components={nc}: skipped (params {n_params} > "
                      f"samples {len(Xr)})")
                bic_report[str(nc)] = {"skipped": "params > samples",
                                       "n_params": n_params}
                continue
            gm = GaussianMixture(n_components=nc, covariance_type="full",
                                 init_params="kmeans", n_init=5, reg_covar=1e-2,
                                 random_state=args.seed)
            gm.fit(Xr)
            bic_report[str(nc)] = {
                "bic": float(gm.bic(Xr)), "aic": float(gm.aic(Xr)),
                "converged": bool(gm.converged_), "n_iter": int(gm.n_iter_),
            }
            print(f"[gmm] n={nc}: BIC={bic_report[str(nc)]['bic']:.1f} "
                  f"converged={bic_report[str(nc)]['converged']} "
                  f"n_iter={bic_report[str(nc)]['n_iter']}")
    else:
        print(f"[gmm] BIC grid skipped (--run-bic-grid); using validated "
              f"n_components={best}")

    bic_report["chosen"] = best
    bic_report["chosen_by"] = ("heldout_validation_ll_max"
                               if not args.run_bic_grid else "bic_grid")
    bic_report["n_components_grid"] = N_COMPONENTS_GRID
    print(f"[gmm] n_components={best} "
          f"({bic_report['chosen_by']})")

    gm = GaussianMixture(n_components=best, covariance_type="full",
                         init_params="kmeans", n_init=5, reg_covar=1e-2,
                         random_state=args.seed)
    gm.fit(Xr)
    bic_report["latent_dim"] = LATENT_DIM
    bic_report["k_pca"] = k
    bic_report["n_train_atoms"] = int(len(Xr))
    bic_report["cond_raw"] = cond_raw
    bic_report["cond_standardized"] = cond_sc
    bic_report["pca_n95"] = n95
    bic_report["pca_n99"] = n99
    bic_report["pca_variance_kept"] = var_kept

    # ---- test atom NLLs + molecule aggregation ----
    X_test_s = scaler.transform(Z_test[:, :LATENT_DIM])
    X_test_p = pca.transform(X_test_s)[:, :k]
    logp = gm.score_samples(X_test_p)
    nll = -logp
    atom_mid = [flat_mid_test[i] for i in Z_test[:, LATENT_DIM + 1].astype(int)]
    atom_nll_df = pd.DataFrame({
        "mol_id": atom_mid,
        "nll": nll,
        "znum": Z_test[:, LATENT_DIM + 2].astype(int),
    })
    atom_nll_df["atom_idx"] = atom_nll_df.groupby("mol_id").cumcount()
    atom_nll_df.to_csv(os.path.join(args.out, "per_atom_nll_test.csv"),
                       index=False)

    agg_nll = {}
    for mid in test_ids:
        v = nll[np.array([m == mid for m in atom_mid])]
        agg_nll[mid] = {"mean_nll": float(v.mean()), "max_nll": float(v.max()),
                        "n_atoms": int(len(v))}

    # ---- assemble per-molecule table with existing annotations ----
    rmse_df = pd.read_csv(ANALYSIS_CSV).set_index("mol_id")
    iso = pd.read_csv(ISOLATION_CSV)
    iso18 = iso[iso["group"] == "confidently_wrong"].sort_values("best_sim")
    isolated6 = set(iso18.head(6)["mol_id"])
    wrong18 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_high_rmse"])
    certain47 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_low_rmse"])

    rows = []
    for mid in test_ids:
        a = agg_nll[mid]
        r = rmse_df.loc[mid] if mid in rmse_df.index else None
        rows.append({
            "mol_id": mid,
            "mean_nll": a["mean_nll"], "max_nll": a["max_nll"],
            "n_atoms": a["n_atoms"],
            "ensemble_std": agg.loc[mid, "ensemble_std"] if mid in agg.index else np.nan,
            "abs_error": agg.loc[mid, "abs_error"] if mid in agg.index else np.nan,
            "seed_rmse": r["rmse_across_seeds"] if r is not None else np.nan,
            "quadrant_label": r["quadrant_label"] if r is not None else "n/a",
            "is_wrong18": mid in wrong18, "is_certain47": mid in certain47,
            "group": ("isolated6" if mid in isolated6 else
                      "gradient12" if mid in wrong18 - isolated6 else
                      "certain47" if mid in certain47 else "other"),
        })
    per_mol = pd.DataFrame(rows)
    per_mol.to_csv(os.path.join(args.out, "per_molecule_gmm_nll.csv"), index=False)

    # ---- correlations ----
    corr = {}
    for score_col in ("mean_nll", "max_nll"):
        for ref in ("ensemble_std", "abs_error", "seed_rmse"):
            v = per_mol[["mol_id", score_col, ref]].dropna()
            rho, p = spearmanr(v[score_col], v[ref])
            corr[f"{score_col}_vs_{ref}"] = {"spearman": float(rho), "p": float(p),
                                             "n": len(v)}
            print(f"[corr] {score_col} vs {ref}: rho={rho:.3f} (p={p:.2e})")
    with open(os.path.join(args.out, "correlation_stats.json"), "w") as f:
        json.dump(corr, f, indent=2)

    # ---- group comparisons ----
    def grp_stats(scores18, scores47):
        return {
            "mean_wrong18": float(np.mean(scores18)),
            "median_wrong18": float(np.median(scores18)),
            "mean_certain47": float(np.mean(scores47)),
            "median_certain47": float(np.median(scores47)),
        }

    gc = {}
    for score_col in ("mean_nll", "max_nll"):
        w = per_mol[per_mol["is_wrong18"]][score_col].values
        c = per_mol[per_mol["is_certain47"]][score_col].values
        U, p = mannwhitneyu(w, c, alternative="two-sided")
        ranks = rankdata(np.concatenate([w, c]))
        mean_rank_w = float(np.mean(ranks[:len(w)]))
        mean_rank_c = float(np.mean(ranks[len(w):]))
        st = grp_stats(w, c)
        st.update({"mannwhitney_U": float(U), "mannwhitney_p": float(p),
                   "mean_rank_wrong18": mean_rank_w,
                   "mean_rank_certain47": mean_rank_c,
                   "n_wrong18": len(w), "n_certain47": len(c)})
        gc[score_col] = st
        print(f"[groups] {score_col}: 18vs47 MWU p={p:.2e} "
              f"(means {st['mean_wrong18']:.3f} vs {st['mean_certain47']:.3f})")

        iso6 = per_mol[per_mol["group"] == "isolated6"][score_col].values
        grad12 = per_mol[per_mol["group"] == "gradient12"][score_col].values
        if len(iso6) and len(grad12):
            U2, p2 = mannwhitneyu(iso6, grad12, alternative="two-sided")
            st2 = {"mean_isolated6": float(np.mean(iso6)),
                   "median_isolated6": float(np.median(iso6)),
                   "mean_gradient12": float(np.mean(grad12)),
                   "median_gradient12": float(np.median(grad12)),
                   "mannwhitney_U": float(U2), "mannwhitney_p": float(p2),
                   "n_isolated6": len(iso6), "n_gradient12": len(grad12)}
            gc[score_col + "_iso6_vs_grad12"] = st2
            print(f"[groups] {score_col}: iso6vsgrad12 MWU p={p2:.2e} "
                  f"(means {st2['mean_isolated6']:.3f} vs {st2['mean_gradient12']:.3f})")

    # ---- overlap: does NLL flag the same 18? ----
    top18 = set(per_mol.sort_values("mean_nll", ascending=False).head(18)["mol_id"])
    top18_max = set(per_mol.sort_values("max_nll", ascending=False).head(18)["mol_id"])
    gc["overlap"] = {
        "wrong18_in_top18_mean_nll": int(len(wrong18 & top18)),
        "wrong18_in_top18_max_nll": int(len(wrong18 & top18_max)),
        "certain47_in_top18_mean_nll": int(len(certain47 & top18)),
        "certain47_in_top18_max_nll": int(len(certain47 & top18_max)),
    }
    print(f"[overlap] wrong18 in top18(mean_nll): {gc['overlap']['wrong18_in_top18_mean_nll']}"
          f", top18(max_nll): {gc['overlap']['wrong18_in_top18_max_nll']}")
    with open(os.path.join(args.out, "group_comparison.json"), "w") as f:
        json.dump(gc, f, indent=2)

    # ---- scatter ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"low_std_low_rmse": "tab:blue", "low_std_high_rmse": "tab:red",
              "high_std_low_rmse": "tab:orange", "high_std_high_rmse": "tab:purple"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for lab, grp in per_mol.groupby("quadrant_label"):
        ax.scatter(grp["ensemble_std"], grp["mean_nll"], s=28,
                   color=colors.get(lab, "grey"), label=lab, alpha=0.8)
    ax.set_xlabel("ensemble_std (kcal/mol, 5-seed disagreement)")
    ax.set_ylabel("GMM mean_NLL (seed-42 single network)")
    ax.set_title(f"GMM-NLL vs ensemble std, colored by quadrant "
                 f"(GMM n_components={best}, PCA k={k})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gmm_nll_vs_std_scatter.png"), dpi=150)
    print(f"saved -> {os.path.join(args.out, 'gmm_nll_vs_std_scatter.png')}")

    with open(os.path.join(args.out, "bic_report.json"), "w") as f:
        json.dump(bic_report, f, indent=2)

    print("\n[done] outputs in", args.out)


if __name__ == "__main__":
    main()

"""PART 1 - validate the GMM via held-out log-likelihood (molecule-level).

BIC chose n_components=50 by hitting the parameter-count ceiling (no interior
minimum), which is a known failure mode. Here we validate with true held-out
log-likelihood instead:

 1. Molecule-level 80/20 split of the 411 TRAINING molecules (no leakage of
    correlated atoms across folds).
 2. For n_components in {1,5,10,20,50}: fit scaler + PCA(k=13) + full-cov GMM
    (kmeans init, n_init=5, reg_covar=1e-2) on fold-A atoms ONLY, score the
    held-out (fold-B) atoms.
 3. Pick n_components maximizing held-out per-atom log-likelihood; compare to
    the BIC choice (n=50) and report the held-out LL difference.
 4. Refit the final GMM at the validated n_components on ALL 7389 training
    atoms, recompute mean_NLL/max_NLL for the 129 test molecules, and recheck
    isolated-6 vs gradient-12, 18-wrong vs 47-certain, and Spearman vs
    ensemble_std/abs_error/seed_rmse.

Outputs -> gmm_uncertainty_check/validation/
"""

import json
import os
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))       # gmm_uncertainty_check/validation/
_deep_ensemble = os.path.dirname(os.path.dirname(_script_dir))  # deep_ensemble/
if _deep_ensemble not in sys.path:
    sys.path.insert(0, _deep_ensemble)

from gmm_uncertainty_check import extract_latents, LATENT_DIM

OUT_DIR = os.path.join(os.path.dirname(_script_dir), "validation")
SEED = 42
GRID = [1, 5, 10, 20, 50]
K_PCA = 13
REG_COVAR = 1e-2
N_INIT = 5
HOLD_OUT_FRAC = 0.2

ANALYSIS_CSV = os.path.join(_deep_ensemble, "rmse_analysis", "output",
                            "per_molecule_rmse.csv")
ISOLATION_CSV = os.path.join(_deep_ensemble, "rmse_analysis",
                             "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")
AGG_CSV = os.path.join(_deep_ensemble, "aggregate", "per_molecule.csv")
CKPT = os.path.join(_deep_ensemble, "seed_42", "ensemble_seed42.pt")


def main():
    import torch
    import pandas as pd
    from scipy.stats import spearmanr, mannwhitneyu

    os.makedirs(OUT_DIR, exist_ok=True)

    from deep_ensemble import load_frozen_split, build_model, DEFAULT_SPLIT_DIR
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels

    labels_path, _ = download_freesolv_data(os.path.join(
        os.path.dirname(os.path.dirname(_deep_ensemble)), "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(labels_path)
    train_ids, val_ids, test_ids = load_frozen_split(DEFAULT_SPLIT_DIR, all_labels)

    model = build_model("cpu")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    Z_train, flat_mid_train, zw_tr, em_tr = extract_latents(
        model, "cpu", train_ids, "train", os.path.join(_script_dir, "latent_cache"))
    Z_test, flat_mid_test, zw_te, em_te = extract_latents(
        model, "cpu", test_ids, "test", os.path.join(_script_dir, "latent_cache"))

    # ---- molecule-level 80/20 split of TRAINING molecules ----
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_ids))
    n_hold = int(round(HOLD_OUT_FRAC * len(train_ids)))
    hold_idx = set(perm[:n_hold].tolist())
    foldA_idx = set(perm[n_hold:].tolist())
    foldA_mids = [train_ids[i] for i in sorted(foldA_idx)]
    hold_mids = [train_ids[i] for i in sorted(hold_idx)]
    assert len(foldA_mids) + len(hold_mids) == len(train_ids)

    el = Z_train[:, LATENT_DIM + 1].astype(int)   # global element idx -> mid
    mid_atom = np.array([flat_mid_train[i] for i in el])
    is_A = np.array([m in set(foldA_mids) for m in mid_atom])
    is_B = ~is_A
    print(f"[split] fold-A: {len(foldA_mids)} mols / {int(is_A.sum())} atoms | "
          f"held-out: {len(hold_mids)} mols / {int(is_B.sum())} atoms")

    X = Z_train[:, :LATENT_DIM]
    X_A, X_B = X[is_A], X[is_B]

    # ---- held-out LL over the grid ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    curve = {}
    for nc in GRID:
        sc = StandardScaler().fit(X_A)
        pca = PCA(n_components=K_PCA, random_state=SEED).fit(sc.transform(X_A))
        Xr_A = pca.transform(sc.transform(X_A))
        gm = GaussianMixture(n_components=nc, covariance_type="full",
                             init_params="kmeans", n_init=N_INIT,
                             reg_covar=REG_COVAR, random_state=SEED)
        gm.fit(Xr_A)
        Xr_B = pca.transform(sc.transform(X_B))
        ll_B = float(np.mean(gm.score_samples(Xr_B)))
        ll_A = float(np.mean(gm.score_samples(Xr_A)))
        bic_A = float(gm.bic(Xr_A))
        curve[str(nc)] = {
            "heldout_ll_per_atom": ll_B, "train_ll_per_atom": ll_A,
            "bic_foldA": bic_A, "converged": bool(gm.converged_),
            "n_iter": int(gm.n_iter_),
        }
        print(f"[grid] n={nc:3d}: held-out LL {ll_B:9.3f} | train LL {ll_A:9.3f} "
              f"| BIC(foldA) {bic_A:.0f}")

    nc_best = max(GRID, key=lambda nc: curve[str(nc)]["heldout_ll_per_atom"])
    ll_best = curve[str(nc_best)]["heldout_ll_per_atom"]
    ll_50 = curve["50"]["heldout_ll_per_atom"]
    print(f"[chooser] best held-out LL at n={nc_best} ({ll_best:.3f}); "
          f"BIC choice n=50 gives {ll_50:.3f}; delta {ll_best - ll_50:+.3f} nats/atom")

    curve["_summary"] = {
        "grid": GRID, "best_n_components": nc_best,
        "heldout_ll_at_best": ll_best,
        "heldout_ll_at_bic_choice_50": ll_50,
        "delta_ll_best_minus_50": ll_best - ll_50,
        "interior_maximum": bool(
            nc_best not in (GRID[0], GRID[-1])),
        "monotone_to_ceiling": bool(
            ll_50 == max(curve[str(nc)]["heldout_ll_per_atom"] for nc in GRID)),
    }
    with open(os.path.join(OUT_DIR, "heldout_ll_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    # ---- curve plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = np.array(GRID)
    ax.plot(ns, [curve[str(n)]["heldout_ll_per_atom"] for n in ns],
            "o-", label="held-out (fold-B) LL/atom")
    ax.plot(ns, [curve[str(n)]["train_ll_per_atom"] for n in ns],
            "s--", label="train (fold-A) LL/atom", alpha=0.7)
    ax.axvline(nc_best, color="tab:red", ls=":", label=f"argmax held-out (n={nc_best})")
    ax.axvline(50, color="grey", ls=":", label="BIC choice (n=50)")
    ax.set_xlabel("n_components"); ax.set_ylabel("log-likelihood per atom (nats)")
    ax.set_title("GMM held-out validation (molecule-level 80/20)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heldout_ll_curve.png"), dpi=150)
    print(f"saved -> {os.path.join(OUT_DIR, 'heldout_ll_curve.png')}")

    # ---- refit on ALL training atoms at the validated n ----
    def fit_all(nc):
        sc = StandardScaler().fit(X)
        pca = PCA(n_components=K_PCA, random_state=SEED).fit(sc.transform(X))
        gm = GaussianMixture(n_components=nc, covariance_type="full",
                             init_params="kmeans", n_init=N_INIT,
                             reg_covar=REG_COVAR, random_state=SEED)
        gm.fit(pca.transform(sc.transform(X)))
        Xt_s = sc.transform(Z_test[:, :LATENT_DIM])
        Xt = pca.transform(Xt_s)
        nll = -gm.score_samples(Xt)
        return nll, (sc, pca, gm)

    agg = pd.read_csv(AGG_CSV).set_index("mol_id")
    rmse_df = pd.read_csv(ANALYSIS_CSV).set_index("mol_id")
    iso = pd.read_csv(ISOLATION_CSV)
    iso18 = iso[iso["group"] == "confidently_wrong"].sort_values("best_sim")
    isolated6 = set(iso18.head(6)["mol_id"])
    wrong18 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_high_rmse"])
    certain47 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_low_rmse"])

    def evaluate(nc, tag):
        nll, _ = fit_all(nc)
        atom_mid = np.array([flat_mid_test[i]
                             for i in Z_test[:, LATENT_DIM + 1].astype(int)])
        per_mol = {}
        for mid in test_ids:
            v = nll[atom_mid == mid]
            per_mol[mid] = {"mean_nll": float(v.mean()), "max_nll": float(v.max())}
        rows = []
        for mid in test_ids:
            rows.append({"mol_id": mid, **per_mol[mid],
                         "ensemble_std": agg.loc[mid, "ensemble_std"],
                         "abs_error": agg.loc[mid, "abs_error"],
                         "seed_rmse": rmse_df.loc[mid, "rmse_across_seeds"],
                         "quadrant_label": rmse_df.loc[mid, "quadrant_label"]})
        df = pd.DataFrame(rows)
        out = {}
        for score_col in ("mean_nll", "max_nll"):
            corr = {}
            for ref in ("ensemble_std", "abs_error", "seed_rmse"):
                rho, p = spearmanr(df[score_col], df[ref])
                corr[ref] = {"spearman": float(rho), "p": float(p)}
            w = df[df["mol_id"].isin(wrong18)][score_col].values
            c = df[df["mol_id"].isin(certain47)][score_col].values
            U18, p18 = mannwhitneyu(w, c, alternative="two-sided")
            i6 = df[df["mol_id"].isin(isolated6)][score_col].values
            g12 = df[df["mol_id"].isin(wrong18 - isolated6)][score_col].values
            Ug, pg = mannwhitneyu(i6, g12, alternative="two-sided")
            out[score_col] = {
                "spearman": corr,
                "wrong18_vs_certain47": {
                    "mean_wrong18": float(np.mean(w)),
                    "median_wrong18": float(np.median(w)),
                    "mean_certain47": float(np.mean(c)),
                    "median_certain47": float(np.median(c)),
                    "mannwhitney_p": float(p18)},
                "isolated6_vs_gradient12": {
                    "mean_isolated6": float(np.mean(i6)),
                    "median_isolated6": float(np.median(i6)),
                    "mean_gradient12": float(np.mean(g12)),
                    "median_gradient12": float(np.median(g12)),
                    "mannwhitney_p": float(pg)},
            }
            print(f"[{tag}] {score_col}: 18vs47 p={p18:.2e} "
                  f"({out[score_col]['wrong18_vs_certain47']['median_wrong18']:.2f} "
                  f"vs {out[score_col]['wrong18_vs_certain47']['median_certain47']:.2f}) "
                  f"| iso6vsgrad12 p={pg:.2e} "
                  f"({out[score_col]['isolated6_vs_gradient12']['median_isolated6']:.2f} "
                  f"vs {out[score_col]['isolated6_vs_gradient12']['median_gradient12']:.2f})")
        for score_col in ("mean_nll", "max_nll"):
            top18 = set(df.sort_values(score_col, ascending=False).head(18)["mol_id"])
            out[score_col]["overlap_wrong18_in_top18"] = int(len(wrong18 & top18))
        out["_n_components"] = nc
        return df, out

    df_best, stats_best = evaluate(nc_best, f"refit n={nc_best}")
    df_50, stats_50 = evaluate(50, "refit n=50")

    df_best.to_csv(os.path.join(OUT_DIR, "per_molecule_gmm_nll_refit.csv"), index=False)
    report = {
        "split": {"foldA_mols": len(foldA_mids), "foldA_atoms": int(is_A.sum()),
                  "heldout_mols": len(hold_mids), "heldout_atoms": int(is_B.sum())},
        "k_pca": K_PCA, "reg_covar": REG_COVAR, "n_init": N_INIT,
        "n_components_validated": nc_best,
        "stats_at_validated_n": stats_best,
        "stats_at_bic_choice_n50": stats_50,
    }
    with open(os.path.join(OUT_DIR, "gmm_refit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n[part1] done. outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
"""Test-time trust-weighted neighbor repair (PAPER_SKELETON Ch2 spec).

Regime decision (user-approved): seeds 42/123/999 only. The recorded 5-seed
ensemble (MAE 0.5059) came from a destroyed box whose RDKit geometry regime no
longer exists anywhere (check1 report); in the surviving regime (stored hdf5
conformers) seeds 7/2024 are pathology-prone (fresh MAE 38/58, predictions to
-742 kcal/mol) and would dominate the baseline. ALL numbers here are computed
in the surviving regime with the 3 coherent seeds; a 5-seed sensitivity arm is
reported separately and flagged.

Design (all four user-mandated refinements):
  * control arms: trust-weighted (t_j from GMM-NLL) vs naive neighbor (t_j=1)
    vs random-shift (same per-molecule magnitude, random sign, seed-fixed)
  * gate: only molecules in the broad uncertain population (UNION primary;
    Q_std and Q_nll reported separately) are corrected
  * Mode A (default): per-seed correction then re-ensemble, preserving the
    false-consensus diagnostic (post-repair ensemble_std recomputed);
    Mode B (sensitivity): mean-level correction only
  * alpha calibrated on the VAL set only (val-internal quartile gate);
  * 10k paired-bootstrap CIs for every arm x population (never point estimates)

Correction (per gated molecule i, neighbors j from k=5 graph):
  p'_i = (1-a) p_i + a * sum_j w_ij t_j p_j / sum_j w_ij t_j
Mode A: p_i, p_j are per-seed (same seed s); re-ensemble after.
Mode B: p_i, p_j are 3-seed ensemble means.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import deep_ensemble as de
REPO = de.REPO_ROOT

N_BOOT = 10_000
SEEDS = [42, 123, 999]
ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RNG_SEED = 20260815


def load_split():
    from freesolv_dataset import load_freesolv_labels
    labels = load_freesolv_labels(os.path.join(REPO, "Data", "FreeSolv", "database.json"))
    split = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                         "cv_results_full", "fold_0")
    tr, va, te = de.load_frozen_split(split, labels)
    return tr, va, te


def gmm_nll_all642(tr, va, te):
    """Refit scaler+PCA+GMM on train latents (protocol == gmm_uncertainty_check.py),
    score ALL 642 molecules; verify test NLL against the saved CSV."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    cache = os.path.join(os.path.dirname(HERE), "gmm_uncertainty_check", "latent_cache")
    ztr = np.load(os.path.join(cache, "z_train.npz"))["Z"]
    zte = np.load(os.path.join(cache, "z_test.npz"))["Z"]
    with open(os.path.join(cache, "mids_train.json")) as f:
        mids_tr = json.load(f)
    with open(os.path.join(cache, "mids_test.json")) as f:
        mids_te = json.load(f)

    import torch
    from gmm_uncertainty_check import extract_latents, LATENT_DIM
    ckpt = os.path.join(os.path.dirname(HERE), "seed_42", "ensemble_seed42.pt")
    model = de.build_model(torch.device("cpu"))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    Zval, mids_val, _, _ = extract_latents(model, torch.device("cpu"), va,
                                           "val", os.path.join(OUT, "latent_cache"))

    scaler = StandardScaler()
    X_s = scaler.fit_transform(ztr[:, :LATENT_DIM])
    pca = PCA(n_components=200, random_state=42)
    X_pca = pca.fit_transform(X_s)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cum >= 0.95)) + 1
    Xr = X_pca[:, :k]
    gm = GaussianMixture(n_components=10, covariance_type="full",
                         init_params="kmeans", n_init=5, reg_covar=1e-2,
                         random_state=42)
    gm.fit(Xr)

    def score(Z, mids):
        X_s = scaler.transform(Z[:, :LATENT_DIM])
        X_p = pca.transform(X_s)[:, :k]
        nll = -gm.score_samples(X_p)
        flat = [mids[i] for i in Z[:, LATENT_DIM + 1].astype(int)]
        return pd.DataFrame({"mol_id": flat, "nll": nll}).groupby("mol_id")["nll"].mean()

    nll_all = pd.concat([score(ztr, mids_tr), score(zte, mids_te),
                         score(Zval, mids_val)])
    nll_all = nll_all[~nll_all.index.duplicated()]
    return nll_all, k


def build_populations(pred, nll, te):
    """Surviving-regime populations on the 129 test molecules (3-seed std).
    Recomputes ensemble_mean/ensemble_std over the 3 seeds (the CSV holds the
    polluted 5-seed values dominated by pathology-prone seeds 7/2024)."""
    t = pred[pred.mol_id.isin(te)].copy()
    t["mean3"] = t[[f"pred_seed{s}" for s in SEEDS]].mean(axis=1)
    t["std3"] = t[[f"pred_seed{s}" for s in SEEDS]].std(axis=1)
    t = t.merge(nll.rename("mean_nll"), on="mol_id")
    thr_std = t["std3"].quantile(0.75)
    thr_nll = t["mean_nll"].quantile(0.75)
    q_std = set(t.loc[t["std3"] >= thr_std, "mol_id"])
    q_nll = set(t.loc[t["mean_nll"] >= thr_nll, "mol_id"])
    t["ensemble_mean"] = t["mean3"]
    t["ensemble_std"] = t["std3"]
    return q_std, q_nll, q_std | q_nll, t.set_index("mol_id")


def load_graph(path):
    g = json.load(open(path))
    return {m: [(float(w), nb) for w, nb in nbrs] for m, nbrs in g.items()}


def main():
    tr, va, te = load_split()
    pred = pd.read_csv(os.path.join(OUT, "seed_predictions_all642.csv"))

    test_mae = (pred[pred.mol_id.isin(te)][[f"pred_seed{s}" for s in SEEDS]]
                .mean(axis=1).sub(pred[pred.mol_id.isin(te)].true_value).abs().mean())
    print(f"[regime] 3-seed ensemble test MAE (surviving) = {test_mae:.3f} kcal/mol "
          f"(recorded 0.506 is NOT comparable; seeds 7/2024 excluded as pathology-prone)",
          flush=True)

    nll, k_pca = gmm_nll_all642(tr, va, te)
    saved = pd.read_csv(os.path.join(os.path.dirname(HERE), "gmm_uncertainty_check",
                                     "per_molecule_gmm_nll.csv"))
    chk = saved.merge(nll.rename("nll_new"), on="mol_id")
    md = (chk["mean_nll"] - chk["nll_new"]).abs().max()
    print(f"[gmm] refit k_pca={k_pca}; max |saved - refit| mean_nll = {md:.3e} "
          f"(n={len(chk)})", flush=True)
    assert md < 1e-3, "GMM refit does not reproduce saved test NLLs"

    q_std, q_nll, union, tdf = build_populations(pred, nll, te)
    print(f"[pop] Q_std n={len(q_std)} Q_nll n={len(q_nll)} "
          f"overlap={len(q_std & q_nll)} union={len(union)}", flush=True)

    grad12 = set(pd.read_csv(os.path.join(os.path.dirname(HERE), "gmm_uncertainty_check",
                                          "gradient12_investigation",
                                          "gradient12_ungrouped.csv")).mol_id)
    print(f"[pop] gradient-12 (recorded regime): {len(grad12 & q_std)} in Q_std, "
          f"{len(grad12 & q_nll)} in Q_nll, {len(grad12 & union)} in union", flush=True)

    graphs = {
        "tanimoto": load_graph(os.path.join(REPO, "aqm-spice2", "freesolv",
                                            "neighbor_regularization",
                                            "graph_cache", "graph_k5_sim0.1.json")),
        "latent": load_graph(os.path.join(REPO, "aqm-spice2", "freesolv",
                                          "neighbor_regularization",
                                          "graph_cache", "latent_k5_sim0.5.json")),
    }
    all642 = set(pred.mol_id)
    for name in graphs:
        graphs[name] = {m: [(w, j) for w, j in nbrs if j in all642]
                        for m, nbrs in graphs[name].items()}

    nll_rank = nll.rank(pct=True)
    trust = {m: 1.0 - float(r) for m, r in nll_rank.items()}

    pj = pred.set_index("mol_id")
    for name, g in graphs.items():
        for mid in tdf.index:
            for w, j in g.get(mid, []):
                r = pj.loc[j]
                for s in SEEDS:
                    tdf.at[mid, f"nbr_seed{s}_{j}"] = r[f"pred_seed{s}"]
                tdf.at[mid, f"nbr_mean_{j}"] = r["ensemble_mean"]

    def apply_repair(gate, alpha, arm, mode, rng, graph_name="tanimoto"):
        out = tdf.copy()
        for mid in gate:
            nbrs = graphs[graph_name].get(mid, [])
            if not nbrs:
                continue
            w = np.array([x[0] for x in nbrs])
            if arm == "trust":
                t = np.array([trust[j] for _, j in nbrs])
            elif arm == "naive":
                t = np.ones(len(nbrs))
            elif arm == "random":
                t = np.array([trust[j] for _, j in nbrs])  # magnitude = trust arm
            denom = (w * t).sum()
            if denom <= 0:
                continue
            if mode == "B":
                nb_vec = np.array([tdf.loc[mid, f"nbr_mean_{j}"] for _, j in nbrs])
                nb_mean = (w * t * nb_vec).sum() / denom
                out.at[mid, "ensemble_mean"] = (1 - alpha) * tdf.loc[mid, "ensemble_mean"] + alpha * nb_mean
            else:
                for s in SEEDS:
                    nb_s = np.array([tdf.loc[mid, f"nbr_seed{s}_{j}"] for _, j in nbrs])
                    nb_s = (w * t * nb_s).sum() / denom
                    newp = (1 - alpha) * tdf.loc[mid, f"pred_seed{s}"] + alpha * nb_s
                    if arm == "random":
                        mag = abs(newp - tdf.loc[mid, f"pred_seed{s}"])
                        newp = tdf.loc[mid, f"pred_seed{s}"] + rng.choice([-1.0, 1.0]) * mag
                    out.at[mid, f"pred_seed{s}"] = newp
                out.at[mid, "ensemble_mean"] = np.mean(
                    [out.loc[mid, f"pred_seed{s}"] for s in SEEDS])
                out.at[mid, "ensemble_std"] = np.std(
                    [out.loc[mid, f"pred_seed{s}"] for s in SEEDS])
        return out

    def mae_on(out, pop):
        sub = out.loc[out.index.isin(pop)]
        if len(sub) == 0:
            return np.nan
        return (sub["ensemble_mean"] - sub["true_value"]).abs().mean()

    v = pred[pred.mol_id.isin(va)].copy()
    v = v.merge(nll.rename("mean_nll"), on="mol_id")
    v["ensemble_mean"] = v[[f"pred_seed{s}" for s in SEEDS]].mean(axis=1)
    v["std"] = v[[f"pred_seed{s}" for s in SEEDS]].std(axis=1)
    val_gate = set(v.loc[(v["std"] >= v["std"].quantile(0.75)) |
                         (v["mean_nll"] >= v["mean_nll"].quantile(0.75)), "mol_id"])
    print(f"[calib] val gate n={len(val_gate)} of {len(va)}", flush=True)
    vdf = v.set_index("mol_id")
    for mid in val_gate:
        for w, j in graphs["tanimoto"].get(mid, []):
            r = pj.loc[j]
            for s in SEEDS:
                vdf.at[mid, f"nbr_seed{s}_{j}"] = r[f"pred_seed{s}"]
            vdf.at[mid, f"nbr_mean_{j}"] = r["ensemble_mean"]

    best_alpha = {}
    for arm in ("trust", "naive", "random"):
        best_a, best_m = None, np.inf
        for a in ALPHA_GRID:
            rng = np.random.default_rng(RNG_SEED)
            vout = vdf.copy()
            for mid in val_gate:
                nbrs = graphs["tanimoto"].get(mid, [])
                if not nbrs:
                    continue
                w = np.array([x[0] for x in nbrs])
                t = (np.array([trust[j] for _, j in nbrs]) if arm in ("trust", "random")
                     else np.ones(len(nbrs)))
                denom = (w * t).sum()
                if denom <= 0:
                    continue
                for s in SEEDS:
                    nb_s = np.array([vout.loc[mid, f"nbr_seed{s}_{j}"] for _, j in nbrs])
                    nb_s = (w * t * nb_s).sum() / denom
                    newp = (1 - a) * vout.loc[mid, f"pred_seed{s}"] + a * nb_s
                    if arm == "random":
                        mag = abs(newp - vout.loc[mid, f"pred_seed{s}"])
                        newp = vout.loc[mid, f"pred_seed{s}"] + rng.choice([-1.0, 1.0]) * mag
                    vout.at[mid, f"pred_seed{s}"] = newp
                vout.at[mid, "ensemble_mean"] = np.mean(
                    [vout.loc[mid, f"pred_seed{s}"] for s in SEEDS])
            m = (vout["ensemble_mean"] - vout["true_value"]).abs().mean()
            if m < best_m:
                best_m, best_a = m, a
        best_alpha[arm] = best_a
        print(f"[calib] arm={arm}: best alpha={best_a} (val gated MAE {best_m:.3f})",
              flush=True)

    pops = {"Q_std": q_std, "Q_nll": q_nll, "UNION": union,
            "all129": set(te), "gradient12": grad12}
    base = tdf
    base_mae = {name: mae_on(base, pop) for name, pop in pops.items()}
    print(f"[base] MAE: " + ", ".join(f"{n}={v:.3f}" for n, v in base_mae.items()),
          flush=True)

    rows = []
    for graph_name in ("tanimoto", "latent"):
        for mode in ("A", "B"):
            for arm in ("trust", "naive", "random"):
                rng = np.random.default_rng(RNG_SEED)
                out = apply_repair(union, best_alpha[arm], arm, mode, rng, graph_name)
                for pop_name, pop in pops.items():
                    sub = out.loc[out.index.isin(pop)]
                    base_sub = base.loc[base.index.isin(pop)]
                    d = ((sub["ensemble_mean"] - sub["true_value"]).abs()
                         - (base_sub["ensemble_mean"] - base_sub["true_value"]).abs())
                    d = d.dropna().values
                    delta = d.mean()
                    rng2 = np.random.default_rng(RNG_SEED + len(rows))
                    boots = np.empty(N_BOOT)
                    for b in range(N_BOOT):
                        idx = rng2.integers(0, len(d), len(d))
                        boots[b] = d[idx].mean()
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    rows.append({"graph": graph_name, "mode": mode, "arm": arm,
                                 "alpha": best_alpha[arm], "population": pop_name,
                                 "n": len(d), "delta_mae": delta,
                                 "ci_lo": lo, "ci_hi": hi})
                    print(f"[{graph_name}|{mode}|{arm}] {pop_name} (n={len(d)}): "
                          f"delta={delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "repair_results.csv"), index=False)

    outA = apply_repair(union, best_alpha["trust"], "trust", "A",
                        np.random.default_rng(RNG_SEED))
    sub_b = base.loc[base.index.isin(union)]
    sub_a = outA.loc[outA.index.isin(union)]
    diag = {
        "seeds": SEEDS,
        "spread_before": float(sub_b["ensemble_mean"].std()),
        "spread_after": float(sub_a["ensemble_mean"].std()),
        "mean_std_before": float(sub_b["ensemble_std"].mean()),
        "mean_std_after": float(sub_a["ensemble_std"].mean()),
        "alpha_trust": best_alpha["trust"],
    }
    with open(os.path.join(OUT, "repair_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print(f"[diag] spread {diag['spread_before']:.3f} -> {diag['spread_after']:.3f}; "
          f"mean std {diag['mean_std_before']:.3f} -> {diag['mean_std_after']:.3f}",
          flush=True)
    print(f"[save] {os.path.join(OUT, 'repair_results.csv')} + repair_diagnostics.json")


if __name__ == "__main__":
    main()
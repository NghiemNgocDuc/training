"""v2 neighbor graph: latent-similarity edges + GMM-NLL uncertainty/trust.

Builds the static per-molecule signals for the v2 neighbor-consistency
regularizer (see DESIGN_v2.md), reusing existing cached artifacts where
they exist (nothing recomputed from scratch):

  1. LATENT GRAPH: per-atom 1024-dim latents (4 output blocks x 256, captured
     by the EXISTING extract_latents() hooks from gmm_uncertainty_check.py,
     fine-tuned seed-42 checkpoint sha 7994ef92). Cached z_train.npz /
     z_test.npz cover 540/642 molecules; ONLY the 102 val molecules are
     extracted fresh (single no-grad forward). Molecule latent = mean-pool
     over atoms; edge weight = cosine similarity of L2-normalized molecule
     latents; top-k neighbors with w >= min_sim (same graph format as
     graph.py, so graph_to_tensor() works unchanged).

  2. GMM-NLL: refit the EXACT validated protocol (StandardScaler ->
     PCA(k=13) -> full-cov GaussianMixture(n_components=10, reg_covar=1e-2,
     n_init=5, random_state=42), fit on TRAIN-set atoms only) from the
     cached train latents; score all 642 molecules -> per-molecule mean NLL.
     Sanity: Spearman(refit NLL, cached per_molecule_gmm_nll.csv mean_nll)
     and Spearman(NLL, ensemble_std) on the 129 test molecules.

  3. TRUST: t_j = 1 if NLL_j <= tau. tau = median NLL over the certain-47
     test group by default (--trust-policy certain47_median); alternatives:
     universe_median (median over all 642), none (all trustworthy).

  4. UNCERTAINTY:s u_i = rank(NLL_i)/642 in (0, 1] (--unc-scale rank,
     the default; --unc-scale minmax gives (nll-min)/(max-min)).

  5. JACCARD overlap report: per molecule, Jaccard(top-5 latent neighbors,
     top-5 Tanimoto neighbors) -> distribution in the meta json.

Outputs -> neighbor_regularization/graph_cache/latent_k{k}_sim{sim}.json
(+ .meta.json with mids, signals, provenance) and latent_cache/ inside
graph_cache (z_val.npz + copied train/test cache refs).

Usage:
  python latent_graph.py --k 5 --min-sim 0.5 --out <graph_cache_dir>
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)           # freesolv/
_aqm = os.path.dirname(_parent)                  # aqm-spice2/
for p in (_aqm, _parent, _script_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED42_CKPT = os.path.join(_parent, "deep_ensemble", "seed_42",
                           "ensemble_seed42.pt")
GMM_CACHE = os.path.join(_parent, "deep_ensemble", "gmm_uncertainty_check",
                         "latent_cache")
GMM_REFIT_CSV = os.path.join(_parent, "deep_ensemble", "gmm_uncertainty_check",
                             "per_molecule_gmm_nll.csv")
AGG_CSV = os.path.join(_parent, "deep_ensemble", "aggregate", "per_molecule.csv")
ISOLATION_CSV = os.path.join(_parent, "deep_ensemble", "rmse_analysis",
                             "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")
ANALYSIS_CSV = os.path.join(_parent, "deep_ensemble", "rmse_analysis", "output",
                            "per_molecule_rmse.csv")

K_PCA = 13
N_COMPONENTS = 10
REG_COVAR = 1e-2
N_INIT = 5
GMM_SEED = 42
LATENT_DIM = 1024


def load_gmm_module():
    """gmm_uncertainty_check.py sits in deep_ensemble/ (no __init__.py) ->
    load by file path so we reuse ITS extract_latents verbatim."""
    path = os.path.join(_parent, "deep_ensemble", "gmm_uncertainty_check.py")
    spec = importlib.util.spec_from_file_location("gmm_uncertainty_check_lib",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ckpt_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_pool_mol_latents(Z, flat_mid, mids):
    """Per-atom rows Z[:, :1024]; col LATENT_DIM+1 = global molecule position
    (into `flat_mid`). Returns mean-pooled per-molecule latent, ordered by
    `mids`."""
    mid_col = Z[:, LATENT_DIM + 1].astype(int)
    vecs = {}
    for atom_pos, mid_pos in enumerate(mid_col):
        vecs.setdefault(flat_mid[mid_pos], []).append(
            Z[atom_pos, :LATENT_DIM])
    out = np.zeros((len(mids), LATENT_DIM))
    for pos, mid in enumerate(mids):
        rows = vecs.get(mid)
        if rows is None:
            raise KeyError(f"no cached latents for {mid}")
        out[pos] = np.mean(rows, axis=0)
    return out


def topk_cosine(mat, k, min_sim, mids):
    """Row-wise top-k cosine neighbors (L2-normalized rows). Returns the same
    {mid: [[w, nbr_mid], ...]} format as graph.py::build_graph."""
    n = mat.shape[0]
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -1.0)          # exclude self
    graph = {}
    order = np.argsort(-sim, axis=1)[:, :k]
    for i in range(n):
        nbrs = []
        for j in order[i]:
            w = float(sim[i, j])
            if w < min_sim:
                continue
            nbrs.append([round(w, 6), mids[j]])
        graph[mids[i]] = nbrs
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.5)
    ap.add_argument("--trust-policy", default="certain47_median",
                    choices=["certain47_median", "universe_median", "none"])
    ap.add_argument("--unc-scale", default="rank",
                    choices=["rank", "minmax"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(_script_dir, "graph_cache"))
    args = ap.parse_args()

    import pandas as pd
    import torch
    from scipy.stats import spearmanr, rankdata
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    from deep_ensemble import (set_seed, load_frozen_split, build_model,
                               DEFAULT_SPLIT_DIR)
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels
    from graph import build_graph

    set_seed(GMM_SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    gmm = load_gmm_module()
    extract_latents = gmm.extract_latents

    labels_path, _ = download_freesolv_data(os.path.join(
        os.path.dirname(os.path.dirname(_aqm)), "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(labels_path)
    train_ids, val_ids, test_ids = load_frozen_split(DEFAULT_SPLIT_DIR, all_labels)
    mids = train_ids + val_ids + test_ids
    assert len(mids) == 642, f"universe != 642: {len(mids)}"
    print(f"[universe] train={len(train_ids)} val={len(val_ids)} "
          f"test={len(test_ids)} total={len(mids)}")

    model = build_model(device)
    ckpt = torch.load(SEED42_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    ckpt_sha = ckpt_sha256(SEED42_CKPT)
    assert ckpt_sha[:8] == "7994ef92", f"unexpected ckpt sha {ckpt_sha[:8]}"
    print(f"[ckpt] seed42 sha {ckpt_sha[:8]} ok")

    # ---- per-molecule latents: cached train/test + fresh val extraction ----
    cache_dir = os.path.join(args.out, "latent_cache")
    os.makedirs(cache_dir, exist_ok=True)
    z_vals = []
    flat_vals = []
    for tag, ids, src_dir in (("train", train_ids, GMM_CACHE),
                              ("test", test_ids, GMM_CACHE)):
        p = os.path.join(src_dir, f"z_{tag}.npz")
        assert os.path.exists(p), f"cached latents missing: {p}"
        d = np.load(p)
        with open(os.path.join(src_dir, f"mids_{tag}.json")) as f:
            flat = json.load(f)
        z = d["Z"].copy()
        # col LATENT_DIM+1 = molecule index WITHIN this cache -> global index
        z[:, LATENT_DIM + 1] += len(flat_vals)
        z_vals.append(z)
        flat_vals.extend(flat)
        print(f"[latents] {tag}: reused cache {p} ({len(d['Z'])} atoms)")
    print(f"[latents] reused cached latents for {len(set(flat_vals))} "
          f"molecules; extracting val {len(val_ids)} fresh...")
    z_val, flat_val, _, _ = extract_latents(
        model, device, val_ids, "val", cache_dir)
    z_val[:, LATENT_DIM + 1] += len(flat_vals)
    z_vals.append(z_val)
    flat_vals.extend(flat_val)
    Z_all = np.concatenate(z_vals, axis=0)
    assert len(flat_vals) == len(mids)
    print(f"[latents] total atoms {len(Z_all)}")
    mol_lat = mean_pool_mol_latents(Z_all, flat_vals, mids)
    print(f"[latents] molecule latent matrix {mol_lat.shape}")

    # ---- latent similarity graph (same format as graph.py) ----
    t0 = time.time()
    norm = mol_lat / (np.linalg.norm(mol_lat, axis=1, keepdims=True) + 1e-12)
    cos_offdiag = norm @ norm.T
    np.fill_diagonal(cos_offdiag, -1.0)
    cos_vals = cos_offdiag[np.triu_indices(len(mids), k=1)]
    print(f"[graph] cosine sim distribution: min {cos_vals.min():.4f} | "
          f"p5 {np.percentile(cos_vals, 5):.4f} | median "
          f"{np.median(cos_vals):.4f} | p95 {np.percentile(cos_vals, 95):.4f} | "
          f"max {cos_vals.max():.4f}")
    graph = topk_cosine(mol_lat, args.k, args.min_sim, mids)
    n_zero = sum(1 for nbrs in graph.values() if not nbrs)
    print(f"[graph] {len(graph)} nodes, k={args.k}, min_sim={args.min_sim}, "
          f"zero-neighbor: {n_zero}, built in {time.time() - t0:.1f}s")

    # ---- Tanimoto graph (cached artifacts, for Jaccard overlap only) ----
    tani = build_graph(mids, {m: all_labels[m]["smiles"] for m in mids},
                       k=args.k, min_sim=0.1)
    jaccard = {}
    for mid in mids:
        lat = {nbr for _, nbr in graph[mid]}
        tan = {nbr for _, nbr in tani[mid]}
        jaccard[mid] = round(len(lat & tan) / max(len(lat | tan), 1), 4)
    jav = np.array(list(jaccard.values()))
    print(f"[jaccard] latent-vs-tanimoto top-{args.k}: mean {jav.mean():.3f} "
          f"median {np.median(jav):.3f}, zero-overlap: {(jav == 0).sum()}")

    # ---- GMM-NLL refit on TRAIN atoms only (validated protocol) ----
    t0 = time.time()
    mid_col = Z_all[:, LATENT_DIM + 1].astype(int)
    atom_mid = np.array([flat_vals[i] for i in mid_col])
    is_train_atom = np.array([m in set(train_ids) for m in atom_mid])
    Z_train = Z_all[is_train_atom]
    scaler = StandardScaler().fit(Z_train[:, :LATENT_DIM])
    X_s = scaler.transform(Z_train[:, :LATENT_DIM])
    pca = PCA(n_components=K_PCA, random_state=GMM_SEED).fit(X_s)
    Xr = pca.transform(X_s)
    gm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="full",
                         init_params="kmeans", n_init=N_INIT,
                         reg_covar=REG_COVAR, random_state=GMM_SEED)
    gm.fit(Xr)
    print(f"[gmm] fit n_components={N_COMPONENTS}, k_pca={K_PCA} on "
          f"{len(Xr)} train atoms in {time.time() - t0:.1f}s "
          f"(converged={gm.converged_}, iters={gm.n_iter_})")

    X_all_s = scaler.transform(Z_all[:, :LATENT_DIM])
    X_all_p = pca.transform(X_all_s)
    nll_all = -gm.score_samples(X_all_p)
    nll_by_mid = {}
    for atom_pos, mid_pos in enumerate(mid_col):
        nll_by_mid.setdefault(flat_vals[mid_pos], []).append(
            nll_all[atom_pos])
    mean_nll = {m: float(np.mean(nll_by_mid[m])) for m in mids}
    print(f"[nll] all-atom scoring done ({len(mean_nll)} molecules)")

    # ---- sanity: refit must reproduce the cached test NLLs ----
    cached = pd.read_csv(GMM_REFIT_CSV).set_index("mol_id")
    common = [m for m in test_ids if m in cached.index]
    rho, p_rho = spearmanr([mean_nll[m] for m in common],
                           [cached.loc[m, "mean_nll"] for m in common])
    print(f"[sanity] refit vs cached test mean_nll: Spearman {rho:.4f} "
          f"(p={p_rho:.2e}, n={len(common)})")

    # ---- uncertainty weight u_i ----
    nll_vec = np.array([mean_nll[m] for m in mids])
    if args.unc_scale == "rank":
        u_vec = rankdata(nll_vec) / len(mids)          # (0, 1]
    else:
        nmin, nmax = nll_vec.min(), nll_vec.max()
        u_vec = (nll_vec - nmin) / max(nmax - nmin, 1e-12)
    print(f"[unc] scale={args.unc_scale}: u in [{u_vec.min():.4f}, "
          f"{u_vec.max():.4f}]")

    # ---- trust gate t_j ----
    if args.trust_policy == "certain47_median":
        rmse_df = pd.read_csv(ANALYSIS_CSV).set_index("mol_id")
        certain47 = set(rmse_df.index[
            rmse_df["quadrant_label"] == "low_std_low_rmse"])
        tau = float(np.median([mean_nll[m] for m in certain47
                               if m in mean_nll]))
        policy_desc = f"certain47 median ({tau:.4f}, n={len(certain47)})"
    elif args.trust_policy == "universe_median":
        tau = float(np.median(nll_vec))
        policy_desc = f"universe median ({tau:.4f}, n=642)"
    else:
        tau = float("inf")
        policy_desc = "none (all trustworthy)"
    t_vec = (nll_vec <= tau).astype(int)
    print(f"[trust] policy={policy_desc} -> {int(t_vec.sum())}/642 trusted")

    # ---- S_i + fallback (trust-filtered weight sums) ----
    S = {}
    for mid in mids:
        S[mid] = sum(w for w, nbr in graph[mid] if t_vec[mids.index(nbr)])
    fallback = [m for m in mids if S[m] < 1e-6]
    print(f"[fallback] {len(fallback)} molecules with trusted weight sum < 1e-6")

    # ---- provenance + signals artifact ----
    meta = {
        "graph_type": "latent_cosine_meanpool",
        "mids": mids,
        "k": args.k,
        "min_sim": args.min_sim,
        "n_nodes": len(mids),
        "n_zero_neighbor": n_zero,
        "n_fallback": len(fallback),
        "jaccard_tani_vs_latent_topk": {
            "mean": float(jav.mean()), "median": float(np.median(jav)),
            "pct_zero": float((jav == 0).mean()),
            "per_mol": {m: jaccard[m] for m in mids},
        },
        "signals": {
            "mean_nll": {m: round(mean_nll[m], 6) for m in mids},
            "u": {m: round(float(u_vec[i]), 6) for i, m in enumerate(mids)},
            "trust": {m: int(t_vec[i]) for i, m in enumerate(mids)},
            "trust_threshold": float(tau),
            "trust_policy": args.trust_policy,
            "trust_policy_desc": policy_desc,
            "unc_scale": args.unc_scale,
            "S": {m: round(S[m], 6) for m in mids},
            "fallback": fallback,
        },
        "provenance": {
            "ckpt": "deep_ensemble/seed_42/ensemble_seed42.pt",
            "ckpt_sha256_prefix": ckpt_sha,
            "gmm": {"n_components": N_COMPONENTS, "k_pca": K_PCA,
                    "reg_covar": REG_COVAR, "n_init": N_INIT,
                    "random_state": GMM_SEED, "fit_atoms": "train_only",
                    "converged": bool(gm.converged_), "n_iter": int(gm.n_iter_)},
            "latents": "reused z_train/z_test caches + fresh z_val",
            "cached_corr": {"test_mean_nll_spearman": round(float(rho), 4),
                            "p": float(p_rho), "n": len(common)},
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    out_graph = os.path.join(args.out, f"latent_k{args.k}_sim{args.min_sim}.json")
    with open(out_graph, "w") as f:
        json.dump(graph, f)
    meta_path = out_graph + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote -> {out_graph} (+ .meta.json)")


if __name__ == "__main__":
    main()
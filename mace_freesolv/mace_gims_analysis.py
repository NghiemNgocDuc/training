"""MACE GIMS / VW / uniform on FreeSolv fold-0 — mirrors the DimeNet GIMS audit.

Inputs: mace_freesolv/fold0_ensemble per-atom + predictions (this repo),
        frozen fold-0 split, DimeNet paper populations (same molecule sets:
        Q_std/Q_nll/UNION/all129/gradient12) + H1/H2 split for holdout.
Method: identical to aqm-spice2/.../gauge_invariant_shrinkage.py, except
  arrays come from MACE node_energy (node key verified, gate 1.4e-05).
  tau recalibrated on the MACE 102-mol val split over its own 37-pt grid.
Outputs -> mace_freesolv/fold0_ensemble/analysis/.
Usage: py -V:3.12 mace_freesolv/mace_gims_analysis.py
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACE_DIR = os.path.join(REPO, "mace_freesolv", "fold0_ensemble")
OUT = os.path.join(MACE_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)

FREESOLV = os.path.join(REPO, "aqm-spice2", "freesolv")
SPLIT_DIR = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                         "cv_results_full", "fold_0")
D_PRED = os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                      "seed_predictions_all642.csv")
D_NLL = os.path.join(FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
                     "per_molecule_gmm_nll.csv")
D_GRAD12 = os.path.join(FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
                        "gradient12_investigation", "gradient12_ungrouped.csv")
H2_SPLIT = os.path.join(FREESOLV, "node_refinement", "holdout_validation",
                        "b8_split.json")

SEEDS = [42, 123, 999]
DSEEDS3 = [42, 123, 999]  # DimeNet paper-population columns (fixed)
N_BOOT = 10_000
RNG = 20260918
POP_NAMES = ["Q_std", "Q_nll", "UNION", "all129", "gradient12"]
EPS_GAUGE = 1.0


def bootstrap(d, off):
    rng = np.random.default_rng(RNG + off)
    m = np.empty(N_BOOT)
    for b in tqdm(range(N_BOOT), desc="bootstrap", leave=False, unit="draw"):
        m[b] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi), float((m < 0).mean())


def main():
    global MACE_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=MACE_DIR,
                    help="Ensemble dir (fold0_ensemble or fold0_ensemble_off23)")
    a = ap.parse_args()
    MACE_DIR = a.base_dir
    OUT = os.path.join(MACE_DIR, "analysis")
    os.makedirs(OUT, exist_ok=True)
    global SEEDS
    found = sorted(int(os.path.basename(p).split("seed")[1].split(".pkl")[0])
                   for p in glob.glob(os.path.join(
                       MACE_DIR, "peratom_mace_seed*.pkl")))
    if found:
        SEEDS = found
    K = len(SEEDS)
    print(f"[seeds] MACE ensemble: {SEEDS} (K={K})", flush=True)
    t0 = time.time()
    nodes = pd.read_csv(os.path.join(MACE_DIR, "mace_node_contributions.csv"))
    pred = pd.read_csv(os.path.join(MACE_DIR,
                                    "mace_seed_predictions_all642.csv"))
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te
    got = set(dict.fromkeys(nodes.mol_id))
    assert got == set(all_ids), \
        f"node mols != split: missing={len(set(all_ids)-got)} extra={len(got-set(all_ids))}"

    P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS], axis=1)
    pool = nodes["mol_id"].to_numpy()

    def gsum(v):
        return pd.Series(v).groupby(pd.Series(pool)).sum().reindex(
            all_ids).to_numpy()

    def gmean(v):
        return pd.Series(v).groupby(pd.Series(pool)).mean().reindex(
            all_ids).to_numpy()

    E3 = np.stack([gsum(P3[:, q]) for q in range(K)], axis=1)
    P3pred = pred.set_index("mol_id")[[f"pred_seed{s}" for s in SEEDS]].reindex(
        all_ids).to_numpy()
    chk = float(np.abs(E3 - P3pred).max())
    print(f"[gate] max|node-sum - mol pred| = {chk:.2e}", flush=True)
    assert chk < 1e-3
    truth = pred.set_index("mol_id")["true_value"].reindex(
        all_ids).to_numpy()
    raw = P3pred.mean(axis=1)
    n_atoms = nodes.groupby("mol_id").size().reindex(all_ids).to_numpy()
    tmask = np.isin(pool, tr)
    mu_train = P3[tmask].mean(axis=0)
    print("[mu_train] " + ", ".join(f"{v:.6f}" for v in mu_train), flush=True)
    sigma2 = np.var(P3, axis=1, ddof=1)
    print(f"[spread] per-atom std: mean={np.sqrt(sigma2).mean():.5f} "
          f"max={np.sqrt(sigma2).max():.4f} kcal/mol", flush=True)

    # paper populations (DimeNet-defined molecule sets)
    dpred = pd.read_csv(D_PRED)
    t = dpred[dpred.mol_id.isin(te)].copy()
    t["mean3"] = t[[f"pred_seed{s}" for s in DSEEDS3]].mean(axis=1)
    t["std3"] = t[[f"pred_seed{s}" for s in DSEEDS3]].std(axis=1)
    nll = pd.read_csv(D_NLL)[["mol_id", "mean_nll"]]
    t = t.merge(nll, on="mol_id")
    q_std = set(t.loc[t["std3"] >= t["std3"].quantile(0.75), "mol_id"])
    q_nll = set(t.loc[t["mean_nll"] >= t["mean_nll"].quantile(0.75), "mol_id"])
    grad12 = set(pd.read_csv(D_GRAD12).mol_id)
    pops = {"Q_std": q_std, "Q_nll": q_nll, "UNION": q_std | q_nll,
            "all129": set(te), "gradient12": grad12 & set(te)}
    print({k: len(v) for k, v in pops.items()}, flush=True)
    id2i = {m: i for i, m in enumerate(all_ids)}

    def deltas(pmean, sub_pops=None):
        pops_ = sub_pops or pops
        o = {}
        for name, pop in pops_.items():
            idx = np.array([id2i[m] for m in pop], dtype=int)
            o[name] = (np.abs(pmean[idx] - truth[idx])
                       - np.abs(raw[idx] - truth[idx]))
        return o

    def val_delta(pmean, mols):
        idx = np.array([id2i[m] for m in mols], dtype=int)
        return float((np.abs(pmean[idx] - truth[idx])
                      - np.abs(raw[idx] - truth[idx])).mean())

    def lam_of(tau2):
        return np.zeros_like(sigma2) if np.isinf(tau2) else sigma2 / (sigma2 + tau2)

    def gims(lam, mu):
        Lm = gmean(lam)
        e = np.stack([(1 - Lm) * E3[:, q] + Lm * n_atoms * mu[q]
                      for q in range(K)], axis=1)
        return e.mean(axis=1), Lm

    def vw(lam, mu):
        e = np.stack([gsum(lam * mu[q] + (1 - lam) * P3[:, q])
                      for q in range(K)], axis=1)
        return e.mean(axis=1)

    def gtotal(tau2, mu):
        mv = np.var(E3, axis=1, ddof=1)
        Lm = np.zeros_like(mv) if np.isinf(tau2) else mv / (mv + tau2)
        e = np.stack([(1 - Lm) * E3[:, q] + Lm * n_atoms * mu[q]
                      for q in range(K)], axis=1)
        return e.mean(axis=1), Lm

    V = float(np.var(sigma2))
    grid = np.geomspace(1e-8 * V, 100 * V, 37).tolist() + [np.inf]
    print(f"[grid] var(sigma2)={V:.3e} range {grid[0]:.3e}..{grid[-2]:.3e}",
          flush=True)

    def calib(kind):
        best, bd = None, 1e9
        curve = []
        for i, tau2 in enumerate(tqdm(grid, desc=f"calib {kind}", unit="tau")):
            if kind == "gims":
                pm, _ = gims(lam_of(tau2), mu_train)
            elif kind == "vw":
                pm = vw(lam_of(tau2), mu_train)
            else:
                pm, _ = gtotal(tau2, mu_train)
            vd = val_delta(pm, va)
            curve.append(vd)
            if vd < bd:
                bd, best = vd, (i, tau2)
        return best, bd, curve

    (gi, gtau), gvd, _ = calib("gims")
    (vi, vtau), vvd, _ = calib("vw")
    (ti, ttau), tvd, _ = calib("gims_total")
    print(f"[calib] gims tau={gtau:.3e} val={gvd:+.4f} | vw tau={vtau:.3e} "
          f"val={vvd:+.4f} | total tau={ttau:.3e} val={tvd:+.4f}", flush=True)

    pm_g, Lm_g = gims(lam_of(gtau), mu_train)
    pm_v = vw(lam_of(vtau), mu_train)
    pm_t, _ = gtotal(ttau, mu_train)
    tr_idx = np.array([id2i[m] for m in tr], dtype=int)
    ubar = float(Lm_g[tr_idx].mean())
    pm_u, _ = gims(np.full_like(sigma2, ubar), mu_train)
    print(f"[uniform] matched Lambda={ubar:.6f}", flush=True)

    arms = {"gims": pm_g, "vw": pm_v, "gims_total": pm_t, "uniform": pm_u}
    rows = []
    D = {a: deltas(pm) for a, pm in arms.items()}
    base = {"gims": 6100, "vw": 6200, "gims_total": 6300, "uniform": 6400}
    for a, pm in arms.items():
        idx_all = np.array([id2i[m] for m in all_ids])
        _ = idx_all
        for pi, name in enumerate(POP_NAMES):
            d = D[a][name]
            idx = np.array([id2i[m] for m in pops[name]], dtype=int)
            before = float(np.abs(raw[idx] - truth[idx]).mean())
            lo, hi, pw = bootstrap(d, base[a] + pi)
            rows.append({"stage": "test", "arm": a, "population": name,
                         "n": len(d), "delta_mae": float(d.mean()),
                         "before_mae": before,
                         "after_mae": before + float(d.mean()),
                         "ci_lo": lo, "ci_hi": hi, "p_improves": pw})
    for pair, (L, Rr) in {"gims_minus_vw": ("gims", "vw"),
                          "gims_minus_uniform": ("gims", "uniform"),
                          "vw_minus_uniform": ("vw", "uniform")}.items():
        for pi, name in enumerate(POP_NAMES):
            diff = D[L][name] - D[Rr][name]
            lo, hi, pw = bootstrap(diff, 6600 + pi)
            rows.append({"stage": "test", "arm": pair, "population": name,
                         "n": len(diff), "delta_mae": float(diff.mean()),
                         "ci_lo": lo, "ci_hi": hi, "p_left_better": pw})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "mace_gims_results.csv"), index=False)
    piv = res[res.stage == "test"].pivot(index="arm", columns="population",
                                         values="delta_mae")
    print(piv.round(3).to_string(), flush=True)

    # gauge stress (same construction as DimeNet audit)
    lam = lam_of(gtau)
    Lm = gmean(lam)
    g = np.zeros(len(lam))
    for m in all_ids:
        ix = np.flatnonzero(pool == m)
        c = lam[ix] - Lm[id2i[m]]
        rms = float(np.sqrt(np.mean(c ** 2)))
        if rms > 1e-14:
            g[ix] = EPS_GAUGE * c / rms
            g[ix[-1]] -= g[ix].sum()
    Pp = P3 + g[:, None]
    mu_p = Pp[tmask].mean(axis=0)

    def gsum_p(v):
        return pd.Series(v).groupby(pd.Series(pool)).sum().reindex(
            all_ids).to_numpy()

    def gmean_p(v):
        return pd.Series(v).groupby(pd.Series(pool)).mean().reindex(
            all_ids).to_numpy()

    Ep = np.stack([gsum_p(Pp[:, q]) for q in range(K)], axis=1)
    Lmp = gmean_p(lam)
    gims_p = ((1 - Lmp) * Ep.mean(axis=1)
              + Lmp * n_atoms * float(mu_p.mean()))
    gims_0 = ((1 - Lm) * E3.mean(axis=1)
              + Lm * n_atoms * float(mu_train.mean()))
    vw_p = np.stack([gsum_p(lam * mu_p[q] + (1 - lam) * Pp[:, q])
                     for q in range(K)], axis=1).mean(axis=1)
    vw_0 = np.stack([gsum(lam * mu_train[q] + (1 - lam) * P3[:, q])
                     for q in range(K)], axis=1).mean(axis=1)
    te_mask = np.isin(all_ids, te)
    print(f"[gauge] GIMS max shift={np.abs(gims_p-gims_0).max():.2e} | "
          f"VW mean|shift| test={np.abs(vw_p-vw_0)[te_mask].mean():.4f} "
          f"per {EPS_GAUGE:.1f} kcal/mol atom-RMS", flush=True)
    rep = {"tau_gims": float(gtau), "tau_vw": float(vtau),
           "tau_total": float(ttau), "uniform_lambda": ubar,
           "mu_train": mu_train.tolist(),
           "gauge_gims_max": float(np.abs(gims_p - gims_0).max()),
           "gauge_vw_mean_test": float(np.abs(vw_p - vw_0)[te_mask].mean()),
           "rows": res.to_dict("records"),
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "mace_gims_report.json"), "w"),
              indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

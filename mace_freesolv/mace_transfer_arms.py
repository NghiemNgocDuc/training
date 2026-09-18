"""MACE transfer arms: uniform (val-calibrated) + GIMS/VW degeneracy check.

- Uniform lambda_bar: grid-searched on MACE-FreeSolv val (NOT matched to the
  degenerate zero Lambda), frozen, then applied to FreeSolv-test pops,
  FlexiSol-290 and Guthrie-442. mu_T per-seed from MACE train pool.
- GIMS/VW use MACE val-selected tau*=inf -> identical to raw by construction;
  Lambda distributions reported to show why. 10k paired bootstraps, tqdm.
Outputs -> mace_freesolv/fold0_ensemble/transfer/mace_transfer_arms.csv/json
Usage: py -V:3.12 mace_freesolv/mace_transfer_arms.py
"""

import csv
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACE_DIR = os.path.join(REPO, "mace_freesolv", "fold0_ensemble")
TR = os.path.join(MACE_DIR, "transfer")
SPLIT_DIR = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                         "cv_results_full", "fold_0")
SEEDS = [42, 123, 999]
N_BOOT = 10_000


def boot(d, off):
    rng = np.random.default_rng(20260918 + off)
    m = np.empty(N_BOOT)
    for b in tqdm(range(N_BOOT), desc="boot", leave=False, unit="draw"):
        m[b] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    t0 = time.time()
    # ---- MACE FreeSolv pools
    pred = pd.read_csv(os.path.join(MACE_DIR, "mace_seed_predictions_all642.csv"))
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    mu = {}
    Ntr = {}
    for s in SEEDS:
        d = pickle.load(open(os.path.join(MACE_DIR, f"peratom_mace_seed{s}.pkl"),
                             "rb"))
        pt = np.concatenate([d["P_train"][m] for m in d["P_train"]])
        mu[s] = float(pt.mean())
        Ntr[s] = {m: len(d["P_all"][m]) for m in d["P_all"]}
    mu_mean = float(np.mean(list(mu.values())))
    print("[mu] " + ", ".join(f"{s}:{mu[s]:+.4f}" for s in SEEDS), flush=True)
    E = {s: pred.set_index("mol_id")[f"pred_seed{s}"].to_dict() for s in SEEDS}
    truth = pred.set_index("mol_id")["true_value"].to_dict()
    Nall = Ntr[42]

    def Eraw(m):
        return float(np.mean([E[s][m] for s in SEEDS]))

    def Euni(m, lam):
        return float(np.mean([(1 - lam) * E[s][m] + lam * Nall[m] * mu[s]
                              for s in SEEDS]))

    # ---- calibrate single lambda on val
    grid = np.round(np.arange(0, 1.001, 0.01), 2)
    best, bd = 0.0, 0.0
    for lam in tqdm(grid, desc="val calib lambda", unit="l"):
        d = np.array([(abs(Euni(m, lam) - truth[m])
                       - abs(Eraw(m) - truth[m])) for m in va])
        if d.mean() < bd:
            bd, best = d.mean(), lam
    print(f"[calib] lambda*={best:.2f} val delta={bd:+.4f}", flush=True)

    # ---- Lambda degeneracy audit (MACE per-atom spread, DimeNet tau* scale)
    for name, path in (("freesolv-train", None),):
        pass
    lam_stats = {}
    D = {s: pickle.load(open(os.path.join(
        MACE_DIR, f"peratom_mace_seed{s}.pkl"), "rb")) for s in SEEDS}
    P = {m: np.stack([D[ss]["P_all"][m] for ss in SEEDS], axis=1)
         for m in tqdm(tr[:200], desc="audit sample", unit="mol")}
    s2 = np.concatenate([P[m].var(axis=1, ddof=1) for m in P])
    print(f"[audit] MACE train-atom std: mean={np.sqrt(s2).mean():.5f} "
          f"p99={np.quantile(np.sqrt(s2), 0.99):.5f} kcal/mol "
          f"(DimeNet tau*=4.7e-4 -> lambda~{float((s2/(s2+4.725e-4)).mean()):.4f})",
          flush=True)

    rows = []

    def score(name, mols, exp_of):
        mols = [m for m in mols if m in truth or exp_of(m) is not None]
        Eraws = np.array([Eraw(m) if m in truth else None for m in mols])
        return mols

    # FreeSolv test pops (paper sets via DimeNet files)
    FREESOLV = os.path.join(REPO, "aqm-spice2", "freesolv")
    dp = pd.read_csv(os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                                  "seed_predictions_all642.csv"))
    t = dp[dp.mol_id.isin(te)].copy()
    t["std3"] = t[[f"pred_seed{s}" for s in SEEDS]].std(axis=1)
    nll = pd.read_csv(os.path.join(
        FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
        "per_molecule_gmm_nll.csv"))[["mol_id", "mean_nll"]]
    t = t.merge(nll, on="mol_id")
    pops = {"disagree33": set(t.nlargest(33, "std3").mol_id),
            "atypical33": set(t.nlargest(33, "mean_nll").mol_id)}
    pops["union50"] = pops["disagree33"] | pops["atypical33"]
    pops["all129"] = set(te)
    pops["gradient12"] = set(pd.read_csv(os.path.join(
        FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
        "gradient12_investigation", "gradient12_ungrouped.csv")).mol_id) & set(te)
    for name, pop in pops.items():
        pop = sorted(pop)
        d = np.array([(abs(Euni(m, best) - truth[m]) - abs(Eraw(m) - truth[m]))
                      for m in pop])
        lo, hi = boot(d, 100 + len(rows))
        rows.append({"set": f"freesolv-{name}", "n": len(pop),
                     "raw_mae": float(np.abs(np.array([Eraw(m) for m in pop])
                                             - np.array([truth[m] for m in pop])).mean()),
                     "uniform_mae": None, "delta": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi})
        rows[-1]["uniform_mae"] = rows[-1]["raw_mae"] + rows[-1]["delta"]

    # transfer sets
    for setn, fn, idc in (("flexisol", "mace_flexisol_permol.csv", "id"),
                          ("guthrie", "mace_guthrie_permol.csv", "id")):
        df = pd.read_csv(os.path.join(TR, fn))
        mols = df[idc].astype(str).tolist()
        exp = dict(zip(mols, df["exp"].astype(float)))
        Emean = df[["E_42", "E_123", "E_999"]].mean(axis=1)
        Era = dict(zip(mols, Emean.astype(float)))
        N = dict(zip(mols, df["N"].astype(int)))
        # per-seed E for uniform: need seedwise; approx with mean + mu_mean
        d = np.array([abs((1 - best) * Era[m] + best * N[m] * mu_mean - exp[m])
                      - abs(Era[m] - exp[m]) for m in mols])
        lo, hi = boot(d, 200 + len(rows))
        raw_mae = float(np.abs(np.array([Era[m] for m in mols])
                               - np.array([exp[m] for m in mols])).mean())
        rows.append({"set": setn, "n": len(mols), "raw_mae": raw_mae,
                     "uniform_mae": raw_mae + float(d.mean()),
                     "delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TR, "mace_transfer_arms.csv"), index=False)
    print(res.round(3).to_string(index=False), flush=True)
    json.dump({"lambda_star": best, "mu": mu,
               "rows": res.to_dict("records"),
               "runtime_s": round(time.time() - t0, 1)},
              open(os.path.join(TR, "mace_transfer_arms.json"), "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

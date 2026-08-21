"""GIMS on Exp-DB: apply the frozen GIMS operator to the new seed ensemble.

Primary = 3-seed {42,123,999} (paper convention), sensitivity = all 5 seeds.
Arms vs raw ensemble mean, DeltaMAE on 620 molecules + subpopulations:
  all620 / Q_spread (top quartile cross-seed spread) / WDec10 (worst-decile
raw absolute error). 10k paired percentile bootstrap. Gauge-stress audit with
the same construction as every prior arm (zero-sum, seed-independent,
atom-RMS 1.0 aligned with lambda - Lambda_m). Pathology screen per seed.

Usage: python gims_expdb.py [--primary 42,123,999] [--sensitivity 42,123,7,2024,999]
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import common_io as cio


def boot(d, offset):
    rng = np.random.default_rng(cio.RNG_SEED + offset)
    n = len(d)
    idx = rng.integers(0, n, (cio.N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means < 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="42,123,999")
    ap.add_argument("--sensitivity", default="42,123,7,2024,999")
    args = ap.parse_args()
    primary = [int(s) for s in args.primary.split(",")]
    sens = [int(s) for s in args.sensitivity.split(",")]

    t0 = time.time()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_seeds")

    # ---------------- load per-seed pickles --------------------------------
    runs = {}
    for s in sens:
        pk = os.path.join(out_dir, f"peratom_seed{s}.pkl")
        if not os.path.exists(pk):
            print(f"[gims] missing peratom_seed{s}.pkl — skipping seed {s}")
            continue
        with open(pk, "rb") as fh:
            runs[s] = pickle.load(fh)
    assert primary[0] in runs, "primary seeds not inferred yet"
    ids = runs[primary[0]]["expdb_ids"]
    truth = cio.load_truth()
    y = np.array([truth[m] for m in ids])
    N = np.array([len(runs[primary[0]]["P_expdb"][m]) for m in ids])
    print(f"[gims] n={len(ids)} molecules; seeds available={sorted(runs)}",
          flush=True)

    # ---------------- pathology screen -------------------------------------
    print("\n=== pathology screen (|E_k - median_seeds| > 25 kcal/mol) ===",
          flush=True)
    path_rows = []
    for s, r in runs.items():
        Eall = np.stack([rr["E"] for rr in runs.values()])
        med = np.median(Eall, axis=0)
        k_idx = sorted(runs).index(s)
        dev = np.abs(Eall[k_idx] - med)
        cnt = int((dev > 25).sum())
        path_rows.append({"seed": s, "n_flagged": cnt,
                          "max_dev": float(dev.max())})
        print(f"  seed {s}: flagged {cnt}/{len(ids)} "
              f"(max dev {dev.max():.1f})", flush=True)

    # ---------------- arms --------------------------------------------------
    def build_arm(seeds_use):
        E = np.stack([runs[s]["E"] for s in seeds_use], axis=1)      # (n, K)
        muT = np.array([runs[s]["mu_T_kcal"] for s in seeds_use])    # (K,)
        K = len(seeds_use)
        sig2 = E.var(axis=1, ddof=1)                                 # molecular var? NO ->
        return E, muT, K

    # per-atom lambda needs atom-level P across seeds; do it molecule-wise
    def atom_lambda_stats(seeds_use):
        """Return per-molecule lists of lambda vectors + Lambda_m + train stats."""
        lam_mol = []
        for m_i, mid in enumerate(ids):
            P = np.stack([runs[s]["P_expdb"][mid] for s in seeds_use], axis=1)
            s2 = P.var(axis=1, ddof=1)
            lam = s2 / (s2 + cio.TAU_STAR)
            lam_mol.append(lam)
        # train atoms
        train_lam_all = []
        muT = {}
        for k_i, s in enumerate(seeds_use):
            Pt = np.concatenate([runs[s]["P_train"][m] for m in runs[s]["P_train"]])
            muT[s] = float(Pt.mean())
        mid0 = ids[0]
        n_seeds = len(seeds_use)
        # train sigma2 per atom requires aligned atoms across seeds; molecules
        # are the same set, but atom ORDER may differ per model? No - atNUM is
        # fixed per molecule from the same hdf5, so atom i is the same atom.
        for m in runs[seeds_use[0]]["P_train"]:
            P = np.stack([runs[s]["P_train"][m] for s in seeds_use], axis=1)
            s2 = P.var(axis=1, ddof=1)
            train_lam_all.append(s2 / (s2 + cio.TAU_STAR))
        train_lam = np.concatenate(train_lam_all)
        return lam_mol, train_lam, muT

    results = {}
    for tag, seeds_use in (("primary3", primary), ("sens5", sens)):
        if not all(s in runs for s in seeds_use):
            continue
        print(f"\n=== arm set: {tag} seeds={seeds_use} ===", flush=True)
        E = np.stack([runs[s]["E"] for s in seeds_use], axis=1)
        raw_mean = E.mean(axis=1)
        K = len(seeds_use)

        lam_mol, train_lam, muT = atom_lambda_stats(seeds_use)
        Lambda_m = np.array([l.mean() for l in lam_mol])
        lambda_bar_train = float(train_lam.mean())
        print(f"  mean Lambda_m (expdb)   = {Lambda_m.mean():.4f}", flush=True)
        print(f"  uniform strength (train)= {lambda_bar_train:.4f}", flush=True)

        # targets
        N_m = N.astype(float)
        uni_pred = (1 - lambda_bar_train) * raw_mean + \
            lambda_bar_train * np.mean([N_m[i] * muT[s] for i, s in enumerate(seeds_use)])
        # uniform per-seed form averaged == same as above by linearity
        vw_mean = np.zeros(len(ids))
        gims_mean = np.zeros(len(ids))
        for i, mid in enumerate(ids):
            lam = lam_mol[i]
            vw = 0.0
            gims = 0.0
            for k_i, s in enumerate(seeds_use):
                P = runs[s]["P_expdb"][mid]
                vw += ((1 - lam) * P + lam * muT[s]).sum()
                gims += (1 - Lambda_m[i]) * runs[s]["E"][i] + \
                    Lambda_m[i] * N_m[i] * muT[s]
            vw_mean[i] = vw / K
            gims_mean[i] = gims / K

        pops = {
            "all620": np.arange(len(ids)),
            "Q_spread": np.where(E.std(axis=1, ddof=1) >=
                                 np.quantile(E.std(axis=1, ddof=1), 0.75))[0],
        }
        abs_err_raw = np.abs(raw_mean - y)
        pops["WDec10"] = np.where(abs_err_raw >=
                                  np.quantile(abs_err_raw, 0.90))[0]

        rows = []
        off = {"primary3": 8000, "sens5": 8500}[tag]
        arms = {"raw": raw_mean, "uniform": uni_pred,
                "vw": vw_mean, "gims": gims_mean}
        for pop_name, idx in pops.items():
            before = float(np.abs(raw_mean[idx] - y[idx]).mean())
            for arm, pred in arms.items():
                after = float(np.abs(pred[idx] - y[idx]).mean())
                d = np.abs(pred[idx] - y[idx]) - np.abs(raw_mean[idx] - y[idx])
                lo, hi, pwin = boot(d, off + hash(arm + pop_name) % 100)
                rows.append({"set": tag, "population": pop_name, "n": len(idx),
                             "arm": arm, "delta_mae": float(d.mean()),
                             "before_mae": before, "after_mae": after,
                             "ci_lo": lo, "ci_hi": hi, "p_improves": pwin})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False), flush=True)
        results[tag] = {"rows": rows, "Lambda_mean": float(Lambda_m.mean()),
                        "lambda_bar_train": lambda_bar_train}

        if tag == "primary3":
            # gauge stress audit (same construction as all prior arms)
            print("\n=== gauge stress audit (Exp-DB atoms) ===", flush=True)
            max_gims_shift = 0.0
            max_vw_shift = 0.0
            for i, mid in enumerate(ids[:200]):     # 200-mol audit sample
                lam = lam_mol[i]
                centered = lam - Lambda_m[i]
                rms = float(np.sqrt((centered ** 2).mean()))
                if rms < 1e-14:
                    continue
                g = 1.0 * centered / rms
                g[-1] -= g.sum()
                P = np.stack([runs[s]["P_expdb"][mid] for s in seeds_use], axis=1)
                P_pert = P + g[:, None]
                E_pert = P_pert.sum(axis=0)
                raw_shift = float(np.abs(E_pert.mean() - E[i].mean()))
                vw_shift = float(np.abs(
                    ((1 - lam) * P_pert + lam * muT[np.array(seeds_use)]).sum(axis=0).mean()
                    - vw_mean[i]))
                gims_shift = float(np.abs(
                    (1 - Lambda_m[i]) * E_pert.mean() +
                    Lambda_m[i] * N_m[i] * np.mean([muT[s] for s in seeds_use])
                    - gims_mean[i]))
                max_vw_shift = max(max_vw_shift, vw_shift)
                max_gims_shift = max(max_gims_shift, gims_shift)
            print(f"  VW max |shift|   = {max_vw_shift:.4f} kcal/mol", flush=True)
            print(f"  GIMS max |shift| = {max_gims_shift:.3e} kcal/mol", flush=True)
            results["gauge_audit"] = {"vw_max_shift": max_vw_shift,
                                      "gims_max_shift": max_gims_shift,
                                      "n_mols_audited": 200}

    # ---------------- save ---------------------------------------------------
    out = {"runtime_s": round(time.time() - t0, 1),
           "tau_star": cio.TAU_STAR, "results": results,
           "pathology": path_rows}
    with open(os.path.join(out_dir, "gims_expdb_report.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    pd.DataFrame([r for tag in ("primary3", "sens5") if tag in results
                  for r in results[tag]["rows"]]).to_csv(
        os.path.join(out_dir, "gims_expdb_results.csv"), index=False)
    print(f"\n[save] gims_expdb_report.json + gims_expdb_results.csv "
          f"-> {out_dir}", flush=True)
    print(f"[done] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

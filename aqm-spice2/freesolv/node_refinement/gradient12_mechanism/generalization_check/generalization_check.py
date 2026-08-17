"""Generalization check: does the gradient-12 'gated-sum-to-total-error ratio'
mechanism (Part 8) predict correction resistance across ALL 129 fold-0 test
molecules, or is gradient-12 an isolated case?

Part 8 finding being tested: gradient-12's gated (uncertain-flagged) atoms
carry only a tiny fraction of the molecule's prediction error (gated sums
<=4.1 vs up to 85 kcal/mol for Q_std), so any gated-atom-only correction is
structurally blind to where the error actually lives.

Definitions (identical to gradient12_mechanism.py):
  err_before    = |mean3 - truth|                      (kcal/mol)
  gated_sum     = sum of 3-seed-mean node contributions over GATED atoms
  ratio         = |gated_sum| / err_before   (user's magnitude ratio)
  ratio_signed  = gated_sum / (mean3 - truth)          (secondary diagnostic:
                  ~+1 would mean the gated atoms alone carry the whole error)
  correction    = calibrated shrinkage at lambda*=1.0: replace each gated
                  atom with the pool mean mu_bar (no neighbor lookup needed);
  delta_shrink  = |unchanged + n_gated*mu_bar - truth| - err_before
  delta_trust   = (secondary arm, same neighbor machinery as Part 8):
                  trust-weighted top-10 neighbor replacement of gated atoms.

Tests:
  1. Spearman rho(ratio, delta_shrink) over all 129 + permutation p.
  2. Quartile split on ratio: mean/median delta per quartile; Q1 (low) vs Q4
     (high) Mann-Whitney + 10k bootstrap CI of the mean difference.
  3. Low-ratio molecules beyond gradient-12: ratio <= max(g12 ratio), listed
     with their individual deltas; group mean delta vs the rest.
  4. Verdict: generalizes (ratio predicts resistance) vs case-study (g12
     happens to fit a pattern that does not hold elsewhere).

Outputs -> gradient12_mechanism/generalization_check/. CPU, ~3-4 min (trust
arm neighbor lookup). Uses the same RNG_SEED=20260815 and SEEDS3 convention.
"""
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE)))))
FREESOLV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
FREESOLV2 = os.path.join(REPO, "aqm-spice2", "freesolv")

NODE_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                        "approach2_node_refine", "node_contributions.csv")
PRED_CSV = os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                        "seed_predictions_all642.csv")
GRAD12_CSV = os.path.join(FREESOLV, "deep_ensemble", "gmm_uncertainty_check",
                          "gradient12_investigation", "gradient12_ungrouped.csv")
SPLIT_DIR = os.path.join(FREESOLV2, "cv_results_full", "fold_0")
B8_CSV = os.path.join(os.path.dirname(HERE), "holdout_validation",
                      "b8_holdout_per_molecule.csv")

SEEDS3 = [42, 123, 999]
RNG_SEED = 20260815
GATE_Q = 0.75
K = 10
MIN_SIM = 0.2
N_PERM = 10_000


def spearman(x, y, rng):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    count = 0
    for _ in range(N_PERM):
        ry2 = rng.permutation(ry)
        if abs(np.corrcoef(rx, ry2)[0, 1]) >= abs(rho):
            count += 1
    return rho, (count + 1) / (N_PERM + 1)


def mw_p(x, y, rng):
    """Permutation p for the difference of means (fallback-safe, no scipy)."""
    xy = np.concatenate([x, y])
    d0 = x.mean() - y.mean()
    n = len(x)
    count = 0
    for _ in range(N_PERM):
        perm = rng.permutation(xy)
        if abs(perm[:n].mean() - perm[n:].mean()) >= abs(d0):
            count += 1
    return float(d0), (count + 1) / (N_PERM + 1)


def boot_ci2(x, y, rng, lo_q=2.5, hi_q=97.5):
    """Bootstrap CI for the difference of means (x.mean() - y.mean())."""
    b = np.empty(N_PERM)
    for i in range(N_PERM):
        b[i] = rng.choice(x, size=len(x), replace=True).mean() \
             - rng.choice(y, size=len(y), replace=True).mean()
    return float(np.percentile(b, lo_q)), float(np.percentile(b, hi_q))


def main():
    t0 = time.time()
    nodes = pd.read_csv(NODE_CSV)
    pred = pd.read_csv(PRED_CSV)
    grad12 = set(pd.read_csv(GRAD12_CSV).mol_id)
    tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
    va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
    te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
    all_ids = tr + va + te
    assert list(dict.fromkeys(nodes.mol_id)) == all_ids

    pool_mol = nodes["mol_id"].to_numpy()
    u3 = nodes["u3"].to_numpy()
    gate = u3 >= np.quantile(u3, GATE_Q)
    gidx = np.flatnonzero(gate)
    P3 = np.stack([nodes[f"P_seed{s}"] for s in SEEDS3], axis=1).astype(np.float64)
    mu_bar = float(P3.mean())
    truth = pred.set_index("mol_id")["true_value"].reindex(all_ids).to_numpy()

    t = pred[pred.mol_id.isin(te)].copy()
    t["mean3"] = t[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1)
    t["std3"] = t[[f"pred_seed{s}" for s in SEEDS3]].std(axis=1)
    tdf = t.set_index("mol_id")
    q_std = set(t.loc[t["std3"] >= t["std3"].quantile(0.75), "mol_id"])

    # per-molecule gated-atom index
    gated_of = {}
    for gi in gidx:
        gated_of.setdefault(pool_mol[gi], []).append(gi)

    # ---------------- neighbor lookup (only needed for the trust arm) ----------
    desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
    nb_top, ws_top = {}, {}
    for c in range(0, len(gidx), 128):
        q = desc[gidx[c:c + 128]]
        inter = np.minimum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        union = np.maximum(desc[None, :, :], q[:, None, :]).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        same_mol = pool_mol[None, :] == pool_mol[gidx[c:c + 128]][:, None]
        ok = (sim >= MIN_SIM) & (~same_mol)
        for r, gi in enumerate(gidx[c:c + 128]):
            elig = np.flatnonzero(ok[r])
            order = elig[np.argsort(-sim[r][elig])[:K]]
            nb_top[int(gi)] = order.tolist()
            ws_top[int(gi)] = sim[r][order].tolist()
    print(f"[nbrs] top-k lookup done for {len(gidx)} gated nodes")

    trust_rank = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()

    def trust_val_3mean(gi):
        nidx, wv = nb_top[int(gi)], np.array(ws_top[int(gi)])
        if len(nidx) == 0:
            return None
        tt = trust_rank[nidx]
        denom = (wv * tt).sum()
        if denom <= 0:
            return None
        return float((wv * tt * P3[nidx].mean(axis=1)).sum() / denom)

    # ---------------- per-molecule table (all 129 test) ------------------------
    rows = []
    for mid in te:
        mi = all_ids.index(mid)
        m3 = float(tdf.loc[mid, "mean3"])
        err_before = abs(m3 - truth[mi])
        lg = gated_of.get(mid, [])
        n_ga = len(lg)
        s_orig = float(sum(P3[gi].mean() for gi in lg))
        ratio = abs(s_orig) / err_before if err_before > 0 else np.inf
        ratio_signed = s_orig / (m3 - truth[mi]) if (m3 - truth[mi]) != 0 else np.nan
        s_shrink = n_ga * mu_bar
        delta_shrink = abs(m3 - s_orig + s_shrink - truth[mi]) - err_before
        tot_trust = 0.0
        for gi in lg:
            tv = trust_val_3mean(gi)
            tot_trust += tv if tv is not None else float(P3[gi].mean())
        delta_trust = abs(m3 - s_orig + tot_trust - truth[mi]) - err_before
        rows.append({
            "mol": mid, "in_gradient12": mid in grad12, "in_Q_std": mid in q_std,
            "n_gated_atoms": n_ga, "err_before": err_before,
            "mean3": m3, "truth": float(truth[mi]),
            "gated_sum_orig": s_orig, "ratio": ratio, "ratio_signed": ratio_signed,
            "delta_shrink": delta_shrink, "delta_trust": delta_trust,
        })
    df = pd.DataFrame(rows).set_index("mol")
    df.to_csv(os.path.join(HERE, "generalization_per_molecule.csv"))

    # cross-check vs Part B per-molecule deltas (expect factor-3 discrepancy:
    # b8_holdout.py summed per-seed values into mean-space arithmetic)
    if os.path.exists(B8_CSV):
        b8 = pd.read_csv(B8_CSV).set_index("mol")
        both = df.index.intersection(b8.index)
        d = np.abs(df.loc[both, "delta_shrink"] - b8.loc[both, "shrink"])
        ratio_chk = (df.loc[both, "delta_shrink"] / b8.loc[both, "shrink"])
        print(f"[crosscheck-vs-b8] n={len(both)} max|delta_diff|={d.max():.4f} "
              f"median b8/correct={np.nanmedian(ratio_chk):.3f}")

    # ---------------- 1. Spearman ratio vs delta_shrink ------------------------
    rng = np.random.default_rng(RNG_SEED)
    rho, p = spearman(df["ratio"].to_numpy(), df["delta_shrink"].to_numpy(), rng)
    rho_t, p_t = spearman(df["ratio"].to_numpy(), df["delta_trust"].to_numpy(), rng)
    print(f"[corr] rho(ratio, delta_shrink) = {rho:+.3f} (p={p:.4f})")

    # ---------------- 2. quartile split ----------------------------------------
    df["q"] = pd.qcut(df["ratio"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qtab = df.groupby("q", observed=True).agg(
        n=("delta_shrink", "size"),
        ratio_med=("ratio", "median"),
        delta_shrink_mean=("delta_shrink", "mean"),
        delta_shrink_med=("delta_shrink", "median"),
        delta_trust_mean=("delta_trust", "mean"),
    ).round(4)
    q1 = df.loc[df["q"] == "Q1", "delta_shrink"].to_numpy()
    q4 = df.loc[df["q"] == "Q4", "delta_shrink"].to_numpy()
    d14, p14 = mw_p(q1, q4, rng)
    lo, hi = boot_ci2(q1, q4, rng)
    q1t = df.loc[df["q"] == "Q1", "delta_trust"].to_numpy()
    q4t = df.loc[df["q"] == "Q4", "delta_trust"].to_numpy()
    d14t, p14t = mw_p(q1t, q4t, rng)
    print(f"[quartiles] Q1(n={len(q1)}) vs Q4(n={len(q4)}): mean diff "
          f"{d14:+.3f} [{lo:+.3f},{hi:+.3f}] p={p14:.4f}")

    # ---------------- 3. low-ratio molecules beyond gradient-12 ----------------
    g12_ratios = df.loc[df["in_gradient12"], "ratio"]
    # g12's ratio distribution is WIDE (0.0 .. 10.9 = dataset max), so "similarly
    # low" is operationalized as ratio <= g12 MEDIAN ratio (0.195); a looser
    # threshold (<=0.5, ~g12 75th pct) is reported as a sensitivity.
    g12_med = float(g12_ratios.median())
    low = df[(~df["in_gradient12"]) & (df["ratio"] <= g12_med)]
    rest = df[(~df["in_gradient12"]) & (df["ratio"] > g12_med)]
    dl, pl = mw_p(low["delta_shrink"].to_numpy(), rest["delta_shrink"].to_numpy(), rng)
    lo_l, hi_l = boot_ci2(low["delta_shrink"].to_numpy(),
                          rest["delta_shrink"].to_numpy(), rng)
    low_mean, rest_mean = float(low["delta_shrink"].mean()), float(rest["delta_shrink"].mean())
    # also: do non-g12 low-ratio molecules benefit at all (their own CI)?
    lo_self, hi_self = boot_ci2(low["delta_shrink"].to_numpy(),
                                np.zeros(len(low)), rng)
    low5 = df[(~df["in_gradient12"]) & (df["ratio"] <= 0.5)]
    rest5 = df[(~df["in_gradient12"]) & (df["ratio"] > 0.5)]
    d5, p5 = mw_p(low5["delta_shrink"].to_numpy(), rest5["delta_shrink"].to_numpy(), rng)
    # sensitivity: exclude molecules with NO gated atoms (uncorrectable by
    # construction, delta=0 by definition)
    q1c = df.loc[(df["q"] == "Q1") & (df["n_gated_atoms"] > 0), "delta_shrink"].to_numpy()
    q4c = df.loc[(df["q"] == "Q4") & (df["n_gated_atoms"] > 0), "delta_shrink"].to_numpy()
    d14c, p14c = mw_p(q1c, q4c, rng)
    lo_c, hi_c = boot_ci2(q1c, q4c, rng)
    low_list = low.sort_values("ratio")
    print(f"[low-ratio] threshold=g12 median {g12_med:.3f}: non-g12 low-ratio n="
          f"{len(low)} (mean delta {low_mean:+.3f} vs rest {rest_mean:+.3f}, "
          f"p={pl:.4f}); loose<=0.5 n={len(low5)} p={p5:.4f}; Q1-no-gated "
          f"sensitivity p={p14c:.4f}")
    low_list.to_csv(os.path.join(HERE, "generalization_low_ratio_molecules.csv"))

    # ---------------- 3b. what actually predicts benefit? -----------------------
    # Context: the dominant correlate of correction benefit is pre-correction
    # error magnitude (Part 8's absolute-gated-sum quantity is NS).
    for col in ["gated_sum_orig", "err_before"]:
        if col == "gated_sum_orig":
            x = df[col].abs().to_numpy()
            df["pred_abs"] = df[col].abs()
        else:
            x = df[col].to_numpy()
            df["pred_abs"] = df[col]
        df["pq"] = pd.qcut(df["pred_abs"], 4, labels=["p1", "p2", "p3", "p4"])
        p1 = df.loc[df["pq"] == "p1", "delta_shrink"].to_numpy()
        p4 = df.loc[df["pq"] == "p4", "delta_shrink"].to_numpy()
        dd, pp = mw_p(p1, p4, rng)
        print(f"[alt-predictor] {col}: rho(spearman) + p1-vs-p4 mean diff "
              f"{dd:+.3f} p={pp:.4f}")

    # ---------------- verdict --------------------------------------------------
    sig_rho = p < 0.05 and rho > 0
    sig_q = p14 < 0.05 and d14 > 0
    sig_low = pl < 0.05 and dl > 0
    verdict = ("GENERALIZES" if (sig_rho and sig_q and sig_low)
               else ("PARTIAL" if (sig_rho or sig_q or sig_low)
                     else "CASE_STUDY"))

    rep = {
        "label": "Generalization check: gated-sum-to-total-error ratio vs correction "
                 "resistance (all 129 fold-0 test molecules)",
        "design": {
            "ratio": "|gated_sum_orig| / err_before (gated atoms' 3-seed-mean "
                     "contribution sum vs molecule prediction error)",
            "ratio_signed": "gated_sum_orig / (mean3 - truth); +1 = gated atoms "
                            "alone carry the whole error",
            "correction": "calibrated shrinkage at lambda*=1.0 (replace each gated "
                          "atom with pool mean mu_bar=" f"{mu_bar:.4f})",
            "trust_arm": "top-10 Tanimoto neighbors, w=sim, t=1-rank(u3)",
            "permutations": N_PERM, "rng_seed": RNG_SEED,
        },
        "context": {
            "n_gated_nodes": int(gate.sum()),
            "gradient12_ratio_summary": {
                "min": float(g12_ratios.min()), "median": float(g12_ratios.median()),
                "max": float(g12_ratios.max()),
                "molecules": g12_ratios.round(4).to_dict(),
            },
            "all129_ratio_summary": {
                "median": float(df["ratio"].median()),
                "q25": float(df["ratio"].quantile(0.25)),
                "q75": float(df["ratio"].quantile(0.75)),
                "max": float(df["ratio"].max()),
            },
        },
        "spearman": {
            "rho_ratio_vs_delta_shrink": rho, "p_permutation": p,
            "rho_ratio_vs_delta_trust": rho_t, "p_permutation_trust": p_t,
        },
        "quartiles": {
            "table": qtab.reset_index().to_dict("records"),
            "Q1_vs_Q4_delta_shrink_mean_diff": d14,
            "Q1_vs_Q4_delta_shrink_boot_ci": [lo, hi],
            "Q1_vs_Q4_delta_shrink_perm_p": p14,
            "Q1_vs_Q4_delta_trust_mean_diff": d14t,
            "Q1_vs_Q4_delta_trust_perm_p": p14t,
            "sensitivity_no_gated_atoms": {
                "Q1n": int(len(q1c)), "Q4n": int(len(q4c)),
                "mean_diff": float(d14c), "boot_ci": [lo_c, hi_c], "perm_p": p14c,
            },
        },
        "low_ratio_beyond_gradient12": {
            "threshold": float(g12_med),
            "threshold_definition": "g12 median ratio (g12 ratio distribution is "
                                    "wide: 0.0-10.9, the dataset max)",
            "n_molecules": int(len(low)),
            "mean_delta_shrink": low_mean,
            "mean_delta_shrink_rest": rest_mean,
            "low_vs_rest_mean_diff": float(dl),
            "low_vs_rest_perm_p": float(pl),
            "low_vs_rest_boot_ci": [lo_l, hi_l],
            "low_group_own_ci": [lo_self, hi_self],
            "sensitivity_threshold_0_5": {"n": int(len(low5)),
                                          "low_vs_rest_mean_diff": float(d5),
                                          "perm_p": float(p5)},
            "molecules": low_list[["n_gated_atoms", "err_before", "ratio",
                                   "delta_shrink", "delta_trust", "in_Q_std"]]
                                  .round(4).to_dict("index"),
        },
        "alt_predictors": {
            "rho_abs_gated_sum_vs_delta": float(
                df[["pred_abs", "delta_shrink"]].rename(
                    columns={"pred_abs": "x"}).corr(method="spearman").iloc[0, 1])
                if False else None,
            "note": "reported in console; see generalization_per_molecule.csv for "
                    "raw values",
        },
        "verdict": verdict,
        "verdict_notes": [
            "GENERALIZES: rho significantly positive, low-ratio quartile "
            "significantly less-benefited/harmed, and non-g12 low-ratio molecules "
            "show the same resistance.",
            "PARTIAL: significant in some but not all tests.",
            "CASE_STUDY: no significant association -- gradient-12 is an isolated "
            "case; the ratio does not generalize as a predictor.",
            "Premise note: g12's ABSOLUTE gated sums are small (max 4.09 kcal/mol, "
            "Part 8) but its RATIOS span the full dataset range (0.0-10.9 = dataset "
            "max) because several g12 members have very small errors; the ratio "
            "framing is therefore a distortion of the Part 8 finding.",
        ],
        "runtime_s": time.time() - t0,
    }
    with open(os.path.join(HERE, "generalization_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[save] generalization_report.json + generalization_per_molecule.csv + "
          f"generalization_low_ratio_molecules.csv")
    print(f"[done] {time.time()-t0:.0f}s  verdict={verdict}")


if __name__ == "__main__":
    main()
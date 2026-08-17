"""Broad-uncertainty re-analysis of the 17-run neighbor-regularization sweep.

Uses only saved per-molecule predictions (no new training).

Populations (fold-0 test, n=129):
  - Q_std : top quartile by baseline 5-seed ensemble_std
  - Q_nll : top quartile by GMM mean_NLL
  - UNION  : Q_std | Q_nll (the "broad uncertain" population)
  - overlap between the two definitions reported separately

Per run (all single-seed, seed 42):
  - MAE on all129 / Q_std / Q_nll / UNION, delta vs same-env lambda=0 baseline,
    paired-bootstrap 95% CI
  - convergence diagnostics: per-molecule prediction shift, distance to
    neighbor-mean (own graph source) before vs after, test-test neighbor edge
    convergence, population prediction spread, cross-run agreement vs truth

NOTE: sweep runs are single-seed (config.json seed=42), so a true 5-seed
post-regularization ensemble_std does not exist. "Std after" is proxied by
(a) within-run neighbor convergence, (b) cross-run consensus spread, and
(c) per-molecule prediction movement. Flagged explicitly in the report.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\User\Documents\Data"
SWEEP = os.path.join(
    ROOT, "aqm-spice2", "freesolv", "neighbor_regularization",
    "aqm-spice2", "freesolv", "neighbor_regularization",
)
OUT = os.path.join(ROOT, "aqm-spice2", "freesolv", "neighbor_regularization",
                   "broad_uncertainty_reanalysis")
os.makedirs(OUT, exist_ok=True)

AGG = os.path.join(ROOT, "aqm-spice2", "freesolv", "deep_ensemble",
                   "aggregate", "per_molecule.csv")
NLL = os.path.join(ROOT, "aqm-spice2", "freesolv", "deep_ensemble",
                   "gmm_uncertainty_check", "per_molecule_gmm_nll.csv")
G12 = os.path.join(ROOT, "aqm-spice2", "freesolv", "deep_ensemble",
                   "gmm_uncertainty_check", "gradient12_investigation",
                   "gradient12_ungrouped.csv")
GRAPH_TANI = os.path.join(ROOT, "aqm-spice2", "freesolv", "neighbor_regularization",
                          "graph_cache", "graph_k5_sim0.1.json")
GRAPH_LAT = os.path.join(ROOT, "aqm-spice2", "freesolv", "neighbor_regularization",
                         "graph_cache", "latent_k5_sim0.5.json")

RNG = np.random.default_rng(0)


def load_run(rel):
    d = os.path.join(SWEEP, rel)
    pred = pd.read_csv(os.path.join(d, "augmented_predictions.csv"))
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
    return pred, cfg, met


def run_label(cfg):
    src = cfg.get("neighbor_source", "tanimoto")
    norm = cfg.get("normalize_nbr", False)
    lam = cfg.get("lambda_nbr", 0.0)
    return f"{src}_{'normalized' if norm else 'raw'}_lam{lam:g}"


def collect_runs():
    """Discover the 17 runs; returns DataFrame with pred, cfg, meta per run."""
    runs = []
    groups = {
        "baseline": ["lambda0_seed42"],
        "raw": ["lambda0.001_seed42", "lambda0.003_seed42", "lambda0.01_seed42", "lambda0.03_seed42"],
        "normalized": ["lambda0.05_seed42", "lambda0.1_seed42", "lambda0.3_seed42", "lambda1.0_seed42"],
        "v2_latent": [f"raw_lambda{l}_seed42" for l in ["0.001", "0.003", "0.01", "0.03"]],
        "v2_latent/normalized": [f"lambda{l}_seed42" for l in ["0.05", "0.1", "0.3", "1.0"]],
    }
    for grp, names in groups.items():
        for n in names:
            rel = os.path.join(grp, n)
            pred, cfg, met = load_run(rel)
            runs.append({"rel": rel, "label": run_label(cfg), "group": grp,
                         "cfg": cfg, "met": met, "pred": pred})
    return runs


def paired_boot_ci(delta, n_boot=10000, alpha=0.05):
    """95% CI on mean of per-molecule deltas (paired bootstrap)."""
    delta = np.asarray(delta, dtype=float)
    idx = RNG.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def main():
    # ---- baseline ensemble + NLL + groups -------------------------------
    agg = pd.read_csv(AGG)
    nll = pd.read_csv(NLL)[["mol_id", "mean_nll"]]
    g12 = set(pd.read_csv(G12).mol_id)

    info = agg.merge(nll, on="mol_id", how="left")
    info["is_gradient12"] = info.mol_id.isin(g12)

    # quartile thresholds (top quartile = highest uncertainty)
    q_std = info.ensemble_std.quantile(0.75)
    q_nll = info.mean_nll.quantile(0.75)
    info["Q_std"] = info.ensemble_std > q_std
    info["Q_nll"] = info.mean_nll > q_nll
    info["UNION"] = info.Q_std | info.Q_nll

    n_qstd, n_qnll, n_union = info.Q_std.sum(), info.Q_nll.sum(), info.UNION.sum()
    n_overlap = (info.Q_std & info.Q_nll).sum()
    print(f"quartiles: Q_std n={n_qstd} (thr {q_std:.4f}), "
          f"Q_nll n={n_qnll} (thr {q_nll:.4f}), overlap n={n_overlap}, "
          f"union n={n_union}")
    print(f"gradient-12 inside: Q_std {info[info.is_gradient12].Q_std.sum()}, "
          f"Q_nll {info[info.is_gradient12].Q_nll.sum()}, "
          f"union {info[info.is_gradient12].UNION.sum()}")

    info.to_csv(os.path.join(OUT, "uncertain_population.csv"), index=False)

    # ---- runs ------------------------------------------------------------
    runs = collect_runs()
    assert len(runs) == 17, f"expected 17 runs, got {len(runs)}"

    base = next(r for r in runs if r["label"] == "tanimoto_raw_lam0")
    base_pred = base["pred"].set_index("mol_id").dG_pred_kcal
    base_err = (base_pred - base["pred"].set_index("mol_id").dG_exp_kcal).abs()

    rows_mae, rows_diag = [], []

    for r in runs:
        pred = r["pred"].set_index("mol_id")
        err = (pred.dG_pred_kcal - pred.dG_exp_kcal).abs()

        lam = r["cfg"].get("lambda_nbr", 0.0)

        mae_all = err.mean()
        row_mae = {"run": r["rel"], "label": r["label"], "lambda": lam,
                   "group": r["group"], "test_mae_tta_metrics": r["met"].get("test_mae_tta_kcal"),
                   "all129_mae": mae_all}
        for pop in ["Q_std", "Q_nll", "UNION"]:
            ids = info[info[pop]].mol_id
            e_pop = err.reindex(ids)
            e_base = base_err.reindex(ids)
            delta = e_pop.values - e_base.values
            ci = paired_boot_ci(delta)
            row_mae[f"{pop}_mae"] = e_pop.mean()
            row_mae[f"{pop}_delta"] = delta.mean()
            row_mae[f"{pop}_ci_lo"] = ci[0]
            row_mae[f"{pop}_ci_hi"] = ci[1]
        rows_mae.append(row_mae)

        # ---- convergence diagnostics on UNION population ----
        # NOTE: sweep runs are single-seed (seed 42); a true 5-seed
        # post-regularization ensemble_std does not exist. Proxies used:
        #   (a) per-molecule prediction shift vs baseline seed-42
        #   (b) population prediction spread (across molecules) before vs after
        #   (c) cross-run consensus (16 variants as pseudo-seeds) in the
        #       separate consensus table
        union_ids = list(info[info.UNION].mol_id)
        shift = (pred.dG_pred_kcal - base_pred).abs().reindex(union_ids)
        d_err = (err - base_err).reindex(union_ids)

        # population prediction spread (across molecules) before vs after
        spread_b = base_pred.reindex(union_ids).std()
        spread_a = pred.dG_pred_kcal.reindex(union_ids).std()

        def sp(x, y):
            m = x.dropna().index.intersection(y.dropna().index)
            if len(m) < 5 or x[m].nunique() < 2 or y[m].nunique() < 2:
                return np.nan, np.nan
            return stats.spearmanr(x[m], y[m]).statistic, stats.spearmanr(x[m], y[m]).pvalue

        rho_shift_err, p_shift_err = sp(shift, d_err)   # moved-more <-> error change

        # movement toward truth among improved molecules: for each improved
        # molecule, signed move along (truth - baseline_pred) axis; positive = toward truth
        improved = list(d_err.index[d_err < 0])
        toward_truth = (pred.dG_pred_kcal - base_pred).reindex(improved)
        sign_axis = np.sign(base_pred.reindex(improved) - pred.dG_exp_kcal.reindex(improved))
        toward_truth = -(toward_truth * sign_axis)

        rows_diag.append({
            "run": r["rel"], "label": r["label"], "lambda": lam, "group": r["group"],
            "union_n": len(union_ids),
            "union_delta_mae": row_mae["UNION_delta"],
            "median_abs_shift": shift.median(),
            "pop_pred_spread_before": spread_b,
            "pop_pred_spread_after": spread_a,
            "rho_shift_vs_err": rho_shift_err, "p_shift_vs_err": p_shift_err,
            "n_improved": len(improved),
            "frac_improved": len(improved) / len(union_ids) if union_ids else np.nan,
            "mean_toward_truth_given_improved": toward_truth.mean() if len(improved) else np.nan,
        })

    df_mae = pd.DataFrame(rows_mae)
    df_diag = pd.DataFrame(rows_diag)
    df_mae.to_csv(os.path.join(OUT, "per_run_broad_mae.csv"), index=False)
    df_diag.to_csv(os.path.join(OUT, "convergence_diagnostics.csv"), index=False)

    # ---- cross-run consensus: 16 variants as pseudo-seeds ----
    var_runs = [r for r in runs if r["cfg"].get("lambda_nbr", 0.0) > 0]
    preds = pd.DataFrame({r["label"]: r["pred"].set_index("mol_id").dG_pred_kcal
                          for r in var_runs})
    preds = preds.reindex(info.mol_id)
    consensus = pd.DataFrame({
        "mol_id": info.mol_id,
        "cross_run_mean": preds.mean(axis=1),
        "cross_run_std": preds.std(axis=1),
        "base_ensemble_mean": agg.set_index("mol_id").ensemble_mean.reindex(info.mol_id),
        "base_ensemble_std": agg.set_index("mol_id").ensemble_std.reindex(info.mol_id),
        "base_seed42_pred": base_pred.reindex(info.mol_id),
        "exp": agg.set_index("mol_id").true_value.reindex(info.mol_id),
        "is_gradient12": info.is_gradient12,
    })
    consensus["cross_run_err"] = (consensus.cross_run_mean - consensus.exp).abs()
    consensus["base_ens_err"] = (consensus.base_ensemble_mean - consensus.exp).abs()
    consensus["base_seed42_err"] = (consensus.base_seed42_pred - consensus.exp).abs()
    consensus.to_csv(os.path.join(OUT, "cross_run_consensus.csv"), index=False)

    print("\nMAE table (kcal/mol):")
    print(df_mae[["label", "all129_mae", "Q_std_mae", "Q_nll_mae", "UNION_mae",
                  "UNION_delta"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nCross-run consensus (all 129):")
    print(f"  mean abs err: baseline ensemble {consensus.base_ens_err.mean():.3f}, "
          f"seed42 {consensus.base_seed42_err.mean():.3f}, "
          f"cross-run consensus {consensus.cross_run_err.mean():.3f}")
    print(f"  mean std: baseline 5-seed {consensus.base_ensemble_std.mean():.3f}, "
          f"cross-run (16 variants) {consensus.cross_run_std.mean():.3f}")
    union_ids_all = list(info[info.UNION].mol_id)
    print(f"  union pop: consensus {consensus.loc[union_ids_all, 'cross_run_err'].mean():.3f} "
          f"vs baseline ens {consensus.loc[union_ids_all, 'base_ens_err'].mean():.3f} "
          f"vs seed42 {consensus.loc[union_ids_all, 'base_seed42_err'].mean():.3f}")


if __name__ == "__main__":
    main()
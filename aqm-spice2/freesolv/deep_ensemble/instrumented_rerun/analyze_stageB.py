"""Stage B cross-seed trajectory analysis: gradient-12 vs certain-47, across all
5 instrumented seeds (42, 123, 7, 2024, 999).

Questions asked (per seed, then pooled):
  Q1  When does each molecule's prediction first get within tol of the true
      value - and does it STAY there, or drift back out? (first-hit epoch,
      stability after hit, drift-out count, longest out-streak)
  Q2  Post-convergence oscillation: std of per-molecule prediction over the
      last 20% of training.
  Q3  Is the pooled-val best epoch actually optimal for gradient-12, or does
      the pooled choice sacrifice this subgroup? (err at pooled-best epoch vs
      err at the group's own optimal epoch, per group)
  Q4  Cross-seed consistency: same direction in how many of the 5 seeds?
      (per-metric direction + MWU per seed; per-molecule "slow" consistency)

Outputs (instrumented_rerun/analysis_stageB/):
  stageB_report.json          full numbers
  stageB_curves_panel.png     group mean |err| curves, one panel per seed
  stageB_g12_trajectories.png individual gradient-12 |err| curves per seed
  stageB_crossseed.png        pooled boxplots + per-seed direction/p panels

Usage: python analyze_stageB.py [--seeds 42 123 7 2024 999]
                                [--instrumented <root>] [--out <dir>]
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_deep_ensemble = os.path.dirname(_script_dir)
_freesolv = os.path.dirname(_deep_ensemble)
if _freesolv not in sys.path:
    sys.path.insert(0, _freesolv)

ANALYSIS_CSV = os.path.join(_deep_ensemble, "rmse_analysis", "output",
                            "per_molecule_rmse.csv")
ISOLATION_CSV = os.path.join(_deep_ensemble, "rmse_analysis",
                             "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")
TOL_KCAL = 0.5
RANDOM_SEED = 42


def load_groups():
    import pandas as pd
    rmse_df = pd.read_csv(ANALYSIS_CSV).set_index("mol_id")
    iso = pd.read_csv(ISOLATION_CSV)
    iso18 = iso[iso["group"] == "confidently_wrong"].sort_values("best_sim")
    isolated6 = set(iso18.head(6)["mol_id"])
    wrong18 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_high_rmse"])
    certain47 = sorted(set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_low_rmse"]))
    grad12 = sorted(wrong18 - isolated6)
    return grad12, certain47, sorted(isolated6), sorted(wrong18)


def mol_stats_row(d, best_val_epoch, n_epochs):
    """Per-molecule trajectory stats from its epoch_predictions slice."""
    import numpy as np
    err = d["abs_err_kcal"].values
    pred = d["dG_pred_kcal"].values
    epochs = d["epoch"].values
    hit = err < TOL_KCAL
    first = int(np.argmax(hit)) if np.any(hit) else len(err)
    first_hit_epoch = int(epochs[first]) if first < len(err) else n_epochs + 1
    if first < len(err):
        after = hit[first:]
        stability = float(after.mean())
        runs = 0
        max_streak = 0
        streak = 0
        prev = True
        for v in after:
            if not v and prev:
                runs += 1
            if not v:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
            prev = bool(v)
        drift_outs = runs
        max_out_streak = max_streak
    else:
        stability = 0.0
        drift_outs = 0
        max_out_streak = 0
    tail = int(np.ceil(len(err) * 0.2))
    osc = float(np.std(pred[-tail:])) if len(pred) >= 2 else float("nan")
    idx_pooled = int(np.where(epochs == best_val_epoch)[0][0]) \
        if np.any(epochs == best_val_epoch) else len(err) - 1
    own_best = int(np.argmin(err))
    own_best_err = float(err[own_best])
    err_at_pooled = float(err[idx_pooled])
    return {
        "mol_id": d["mol_id"].iloc[0],
        "first_hit_epoch": first_hit_epoch,
        "never_within_tol": int(first_hit_epoch > n_epochs),
        "stability_after_hit": stability,
        "drift_outs_after_hit": drift_outs,
        "max_out_streak_epochs": max_out_streak,
        "tail_oscillation_std_kcal": osc,
        "err_at_pooled_best": err_at_pooled,
        "own_best_err_kcal": own_best_err,
        "pooled_sacrifice_kcal": err_at_pooled - own_best_err,
        "own_best_before_pooled": int(own_best < idx_pooled),
        "last_epoch_abs_err": float(err[-1]),
    }


def analyze_seed(seed_dir, grad12, certain47):
    """Per-seed run: group summaries, MWU tests, Q3 pooled-vs-group-optimal."""
    import json as _json
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu

    ep = pd.read_csv(os.path.join(seed_dir, "epoch_predictions.csv"))
    with open(os.path.join(seed_dir, "metrics.json")) as f:
        metrics = _json.load(f)
    n_epochs = int(ep["epoch"].max())
    # authoritative: the epoch the training actually kept for early stopping.
    # (val_history.csv has a warm-start row at epoch 0 whose value can win a
    # naive min-sort - never derive best_val_epoch from it.)
    best_val_epoch = int(metrics.get("best_val_epoch", -1))
    if best_val_epoch <= 0:
        val = pd.read_csv(os.path.join(seed_dir, "val_history.csv"), index_col=False)
        best_val_epoch = int(val.sort_values("val_mae_kcal").iloc[0]["epoch"])

    def stats_for(grp):
        rows = [mol_stats_row(ep[ep["mol_id"] == m].sort_values("epoch"),
                              best_val_epoch, n_epochs) for m in grp]
        rows = [r for r in rows if r is not None]
        return pd.DataFrame(rows), len(grp) - len(rows)

    df_g, miss_g = stats_for(grad12)
    df_c, miss_c = stats_for(certain47)

    def med(s):
        return float(np.median(s)) if len(s) else float("nan")

    mwu = {}
    for key in ["first_hit_epoch", "stability_after_hit", "drift_outs_after_hit",
                "max_out_streak_epochs", "tail_oscillation_std_kcal",
                "err_at_pooled_best", "pooled_sacrifice_kcal",
                "last_epoch_abs_err"]:
        a = df_g[key].values
        b = df_c[key].values
        if len(a) and len(b):
            U, p = mannwhitneyu(a, b, alternative="two-sided")
            mwu[key] = {"p": float(p), "med_g12": med(a), "med_c47": med(b)}
        else:
            mwu[key] = {"p": float("nan"), "med_g12": float("nan"),
                        "med_c47": float("nan")}

    # Q3: pooled best epoch vs each group's own optimal epoch
    group_optimal = {}
    for name, grp in [("gradient12", grad12), ("certain47", certain47)]:
        g = ep[ep["mol_id"].isin(grp)].groupby("epoch")["abs_err_kcal"].mean()
        opt = int(g.idxmin())
        group_optimal[name] = {
            "n": len(grp),
            "group_optimal_epoch": opt,
            "err_at_group_optimal": float(g.min()),
            "err_at_pooled_best": float(g.loc[best_val_epoch])
            if best_val_epoch in g.index else float("nan"),
            "sacrifice_kcal": float(g.loc[best_val_epoch] - g.min())
            if best_val_epoch in g.index else float("nan"),
        }

    summary = {
        "n_epochs": n_epochs, "best_val_epoch": best_val_epoch,
        "gradient12": {
            "n": len(grad12), "missing_from_log": miss_g,
            "first_hit_epoch_median": med(df_g["first_hit_epoch"]),
            "never_within_tol": int(df_g["never_within_tol"].sum()),
            "stability_after_hit_median": med(df_g["stability_after_hit"]),
            "drift_outs_after_hit_median": med(df_g["drift_outs_after_hit"]),
            "max_out_streak_median": med(df_g["max_out_streak_epochs"]),
            "tail_oscillation_std_median": med(df_g["tail_oscillation_std_kcal"]),
            "err_at_pooled_best_median": med(df_g["err_at_pooled_best"]),
            "own_best_err_median": med(df_g["own_best_err_kcal"]),
            "pooled_sacrifice_median": med(df_g["pooled_sacrifice_kcal"]),
            "own_best_before_pooled_frac": float(df_g["own_best_before_pooled"].mean()),
            "last_epoch_abs_err_median": med(df_g["last_epoch_abs_err"]),
        },
        "certain47": {
            "n": len(certain47), "missing_from_log": miss_c,
            "first_hit_epoch_median": med(df_c["first_hit_epoch"]),
            "never_within_tol": int(df_c["never_within_tol"].sum()),
            "stability_after_hit_median": med(df_c["stability_after_hit"]),
            "drift_outs_after_hit_median": med(df_c["drift_outs_after_hit"]),
            "max_out_streak_median": med(df_c["max_out_streak_epochs"]),
            "tail_oscillation_std_median": med(df_c["tail_oscillation_std_kcal"]),
            "err_at_pooled_best_median": med(df_c["err_at_pooled_best"]),
            "own_best_err_median": med(df_c["own_best_err_kcal"]),
            "pooled_sacrifice_median": med(df_c["pooled_sacrifice_kcal"]),
            "own_best_before_pooled_frac": float(df_c["own_best_before_pooled"].mean()),
            "last_epoch_abs_err_median": med(df_c["last_epoch_abs_err"]),
        },
        "mwu": mwu,
        "group_optimal": group_optimal,
    }

    print(f"[seed analy] {os.path.basename(seed_dir)}: n_ep={n_epochs} "
          f"best_val={best_val_epoch} | g12 first-hit med "
          f"{summary['gradient12']['first_hit_epoch_median']:.1f} vs c47 "
          f"{summary['certain47']['first_hit_epoch_median']:.1f} (p="
          f"{mwu['first_hit_epoch']['p']:.3f}) | tail-osc g12 "
          f"{summary['gradient12']['tail_oscillation_std_median']:.3f} vs c47 "
          f"{summary['certain47']['tail_oscillation_std_median']:.3f} (p="
          f"{mwu['tail_oscillation_std_kcal']['p']:.3f}) | Q3 g12 sacrifice "
          f"{group_optimal['gradient12']['sacrifice_kcal']:.3f} kcal")

    return {"summary": summary, "ep": ep, "df_g": df_g, "df_c": df_c,
            "n_epochs": n_epochs, "best_val_epoch": best_val_epoch}


def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 7, 2024, 999])
    ap.add_argument("--instrumented", default=_script_dir)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.instrumented, "analysis_stageB")
    os.makedirs(out_dir, exist_ok=True)

    grad12, certain47, isolated6, wrong18 = load_groups()
    print(f"[stageB] groups: gradient12 n={len(grad12)} | certain47 n={len(certain47)} "
          f"| tol={TOL_KCAL} kcal")

    per_seed = {}
    seed_data = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.instrumented, f"seed_{seed}")
        ep_csv = os.path.join(seed_dir, "epoch_predictions.csv")
        if not os.path.exists(ep_csv):
            print(f"[stageB] WARNING: {ep_csv} missing - skipping seed {seed}")
            continue
        res = analyze_seed(seed_dir, grad12, certain47)
        per_seed[str(seed)] = res["summary"]
        seed_data.append((int(seed), res))

    if not seed_data:
        print("ERROR: no seeds with epoch_predictions.csv found")
        sys.exit(2)

    # Q4: cross-seed consistency
    metrics = ["first_hit_epoch", "stability_after_hit", "drift_outs_after_hit",
               "max_out_streak_epochs", "tail_oscillation_std_kcal",
               "err_at_pooled_best", "pooled_sacrifice_kcal", "last_epoch_abs_err"]
    worse_dir = {"first_hit_epoch": 1, "stability_after_hit": -1,
                 "drift_outs_after_hit": 1, "max_out_streak_epochs": 1,
                 "tail_oscillation_std_kcal": 1, "err_at_pooled_best": 1,
                 "pooled_sacrifice_kcal": 1, "last_epoch_abs_err": 1}

    consistency = {}
    for key in metrics:
        rows = []
        for seed, res in seed_data:
            m = res["summary"]["mwu"][key]
            rows.append({"seed": seed, "med_g12": m["med_g12"],
                         "med_c47": m["med_c47"], "p": m["p"]})
        signs = []
        for r in rows:
            d = (r["med_g12"] - r["med_c47"]) * worse_dir[key]
            signs.append(1 if d > 0 else (-1 if d < 0 else 0))
        pos = sum(1 for s in signs if s == 1)   # seeds where g12 is WORSE
        neg = sum(1 for s in signs if s == -1)  # seeds where g12 is BETTER
        direction = "g12_worse" if pos > neg else ("g12_better" if neg > pos else "tie")
        n_sig = sum(1 for r in rows if r["p"] < 0.05)
        consistency[key] = {
            "per_seed": rows,
            "n_seeds_g12_worse": pos, "n_seeds_g12_better": neg,
            "direction": direction,
            "n_seeds_sig_p005": n_sig,
        }
        worse = "g12 worse" if direction == "g12_worse" else "g12 better"
        print(f"[stageB] {key:<24} direction={worse:<11} ({pos}/{len(rows)} seeds worse) "
              f"sig p<0.05: {n_sig}/{len(rows)}")

    # per-molecule consistency: is each gradient-12 molecule consistently slow?
    c47_median_hit = {}
    for seed, res in seed_data:
        c47_median_hit[seed] = float(np.median(res["df_c"]["first_hit_epoch"]))
    per_mol = {}
    for mid in grad12:
        hits = []
        for seed, res in seed_data:
            row = res["df_g"][res["df_g"]["mol_id"] == mid]
            if len(row):
                hits.append((seed, int(row.iloc[0]["first_hit_epoch"])))
        if not hits:
            continue
        late = sum(1 for s, h in hits if h > c47_median_hit[s])
        per_mol[mid] = {
            "seeds_with_data": len(hits),
            "first_hit_per_seed": dict(hits),
            "frac_slower_than_c47_median": round(late / len(hits), 2),
        }

    report = {
        "tol_kcal": TOL_KCAL,
        "seeds": [s for s, _ in seed_data],
        "groups": {"gradient12": grad12, "certain47": certain47,
                   "isolated6": isolated6, "wrong18": wrong18},
        "per_seed": per_seed,
        "cross_seed": {"per_metric_direction": consistency,
                       "per_molecule_gradient12": per_mol},
        "note": ("Q1 first-hit+stability per molecule; Q2 tail oscillation last "
                 "20%; Q3 pooled-val best epoch vs group-optimal epoch; Q4 "
                 "direction + MWU per seed"),
    }
    with open(os.path.join(out_dir, "stageB_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_curves_panel(seed_data, out_dir)
    plot_g12_trajectories(seed_data, grad12, out_dir)
    plot_crossseed(seed_data, metrics, consistency, out_dir)
    print(f"\n[stageB] report + 3 PNGs -> {out_dir}")


def plot_curves_panel(seed_data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seed_data)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6),
                             squeeze=False, sharey=True)
    flat = axes.flatten()
    for i, (seed, res) in enumerate(seed_data):
        ax = flat[i]
        ep = res["ep"]
        for name, mids, color, ls in [("gradient12", res["df_g"]["mol_id"],
                                       "crimson", "-"),
                                      ("certain47", res["df_c"]["mol_id"],
                                       "tab:blue", "--")]:
            g = ep[ep["mol_id"].isin(mids)].groupby("epoch")["abs_err_kcal"].mean()
            ax.plot(g.index, g.values, ls, color=color, lw=1.3,
                    label=name if i == 0 else None)
        ax.axvline(res["best_val_epoch"], color="k", lw=0.8, ls=":")
        ax.set_title(f"seed {seed} (best-ep {res['best_val_epoch']})")
        ax.grid(alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend()
    fig.suptitle(f"Group mean |err| per seed (tol={TOL_KCAL} kcal)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stageB_curves_panel.png"), dpi=150)
    plt.close(fig)


def plot_g12_trajectories(seed_data, grad12, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seed_data)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 9),
                             squeeze=False)
    flat = axes.flatten()
    for i, (seed, res) in enumerate(seed_data):
        ax = flat[i]
        ep = res["ep"]
        for mid in grad12:
            dd = ep[ep["mol_id"] == mid]
            ax.plot(dd["epoch"], dd["abs_err_kcal"], lw=1, alpha=0.85)
        ax.axhline(TOL_KCAL, color="k", ls=":", lw=1)
        ax.axvline(res["best_val_epoch"], color="tab:gray", ls=":", lw=1)
        ax.set_title(f"seed {seed}")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle(f"gradient-12 individual |err| trajectories (tol={TOL_KCAL})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stageB_g12_trajectories.png"), dpi=150)
    plt.close(fig)


def plot_crossseed(seed_data, metrics, consistency, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for key, ax in zip(["first_hit_epoch", "tail_oscillation_std_kcal",
                        "err_at_pooled_best"], axes):
        g = []; c = []
        for seed, res in seed_data:
            g += list(res["df_g"][key].values)
            c += list(res["df_c"][key].values)
        ax.boxplot([g, c], showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["gradient-12", "certain-47"])
        ax.set_title(key.replace("_", " "))
        ax.grid(alpha=0.3)
    fig.suptitle("Pooled across seeds (fliers hidden)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stageB_crossseed.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
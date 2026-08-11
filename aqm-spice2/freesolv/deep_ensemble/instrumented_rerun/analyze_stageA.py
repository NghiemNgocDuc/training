"""Stage A early read (n=1 seed, seed 42): gradient-12 vs a matched random
sample of 12 from certain-47, from the per-epoch instrumented log.

Metrics per molecule (from epoch_predictions.csv):
  * first-hit epoch: first epoch with |err| < 0.5 kcal/mol (tol)
  * stability: fraction of epochs AFTER first-hit that stay within tol
  * post-convergence oscillation: std of prediction over the last 20% epochs
  * per-molecule best epoch: its own lowest |err| epoch, vs the pooled
    best-val epoch (does pooled early stopping sacrifice this subgroup?)
  * last-epoch |err| (model at best-val checkpoint, single conf)

Group mean |err| curves, MWU tests, and three plots. Early read only,
not a final conclusion.

Usage: python analyze_stageA.py [--seed 42] [--instrumented <dir>]
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))          # .../deep_ensemble/instrumented_rerun
_deep_ensemble = os.path.dirname(_script_dir)                     # .../deep_ensemble (original outputs)
_freesolv = os.path.dirname(_deep_ensemble)                       # .../freesolv (deep_ensemble.py lives here)
if _freesolv not in sys.path:
    sys.path.insert(0, _freesolv)

ANALYSIS_CSV = os.path.join(_deep_ensemble, "rmse_analysis", "output",
                            "per_molecule_rmse.csv")
ISOLATION_CSV = os.path.join(_deep_ensemble, "rmse_analysis",
                             "neighbor_isolation_check",
                             "neighbor_similarity_results.csv")
TOL_KCAL = 0.5
RANDOM_SEED = 42


def main():
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instrumented", default=_script_dir)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_dir = os.path.join(args.instrumented, f"seed_{args.seed}")
    out_dir = args.out or seed_dir
    os.makedirs(out_dir, exist_ok=True)

    ep = pd.read_csv(os.path.join(seed_dir, "epoch_predictions.csv"))
    val = pd.read_csv(os.path.join(seed_dir, "val_history.csv"))
    n_epochs = int(ep["epoch"].max())
    best_val_epoch = int(val.sort_values("val_mae_kcal").iloc[0]["epoch"])

    rmse_df = pd.read_csv(ANALYSIS_CSV).set_index("mol_id")
    iso = pd.read_csv(ISOLATION_CSV)
    iso18 = iso[iso["group"] == "confidently_wrong"].sort_values("best_sim")
    isolated6 = set(iso18.head(6)["mol_id"])
    wrong18 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_high_rmse"])
    certain47 = set(rmse_df.index[rmse_df["quadrant_label"] == "low_std_low_rmse"])
    grad12 = sorted(wrong18 - isolated6)
    rng = np.random.RandomState(RANDOM_SEED)
    rand12 = sorted(rng.choice(sorted(certain47), 12, replace=False))

    print(f"[stageA] seed={args.seed} epochs={n_epochs} best_val_epoch={best_val_epoch} "
          f"tol={TOL_KCAL} kcal")
    print(f"[stageA] gradient12 n={len(grad12)} | random certain-47 n={len(rand12)}")

    def mol_stats(mid):
        d = ep[ep["mol_id"] == mid].sort_values("epoch")
        if len(d) == 0:
            return None
        err = d["abs_err_kcal"].values
        pred = d["dG_pred_kcal"].values
        epochs = d["epoch"].values
        first_hit = int(np.argmax(err < TOL_KCAL)) if np.any(err < TOL_KCAL) else len(err)
        hit_epoch = int(epochs[first_hit]) if first_hit < len(err) else n_epochs + 1
        stay = float(np.mean(err[first_hit:] < TOL_KCAL)) if first_hit < len(err) else 0.0
        tail = int(np.ceil(len(err) * 0.2))
        osc = float(np.std(pred[-tail:])) if len(pred) >= 2 else float("nan")
        own_best = int(np.argmin(err))
        idx_best = int(np.where(epochs == best_val_epoch)[0][0]) \
            if np.any(epochs == best_val_epoch) else len(err) - 1
        return {
            "mol_id": mid,
            "first_hit_epoch": hit_epoch,
            "stability_after_hit": stay,
            "tail_oscillation_std_kcal": osc,
            "own_best_epoch": own_best,
            "err_to_pooled_best_epoch": float(err[idx_best]),
            "last_epoch_abs_err": float(err[-1]),
            "ahead_of_pooled_best": own_best < int(np.where(epochs == best_val_epoch)[0][0]) if np.any(epochs == best_val_epoch) else bool(False),
        }

    def summarize(group, name):
        rows = [mol_stats(m) for m in group]
        rows = [r for r in rows if r is not None]
        missing = len(group) - len(rows)
        df = pd.DataFrame(rows)
        out = {
            "n": len(group),
            "missing_from_log": missing,
            "first_hit_epoch_median": float(df["first_hit_epoch"].median()),
            "first_hit_epoch_mean": float(df["first_hit_epoch"].mean()),
            "never_within_tol_count": int((df["first_hit_epoch"] > n_epochs).sum()),
            "stability_after_hit_median": float(df["stability_after_hit"].median()),
            "tail_oscillation_std_median": float(df["tail_oscillation_std_kcal"].median()),
            "tail_oscillation_std_mean": float(df["tail_oscillation_std_kcal"].mean()),
            "own_best_before_pooled_best_frac": float(df["ahead_of_pooled_best"].mean()),
            "err_at_pooled_best_median": float(df["err_to_pooled_best_epoch"].median()),
            "last_epoch_abs_err_median": float(df["last_epoch_abs_err"].median()),
            "last_epoch_abs_err_mean": float(df["last_epoch_abs_err"].mean()),
        }
        print(f"[{name}] first-hit med {out['first_hit_epoch_median']:.1f} "
              f"(never {out['never_within_tol_count']}/{out['n']}) | stability med "
              f"{out['stability_after_hit_median']:.2f} | tail-osc med "
              f"{out['tail_oscillation_std_median']:.3f} | own-best<pooled-best "
              f"{out['own_best_before_pooled_best_frac']:.2f} | err@pooled-best med "
              f"{out['err_at_pooled_best_median']:.3f} | last-err med "
              f"{out['last_epoch_abs_err_median']:.3f}")
        return out, df

    s_g12, df_g12 = summarize(grad12, "gradient12")
    s_r12, df_r12 = summarize(rand12, "certain47-rand12")

    mwu = {}
    for key, col in [("first_hit_epoch", "first_hit_epoch"),
                     ("stability_after_hit", "stability_after_hit"),
                     ("tail_oscillation_std", "tail_oscillation_std_kcal"),
                     ("err_at_pooled_best", "err_to_pooled_best_epoch"),
                     ("last_epoch_abs_err", "last_epoch_abs_err")]:
        a = df_g12[col].values
        b = df_r12[col].values
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        mwu[key] = {"U": float(U), "p": float(p),
                    "median_g12": float(np.median(a)), "median_certain47": float(np.median(b))}
        print(f"[mwu] {key}: g12 med {np.median(a):.3f} vs c47 med {np.median(b):.3f} "
              f"p={p:.3f}")

    group_curves = {}
    for name, grp in [("gradient12", grad12), ("certain47_rand12", rand12),
                      ("certain47_all", sorted(certain47))]:
        g = ep[ep["mol_id"].isin(grp)].groupby("epoch")["abs_err_kcal"].mean()
        group_curves[name] = {"epochs": [int(i) for i in g.index],
                              "mean_abs_err_kcal": [float(v) for v in g.values]}

    report = {
        "seed": args.seed, "n_epochs": n_epochs, "best_val_epoch": best_val_epoch,
        "tol_kcal": TOL_KCAL,
        "gradient12": s_g12, "certain47_rand12": s_r12,
        "mwu": mwu,
        "group_mean_curves": group_curves,
        "note": "Stage A early read, n=1 seed. Not a final conclusion.",
    }
    with open(os.path.join(out_dir, "stageA_early_read.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_curves(ep, grad12, rand12, out_dir)
    print(f"\n[stageA] plots + json -> {out_dir}")


def plot_curves(ep, grad12, rand12, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted(ep["epoch"].unique())
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    g = ep[ep["mol_id"].isin(grad12)].groupby("epoch")["abs_err_kcal"].mean()
    r = ep[ep["mol_id"].isin(rand12)].groupby("epoch")["abs_err_kcal"].mean()
    ax[0].plot(g.index, g.values, "o-", ms=3, label="gradient-12 (n=12)", color="crimson")
    ax[0].plot(r.index, r.values, "s-", ms=3, label="certain-47 rand 12", color="tab:blue")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean |err| (kcal/mol)")
    ax[0].set_title("Group mean error trajectory"); ax[0].legend(); ax[0].grid(alpha=0.3)

    d = ep[ep["mol_id"].isin(grad12)]
    for mid in grad12:
        dd = d[d["mol_id"] == mid]
        ax[1].plot(dd["epoch"], dd["abs_err_kcal"], lw=1, alpha=0.8, label=mid if mid == grad12[0] else None)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("|err| (kcal/mol)")
    ax[1].set_title("Individual gradient-12 trajectories"); ax[1].grid(alpha=0.3)

    for name, grp, color in [("g12", grad12, "crimson"), ("c47", rand12, "tab:blue")]:
        g2 = ep[ep["mol_id"].isin(grp)].groupby("mol_id")["abs_err_kcal"].min()
        ax[2].hist(g2.values, bins=20, alpha=0.55, label=name, color=color)
    ax[2].set_xlabel("best single-epoch |err| (kcal/mol)"); ax[2].set_ylabel("molecules")
    ax[2].set_title("Best-epoch error distribution"); ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stageA_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
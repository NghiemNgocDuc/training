"""Plot per-epoch curves from neighbor-regularization runs.

Reads epoch_history.csv from each run and plots:
  A) raw L_neighbor magnitude (eV^2) over epochs  - the key stability check
  B) L_neighbor value actually used in the loss (raw or normalized)
  C) task MSE (eV^2, log scale) - is learning still happening?
  D) prediction variance var(p) over epochs - drift of the raw formula's
     effective strength as the variance shrinks

Outputs: plot_l_nbr.png (all runs) and plot_l_nbr_<variant>.png per variant.

Usage:
  python plot_l_nbr.py --runs raw/lambda0.001_seed42 raw/lambda0.01_seed42 \
      normalized/lambda0.1_seed42 normalized/lambda1.0_seed42
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (relative to this script dir)")
    args = ap.parse_args()

    runs = []
    for rel in args.runs:
        run_dir = os.path.join(_script_dir, rel)
        ep = os.path.join(run_dir, "epoch_history.csv")
        if not os.path.exists(ep):
            print(f"!! no epoch_history.csv in {run_dir}")
            continue
        with open(os.path.join(run_dir, "metrics.json")) as f:
            import json
            m = json.load(f)
        runs.append((rel, m, pd.read_csv(ep)))
    if not runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (rel, m, eh) in enumerate(runs):
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        mk = MARKERS[idx % len(MARKERS)]
        label = f"{rel} (l={m['lambda_nbr']})"
        lr = eh["l_nbr_raw_eV2"].dropna()
        if len(lr):
            axes[0, 0].plot(eh["epoch"][lr.index], lr.values, ls=ls, marker=mk,
                            ms=3, label=label)
        lu = eh["l_nbr_used"].dropna()
        if len(lu):
            axes[0, 1].plot(eh["epoch"][lu.index], lu.values, ls=ls, marker=mk,
                            ms=3, label=label)
        axes[1, 0].plot(eh["epoch"], eh["task_mse_train_eV2"], ls=ls, marker=mk,
                        ms=3, label=label)
        vp = eh["var_p_eV2"].dropna()
        if len(vp):
            axes[1, 1].plot(eh["epoch"][vp.index], vp.values, ls=ls, marker=mk,
                            ms=3, label=label)

    axes[0, 0].set_ylabel("L_neighbor raw (eV^2)"); axes[0, 0].set_xlabel("epoch")
    axes[0, 1].set_ylabel("L_neighbor used in loss"); axes[0, 1].set_xlabel("epoch")
    axes[1, 0].set_yscale("log"); axes[1, 0].set_ylabel("task MSE train (eV^2)")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_ylabel("var(pred) (eV^2)"); axes[1, 1].set_xlabel("epoch")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Neighbor-regularization per-epoch curves (fold-0, seed 42 smoke)")
    fig.tight_layout()
    fig.savefig(os.path.join(_script_dir, "plot_l_nbr.png"), dpi=150)
    print(f"saved -> {os.path.join(_script_dir, 'plot_l_nbr.png')}")

    for variant in ("raw", "normalized", "baseline"):
        sub = [(rel, m, eh) for rel, m, eh in runs
               if ("normalize_nbr" in m and m["normalize_nbr"]) == (variant == "normalized")]
        if variant == "baseline" and not sub:
            continue
        if not sub:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (rel, m, eh) in enumerate(sub):
            ls = LINE_STYLES[idx % len(LINE_STYLES)]
            mk = MARKERS[idx % len(MARKERS)]
            label = f"{rel} (l={m['lambda_nbr']})"
            lr = eh["l_nbr_raw_eV2"].dropna()
            if len(lr):
                axes[0, 0].plot(eh["epoch"][lr.index], lr.values, ls=ls,
                                marker=mk, ms=3, label=label)
            lu = eh["l_nbr_used"].dropna()
            if len(lu):
                axes[0, 1].plot(eh["epoch"][lu.index], lu.values, ls=ls,
                                marker=mk, ms=3, label=label)
            axes[1, 0].plot(eh["epoch"], eh["task_mse_train_eV2"], ls=ls,
                            marker=mk, ms=3, label=label)
            vp = eh["var_p_eV2"].dropna()
            if len(vp):
                axes[1, 1].plot(eh["epoch"][vp.index], vp.values, ls=ls,
                                marker=mk, ms=3, label=label)
        axes[0, 0].set_ylabel("L_neighbor raw (eV^2)"); axes[0, 0].set_xlabel("epoch")
        axes[0, 1].set_ylabel("L_neighbor used"); axes[0, 1].set_xlabel("epoch")
        axes[1, 0].set_yscale("log"); axes[1, 0].set_ylabel("task MSE train (eV^2)")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 1].set_ylabel("var(pred) (eV^2)"); axes[1, 1].set_xlabel("epoch")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"variant: {variant}")
        fig.tight_layout()
        p = os.path.join(_script_dir, f"plot_l_nbr_{variant}.png")
        fig.savefig(p, dpi=150)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()

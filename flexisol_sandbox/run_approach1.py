"""Run the (untouched) approach-1 weighted-retrain experiment on FlexiSol-water.

The approach-1 script in experimental_uncertainty_refine takes every input
via --flags, so porting = passing FlexiSol paths.  This driver does exactly
that; it does NOT modify that script.  Uses only the original alpha sweep
(defaults [0.0, 0.5, 1.0, 2.0]) - hard-mask mode was removed upstream.

Requires: the FlexiSol ensemble trained by train_flexisol_ensemble.py
(its aggregate per_molecule.csv is the uncertainty source), plus the built
dataset (build_hdf5.py) and its split.

Usage:
  python run_approach1.py [--smoke]
"""

import argparse
import os
import subprocess
import sys

SANDBOX_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SANDBOX_ROOT, "out")
ENSEMBLE = os.path.join(DATA, "ensemble")
APPROACH1 = os.path.abspath(os.path.join(
    SANDBOX_ROOT, "..", "aqm-spice2", "freesolv",
    "experimental_uncertainty_refine", "approach1_weighted_retrain.py"))
OUTPUT = os.path.join(DATA, "approach1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    cmd = [sys.executable, APPROACH1,
           "--split_dir", os.path.join(DATA, "split"),
           "--ensemble_dir", ENSEMBLE,
           "--conformers", os.path.join(DATA, "flexisol_water.hdf5"),
           "--labels_json", os.path.join(DATA, "labels.json"),
           "--per_molecule", os.path.join(ENSEMBLE, "aggregate", "per_molecule.csv"),
           "--output_dir", OUTPUT,
           "--device", args.device,
           "--epochs", str(args.epochs)]
    for a in args.alphas:
        cmd += ["--alphas", str(a)]
    if args.smoke:
        cmd.append("--smoke")

    print("running:", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
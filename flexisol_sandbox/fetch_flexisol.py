"""Fetch the FlexiSol benchmark repo (grimme-lab/flexisol, MIT).

The repo already contains experimental references AND the full 3D
conformer/tautomer geometry sets, so a plain shallow clone is all we
need -- no separate data download.

Sandbox local layout (created):

  flexisol_sandbox/
    data/flexisol_repo/            # shallow clone of grimme-lab/flexisol
    flexisol_water.hdf5            # built by build_hdf5.py
    labels.json                    # same schema as the FreeSolv loader
    split_train.json / val / test  # frozen split, mol_id lists
"""

import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_url", default="https://github.com/grimme-lab/flexisol.git")
    ap.add_argument("--dest", default="data/flexisol_repo")
    args = ap.parse_args()

    if os.path.isdir(os.path.join(args.dest, ".git")):
        print(f"[ok] already cloned at {args.dest}")
        return
    os.makedirs(args.dest, exist_ok=True)
    print(f"cloning {args.repo_url} (shallow) -> {args.dest}")
    subprocess.check_call(["git", "clone", "--depth", "1", args.repo_url, args.dest])
    print("done")


if __name__ == "__main__":
    main()
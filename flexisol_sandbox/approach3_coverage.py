"""Approach 3 (sandbox): active-learning-style coverage diagnostic on FlexiSol.

Diagnostic, no retraining.  For the test molecules the ensemble is most
uncertain about (highest ensemble std), measure how well their chemical
space is covered by the FlexiSol training pool:

  * Tanimoto NN (Morgan r=2, 2048 bits) of each top-std test molecule
    against every training molecule: max sim, mean of top-3, count of
    neighbors >= 0.7.
  * Spearman(ensemble_std, 1 / max_Tanimoto) across ALL test molecules --
    if uncertainty tracks lack of coverage, that is the finding.
  * Same coverage vs the full 297-molecule FlexiSol pool (train+val+test),
    to show whether close neighbors exist in-pool (just not in train).

Consumes only artifacts produced by the frozen pipeline:
  out/ensemble*/aggregate/per_molecule.csv (from train_flexisol_ensemble.py
  on Vast), out/labels.json, out/split/*.

Usage:
  python approach3_coverage.py --aggregate out/ensemble_full/aggregate/per_molecule.csv
"""

import argparse
import json
import os
import sys

import numpy as np

SANDBOX_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_csv_rows(path):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True,
                    help="per_molecule.csv from the ensemble aggregate dir")
    ap.add_argument("--out", default=os.path.join(SANDBOX_ROOT, "out", "approach3"))
    args = ap.parse_args()

    labels = json.load(open(os.path.join(SANDBOX_ROOT, "out", "labels.json")))
    split_dir = os.path.join(SANDBOX_ROOT, "out", "split")
    train_ids = json.load(open(os.path.join(split_dir, "train_ids.json")))
    test_ids = json.load(open(os.path.join(split_dir, "test_ids.json")))

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError as e:
        sys.exit(f"rdkit required: {e}")

    def fp(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

    train_fps = {}
    missing = []
    for mid in train_ids:
        f = fp(labels[mid]["smiles"])
        if f is None:
            missing.append((mid, labels[mid]["smiles"]))
            continue
        train_fps[mid] = f
    if missing:
        print(f"[warn] {len(missing)} train SMILES failed to parse: {missing[:5]}")
    if not train_fps:
        sys.exit("no parseable train SMILES, abort")

    rows = load_csv_rows(args.aggregate)
    test_rows = [r for r in rows if r["mol_id"] in set(test_ids)]
    print(f"aggregate rows: {len(rows)}  test rows: {len(test_rows)}")

    parsed = []
    for r in test_rows:
        f = fp(labels[r["mol_id"]]["smiles"])
        if f is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, list(train_fps.values()))
        sims = np.asarray(sims, dtype=float)
        parsed.append((r["mol_id"], float(r["ensemble_std"]), float(r["abs_error"]),
                       sims.max(), np.sort(sims)[-3:].mean(), int((sims >= 0.7).sum())))
    if not parsed:
        sys.exit("no parseable test rows")

    header = ["mol_id", "ensemble_std", "abs_error", "max_train_tanimoto",
              "mean_top3_tanimoto", "n_neighbors_ge0.7"]
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "coverage_per_molecule.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(header)
        for row in sorted(parsed, key=lambda t: -t[1]):
            w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}",
                        f"{row[3]:.4f}", f"{row[4]:.4f}", row[5]])

    n = len(parsed)
    stds = np.array([p[1] for p in parsed])
    maxsim = np.array([p[3] for p in parsed])
    errs = np.array([p[2] for p in parsed])
    inv = 1.0 - maxsim
    rho = float(np.corrcoef(stds, inv)[0, 1])
    from scipy.stats import spearmanr
    rho_s, p_s = spearmanr(stds, inv)
    rho_s_err, p_err = spearmanr(stds, errs)

    print("\n=== ALL TEST (n=%d) ===" % n)
    print(f"Spearman(ensemble_std, 1-max_tanimoto): {rho_s:.3f} (p={p_s:.2e})")
    print(f"Pearson(ensemble_std, 1-max_tanimoto):  {rho:.3f}")
    print(f"Spearman(ensemble_std, abs_error):      {rho_s_err:.3f} (p={p_err:.2e})")

    print("\n=== TOP-15 HIGHEST-STD TEST MOLECULES ===")
    for p in sorted(parsed, key=lambda t: -t[1])[:15]:
        print(f"  {p[0]:<10} std={p[1]:6.3f}  err={p[2]:6.3f}  "
              f"max_sim={p[3]:.3f}  top3_mean={p[4]:.3f}  nn>=0.7: {p[5]}")

    print(f"\ncoverage table -> {csv_path}")
    print("NOTE: run this again with the FULL 200-epoch ensemble aggregate "
          "from Vast for the final report.")


if __name__ == "__main__":
    main()
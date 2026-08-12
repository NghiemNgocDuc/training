"""Gradient-12 signed-error & physicochemical descriptor check.

Part A: signed (pred - true) error analysis for wrong18 / gradient12 / isolated6 /
certain-47 groups from the 5-seed ensemble mean (deep_ensemble/aggregate/per_molecule.csv),
plus per-seed sign stability across the 5 seeds.

Part B: RDKit descriptors (LogP, TPSA, RotBonds, HBD, HBA, MW, Rings, FracCSP3,
formal charge) for all 129 fold-0 test molecules from Data/FreeSolv/database.json.
Mann-Whitney gradient12 vs certain47 (all descriptors reported, none cherry-picked),
then Spearman correlation of every descriptor (plus GMM mean_nll and Tanimoto best_sim
as controls) against SIGNED error and |error| across all 129 molecules.

Part C: summary report, JSON stats, and scatter plot saved to
deep_ensemble/gradient12_descriptor_check/.

NOTE: p-values are NOT multiple-testing corrected (8+ tests); treat as
hypothesis-generating, consistent with prior analyses.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "gradient12_descriptor_check")
AGG_CSV = os.path.join(os.path.dirname(__file__), "aggregate", "per_molecule.csv")
RMSE_CSV = os.path.join(os.path.dirname(__file__), "rmse_analysis", "output", "per_molecule_rmse.csv")
NEIGH_CSV = os.path.join(
    os.path.dirname(__file__), "rmse_analysis", "neighbor_isolation_check",
    "neighbor_similarity_results.csv")
NLL_CSV = os.path.join(os.path.dirname(__file__), "gmm_uncertainty_check", "per_molecule_gmm_nll.csv")
DB_JSON = os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")

SEED_COLS = ["pred_seed42", "pred_seed123", "pred_seed7", "pred_seed2024", "pred_seed999"]

DESCRIPTOR_FUNCS = {
    "logp_crippen": lambda m: Crippen.MolLogP(m),
    "tpsa": lambda m: rdMolDescriptors.CalcTPSA(m),
    "rotatable_bonds": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    "h_bond_donors": lambda m: rdMolDescriptors.CalcNumHBD(m),
    "h_bond_acceptors": lambda m: rdMolDescriptors.CalcNumHBA(m),
    "molecular_weight": lambda m: Descriptors.MolWt(m),
    "num_rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "fraction_csp3": lambda m: Descriptors.FractionCSP3(m),
    "formal_charge": lambda m: float(sum(a.GetFormalCharge() for a in m.GetAtoms())),
}


def load_data():
    df = pd.read_csv(AGG_CSV)
    rmse = pd.read_csv(RMSE_CSV)
    neigh = pd.read_csv(NEIGH_CSV)
    with open(DB_JSON) as f:
        db = json.load(f)
    nll = pd.read_csv(NLL_CSV)[["mol_id", "mean_nll"]]
    df["smiles"] = df["mol_id"].map(lambda m: db.get(m, {}).get("smiles", None))
    df["mean_nll"] = df["mol_id"].map(dict(zip(nll["mol_id"], nll["mean_nll"])))
    best_sim = neigh.set_index("mol_id")["best_sim"]
    df["best_sim"] = df["mol_id"].map(best_sim)
    isolated6 = set(neigh[neigh["group"] == "confidently_wrong"]
                    .sort_values("best_sim").head(6)["mol_id"])
    wrong18 = set(rmse.loc[rmse["quadrant_label"] == "low_std_high_rmse", "mol_id"])
    certain47 = set(rmse.loc[rmse["quadrant_label"] == "low_std_low_rmse", "mol_id"])
    grad12 = sorted(wrong18 - isolated6)
    df["group"] = df["mol_id"].map(
        lambda m: "gradient12" if m in grad12 else
        "isolated6" if m in isolated6 else
        "certain47" if m in certain47 else "other")
    df["signed_error"] = df["ensemble_mean"] - df["true_value"]
    for c in SEED_COLS:
        df[f"signed_{c}"] = df[c] - df["true_value"]
    return df, {"gradient12": grad12, "isolated6": sorted(isolated6),
                "wrong18": sorted(wrong18), "certain47": sorted(certain47)}


def part_a_sign_analysis(df, groups):
    out = {}
    for gname, mids in groups.items():
        sub = df[df["mol_id"].isin(mids)]
        se = sub["signed_error"].values
        n_over = int((se < 0).sum())
        n_under = int((se > 0).sum())
        n_zero = int((se == 0).sum())
        k = max(n_over, n_under)
        n = n_over + n_under
        binom_p = stats.binomtest(k, n, 0.5).pvalue if n > 0 else np.nan
        wilcox = stats.wilcoxon(se, alternative="two-sided") if len(se) >= 5 else None
        out[gname] = {
            "n": len(sub), "n_over_predict": n_over, "n_under_predict": n_under,
            "n_zero": n_zero, "pct_over": round(100.0 * n_over / len(sub), 1) if len(sub) else None,
            "pct_under": round(100.0 * n_under / len(sub), 1) if len(sub) else None,
            "majority_fraction": round(k / n, 3) if n else None,
            "binomial_p_two_sided": float(binom_p),
            "mean_signed_error": float(np.mean(se)),
            "median_signed_error": float(np.median(se)),
            "wilcoxon_vs_zero_p": float(wilcox.pvalue) if wilcox else None,
        }
    per_mol = {}
    for gname, mids in groups.items():
        if gname not in ("gradient12", "isolated6"):
            continue
        sub = df[df["mol_id"].isin(mids)]
        for _, r in sub.iterrows():
            seed_signs = [int(np.sign(r[f"signed_{c}"])) for c in SEED_COLS]
            per_mol[r["mol_id"]] = {
                "group": gname, "signed_error": float(r["signed_error"]),
                "direction": "over" if r["signed_error"] < 0 else "under",
                "seed_signs": seed_signs,
                "n_seeds_matching_majority_sign": max(
                    Counter(seed_signs).most_common()[0][1], 1) if seed_signs else None,
            }
    out["per_molecule"] = per_mol
    return out


def compute_descriptors(df):
    desc = pd.DataFrame({"mol_id": df["mol_id"]})
    failed = []
    for name, fn in DESCRIPTOR_FUNCS.items():
        vals = []
        for smi in df["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                vals.append(np.nan)
            else:
                vals.append(fn(mol))
        desc[name] = vals
    desc = desc.merge(df[["mol_id", "group", "signed_error", "abs_error",
                          "mean_nll", "best_sim"]], on="mol_id", how="left")
    return desc, failed


def part_b_stats(desc, groups):
    g12 = desc[desc["group"] == "gradient12"]
    c47 = desc[desc["group"] == "certain47"]
    stats_out = {"mannwhitney_gradient12_vs_certain47": {},
                 "spearman_vs_error_all_129": {}}
    for name in DESCRIPTOR_FUNCS:
        a = g12[name].dropna().values
        b = c47[name].dropna().values
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        stats_out["mannwhitney_gradient12_vs_certain47"][name] = {
            "median_gradient12": float(np.median(a)), "median_certain47": float(np.median(b)),
            "mean_gradient12": float(np.mean(a)), "mean_certain47": float(np.mean(b)),
            "u": float(u), "p": float(p),
            "significant_p_lt_0.05": bool(p < 0.05), "near_significant_p_lt_0.10": bool(p < 0.10),
        }
    correlate_cols = list(DESCRIPTOR_FUNCS) + ["mean_nll", "best_sim"]
    for name in correlate_cols:
        col = desc[name]
        valid = col.notna() & desc["signed_error"].notna()
        constant = col[valid].nunique(dropna=True) <= 1 if valid.sum() > 0 else True
        if constant:
            stats_out["spearman_vs_error_all_129"][name] = {
                "n": int(valid.sum()), "constant": True,
                "rho_vs_signed_error": None, "p_vs_signed_error": None,
                "rho_vs_abs_error": None, "p_vs_abs_error": None,
                "significant_vs_signed_p_lt_0.05": False,
                "significant_vs_abs_p_lt_0.05": False}
            continue
        rho_signed, p_signed = stats.spearmanr(col[valid], desc.loc[valid, "signed_error"])
        valid_a = col.notna() & desc["abs_error"].notna()
        rho_abs, p_abs = stats.spearmanr(col[valid_a], desc.loc[valid_a, "abs_error"])
        stats_out["spearman_vs_error_all_129"][name] = {
            "n": int(valid.sum()), "constant": False,
            "rho_vs_signed_error": float(rho_signed), "p_vs_signed_error": float(p_signed),
            "rho_vs_abs_error": float(rho_abs), "p_vs_abs_error": float(p_abs),
            "significant_vs_signed_p_lt_0.05": bool(p_signed < 0.05),
            "significant_vs_abs_p_lt_0.05": bool(p_abs < 0.05),
        }
    return stats_out


def write_report(df, sign_a, part_b, top_desc):
    lines = [
        "# Gradient-12 signed-error & physicochemical descriptor check",
        "",
        "p-values are **NOT multiple-testing corrected** (9 descriptors + 2 controls); "
        "treat everything here as hypothesis-generating, consistent with prior analyses.",
        "",
        "Signed error convention: `ensemble_mean - true_value` (kcal/mol).",
        "`signed_error < 0` = prediction MORE negative than experiment = OVER-prediction "
        "(model says solvation is more favorable than measured).",
        "`signed_error > 0` = UNDER-prediction.",
        "",
        "## Groups",
        f"- wrong18 (low_std_high_rmse): {sign_a['wrong18']['n']} | "
        f"gradient12 = wrong18 - isolated6: {sign_a['gradient12']['n']} | "
        f"isolated6: {sign_a['isolated6']['n']} | certain47: {sign_a['certain47']['n']}",
        "",
        "## Part A - sign of errors",
        "",
        "| group | n | over | under | %over | %under | majority frac | binomial p | mean signed | med signed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in ("gradient12", "isolated6", "wrong18", "certain47"):
        s = sign_a[g]
        lines.append(
            f"| {g} | {s['n']} | {s['n_over_predict']} | {s['n_under_predict']} | "
            f"{s['pct_over']} | {s['pct_under']} | {s['majority_fraction']} | "
            f"{s['binomial_p_two_sided']:.4f} | {s['mean_signed_error']:.3f} | "
            f"{s['median_signed_error']:.3f} |")
    lines += [
        "",
        "### Per-molecule (gradient-12 & isolated-6) with per-seed sign stability",
        "",
        "| mol_id | group | signed err | direction | seed signs (42,123,7,2024,999) |",
        "|---|---|---|---|---|",
    ]
    for mid, info in sign_a["per_molecule"].items():
        sgn = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in info["seed_signs"])
        lines.append(
            f"| {mid} | {info['group']} | {info['signed_error']:+.3f} | {info['direction']} | {sgn} |")
    lines += ["", "## Part B - descriptors (Mann-Whitney gradient12 vs certain47)",
              "", "| descriptor | med g12 | med c47 | mean g12 | mean c47 | U | p |",
              "|---|---|---|---|---|---|---|"]
    for name, s in part_b["mannwhitney_gradient12_vs_certain47"].items():
        mark = " **" if s["significant_p_lt_0.05"] else (" *" if s["near_significant_p_lt_0.10"] else "")
        lines.append(
            f"| {name} | {s['median_gradient12']:.3f} | {s['median_certain47']:.3f} | "
            f"{s['mean_gradient12']:.3f} | {s['mean_certain47']:.3f} | "
            f"{s['u']:.1f} | {s['p']:.4f}{mark} |")
    lines += ["", "## Part B - Spearman across ALL 129 test molecules (continuous check)",
              "", "| variable | n | rho vs signed | p vs signed | rho vs |err| | p vs |err| |",
              "|---|---|---|---|---|---|"]
    for name, s in part_b["spearman_vs_error_all_129"].items():
        if s.get("constant"):
            lines.append(f"| {name} | {s['n']} | constant (no variance) | - | - | - |")
            continue
        mark = " **" if s["significant_vs_signed_p_lt_0.05"] else ""
        lines.append(
            f"| {name} | {s['n']} | {s['rho_vs_signed_error']:.3f} | "
            f"{s['p_vs_signed_error']:.4f}{mark} | {s['rho_vs_abs_error']:.3f} | "
            f"{s['p_vs_abs_error']:.4f} |")
    lines += [
        "",
        "## Part C - headline",
        f"- Most-correlated continuous descriptor: **{top_desc}** "
        "(see scatter_signed_error_vs_descriptor.png).",
    ]
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))


def make_plot(df, top_desc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gradient12": "tab:red", "isolated6": "tab:orange",
              "certain47": "tab:blue", "other": "lightgray"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for gname, colr in colors.items():
        sub = df[df["group"] == gname]
        axs[0].scatter(sub[top_desc], sub["signed_error"], c=colr, s=28,
                       alpha=0.85, edgecolor="none", label=gname, zorder=3 if gname != "other" else 1)
        axs[1].scatter(sub[top_desc], sub["abs_error"], c=colr, s=28,
                       alpha=0.85, edgecolor="none", label=gname, zorder=3 if gname != "other" else 1)
    for ax, ylab in ((axs[0], "signed error (pred - true), kcal/mol"),
                     (axs[1], "|signed error|, kcal/mol")):
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel(top_desc)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Gradient-12 vs certain-47: {top_desc}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scatter_signed_error_vs_descriptor.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, groups = load_data()
    if len(df) != 129:
        print(f"[warn] per_molecule.csv has {len(df)} rows, expected 129")
    sign_a = part_a_sign_analysis(df, groups)
    desc, failed = compute_descriptors(df)
    if failed:
        print(f"[warn] descriptors failed for: {failed}")
    part_b = part_b_stats(desc, groups)
    mwu = part_b["mannwhitney_gradient12_vs_certain47"]
    sig_mwu = [k for k, v in mwu.items() if v["significant_p_lt_0.05"] or v["near_significant_p_lt_0.10"]]
    spe = part_b["spearman_vs_error_all_129"]
    candidates = [k for k in sig_mwu if k in spe]
    if candidates:
        top_desc = max(candidates, key=lambda k: abs(spe[k]["rho_vs_signed_error"]))
    else:
        top_desc = max(spe, key=lambda k: abs(spe[k]["rho_vs_signed_error"] or 0.0) if k in DESCRIPTOR_FUNCS else 0)
    full = desc.merge(df[["mol_id", "smiles", "ensemble_mean", "true_value"]], on="mol_id", how="left")
    full.to_csv(os.path.join(OUT_DIR, "descriptors_all_129.csv"), index=False)
    with open(os.path.join(OUT_DIR, "sign_analysis.json"), "w") as f:
        json.dump(sign_a, f, indent=2)
    with open(os.path.join(OUT_DIR, "descriptor_stats.json"), "w") as f:
        json.dump(part_b, f, indent=2)
    write_report(df, sign_a, part_b, top_desc)
    make_plot(full, top_desc)
    print(f"[desc] {len(df)} molecules, groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    g = sign_a["gradient12"]
    print(f"[desc] gradient12 signed: over={g['n_over_predict']}/{g['n']} "
          f"under={g['n_under_predict']}/{g['n']} binomial_p={g['binomial_p_two_sided']:.4f} "
          f"median={g['median_signed_error']:+.3f}")
    print(f"[desc] MWU sig/near-sig: {sig_mwu}")
    for name, s in sorted(spe.items()):
        if s.get("constant"):
            print(f"[desc] spearman {name:16s} constant (no variance)")
            continue
        print(f"[desc] spearman {name:16s} signed rho={s['rho_vs_signed_error']:+.3f} "
              f"p={s['p_vs_signed_error']:.4f} | abs rho={s['rho_vs_abs_error']:+.3f} "
              f"p={s['p_vs_abs_error']:.4f}")
    print(f"[desc] top descriptor for plot: {top_desc}")
    print(f"[desc] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

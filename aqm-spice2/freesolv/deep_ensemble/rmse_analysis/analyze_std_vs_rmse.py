"""Per-molecule seed-RMSE vs ensemble-std analysis for the fold-0 deep ensemble.

Input : ../aggregate/per_molecule.csv (129 rows, fold-0 test set, NO header)
        columns: mol_id, pred_seed42, pred_seed123, pred_seed7, pred_seed2024,
                 pred_seed999, ensemble_mean, ensemble_std, true_value,
                 abs_error, has_halogen_Br_I
Output: <this dir>/output/per_molecule_rmse.csv      (mol_id, rmse_across_seeds, quadrant_label)
        <this dir>/output/std_vs_rmse_scatter.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, fisher_exact

SEED_COLS = ["pred_seed42", "pred_seed123", "pred_seed7", "pred_seed2024", "pred_seed999"]
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "aggregate", "per_molecule.csv")
OUT = os.path.join(HERE, "output")

COLS = ["mol_id", *SEED_COLS, "ensemble_mean", "ensemble_std",
        "true_value", "abs_error", "has_halogen_Br_I"]
raw = pd.read_csv(SRC, header=None, dtype=str)
if raw.iloc[0, 0] == "mol_id":
    raw = raw.iloc[1:]
df = raw.copy()
df.columns = COLS
for c in COLS[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if df["has_halogen_Br_I"].isna().any():
    raise SystemExit("bad parse: some rows failed to convert to numeric")
df["mol_id"] = df["mol_id"].str.strip()
df["has_halogen_Br_I"] = df["has_halogen_Br_I"].astype(int)

df["rmse_across_seeds"] = np.sqrt(
    ((df[SEED_COLS].values - df["true_value"].values[:, None]) ** 2).mean(axis=1))

std_med = df["ensemble_std"].median()
rmse_med = df["rmse_across_seeds"].median()


def quadrant(std, rmse):
    if std <= std_med and rmse <= rmse_med:
        return "low_std_low_rmse"
    if std <= std_med and rmse > rmse_med:
        return "low_std_high_rmse"
    if std > std_med and rmse <= rmse_med:
        return "high_std_low_rmse"
    return "high_std_high_rmse"


df["quadrant_label"] = [quadrant(s, r) for s, r in zip(df["ensemble_std"], df["rmse_across_seeds"])]

print("=" * 78)
print(f"rows={len(df)}  |  std median={std_med:.4f} kcal/mol  |  rmse across seeds median={rmse_med:.4f} kcal/mol")
print("quadrant split uses '<= median' -> low, '> median' -> high on both axes")
print("=" * 78)

print("\n--- 2. Quadrant cross-tabulation (counts, % of 129) ---")
quad_order = ["low_std_low_rmse", "low_std_high_rmse", "high_std_low_rmse", "high_std_high_rmse"]
tab = df["quadrant_label"].value_counts().reindex(quad_order)
for q in quad_order:
    print(f"  {q:20s} : {int(tab[q]):3d}  ({100 * tab[q] / len(df):5.1f}%)")

conf_wrong = df[df["quadrant_label"] == "low_std_high_rmse"].sort_values("rmse_across_seeds", ascending=False)
print(f"\n--- CONFIDENTLY WRONG (low std, high rmse): n={len(conf_wrong)} ---")
print(conf_wrong[["mol_id", "ensemble_std", "rmse_across_seeds", "true_value", "abs_error", "has_halogen_Br_I"]].to_string(index=False))

print("\n--- 3. Spearman: ensemble_std vs per-molecule rmse_across_seeds ---")
rho_rmse, p_rmse = spearmanr(df["ensemble_std"], df["rmse_across_seeds"])
rho_abs, p_abs = spearmanr(df["ensemble_std"], df["abs_error"])
print(f"  std vs rmse_across_seeds : rho={rho_rmse:.3f}  p={p_rmse:.3g}")
print(f"  std vs abs_error          : rho={rho_abs:.3f}  p={p_abs:.3g}   (prior report: rho=0.496)")

print("\n--- 4. Halogen overrepresentation ---")
n_halo = int(df["has_halogen_Br_I"].sum())
n_conf = int(conf_wrong["has_halogen_Br_I"].sum())
base_pct = 100 * n_halo / len(df)
conf_pct = 100 * n_conf / len(conf_wrong)
table = [[n_conf, len(conf_wrong) - n_conf],
         [n_halo - n_conf, (len(df) - len(conf_wrong)) - (n_halo - n_conf)]]
odds, p_fish = fisher_exact(table)
print(f"  halogen base rate        : {n_halo:3d}/{len(df)} = {base_pct:5.1f}%")
print(f"  halogen in conf-wrong    : {n_conf:3d}/{len(conf_wrong)} = {conf_pct:5.1f}%")
print(f"  fisher_exact odds ratio={odds:.2f}  p={p_fish:.3f}")

os.makedirs(OUT, exist_ok=True)
df[["mol_id", "rmse_across_seeds", "quadrant_label"]].to_csv(
    os.path.join(OUT, "per_molecule_rmse.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 6.5))
for flag, color, label in [(0, "#1f77b4", "no Br/I"), (1, "#d62728", "has Br/I")]:
    sub = df[df["has_halogen_Br_I"] == flag]
    ax.scatter(sub["ensemble_std"], sub["rmse_across_seeds"],
               c=color, s=42, alpha=0.8, edgecolors="none", label=label)
ax.axvline(std_med, ls="--", c="gray", lw=1)
ax.axhline(rmse_med, ls="--", c="gray", lw=1)
ax.annotate("CONFIDENTLY WRONG\n(low std, high rmse)", xy=(0.02, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=9, color="#d62728", fontweight="bold")
ax.annotate(f"std median = {std_med:.3f}", xy=(std_med, ax.get_ylim()[1]), xytext=(8, 8),
            textcoords="offset points", fontsize=8, color="gray")
ax.annotate(f"rmse median = {rmse_med:.3f}", xy=(1.0, rmse_med), xytext=(8, -14),
            textcoords="offset points", fontsize=8, color="gray", ha="right")
ax.set_xlabel("ensemble_std (kcal/mol) â€” disagreement across 5 seeds")
ax.set_ylabel("per-molecule RMSE across 5 seeds (kcal/mol)")
ax.set_title(f"Fold-0 deep ensemble: seed RMSE vs disagreement (n={len(df)}, "
             f"spearman rho={rho_rmse:.3f})")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "std_vs_rmse_scatter.png"), dpi=150)
print(f"\n--- 5. artifacts written ---")
print(f"  {os.path.join(OUT, 'per_molecule_rmse.csv')}")
print(f"  {os.path.join(OUT, 'std_vs_rmse_scatter.png')}")

print("\n--- 6. Plain-English summary ---")
print(f"Of {len(df)} fold-0 test molecules, {int(tab['high_std_high_rmse'])} ({100 * tab['high_std_high_rmse'] / len(df):.0f}%) are BOTH high-disagreement and high-RMSE "
      f"(disagreement correctly flags them), while {len(conf_wrong)} ({100 * len(conf_wrong) / len(df):.0f}%) fall in the 'confidently wrong' quadrant "
      f"where the ensemble AGREES (std below median {std_med:.2f}) yet is uniformly wrong (seed-RMSE above median {rmse_med:.2f} kcal/mol) - "
      f"these are the blind spot of disagreement-based uncertainty.")
print(f"Spearman std-vs-seedRMSE = {rho_rmse:.3f} (p={p_rmse:.2g}) vs std-vs-abs_error {rho_abs:.3f} (prior report 0.496) - the seed-RMSE target raises rho by +{rho_rmse - rho_abs:.3f}, "
      f"but note rmse_across_seeds mechanically embeds the seed spread (rmse^2 = bias^2 + std^2*0.8), so part of that lift is by construction. "
      f"Either way the blind spot stands: disagreement catches ~{100 * tab['high_std_high_rmse'] / len(df):.0f}% of bad molecules but misses ~{100 * len(conf_wrong) / len(df):.0f}% "
      f"that are quietly wrong.")
print(f"Halogens: {n_halo}/{len(df)} ({base_pct:.1f}%) base rate vs {n_conf}/{len(conf_wrong)} ({conf_pct:.1f}%) in the confidently-wrong quadrant"
      + (f" - overrepresented (Fisher odds {odds:.2f}, p={p_fish:.3f})." if p_fish < 0.05 and n_conf > 0 else " - not a statistically significant overrepresentation."))

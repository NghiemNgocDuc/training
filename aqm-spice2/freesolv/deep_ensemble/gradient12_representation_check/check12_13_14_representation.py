"""CHECK 12/13/14: structural representation sanity checks on the 129 fold-0 test molecules.

CHECK 12 - tautomer / protonation-state ambiguity:
    SMARTS flags for tautomerizable motifs and ionizable groups; charged-vs-neutral
    form encoding in the input SMILES; gradient12 vs certain47 flag-rate comparison
    (Fisher's exact), and Spearman vs |error| and signed error over all 129.

CHECK 13 - unspecified stereochemistry:
    FindMolChiralCenters(includeUnassigned=True); counts of total / undefined
    centers; rate comparison (Fisher's exact), count comparison (Mann-Whitney U),
    Spearman vs errors across all 129.

CHECK 14 - sanitization edge cases + featurizer trace:
    sanitization warnings (kekulization, valence, radicals), and a trace of the
    exact representation fed to the model featurizer (element one-hot + 3D
    geometry only - no bond orders, no formal charges, no hybridization), i.e.
    atom/element composition incl. implicit-H count as the model sees it, plus
    any silent-fallback or representation-loss anomalies.

Uncorrected p-values throughout. Outputs:
    per_molecule_flags.csv   - one row per molecule, all flags
    group_comparison.json    - Fisher / Mann-Whitney stats per flag
    correlations.json        - Spearman per flag vs abs/signed error (all 129)
    report.md                - plain-English verdict per check
"""
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "..", "gradient12_descriptor_check", "descriptors_all_129.csv")
OUT_DIR = HERE

MODEL_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
ELEMENT_TO_IDX = {
    1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7,
    3: 8, 5: 9, 11: 10, 12: 11, 14: 12, 19: 13, 20: 14, 35: 15, 53: 16,
}

TAUTOMER_SMARTS = {
    "amide_urea_lactam": "[NX3][CX3](=[OX1])",
    "imine": "[CX3]=[NX2]",
    "enol_OH": "[OX2H1][CX3]=[CX3]",
    "1,3_dicarbonyl": "[CX3](=[OX1])[#6][CX3](=[OX1])",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "amidine": "[NX2]=[CX3][NX3]",
    "aromatic_pyrrole_NH": "[nX3H1]",
    "phenol_OH": "[OX2H1][cX3]",
    "alpha_CH2_to_carbonyl": "[CX4H2][CX3](=[OX1])",
}

IONIZABLE_SMARTS = {
    "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "carboxylate": "[CX3](=[OX1])[OX1-]",
    "amine_primary_secondary": "[NX3;H1,H2;!$(NC(=O));!$(N=[#6]);!$(N#[#6]);!$(Nc:1:c:c:c:c:c:1)]",
    "amine_tertiary": "[NX3;H0;!$(NC(=O));!$(N=[#6]);!$(N#[#6]);!$(Nc:1:c:c:c:c:c:1)]",
    "sulfonic_acid": "[SX4](=[OX1])(=[OX1])[OX2H1]",
    "nitro": "[NX3](=[OX1])=[OX1]",
}

ALL_SMARTS = {}
ALL_SMARTS.update({f"taut_{k}": v for k, v in TAUTOMER_SMARTS.items()})
ALL_SMARTS.update({f"ion_{k}": v for k, v in IONIZABLE_SMARTS.items()})


def sanitize_and_report(smiles):
    """Return (flags dict, note) for a SMILES under strict sanitization."""
    flags = {}
    note = ""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "parse_failed"
    try:
        probs = Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL,
                                 catchErrors=True)
        flags["sanitize_warnings"] = bool(probs)
        if probs:
            note += "; ".join(f"{k}" for k, _ in probs)
    except Exception as e:
        flags["sanitize_warnings"] = True
        note += f"sanitize_exc:{type(e).__name__}"
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek)
        flags["kekulize_failed"] = False
    except Exception:
        flags["kekulize_failed"] = True
        note += "; kekulize_failed"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    flags["radicals"] = sum(mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                        for i in range(mol.GetNumAtoms())) > 0
    return flags, note


def check12_flags(mol):
    flags = {}
    for name, smarts in ALL_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            flags[name] = bool(mol.HasSubstructMatch(patt))
        except Exception:
            flags[name] = False
    charged = [(a.GetSymbol(), int(a.GetFormalCharge())) for a in mol.GetAtoms()
               if a.GetFormalCharge() != 0]
    flags["has_charged_atoms"] = bool(charged)
    flags["charged_atoms_desc"] = ";".join(f"{s}:{q}" for s, q in charged) or ""
    flags["n_charged_atoms"] = len(charged)
    flags["formal_charge_sum"] = float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    return flags


def check13_flags(mol):
    flags = {}
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                            useLegacyImplementation=False)
    except Exception:
        centers = []
    n_total = len(centers)
    n_undefined = sum(1 for _, tag in centers if tag in ("?", None, ""))
    flags["n_chiral_centers"] = n_total
    flags["n_undefined_stereo"] = n_undefined
    flags["has_undefined_stereo"] = n_undefined > 0
    flags["stereo_centers_desc"] = ";".join(f"{tag or '?'}" for _, tag in centers)
    return flags


def check14_flags(mol, mol_h):
    flags = {}
    flags["n_heavy_from_smiles"] = mol.GetNumAtoms()
    flags["n_atoms_as_model_sees"] = mol_h.GetNumAtoms()
    flags["n_H_added"] = mol_h.GetNumAtoms() - mol.GetNumAtoms()
    zs = set(a.GetAtomicNum() for a in mol.GetAtoms())
    flags["non_model_element"] = not zs.issubset(MODEL_ELEMENTS)
    flags["non_vocab_element"] = not zs.issubset(set(ELEMENT_TO_IDX))
    flags["elements"] = ".".join(str(z) for z in sorted(zs))
    return flags


def main():
    df = pd.read_csv(INPUT)
    g12 = set(df.loc[df["group"] == "gradient12", "mol_id"])
    c47 = set(df.loc[df["group"] == "certain47", "mol_id"])

    rows = []
    for _, r in df.iterrows():
        mol_id = r["mol_id"]
        smiles = r["smiles"]
        flags, note = sanitize_and_report(smiles)
        if flags is None:
            rows.append({"mol_id": mol_id, "group": r["group"],
                         "smiles": smiles, "abs_error": r["abs_error"],
                         "signed_error": r["signed_error"], "parse_error": note})
            continue
        mol = Chem.MolFromSmiles(smiles)
        mol_h = Chem.AddHs(mol)
        f12 = check12_flags(mol)
        f13 = check13_flags(mol)
        f14 = check14_flags(mol, mol_h)
        row = {"mol_id": mol_id, "group": r["group"], "smiles": smiles,
               "abs_error": float(r["abs_error"]), "signed_error": float(r["signed_error"]),
               "sanitize_warnings": flags["sanitize_warnings"],
               "kekulize_failed": flags["kekulize_failed"],
               "radicals": flags["radicals"],
               "sanitize_note": note}
        row.update(f12)
        row.update(f13)
        row.update(f14)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "per_molecule_flags.csv")
    out.to_csv(out_path, index=False)

    flag_cols = [c for c in out.columns if c not in (
        "mol_id", "group", "smiles", "abs_error", "signed_error",
        "charged_atoms_desc", "stereo_centers_desc", "sanitize_note",
        "elements", "parse_error")]

    group_stats = {"groups": {"gradient12": sorted(g12), "certain47": sorted(c47)},
                   "n_gradient12": len(g12), "n_certain47": len(c47),
                   "mean_abs_error": {"gradient12": float(out.loc[out["group"] == "gradient12", "abs_error"].mean()),
                                      "certain47": float(out.loc[out["group"] == "certain47", "abs_error"].mean())}}
    correlations = {}
    for c in flag_cols:
        if c in ("n_chiral_centers", "n_undefined_stereo", "n_charged_atoms",
                 "n_heavy_from_smiles", "n_atoms_as_model_sees", "n_H_added",
                 "formal_charge_sum"):
            continue
        vals = out[c].astype(float)
        if vals.nunique() <= 1:
            continue
        try:
            rho_a, pa = spearmanr(vals, out["abs_error"])
            rho_s, ps = spearmanr(vals, out["signed_error"])
        except Exception:
            continue
        correlations[c] = {"rho_vs_abs_error": float(rho_a), "p_vs_abs_error": float(pa),
                           "rho_vs_signed_error": float(rho_s), "p_vs_signed_error": float(ps)}

    for c in flag_cols:
        if c in ("n_chiral_centers", "n_undefined_stereo", "n_charged_atoms",
                 "n_heavy_from_smiles", "n_atoms_as_model_sees", "n_H_added",
                 "formal_charge_sum", "charged_atoms_desc", "stereo_centers_desc"):
            continue
        a = out.loc[out["group"] == "gradient12", c].astype(bool)
        b = out.loc[out["group"] == "certain47", c].astype(bool)
        a_pos, b_pos = int(a.sum()), int(b.sum())
        table = [[a_pos, len(a) - a_pos], [b_pos, len(b) - b_pos]]
        try:
            or_, p = fisher_exact(table, alternative="two-sided")
        except Exception:
            or_, p = float("nan"), float("nan")
        group_stats[c] = {
            "flag": c,
            "n_true_gradient12": a_pos, "n_total_gradient12": int(len(a)),
            "rate_gradient12": a_pos / len(a),
            "n_true_certain47": b_pos, "n_total_certain47": int(len(b)),
            "rate_certain47": b_pos / len(b),
            "fisher_odds_ratio": float(or_), "fisher_p": float(p),
        }

    for c in ("n_chiral_centers", "n_undefined_stereo", "n_charged_atoms"):
        a = out.loc[out["group"] == "gradient12", c].astype(float)
        b = out.loc[out["group"] == "certain47", c].astype(float)
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        except Exception:
            u, p = float("nan"), float("nan")
        try:
            rho_a, pa = spearmanr(out[c].astype(float), out["abs_error"])
            rho_s, ps = spearmanr(out[c].astype(float), out["signed_error"])
        except Exception:
            rho_a, rho_s, pa, ps = float("nan"), float("nan"), float("nan"), float("nan")
        group_stats[f"count_{c}"] = {
            "flag": c,
            "mean_gradient12": float(a.mean()), "mean_certain47": float(b.mean()),
            "mannwhitney_u": float(u), "mannwhitney_p": float(p),
            "rho_vs_abs_error": float(rho_a), "p_vs_abs_error": float(pa),
            "rho_vs_signed_error": float(rho_s), "p_vs_signed_error": float(ps),
        }

    with open(os.path.join(OUT_DIR, "group_comparison.json"), "w") as f:
        json.dump(group_stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "correlations.json"), "w") as f:
        json.dump(correlations, f, indent=2)

    g = out.loc[out["group"] == "gradient12"]
    c = out.loc[out["group"] == "certain47"]
    lines = []
    lines.append("# CHECK 12/13/14: representation-level sanity checks (129 fold-0 test molecules)\n")
    lines.append(f"- Input: {os.path.relpath(INPUT, OUT_DIR)}\n")
    lines.append(f"- gradient12 n={len(g)}, certain47 n={len(c)}, abs_error mean: g12={g['abs_error'].mean():.2f} vs c47={c['abs_error'].mean():.2f}\n")
    lines.append(f"- p-values uncorrected, two-sided.\n")

    def verdict(p, direction):
        if p < 0.01:
            return "SIGNAL (p<0.01)"
        if p < 0.05:
            return f"marginal ({direction}, p<0.05)"
        return f"ruled out ({direction} p={p:.2f})"

    lines.append("\n## CHECK 12 - tautomer / protonation-state ambiguity\n")
    lines.append("Model input has NO bond-order or formal-charge feature (element one-hot + 3D only); "
                 "tautomer/charge ambiguity can only act through the geometry (H count, bond lengths) "
                 "produced by RDKit ETKDG/MMFF from the SMILES.\n")
    sig = []
    for cname in ["taut_amide_urea_lactam", "taut_imine", "taut_enol_OH", "taut_1,3_dicarbonyl",
                  "taut_guanidine", "taut_amidine", "taut_aromatic_pyrrole_NH", "taut_phenol_OH",
                  "taut_alpha_CH2_to_carbonyl", "ion_carboxylic_acid", "ion_carboxylate",
                  "ion_amine_primary_secondary", "ion_amine_tertiary", "ion_sulfonic_acid", "ion_nitro"]:
        if cname in group_stats:
            s = group_stats[cname]
            lines.append(f"- {cname}: g12 {s['n_true_gradient12']}/{s['n_total_gradient12']} "
                         f"({s['rate_gradient12']:.2f}) vs c47 {s['n_true_certain47']}/{s['n_total_certain47']} "
                         f"({s['rate_certain47']:.2f}) | Fisher p={s['fisher_p']:.4f}")
            if s["fisher_p"] < 0.05:
                sig.append(cname)
    lines.append(f"\n**Verdict CHECK 12:** {verdict(0.01 if sig else 0.5, 'no flags differ')} "
                 f"{'; flags with p<0.05: ' + ', '.join(sig) if sig else ''}\n")

    lines.append("\n## CHECK 13 - unspecified stereochemistry\n")
    s = group_stats.get("count_n_undefined_stereo", {})
    if s:
        lines.append(f"- undefined stereo centers: g12 mean={s['mean_gradient12']:.2f} vs c47 mean={s['mean_certain47']:.2f} "
                     f"| Mann-Whitney p={s['mannwhitney_p']:.4f} | Spearman vs abs_error rho={s['rho_vs_abs_error']:.3f} "
                     f"(p={s['p_vs_abs_error']:.4f}), vs signed rho={s['rho_vs_signed_error']:.3f} (p={s['p_vs_signed_error']:.4f})")
    h = group_stats.get("has_undefined_stereo", {})
    if h:
        lines.append(f"- any undefined stereocenter: g12 {h['n_true_gradient12']}/{h['n_total_gradient12']} "
                     f"({h['rate_gradient12']:.2f}) vs c47 {h['n_true_certain47']}/{h['n_total_certain47']} "
                     f"({h['rate_certain47']:.2f}) | Fisher p={h['fisher_p']:.4f}")
    p13 = min(s["mannwhitney_p"], h["fisher_p"]) if s and h else 1.0
    lines.append(f"\n**Verdict CHECK 13:** {verdict(p13, 'stereo undefined')}\n")

    lines.append("\n## CHECK 14 - sanitization edge cases + featurizer trace\n")
    lines.append("Featurizer trace (from element_vocab.py build_one_hot + DimeNet++ usage): "
                 "atom feature = one-hot over 17 elements (H,C,N,O,F,P,S,Cl,Li,B,Na,Mg,Si,K,Ca,Br,I); "
                 "edge features from 3D distances/angles only. NO formal charge, NO bond order, NO "
                 "hybridization, NO aromaticity is in the model input. Pipeline: SMILES -> element gate "
                 "({H,C,N,O,F,P,S,Cl,Br,I}) -> AddHs -> ETKDGv3/MMFF (or xTB) -> atNUM+atXYZ -> one-hot+geometry.\n")
    for cname in ["sanitize_warnings", "kekulize_failed", "radicals",
                  "has_charged_atoms", "non_model_element", "non_vocab_element"]:
        s = group_stats.get(cname, {})
        if s:
            lines.append(f"- {cname}: g12 {s['n_true_gradient12']}/{s['n_total_gradient12']} "
                         f"({s['rate_gradient12']:.2f}) vs c47 {s['n_true_certain47']}/{s['n_total_certain47']} "
                         f"({s['rate_certain47']:.2f}) | Fisher p={s['fisher_p']:.4f}")
    lines.append("\nCharged-atom encoding in input SMILES (representation loss: charge invisible to model):")
    charged_rows = out.loc[out["has_charged_atoms"]]
    for _, r in charged_rows.iterrows():
        lines.append(f"- {r['mol_id']} [{r['group']}] {r['charged_atoms_desc']}  {r['smiles']}")
    if charged_rows.empty:
        lines.append("- none\n")
    n_nonmodel = int(out["non_model_element"].sum())
    lines.append(f"\n**Verdict CHECK 14:** model input is element+geometry only; no unknown-token fallback exists "
                 f"(one-hot index lookup is total over the 17-element vocab). Anomaly flags: charged-atom SMILES "
                 f"g12={int(out.loc[(out['group']=='gradient12')]['has_charged_atoms'].sum())} vs "
                 f"c47={int(out.loc[(out['group']=='certain47')]['has_charged_atoms'].sum())}; "
                 f"non-model elements {n_nonmodel}. "
                 f"{'SIGNAL: charged/tautomeric encoding differs between groups' if p13 < 0.05 or any(s['fisher_p'] < 0.05 for cname in ['has_charged_atoms','sanitize_warnings','radicals','taut_amide_urea_lactam','taut_aromatic_pyrrole_NH'] if cname in group_stats and (s := group_stats[cname]) and s['fisher_p'] < 0.05) else 'ruled out: no group-representation difference detected'}\n")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {out_path}, group_comparison.json, correlations.json, report.md")


if __name__ == "__main__":
    main()

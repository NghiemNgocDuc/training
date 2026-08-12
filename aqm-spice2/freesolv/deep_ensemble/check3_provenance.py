"""Check 3: data provenance / source heterogeneity for gradient-12.

FreeSolv is compiled from multiple experimental campaigns. This script inspects
the REAL metadata fields present in Data/FreeSolv/database.json (no assumed field
names), then cross-tabs gradient-12 / wrong-18 against source (expt_reference),
functional-group tags, and notes, vs their base rate in the full 643-molecule DB.
Fisher's exact test used for 2x2 tables (small counts)."

Outputs -> deep_ensemble/gradient12_conformer_provenance_check/
"""

import json
import os
from collections import Counter

import pandas as pd
from scipy import stats

_script_dir = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_script_dir, "gradient12_conformer_provenance_check")


def find_repo_root():
    d = _script_dir
    while True:
        if os.path.exists(os.path.join(d, "freesolv_conformers.hdf5")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("repo root (freesolv_conformers.hdf5) not found above script")
        d = parent


DB_CANDIDATES = ("Data/FreeSolv/database.json", "aqm-spice2/Data/FreeSolv/database.json",
                 "aqm-spice2/aqm-spice2/Data/FreeSolv/database.json")
REPO_ROOT = find_repo_root()
DB_JSON = next((os.path.join(REPO_ROOT, rel) for rel in DB_CANDIDATES
                if os.path.exists(os.path.join(REPO_ROOT, rel))), None)
if DB_JSON is None:
    raise SystemExit(f"database.json not found under {REPO_ROOT} (tried {DB_CANDIDATES})")
RMSE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rmse_analysis", "output", "per_molecule_rmse.csv")
NEIGH_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "rmse_analysis", "neighbor_isolation_check",
                         "neighbor_similarity_results.csv")
AGG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "aggregate", "per_molecule.csv")


def load_groups():
    rmse = pd.read_csv(RMSE_CSV)
    neigh = pd.read_csv(NEIGH_CSV)
    isolated6 = set(neigh[neigh["group"] == "confidently_wrong"]
                    .sort_values("best_sim").head(6)["mol_id"])
    wrong18 = set(rmse.loc[rmse["quadrant_label"] == "low_std_high_rmse", "mol_id"])
    certain47 = set(rmse.loc[rmse["quadrant_label"] == "low_std_low_rmse", "mol_id"])
    return sorted(wrong18 - isolated6), sorted(isolated6), sorted(wrong18), sorted(certain47)


def find_seed42_dir():
    cand = os.path.join(_script_dir, "seed_42")
    if os.path.exists(os.path.join(cand, "test_ids.json")):
        return cand
    for levels in (1, 2, 3, 4):
        base = _script_dir
        for _ in range(levels):
            base = os.path.dirname(base)
        cand = os.path.join(base, "deep_ensemble", "seed_42")
        if os.path.exists(os.path.join(cand, "test_ids.json")):
            return cand
    raise SystemExit("seed_42/test_ids.json not found")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grad12, iso6, wrong18, c47 = load_groups()
    with open(DB_JSON) as f:
        db = json.load(f)
    all_ids = list(db.keys())
    print(f"[prov] database.json: {len(all_ids)} records")

    keys = Counter()
    for mid in all_ids:
        keys.update(db[mid].keys())
    print("[prov] metadata fields present (top-level): "
          + ", ".join(f"{k} x{n}" for k, n in keys.most_common()))

    agg = pd.read_csv(AGG_CSV)[["mol_id", "abs_error"]]
    rows = []
    for mid in all_ids:
        e = db[mid]
        rows.append({
            "mol_id": mid,
            "group": ("gradient12" if mid in grad12 else
                      "isolated6" if mid in iso6 else
                      "certain47" if mid in c47 else
                      "wrong18" if mid in wrong18 else "other"),
            "expt_reference": e.get("expt_reference"),
            "calc_reference": e.get("calc_reference"),
            "groups_tag": ", ".join(e.get("groups", []) or []),
            "notes": " | ".join(e.get("notes", []) or []),
            "iupac": e.get("iupac"),
            "expt_kcal": e.get("expt"),
            "abs_error": float(agg.loc[agg["mol_id"] == mid, "abs_error"].iloc[0])
            if mid in set(agg["mol_id"]) else None,
        })
    df = pd.DataFrame(rows)
    n_all = len(all_ids)
    n_g12 = len(grad12)

    report = {"n_all": n_all, "n_gradient12": n_g12,
              "gradient12": grad12, "wrong18": wrong18,
              "fields_present": {k: int(v) for k, v in keys.most_common()}}

    test_ids = set(json.load(open(os.path.join(find_seed42_dir(), "test_ids.json"))))
    n_test = len(test_ids)

    def crosstab(column, min_share=0.02):
        cols = {}
        for mid in all_ids:
            val = db[mid].get(column)
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                if v is None or (isinstance(v, str) and v.strip() in ("", "Not available")):
                    continue
                cols.setdefault(v, {"all": 0, "test": 0, "g12": 0, "wrong18": 0, "iso6": 0})
                cols[v]["all"] += 1
                if mid in test_ids:
                    cols[v]["test"] += 1
                if mid in grad12:
                    cols[v]["g12"] += 1
                if mid in wrong18:
                    cols[v]["wrong18"] += 1
                if mid in iso6:
                    cols[v]["iso6"] += 1
        tab = []
        for v, c in sorted(cols.items(), key=lambda kv: -kv[1]["all"]):
            if c["all"] < max(3, n_all * min_share):
                continue
            tab.append({"value": v, "n_all": c["all"], "n_test": c["test"],
                        "n_gradient12": c["g12"], "n_wrong18": c["wrong18"],
                        "n_iso6": c["iso6"],
                        "g12_share": round(c["g12"] / c["all"], 3),
                        "test_share": round(c["test"] / n_test, 3),
                        "base_rate_g12": round(n_g12 / n_all, 3)})
        return pd.DataFrame(tab)

    fisher_out = {}
    for column in ("expt_reference", "calc_reference", "groups"):
        sub = crosstab(column)
        sub.to_csv(os.path.join(OUT_DIR, f"crosstab_{column}.csv"), index=False)
        row_out = []
        for _, r in sub.iterrows():
            if r["n_all"] < 5:
                continue
            a = r["n_gradient12"]
            b = r["n_all"] - r["n_gradient12"]
            c = n_g12 - r["n_gradient12"]
            d = (n_all - r["n_all"]) - (n_g12 - r["n_gradient12"])
            orr, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            row_out.append({"value": r["value"], "in_group": int(a), "not_in_group": int(b),
                            "n_g12_total": n_g12, "n_all_total": n_all,
                            "odds_ratio": float(orr), "fisher_p": float(p)})
        if row_out:
            fisher_out[column] = row_out
            print(f"[prov] {column}: fisher rows = {len(row_out)}")
            for r in row_out[:8]:
                print(f"    {r['value'][:45]:47s} g12 {r['in_group']}/{r['n_g12_total']} "
                      f"ref {r['not_in_group']}/{r['n_all_total']} "
                      f"OR {r['odds_ratio']:.2f} p={r['fisher_p']:.4f}")

    def flag_stats(fn, label):
        n_g = sum(1 for m in grad12 if fn(db[m]))
        n_t = sum(1 for m in test_ids if fn(db[m]))
        n_all_f = sum(1 for m in all_ids if fn(db[m]))
        orr, p = stats.fisher_exact([[n_g, n_t - n_g], [n_g12 - n_g,
                                   (n_test - n_t) - (n_g12 - n_g)]], alternative="two-sided")
        out = {"n_gradient12": int(n_g), "n_test": int(n_t), "n_all": int(n_all_f),
               "test_base_rate": round(n_t / n_test, 3),
               "gradient12_rate": round(n_g / n_g12, 3),
               "odds_ratio": float(orr), "fisher_p_test_set_null": float(p)}
        print(f"[prov] {label}: g12 {n_g}/{n_g12} | test {n_t}/{n_test} "
              f"({n_t/n_test:.1%}) | full {n_all_f}/{n_all} | "
              f"fisher_p(vs test base) = {p:.4f}")
        return out

    report["flag_default_uncertainty_note"] = flag_stats(
        lambda e: any("Experimental uncertainty not presently available" in n
                      for n in (e.get("notes") or [])), "default-uncertainty note")
    report["flag_expt_uncertainty_missing"] = flag_stats(
        lambda e: str(e.get("d_expt", "")).strip().lower() in ("not available", "", "nan"),
        "d_expt = Not available")
    report["flag_calc_reference_same_as_expt"] = flag_stats(
        lambda e: e.get("expt_reference") == e.get("calc_reference"),
        "expt_ref == calc_ref (value is computed, not measured)")

    g12_notes = df[df["group"] == "gradient12"][
        ["mol_id", "expt_reference", "groups_tag", "notes", "iupac", "abs_error"]]
    g12_notes.to_csv(os.path.join(OUT_DIR, "gradient12_provenance_detail.csv"), index=False)
    report["fisher_tests"] = fisher_out
    report["gradient12_provenance_detail_csv"] = os.path.join(
        OUT_DIR, "gradient12_provenance_detail.csv")

    with open(os.path.join(OUT_DIR, "provenance_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[prov] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

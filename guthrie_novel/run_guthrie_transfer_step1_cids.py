"""Step 1: aggregate Guthrie final-kcal values per SMILES, resolve PubChem CIDs.

Reads the cloned GuthrieSolv CSV (no RDKit needed), means replicate rows,
queries PUG-REST fastidentity/smiles -> CID with throttle+retry, tqdm bars.
Writes guthrie_cids.json: {smiles: {cid, value_kcal, n_rows}}.
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from tqdm import tqdm

GUTH_CSV = r"C:\Users\User\AppData\Local\Temp\opencode\guthrie\guthrie_database.csv"
OUT = r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run\guthrie_cids.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def resolve_cid(smiles, tries=3):
    enc = urllib.parse.quote(smiles, safe="")
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
           f"fastidentity/smiles/{enc}/cids/JSON")
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            cids = d.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
        except Exception as e:
            if t == tries - 1:
                return f"ERR:{type(e).__name__}"
            time.sleep(1.5)
    return None


def main():
    agg = {}
    with open(GUTH_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in tqdm(csv.DictReader(f), desc="scan csv", unit="row",
                        total=53895):
            if row.get("dimension3") != "kcal/mol" or not row.get("mol"):
                continue
            try:
                v = float(row["final"])
            except (ValueError, TypeError):
                continue
            agg.setdefault(row["mol"], []).append(v)
    print(f"SMILES with final-kcal: {len(agg)}", flush=True)

    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
        print(f"resuming: {len(out)} already resolved", flush=True)
    todo = [s for s in agg if s not in out]
    print(f"to resolve: {len(todo)}", flush=True)
    for s in tqdm(todo, desc="PUG-REST CID", unit="mol"):
        vals = agg[s]
        out[s] = {"cid": resolve_cid(s),
                  "value_kcal": sum(vals) / len(vals),
                  "n_rows": len(vals)}
        time.sleep(0.2)  # throttle ~5/s
        if len(out) % 50 == 0:
            json.dump(out, open(OUT, "w"))
    json.dump(out, open(OUT, "w"))
    n_ok = sum(1 for v in out.values() if isinstance(v["cid"], int))
    print(f"done: {len(out)} total, {n_ok} with CID -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

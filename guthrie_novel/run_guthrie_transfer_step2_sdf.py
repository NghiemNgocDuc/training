"""Step 2: exact CID dedup vs FreeSolv, overlap-value check, batch 3D-SDF download.

Writes: guthrie_final.json {cid: value}, guthrie_overlap_check.json,
        guthrie_3d.pkl {cid: {symbols:[...], xyz:[[..]]}} (pure-python SDF parse).
"""
import json
import os
import time
import urllib.request
from tqdm import tqdm

RUN = r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run"
CIDS = os.path.join(RUN, "guthrie_cids.json")
NOV = os.path.join(RUN, "guthrie_novel_approx.json")
OUT_FINAL = os.path.join(RUN, "guthrie_final.json")
OUT_SDF = os.path.join(RUN, "guthrie_3d.pkl")
REPO = r"C:\Users\User\Documents\Data"

cids = json.load(open(CIDS))
nov = json.load(open(NOV))
flab = json.load(open(os.path.join(REPO, r"Data\FreeSolv\database.json")))
fs_cids = {v["PubChemID"] for v in flab.values()}

# exact overlap via CID (all 663, incl string-match overlaps)
cid_of = {s: v["cid"] for s, v in cids.items() if isinstance(v.get("cid"), int)}
overlap_vals = [(s, v["value_kcal"]) for s, v in cids.items()
                if isinstance(v.get("cid"), int) and v["cid"] in fs_cids]
fsexp = {v["PubChemID"]: v["expt"] for v in flab.values()}
diffs = [val - fsexp[cids[s]["cid"]] for s, val in overlap_vals]
import numpy as np
d = np.array(diffs)
print(f"resolved={len(cid_of)} exact-CID overlap={len(d)} "
      f"MAE={np.abs(d).mean():.3f} max={np.abs(d).max():.3f}", flush=True)
json.dump({"n": len(d), "mae": float(np.abs(d).mean()),
           "max": float(np.abs(d).max())}, open(
               os.path.join(RUN, "guthrie_overlap_check.json"), "w"))

final = {}
for s in nov:
    v = cids.get(s)
    if v and isinstance(v.get("cid"), int) and v["cid"] not in fs_cids:
        final[str(v["cid"])] = v["value_kcal"]
print(f"final novel set: {len(final)} molecules", flush=True)
vals = list(final.values())
print(f"value mean {np.mean(vals):.2f} min {min(vals):.2f} max {max(vals):.2f}",
      flush=True)
json.dump(final, open(OUT_FINAL, "w"))


def fetch_sdf(cid_chunk):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           + ",".join(cid_chunk) + "/SDF?record_type=3d")
    req = urllib.request.Request(url, headers={"Accept": "chemical/x-mdl-sdfile"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_sdf(text):
    """Parse concatenated V2000 SDFs -> {cid: (symbols, xyz)} using $$$$ + CID."""
    out = {}
    for block in text.split("$$$$"):
        lines = block.strip().splitlines()
        if len(lines) < 5:
            continue
        cid = None
        for i, ln in enumerate(lines):
            if ln.strip() == "> <PUBCHEM_COMPOUND_CID>" and i + 1 < len(lines):
                cid = lines[i + 1].strip()
                break
        try:
            n_atoms = int(lines[3][:3])
        except (ValueError, IndexError):
            continue
        syms, xyz = [], []
        for ln in lines[4:4 + n_atoms]:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
                syms.append(parts[3])
            except ValueError:
                continue
        if cid and syms:
            out[cid] = (syms, xyz)
    return out


all_cids = sorted(final, key=int)
store = {}
CH = 60
for i in tqdm(range(0, len(all_cids), CH), desc="SDF 3D chunks", unit="chunk"):
    chunk = all_cids[i:i + CH]
    for tries in range(3):
        try:
            store.update(parse_sdf(fetch_sdf(chunk)))
            break
        except Exception as e:
            print(f"  chunk retry ({type(e).__name__})", flush=True)
            time.sleep(3)
    time.sleep(0.5)
print(f"SDF parsed: {len(store)}/{len(all_cids)}", flush=True)
import collections
els = collections.Counter(e for v in store.values() for e in v[0])
print("elements:", dict(els), flush=True)
nh = sum(1 for v in store.values() if "H" in v[0])
print(f"molecules with explicit H: {nh}/{len(store)}", flush=True)
import pickle
pickle.dump({"mols": store, "values": final},
            open(OUT_SDF, "wb"))
print(f"saved {OUT_SDF}", flush=True)

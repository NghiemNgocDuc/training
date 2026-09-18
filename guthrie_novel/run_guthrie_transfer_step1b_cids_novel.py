"""Step 1b: resolve CIDs only for approx-novel SMILES missing them (fast finish)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_cids import resolve_cid

RUN = r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run"
CIDS = os.path.join(RUN, "guthrie_cids.json")
NOV = os.path.join(RUN, "guthrie_novel_approx.json")

cids = json.load(open(CIDS)) if os.path.exists(CIDS) else {}
nov = json.load(open(NOV))
todo = [s for s in nov if s not in cids]
print(f"novel={len(nov)} resolved={len(cids)} todo={len(todo)}", flush=True)
for i, s in enumerate(todo):
    cids[s] = {"cid": resolve_cid(s), "value_kcal": nov[s], "n_rows": -1}
    time.sleep(0.2)
    if (i + 1) % 25 == 0:
        json.dump(cids, open(CIDS, "w"))
        print(f"  {i+1}/{len(todo)}", flush=True)
json.dump(cids, open(CIDS, "w"))
n_ok = sum(1 for s in nov if isinstance(cids.get(s, {}).get("cid"), int))
print(f"done: {n_ok}/{len(nov)} novel with CID", flush=True)

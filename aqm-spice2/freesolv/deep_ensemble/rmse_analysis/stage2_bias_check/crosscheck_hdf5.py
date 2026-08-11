"""Cross-check: Stage-2 predictions on the STORED hdf5 conformer (single best-MMFF
of 10, the historic predict_freesolv protocol) for the same 36 molecules, to test
whether the heavy-tail Stage-2 errors are fresh-embedding geometry artifacts."""

import sys, os, json, numpy as np, pandas as pd, torch, h5py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
FREESOLV_DIR = os.path.join(REPO_ROOT, "aqm-spice2", "freesolv")
sys.path.insert(0, FREESOLV_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "aqm-spice2"))
from predict_freesolv import build_model
from freesolv_dataset import EV_TO_KCAL, load_freesolv_labels
from element_vocab import build_one_hot
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

CKPT = os.path.join(REPO_ROOT, "aqm-spice2", "aqm-spice2", "pipeline", "results_full", "stage2_correction.pt")
H5 = os.path.join(REPO_ROOT, "freesolv_conformers.hdf5")
LABELS = json.load(open(os.path.join(REPO_ROOT, "Data", "FreeSolv", "database.json")))
out = pd.read_csv(os.path.join(REPO_ROOT, "aqm-spice2", "freesolv", "deep_ensemble", "rmse_analysis", "stage2_bias_check", "stage2_predictions.csv"))
ids = out.mol_id.tolist()

model = build_model(num_blocks=3)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()

data_list = []
with h5py.File(H5, "r") as f:
    for mid in ids:
        g = f[mid]
        data_list.append(Data(mid=mid,
                              z=torch.tensor(g["atNUM"][...], dtype=torch.long),
                              pos=torch.tensor(g["atXYZ"][...], dtype=torch.float)))
loader = DataLoader(data_list, batch_size=32, shuffle=False)
preds = {}
with torch.no_grad():
    for d in loader:
        mids = d.mid
        x = build_one_hot(d, torch.device("cpu"))
        vals = model(x, d.pos, d.batch).view(-1).cpu() * EV_TO_KCAL
        for i, m in enumerate(mids):
            preds[m] = float(vals[i])

rows = []
group_of = dict(zip(out.mol_id, out.group))
for mid in ids:
    rows.append({"mol_id": mid,
                 "group": group_of[mid],
                 "hdf5_stage2_pred_kcal": preds[mid],
                 "hdf5_stage2_abs_err_kcal": abs(preds[mid] - float(LABELS[mid]["expt"])),
                 "fresh_tta_stage2_abs_err_kcal": float(out.set_index("mol_id").loc[mid, "stage2_abs_err_kcal"])})
r = pd.DataFrame(rows)
OUT_DIR = os.path.join(REPO_ROOT, "aqm-spice2", "freesolv", "deep_ensemble", "rmse_analysis", "stage2_bias_check")
r.to_csv(os.path.join(OUT_DIR, "stage2_hdf5_crosscheck.csv"), index=False)
for grp in ("confidently_wrong", "control"):
    g = r[r.group == grp]
    print(f"{grp}: hdf5-protocol median |err|={g.hdf5_stage2_abs_err_kcal.median():.3f} mean={g.hdf5_stage2_abs_err_kcal.mean():.3f} max={g.hdf5_stage2_abs_err_kcal.max():.3f}")
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(r[r.group == "confidently_wrong"].hdf5_stage2_abs_err_kcal, r[r.group == "control"].hdf5_stage2_abs_err_kcal)
print(f"Mann-Whitney hdf5-protocol: p={p:.4f}")
print(r[r.hdf5_stage2_abs_err_kcal > 10][["mol_id", "group", "hdf5_stage2_abs_err_kcal", "fresh_tta_stage2_abs_err_kcal"]].to_string(index=False))
"""Debug: verbatim replication of approach2's Mode A random arm only."""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
FREESOLV = os.path.join(REPO, "aqm-spice2", "freesolv")
NODE_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                        "approach2_node_refine", "node_contributions.csv")
PRED_CSV = os.path.join(FREESOLV, "deep_ensemble", "repair_data",
                        "seed_predictions_all642.csv")
SPLIT_DIR = os.path.join(REPO, "aqm-spice2", "aqm-spice2", "freesolv",
                         "cv_results_full", "fold_0")
RESULTS_CSV = os.path.join(FREESOLV, "experimental_uncertainty_refine", "output",
                           "approach2_node_refine", "results.csv")

SEEDS3 = [42, 123, 999]
RNG_SEED = 20260815

nodes = pd.read_csv(NODE_CSV)
pred = pd.read_csv(PRED_CSV)
tr = json.load(open(os.path.join(SPLIT_DIR, "train_ids.json")))
va = json.load(open(os.path.join(SPLIT_DIR, "val_ids.json")))
te = json.load(open(os.path.join(SPLIT_DIR, "test_ids.json")))
all_ids = tr + va + te

desc = np.array([json.loads(d) for d in nodes["desc"]], dtype=np.float32)
pool_mol = nodes["mol_id"].to_numpy()
u3 = nodes["u3"].to_numpy()
gate = u3 >= np.quantile(u3, 0.75)
trust = 1.0 - pd.Series(u3).rank(pct=True).to_numpy()
P3 = np.stack([nodes[f"P_seed{s}"].to_numpy() for s in SEEDS3], axis=1)

gidx = np.flatnonzero(gate)
qdesc = desc[gidx]
inter = np.minimum(desc[None, :, :], qdesc[:, None, :]).sum(axis=2)
union = np.maximum(desc[None, :, :], qdesc[:, None, :]).sum(axis=2)
with np.errstate(divide="ignore", invalid="ignore"):
    sim = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
same_mol = pool_mol[None, :] == pool_mol[gidx][:, None]
ok = (sim >= 0.2) & (~same_mol)
nbrs, ws = {}, {}
for i, gi in enumerate(gidx):
    o = np.flatnonzero(ok[i])
    order = o[np.argsort(-sim[i][o])[:10]]
    nbrs[int(gi)] = order.tolist()
    ws[int(gi)] = sim[i][order].tolist()
print("nbrs built, gated:", len(gidx))


def refine_verbatim(arm, alpha, seeds, trust, pool_P, rng):
    col = {s: i for i, s in enumerate(seeds)}
    res = {}
    for mid in all_ids:
        m = np.flatnonzero(pool_mol == mid)
        start = m[0]
        n_atoms = len(m)
        gated = np.flatnonzero(gate[start:start + n_atoms])
        if len(gated) == 0:
            res[mid] = {s: float(pool_P[start:start + n_atoms, col[s]].sum())
                        for s in seeds}
            continue
        gl = [(nbrs[int(start + g)], ws[int(start + g)]) for g in gated]
        out_m = {}
        for s in seeds:
            P = pool_P[start:start + n_atoms, col[s]]
            newp = P.copy()
            for gi2, (nidx, w) in enumerate(gl):
                if len(nidx) == 0:
                    continue
                t = trust[nidx] if arm != "naive" else np.ones(len(nidx))
                w = np.array(w)
                denom = (w * t).sum()
                if denom <= 0:
                    continue
                nb = (w * t * pool_P[nidx, col[s]]).sum() / denom
                newp[gated[gi2]] = (1 - alpha) * P[gated[gi2]] + alpha * nb
            if arm == "random":
                mag = np.abs(newp - P)
                sign = rng.choice([-1.0, 1.0], size=len(newp))
                newp = np.where(gate[start:start + n_atoms],
                                P + sign * mag, newp)
            out_m[s] = float(newp.sum())
        res[mid] = out_m
    return res


alpha = 1.0
res = refine_verbatim("random", alpha, SEEDS3, trust, P3,
                      np.random.default_rng(RNG_SEED))
pmean = {m: float(np.mean([res[m][s] for s in SEEDS3])) for m in te}


def refine_v1_dbg(arm, rng):
    """Mirror of v1_verify.refine_A with value logging for first molecules."""
    gidx = np.flatnonzero(gate)
    new = P3[gidx].copy()
    for gi, nidx in enumerate([nbrs[int(i)] for i in gidx]):
        if len(nidx) == 0:
            continue
        tt = trust[nidx] if arm != "naive" else np.ones(len(nidx))
        wv = np.array(ws[int(gidx[gi])])
        denom = (wv * tt).sum()
        if denom <= 0:
            continue
        for s in range(3):
            nb = (wv * tt * P3[nidx, s]).sum() / denom
            new[gi, s] = nb
    gp = np.flatnonzero(gate)
    log = []
    for mid in all_ids:
        m = np.flatnonzero(pool_mol == mid)
        n_atoms = len(m)
        start = m[0]
        gm = np.flatnonzero(gate[start:start + n_atoms])
        if len(gm) == 0:
            continue
        k1 = int((gp < start).sum())
        gk = np.arange(k1, k1 + len(gm))
        for s in range(3):
            mag = np.abs(new[gk, s] - P3[gk, s])
            sign_full = rng.choice([-1.0, 1.0], size=n_atoms)
            new[gk, s] = P3[gk, s] + sign_full[gm] * mag
            if mid in te[:6]:
                log.append((mid, s, int(gm[0]), float(mag[0]),
                            float(sign_full[gm[0]]), float(new[gk[0], s]),
                            float(P3[gk[0], s])))
    return new, log


newv1, logv1 = refine_v1_dbg("random", np.random.default_rng(RNG_SEED))
print("v1 log (mid, s, gm0, mag0, sign0, newval, P):")
for r in logv1[:9]:
    print("  ", r)


def refine_verbatim_dbg(arm, alpha, seeds, trust, pool_P, rng):
    col = {s: i for i, s in enumerate(seeds)}
    res = {}
    log = []
    for mid in all_ids:
        m = np.flatnonzero(pool_mol == mid)
        start = m[0]
        n_atoms = len(m)
        gated = np.flatnonzero(gate[start:start + n_atoms])
        if len(gated) == 0:
            res[mid] = {s: float(pool_P[start:start + n_atoms, col[s]].sum())
                        for s in seeds}
            continue
        gl = [(nbrs[int(start + g)], ws[int(start + g)]) for g in gated]
        out_m = {}
        for s in seeds:
            P = pool_P[start:start + n_atoms, col[s]]
            newp = P.copy()
            for gi2, (nidx, w) in enumerate(gl):
                if len(nidx) == 0:
                    continue
                t = trust[nidx] if arm != "naive" else np.ones(len(nidx))
                w = np.array(w)
                denom = (w * t).sum()
                if denom <= 0:
                    continue
                nb = (w * t * pool_P[nidx, col[s]]).sum() / denom
                newp[gated[gi2]] = (1 - alpha) * P[gated[gi2]] + alpha * nb
            if arm == "random":
                mag = np.abs(newp - P)
                sign = rng.choice([-1.0, 1.0], size=len(newp))
                newp = np.where(gate[start:start + n_atoms],
                                P + sign * mag, newp)
                if mid in te[:6]:
                    log.append((mid, s, int(gated[0]), float(mag[gated[0]]),
                                float(sign[gated[0]]), float(newp[gated[0]]),
                                float(P[gated[0]])))
            out_m[s] = float(newp.sum())
        res[mid] = out_m
    return res, log


res2, log2 = refine_verbatim_dbg("random", 1.0, SEEDS3, trust, P3,
                                 np.random.default_rng(RNG_SEED))
print("verbatim log (mid, s, gm0, mag0, sign0, newval, P):")
for r in log2[:9]:
    print("  ", r)

mid0 = te[0]
gp = np.flatnonzero(gate)
m = np.flatnonzero(pool_mol == mid0)
start = m[0]
gm = np.flatnonzero(gate[start:start + len(m)])
k1 = int((gp < start).sum())
print(f"\nindex check for {mid0}: start={start} n_atoms={len(m)} "
      f"n_gated={len(gm)} k1={k1}")
print("  gp[k1:k1+n_gated] == start+gm :",
      np.array_equal(gp[k1:k1 + len(gm)], start + gm))
print(f"  P3[gp[k1], 0] (gated-array idx) = {P3[gp[k1], 0]:.6f}")
print(f"  P3[start+19, 0] (full-array idx) = {P3[start + 19, 0]:.6f}")


t = pred[pred.mol_id.isin(te)].copy()
t["mean3"] = t[[f"pred_seed{s}" for s in SEEDS3]].mean(axis=1)
t["std3"] = t[[f"pred_seed{s}" for s in SEEDS3]].std(axis=1)
thr_std = t["std3"].quantile(0.75)
q_std = set(t.loc[t["std3"] >= thr_std, "mol_id"])
tdf = t.set_index("mol_id")
truth = tdf["true_value"]

for name, pop in [("Q_std", q_std), ("all129", set(te))]:
    sub = [m for m in pop if m in tdf.index]
    d = (np.abs(np.array([pmean[m] for m in sub]) - tdf.loc[sub, "true_value"].to_numpy())
         - np.abs(tdf.loc[sub, "mean3"].to_numpy() - tdf.loc[sub, "true_value"].to_numpy()))
    print(f"verbatim A-random {name}: delta={d.mean():.4f}")
    row = pd.read_csv(RESULTS_CSV)
    r = row[(row.mode == "A") & (row.arm == "random") & (row.population == name)]
    print(f"  saved A-random {name}: delta={r.delta_mae.values[0]:.4f}")
print("sample molecules (verbatim pmean vs mean3):")
for m in te[:5]:
    print(f"  {m}: pmean={pmean[m]:.4f} mean3={tdf.loc[m, 'mean3']:.4f} "
          f"truth={tdf.loc[m, 'true_value']:.4f}")
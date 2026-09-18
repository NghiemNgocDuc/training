import csv
import statistics
import numpy as np

rows = list(csv.DictReader(
    open(r"C:\Users\User\AppData\Local\Temp\opencode\guthrie_run\guthrie_all_methods.csv")))
for r in rows:
    for k in ("exp", "raw", "gims", "uniform", "vw", "Lambda", "N"):
        r[k] = float(r[k])
print("n =", len(rows))


def rep(name, sub):
    exp = np.array([r["exp"] for r in sub])
    out = {}
    for arm in ("raw", "gims", "uniform", "vw"):
        out[arm] = float(np.abs(np.array([r[arm] for r in sub]) - exp).mean())
    g_vw = out["gims"] - out["vw"]
    g_un = out["gims"] - out["uniform"]
    flag = "  <-- GIMS WINS" if (g_vw < 0 and g_un < 0) else ""
    print(namehoz := (f"{name:24s} n={len(sub):4d} raw={out['raw']:.3f} "
                      f"gims={out['gims']:.3f}({out['gims']-out['raw']:+.3f}) "
                      f"vw={out['vw']:.3f}({out['vw']-out['raw']:+.3f}) "
                      f"uni={out['uniform']:.3f}({out['uniform']-out['raw']:+.3f}) "
                      f"G-VW={g_vw:+.3f} G-Uni={g_un:+.3f}{flag}"))


rep("all", rows)
med = statistics.median(r["N"] for r in rows)
rep(f"heavy N>{med:.0f}", [r for r in rows if r["N"] > med])
rep(f"light N<={med:.0f}", [r for r in rows if r["N"] <= med])
rep("N>=25", [r for r in rows if r["N"] >= 25])
rep("N>=30", [r for r in rows if r["N"] >= 30])
rep("Lambda>0.9", [r for r in rows if r["Lambda"] > 0.9])
rep("exp<-8 hydrophilic", [r for r in rows if r["exp"] < -8])
rep("exp>-4 hydrophobic", [r for r in rows if r["exp"] > -4])

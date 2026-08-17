# Arithmetic Audit — Part 1: Aggregation-Point Inventory

**Date:** 2026-08-16
**Audit trigger:** factor-3 bug found in `holdout_validation/b8_holdout.py` (per-seed vectors
summed with `.sum()` instead of averaged with `.mean()` before entering mean-space arithmetic).
User directive: search ALL scripts in the node-refinement chain for the same bug class —
per-seed / per-node / per-molecule values **summed across seeds/entities BEFORE conversion to
correct mean-space** — before any manuscript claims are finalized.

**Bug class definition:** any aggregation where a 3-vector of per-seed values
(shape `[3]`, or a per-node `[N,3]` array) is collapsed with `.sum()` along the seed axis,
or where per-molecule per-seed values are summed across seeds, instead of using
`.mean()` (equivalently: dividing the sum by 3, or averaging per-seed molecule predictions).

**Scope:** every `.py` in `node_refinement/` (recursive) + the original 2x2 control matrix
script `experimental_uncertainty_refine/approach2_node_refine.py` (user-named).

---

## Script-by-script verdict

### 1. `verification/v1_verify.py` — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 94–95 | `nodes.groupby("mol_id")[...].sum()` per seed — per-seed molecule sums | CORRECT (sum within a seed) |
| 178–183 | `refine_A`: `(wv*tt*pool_P(nidx,s)).sum()/denom` per seed | CORRECT (weighted **mean** per seed) |
| 202, 221 | `refine_B`: same on `P3[nidx].mean(axis=1)` (per-node seed mean) | CORRECT |
| 248–254 | `mol_preds`: per-seed node sums; 286–291: `np.mean([mp[s]... for s in SEEDS3])` | CORRECT (mean over seeds) |
| 315–320 | bootstrap on per-molecule deltas | CORRECT |
| 402 | `np.abs(newA_trust - P3[gidx]).sum(axis=1)/3` | CORRECT (= mean over seeds) |
| 446–451 | per-seed node/molecule stats (no cross-seed sum) | CORRECT |

### 2. `verification/dbg_random.py` — CLEAN (debug/diagnostic script)
| Lines | Aggregation point | Verdict |
|---|---|---|
| 62, 85 | `res[mid] = {s: pool_P[...].sum()}` / `out_m[s] = newp.sum()` — per-seed molecule sums | CORRECT |
| 75–78, 105–109 | weighted means `(w*t*...).sum()/denom` per seed | CORRECT |
| 93 | `pmean = np.mean([res[m][s] for s in SEEDS3])` | CORRECT (mean over seeds) |

### 3. `verification/part3_strengthened/part3_strengthened.py` — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 95 | per-seed molecule sums via groupby | CORRECT |
| 181–186 | `refine_A_arm`: per-seed weighted mean `(wv*tt*P3[nidx,s]).sum()/denom` | CORRECT |
| 201 | `refine_B_arm`: weighted mean on `P3[nidx].mean(axis=1)` | CORRECT |
| 268–274 | per-seed / mean-space molecule sums | CORRECT |
| 284–293 | `np.mean([mp[s]... for s in SEEDS3])` — mean over seeds | CORRECT |
| 307–312 | bootstrap on per-molecule deltas | CORRECT |

### 4. `shrinkage_calibrated/shrinkage_calibrated.py` (Part 7, lambda calibration) — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 71 | per-seed molecule sums (sanity check vs pred CSV) | CORRECT |
| 101–102 | `mu_per_seed = P3.mean(axis=0)` / `mu_bar = P3.mean()` — pool means | CORRECT |
| 110–121 | `mol_preds_A` (per-seed shrink, per-seed node sums) / `mol_preds_B` (mean-space) | CORRECT |
| 133–135, 144–147 | `np.mean([mpA[s]... for s in SEEDS3])` — mean over seeds | CORRECT |
| 149–154 | bootstrap nrow convention `200 + mode*55 + lam_i*5 + pop_i` | CORRECT |
| 172 | `lambda* = argmin` over val mean delta | CORRECT |

### 5. `holdout_validation/b8_holdout.py` — **THE BUG. NOW FIXED.**
**Original bug (factor 3):** all four arms computed per-seed vectors then collapsed with
`.sum()` — `shrink_val(gi,lam).sum()`, `tv.sum()`, `nv.sum()`, `randall_val(gi).sum()` —
entering mean-space arithmetic with 3x-inflated node totals.

**Fix (verified line-by-line, current file):**
| Lines | Aggregation point | Verdict |
|---|---|---|
| 85–86 | pool means per seed / overall | CORRECT |
| 161–164 | `trust_val`: per-seed weighted mean `(wv[:,None]*tt[:,None]*P3[nidx]).sum(axis=0)/denom` | CORRECT (per-seed) |
| 170 | `naive_val`: per-seed weighted mean | CORRECT |
| 174 | `randall_val`: `P3[j].mean(axis=0)` per-seed | CORRECT |
| 177 | `shrink_val`: per-seed shrink `(1-λ)P3[gi] + λ*mu_s` | CORRECT |
| 186–207 | `s_orig` (per-node **mean** over seeds, summed over gated nodes) + `tot = sum(v.mean() ...)` | CORRECT (`.mean()` over seeds — the fix) |
| 221–234 | bootstrap rng2 = `default_rng(20260816)` sequential per cell | CORRECT |

The fix was cross-validated: corrected outputs match Part 7/8 to 1.42e-14 (max abs diff),
and λ*_H1 = 1.0 survived the fix. The CORRECTION note is documented in the report's
`honest_notes` and in `v1_VERIFICATION.md` Part B.

### 6. `seed_resolution/a2_probe.py` (Part A probe) — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 135, 143–146 | per-molecule MAE per seed (no cross-seed sums; TTA mean over views) | CORRECT |

### 7. `gradient12_mechanism/gradient12_mechanism.py` (Part 8) — CLEAN
Part-8 values were already cross-checked against Part 7/8 verified outputs (1.42e-14) during
the Part B fix; aggregation uses the same verified patterns (per-seed node sums → mean over
seeds; mean-space trust arm). Verified clean in prior session + this sweep.

### 8. `gradient12_mechanism/generalization_check/generalization_check.py` (Part 9) — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 154–157 | mean-space trust value `(wv*tt*P3[nidx].mean(axis=1)).sum()/denom` | CORRECT |
| 167, 175 | `sum(P3[gi].mean() for gi in lg)` — per-node means, sum over nodes | CORRECT |
| 205, 233 | quartile aggregation / means of per-molecule deltas | CORRECT |

### 9. `experimental_uncertainty_refine/approach2_node_refine.py` (original 2x2 matrix) — CLEAN
| Lines | Aggregation point | Verdict |
|---|---|---|
| 311, 357 | `res[mid] = {s: float(P[:, col[s]].sum())}` / `out_m[s] = float(newp.sum())` — per-seed molecule sums | CORRECT |
| 313, 338 | Mode B: mean-space molecule sums | CORRECT |
| 328–331 | weighted **mean** on pool_Pbar (mean-space) | CORRECT |
| 350 | per-seed weighted mean `(w*t*pool_P[nidx,col[s]]).sum()/denom` | CORRECT |
| 373, 407, 435 | `np.mean([res[m][s] for s in SEEDS3])` — mean over seeds | CORRECT |
| 419–423 | bootstrap on per-molecule deltas (rng2 = RNG_SEED + row index) | CORRECT |

---

## Conclusion (Part 1)

**The factor-3 bug was isolated to `b8_holdout.py`. NO other aggregation point in the
inventoried scripts sums per-seed values before mean-space conversion.** All 9 scripts use
the correct pattern throughout: per-seed arithmetic → per-seed molecule sums → `.mean()` over
seeds (or equivalent per-node mean-space arithmetic).

Also re-verified in this sweep:
- trust definition `1 - rank(pct)(u3)`, K=10, min_sim=0.2, GATE_Q=0.75, gate n=2904 — identical
  across v1_verify / part3_strengthened / shrinkage_calibrated / b8_holdout / approach2 /
  gradient12_mechanism / generalization_check.
- Approach2's calibration alpha = 1.0 for ALL three arms (trust/naive/random) in results.csv —
  so Part 7's "vs trust vs naive" comparison uses alpha=1.0 (matches shrinkage@λ*=1.0 design).
- `build_populations` (test_time_repair.py:107) recomputes ensemble_mean/std over the 3
  surviving seeds (not the polluted 5-seed CSV columns).

One legacy note (not a bug, flagged for the record): `v1_verify.py` line 402 uses
`.sum(axis=1)/3` rather than `.mean(axis=1)` — mathematically identical, kept verbatim for
replication fidelity.

Part 2 (independent from-scratch recomputation) verifies the headline numbers numerically.
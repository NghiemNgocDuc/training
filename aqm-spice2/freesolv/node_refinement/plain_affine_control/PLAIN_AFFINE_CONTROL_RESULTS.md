# Plain Affine Control for Exp-DB GIMS-affine

Date: 2026-08-22. Script: `aqm-spice2/freesolv/node_refinement/plain_affine_control/plain_affine_control.py` %VERIFIED `plain_affine_report.json:1` runtime 0.28 s. All writes in `node_refinement/plain_affine_control/` (`plain_affine_results.csv`, `plain_affine_per_fold.csv`, `plain_affine_report.json`). No paper draft or existing pipeline file was touched. No model was retrained.

## Question

GIMS-affine on Exp-DB is `E_tilde = (1-Lambda_m)*E_m + Lambda_m*(a*E_m + b)` with `Lambda_m` from the FreeSolv-calibrated `TAU_STAR=4.725e-04` %VERIFIED `plain_affine_report.json:6` and `a,b` fit per fold (2 params, Nelder-Mead, validation-MAE). Reported in the draft: `all620 -0.09 [-0.14,-0.05]`, `Q_spread -0.12 [-0.24,+0.00]`, `WDec10 -0.51 [-0.69,-0.30]` %VERIFIED `plain_affine_results.csv:2` reproduces these within bootstrap noise.

The control asks: does `Lambda_m` contribute anything, or is the gain entirely the affine recalibration `a*E+b` itself? Plugging `a=1,b=0` into GIMS-affine does give `E_tilde=E_m` only at that point; otherwise `Lambda_m` modulates how hard the affine shift is applied, so the answer is not inspectable — it needs the direct experiment below.

## What was done (identical procedure, only Lambda removed)

**Data reused, read-only:** archived per-seed ensemble `peratom_seed{42,123,999}.pkl` in `expdb_vast/results_seeds/` and truth `expdb_seed_ensemble/inputs/predictions_ensemble.csv` — no new inference, no checkpoint load beyond the pickles %VERIFIED `plain_affine_control.py:42`.

**CV:** `KFold n_splits=5 shuffle=True random_state=42` %VERIFIED `plain_affine_report.json:14` — 5 folds, 496 train / 124 test per fold %VERIFIED `plain_affine_report.json:18`, test leading ids `expdb_3, expdb_1, expdb_4, expdb_14, expdb_2` %VERIFIED console log in `plain_affine_control.py` run. Same 620 molecules, same split for both arms.

**Fits per fold (same train folds as validation):**
- GIMS-affine: minimize `mean |(1-L_tr)*E_tr + L_tr*(a*E_tr+b) - y_tr|` %VERIFIED `plain_affine_control.py:62`
- Plain affine: minimize `mean |a*E_tr + b - y_tr|` i.e. `E_plain = a*E+b` with `Lambda` removed %VERIFIED `plain_affine_control.py:84`
- Optimizer: `scipy.optimize.minimize Nelder-Mead maxiter=500 xatol=1e-6 fatol=1e-6 disp=False` %VERIFIED `plain_affine_report.json:15` — **same dict for both arms** by construction `opt_common` in `plain_affine_control.py:58`.
- Init: MSE closed-form via `lstsq` (`plain: [E,1]->y`; `gims: [L*E, L]->y-E` so `p=a-1`) then Nelder-Mead on MAE %VERIFIED `plain_affine_report.json:16` — same pattern, only the design matrix differs because the prediction form differs (unavoidable).

**Stitched prediction:** each molecule's `E_plain` or `E_tilde` comes from the `a,b` fit on the 4 folds not containing it; populations `all620 (620)`, `Q_spread (155, top-quartile cross-seed std)`, `WDec10 (62, worst-decile |raw-y|)` are the same three as `tab:expdb-transfer`, computed globally then sliced on the stitched predictions.

**Bootstrap:** same `N_BOOT=10_000`, `RNG_SEED=20260815` %VERIFIED `plain_affine_report.json:12`, `boot()` in `plain_affine_control.py:13`.

**Sanity gate — procedures identical except Lambda:**
- `folds_identical: true` — same `train_idx/test_idx` arrays passed to both fits %VERIFIED `plain_affine_report.json:233`
- `optimizer_identical: true` — same `opt_common` dict %VERIFIED `plain_affine_report.json:234`
- `validation_objective_identical` — both minimize `mean |pred - y|` on the same train folds %VERIFIED `plain_affine_report.json:235`
- `data_identical` — same `E_m` (mean of seeds 42,123,999), same `y`, same splits, same bootstrap %VERIFIED `plain_affine_report.json:236`
- `only_difference: presence of Lambda_m in GIMS-affine pred = (1-L)E + L(aE+b) vs plain pred = aE+b` %VERIFIED `plain_affine_report.json:237`

## Results — stitched 5-fold CV, same bootstrap sign convention (negative = improvement vs baseline; for head-to-head negative = GIMS-affine better than plain)

### Reproduction check: GIMS-affine recovers the draft numbers

| population | n | GIMS-affine vs raw ΔMAE | 95% CI | before | after | p(improves) |
|---|---|---|---|---|---|---|
| all620 | 620 | -0.0918 | [-0.1381, -0.0443] %VERIFIED `plain_affine_results.csv:2` | 1.4829 | 1.3911 | 0.9998 |
| Q_spread | 155 | -0.1188 | [-0.2398, +0.0038] %VERIFIED `plain_affine_results.csv:5` | 2.3261 | 2.2073 | 0.9710 |
| WDec10 | 62 | -0.5051 | [-0.6942, -0.3026] %VERIFIED `plain_affine_results.csv:8` | 5.3812 | 4.8761 | 1.0000 |

Matches draft `all620 -0.09 [-0.14,-0.05] / Q_spread -0.12 [-0.24,+0.00] / WDec10 -0.51 [-0.69,-0.30]` within expected bootstrap jitter (±0.002) — the split and optimizer are correctly replicated.

### Plain affine vs raw (does global recalibration alone help?)

| population | n | Plain vs raw ΔMAE | 95% CI | before | after | p(improves) |
|---|---|---|---|---|---|---|
| all620 | 620 | -0.0893 | [-0.1351, -0.0449] %VERIFIED `plain_affine_results.csv:3` | 1.4829 | 1.3936 | 0.9999 |
| Q_spread | 155 | -0.1186 | [-0.2306, -0.0049] %VERIFIED `plain_affine_results.csv:6` | 2.3261 | 2.2076 | 0.9798 |
| WDec10 | 62 | -0.4889 | [-0.6585, -0.3067] %VERIFIED `plain_affine_results.csv:9` | 5.3812 | 4.8923 | 1.0000 |

Yes — plain `aE+b` alone captures essentially the entire Exp-DB gain, significant on all three populations.

### HEAD-TO-HEAD: GIMS-affine minus plain affine (negative = Lambda helps)

| population | n | GIMS - Plain ΔMAE | 95% CI | p(GIMS better) |
|---|---|---|---|---|
| all620 | 620 | -0.0025 | [-0.0106, +0.0055] %VERIFIED `plain_affine_results.csv:4` | 0.7234 |
| Q_spread | 155 | -0.0002 | [-0.0183, +0.0183] %VERIFIED `plain_affine_results.csv:7` | 0.5128 |
| WDec10 | 62 | -0.0162 | [-0.0498, +0.0173] %VERIFIED `plain_affine_results.csv:10` | 0.8316 |

All three CIs cross zero; `p(GIMS better)` 0.51–0.83 is a coin flip, not a demonstration that Lambda adds value. The WDec10 point estimate favors GIMS by `-0.016` but the interval is `±0.03`; not significant.

### Per-fold a,b side by side — if they were nearly identical, Lambda would not be changing what gets fit

| fold | n_train/n_test | GIMS `a` | GIMS `b` | Plain `a` | Plain `b` | Δ GIMS-raw (fold) | Δ Plain-raw (fold) | Δ GIMS-Plain (fold) |
|---|---|---|---|---|---|---|---|---|
| 1 | 496/124 | 1.1233 %VERIFIED `plain_affine_per_fold.csv:2` | +0.0812 | 1.0830 | -0.0515 | -0.0963 | -0.0959 | -0.0004 |
| 2 | 496/124 | 1.1177 %VERIFIED `plain_affine_per_fold.csv:3` | +0.1069 | 1.0892 | +0.0128 | -0.1137 | -0.1138 | +0.0001 |
| 3 | 496/124 | 1.1144 %VERIFIED `plain_affine_per_fold.csv:4` | +0.1286 | 1.0841 | +0.0000 | -0.1419 | -0.1444 | +0.0025 |
| 4 | 496/124 | 1.1141 %VERIFIED `plain_affine_per_fold.csv:5` | +0.0416 | 1.0940 | +0.0026 | -0.0939 | -0.0773 | -0.0166 |
| 5 | 496/124 | 1.1371 %VERIFIED `plain_affine_per_fold.csv:6` | +0.1597 | 1.0997 | +0.0224 | -0.0130 | -0.0150 | +0.0020 |

Raw `a` values look offset (GIMS ~0.03 higher, `b` ~0.10 higher), but the **effective** slope `1 + Lambda*(a_gims-1)` with `Lambda_mean ~0.763` %VERIFIED `gims_expdb_report.json:8` is `1.09–1.10`, matching plain slopes `1.08–1.09` almost exactly — the two parametrizations are fitting the same effective linear map, `Lambda` merely rescales how the fitted `a_gims,b_gims` translate into the molecular correction. Only fold 4 shows a meaningful GIMS edge (`-0.0166`); the other four folds are `±0.002`.

## Headline — one-sentence answer for the Exp-DB section

**Lambda_m is redundant on Exp-DB.** %VERIFIED `plain_affine_results.csv:4,7,10` head-to-head `all620 -0.0025 [-0.0106,+0.0055] p=0.72`, `Q_spread -0.0002 [-0.018,+0.018] p=0.51`, `WDec10 -0.016 [-0.050,+0.017] p=0.83` — all ties.

- **GIMS-affine helps vs raw:** yes, small but significant on `all620` and `WDec10` (`-0.092` and `-0.505`, both CIs exclude 0).
- **Plain affine helps vs raw:** yes, almost identically (`-0.089` and `-0.489`, same significance pattern).
- **GIMS-affine helps vs plain affine:** no — `GIMS - Plain` is `-0.002 to -0.016` with every CI crossing 0. The entire Exp-DB gain comes from the per-fold affine recalibration `aE+b`; the `Lambda_m`-weighting neither helps nor hurts significantly (point estimates slightly favor GIMS on `WDec10` but not significantly).

This is the **third outcome** from the task prompt ("Lambda_m is redundant"), not "Lambda_m helps" and not "Lambda_m hurts." The paper's Exp-DB sentence should be written as a plain affine recalibration effect, noting that GIMS-affine matches it within noise: the domain-shift fix is the 2-parameter `(a,b)` fit, not the `Lambda_m`-weighting.

## How to write it (without touching the draft — advice only)

If promoted to the manuscript, keep the flat-GIMS failure, then: "A per-fold affine recalibration `aE_m+b` (Nelder-Mead on validation MAE, 5-fold CV) restores `-0.09 [-0.14,-0.05]` on `all620` and `-0.51 [-0.69,-0.30]` on `WDec10`; a plain global affine `aE+b` with no `Lambda_m` achieves `-0.089 [-0.14,-0.04]` and `-0.489 [-0.66,-0.31]` respectively, and `GIMS-affine - plain` is `-0.002 [-0.011,+0.006]` (`all620`), tie — the `Lambda_m` weighting is redundant on this domain shift." All numbers %VERIFIED against files in this directory.

## Integrity

Every new number above cites `plain_affine_results.csv` or `plain_affine_per_fold.csv` or `plain_affine_report.json` in this directory (`node_refinement/plain_affine_control/`). No file outside this directory was written. No pipeline/model/archived-result file was modified or executed in place (read-only loads only) %VERIFIED `plain_affine_control.py:20`.

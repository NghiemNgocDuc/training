# Gradient-12: conformer & provenance checks — final (box-truth)

Groups as in all prior analyses: gradient12 = wrong18 − isolated6 (12),
isolated6 (6), wrong18 (18, quadrant `low_std_high_rmse`), certain47 (47,
quadrant `low_std_low_rmse`). P-values are NOT multiple-testing corrected
(hypothesis-generating, consistent with prior analyses).

All box runs below: `root@C.47518475:/workspace/training`, torch 2.12.0+cu130,
rdkit 2026.03.5, GPU RTX 5080 (check2), seed-42 checkpoint (sha `7994ef…`).

---

## 0. The recorded-metrics mystery — resolved (read first)

The original training-time evals (single-conf MAE 0.5313; 5-conf TTA 0.5048 per
`predictions.csv`) cannot be reproduced on ANY surviving machine:

- **Local (Windows, rdkit 2026.03.4)** and **the new box (rdkit 2026.03.5)**
  produce byte-identical results (per-seed draw deltas match exactly; both give
  stored-conformer single-conf MAE 2.5155 on the 59-mol subset) → conformer
  generation agrees across all surviving environments.
- The old training instance is destroyed (only C.47518475 exists). Its
  environment — and with it the 0.5048/0.5313 numbers — is gone.
- `freesolv_conformers.hdf5` was **not** the input to the recorded evals:
  git history holds exactly one blob (`61c48f1`, "Track freesolv_conformers.hdf5
  (1.7 MB, un-ignored from *.hdf5 pattern)") == the surviving stale file, created
  locally 7/21/2026 — *after* training. The recorded metrics were computed from
  fresh in-memory conformers on the old box with its training-time RDKit; that
  geometry regime differs from the surviving 2026.03 family in a way that changes
  per-molecule predictions by up to ~20 kcal/mol (glucose: −2.3…−5.9 surviving
  environment vs −24.24 recorded, true −25.47), yet the checkpoint, DimeModels.py
  (unchanged since `b4c0b56`) and hdf5 structure are all authentic.

**Consequences:**
1. Absolute MAEs from the surviving environments (3.4 kcal/mol single-conf,
   3.409 TTA-5) are NOT comparable to the recorded box metrics. All
   within-machinery RELATIVE comparisons (g12 vs c47) remain valid as controlled
   experiments — identical pipeline for every molecule.
2. The 0.5048/0.5313 record is unrecoverable. Do not compare new numbers to it.

---

## Check 1: conformer ensemble instability (box; == local results)

Protocol identical to `deep_ensemble.conformer_average`: ETKDGv3 (randomSeed 42,
pruneRmsThresh 0.5) + MMFF, 50 conformers requested per molecule (848 kept in
total — pruning leaves 1–3 distinct for these small molecules), seed-42
checkpoint, `model(x, pos, batch) × EV_TO_KCAL`.

| metric | gradient12 med | certain47 med | MWU p |
|---|---|---|---|
| prediction std across conformers (kcal/mol) | 0.095 | 0.256 | 0.76 |
| MMFF energy spread of conformers (kcal/mol) | 0.131 | 0.588 | 0.26 |
| n distinct conformers kept (of 50 requested) | 1.5 | 3.0 | 0.19 |

- **gradient-12 does NOT show higher conformer instability**; direction is
  opposite and gradient-12 keeps FEWER distinct conformers (more rigid), which
  caps their prediction std. All n.s.
- Spearman across all 129: conformer-pred-std vs recorded 5-conf-TTA error
  rho = −0.118 (p = 0.18); other-64 +0.092 (n.s.). MMFF spread vs error −0.104
  (p = 0.24). Conformer instability does not predict error.
- 50- vs 5-conformer TTA (identical machinery): g12 MAE 3.057 → 3.083, all
  3.409 → 3.402; **2/12 gradient-12 improved**. More conformers do not help.
- `check1_box.log`; per-conformer `per_conformer_predictions.csv`, per-molecule
  `per_molecule_conformer_stats.csv`, `calibration_stored_conformer.csv`.

**Verdict: conformer instability does NOT explain gradient-12. Rule-out #6.**

## Check 3: data provenance (box; environment-independent)

`database.json` (642 records): `expt_reference`, `calc_reference`, `groups`,
`notes`, etc.

- **All 12/12 gradient-12 carry `expt_reference = 10.1021/ct050097l`**
  (Mobley et al., JCTC 2006 — dominant FreeSolv source: 422/642 full-DB, 87/129
  test, 67.4% base rate). Fisher vs full-DB: p = 0.0106; vs test-set base:
  p = 0.0105. Real but not absolute; single dominant source, not a discriminator.
- Default-uncertainty note in 120/129 (93%) of the whole test set → not a
  discriminator (p = 1.0). `d_expt` never "Not available"; no
  computed-value-as-experiment flags in this DB copy.
- `groups` tags: no enrichment (largest p = 0.39, 22 tags).
- Cross-tabs: `crosstab_*.csv`, `gradient12_provenance_detail.csv`,
  `provenance_report.json`.

**Verdict: no strong provenance story; weak single-source enrichment only
(Fisher p ≈ 0.01, uncorrected). Rule-out #7, with a footnote.**

## Check 2: conformer-DRAW sensitivity (box; new test, hypothesis 8)

Question: would gradient-12's wrong predictions change if a DIFFERENT, equally
valid conformer draw had been frozen into the pipeline? For each of the 12 + 47
molecules: 5 fresh conformers (ETKDGv3, pruneRmsThresh 0.5, +MMFF) with
embedding seeds **7, 123, 2024, 999** (≠ training seed 42), predicted with the
seed-42 checkpoint; per-molecule sensitivity = |fresh-draw TTA-5 − baseline|
over the 4 seeds, vs two baselines: (a) stored-conformer single prediction
(same environment), (b) recorded training-time TTA-5 (`predictions.csv`).

| test | gradient12 med | certain47 med | MWU p |
|---|---|---|---|
| delta vs stored (same-env, clean) | 0.062 | 0.228 | 0.90 |
| max delta vs stored | 0.120 | 0.364 | 0.92 |
| delta vs recorded TTA-5 (cross-env*) | 2.69 | 1.75 | 0.63 |
| per-seed delta vs TTA-5 (7/123/2024/999) | — | — | 0.64/0.67/0.46/0.89 |

\* contaminated: recorded TTA-5 lives in the destroyed environment (see §0); the
~2.7 kcal/mol g12 value is the environment gap, not draw randomness.

- **Predictions barely move when the conformer draw changes** (0.06–0.23
  kcal/mol, same environment), and gradient-12 is, if anything, LESS
  draw-sensitive (p = 0.90, direction negative). A different equally-valid draw
  would have produced essentially the same (wrong) answer.
- Pooling 5 seeds × 5 conformers does NOT fix gradient-12: cross-env comparison
  invalid (g12 0.572 → 2.423, 2/12 improved); the internally-consistent local
  pooling (all draws same environment) gives g12 3.057 → 3.028, p = 0.34, 8/12
  improved — noise-level. **Conformer-seed ensembling is not an actionable fix.**
- `check2_box.log`, `check2_sensitivity.csv`, `check2_fresh_draw_predictions.csv`,
  `check2_report.json`.

**Verdict: conformer-draw sensitivity does NOT explain gradient-12. Rule-out #8.**

---

## Bottom line

Eight hypotheses ruled out, in order: Tanimoto similarity, GMM-NLL confidence,
physicochemical descriptors, sign consistency, conformer-ensemble instability,
data-source provenance, conformer-draw sensitivity, conformer-seed ensembling.
The gradient-12 cluster is not an artifact of which valid geometry was frozen
into the pipeline, and no conformer-related remedy helps it.

The only durable finding of this round is negative for reproducibility: the
recorded training metrics (0.5048/0.5313) originated in a now-destroyed Vast
instance whose conformer-generation regime cannot be recovered from git, the
checkpoint, or any surviving environment — treat those recorded numbers as
unverifiable historical values.

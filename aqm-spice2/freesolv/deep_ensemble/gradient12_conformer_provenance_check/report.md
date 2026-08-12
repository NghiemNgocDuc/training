# Gradient-12: conformer-instability (Check 1) & data-provenance (Check 3)

Both checks use the same groups as all prior analyses: gradient12 = wrong18 − isolated6
(12), isolated6 (6), wrong18 (18, quadrant `low_std_high_rmse`), certain47 (47,
quadrant `low_std_low_rmse`). P-values are NOT multiple-testing corrected
(hypothesis-generating, consistent with prior analyses).

---

## IMPORTANT calibration finding (read first)

Local (Windows) inference with the **sha-verified** original checkpoint
(`deep_ensemble/seed_42/ensemble_seed42.pt`, sha256 `7994ef…` matches
`metrics.json`) over the **stored conformer set** (`Data/freesolv_conformers.hdf5`)
does NOT reproduce the recorded training-time single-conformer test MAE:

- recorded (box, training time): single-conf MAE **0.5313** kcal/mol
- local (same checkpoint + stored conformers, `de.conformer_average` fallback path):
  **3.398** kcal/mol
- per-molecule local single-conf vs box 5-conf-TTA: Spearman rho = **0.177**

The 8/5 ensemble_average_report claims "matched exactly" but stores no numeric
evidence. Given checkpoint sha, code (`DimeModels.py`) and hdf5 structure are all
consistent with the training state, the discrepancy is attributed to a changed
conformer-generation environment (local RDKit 2026.03.4 ETKDGv3/MMFF vs the box's
training-time RDKit; local MMFF post-optimization energies for e.g. glucose are
+80…+90 kcal/mol, i.e. strained, and local fresh conformers of glucose all predict
−2.3…−5.9 kcal/mol while the box's TTA-5 was −24.24 vs true −25.47).

**Consequences:**
1. Absolute prediction values / MAEs computed locally are NOT comparable to box
   metrics. All within-machinery RELATIVE comparisons below (g12 vs c47, 5-conf vs
   50-conf TTA, correlations) remain valid as controlled experiments (identical
   pipeline for all 129 molecules).
2. The model is EXTREMELY geometry-sensitive: conformer generation differences
   change per-molecule predictions by up to ~20 kcal/mol (glucose), which itself
   supports a "conformer machinery" sensitivity story — but not one that
   differentiates gradient-12 from certain-47 locally.
3. **The definitive run is `box_conformer_checks.sh` on the Vast box**, which has
   the training-time RDKit, environment, and the true `freesolv_conformers.hdf5`.
   Its `calibration:` line should print ~0.5313 if the box files are intact.

---

## Check 1: conformer ensemble instability (local, internally consistent)

Protocol reused exactly from `deep_ensemble.conformer_average`: ETKDGv3
(randomSeed 42, pruneRmsThresh 0.5) + MMFF optimize, 50 conformers requested per
molecule, seed-42 checkpoint, `build_one_hot` + `model(x, pos, batch) × EV_TO_KCAL`.
Also ran a local 5-conformer TTA pass for the conformer-count comparison.

Per-molecule (all 129 test molecules):

| metric | gradient12 med | certain47 med | MWU p |
|---|---|---|---|
| prediction std across conformers (kcal/mol) | 0.095 | 0.256 | 0.76 |
| MMFF energy spread of conformers (kcal/mol) | 0.131 | 0.588 | 0.26 |
| n distinct conformers kept (of 50 requested) | 1.5 | 3.0 | 0.19 |

- **gradient-12 does NOT show higher conformer instability** — if anything the
  direction is opposite, and gradient-12 keeps FEWER distinct conformers (more
  rigid molecules), which mechanically caps their prediction std. All n.s.
- Spearman across all 129: conformer-pred-std vs original 5-conf-TTA error
  rho = **−0.119** (p = 0.18); within other-64: +0.087 (n.s.). MMFF energy spread
  vs error: −0.103 (p = 0.24). Conformer instability does NOT predict error
  generally.
- 50-conformer vs 5-conformer TTA (identical local machinery): g12 MAE
  3.057 → 3.083, all-129 3.398 → 3.392; **2/12 gradient-12 improved**. More
  conformers do NOT improve accuracy (pruning at 0.5 Å leaves only 1–3 distinct
  conformers for these small molecules, so 50 ≈ 5).
- Per-conformer data: `per_conformer_predictions.csv`; per-molecule:
  `per_molecule_conformer_stats.csv`; calibration: `calibration_stored_conformer.csv`.

**Verdict (local, to be confirmed on the box): conformer instability does NOT
explain gradient-12. Rule-out #6.**

## Check 3: data provenance / source heterogeneity

`Data/FreeSolv/database.json` (642 records) has rich metadata: `expt_reference`,
`calc_reference`, `groups` (functional-group tags), `notes`, `iupac`,
`expt_h_reference`, etc. (full field inventory in `provenance_report.json`).

- **All 12/12 gradient-12 molecules carry `expt_reference = 10.1021/ct050097l`**
  (Mobley et al., JCTC 2006 — the dominant FreeSolv experimental source: 422/642
  full-DB, 87/129 test set, 67.4% base rate). Fisher's exact vs full-DB base:
  **p = 0.0106**; vs the more appropriate test-set base rate (87/129): p = 0.0105.
  Enrichment is real but not absolute — it is the single dominant source, and
  gradient-12 is 100% from it. (Uncorrected; one of several tables tested.)
- Default-uncertainty note ("Experimental uncertainty not presently available…"):
  present in **120/129 (93%)** of the whole test set → NOT a discriminator
  (p = 1.0). `d_expt` is never "Not available" locally; no
  "computed-value-used-as-experiment" flags exist in this DB copy.
- `groups` tags: no enrichment (largest p = 0.39; 22 tags tested).
- Cross-tabs: `crosstab_expt_reference.csv`, `crosstab_calc_reference.csv`,
  `crosstab_groups.csv`; per-molecule detail: `gradient12_provenance_detail.csv`.

**Verdict: no strong provenance story. A weak statistical enrichment toward the
single dominant source (Fisher p ≈ 0.01, uncorrected) — worth noting, but it
does not mechanistically explain drift-out. Rule-out #7, with a footnote.**

---

## Bottom line

Sixth and seventh hypotheses ruled out. Neither conformer-ensemble instability
(with 50-conformer TTA offering no improvement) nor data-source provenance
explains gradient-12. The one live thread from this round is the calibration
discrepancy itself (local conformer machinery ≠ box training-time machinery, with
the model being extremely sensitive to it) — run `box_conformer_checks.sh` on the
Vast box for the definitive Check-1 numbers, and confirm the box's stored
conformers reproduce the recorded 0.5313 single-conf MAE.

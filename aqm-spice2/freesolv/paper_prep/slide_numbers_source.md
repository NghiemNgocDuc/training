# Paper_Structure_Deck.pptx — Source Numbers Table
Generated from build.js. Cross-check against this file, not chat transcripts.
Last corrected: trust-weighting reframe (premise falsified, not effect) + Category 7 holdout numbers + rho=0.496 fix.

## Slide 4 — Headline Accuracy
- 0.549 ± 0.024 kcal/mol MAE, 642 molecules, 5-fold CV (fold ensemble)
- Per-fold: 0.515, 0.537, 0.581, 0.570, 0.540
- SOTA table: This work 0.549 | Zhang 2022 0.417 | COSMO-RS 0.520 | ReSolv 0.630 | GAFF/CGenFF 1.11-1.18

## Slide 5 — Uncertainty Blind Spot
- Spearman rho = 0.496 (p=2.3e-9, N=129)
- 18/129 (14%) confidently-wrong molecules
- Isolated-6 (explained) / gradient-12 (case-study explained in §3.6)

## Slide 6 — Molecule-Level Correction
- 17 configurations tested (2 similarity metrics x raw/normalized loss x lambda sweep)
- Negative result, reframed as granularity finding

## Slide 7 — Node-Level Trust-Weighting Reversal
- Trust-weighting genuinely beats no-correction (real, significant improvement)
- Random-neighbor equal-weight placebo beats trust-weighting significantly on every population
- CORRECTED FRAMING: similarity/trust premise falsified; correction EFFECT is real, dominated by shrinkage

## Slide 8 — Calibrated Shrinkage (H1/H2 holdout, all deltas vs uncorrected baseline, negative = improvement)
| Population | Shrink @ lambda* | Trust | Naive |
|---|---|---|---|
| Q_std | -8.22 (sig) | -5.22 (sig, less) | -0.39 (null) |
| Q_nll | -10.73 (sig) | -6.50 NS | -0.06 (null) |
| UNION | -6.65 (sig) | -4.08 NS | -0.15 (null) |
| all H2 (n=65) | -2.70 (sig) | -1.44 NS | +0.10 (null) |
lambda*_H1 = 1.0 (independently recalibrated on H1, confirms lambda*=1.0 from full-data calibration)

## Slide 9 — Gradient-12 Mechanism
- Gated sums <=4.1 vs up to 85 kcal/mol (other groups)
- Neighbor uncertainty: 45% below pool median vs Q_std's 28%
- Replacement alignment: +0.09 vs Q_std's +0.64
- Generalization check: rho=0.17, NS (does not generalize beyond gradient-12)

## Slide 10 — Discussion
- Seed 7 vs seed 2024 failure-pattern correlation: rho=0.96

## Slide 13 — Target Journal
- Journal of Molecular Graphics and Modelling (Elsevier, Scimago Q2, h-index 95), $0 standard route
- Backup: Molecular Informatics (Wiley), Q2-leaning-Q1, $0 standard route
- SIAM CSE27 poster (Feb 2027, Pittsburgh) planned alongside, no conflict

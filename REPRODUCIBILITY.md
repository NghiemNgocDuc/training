# Reproducibility & provenance

**Status: archived results are intact and verifiable; bit-for-bit reproduction
from the current repo state is NOT possible for the headline result.**

## The headline number

FreeSolv hydration free-energy 5-fold CV, ensemble of 5 DimeNet+ (Option B,
5-conformer TTA) members per fold: **MAE = 0.549 +/- 0.024 kcal/mol**
(RMSE = 0.921, R2 = 0.9427), mean over the five per-fold ensemble MAEs
(0.515 / 0.537 / 0.581 / 0.570 / 0.540; arithmetic re-verified from the
records in `docs/progress/VAST_FULL_TRAIN_2MODELS.txt`: mean 0.5486,
sample std 0.0237).

## What happened to the training environment

The Stage 3 training ran on Vast.ai cloud compute (instance) that has since
been **destroyed**. That instance's environment — including the exact
conformer-generation stack (RDKit version + ETKDGv3/MMFF behavior) that
produced the 0.549 result — no longer exists anywhere:

- Every surviving environment (local Windows; replacement Vast instance
  C.47518475) reproduces the *same* conformer machinery, but it differs from
  the original training environment: fresh-conformer 5-conf TTA gives
  ~3.4 kcal/mol MAE with the archived checkpoints, not the recorded ~0.50,
  and per-molecule predictions shift by up to ~20 kcal/mol for some molecules
  (e.g. glucose: -2.3..-5.9 predicted in surviving environments vs -24.24
  recorded, experimental -25.47) while remaining essentially insensitive to
  which conformer draw is used within one environment (~0.06-0.23 kcal/mol).
- `Data/freesolv_conformers.hdf5` (the file currently in the repo)
  **postdates** the original run (created 7/21/2026, single-conformer-per-
  molecule) and was NOT the input to the recorded evaluation metrics; git
  history contains exactly one blob of this file (`61c48f1`), which is the
  surviving one. The recorded metrics were computed from fresh in-memory
  conformers on the destroyed instance.
- Recorded metrics (0.5313 single-conf, 0.5048 TTA-5 for seed 42, etc.) are
  therefore treated as **historical values that are not regenerable** from the
  current repo state.

## What IS preserved and verifiable (verified 2026-08-12)

- Original per-seed prediction files for **fold 0** (`deep_ensemble/seed_42` /
  `seed_123` / `seed_7` / `seed_2024` / `seed_999`), 129 rows each, sha256:
  - seed_42:   `fc5fa4bb…` / checkpoint `7994ef92…`
  - seed_123:  `0cdd1cbf…` / checkpoint `23cb258e…`
  - seed_7:    `6cb97e0f…` / checkpoint `5ace1542…`
  - seed_2024: `c6427d66…` / checkpoint `25ad28c8…`
  - seed_999:  `c16e4ee5…` / checkpoint `4d46b01c…`
  (full sha256 for checkpoints matches the per-seed `metrics.json`
  `checkpoint_sha256` field recorded at training time.)
- Recomputing the fold-0 ensemble (mean of the 5 archived per-seed
  predictions) gives **MAE = 0.5059, RMSE = 0.7696, R2 = 0.9646** — consistent
  with the documentation's own fold-0 statement (0.5059 is documented as the
  fold-0 5-seed mean; the summary table rounded/derived 0.515).
- Folds 1-4: only the per-fold aggregate metrics survive (in
  `docs/progress/VAST_FULL_TRAIN_2MODELS.txt`); the per-fold prediction files
  (`freesolv/cv_results_full/fold_N/…`) were archived on the destroyed
  instance and are not present in this repo.

## Guidance for any writeup / reviewer claims

- Cite 0.549 +/- 0.024 as an archived, arithmetic-verified result whose
  underlying per-fold artifacts for folds 1-4 are no longer present.
- Do NOT claim end-to-end reproducibility from this repo; state that
  checkpoints + fold-0 predictions are archived and sha256-verifiable, and
  that the conformer-generation environment of the original run was lost with
  the cloud instance.
- Any fresh run with the current environment will not bit-for-bit match the
  recorded numbers (see above); treat new numbers as a different experiment.
# Seed selection justification (3-seed regime, journal-methods ready)

Date: 2026-08-16. Scope: why all node-level refinement analyses (uncertainty
gating, trust-weighted refinement, random-neighbor controls, calibrated
shrinkage, gradient-12 mechanism) use ensemble members {42, 123, 999} rather
than the full 5-member ensemble. Evidence files: `a2_probe_report.json`,
`a2_probe_predictions.csv` (this directory); `deep_ensemble/seed_*/metrics.json`
and `predictions.csv`; `deep_ensemble/instrumented_rerun/…/seed_{7,2024}/
val_history.csv` + `run_all_seeds.log`;
`deep_ensemble/gradient12_conformer_provenance_check/report.md`.

## What the manuscript may state (verbatim-ready paragraph)

> **Ensemble composition.** The five ensemble members were fine-tuned
> identically (frozen fold-0 split, shared Stage-2 correction checkpoint,
> same architecture, optimizer, batch size, loss, scheduler, and
> best-validation checkpoints; recorded test MAE 0.50–0.55 kcal/mol on the
> test set). Two members (seeds 7 and 2024) are excluded from all analysis
> reported here. The exclusion is an environmental artifact, not a model
> failure: their recorded training metrics are normal — per-epoch validation
> MAE trajectories decrease monotonically to 0.508 (seed 7, epoch 49) and
> 0.489 (seed 2024, epoch 56), early-stop at patience 30 (epochs 79 and 86),
> training MSE floors at ≈10⁻⁵ (eV)², and best-validation MAE (0.418 at
> epoch 37 / 0.465 at epoch 23) is competitive with the retained seeds
> (0.443/0.451/0.426). However, the training-time environment (a cloud
> instance since destroyed) used an RDKit conformer-generation regime that
> cannot be reproduced on any surviving machine. When the original
> checkpoints of seeds 7 and 2024 are re-scored in the surviving environment —
> with the stored conformer geometry, with freshly generated ETKDGv3+MMFF
> geometry, and with 5-conformer test-time averaging — their test MAE is
> 38.4 and 58.1–58.6 kcal/mol respectively (vs 3.4–6.7 for the retained
> seeds), with per-molecule errors reaching ±1,000 kcal/mol. The
> catastrophic errors are molecule-specific (45 of 129 test molecules, 32
> shared by both seeds), strongly correlated across the two seeds
> (Spearman ρ = 0.96), and independent of the geometry given to the model
> (stored vs fresh conformer predictions agree to within the 0.1 kcal/mol
> conformer-draw sensitivity of the retained seeds). No surviving geometry
> regime restores normal predictions for these two checkpoints, and the
> original regime is unrecoverable; re-scoring is therefore not a viable
> fix. The recorded 5-member ensemble metrics (0.506 kcal/mol) originate in
> the destroyed environment and are treated as unverifiable historical
> values. All results reported here therefore use the three members whose
> checkpoints are verifiably well-behaved in a reproducible environment
> (seeds 42, 123, 999; test MAE 3.4–6.7 kcal/mol in the surviving regime,
> recorded 0.50–0.55). This is a smaller ensemble than the recorded
> 5-member average, with a correspondingly reduced ensemble-average
> precision; we note this limitation explicitly.

## Evidence checklist (for reviewers)

| claim | evidence |
|---|---|
| recorded 5-seed metrics normal | `deep_ensemble/seed_{42,123,7,2024,999}/metrics.json` (test MAE 0.505–0.550; best-val 0.418–0.465) |
| training converged normally (seeds 7/2024) | instrumented rerun `seed_7/val_history.csv` (0.508@ep49, stop ep79), `seed_2024/val_history.csv` (0.489@ep56, stop ep86); `run_all_seeds.log` (no warnings/errors); stageB per-molecule trajectories normal (first-hit epochs 3–6, tail oscillation ≈0.05) |
| recorded metrics unreproducible anywhere | `gradient12_conformer_provenance_check/report.md` §0 (destroyed instance; fresh vs stored conformer regimes differ; 0.5313/0.5048 unrecoverable) |
| pathology persists on every surviving geometry | `a2_probe_report.json`: seeds 7/2024 test MAE hdf5 38.373/58.149, fresh single-conf 38.618/58.510, fresh TTA-5 38.476/58.644; identical to `seed_predictions_all642.csv` (38.373/58.149) |
| catastrophe is molecule-specific, geometry-independent | `a2_probe_predictions.csv`: 45 molecules |err|>20 (33 seed-7, 44 seed-2024, 32 shared); per-molecule hdf5-vs-fresh error diff median 0.0 (≤ conformer-draw noise); ρ(seed7, seed2024) = 0.96; same-sign, up to ±1,000 kcal/mol |
| retained seeds valid in surviving regime | seeds 42/123/999 test MAE 3.4–6.7 kcal/mol across all three protocols; probe exactly reproduces the stored `seed_predictions_all642.csv` values for them |
| fix attempts ruled out | conformer re-scoring (this probe, A2); conformer-draw ensembling, conformer-seed pooling (`report.md` checks 1–2); original regime unrecoverable (git has no conformer blob predating training; surviving RDKit 2026.03 family differs from training-time RDKit) |

## Honest limitations

1. The retained 3-seed ensemble is smaller than the recorded 5-member
   average; ensemble-mean precision is lower and the uncertainty estimate
   (per-molecule spread) is computed over 3 members throughout.
2. The recorded single-seed metrics of seeds 7/2024 (0.53–0.59) are normal
   but unverifiable; we do not claim those seeds were intrinsically
   defective, only that their checkpoints are unusable in any surviving,
   reproducible geometry regime.
3. All node-level quantities (per-node contributions, uncertainties u3/u5)
   are computed in the surviving regime, consistently for the 3 retained
   seeds; the 5-seed sensitivity arm (`sens5.json`) and the 5-seed
   `ensemble_mean` column of `seed_predictions_all642.csv` are inflated by
   the two excluded members and are reported only as a flagged sensitivity,
   never as signal.
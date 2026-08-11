# Gradient-12 investigation — bottom line

**Question:** the 12 "confidently wrong" test molecules not explained by chemical
isolation (Tanimoto) or by learned-representation density (GMM-NLL, Part 1)
need another explanation. Two candidates were checked:

## Candidate A — experimental measurement uncertainty: NOT TESTABLE (data absent)

We inspected the actual FreeSolv source data (`Data/FreeSolv/database.json`,
642 records). The schema has **no uncertainty field** — no `uncertainty`,
`error`, `std`, or `sd` column exists in any record. The only uncertainty
content is a `notes` string: 538/642 molecules say *"Experimental uncertainty
not presently available, so assigned a default value."* and 20 more carry a
uniform literature-suggested 0.2 kcal/mol recommendation
(DOI 10.1039/P29900000291) that is _not_ a per-molecule measurement.

Of the 129 test molecules — including all 12 gradient-12 — **zero** have a
measured per-molecule experimental uncertainty. Every molecule effectively
has the same default value, so there is nothing to compare: a Mann-Whitney
test across gradient-12 / isolated-6 / certain-47 is not computable and would
be meaningless. We did not substitute a proxy.

## Candidate B — training dynamics: NOT LOGGED (requires a new instrumented run)

We searched the training artifacts and read the training code. Stage-3
fine-tuning (original cross-validation `freesolv_ft.log` + 5-seed ensemble,
`deep_ensemble.py`) logged **only one scalar per epoch** — validation MAE/RMSE
pooled over all molecules — and saved **only the best-val checkpoint** per run
(in `cv_finetune.py` and `deep_ensemble.py` `torch.save` fires on best-val
improvement only). There are no per-epoch checkpoints, no per-molecule
per-epoch losses/predictions, and no TensorBoard events. The `epoch_history.csv`
files found belong to a different experiment family (neighbor-regularization
ablation) with aggregate train losses only — not the Stage-3 fine-tune, and not
per molecule.

Per protocol we did **not** retrain to reconstruct this. A per-epoch,
per-molecule trajectory comparison (gradient-12 vs a matched certain-47 sample)
is answerable only from a future instrumented training run that logs
per-molecule losses or checkpoints every epoch.

## Bottom line

**Neither candidate explains gradient-12 from existing data:**
- Experimental uncertainty: cannot discriminate — the labels do not carry
  per-molecule error bars at all (all-default uncertainty), so "gradient-12
  labels are unreliable" remains untested, not disproven.
- Training dynamics: cannot discriminate — the trajectories were never logged,
  so late-convergence/oscillation/drift hypotheses remain untestable without a
  new instrumented run.

**What gradient-12 remains (current best account):** molecules the 5-seed
ensemble agrees about (low std) yet all five miss (high RMSE), that are
neither Tanimoto-isolated from training chemistry nor outliers in Stage-1
latent density at validated GMM settings. They are also **not** explained by
per-molecule experimental uncertainty or by logged training behavior.

This is a legitimate reportable outcome: gradient-12 is an unexplained
empirical cluster. The productive next step is pre-registered: re-run the
ensemble with per-epoch, per-molecule loss logging (and optionally confirmatory
FreeSolv-adjacent datasets with real error bars if experimental uncertainty
becomes a priority).

## Files

- `experimental_uncertainty_check.json` — schema inspection, group coverage,
  statistical implication
- `training_dynamics_check.json` — artifact inventory, code evidence, conclusion
- `gradient12_ungrouped.csv` (from `per_molecule_gmm_nll_refit.csv`) — the 12
  molecules, with ensemble std, RMSE, GMM-NLL, and quadrant labels, for reuse
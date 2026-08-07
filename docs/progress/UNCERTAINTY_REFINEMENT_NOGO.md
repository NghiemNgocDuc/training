# Uncertainty-Refinement Experiment: Final NO-GO Report

Status: **CLOSED — all approaches tested are NO-GO.** No code changes to the
production experiment pipeline; all work was sandboxed.

Date: 2026-08-07. Datasets: FreeSolv (642 mols) and FlexiSol-water (297 mols).

## Question

Can a 5-seed DimeNet+ deep ensemble's *disagreement* be used to actively
improve solvation free-energy predictions — either by re-weighted retraining
or by targeted data augmentation?

## Setup (shared)

- 5 members, seeds {42, 123, 7, 2024, 999}, DimeNet+, fold-0 split
  (FreeSolv 411/102/129), MSE-in-eV, Adam lr=1e-4 wd=1e-5, batch 8,
  epochs 200, patience 30, TTA eval. Init from `stage2_correction.pt`.
- FreeSol baseline anchors: seed-42 test MAE 0.5313 (single) / 0.5048 (TTA).
- FlexSol-water: 297 mols, 11 elements (all in the 17-elem vocab), 239/29/29
  split; fresh ensemble, val MAE ~1.7–1.9 kcal/mol (239-train scale).

## Results

| # | Experiment | Signal | Outcome |
|---|---|---|---|
| 0 | Iteration-variance (go/no-go across SGD iters) | Spearman(iter_std, |err|)=0.075 p=0.40; vs ens_std 0.393 (rho²≈0.25) | NO-GO |
| 1 | Alpha-sweep uncertainty re-weighting (seed-42) | α=0.0→0.5507 TTA; α=0.5→0.6857; α=1.0→0.5807; α=2.0→0.5970 | NO-GO (no α beats uniform retrain 0.5507; frozen member 0.5048) |
| 1m | Hard-mask "freeze-the-rest" (top-20%-std, 83/411) | test MAE 1.87 | catastrophic NO-GO |
| 3 | Coverage-based augmentation diagnostic (FlexSol-water, n_test=29) | Spearman(ens_std, 1−max_tanimoto)=**−0.172** p=0.37; vs |err|=0.116 p=0.55 | NO-GO (sign wrong) |

Resource-honest context: control comparisons at the fair point (fresh retrain
from the same init, α=0.0 = 0.5507) — the frozen baseline 0.5048 is not a
fair α=0 reference, because uniform retrain is bootstrapped from scratch, not
from a trained member.

## Why the idea is dead (mechanistic)

- Ensemble std does **not** explain |error| (FreeSol: rho ≈ 0.075–0.12).
- Ensemble std does **not** explain lack of coverage either — on FlexSol the
  correlation is *negative*: high-std molecules are well-covered
  (`dibenzo-24-crown-8`: max_sim = 1.000 yet std=2.05, err=7.80).
- Disagreement is dominated by *hard chemistry* (rigidity, ring strain,
  macrocycles, sugars) and not by coverage gaps or by achievable error.
  Hard-mole errors are instructive already in the input: reweighting the loss
  cannot create signal where the base model + labels are integration-noise-bound.
- Soft weighting (α-sweep) and hard masking (freeze rest) both fail in the
  same direction, which rules out a monotone-re-importance artifact.

## Standing conclusions (what still holds)

- Ensemble **averaging itself helps**: 0.5048 (TTA) vs 0.5313 (single) on
  FreeSol, ~5%. This is the only positive effect of the deep ensemble.
- Uncertainty of the base model is not an active-learing signal for this
  target. Frag20/SPICE/solvation-supplementation is NOT a targeted fix for
  ensemble disagreement (Approach 3 negative on FlexSol).

## Files

- Experiment dir (unchanged, read-only): `aqm-spice2/freesolv/experimental_uncertainty_refine/`
- Sandboxes (delete-free): `flexisol_sandbox/` (fetch/build/espc ensemble,
  approach0, approach1, approach3_coverage, run_vast.sh, README with results)
- Aggregate tables: `flexisol_sandbox/out/ensemble_full/aggregate/per_molecule.csv`
  and `out/approach3/coverage_per_molecule.csv` (gitignored, on Vast).
- Full traces: git history (b903ab1 revert of hard-mask, c021214 etc).

## Recommendation

Stop all uncertainty-refinement approaches. Keep the ensemble-average+TTA
as the deployed gain. Do not build Frag20/SPICE supplement to fix
disagreement — the mechanism argues it will not help. If any future idea
needs testing, add it in a sandbox and reuse the frozen evaluation recipe.
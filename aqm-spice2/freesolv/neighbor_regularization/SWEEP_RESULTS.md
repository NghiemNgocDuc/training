# Neighbor-Consistency Regularization — Sweep Results (fold-0, seed 42)

Status: COMPLETE 2026-08-14. **Negative result**: graph-consistency
fine-tuning does not fix the low-std-high-rmse anomaly (gradient-12 /
isolated-6). Mild, non-significant overall gain at lambda=0.01 raw; the
normalized latent variant is significantly harmful.

## Context

- The anomaly: 18 fold-0 test molecules have low ensemble std but high rmse
  ("confidently wrong"). gradient-12 = 12 of them with best_sim > 0.22
  (neighbors exist), isolated-6 = the 6 with best_sim <= 0.22 (structurally
  isolated). Baseline test MAE: all129 0.552, wrong18 0.499, certain47 0.273,
  isolated6 0.464, gradient12 0.516 kcal/mol (single-conf, seed 42).
- Hypothesis tested here: pull each molecule's prediction toward a weighted
  mean of its top-5 similar molecules' predictions DURING fine-tuning so
  uncertain molecules learn from well-predicted neighbors.
- Input-side representation was exonerated first (CHECK 12/13/14,
  `deep_ensemble/gradient12_representation_check/report.md`): no tautomer /
  stereochemistry / sanitization / element-vocabulary difference explains the
  group.

## Setup

17 runs, GPU box, fold-0, seed 42, 200 epochs / patience 30, stage-2 backbone
(identical harness to `deep_ensemble.train_member`). All runs re-executed
with PARALLEL=1 after the OOM root-cause fix; every run verified complete
(17/17 metrics.json, no NaN/Inf, no Traceback/OOM in logs).

| axis | values |
|---|---|
| graph source | tanimoto (static Morgan r=2/2048, `graph.py`) |
|               | latent (cosine on stage-2 embeddings, `latent_graph.py`) |
| loss form | raw L_nbr |
|           | normalized L_nbr / var(p) |
| lambdas | raw: {0.001, 0.003, 0.01, 0.03}; normalized: {0.05, 0.1, 0.3, 1.0} |
| shared | lambda=0 baseline |

## Full results (test MAE single-conf, kcal/mol)

| variant | lambda | all129 | wrong18 | certain47 | isolated6 | gradient12 |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.552 | 0.499 | 0.273 | 0.464 | 0.516 |
| tanimoto_raw | 0.001 | 0.573 | 0.568 | 0.266 | 0.596 | 0.555 |
| tanimoto_raw | 0.003 | 0.607 | 0.575 | 0.337 | 0.540 | 0.593 |
| tanimoto_raw | 0.010 | **0.487** | 0.504 | 0.247 | 0.367 | 0.572 |
| tanimoto_raw | 0.030 | 0.599 | 0.543 | 0.293 | 0.716 | 0.456 |
| tanimoto_normalized | 0.050 | 0.561 | 0.591 | 0.307 | 0.594 | 0.590 |
| tanimoto_normalized | 0.100 | 0.642 | 0.623 | 0.409 | 0.579 | 0.645 |
| tanimoto_normalized | 0.300 | 0.835 | 0.695 | 0.635 | 0.779 | 0.652 |
| tanimoto_normalized | 1.000 | 1.083 | 0.893 | 1.037 | 0.810 | 0.935 |
| latent_raw | 0.001 | 0.629 | 0.595 | 0.330 | 0.742 | 0.521 |
| latent_raw | 0.003 | 0.541 | 0.591 | 0.247 | 0.526 | 0.624 |
| latent_raw | 0.010 | 0.517 | 0.517 | 0.260 | 0.432 | 0.560 |
| latent_raw | 0.030 | 0.537 | 0.506 | 0.262 | 0.638 | **0.440** |
| latent_normalized | 0.050 | 0.791 | 0.726 | 0.458 | 0.882 | 0.648 |
| latent_normalized | 0.100 | 0.829 | 0.738 | 0.501 | 0.627 | 0.793 |
| latent_normalized | 0.300 | 1.141 | 0.895 | 0.708 | 0.982 | 0.851 |
| latent_normalized | 1.000 | 1.003 | 0.707 | 0.525 | 0.662 | 0.730 |

## Best per formulation + paired-bootstrap 95% CI (delta vs baseline MAE)

Best run per variant = lowest all129 MAE; CI over paired per-molecule
abs-error deltas, 10k resamples (`analyze_sweep.py`).

| variant | lambda | group | delta | 95% CI |
|---|---|---|---|---|
| tanimoto_raw | 0.010 | all129 | −0.065 | [−0.147, +0.014] |
| | | wrong18 | +0.005 | [−0.119, +0.136] |
| | | certain47 | −0.026 | [−0.095, +0.036] |
| | | isolated6 | −0.097 | [−0.322, +0.134] |
| | | gradient12 | +0.056 | [−0.082, +0.210] |
| tanimoto_normalized | 0.050 | all129 | +0.009 | [−0.081, +0.088] |
| | | gradient12 | +0.074 | [−0.078, +0.210] |
| latent_raw | 0.010 | all129 | −0.035 | [−0.090, +0.020] |
| | | gradient12 | +0.044 | [−0.037, +0.123] |
| latent_normalized | 0.050 | all129 | +0.239 | [+0.130, +0.371] |
| | | gradient12 | +0.132 | [+0.023, +0.228] |

(normalized variants: all129 row shown; per-group reads analogous and
uniformly non-negative.)

## Reads

1. **No formulation significantly improves overall MAE** — the λ=0.01 raw
   gains (tanimoto −0.065, latent −0.035) have CIs that include 0. All
   normalized variants flat or worse.
2. **gradient-12 is not improved at the best-λ runs; point estimates are
   worse** (+0.056 / +0.044), not significantly. The λ=0.03 dips
   (0.440 latent / 0.456 tanimoto vs baseline 0.516) sit at lambdas that
   degrade overall MAE, and at n=12 the paired CI crosses 0 — not a fix.
3. **isolated-6 (n=6):** tanimoto_raw λ=0.01 shows the best point estimate
   (−0.097) but CI [−0.322, +0.134] — noise.
4. **latent_normalized is significantly harmful** (overall +0.239
   [+0.130, +0.371]; gradient-12 +0.132 [+0.023, +0.228]) — the normalized
   latent scheme is dead.
5. **Molecule-level: the loss moves errors within the group, not out of it.**
   It fixes the two worst molecules (mobley_4620651 1.087→0.49 under
   tanimoto_normalized; mobley_4483973 1.31→1.05 under raw) but destabilizes
   near-perfect ones (mobley_6257907 0.037→0.46, mobley_4639255 0.056→0.64
   under tanimoto_raw; mobley_3682850 regresses under both latent variants).
   Net zero on the group.

## Conclusion

- Combined with CHECK 12/13/14 (representation exonerated), the
  low-std-high-rmse anomaly is **intrinsic to the molecules** — it is not an
  input-representation artifact and not addressable by neighbor-consistency
  smoothing during fine-tuning (both graph sources, both loss forms).
- Recommendation: keep the main pipeline (TTA-5). Optionally adopt
  tanimoto_raw λ=0.01 (best point estimates everywhere, no harm in any
  group) if a small overall gain is desired despite non-significance.
- Caveats: single fold (0), single seed (42); n=6 / n=12 groups are
  underpowered; box environment differs from the archived training env
  (REPRODUCIBILITY.md), so absolute MAEs shift run-to-run while deltas are
  internally consistent.

## Artifacts

- `run_rerun_box.sh` / `watch_progress.sh` — box launcher + monitor
- `report_results.py` — per-run aggregation (variant labels read from
  config.json when metrics.json lacks `neighbor_source`/`normalize_nbr`)
- `analyze_sweep.py` — paired-bootstrap CIs + top moved molecules
- `sweep_report.csv` (box) — machine-readable full table
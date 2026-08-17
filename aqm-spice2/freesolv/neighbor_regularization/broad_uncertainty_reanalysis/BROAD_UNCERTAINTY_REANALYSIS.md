# Broad-Uncertainty Reanalysis of the Neighbor-Regularization Sweep

Re-analyzes the 17 completed sweep runs (single-seed, seed 42, TTA-5
predictions from `augmented_predictions.csv` — verified identical to
`metrics.json` `test_mae_tta_kcal`, max diff 5.5e-8) on broad "uncertain"
populations instead of the narrow gradient-12 / isolated-6 subgroups.
No new training; no model inference. Script: `broad_uncertainty_reanalysis.py`.

## 1. Uncertain-population definitions (fold-0 test, n=129)

Top quartile = 32 molecules by each criterion (strictly above the 75th pct).

| population | criterion | n | mean baseline ens_std | baseline TTA MAE (kcal/mol) |
|---|---|---|---|---|
| Q_std | top quartile by 5-seed ensemble_std (>0.217) | 32 | 0.402 | 0.948 |
| Q_nll | top quartile by GMM mean-NLL (>27.09) | 32 | 0.239 | 0.734 |
| UNION | Q_std OR Q_nll | 53 | 0.302 | 0.841 |

**Overlap: only 11 molecules (Jaccard 0.21).** The two uncertainty
definitions barely agree — they select nearly disjoint sets. The union
population is where the model is worst overall (MAE 0.841 vs 0.552 all-129).

**Critical: gradient-12 is essentially DISJOINT from the broad uncertain
population** — 0/12 in Q_std, 1/12 in Q_nll, 1/12 in union. This is
structural: gradient-12 was defined as *low*-std-high-rmse ("confidently
wrong"), while Q_std is *high*-std. The two analyses therefore test
different populations, not a subset/superset pair. The narrow gradient-12
analysis did not "miss" a subset of the uncertain population — the groups
barely intersect.

## 2. Per-run MAE on broad uncertain populations (delta vs same-env λ=0 baseline, paired bootstrap 95% CI)

| run | Q_std Δ | CI | Q_nll Δ | CI | UNION Δ | CI |
|---|---|---|---|---|---|---|
| tanimoto_raw 0.001 | +0.131 | [−0.079, +0.366] | +0.015 | [−0.190, +0.236] | +0.048 | [−0.102, +0.208] |
| tanimoto_raw 0.003 | +0.094 | [−0.107, +0.326] | +0.099 | [−0.115, +0.340] | +0.093 | [−0.045, +0.245] |
| **tanimoto_raw 0.01** | −0.044 | [−0.262, +0.191] | −0.114 | [−0.383, +0.151] | **−0.117** | [−0.289, +0.059] |
| tanimoto_raw 0.03 | +0.104 | [−0.075, +0.302] | +0.207 | [+0.005, +0.421] | +0.138 | [+0.006, +0.279] |
| tanimoto_norm 0.05 | +0.002 | [−0.240, +0.223] | +0.003 | [−0.296, +0.267] | −0.018 | [−0.219, +0.161] |
| tanimoto_norm 0.1 | +0.037 | [−0.188, +0.245] | +0.076 | [−0.221, +0.340] | +0.077 | [−0.119, +0.256] |
| tanimoto_norm 0.3 | +0.179 | [−0.096, +0.456] | +0.302 | [+0.043, +0.550] | +0.332 | [+0.134, +0.527] |
| tanimoto_norm 1.0 | +0.329 | [−0.087, +0.820] | +0.367 | [+0.039, +0.682] | +0.449 | [+0.156, +0.773] |
| latent_raw 0.001 | +0.079 | [−0.189, +0.439] | +0.237 | [−0.086, +0.636] | +0.142 | [−0.067, +0.395] |
| latent_raw 0.003 | −0.067 | [−0.247, +0.120] | +0.062 | [−0.120, +0.250] | +0.002 | [−0.130, +0.131] |
| **latent_raw 0.01** | −0.017 | [−0.172, +0.143] | −0.064 | [−0.245, +0.117] | **−0.072** | [−0.197, +0.051] |
| latent_raw 0.03 | −0.056 | [−0.210, +0.100] | +0.076 | [−0.112, +0.265] | +0.019 | [−0.106, +0.149] |
| latent_norm 0.05 | +0.291 | [−0.087, +0.782] | +0.490 | [+0.130, +0.967] | +0.352 | [+0.106, +0.639] |
| latent_norm 0.1 | +0.451 | [−0.010, +1.187] | +0.559 | [+0.122, +1.260] | +0.420 | [+0.125, +0.883] |
| latent_norm 0.3 | +1.105 | [+0.299, +2.424] | +1.111 | [+0.375, +2.334] | +0.908 | [+0.393, +1.720] |
| latent_norm 1.0 | +0.957 | [+0.159, +2.148] | +1.059 | [+0.317, +2.254] | +0.803 | [+0.307, +1.576] |

Full table incl. absolute MAEs: `per_run_broad_mae.csv`.

**No run significantly improves any broad uncertain population.** The best
point estimates (tanimoto_raw λ=0.01: UNION −0.117; latent_raw λ=0.01:
−0.072) are the same two runs flagged in the narrow analysis, and their CIs
cross zero. All normalized variants ≥0.3 are significantly harmful on Q_nll
and UNION; latent_normalized is harmful even at λ=0.05.

## 3. Std before vs after — the "converging on each other" check

**Data limitation, stated plainly:** the sweep runs are single-seed (seed 42
only — `config.json`), so a true 5-seed *post-regularization* ensemble_std
does not exist. The flagged risk (apparent gains coming from seeds agreeing
with each other rather than with truth) cannot be measured directly.
Delivered instead — three independent proxies, all pointing the same way:

**(a) Population prediction spread across the 53 union molecules.**
Baseline spread 4.861 kcal/mol. After: 4.35–6.22 across the 17 runs.
The improving runs *preserve or widen* spread (tanimoto_raw 0.01 → 4.980,
latent_raw 0.01 → 4.939); only harmful high-λ normalized runs shrink it
(tanimoto_norm 1.0 → 4.354). Predictions did not collapse toward a common
value in any run that improved error.

**(b) Shift-vs-error Spearman on union (did movement track truth?).**
Improving runs show *negative* rho (bigger move → bigger error reduction):
tanimoto_raw 0.01 ρ=−0.217 (p=0.12), latent_raw 0.01 ρ=−0.248 (p=0.07).
Harmful normalized runs show strongly *positive* significant rho
(+0.44 to +0.70, p<0.01): bigger moves → bigger error *increase*. Movement
is directional, not a random walk toward mutual agreement.

**(c) Cross-run consensus (16 variants as pseudo-seeds) vs baseline 5 seeds.**
Per-molecule: cross-run std (16 variants) mean **0.481** vs baseline 5-seed
std mean **0.188** — the regularized family does NOT agree more than the
baseline ensemble; it disagrees ~2.5× more. And its consensus is *worse*:
cross-run mean MAE 0.632 vs baseline ensemble 0.506 on all-129; on union
0.951 vs 0.828. Where the variants do agree, they agree on worse answers.

**Verdict on the flagged risk:** no evidence that any apparent improvement
comes from seeds converging on each other. If anything the regularization
*inflates* dispersion; the mild gains in raw λ=0.01 runs coincide with
moves toward the experimental values (mean toward-truth movement among
improved molecules: +0.445 tanimoto, +0.413 latent), not toward each other.

## 4. Was there a broad improvement the narrow analysis missed?

**No.** Two conclusions:

1. The best runs on the broad populations are the same two raw λ=0.01 runs
   that looked best on gradient-12/isolated-6 — no new winner surfaced, and
   their broad-population CIs all include zero.
2. The premise of "gradient-12 = a subset of the top-quartile uncertain
   population" is false in the data: 0/12 gradient-12 molecules are in
   Q_std (they are low-std by definition) and 1/12 in Q_nll. The narrow
   analysis and this broad analysis test nearly disjoint sets, and both
   conclude the same thing: neighbor-consistency regularization does not
   fix the model's worst, most uncertain predictions.

## Conclusion

- The 17-run sweep verdict is unchanged under broad uncertainty
  definitions: no significant improvement anywhere; raw λ=0.01 runs are
  best-but-null; normalized variants, especially latent, are harmful.
- The uncertainty-signal-integrity concern (seeds merely converging on each
  other) is not supported by any of three independent proxies.
- Notable side finding: **"confidently wrong" (low std, high error) and
  "uncertain" (top-quartile std/NLL) are nearly disjoint populations** — a
  fact the earlier analyses implicitly assumed away, and which any future
  uncertainty-repair method (e.g. targeted augmentation) must design around.

## Artifacts

- `broad_uncertainty_reanalysis.py` — the analysis script (17 runs, 10k
  paired bootstrap CIs, Spearman diagnostics, cross-run consensus)
- `uncertain_population.csv` — per-molecule quartile membership + overlap
- `per_run_broad_mae.csv` — per-run MAE / delta / CI on Q_std, Q_nll, UNION
- `convergence_diagnostics.csv` — per-run shift, spread, rho, movement
- `cross_run_consensus.csv` — per-molecule cross-run stats vs baseline ensemble
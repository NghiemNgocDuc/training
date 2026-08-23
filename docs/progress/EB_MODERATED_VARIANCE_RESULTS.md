# Empirical-Bayes Moderated Variance (Part C'') — results

Date: 2026-08-22. Script: `aqm-spice2/freesolv/node_refinement/eb_moderated_variance/eb_moderated_variance.py` (38.6 s). All outputs in `node_refinement/eb_moderated_variance/` (`eb_moderated_report.json`, `eb_moderated_at_star.csv`, `eb_moderated_contrasts.csv`, `eb_moderated_bias_variance.csv`, `eb_moderated_h2_holdout.csv`, `eb_moderated_lambda_diagnostics.csv`, `run.log`).

## Question

Math-advisor proposal: shrink per-atom variance before VW as

```
tilde_i = (d0*s0^2 + (K-1)*sigma2_hat_i) / (d0 + K - 1)
lambda_i = tilde_i / (tilde_i + tau^2)
x_hat_i = lambda_i*mu + (1 - lambda_i)*x_i
```

with K=3 => `sigma2_hat|sigma2 ~ Exp(mean=sigma2)` exactly (chi2_2 = Exp), CV=1. Place inverse-gamma prior `sigma2 ~ IG(d0/2, d0*s0^2/2)` => marginal `sigma2_hat ~ Lomax(alpha=d0/2, beta=d0*s0^2/2)`. Fit `(d0, s0^2)` by MLE on the pooled sigma2_hat (not by 102-mol val-MAE grid; archived `variance_moderated_shrinkage` already found that grid optimum is `d0*=0`, `tau2*=4.7254e-04`). Then recalibrate `tau^2` on the same 102 VAL molecules over the identical archived 37-pt grid. Distinct nrow seeds (1100/1200/1300 + mode*5 + pop_i), N_BOOT=10_000, SEEDS3=[42,123,999], D=2, u3 gate 0.09716.

## Diagnostic — advisor premise fails on this pool

Pooled over all 642 molecules x 3 seeds: n=11,613 atoms. Normalized sigma2_hat: CV=12.93 (bench 1.0), skew 25.58 (bench 2), excess kurtosis 895 (bench 6). KS vs Exp: stat 0.858, p~0. MOM d0 = 4*CV^2/(CV^2-1) = 4.024. Direct takeaway: the Exp marginal is rejected at machine precision; the pool is heavy-tailed, dominated by a small number of huge variances (max 416.8, mean 0.687, median 0.003296).

Simulation check (ground-truth Lomax alpha=6.5 beta=0.3): recovered alpha 6.18 beta 0.283 d0 12.36 — optimizer is sound.

## MLE fits of (d0, s0^2) on sigma2_hat

- **Full pool (n=11,613):** d0=1.581 alpha=0.791 s0^2=0.003109 beta=0.002458 nll -40743.88 KS Lomax 0.040 p 1.1e-16 (still rejected; Lomax cannot absorb the tail).
- **VAL atoms only (n=1,928):** d0=1.899 alpha=0.950 s0^2=0.003107 nll -7174.35 KS 0.039 p 0.006 (marginal).
- **Mean-fixed (s0^2=0.6867):** d0=1.593 nll -2479 (much worse; mean is outlier-driven).
- **Median-fixed (s0^2=0.003296):** d0=1.611 nll -40738.74 (essentially same as free).
- **2D val-MAE grid (reproduced):** optimum d0=0 tau2=4.725e-04 val_delta -2.3975 — moderation hurts val-MAE under the honest criterion.

Gates: d0=0 max |lambda diff| =1.1e-16 (identity); large d0 lambda std 0.000895 (uniform as expected).

## EB recalibration on 102-VAL molecules

Using d0_hat=1.581 s0^2=0.003109, tau2* = 8.958e-04 (idx 11) vs VW tau2*=4.725e-04 (idx 10). Val delta at star = -2.3687 vs VW -2.3513 at idx10; the 2D grid optimum (-2.3975 at d0=0) still wins. Val curve is flat 0-11 then falls off after idx 12. Taking the MLE literally costs ~0.03 MAE on VAL vs the val-optimum — i.e., moderation does not improve VAL even after refitting tau.

## 5-pop x 2-mode results at EB tau2* (10k bootstrap)

Mode A = pooled-seed mean then MAE; Mode B = seed-then-bootstrap affine (deterministic deltas identical, CIs differ only by historic noise band). All deltas vs raw P are strongly negative except gradient-12 (n=12, noise).

| pop | n | ΔMAE (EB) | before | after | 95% CI (A) |
|-----|---|-----------|--------|-------|------------|
| Q_std | 33 | -6.753 | 10.465 | 3.712 | [-13.03, -2.24] |
| Q_nll | 33 | -6.739 | 10.276 | 3.536 | [-12.93, -2.30] |
| UNION | 50 | -4.652 | 8.294 | 3.642 | [-8.88, -1.69] |
| all129 |129| -1.983 | 4.836 | 2.853 | [-3.67, -0.76] |
| grad12| 12 | -0.421 | 2.914 | 2.493 | [-1.54, +0.62] |

Mode B deltas identical to 1e-12 (CI band [-12.78,-2.31] Q_std etc — historical).

Paired contrasts (10k bootstrap, posterior win = P(Δ<0)):

- **EB - VW:** Q_std -0.028 [-0.210,+0.132] pwin 0.623; Q_nll -0.010 [-0.116,+0.089] 0.560; UNION -0.046 [-0.177,+0.074] 0.756; all129 -0.013 [-0.090,+0.059] 0.624; grad12 +0.091 [-0.030,+0.212] 0.068. All CIs cross 0 — EB indistinguishable from plain VW.
- **EB - uniform:** Q_std -0.881 [-2.11,-0.030] 0.982; Q_nll -0.839 [-2.07,+0.011] 0.972; UNION -0.558 [-1.36,+0.004] 0.973; all129 -0.224 [-0.553,+0.002] 0.973; grad12 +0.052 [-0.075,+0.179] 0.217. Same pattern as VW: atomwise beats uniform on the four real pops, not on grad12.

Bias-variance (exact before/after, identity err <2e-12): EB mirrors VW almost exactly. Fraction of ΔMSE from b^2 ~0.66 on Q_std/UNION/all129 (bias-driven). v_after ~0.54-0.90 on Q sets vs VW 0.55-0.87; grad12 v increases slightly (0.61) and b^2 frac >1 (MSE increased under uniform accounting noise). No variance collapse beyond VW.

## H2 holdout (frozen split, fit on H1 atoms only)

H1 (64 mols) fit: d0_h1=2.311 s0^2=0.003788 (median-fixed would be 0.00329); tau2_h1 fixed at archived 4.725e-04. H2 = 65 mols (Q_std 21, Q_nll 16, UNION 26, grad12 5). EB_H2 deltas larger because H2 is harder:

- Q_std -8.874 [ -18.16,-2.43] (21 mols)
- Q_nll -11.217 [-23.11,-2.87] (16)
- UNION -7.297 [-14.95,-2.00] (26)
- all129 -2.916 [-6.20,-0.70] (65)
- grad12 +0.539 [-0.79,+1.50] (5, noise)

Directionally identical to main; CIs still wide on small n. No claim of moderation helping — the same d0*=0 grid would win on H1 VAL too.

## Gate instability (context, not EB-specific)

2000 draws of g_mi with mean 0: g12 (203 atoms, 28 gated now) mean flip 14.8% (hist 44.8% flip 1, 33% flip 2, 21% flip 3, 5% flip >=5); overall mean flip 13.9%, gated atoms 27.7% vs ungated 9.2%; gated-set Jaccard 0.566 [0.553,0.579]; mean gated count 35.9 vs 28 now — VW's top-quartile hard gate is unstable under null shifts. EB inherits the same λ instability (same sigma2_hat input) but does not fix it.

## Verdict

Negative result. The inverse-gamma → Lomax EB story is misspecified on this FreeSolv pool (CV 12.9 vs 1, KS p ~0/1e-16). MLE-fitted moderation (d0≈1.6–1.9) does not improve over plain VW: paired EB-VW deltas are ~0 ±0.1 with win probs 0.56–0.76, all CIs crossing 0; val-MAE still prefers d0=0. If reported in the paper, it belongs as an appendix negative control ("MLE-moderated VW ≈ VW; val-MAE selects no moderation"), not as a main-method upgrade. The honest next step, if any, is a different prior (e.g. log-normal mixture, Student-t) or direct shrinkage on mu rather than on sigma2, not tuning d0 under Lomax.

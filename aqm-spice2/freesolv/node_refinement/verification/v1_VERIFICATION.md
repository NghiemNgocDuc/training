# Ch2.5 Verification Report (v1)

Date: 2026-08-15. Scope: node-level uncertainty-gated refinement (approach2) on FreeSolv.
Inputs: `experimental_uncertainty_refine/output/approach2_node_refine/` (node_contributions.csv,
results.csv, report.json, val_calibration.json, diagnostics.json) and
`deep_ensemble/repair_data/seed_predictions_all642.csv`. Replication script: `v1_verify.py`
(transductive + `--inductive`). All numbers below are recomputed from primary CSVs, not copied.

## Part 0 — exact replication (prerequisite)

| check | result |
|---|---|
| node sums == molecule preds (642 × 5 seeds) | max diff 1.25e-04 (CSV rounding) |
| gate | 2904 nodes, u3 >= 0.097158 (matches) |
| populations | 33 / 33 / 50 / 12 (matches) |
| gated nodes w/o neighbor (min_sim 0.2) | 0 / 2904 |
| bootstrap table vs saved results.csv (30 rows × 10,000) | max |Δdelta| = 2.78e-07 |
| baseline = 3-seed mean (pred_seed42/123/999) | all129 base 4.83584975 (matches report.json; CSV `ensemble_mean` column is the polluted 5-seed version = 20.5, NOT used) |
| all alphas in val_calibration.json | 1.0 (matches) |

Verdict: **PASS** — the saved results.csv is fully reproducible from primary artifacts.

## Part 1 — real correction vs consensus artifact

Per-molecule (UNION, n=50), trust arm (A):
- before MAE 8.294 -> after 5.934. Deltas: q0.1 −2.65, q0.25 −0.60, median +0.012, q0.75 +0.77, q0.9 +1.82. Count 25 improved / 25 worsened, but improvement magnitudes dominate (asymmetry).
- Spearman(delta, error_before) = −0.339: larger pre-error -> larger improvement (right direction).
- Improvement concentration: worst-20% pre-error quintile carries 89.1% of total improvement; worst-50% carries 95.2%.
- Spread: mean per-node ensemble spread 4.907 -> 1.492 (−69.6%) at molecule level.
- Node shifts: median 0.346, q0.99 9.06, max 16.76 (right-skewed tail) — large shifts are rare, concentrated in few nodes.
- Spearman(node_shift, pre-outlierness) = +0.308 — nodes deviating from their molecule mean shift most.
- 27/50 molecules pulled toward truth, 23 away; magnitude asymmetry makes net negative.

A pure consensus/artifact effect would produce small, symmetric shifts uncorrelated with pre-error.
Observed: shifts scale with pre-error and concentrate in the worst molecules -> correction, not artifact.

Verdict: **PASS**

## Part 2 — bootstrap CIs (10,000 resamples, percentile 2.5/97.5)

A-trust: Q_std −3.807 [−8.41, −0.34] excl 0; Q_nll −3.819 [−8.37, −0.38] excl 0; UNION −2.359
[−5.52, −0.08] excl 0; all129 −0.760 [−1.99, +0.17] **NS**; gradient12 +0.675 [+0.01, +1.48]
excl 0, **harmed**. State plainly: 3/5 populations show a significant correction; the all-129
population does not (CI crosses zero); gradient12 is significantly worsened. 15 A-mode tests;
3 exclude 0; all 3 survive Bonferroni (0.05/15). CIs reproduced from primary data to 2.8e-07.

Verdict: **PASS** (exact reproduction; NS/harm stated plainly)

## Part 3 — controls fair (UPDATED 2026-08-15, strengthened random control)

First pass: naive (same top-k neighbors, t_j=1) and old sign-placebo random were both null on
Mode A. Follow-up audit (`part3_strengthened/part3_strengthened.py`) replaced the sign placebo
with neighbor-IDENTITY randomization: same gated nodes, same trust formula, same bootstrap;
only WHICH neighbors enter changes. Four new arms + one diagnostic:

| arm | neighbors | weights | ΔMAE Q_std | ΔMAE Q_nll | ΔMAE UNION | ΔMAE all129 | ΔMAE grad12 |
|---|---|---|---|---|---|---|---|
| trust (real method) | top-k similar (sim 0.95) | w=sim, t=trust | −3.807 [−8.39,−0.36] | −3.819 [−8.45,−0.34] | −2.359 [−5.56,−0.05] | −0.760 [−2.01,+0.16] NS | **+0.675 [+0.00,+1.47]** |
| naive | top-k similar | w=sim, t=1 | −0.792 NS | −0.538 NS | −0.242 NS | +0.042 NS | +0.540 NS |
| old random (sign placebo) | top-k similar | sign flips | +0.733 NS | +0.692 NS | +0.536 NS | +0.187 NS | +0.017 NS |
| randElig_trust | random from eligible pool (sim 0.60) | w=sim, t=trust | **−6.118 [−12.33,−1.55]** | **−6.049 [−12.38,−1.54]** | **−3.992 [−8.26,−0.96]** | **−1.495 [−3.20,−0.29]** | +0.313 NS |
| randAll_trust | random from ALL nodes (w=1) | t=trust | **−6.012 [−12.10,−1.50]** | **−5.936 [−12.32,−1.49]** | **−3.912 [−8.09,−0.90]** | **−1.490 [−3.22,−0.28]** | +0.216 NS |
| randElig_equal | random from eligible pool | w=sim, t=1 | **−5.906 [−11.88,−1.47]** | **−5.809 [−11.86,−1.41]** | **−3.805 [−7.92,−0.86]** | **−1.474 [−3.14,−0.27]** | +0.348 NS |
| randAll_equal (strictest placebo) | random from ALL nodes | w=1, t=1 | **−6.197 [−12.26,−1.76]** | **−5.817 [−12.10,−1.30]** | **−3.972 [−8.19,−0.98]** | **−1.568 [−3.25,−0.37]** | +0.055 NS |
| shrink_poolmean (diagnostic) | none — gated node value replaced by pool mean | — | **−6.304 [−12.48,−1.83]** | **−6.065 [−12.30,−1.57]** | **−4.103 [−8.28,−1.07]** | **−1.629 [−3.31,−0.42]** | +0.217 NS |

Mode A shown; Mode B identical for all linear-refinement arms (α=1 ⇒ A≡B), verified.

**Result: the strengthened random controls are NOT null — 8/10 CIs exclude 0 for every new arm
(both modes), and the strictest placebo (equal weights, random neighbors from all nodes)
improves Q_std/Q_nll/UNION/all129 MORE than the trust method.** The hoped-for outcome did not
occur; per the audit protocol this must be reported plainly.

**Mechanism (diagnostics in part3_strengthened.json):** improvements concentrate in the
worst-error molecules (spearman(err_before, delta) = −0.54…−0.68 across arms on UNION/Q_std).
The deterministic `shrink_poolmean` arm — which does NO neighbor machinery at all, merely
replacing each gated node's value with the pool mean — matches or beats every random arm and
beats the trust method. Conclusion: the repair is **value shrinkage at high-u3 nodes** (their
seed-averaged values are overdispersed; pulling them toward the pool mean reduces error on
average — a James-Stein/regularization effect). Neighbor similarity and trust-weighting are not
the drivers; random-neighbor averaging is a noisier implementation of the same shrinkage, and
it additionally avoids the trust method's gradient12 harm (all new arms NS there).

Verdict: **REVISED.** Part 3 machinery is fair and correct, but the strengthened control
falsifies the causal claim "improvement comes specifically from trust-weighting applied to
genuinely similar neighbors". The paper must (i) state that equal-weighted random-neighbor
averaging and pure pool-mean shrinkage of gated nodes reproduce or exceed the headline repair
on every population, (ii) add a shrinkage/zero-out baseline to the comparisons, and (iii) reframe
the contribution as "gating identifies nodes worth regularizing; the specific transfer mechanism
is not validated" — with gradient12 as the only population where the trust mechanism is
distinguishable (it is uniquely harmful there).

## Part 4 — transductive vs inductive scope

- Transductive: test gated nodes borrow neighbors: 67.6% train / 16.4% val / 16.1% test. The
  method scores a batch of molecules together; test nodes DO borrow from other test nodes.
- Inductive rerun (train-only pool, 7,389 pool nodes): A-trust Q_std −4.430 [−9.85, −0.56],
  Q_nll −4.440 [−9.79, −0.57], UNION −2.764 [−6.51, −0.17], all129 −0.923 [−2.36, +0.13] NS,
  gradient12 +0.752 [+0.07, +1.51] harmed. Same verdict as transductive; effect is slightly
  stronger, not weaker.

Verdict: **PASS** — the effect is not an artifact of test-node leakage.

## Part 5 — seeds 7/2024 diagnosis

- Node-level: seed7 mean −2.47, std 8.36, 72 nodes |P|>50 (max 117.2); seed2024 mean −3.45,
  std 11.12, 155 nodes |P|>50 (max 113.5). Seeds 42/123/999: 0 nodes >50 (max_abs 7.4/39.4/16.1).
- Molecule-level: seed7 MAE 42.8 (135 mols with error >50), seed2024 MAE 64.7 (168); seeds
  42/123/999 MAE 3.53/6.54/4.33.
- Catastrophic molecules: 25 (seed7) / 46 (seed2024); NONE in gradient12 (frac 0.0), none in
  isolated-6. Shape: mixture of whole-molecule shifts (mobley_1278715: all 8 atoms >50) and
  few-atom spikes (4–37% of atoms).
- Training: rerun logs show normal convergence (seed7 0.508@ep49, seed2024 0.489@ep56); the
  pathology is at inference time — stored-box-conformer inference (check1: stored single-conf MAE
  3.4752 vs training-time 0.5313). Seed-42 original log not retained.
- Consequence: SEEDS3 = {42, 123, 999} for headline; the 5-seed `ensemble_mean` column and 5-seed
  sensitivity are inflated by the two anomalous seeds and are flagged, not reported as signal.

Verdict: **PASS** — seeds 7/2024 diagnosed at node level; excluded correctly; sensitivity
statements in the skeleton must carry the flag.

## Part 6 — gradient-12 mechanism

- 11/12 have gated atoms; 8 worsened / 4 improved; mean delta +0.675 (matches headline).
- Worst 3 (mobley_4883284 +3.97, mobley_6257907 +2.61, mobley_1449384 +1.35) carry 83.5% of total
  worsening.
- Neighbor profiles of worsened molecules: neighbors are low-u3 (trustworthy-looking) and
  chemically distant — e.g. mobley_6257907: 40% train neighbors, neighbor u3 median 0.123;
  mobley_4883284: 60% train, u3 median 0.038; mobley_1449384: 70% train, u3 median 0.066.
- Mechanism statement: the trust gate scores PREDICTED confidence (u3), not chemical relevance;
  large trust-weighted pulls toward chemically dissimilar neighbors move the molecule away from
  its own truth. gradient12 molecules sit in low-u3, dissimilar-neighbor regions — the trust
  signal misfires there by construction.

Verdict: **PASS** — mechanism stated; harm is real, reproduced in both modes, and attributable
to neighbor chemistry vs trust mismatch.

Follow-up refinement (2026-08-15, atom-level analysis, `gradient12_mechanism/`): see Part 8.

## Part 7 — calibrated shrinkage (follow-up; new candidate "best method")

`shrinkage_calibrated/shrinkage_calibrated.py`. Operation: replace each gated node value with
P'_i = (1−λ)·P_i + λ·μ (μ = per-seed pool mean over all 11,613 transductive nodes; μ̄ = −0.177).
Grid λ ∈ {0, 0.1, …, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0}, same 5 populations, 10,000 paired percentile
bootstrap CIs per cell (rng2 stream per λ row; A≡B verified to 7e-16).

- Val calibration (102 val molecules, same split discipline as α selection): val ΔMAE is
  monotone in λ to −1.8048 at λ=1.0, then rises (−1.65, −1.48, −0.97, −0.01 at 2.0) — λ* = 1.0 is
  a genuine interior optimum, not a grid-boundary artifact. The extended grid settles this.
- λ* = 1.0 equals the part3 `shrink_poolmean` arm by construction; the calibration makes it an
  honest single-hyperparameter method rather than a hand-picked diagnostic.
- Test populations at λ*: Q_std **−6.304 [−12.48, −1.84]**, Q_nll **−6.065 [−12.30, −1.53]**,
  UNION **−4.103 [−8.33, −1.08]**, all129 **−1.629 [−3.33, −0.44] (significant — vs the trust
  method's NS −0.760)**, gradient12 +0.217 [−0.26, +0.75] **NS**.
- Improvement is monotone in λ on all positive populations (Q_std −0.67 at λ=0.1 … −6.30 at
  λ=1.0); gradient12 is NS at EVERY λ (best in-sample −0.044 at λ=0.4, not significant).
- In-sample per-population best λ = 1.0 for 4/5 populations (0.4 only for gradient12, NS
  everywhere) — no in-sample/out-of-sample tension.
- vs the trust method: calibrated shrinkage beats trust on every population and is neutral
  (rather than harmful) on gradient12.

Verdict: **calibrated shrinkage (λ* = 1.0, full pool-mean replacement of gated-node values) is
the new candidate best method.** It is simpler, better on all populations, and its one
hyperparameter is calibratable on val. Note the boundary at λ=1.0 (full replacement); λ>1
overshoots and is worse.

## Part 8 — gradient-12 mechanism, atom level (follow-up)

`gradient12_mechanism/gradient12_mechanism.py`. For all 28 gated atoms of the 12 molecules (vs
218 gated atoms of the Q_std population): per-atom original 3-seed value, trust replacement
(top-10 neighbors, w=sim, t=trust), shrinkage replacement (λ*=1.0), neighbor u3 / similarity /
mean value, and molecule error direction. Answers to the audit question:

| question | gradient12 | Q_std (comparison) |
|---|---|---|
| neighbor reliability (median u3 of top-10) | 0.063 ≈ pool median 0.057 (45% of neighbors below pool median) | 0.084 (28% below) — g12 neighbors are NOT unusually unreliable; if anything more "reliable" |
| neighbor similarity | 0.998 | 0.922 — g12 neighbors are essentially identical molecules |
| correlated-error index of trust replacement (Σ(P'_i−μ̄)·sign(err)/Σ\|P'_i−μ̄\|) | **+0.09 ≈ 0** — replacement values are NOT offset along the molecule's error | +0.64 — Q_std replacements are offset along the error but massively reduced in magnitude |
| molecules where trust pull opposes the gap | 4/11 | 18/33 |
| max \|gated-sum\| (orig) | 4.09 | 85.2 |
| mean per-atom \|replacement−orig\|: trust / shrink(λ*) | 0.63 / 0.45 | — |

Mechanism (refines Part 6 and SUMMARY.md): g12's confident wrongness is NOT carried by the gated
atoms (gated sums ≤ 4.1 kJ/mol vs up to 85 for Q_std) — it lives in the non-gated contributions.
The gated atoms' values sit near the pool mean, so ANY rewrite of them is movement without
information (alignment ≈ 0): 8/11 harm under trust, 7/11 under shrinkage. Trust harms more than
shrinkage only because it moves the values more (0.63 vs 0.45 per-atom). The refinement channel
is orthogonal to g12's error source. Consistent with SUMMARY.md's "confidently wrong, not
coverage-driven"; training-dynamics causation remains untestable (never logged).

Verdict: **PASS** — mechanism now quantified atom-by-atom; the trust method's g12 harm is
"movement without information", not unreliable or misleadingly-similar neighbors (both
hypotheses falsified at atom level).

## Part A — seed-resolution audit: 3-seed vs 5-seed (follow-up 2026-08-16)

`node_refinement/seed_resolution/` (`a2_probe.py`, `a2_probe_report.json`,
`a2_probe_predictions.csv`, `3seed_justification.md`). Pre-write-up audit
asking: can the ensemble be restored to 5 members, or is the 3-seed regime
the defensible one?

**A1 — training convergence (re-verified with per-epoch logs):** instrumented
rerun `val_history.csv` (same split/init/arch): seed 7 floors at 0.508 @
epoch 49, seed 2024 at 0.489 @ epoch 56, early-stop at patience 30 (epochs
79/86), train MSE → ~1e-5 (eV)²; recorded `metrics.json` best-val 0.418@37 /
0.465@23 — competitive with retained seeds (0.443/0.451/0.426). No warning/
error/nan in `run_all_seeds.log`; stage-B per-molecule trajectories normal.
The recorded test MAEs (0.537/0.536) exist only in the destroyed training
environment. **PASS: the model itself converged normally; the pathology is
inference/geometry-regime-specific.**

**A2 — can re-scoring fix seeds 7/2024? DECISIVE: NO.** Original checkpoints
× {stored hdf5, fresh ETKDGv3+MMFF single-conf, fresh TTA-5} on 129 test:

| seed | hdf5 | fresh single | fresh TTA-5 | (repair CSV, existing) |
|---|---|---|---|---|
| 42 | 3.475 | 3.392 | 3.398 | 3.475 |
| 123 | 6.659 | 6.588 | 6.584 | 6.659 |
| **7** | **38.373** | **38.618** | **38.476** | 38.373 |
| **2024** | **58.149** | **58.510** | **58.644** | 58.149 |
| 999 | 4.529 | 4.423 | 4.458 | 4.529 |

Catastrophe is molecule-specific (45/129 test molecules, 32 shared),
same-sign, Spearman ρ(7,2024)=0.96, and geometry-independent (per-molecule
hdf5 vs fresh error difference median 0.0, ≤ conformer-draw noise). The
training-time geometry regime is unrecoverable (destroyed instance; no
pre-training conformer blob in git; surviving RDKit family differs). A3
(5-seed retraining) skipped: not requested and changes the ensemble
definition.

**A4/A5 — 3-seed regime justified, verbatim-ready:** see
`3seed_justification.md` (methods paragraph + reviewer evidence table +
honest limitations). PAPER_SKELETON.md §2.5 Method now carries the
Seed-resolution note. Verdict: **PASS** — 3-seed {42,123,999} is the
reproducible ensemble; 5-seed numbers are flagged historical values, never
signal.

## Part B — holdout validation of λ*=1.0 (follow-up 2026-08-16)

`node_refinement/holdout_validation/` (`b8_holdout.py`, `b8_holdout_report.json`,
`b8_holdout_bootstrap.csv`, `b8_holdout_per_molecule.csv`, `b8_holdout_contrasts.csv`,
`b8_split.json`). Audit question: is the calibrated-shrinkage result overfit
to the five fold-0 populations?

**B6 — was any population held out? NO.** All five populations (Q_std,
Q_nll, UNION, all129, gradient12) were reused across the entire comparison
chain (trust eval, naive/random controls, 2×2 matrix, λ grid). λ* itself was
selected on the 102 VAL molecules (never on test outcomes), but the
method-selection chain reused the same test set repeatedly — a real
overfitting risk. This check is a retrospective fix to the 17-run design,
stated honestly as such.

**B8 — genuine held-out evaluation (H1-calibrated, H2-evaluated):** fold-0's
129 test molecules split (rng 20260816, fixed/documented): H1 n=64, H2 n=65;
H2 used in NO prior analysis. FULL calibration re-run on H1 only: population
gates (std3 ≥ 0.930, NLL ≥ 27.177 — H1 quantiles) and λ* recalibrated on H1
molecules → **λ*_H1 = 1.0, identical to the original val-selected λ*** (the
calibration curve's minimum is stable across disjoint data).

> **CORRECTION (2026-08-16):** the first version of this script summed
> per-seed values into mean-space arithmetic (factor-3 error) in all four
> arms; caught by cross-validation against Part 8's per-molecule deltas
> during the generalization check. All Part B outputs were regenerated with
> mean-space arithmetic and now match Part 7/8 exactly (cross-check
> 1.4e-14). Conclusions are unchanged in substance (λ*_H1=1.0; shrink sig on
> 4/5; shrink−trust sig on all five; shrink ≥ placebo); the corrected trust
> arm is NS on allH2 (was spuriously "sig harmful"), and the naive arm is
> neutral on H2 (was spuriously "sig harmful").

H2 results (10k paired bootstrap, untouched half; corrected arithmetic):

| population | n | shrink @λ* | trust | naive | randAll_equal |
|---|---|---|---|---|---|
| Q_std | 21 | **−8.22** [−17.76,−1.67] | −5.22 [−12.03,−0.18] | −0.39 NS | −8.14 [−17.49,−1.59] |
| Q_nll | 16 | **−10.73** [−22.75,−2.34] | −6.50 NS | −0.06 NS | −10.52 [−22.16,−2.11] |
| UNION | 26 | **−6.65** [−14.21,−1.35] | −4.08 NS | −0.15 NS | −6.46 [−14.30,−1.06] |
| allH2 | 65 | **−2.70** [−6.02,−0.50] | −1.44 NS | +0.10 NS | −2.63 [−5.84,−0.41] |
| gradient12 | 5 | −0.10 NS | **+0.75** [+0.09,+1.73] | +0.47 NS | +0.03 NS |

Paired contrasts on H2: **shrink − trust significant on ALL FIVE
populations** (Q_std −3.00 [−6.16,−0.64], Q_nll −4.23 [−8.06,−1.27],
UNION −2.58 [−5.09,−0.69], allH2 −1.26 [−2.37,−0.46], gradient12 −0.85
[−1.56,−0.44]); shrink − naive significant everywhere; shrink − randAll_equal
negative on all five but NS (shrink ≈ the strictest placebo); **trust is
significantly WORSE than the placebo on all five populations** (trust − randAll
positive, CI excludes 0 everywhere). "Worse" means it improves *less*, not that
it harms accuracy: trust's own deltas vs the no-correction baseline are negative
on 4/5 populations (Q_std −5.22 [−12.03,−0.18] sig; Q_nll −6.50, UNION −4.08,
allH2 −1.44 NS — trust genuinely beats doing nothing; the placebo is simply a
stronger shrinker), and only gradient12 (+0.75 sig) is genuinely harmful. The
falsified premise is the similarity/trust-selection causal claim, not trust's
effect. This confirms on unseen data that the trust
method's relative shortfall is real. gradient12 H2 n=5 (the fixed 12-molecule list
landed 7/12 in H1) — underpowered; direction consistent (shrink neutral, trust
harmful).

**B9 — verdict: PASS.** λ*=1.0 survives genuine holdout: significant
improvement on 4/5 H2 populations incl. allH2; decisively better than the
trust method (paired, all 5 populations); equal-or-better than the strictest
placebo; and λ* itself is stable under full recalibration on disjoint data.
The candidate-best-method claim stands, now with an out-of-sample
validation. Remaining caveats: gradient12 H2 n=5 (underpowered, direction
consistent); the split is a retrospective fix (prospective design would
pre-register the split before the original runs).

## Part 9 — gradient-12 mechanism generalization check (follow-up 2026-08-16)

`gradient12_mechanism/generalization_check/` (`generalization_check.py`,
`generalization_per_molecule.csv`, `generalization_low_ratio_molecules.csv`,
`generalization_report.json`). Does the Part 8 gated-sum finding predict
correction resistance across ALL 129 fold-0 test molecules, or is gradient-12
an isolated case? Ratio = |gated-sum of gated atoms| / pre-correction error;
correction = calibrated shrinkage at λ*=1.0 (per-molecule deltas identical to
Part 7/8 arithmetic, cross-checked 1.4e-14 against the corrected Part B).

**Premise check — FAILS for the ratio framing.** Part 8's claim (small
ABSOLUTE gated sums: g12 max 4.09 vs 85.19 dataset-wide) is re-verified. But
g12's RATIOS span the full dataset range (0.000–10.878, incl. the dataset
MAXIMUM — mobley_4883284, err 0.247, gated sum −2.688): because several g12
members have very small errors, their ratios are high, not low. "Gradient-12
is a low-ratio group" is false.

**Hypothesis test — DOES NOT GENERALIZE (CASE_STUDY).**

| test | result |
|---|---|
| Spearman ρ(ratio, Δshrink), n=129 | **+0.171, p=0.053** — weak, marginal, NS at α=0.05 |
| Q1 (low ratio) vs Q4 (high) mean Δ | +4.646 [+0.14,+10.90], p=0.080 — NS; sensitivity excluding uncorrectable (0-gated-atom) molecules p=0.255 |
| within-error-quartile ρ(ratio, Δ) | **+0.430 / +0.157 / −0.088 / −0.598** — sign flips across strata; no stable relationship |
| non-g12 low-ratio molecules (n=56, ratio ≤ g12 median 0.238) | mean Δ **−0.488 [−0.72,−0.26]** — they SIGNIFICANTLY BENEFIT (opposite of g12's resistance); vs rest (−3.04) diff NS (p=0.11) |
| what actually predicts benefit | pre-correction error magnitude: ρ(err_before, Δ) = **−0.490**; |gated-sum| quartile contrast p=0.023 — correction helps big-error / big-gated-sum molecules most; ratio is confounded (ρ(ratio, err_before) = −0.397, ρ(ratio, |gated_sum|) = +0.769) |

Even within g12, harm is spread across ratio levels (the max-ratio member is
harmed +2.33; the Q1 g12 members are neutral/benefited) — the ratio does not
explain g12's own harm; the Part 8 atom-level mechanism (movement without
information; alignment ≈ 0) remains the correct explanation.

Verdict: **PASS as a case study, NOT as a general rule.** Part 8 stands for
gradient-12 specifically; the "low-ratio molecules won't benefit" screening
claim is NOT supported and must not appear in the paper as a general
check. The paper should describe gradient-12 as a defined failure mode with
its atom-level mechanism, and (if used at all) report the ratio only as a
descriptive statistic for that group — with the absolute-gated-sum
comparison as the honest quantity.

## Arithmetic audit (2026-08-16, post-B fix; distinct from Parts A/B/7/8/9)

Triggered by the factor-3 seed-sum bug found and fixed in `b8_holdout.py` (Part B), this sweep
re-audited every aggregation point in the node-refinement chain for the same bug class
(per-seed/per-node/per-molecule values summed across seeds BEFORE conversion to mean-space),
then independently recomputed the headline numbers from raw CSVs with fresh, minimal code
(no reuse of any potentially-affected script).

**Part 1 — inventory of aggregation points (9 scripts):**
`v1_verify.py`, `part3_strengthened.py`, `shrinkage_calibrated.py`, `b8_holdout.py`,
`a2_probe.py`, `gradient12_mechanism.py`, `generalization_check.py`, `dbg_random.py`,
and the original 2x2 matrix `experimental_uncertainty_refine/approach2_node_refine.py`.
Every `.sum()`/`.mean()`/groupby aggregation was manually verified (see
`arithmetic_audit/inventory.md` for line-level detail). **Verdict: the b8 bug was ISOLATED.**
All other scripts use the correct pattern throughout: per-seed arithmetic → per-seed
molecule sums → `.mean()` over seeds (or equivalent mean-space arithmetic). Also re-verified:
trust = 1 − rank(pct)(u3), K=10, min_sim=0.2, gate=2904, λ-grid, all approach2 arms at
alpha=1.0 (calibrated), and `build_populations` recomputes ensemble stats over the 3
surviving seeds.

**Part 2 — independent recomputation (`arithmetic_audit/independent_recompute.py`, fresh
code, raw inputs only; compare in `arithmetic_audit/audit_compare.json`):**
- (a) Val-set λ sweep (Part 7 protocol): λ* = **1.0**, identical to saved (0 diff on the
  entire 16-λ val calibration curve); all 160 bootstrap cells' deltas match to 1.8e-15
  (machine precision). Small CI diffs (≤0.27) are documented bootstrap-ordering noise
  (Part 7 iterates unsorted population sets; hash order varies per process) — deltas, which
  are order-independent, match exactly.
- (b) Headline populations at λ*=1.0 vs trust vs naive (approach2 replication, Mode A/B):
  all 20 cells' deltas match results.csv to ≤2.8e-7 (summation-order noise at the same
  tolerance v1_verify already asserted, <1e-6); CI diffs ≤0.14 (same set-order noise).
- (c) No discrepancy found in any headline number; nothing required investigation.
- (d) H1/H2 holdout (b8 protocol): λ*_H1 = **1.0**; all 25 cells (5 arms × 5 H2 populations)
  match b8_holdout_report.json **exactly** (max |Δdelta| = 0, max |ΔCI| = 0 — b8 sorts
  population members, so it is bit-reproducible).

**Part 3 — conclusion.** The corrected Part B outputs and every headline number in Parts
0–9 are arithmetically sound. The factor-3 bug was confined to the first version of
`b8_holdout.py`; the fix is verified exact. No manuscript limitation or claim needs
adjustment on arithmetic grounds; the audit record is `arithmetic_audit/inventory.md` +
`independent_recompute.py` + `audit_compare.json`.

## Overall

**Revision (2026-08-15, follow-up):** Parts 0/1/2/4/5/6 stand as PASS. Part 3's strengthened
random-neighbor control (4 arms + pool-mean shrinkage diagnostic) **falsifies the causal claim**
of Part 3 as originally passed: random-neighbor averaging — with or without trust weights, from
the eligible pool or from all nodes — reproduces or exceeds the headline repair on every
population, and deterministic pool-mean shrinkage of gated nodes beats the method outright.
The repair is value shrinkage at high-u3 nodes (a regularization effect), not validated
knowledge transfer from similar neighbors. Ch2.5's numbers are real and reproducible, but the
paper's causal story must be revised (see Part 3); the honest residual contribution is the
gating itself (which nodes to regularize) plus the gradient12 distinction (the trust mechanism
is uniquely harmful there, all shrinkage controls are neutral).

**Second revision (2026-08-15, follow-up):** the λ-calibrated shrinkage study (Part 7) makes the
shrinkage interpretation actionable: with λ* = 1.0 calibrated on val (a genuine interior
optimum), it is the new candidate best method — better than trust on every population, all129
significant where trust is NS, gradient12 neutral. The gradient12 atom-level study (Part 8)
closes the mechanism question: g12's errors are not gated-atom-dominated, so any gated-atom
rewrite is movement without information; trust harms more than shrinkage only because it moves
values further. Paper recommendation: headline = calibrated shrinkage (or full-replacement
baseline) as best method; trust-weighted similarity transfer demoted to a hypothesis, reported
with its g12 harm and the random-neighbor controls; g12 presented as a defined failure mode.

Caveats that must stay in the paper: (1) all129 NS for the trust method (significant only for
calibrated shrinkage); (2) gradient12 harmed by the trust method, NS for shrinkage; (3)
strengthened controls show equal-weight random averaging and pool-mean shrinkage beat the trust
method — the transfer mechanism is not validated; (4) transductive scope is batch scoring,
inductive rerun confirms but does not change the headline numbers.

**Third revision (2026-08-16, pre-write-up audit, Parts A and B):** both foundational checks
PASS. (A) The 3-seed regime is the reproducible one: seeds 7/2024 converged normally but their
checkpoints are catastrophically wrong (test MAE 38.4/58.1–58.6) on every surviving geometry
regime, molecule-specifically (45/129, ρ=0.96) and geometry-independently; re-scoring cannot
restore them; verbatim-ready methods justification in `seed_resolution/3seed_justification.md`.
(B) λ*=1.0 is not an artifact of repeated evaluation on the five fold-0 populations: with the
full calibration re-run on a random half (H1) of the test set and evaluation on the untouched
half (H2), λ*_H1 = 1.0 exactly, shrink improves 4/5 H2 populations significantly (allH2 −2.70
[−6.02,−0.50]), and paired shrink−trust is significant on ALL five H2 populations. The
candidate-best-method claim is retained with out-of-sample support; the trust method is
significantly worse than the strictest placebo on all five H2 populations. gradient12's H2
arm is underpowered (n=5) — its verdict rests on the fold-0 12-molecule evidence. Part B
outputs were corrected (2026-08-16) after a factor-3 seed-sum bug was caught in the first
version of the holdout script (mean-space arithmetic restored; conclusions unchanged in
substance — see Part B section).

**Fourth revision (2026-08-16, Part 9):** the gradient-12 mechanism check does NOT generalize
as a predictor. The gated-sum-ratio is not significantly associated with correction outcome
(ρ=+0.171, p=0.053; quartile contrast p=0.080; within-error-quartile ρ flips sign +0.43→−0.60;
non-g12 low-ratio molecules significantly benefit, −0.488 [−0.72,−0.26]). The ratio framing
of the Part 8 finding is a distortion — g12's ratios span the full dataset range (max 10.88 =
dataset max) while its ABSOLUTE gated sums are small (max 4.09 vs 85.19). What predicts
benefit is error magnitude (ρ=−0.490) and absolute gated-sum size, not the ratio. Part 8
remains a valid gradient-12-specific (case-study) explanation; the paper must not present
the ratio as a general screening rule.

**Fifth revision (2026-08-16, arithmetic audit):** independent from-scratch recomputation
confirms every headline number (λ*=1.0 on val and on H1; all Part 7 bootstrap deltas to
machine precision; all approach2 trust/naive deltas to ≤2.8e-7; all 25 H2 holdout cells
bit-exact). The factor-3 b8 bug was isolated (inventory of 9 scripts; see "Arithmetic audit"
section above). No manuscript claim needs arithmetic correction.

Outputs: v1_transductive.json, v1_inductive.json, v1_bootstrap_transductive.csv,
v1_grad12_transductive.csv, v1_union_per_molecule.csv; part3_strengthened/part3_strengthened.py,
part3_strengthened.json, part3_strengthened_all_rows.csv, part3_strengthened_new_arms.csv;
shrinkage_calibrated/shrinkage_calibrated.py, shrinkage_calibrated.json,
shrinkage_calibrated_grid.csv, shrinkage_calibrated_at_lambda_star.csv,
shrinkage_calibrated_vs_reference.csv; gradient12_mechanism/gradient12_mechanism.py,
gradient12_mechanism.json, gradient12_molecules.csv, gradient12_atoms.csv,
q_std_atoms_comparison.csv; seed_resolution/a2_probe.py, a2_probe_report.json,
a2_probe_predictions.csv, 3seed_justification.md (Part A); holdout_validation/b8_holdout.py,
b8_holdout_report.json, b8_holdout_bootstrap.csv, b8_holdout_per_molecule.csv,
b8_holdout_contrasts.csv, b8_split.json (Part B); gradient12_mechanism/generalization_check/
generalization_check.py, generalization_per_molecule.csv, generalization_low_ratio_molecules.csv,
generalization_report.json (Part 9); arithmetic_audit/inventory.md,
arithmetic_audit/independent_recompute.py, arithmetic_audit/audit_compare.json
(arithmetic audit, 2026-08-16).
Debug artifact dbg_random.py (verbatim approach2-structure replication used to isolate a
random-arm indexing bug during the audit) is kept for provenance.
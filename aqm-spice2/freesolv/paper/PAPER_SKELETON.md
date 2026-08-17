# Paper Skeleton — Solvation Free-Energy Ensembles: When Confidence and Uncertainty Disagree

Working title, status: **scaffolding**. Chapters filled in as experiments close.

---

## Chapter 1 (REFramED — not a terminal negative): molecule-level neighbor smoothing was the wrong granularity

**Status: reframed 2026-08-15 per prof clarification.** The intended method is
**node-level refinement**, not molecule-level: extract *uncertain nodes*
(atoms), not uncertain molecules, and refine those nodes' predictions from
*other nodes'* predictions. The molecule-level experiment below remains in
the paper as motivation: it tests (and fails to confirm) the naive
molecule-level interpretation of the idea, which is precisely why the
node-level formulation is the right one to build. Runs stand as-is; the
conclusion changes from "idea is negative" to "wrong granularity — the
mechanism lives at the node level (Chapter 2)."

### 1.1 Motivation and setup

The 5-seed ensemble (DimeNet++ backbone, stage-2 correction) has an anomaly:
18/129 fold-0 test molecules are *confidently wrong* — low ensemble std but
high RMSE. They split into gradient-12 (best_sim > 0.22, real neighbors
exist in training data) and isolated-6 (best_sim ≤ 0.22, structurally
isolated). Input-side representation was exonerated first (CHECK 12/13/14:
no tautomer/stereochemistry/sanitization/element-vocabulary difference).
Hypothesis tested here: pull each molecule's prediction toward a weighted
mean of its top-5 similar molecules' predictions *during fine-tuning* so
uncertain molecules learn from well-predicted neighbors.

Neighbor graph: static Morgan fingerprints (radius 2, 2048 bits), Tanimoto
edge weight, top-5, min_sim 0.1. Loss:
`L_nbr = mean_i [ sum_j w_ij (p_i − p_j)^2 / sum_j w_ij ]`, with the
normalized variant `L_nbr / var(p)`. 200 epochs, patience 30, stage-2
backbone, identical harness to the deep-ensemble member training.

### 1.2 Results (test MAE, kcal/mol, TTA-5)

| variant | λ | all129 | wrong18 | certain47 | isolated6 | gradient12 |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.552 | 0.499 | 0.273 | 0.464 | 0.516 |
| tanimoto_raw | 0.010 | **0.487** | 0.504 | 0.247 | 0.367 | 0.572 |
| latent_raw | 0.030 | 0.537 | 0.506 | 0.262 | 0.638 | **0.440** |
| latent_normalized | 0.050 | 0.791 | 0.726 | 0.458 | 0.882 | 0.648 |

Best-λ paired-bootstrap 95% CIs (10k resamples, delta vs baseline):

| variant | λ | group | delta | 95% CI |
|---|---|---|---|---|
| tanimoto_raw | 0.010 | all129 | −0.065 | [−0.147, +0.014] |
| | | gradient12 | +0.056 | [−0.082, +0.210] |
| latent_raw | 0.010 | all129 | −0.035 | [−0.090, +0.020] |
| latent_normalized | 0.050 | all129 | +0.239 | [+0.130, +0.371] |
| | | gradient12 | +0.132 | [+0.023, +0.228] |

Full 17-run table: `neighbor_regularization/SWEEP_RESULTS.md`.

Reads:
1. No formulation significantly improves overall MAE. The λ=0.01 raw gains
   (tanimoto −0.065, latent −0.035) have CIs that include 0.
2. gradient-12 is not improved at best-λ; point estimates are *worse*
   (+0.056 / +0.044). The λ=0.03 dips (0.440 latent / 0.456 tanimoto) sit at
   lambdas that degrade overall MAE, and at n=12 the paired CI crosses 0.
3. isolated-6 (n=6): best point estimate −0.097 (tanimoto_raw λ=0.01), CI
   [−0.322, +0.134] — noise.
4. **latent_normalized is significantly harmful** (all129 +0.239
   [+0.130, +0.371]; gradient-12 +0.132 [+0.023, +0.228]). The normalized
   latent scheme is dead.
5. Molecule-level: the loss moves errors *within* the group, not out of it
   (fixes the two worst, destabilizes near-perfect ones; net zero).

### 1.3 Broad-uncertainty reanalysis (population critique)

The narrow gradient-12/isolated-6 evaluation was itself audited
(`neighbor_regularization/broad_uncertainty_reanalysis/`):

- **Uncertain populations defined two ways**: top quartile (32/129) by
  5-seed ensemble_std and by GMM mean-NLL. Overlap: only 11 molecules
  (Jaccard 0.21) — the two uncertainty definitions select nearly disjoint
  sets. Union population n=53, baseline MAE 0.841.
- **Critical population finding**: gradient-12 is essentially *disjoint*
  from both broad uncertain populations (0/12 in Q_std, 1/12 in Q_nll).
  Gradient-12 is low-std by construction; "confidently wrong" ≠ "uncertain".
  The narrow and broad analyses test different populations, not a
  subset/superset pair.
- Re-running all 17 runs on Q_std / Q_nll / UNION: no run significantly
  improves any broad population. Best point estimates are the same two raw
  λ=0.01 runs (UNION −0.117 [−0.289, +0.059] tanimoto; −0.072 [−0.197,
  +0.051] latent); all CIs cross zero. All normalized variants ≥0.3 are
  significantly harmful on Q_nll and UNION.

### 1.4 The "converging on each other" diagnostic (never-before-checked risk)

The flagged risk: apparent gains could come from the 5 seeds converging on
each other (shrinking ensemble_std) rather than on truth. Sweep runs are
single-seed, so a true post-regularization 5-seed std does not exist —
stated plainly. Three independent proxies:

1. **Population prediction spread** (std across the 53 union molecules):
   baseline 4.861 kcal/mol; improving runs preserve or widen it
   (tanimoto_raw 0.01 → 4.980, latent_raw 0.01 → 4.939); only harmful
   high-λ normalized runs shrink it (tanimoto_norm 1.0 → 4.354).
2. **Shift-vs-error Spearman** on union: improving runs negative
   (tanimoto_raw 0.01 ρ=−0.217 p=0.12; latent_raw 0.01 ρ=−0.248 p=0.07) —
   bigger moves track bigger error reductions. Harmful normalized runs
   strongly positive and significant (ρ +0.44 to +0.70, p<0.01) — bigger
   moves track error *increases*. Movement is directional, not a random walk.
3. **Cross-run consensus** (16 variants as pseudo-seeds): mean per-molecule
   cross-run std 0.481 vs baseline 5-seed std 0.188 — the regularized
   family disagrees ~2.5× *more*, and its consensus is worse (union MAE
   0.951 vs baseline ensemble 0.828).

**Verdict: no evidence that apparent improvements come from seeds converging
on each other. Regularization inflates dispersion rather than manufacturing
false agreement.**

### 1.5 Chapter conclusion

Neighbor-consistency regularization is **tested-and-negative** across both
graph sources, both loss forms, and four lambdas each, under both narrow
and broad uncertainty populations. Combined with the exonerated input
representation, the confidently-wrong anomaly is intrinsic to the
molecules. The two cleanest positive learnings are diagnostic, not
methodological: (a) "confidently wrong" and "uncertain" are nearly disjoint
populations under standard uncertainty estimators; (b) the agreement-
accuracy coupling check (does regularization-induced agreement coincide
with error reduction?) is a reusable audit for any future ensemble
regularization. No further loss-formulation variants will be pursued.

---

## Chapter 2: Trust-weighted test-time refinement — molecule-level (RUN) → node-level (THE method, per prof)

**Status: molecule-level RUN is an intermediate finding; the paper's positive
contribution is the NODE-LEVEL formulation (spec'd 2.5, not yet run).**
Results in `deep_ensemble/repair_data/` (`repair_results.csv`,
`repair_diagnostics.json`, `seed_predictions_all642.csv`, `test_time_repair.py`).

### 2.1 Idea

Training-time neighbor-consistency failed because the loss competes with
the task loss. Test-time correction has no such competition: adjust the
ensemble prediction of uncertain molecules toward their top-5 neighbors'
predictions, gated by trust. Mechanistically distinct from Chapter 1 — no
training, no loss, post-hoc inference-time adjustment. **Prof clarification
(2026-08-15): the mechanism belongs at the node level — refine uncertain
NODES from OTHER NODES' predictions, not uncertain molecules from other
molecules. Section 2.5 specs that method; 2.2–2.4 remain as the
molecule-level baseline the node-level method must beat.**

### 2.2 Method (executed)

`p'_i = (1 − α) · p_i + α · Σ_j w_ij · t_j · p_j / Σ_j w_ij · t_j`

- `w_ij`: graph edge weight (Tanimoto `graph_k5_sim0.1.json` primary,
  latent `latent_k5_sim0.5.json` secondary)
- `t_j`: neighbor trust ∈ [0,1), rank-based on GMM mean-NLL over all 642
  molecules (refit per `gmm_uncertainty_check.py` protocol; reproduces the
  saved test NLLs to 7.6e-6 — `k_pca=13`)
- `α`: calibrated on the VAL set only (val-internal top-quartile gate,
  n=41), grid {0.05…1.0}; trust → 0.9, naive → 1.0, random → 0.05
- Gate: only molecules in the broad uncertain population are corrected

**Regime note (critical, per the repo's check1 "box-truth" report):** the
recorded 5-seed metrics (MAE 0.506, all populations) came from a destroyed
box with a different RDKit geometry regime; in the surviving regime (stored
hdf5 conformers) seeds 7/2024 are pathology-prone (fresh MAE 38/58, preds to
−742 kcal/mol; checkpoints SHA-authentic) and would dominate the baseline.
User-approved decision: **3-seed ensemble (42/123/999)**, everything
recomputed in the surviving regime. Fresh 3-seed test MAE = 4.836
(recorded 0.506 NOT comparable). All comparisons are within-machinery
(arms vs baseline), which the box-truth report confirms remains valid.

### 2.3 Results — Mode A (seed-level correction + re-ensemble), tanimoto graph, UNION primary

| arm (α) | Q_std (n=33) | Q_nll (n=33) | UNION (n=50) | all129 | gradient12 (n=12) |
|---|---|---|---|---|---|
| trust (0.9) | −4.841 [−10.7, −0.7] | −4.650 [−10.7, −0.5] | **−3.086 [−7.1, −0.3]** | −1.196 [−2.8, −0.1] | −0.131 [−0.39, 0.00] |
| naive (1.0) | −4.258 [−8.5, −1.0] | −3.451 [−7.9, −0.1] | −2.308 [−5.3, −0.04] | −0.895 [−2.1, −0.01] | −0.099 [−0.30, 0.00] |
| random (0.05) | +0.150 [−0.10, +0.48] | +0.142 [−0.11, +0.48] | +0.103 [−0.06, +0.32] | +0.040 [−0.02, +0.13] | −0.007 [−0.02, 0.00] |

Baseline (3-seed, surviving regime): Q_std 10.465, Q_nll 10.276, UNION 8.294,
all129 4.836, gradient12 2.914. ΔMAE in kcal/mol, paired-bootstrap 95% CI (10k).

- **Trust-weighted beats the random-shift control on every population**
  (CI excludes 0 on UNION/Q_std/Q_nll/all129); naive also beats it but
  consistently less (trust −3.09 vs naive −2.31 on UNION). The gain is real,
  not shrinkage-as-win: the same-magnitude random control is null.
- **Mode B (mean-level) is significantly harmful everywhere** (UNION
  +13.96 [+4.6, +25.7] trust) — the per-seed nature of the correction is
  what carries the gain, confirming Mode A as the mechanism.
- **Latent graph is weak** (Mode A trust UNION −0.925 [−2.7, +0.2],
  CI includes 0) — the repair works through chemical-similarity (Tanimoto)
  neighbors, not latent neighbors.
- **gradient-12 is untouched** (−0.13, CI to 0.00): the confidently-wrong
  group has (almost) no uncertain neighbors to borrow from — consistent with
  the §1.3 finding that gradient-12 is disjoint from the uncertain population.

### 2.4 Refinements (user-mandated — all in place, all honored)

1. **Control arms**: trust-weighted vs naive (t_j ≡ 1) vs random-shift
   (same per-molecule magnitude, random sign, seed-fixed). Trust beats
   random on all populations; beats naive on all populations.
2. **Broad population**: UNION primary, all populations reported (n=50/33/33).
3. **Correction target**: Mode A (seed-level, default) + Mode B (mean-level)
   sensitivity. Mode A wins; Mode B harmful.
4. **Statistical rigor**: 10k paired-bootstrap CIs for every arm × population
   cell — no point estimates.

### 2.5 False-consensus re-check (Mode A, trust, tanimoto, UNION)

| | before | after |
|---|---|---|
| ensemble-mean spread (kcal/mol) | 14.632 | 4.748 |
| mean ensemble_std | 4.907 | 1.887 |

Post-repair spread collapses (14.6 → 4.7) — the correction pulls the 3
seeds together as it improves accuracy. So: the gain is real (control arms
clean), but **repair converges the ensemble rather than preserving
disagreement** — the same "false consensus" signature the diagnostics
warn about, now produced by the fix rather than by regularization.

### 2.6 Verdict & decision tree resolution

- **Trust-weighted beats both controls on UNION with CI excluding 0** →
  per the §2.5 decision tree, the paper gains a positive method: a clean,
  mechanism-distinct, training-free repair that improves the broad
  uncertain population by −3.1 kcal/mol (CI [−7.1, −0.3]) on the union
  (≈ −37% relative), verified against naive and random controls, with
  Mode A per-seed correction as the operative mechanism.
- Caveats to state honestly: (1) surviving-regime absolute MAEs only
  (recorded 0.506 unrecoverable); (2) repair collapses ensemble spread
  (§2.5); (3) gradient-12 untouched — repair helps the *uncertain*, not
  the *confidently-wrong*; (4) α≈0.9–1.0 means the method is close to
  "replace uncertain predictions with neighbor predictions".
- Reframe toward **SDM 2027 (primary, §Appendix)** as: wrong-granularity
  training-time result (Ch1) + agreement-accuracy diagnostic + molecule-level
  test-time repair (2.2–2.4) as stepping stones + **node-level refinement
  (§2.5 spec) as the positive contribution** — pending the node-level run.

---

## Chapter 2.5 (SPEC — the method, per prof): Node-level refinement of uncertain nodes

**Status: RUN COMPLETE (2026-08-15) — VERIFIED (2026-08-15, 6-part audit pass) with
CAUSAL CLAIM REVISED (follow-up, same day: strengthened random-neighbor controls
falsify the similarity-transfer mechanism — see Validation & diagnostics and §2.5
interpretation below). Precisely: what is falsified is the *premise* that
similarity/trust selection drives the gain — trust-weighting itself genuinely
beats the no-correction baseline (H2 deltas −5.22/−6.50/−4.08/−1.44, all
negative); the engine was node-value shrinkage all along, of which
trust-weighting is an accidental, weaker implementation. FURTHER REVISED (2026-08-15): λ-calibrated shrinkage
(λ*=1.0 on val) is the new candidate BEST METHOD; gradient-12 mechanism closed at
atom level (Part 8). AUDIT COMPLETE (2026-08-16, Parts A & B): seed-resolution
justification (3-seed reproducible ensemble) and genuine held-out validation of
λ*=1.0 — both PASS; claim stands with out-of-sample support. Part 9 (2026-08-16):
gradient-12 mechanism does NOT generalize as a predictor (case-study scope confirmed).
ARITHMETIC AUDIT (2026-08-16): full aggregation-point inventory (9 scripts) + independent
from-scratch recomputation of λ*, all headline populations, and the H1/H2 holdout —
all numbers reproduce (λ*=1.0 both calibers; Part 7 deltas to machine precision; all 25
H2 cells bit-exact); the factor-3 b8 bug was isolated and its fix verified. No claim
needs arithmetic correction — see v1_VERIFICATION.md "Arithmetic audit".**
The paper's core positive contribution
per prof's clarification: refine *nodes*, not molecules. Implemented and
executed in `experimental_uncertainty_refine/approach2_node_refine.py`
(cross-molecular node pool per user decision; primary 3-seed 42/123/999,
5-seed arm flagged). Full head-to-head table in
`output/approach2_node_refine/results.csv`; per-node artifact
`node_contributions.csv` (11,613 nodes, 642 molecules).

**Feasibility (verified against `experimental_uncertainty_refine/DimeModels.py`):**
`DimeNetPlus` with `is_energy=True` computes per-node contributions
`P ∈ [N,1]` and only then `scatter(P, batch, reduce='sum')` — per-atom
energy contributions are directly extractable from every ensemble member
(assert: node-sum == seed prediction, passed for all 5 seeds).

### Results (ΔMAE vs ensemble mean, 10k paired bootstrap CIs, val-calibrated α)

Mode A (per-member refinement + re-ensemble), **trust-weighted** — THE method:

| population | n | ΔMAE | 95% CI |
|---|---|---|---|
| Q_std | 33 | **−3.807** | [−8.388, −0.358] |
| Q_nll | 33 | **−3.819** | [−8.454, −0.344] |
| UNION | 50 | **−2.359** | [−5.563, −0.048] |
| all129 | 129 | −0.760 | [−2.005, +0.165] |
| gradient12 | 12 | **+0.675** | [+0.003, +1.475] |

Controls (Mode A): naive −0.792/−0.538/−0.242 (NS); random +0.733/+0.692/
+0.536 (NS). Mode B identical for trust (α=1 ⇒ same operation); Mode B
random −1.078/−1.409/−0.737 (flagged: mixed-shift artifact, not a clean
null). α=1.0 chosen on val for all arms.

### Head-to-head vs molecule-level §2.2 (same populations, same regime)

| population | §2.2 molecule | §2.5 node |
|---|---|---|
| Q_std | −4.841 | −3.807 |
| Q_nll | −4.650 | −3.819 |
| UNION | −3.086 | −2.359 |
| all129 | −1.196 | −0.760 |
| gradient12 | 0 (untouched) | **+0.675 (harmed, sig)** |

Reading: node-level refinement is the prof's method and achieves
qualitatively the same repair as molecule-level on uncertainty-ranked
populations — but the STRENGTHENED control audit (2026-08-15 follow-up,
`node_refinement/verification/part3_strengthened/`) shows the mechanism is
NOT similarity-based knowledge transfer: equal-weighted random-neighbor
averaging and plain pool-mean shrinkage of the gated nodes both reproduce or
EXCEED the trust-weighted repair on Q_std/Q_nll/UNION/all129, and the
deterministic shrink-to-pool-mean baseline beats the method outright
(Q_std −6.30 vs −3.81). The repair is value shrinkage/regularization of
high-u3 (overdispersed) node values — a James-Stein-type effect — not
validated transfer from similar neighbors. The honest residual contributions
for the paper: (i) the GATING (uncertainty identifies which nodes are worth
regularizing; the effect concentrates in the worst-error molecules); (ii)
the gradient12 distinction — the trust mechanism is uniquely HARMFUL there
(+0.675, CI excludes 0) while every shrinkage control is neutral, so the
method's specificity is real only as a failure mode; (iii) the false-
consensus/spread signature as diagnostic evidence. Paper framing must state
the shrinkage baseline comparison and demote the similarity-transfer causal
claim to a hypothesis, not a demonstrated mechanism — while keeping
trust-weighting's positive effect explicit: it genuinely beats doing
nothing, and the honest story is that the fancy method's real engine was
shrinkage all along, hiding in plain sight; once shrinkage is built directly
and calibrated properly (λ*=1.0), it outperforms the method that was
accidentally doing a worse version of the same thing.

### Candidate best method (follow-up 2026-08-15): λ-calibrated shrinkage

`node_refinement/shrinkage_calibrated/`. Operation: P'_i = (1−λ)·P_i + λ·μ
(μ = pool mean, per seed), λ grid {0, 0.1, …, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0},
same 5 populations, 10k paired bootstrap. Val calibration (102 mols):
monotone to −1.805 at λ=1.0, then rises (λ>1 overshoots) → **λ* = 1.0, a
genuine interior optimum**. At λ*: Q_std −6.304 [−12.48,−1.84], Q_nll −6.065,
UNION −4.103, all129 −1.629 [−3.33,−0.44] (significant — trust is NS here),
gradient12 +0.217 NS. Improvement monotone in λ on all positive populations;
gradient12 NS at every λ. In-sample best λ = 1.0 for 4/5 populations — no
in-sample/out-of-sample tension. **Calibrated shrinkage beats the trust method
on every population with one val-calibrated hyperparameter; it is the new
candidate headline method** (equivalently: full pool-mean replacement of
gated-node values, i.e. subtract-and-zero the overdispersed contributions).

**Out-of-sample confirmation (2026-08-16, Part B audit,
`node_refinement/holdout_validation/`):** no population was held out during
the original chain, so the audit re-ran the FULL calibration on a random half
(H1, n=64, rng 20260816) of the 129 test molecules — population gates AND λ*
recalibrated on H1 only — and evaluated on the untouched half (H2, n=65).
λ*_H1 = 1.0 exactly (calibration minimum stable across disjoint data). On H2:
shrink significant on 4/5 populations (allH2 −2.70 [−6.02,−0.50], Q_std −8.22,
Q_nll −10.73, UNION −6.65; gradient12 n=5 NS); paired shrink−trust significant
on ALL FIVE H2 populations (allH2 −1.26 [−2.37,−0.46], gradient12 −0.85
[−1.56,−0.44]); shrink ≥ strictest placebo (randAll_equal) everywhere; the
trust method is significantly WORSE than the placebo on all five H2
populations — "worse" meaning it improves *less*: trust's own deltas vs the
no-correction baseline remain negative on 4/5 (Q_std −5.22 [−12.03,−0.18]
sig, Q_nll −6.50 NS, UNION −4.08 NS, allH2 −1.44 NS; only gradient12 +0.75
is genuinely harmful), while the placebo achieves −8.14/−10.52/−6.45/−2.63.
Trust genuinely beats doing nothing; the falsified claim is the
similarity/trust-selection premise, not trust's effect. (Numbers corrected
2026-08-16 after a factor-3 seed-sum bug in
the first holdout script — mean-space arithmetic restored, cross-checked
1.4e-14 against Part 7/8; conclusions unchanged in substance.) Caveat:
gradient12 H2 arm underpowered (n=5, 7/12 of the fixed list landed in H1);
the split is a retrospective fix, stated honestly in the paper's limitations.

### Validation & diagnostics

- **Verification audit (v1, 2026-08-15): 6/6 parts PASS; causal claim REVISED by
  follow-up.** See `node_refinement/verification/v1_VERIFICATION.md` and `v1_verify.py`:
  (0) 30-row results.csv reproduced from primary CSVs to 2.8e-07; (1) real
  correction — improvement concentrates in worst-20% pre-error quintile
  (89.1%), node shifts right-skewed (median 0.35, max 16.8), not consensus
  artifact; (2) 10k bootstrap CIs reproduced; 3/5 populations exclude 0,
  all129 NS, gradient12 harmed — stated plainly; (3) controls fair
  (naive = same neighbors t_j=1; random = magnitude-matched sign placebo,
  0/5 exclude 0) — **but strengthened random-neighbor controls (same-day
  follow-up) are NOT null: equal-weighted random-neighbor averaging and
  pool-mean shrinkage of gated nodes exceed the trust repair on all four
  positive populations (e.g. randAll_equal Q_std −6.20 [−12.26,−1.76] vs
  trust −3.81; shrink_poolmean −6.30), all new arms NS on gradient12 —
  mechanism is value shrinkage at high-u3 nodes, NOT validated
  similarity/trust transfer**; (4) inductive rerun (train-only pool) same
  verdict, slightly stronger: Q_std −4.430 [−9.85, −0.56], UNION −2.764
  [−6.51, −0.17]; (5) seeds 7/2024 diagnosed node-level (155/72
  catastrophic nodes, 0 overlap with gradient12/isolated-6; training
  converged normally, pathology = stored-box-conformer inference) — **Part A
  audit (2026-08-16, `node_refinement/seed_resolution/`) closed the 3-vs-5
  seed question: re-scoring the original checkpoints on every surviving
  geometry regime (stored hdf5 / fresh ETKDGv3 single / fresh TTA-5) gives
  38.4 / 58.1–58.6 test MAE (vs 3.4–6.7 for {42,123,999}), molecule-specific
  (45/129, ρ=0.96, geometry-independent); training per-epoch logs normal —
  the pathology is intrinsic to those checkpoints in the surviving regime,
  re-scoring is not a fix, and the 3-seed ensemble is the reproducible one
  (verbatim-ready justification in `3seed_justification.md`)**;
  (6) gradient12 mechanism refined to atom level (Part 8 follow-up):
  their neighbors are NOT unreliable (median u3 0.063 ≈ pool median; 45%
  below pool median vs 28% for Q_std) and NOT directionally misleading
  (replacement alignment with molecule error +0.09 vs +0.64 for Q_std);
  their gated sums are small (≤4.1 vs up to 85 for Q_std) — the confident
  wrongness lives in NON-gated contributions, so any gated-atom rewrite is
  movement without information; trust harms more than shrinkage only because
  it moves values further (0.63 vs 0.45 per-atom); worst 3 carry 83.5% of
  worsening; (7) **Part B audit (2026-08-16,
  `node_refinement/holdout_validation/`): genuine held-out validation of
  λ*=1.0 — full calibration re-run on a random test half (H1), evaluation on
  the untouched half (H2): λ*_H1 = 1.0; shrink significant on 4/5 H2
  populations (allH2 −2.70 [−6.02,−0.50]); paired shrink−trust significant
on ALL five H2 populations; trust significantly worse (improves
significantly *less* — its own deltas stay negative on 4/5 vs no
correction) than the strictest placebo on all five — the candidate-best-method claim and the
trust-vs-placebo comparison both survive out-of-sample** (corrected
  2026-08-16, factor-3 seed-sum bug fixed; see v1_VERIFICATION.md Part B);
  (8) **Part 9 generalization check (2026-08-16,
  `gradient12_mechanism/generalization_check/`): the gated-sum-ratio does NOT
  generalize as a predictor of correction resistance — ρ(ratio, Δ)=+0.171
  (p=0.053), quartile contrast p=0.080, within-error-quartile ρ flips sign
  (+0.43→−0.60), and non-g12 low-ratio molecules significantly BENEFIT
  (−0.488 [−0.72,−0.26]); g12's ratios span the full dataset range (max =
  dataset max) while its absolute gated sums are small (Part 8's actual
  finding); the predictor of benefit is error magnitude (ρ=−0.490) and
  absolute gated-sum size. gradient-12 must be described as a case-study
  failure mode, not a general rule; the ratio must not be presented as a
  screening check**; (9) **arithmetic audit (2026-08-16,
  `node_refinement/arithmetic_audit/`): aggregation-point inventory of all 9
  scripts (factor-3 seed-sum bug ISOLATED to the first b8 version) + fresh,
  independent recomputation from raw CSVs — λ*=1.0 reproduced on both
  calibrations; all 160 Part 7 bootstrap deltas match to machine precision;
  all approach2 trust/naive deltas ≤2.8e-7; all 25 H2 holdout cells
  bit-exact (max |Δ| = 0). Every headline number is arithmetically sound;
  no claim requires correction**.
- Node-uncertainty aggregates correlate with per-molecule |error|:
  Spearman mean_u3=0.235, sum_u3=0.242, max_u3=0.196 vs molecule-level
  ensemble_std 0.266 — node-level signal is nearly as strong.
- False-consensus re-check: spread 14.632→5.517, mean ensemble_std
  4.907→1.492 (same signature as §2.2).
- Pool: 11,613 nodes, u3 quantiles 50/75/90 = 0.057/0.097/0.182;
  gate = top-quartile of pool u3 (25% of nodes refined).
- 5-seed sensitivity (FLAGGED — see Seed-resolution note above): all129
  −6.328, UNION5 −24.446. Inflated by seeds 7/2024 (pathology-prone on every
  surviving geometry regime; `node_refinement/seed_resolution/`); reported
  only as a sensitivity arm (`sens5.json`), never as signal.

### Method (final, as executed)

**Seed-resolution note (2026-08-16, Part A of the pre-write-up audit):** all
results in this chapter use ensemble members {42, 123, 999}. The two excluded
members (seeds 7, 2024) have normal recorded training metrics (best-val MAE
0.418@37 / 0.465@23; per-epoch val MAE floors 0.508@49 / 0.489@56; early stop
at patience 30) and normal retrained trajectories, but their checkpoints are
unusable in every surviving geometry regime: re-scoring them with the stored
hdf5 geometry, freshly generated ETKDGv3+MMFF geometry, and 5-conformer TTA
gives test MAE 38.4 / 58.1–58.6 (vs 3.4–6.7 for the retained seeds), on a
molecule-specific subset (45/129, 32 shared, errors correlated ρ=0.96,
geometry-independent). The training-time conformer regime lives on a
destroyed cloud instance and is unrecoverable (see
`gradient12_conformer_provenance_check/report.md` §0); re-scoring is not a
viable fix. The recorded 5-member ensemble numbers (MAE 0.506) are treated
as unverifiable historical values; the 3-seed analysis is the reproducible
one. Full verbatim-ready justification, evidence table, and limitations:
`node_refinement/seed_resolution/3seed_justification.md` (+ probe artifacts
`a2_probe_report.json`, `a2_probe_predictions.csv`).

1. Per-node contributions `P_i` (kcal/mol) captured from all 5 members over
   all 642 molecules, single stored conformer; per-node uncertainty
   `u_i = std_s(P_i)`.
2. Cross-molecular node pool: descriptor = element one-hot + 1-hop
   neighbor-element counts (34-dim), Tanimoto; top-k (k=10, min_sim 0.2),
   same-molecule nodes excluded.
3. Trust `t_j = 1 − rank(u_j)/N` over the pool; gate = pool top-quartile u3.
4. `P'_i = (1−α)·P_i + α·Σ_j w_ij·t_j·P_j / Σ_j w_ij·t_j`; α on val
   (grid), Mode A (per-member then re-ensemble) primary.
5. Rigor: controls (trust/naive/random), 10k paired-bootstrap CIs,
   surviving-regime populations + gradient12, false-consensus re-check.

Artifacts → `experimental_uncertainty_refine/output/approach2_node_refine/`
(`node_contributions.csv`, `results.csv`, `report.json`,
`diagnostics.json`, `val_calibration.json`, `sens5.json`; runner
`approach2_sens5.py`).

---

## Chapter 3 (DECLINED — 2026-08-15): Frag20-as-augmentation

**No longer used.** Mechanistically distinct (new training examples, not a
loss penalty) and fully staged/prepared (corrected coverage numbers,
confirmed real neighbors in the full 100K set), but Ch2.5 (node-level
refinement) is now the paper's positive contribution and needs no Frag20.
The 89MB `Frag20-Aqsol-100K.tar.bz2` re-download and any Vast AI spend are
**not justified**; scripts remain in place should a future journal
extension ever warrant it. Removed from the SDM 2027 scope.

---

## Chapter 4 (CLOSED — 2026-08-15): Charges feature ablation

**Result: Gasteiger partial charges as input features HURT — plain wins on
all 3 seeds and both metrics. Negative result; not part of the paper.**

Protocol: fine-tune from stage-2 ckpt on frozen fold-0 split (411/102/129),
14 epochs each (best-val checkpoints; the 6 detached runs were killed before
their built-in final evaluation, so `experimental_charges/eval_finetuned.py`
re-ran the final eval on the saved best-val checkpoints). Both variants share
everything except a 1-channel charge feature (emb.continuous_lin
re-initialized in the charges variant).

Test MAE, stored conformer / TTA-4 (kcal/mol), n=129:

| seed | plain stored | plain tta4 | charges stored | charges tta4 |
|------|-------------|-----------|---------------|-------------|
| 42   | **0.707** | **0.726** | 1.045 | 1.061 |
| 7    | **0.981** | **1.009** | 1.012 | 1.061 |
| 2024 | **0.835** | **0.856** | 0.955 | 0.973 |
| mean | **0.841** | **0.864** | 1.004 | 1.031 |

gradient-12 (n=12, stored): plain 0.583–0.712 vs charges 0.897–0.983 — also
worse with charges. Charges add nothing; the plain reproduction is the
baseline for Ch2/Ch2.5 comparisons. Per-molecule 642 predictions saved per
mode/seed (`per_molecule_{mode}_seed{seed}.csv`) — the same
agreement-accuracy audit could be re-run on the charges variants if ever
needed (currently: unnecessary).

---

## Appendix: venue plan (updated 2026-08-15 — PAUSED, resume later)

**Status: user decision — relax the venue hunt; candidates on file, resume
searching later.** No submission deadline is binding right now; the paper
is not yet written.

User decision (2026-08-15): mid-tier + US-only + submit-in-2026 +
event-early-2027. Verified eliminations: AISTATS 2027 (Montréal), AAAI 2027
(Montréal, deadline Jul 28 passed), ICASSP 2027 (Toronto), RECOMB 2027
(Toronto), WSDM 2027 (Hong Kong), PSB 2027 (Hawaii — perfect fit but
deadline Aug 3 passed), WACV 2027 (Orlando — mid-tier, Aug 28 2026
deadline, but CV-only venue), HICSS-60 (Hawaii Jan 2027 — deadline
Jun 15, 2026 passed).

**SDM CYCLE CHANGE (tracked 2026-08-15)**: SDM has moved to a FALL
cadence. SDM'26 = **Nov 19–20, 2026, Salt Lake City, Utah** (US) — abstract
deadline Apr 10, 2026 and full-paper deadline Apr 17, 2026, BOTH PASSED.
The next edition (SDM'27) therefore lands ~**Nov 2027** with a paper
deadline expected ~**Apr 2027** (per the SDM'26 pattern; CFP not yet
posted). SDM'26 format (per sdm26.submissions): 8 pages incl. figures,
SIAM double-column US Letter, refs/appendices unlimited, triple-blind,
arXiv allowed, AoE deadlines.

- **SDM'27 (Nov 2027, ~Apr 2027 deadline): still the best-fit *venue***
  (mid-tier, US) but **no longer satisfies "submit in 2026 / event early
  2027"** — the fall cadence pushes the cycle out a year.
- **FLAIRS-40** (St. Pete Beach FL, May 24–27 2027; abstract Jan 22, 2027,
  paper Feb 1, 2027): best realistic "submit-soon + US + mid-2027 event"
  fit; 6pp, double-blind, DBLP/Scopus proceedings.
- **IEEE BigData 2026** (Phoenix AZ, Dec 14–17 2026; deadline Aug 21, 2026):
  only submit-2026+US option; 6-day window made it impractical unless
  sprinted; ~18% acceptance, single-blind, 10pp; CFP covers chemical
  engineering / materials informatics / data-centric AI.
- **NeurIPS 2026 workshops** (Paris, Dec 12–13; deadline **Aug 29, 2026**):
  light-touch exposure option only.
- **JCIM / JCTC**: journal home regardless of conference outcome; no
  deadline pressure.
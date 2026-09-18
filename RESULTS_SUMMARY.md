# GIMS Project — All Results Summary

Date: 2026-09-18. Backbone A = DimeNet++ (main). Backbone B = MACE (boundary
condition). Backbone C = AIMNet2 (in progress). Method: Gauge-Invariant
Molecular Shrinkage (GIMS) vs variance-weighted (VW) vs uniform shrinkage,
with 10,000 paired bootstraps unless noted.
`paper.tex` holds the ICLR 2027 draft (DimeNet++ + Guthrie-novel integrated).

## 1. DimeNet++ in-domain (FreeSolv fold-0, 3-seed {42,123,999})

Reference: `mu_T` from 411 train molecules (7,389 atoms); tau2* calibrated on
102-mol val; five test populations (disagree33/atypical33/union50/all129/
gradient12). Five-seed headline 0.549 +/- 0.024 kcal/mol; fold-0 5-seed
0.5059. Shrinkage arms scored vs the 3-seed stored-conformer baseline
(all129 MAE 4.84 — surviving-regime; see REPRODUCIBILITY.md).

| Method (pooled ref) | disagree33 | atypical33 | union50 | all129 | gradient12 |
|---|---|---|---|---|---|
| Baseline MAE | 10.46 | 10.28 | 8.29 | 4.84 | 2.91 |
| Trust-weighted | -3.81 | -3.82 | -2.36 | -0.76 | +0.67 |
| Uniform (matched) | -5.87 | -5.90 | -4.09 | -1.76 | -0.47 |
| Variance-weighted (VW) | -6.72 | -6.73 | -4.61 | -1.97 | -0.51 |
| Gated (top-quartile) | -6.30 | -6.07 | -4.10 | -1.63 | +0.22 |

GIMS (train-only ref): -6.81/-6.80/-4.67/-2.00/-0.52 — best point estimate on
all five. GIMS-VW directional but NS everywhere; GIMS-uniform significant on
4/5 (gradient12 n=12 excepted). Permutation control p=0.0000 on 4/5.
Gauge stress (1.0 kcal/mol atom-RMS): VW shifts 3.26, GIMS 2.5e-14.
Bias-variance: ~2/3 of MSE gain is bias^2 on all arms. Six alternative
atom-level designs all reduce to VW or worse. GIMS-total ablation slightly
worse (-1.95 vs -2.00). Records: `docs/progress/*`, `gims_report.json`.

## 2. DimeNet++ transfer (frozen tau*, mu_T, no retraining)

| set | n | raw | GIMS | VW | uniform |
|---|---|---:|---:|---:|---:|
| FlexiSol water | 297 | 4.80 | 3.98 (-0.83 [-1.15,-0.51]) | 3.99 (-0.82) | 4.00 (-0.80) |
| Guthrie-novel | 445 | 3.99 | 3.13 (-0.87 [-1.12,-0.62]) | 3.12 (-0.87) | 3.15 (-0.85) |

FlexiSol: GIMS > VW > uniform (GIMS-VW -0.011). Guthrie-novel (Guthrie DB
final-kcal slice, CID-deduped vs FreeSolv, PubChem 3D single conformer,
overlap-vs-FreeSolv MAE 0.28): VW ~= GIMS > uniform (GIMS-VW +0.004, NS).
Transfer replicates 2/2 (all methods beat raw); no second GIMS>VW win, and
none is claimed. Records: `flexisol/flexisol_all_methods_raw.csv`,
`guthrie_novel/`.

## 3. MACE from-scratch 3-seed (fold-0, frozen split)

TEST MAE 1.663 x3 (seeds near-identical: per-mol std mean 0.0039).
GIMS/VW/uniform all EXACT no-ops (tau*=inf, lambda*=0.00 on monotonic val
curve). Transfer raws: FlexiSol-290 2.838, Guthrie-442 3.164; arms 0.00.
Records: `mace_freesolv/fold0_ensemble/` + `MACE_GIMS_RESULTS.md`.

## 4. MACE-OFF23 (released) 5-seed (fold-0, frozen split)

TEST MAE 1.360-1.371 (foundation helps raw). Per-mol std mean 0.027 —
7x scratch but still below shrinkage scale. Val again selects exact no-op
for every arm, in-domain and transfer (FlexiSol-290 raw 4.639,
Guthrie-442 raw 3.164; all deltas 0.00). Records:
`mace_freesolv/fold0_ensemble_off23/` + `OFF23_GIMS_RESULTS.md`.

## 5. SPICE leakage check (for OFF23 use)

SPICE PubChem screening pool shares connectivity with 173/642 FreeSolv
(train 119 / val 26 / test 28), 9 FlexiSol, 28 Guthrie entries (upper bounds;
local DES370K-monomer file: 0). No label leakage possible (SPICE has no
hydration labels); standard foundation-model caveat + one-sentence paper
disclosure. Records: `mace_freesolv/SPICE_LEAKAGE_CHECK.md`.

## 6. Reading across backbones

Shrinkage helps iff (a) seeds disagree and (b) systematic bias exists.
DimeNet++ (surviving regime) has both -> large gains, GIMS >= VW > uniform.
MACE (both inits) has neither -> exact no-ops. The no-ops are the theory's
predicted degenerate limit, not failures, but they are not second GIMS wins.

## 7. AIMNet2-2025 (backbone C, in progress)

Why: 14 elements (H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I — zero skips on all 3
sets incl. siloxanes), 4 published ensemble members, MIT, pip-installable,
`aimnet[train]` supported. Probe: member spread 0.23 kcal/mol on ethanol
(~10x OFF23-finetuned); per-atom energies via hook before `atomic_sum`.
CPU benchmark (20 FreeSolv mols, real fwd+bwd): 0.04 s/mol -> ~25 s/epoch ->
~2-3.5 h for 5 seeds locally, or minutes on Vast GPU. Plan: fine-tune
member-0 weights 5x (seeds 42,123,7,2024,999) on frozen fold-0, energy-only
MSE (Stage-3-like: Adam 1e-4, wd 1e-5, <=200 epochs, patience 30), best-val
checkpoints, per-atom dump 642, then GIMS/VW/uniform x FreeSolv/FlexiSol-297/
Guthrie-445. Scripts: `aimnet_freesolv/` (trainer + transfer + arms),
`run_aimnet_ensemble.sh`.

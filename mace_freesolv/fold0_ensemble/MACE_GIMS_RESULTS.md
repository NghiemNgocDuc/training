# MACE Second-Backbone Shrinkage — Results

Date: 2026-09-18. Scripts: `mace_freesolv/mace_gims_analysis.py` (FreeSolv),
`mace_freesolv/mace_transfer_infer.py` (inference),
`mace_freesolv/mace_transfer_arms.py` (transfer arms).
Model: frozen MACE fold-0 3-seed ensemble (`fold0_ensemble/seed_{42,123,999}`,
from-scratch Stage-A + Stage-B, CPU inference). All outputs in
`mace_freesolv/fold0_ensemble/{analysis,transfer}/`.

## Ensemble properties (why this backbone behaves differently)

- Frozen split 411/102/129; TEST MAE 1.6629/1.6624/1.6626 (RMSE ~2.158).
- Per-atom std across seeds: mean 0.00062, p99 0.00293 kcal/mol
  (DimeNet: same quantity is ~100x larger). Shared-init + 500-epoch
  convergence → near-identical weights. Per-molecule std mean 0.0039.
- Per-atom decomposition: `node_energy` key, gate
  max|sum(P)-E| = 1.4e-05 kcal/mol, 642/642 kept, 0 skipped.
- mu_T per-seed (MACE train pool): -0.2123/-0.2122/-0.2124.

## FreeSolv (in-domain, paper populations)

Val calibration selects **tau*=inf (no-op) for GIMS, VW, and GIMS-total**
(val delta +0.0000 at every finite grid point), matched uniform Lambda = 0.
Freely-calibrated single uniform lambda* = 0.00; val curve rises
monotonically (lambda 0.01 -> +0.005, 0.10 -> +0.057, 1.00 -> +1.36), so the
no-op is genuine, not a grid edge artifact.

| arm | disagree33 | atypical33 | union50 | all129 | gradient12 |
|---|---|---|---|---|---|
| GIMS / VW / uniform | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

(all deltas exactly 0.00; MACE all129 raw MAE 1.663 vs DimeNet
surviving-regime 4.84). Gauge stress is vacuous here (both shifts 0).

## Transfer (frozen MACE settings; uniform lambda*=0.00 from val)

| set | n | raw MAE | GIMS d | VW d | uniform d |
|---|---|---:|---:|---:|---:|
| FlexiSol water | 290 | 2.838 | 0.00 | 0.00 | 0.00 |
| Guthrie-novel | 442 | 2.825 | 0.00 | 0.00 | 0.00 |

Notes: FlexiSol uses shipped gas-phase c0 geometries (7 siloxanes skipped —
Si outside MACE 10-elem vocab); Guthrie uses stored PubChem 3D (3 Si mols
skipped). DimeNet raw on the same sets: 4.80 / 3.99 — MACE raw transfers
better and has no systematic bias left for pool-mean shrinkage to fix.

## Verdict

- GIMS/VW/uniform are all exact no-ops on this MACE ensemble, in-domain and
  on both transfer sets. This is the predicted degenerate limit (no ensemble
  disagreement + no systematic bias -> no shrinkage), NOT a failure of the
  invariance construction — the FreeSolv gauge audit code runs green with
  both shifts at 0.
- Paper framing: MACE is a second backbone for the RAW model and for the
  boundary condition of the method, not a second GIMS win. The honest claim
  is conditional: shrinkage helps iff (a) seeds disagree and (b) predictions
  are systematically biased; MACE-from-scratch has neither. A MACE ensemble
  with real diversity (independent inits) would be needed to test GIMS
  proper on this architecture — flagged as future work, not run here.

## Artifacts

- `analysis/mace_gims_report.json`, `analysis/mace_gims_results.csv`
- `transfer/mace_flexisol_permol.csv`, `transfer/mace_flexisol_peratom.pkl`
- `transfer/mace_guthrie_permol.csv`, `transfer/mace_guthrie_peratom.pkl`
- `transfer/mace_transfer_arms.csv`, `transfer/mace_transfer_arms.json`

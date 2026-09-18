# MACE-OFF23 (Released Weights) Shrinkage — Results

Date: 2026-09-18. Ensemble: official MACE-OFF23-medium foundation
(ACEsuit/mace-off, SPICE wB97M-D3(BJ), ASL license) fine-tuned 5x
(seeds 42,123,7,2024,999) on frozen FreeSolv fold-0 (411/102/129), same
hyperparams as the scratch run. Scripts: `mace_freesolv/fold0_ensemble_off23.py`
(Vast GPU), `mace_gims_analysis.py` / `mace_transfer_infer.py` /
`mace_transfer_arms.py --base_dir .../fold0_ensemble_off23` (local CPU).

## Ensemble properties

- TEST MAE 1.361/1.360/1.371/1.365/1.366 (RMSE ~1.80) — better than
  from-scratch (1.663), as expected from foundation init.
- Per-molecule std across 5 seeds: mean 0.027, max 0.19 kcal/mol (~7x the
  scratch ensemble, still ~50x below DimeNet scale).
- Per-atom std (train pool): mean 0.0040, p99 0.028 (at DimeNet tau* this is
  mean Lambda ~0.06 — small but nonzero signal).
- mu_T per-seed: -0.2038/-0.2046/-0.2033/-0.2035/-0.2050. Gate
  max|sum(P)-E| ~1e-05, 642/642 kept.

## FreeSolv (in-domain, paper populations)

Val calibration selects **tau*=inf for GIMS, VW, and GIMS-total**, matched
uniform Lambda = 0, freely-calibrated uniform lambda* = 0.00. All deltas
exactly 0.00 on all five populations (all129 raw MAE 1.366). Gauge stress
vacuous (both shifts 0). So even with official weights and 5 seeds, val
rejects every finite shrinkage strength.

## Transfer (frozen OFF23 settings)

| set | n | raw MAE | GIMS d | VW d | uniform d |
|---|---|---:|---:|---:|---:|
| FlexiSol water | 290 | 4.639 | 0.00 | 0.00 | 0.00 |
| Guthrie-novel | 442 | 3.164 | 0.00 | 0.00 | 0.00 |

Geometries: FlexiSol shipped gas c0 (7 siloxanes skipped, Si outside MACE
vocab); Guthrie stored PubChem 3D (3 Si mols skipped). For reference, DimeNet
raw: 4.80 / 3.99; scratch-MACE raw: 2.84 / 2.83.

## Verdict

Same degenerate conclusion as the scratch run, now with official weights and
5 seeds: no ensemble disagreement worth shrinking + no systematic bias val
wants fixed -> every arm is an exact no-op. This bounds the paper's claim
honestly (shrinkage helps iff seeds disagree and bias exists) but is NOT a
second GIMS win. Testing GIMS proper on MACE needs a truly diverse ensemble
(independent inits) — future work.

## Artifacts (this directory)

- `seed_{42,123,7,2024,999}/model.pt` + `metrics.json` + `test_predictions.csv`
- `mace_node_contributions.csv`, `mace_seed_predictions_all642.csv`,
  `peratom_mace_seed*.pkl` (5), `ensemble_summary.json`, `split_used.json`
- `analysis/mace_gims_report.json`, `analysis/mace_gims_results.csv`
- `transfer/mace_flexisol_per{mol.csv,atom.pkl}`,
  `transfer/mace_guthrie_per{mol.csv,atom.pkl}`,
  `transfer/mace_transfer_arms.csv`, `transfer/mace_transfer_arms.json`

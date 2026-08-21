# Exp-DB seed-ensemble GIMS bundle

Trains a proper 5-seed ensemble of the fold-0 model (seeds 42/123/7/2024/999,
identical recipe to expdb_vast), extracts **per-atom contributions** on the
620 Exp-DB molecules (TTA-5) and on the 411 FreeSolv training molecules, then
applies GIMS / VW / uniform vs raw on Exp-DB — the second-dataset test.

## On your GPU machine (Vast)

```bash
git clone <your-repo-url>
cd <repo>/expdb_seed_ensemble
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 1) verify inputs are present (they ship in inputs/)
python preflight.py

# 2) run everything detached, with logs
nohup bash run_all.sh > run_all.log 2>&1 &

# 3) watch progress
tail -f run_all.log
```

Stages are resumable: finished seeds/pickles are skipped on re-run.

## Outputs (`results_seeds/`)

| file | what |
|---|---|
| `finetuned_seed{S}.pt` | stage-3 fine-tuned fold-0 model, seed S |
| `train_meta_seed{S}.json` | best val MAE / epoch / runtime |
| `peratom_seed{S}.pkl` | per-atom P + molecular E for Exp-DB (TTA-5 avg) and FreeSolv train |
| `gims_expdb_report.json` | gates, gauge audit, pathology screen, Lambda stats |
| `gims_expdb_results.csv` | DeltaMAE table: raw/uniform/VW/GIMS x {all620, Q_spread, WDec10} |

## Method notes

- Ensemble = 5 independently fine-tuned stage-3 runs from the same frozen
  `stage2_correction.pt` on the ARCHIVED fold-0 split (loaded byte-identical
  from `inputs/split_check/`). Primary analysis = 3-seed {42,123,999}
  (paper convention); 5-seed reported as sensitivity.
- Per-atom extraction flips `model.is_energy` at inference only; sums are
  gated against direct energy prediction (<1e-6 kcal/mol).
- mu_T^(k) computed per seed from FreeSolv TRAIN atoms only (deployment-safe).
- tau^2* frozen at the archived 4.725394227550238e-04.
- Gauge-stress audit reuses the standard construction (zero-sum,
  seed-independent, atom-RMS 1.0 aligned with lambda - Lambda_m).

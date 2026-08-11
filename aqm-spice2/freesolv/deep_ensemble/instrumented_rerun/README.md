# Instrumented Stage-3 re-run (gradient-12 training-dynamics diagnostics)

**PURE DIAGNOSTIC ADDITION — NOT a new experiment.** This replicates the
original Stage-3 fine-tune exactly and only adds per-epoch, per-molecule
logging. It never touches the original `deep_ensemble/seed_{42,123,7,2024,999}`
directories: all outputs go under `instrumented_rerun/seed_<s>/`.

## What is reused (byte-for-byte identical to the original seed runs)

| knob | value | source |
|---|---|---|
| split | frozen fold-0 `train_ids/val_ids/test_ids.json` (md5-checked) | `deep_ensemble.py DEFAULT_SPLIT_DIR` |
| init | Stage-2 correction checkpoint `pipeline/results_full/stage2_correction.pt` | `deep_ensemble.py DEFAULT_CORRECTION_CKPT` |
| arch | DimeNetPlus (`build_model`) | `deep_ensemble.py` |
| optimizer | Adam lr=1e-4, wd=1e-5 | `deep_ensemble.py train_member` |
| batch | 8 | same |
| loss | MSE in eV, grad-clip 10.0 | same |
| scheduler | ReduceLROnPlateau(f=0.5, pat=15, min_lr=1e-6), stepped on val MAE | same |
| epochs / patience | 200 / 30, early stop on val MAE | same |
| best-val ckpt rule | save on strict val-MAE improvement | same |
| final eval | single-conf test pass + 5-conf RDKit TTA (`predictions.csv`) | same |
| device | cuda (original ran on cuda) | `--device cuda` |

## Instrumentation (RNG-neutral by construction)

Everything added runs under `model.eval()` + `torch.no_grad()`; the train
loader, optimizer, scheduler, and best-ckpt logic are untouched, so the
training trajectory is unaffected by the logging:

- epoch 0: warm-start (Stage-2 ckpt) test+val predictions, before any step
- every epoch: per-molecule test rows
  `epoch, mol_id, dG_pred_kcal, dG_exp_kcal, abs_err_kcal, mse_ev2`
  → `seed_<s>/epoch_predictions.csv`
- every epoch: `epoch, val_mae_kcal, val_rmse_kcal, train_mse_ev2`
  → `seed_<s>/val_history.csv`
- same end artifacts as the original run (`metrics.json`, `predictions.csv`,
  split copies, `split.md5`) — written only under `seed_<s>/`

Caveat: GPU nondeterminism (cuDNN/torch version on the Vast box vs the box
that produced the recorded seed_42 numbers) can shift best-val epoch by a few
epochs and MAE by ~1e-3. The sanity check tolerates that; anything larger
fails the check and STOPS Stage B.

## Running on Vast AI

1. Sync this repo to the box; keep the same relative layout. Required files:
   - `Data/FreeSolv/database.json` (labels)
   - `freesolv_conformers.hdf5` (repo root, stored conformers)
   - `aqm-spice2/aqm-spice2/pipeline/results_full/stage2_correction.pt`
   - `aqm-spice2/aqm-spice2/freesolv/cv_results_full/fold_0/*_ids.json`
   - the same python env the original runs used (torch_geometric, rdkit,
     scipy, pandas, numpy, matplotlib)
2. Stage A (single seed, then sanity + early read):

   ```
   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_seed42.sh \
       > aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_seed42.log 2>&1 &
   ```
3. Review `seed_42/sanity_report.json` (verdict PASS/FAIL) and
   `seed_42/stageA_early_read.json` + `stageA_curves.png`. **If FAIL: stop**
   and report before Stage B (per protocol).
4. Stage B (remaining seeds, only after Stage A review):

   ```
   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_all_seeds.sh \
       > aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/run_all_seeds.log 2>&1 &
   ```

## Output layout

```
instrumented_rerun/
  instrument_finetune.py   # the instrumented replica (training loop untouched)
  sanity_check.py          # instrumented vs original seed_42 comparison
  analyze_stageA.py        # early read: gradient-12 vs random-12 (seed 42 only)
  run_seed42.sh            # Stage A orchestration
  run_all_seeds.sh         # Stage B orchestration (seeds 123 7 2024 999)
  seed_<s>/
    epoch_predictions.csv  # per-epoch per-molecule test predictions (THE new data)
    val_history.csv        # per-epoch pooled val MAE/RMSE + train MSE
    metrics.json           # same schema as the original
    predictions.csv        # final TTA predictions (same protocol as original)
    sanity_report.json     # seed_42: instrumented-vs-original verdict
    stageA_early_read.json # Stage A metrics + MWU (seed 42)
    stageA_curves.png      # group curves + trajectories + best-err histograms
```

## Stage B trajectory analysis (after all 5 seeds)

Designed outputs (script `analyze_stageB.py` to be written after Stage A
review): per molecule across 5 seeds —
first-epoch-within-tolerance and stay-within stability, tail-20% prediction
oscillation, per-molecule best epoch vs pooled best-val epoch (does pooled
early stopping sacrifice gradient-12?), overlaid representative trajectories,
group-level MWU across seeds. Verdict: late convergence / instability /
early-stopping-vs-subgroup, or inconclusive.

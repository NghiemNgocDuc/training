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

## RNG neutrality: what the audit found and the guard fix

`audit_rng.py` (report: `rng_audit_report.json`) proves the instrumentation is
RNG-neutral AFTER the guard; without the guard it is NOT, and the mechanism is
subtle:

- **`iter()` on ANY DataLoader consumes exactly one draw from torch's DEFAULT
  generator** (`BaseDataLoaderIter.__init__` -> `_base_seed = torch.empty(
  (), dtype=torch.int64).random_()`), regardless of shuffle (verified for
  plain torch and torch_geometric, shuffle True and False).
- The original per-epoch loop draws: `iter(train_loader)` + `iter(val_loader)`.
- The instrumented loop (pre-fix) added `iter(test_loader)` every epoch, plus
  2 extra iters at the epoch-0 warm-start eval. Every draw shifts the state
  from which the NEXT epoch's shuffle permutation is generated -> different
  batch order from epoch 1 on -> the training trajectory diverges while the
  final aggregate MAE stays similar. This is exactly the observed FAIL.
- RDKit ETKDGv3 TTA (`conformer_average`) is NOT the culprit: it runs once at
  the end (as in the original), uses `randomSeed=42`, and is fully RNG-neutral
  (`check2`). The per-epoch eval is single-conformer; no RDKit involved.

The fix (`instrument_finetune.py`): `rng_snapshot()`/`rng_restore()` wrap
every block that does NOT exist in the original script (epoch-0 warm-start
eval, per-epoch test eval, final TTA). They save/restore python `random`,
numpy, torch CPU + CUDA state. Verified: with the guard, the per-epoch RNG
stream is byte-identical to the original script's through 3 epochs
(`seq_sim.py` check); without it, the stream diverges at epoch 1.

Remaining irreducible variance: float nondeterminism (CPU threading / GPU
cuDNN kernels) is NOT covered by RNG guards and can shift trajectories even
with identical batch orders. On the box, if the guarded short run still fails
the sanity check against the RECORDED reference, re-baseline: run the original
`deep_ensemble.py --mode train --seed 42` ON THE BOX and compare against
THAT (same torch/cuDNN/GPU).

## Running on Vast AI

Current protocol (after the RNG-leak fix, commit 7a5ec0c):

1. `git pull` and install the deps listed in the setup block (torch, torch_geometric,
   rdkit, scipy, pandas, numpy, matplotlib, h5py, tqdm).
2. **Phase A** — same-box 5-epoch comparison (orig x2 vs fixed x1), verdict gates
   Phase B (CPU self-noise showed even the ORIGINAL doesn't reproduce itself
   run-to-run; Phase A quantifies the box's real self-noise):

   ```
   nohup bash aqm-spice2/freesolv/deep_ensemble/instrumented_rerun/box_gpu_compare.sh \
       > box_gpu_compare.log 2>&1 &
   ```

   Review `cmp_verdict.json`. PASS auto-launches Phase B; FAIL stops per protocol.
3. **Phase B** — full instrumented seed_42 rerun + full ORIGINAL seed_42 on the
   same box (`box_orig_full/`) + dual-reference sanity check (recorded reference
   gets self-noise-loosened tolerances; same-box reference keeps tight ones):

   ```
   tail -f box_full_rerun.log     # launched by Phase A on PASS
   ```

   Review `seed_42/sanity_report_ref0.json` (vs recorded),
   `seed_42/sanity_report_ref1.json` (vs same-box original),
   `seed_42/sanity_summary.json` (verdict). Only after a PASS proceed to Stage B.

Legacy instructions (single reference, original recorded numbers):

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

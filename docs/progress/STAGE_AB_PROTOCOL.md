==============================================================================
  MACE-OFF23 + FreeSolv — Pipeline Status & Run Guide
  Updated: 2026-07-31
==============================================================================

== OBJECTIVE ==
Fine-tune MACE-OFF23 (pretrained organic-molecule foundation model) on FreeSolv
(642 small organic molecules) to predict solvation free energy ΔG_solv
in kcal/mol. Target: beat SOTA MAE 0.417 (Zhang 2022, A3D-PNAConv-FT).

All ΔG in kcal/mol. 1 kcal/mol = 4.184 kJ/mol.
FreeSolv experimental uncertainty: ~0.1-0.3 kcal/mol.

Metric reference:
  MAE (mean absolute error) = avg |prediction − experiment|
    <0.4 beats SOTA | 0.52 = DimeNet++ FT | 3.0 = MACE-OFF23 zero-shot
  RMSE = sqrt(avg(pred−exp)²) — penalizes outliers
    0.72 = Zhang SOTA | 0.84 = DimeNet++ FT
  R² = fraction of variance explained
    1.0 = perfect | 0.96 = excellent | 0.0 = mean baseline

== FILES (mace_freesolv/) ==
  config.py            — All hyperparameters (epochs, LR, patience, etc.)
  data.py              — MACEFreeSolvDataset, radius_graph, collate_mace
  model.py             — MACEFreeSolv wrapper, atomic ref fitting, calibration, LoRA, init_checkpoint
  train.py             — WarmupWrapper, train_epoch, validate, run_fold, run_cv
  main.py              — CLI entry, SOTA comparison table, eval_checkpoint
  aqm_data.py          — AQMMACEDataset (Stage-A loader, dG = E_sol - E_gas in eV)
  train_stage_a.py     — Stage-A fine-tune on AQM
  run_stage_ab.sh      — one-command two-stage launcher (detached, stage_ab.log)

== CV STRATEGY ==
  Round-robin fold assignment: sort 642 molecules by target ΔG,
  assign i % n_folds. Gives stratified folds by property (not shuffled).

  Each fold: 80% train/val, 20% test (round-robin).
  Train/val split: 80/20 shuffle within train_val molecules.
  Val drives early stopping + ReduceLROnPlateau.
  Test held out until final reload-and-evaluate.

  fold_metadata.json saved alongside checkpoints for eval_checkpoint.

== DEFAULT CONFIG ==
  FREEZE_ATOMIC_ENERGIES=True   — freezes per-element reference energies
  FREEZE_INTERACTIONS=False     — fine-tunes all interaction layers
  PATIENCE=50 epochs            — early stopping
  WARMUP_EPOCHS=10              — linear LR ramp
  N_FOLDS=5                     — cross-validation
  LR=1e-4                       — Adam
  BATCH_SIZE=32
  LOSS_TYPE="mse"               — also supports "huber"
  USE_LORA=False                — full fine-tune by default

== BUGS FIXED (2026-07-30) ==
  1. radius_graph: torch.lexsort does not exist in current PyTorch.
     Replaced with np.lexsort (data.py:31). radius_graph had never
     actually executed — all prior edge indices were empty.
  2. hdf5_path ignored: _build_item hardcoded module-level HDF5_PATH
     instead of self.hdf5_path. Added self.hdf5_path = hdf5_path in
     __init__ (data.py:69), changed _build_item to use it (data.py:94).
  3. Atomic ref / calibration leaked test data: fit_refs in model.py
     built a fresh MACEFreeSolvDataset() with all 642 molecules every
     fold. Added fit_dataset param; train.py:138 passes fit_dataset=train_ds.
  4. Scale cap at 0.001: model.py line 90 now caps at 1.0 with
     warning when it fires. Old cap always clipped reasonable values.
  5. Fold leakage (KFold shuffle): train.py replaced KFold(shuffle=True)
     with round-robin on sorted targets. KFold was re-shuffling the
     sorted list, cancelling the stratification.
  6. No validation set: train.py added 3-way train/val/test split.
     Was evaluating on test set every epoch and picking best test epoch.
  7. Warmup/scheduler ordering: warmup.step() moved before train_epoch().
     First epoch was training at full LR.
  8. Warmup/scheduler overlap: scheduler.step() only called when
     epoch > warmup_epochs to avoid conflicting LR adjustments.
  9. eval_checkpoint fold reconstruction: added round-robin split
     matching train.py, plus fold_metadata.json roundtrip.
  10. loaded.dim() != expected_shape: model.py:160 fixed to
      loaded.shape != expected_shape (dim() returns int, always True).
  11. LoRA print timing: moved after hybrid unfreezing.
  12. Dead config wires: MACE_NUM_INTERACTIONS deleted from config.py;
      VAL_SPLIT imported in train.py; FREEZE_INTERACTIONS wired into
      main.py argparse default.
  13. Dead seed variable: removed from eval_checkpoint (set but unused).

== BENCHMARK ==
  Zhang 2022 (A3D-PNAConv-FT)   0.417    0.719
  COSMO-RS                      0.52      —
  ReSolv                        0.63      0.96
  Fine-tuned DimeNet++ (ours)   0.52      0.84
  MACE-OFF23 fine-tuned         ?         ?

== FILES SAVED ==
  mace_freesolv/results/
    fold_N/model.pt                 — best checkpoint per fold
    fold_N/fold_metadata.json       — seed, n_folds, fold_index
    fold_N/test_preds.npz           — preds vs expts for test set

== TWO-STAGE TRANSFER PROTOCOL (2026-07-31) ==
Goal: beat 0.417 (Zhang 2022). Direct MACE-OFF23+FreeSolv FT plateaued ~1.4 kcal/mol.
Plan: fine-tune MACE on AQM first (Stage A), then on FreeSolv (Stage B, existing pipeline).

CRITICAL DISCOVERY - eSOLV IS NOT A SOLVATION FREE ENERGY:
  AQM's eSOLV field sits on the TOTAL-ENERGY scale (eSOLV ~= E_sol + 0.08 eV).
  The old aqm-spice2 Stage-2a correction model was trained on (eSOLV - E_gas)
  residuals ~= 0.08 eV constant -> zero-shot FreeSolv MAE was 27.1 kcal/mol.
  CORRECT TARGET: dG = E_sol - E_gas from 'ePBE0+MBD' fields of paired
  sol/gas conformers, x 23.0605 = kcal/mol.

AQM-full verified (inspect_full.py):
  sol & gas both 1653 molecules / 59783 conformers, all pairs matched.
  dG stats: mean=-14.29, std=5.47, min=-52.8, max=+1.6 kcal/mol.
  Elements: H,C,N,O,F,P,S,Cl only -> all 1653 molecules in MACE-10 vocab (no Br/I).
  Molecules: 2-92 atoms (mean 40.6, median 42) vs FreeSolv ~20.
  Gas/sol geometries differ slightly (RMSD 0.27 A) - paired-conformer dG is fine.

== HOW TO RUN THE FULL PIPELINE (step by step) ==

== RUN FROM SCRATCH ON A VAST JUPYTER INSTANCE (no local machine needed) ==
Code is already pushed to GitHub — nothing to do locally. On the instance,
open a terminal (or run each line in a Jupyter cell with a leading !):

  # 1 - GPU check
  nvidia-smi

  # 2 - clone repo + install deps
  cd /workspace
  git clone https://github.com/NghiemNgocDuc/training.git 2>/dev/null || git -C training pull
  cd training
  pip install mace-torch tqdm h5py

  # 3 - download the full AQM data (~3.3 GB total, from Zenodo)
  wget -O AQM-gas-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-gas.hdf5?download=1'
  wget -O AQM-sol-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-sol.hdf5?download=1'

  # 4 - run the whole pipeline (detached; Stage A then Stage B, logs to stage_ab.log)
  bash mace_freesolv/run_stage_ab.sh

  # 5 - monitor
  tail -f stage_ab.log                   # live progress
  grep 'test:' stage_ab.log | tail -20   # fold-by-fold test MAE/RMSE
  grep 'PIPELINE DONE' stage_ab.log      # finished marker

Results land in /workspace/training/mace_freesolv/results/ (fold_N/model.pt,
fold_metadata.json, test_preds.npz) and results_stage_a/ (stage_a.pt).

== RUN VIA SSH FROM LOCAL MACHINE (alternative) ==

1) LOCAL MACHINE - commit & push code (only needed if you changed something):
     git add -A
     git commit -m "two-stage protocol"
     git push origin master

2) VAST INSTANCE - start it, then SSH in:
     ssh battery
     ls /workspace                          # confirm repo folder (usually training/)
     cd training
     git pull                               # get the new code
   (first time on a fresh instance, also install deps:
     pip install mace-torch tqdm h5py       # or python3 -m venv venv && source venv/bin/activate first)

3) DATA - only if AQM-sol-full.hdf5 / AQM-gas-full.hdf5 are NOT on the instance:
     wget -O AQM-gas-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-gas.hdf5?download=1'
     wget -O AQM-sol-full.hdf5 'https://zenodo.org/records/10208010/files/AQM-sol.hdf5?download=1'
   (or from local, slower: scp AQM-sol-full.hdf5 AQM-gas-full.hdf5 battery:/workspace/training/)

4) RUN THE WHOLE PIPELINE (one command, detached - safe to close SSH):
     bash mace_freesolv/run_stage_ab.sh
   -> runs Stage A (MACE fine-tune on AQM dG) then Stage B (FreeSolv 5-fold CV
      from stage_a.pt), logs everything to stage_ab.log

5) MONITOR:
     tail -f stage_ab.log                   # live progress
     grep 'test:' stage_ab.log | tail -20   # fold-by-fold test MAE/RMSE
     grep 'PIPELINE DONE' stage_ab.log      # finished marker

6) FETCH RESULTS BACK TO LOCAL:
     scp -r battery:/workspace/training/mace_freesolv/results mace_freesolv/
     scp -r battery:/workspace/training/mace_freesolv/results_stage_a mace_freesolv/
   (run from C:\Users\User\Documents\Data)

Manual run - Stage A (fine-tune MACE-OFF23 on AQM dG):
  python mace_freesolv/train_stage_a.py --hdf5_sol AQM-sol-full.hdf5 --hdf5_gas AQM-gas-full.hdf5 --device cuda
  (defaults: medium, 100 epochs, lr 1e-4, batch 32, patience 20, warmup 10,
   molecule-level 80/20 split - NEVER split conformers of same molecule!)
  Output: mace_freesolv/results_stage_a/stage_a.pt + stage_a_meta.json
  quick test: add --quick_test (2 epochs, 300 samples)

Manual run - Stage B (existing FreeSolv CV pipeline, from Stage-A weights):
  python mace_freesolv/main.py --init_checkpoint mace_freesolv/results_stage_a/stage_a.pt --device cuda
  eval: python mace_freesolv/main.py --eval_only results/fold_N/model.pt --eval_fold N --device cuda

New files: mace_freesolv/aqm_data.py (AQMMACEDataset, lazy LRU-cached, y in eV),
           mace_freesolv/train_stage_a.py; model.py: init_checkpoint param
           (skips atomic-ref fit; uses checkpoint refs/scale); train.py/main.py: --init_checkpoint.

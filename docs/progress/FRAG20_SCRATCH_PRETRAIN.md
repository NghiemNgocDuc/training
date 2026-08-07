# Frag20 from-scratch pretraining -> FreeSolv fine-tune: design & Vast timing

Status: sandbox built + CPU smoke-tested. Full runs NOT launched (user launches
on Vast). Design follows the handoff brief: train DimeNet+ FROM SCRATCH (stage 1
vacuum + stage 2 correction, mirroring the verified AQM pipeline) entirely on
Frag20-Aqsol-100K, then fine-tune on the frozen FreeSolv fold-0 split and compare
against the from-scratch AQM init under identical conditions.

Folder: `frag20/` at the repo root (self-contained; delete for rollback; no
imports from pipeline files). Run pipeline + nohup commands: `frag20/RUN_PIPELINE.md`.

## Vocabulary decision (explicit)
The 17-element AQM vocab (element_vocab.py) ALREADY includes B
(Z=5, index 9). Therefore B-containing Frag20 rows are KEPT - no drop, no vocab
extension. Iodine (Z=53) is in the vocab but has ZERO Frag20 instances (dataset
fact); the I channel stays untrained, exactly as in every prior AQM run. All
filtering is confirmed against the xyz geometry, not just SMILES.

## Design
* prepare_frag20_scratch.py: full-dataset hdf5. SMILES prefilter then tar
  extraction of QM_xyz (B3LYP/6-31G*). Keeps all 17-elem-vocab molecules,
  dataset's own fixed 80K/10K/10K split, plus gas_eV / wat_eV
  (Hartree*27.2114) and calc_sol_kcal. --limit for smoke.
* pretrain_stage1_frag20.py: DimeNet+ (4 blocks, hidden 128) predicts gas_eV,
  energy-only (Frag20 has NO forces -> lambda_force=0) with train-split atomic
  refs. lr 1e-3, batch 32, 200 epochs, patience 10.
* pretrain_stage2_frag20.py: frozen 4-block vacuum + trainable 3-block
  correction. Primary target dG_eV = watEnergy - gasEnergy (the paired
  electronic energy difference; CalcSol = dG*627.509). lambda_total=0.05
  anchors total energy vs watEnergy - refs. lr 1e-3, batch 16, 200 ep, pat 10.
* finetune_freesolv.py: init from stage2_scratch.pt, SAME frozen fold-0 split
  (411/102/129, md5 c0ef293341...), lr 1e-4, wd 1e-5, batch 8, 200 ep,
  patience 30, MSE in eV, 5-conf RDKit TTA. Baselines reported: seed-42 single
  0.5313 / TTA 0.5048.

## Smoke test (local CPU) - PASSED
* prepare --limit 300: 300 mols extracted (241 train/30 valid/29 test),
  element hist H,C,N,O,F,P,S,Cl,Br,B(=4) present, 0 parse fails, 0 vocab drops.
* stage1 --max_structures 64 --epochs 3: runs, refs fit (RMSE ~0.97 eV),
  best val loss 22.57, ckpt saved.
* stage2 --max_structures 64 --epochs 3: frozen vacuum loads, dG head runs,
  val best 1.219 (dG 6.998 kcal RMSE), ckpt saved.
* finetune --quick_test: loads stage2 ckpt, fold-0 split md5 c0ef293341...,
  2 epochs, test MAE 1.616 (smoke only; model barely trained).
- Verified: no existing file modified (only new sandbox files added).

## Vast timing estimates (GPU)
Per-epoch wall time measured on CPU (batch16): stage1 ~19s (64 mols/epoch on
CPU). On a single RTX-4090/A100 the batch-32 stage1 epoch for 80k mols
expects ~20-40s; stage2 batch-16 ~15-30s; finetune batch-8 on 411 mols
already ~<2s/epoch on GPU.

| step | dataset rows | epochs | est. GPU time |
|---|---|---|---|
| prepare (full 100k xyz parse) | 100k | - | 5-15 min (disk-bound) |
| stage1 (batch 32) | 80k train | 200 (early-stop ~100) | 1.5-3 h |
| stage2 (batch 16) | 80k train | 200 (early-stop ~100) | 1-2.5 h |
| finetune (batch 8) | 411 train | 200 (early-stop ~50-80) | 15-40 min |
| TOTAL (single seed 42) | | | ~3-6 h |

For the 5-seed matching ensemble recipe (optional, seeds 42/123/7/2024/999),
multiply stage1+stage2+finetune by 5 (~15-30 h) OR run single-seed seed-42
first for the direct baseline comparison.

## Decisions/risks
- Frag20 SMD-B3LYP energies vs AQM PBE0+MBD: the level-of-theory gap is the
  POINT of the experiment (does scratch pretraining on the cheaper dataset win
  anyway?). The stage-2 head plus fold-0 fine-tune is the comparison axis.
- No forces in Frag20 (documented; energy-only stage1/2). This is a real
  difference vs the AQM 2-stage (which used totFOR lambda_force=1000); the
  large total-energy scale is why both stages use atomic refs for the energy
  loss, same as AQM.
- One geometry per molecule (Frag20) vs AQM multi-conformer: molecule-level
  split is exactly the dataset's own 80/10/10 - no leak, no conformer grouping
  needed; fine-tune uses FreeSolv's own conformers + RDKit TTA.
- single seed 42 first; 5-seed ensemble later if the single-seed signal is
  in the right direction vs AQM-from-scratch.

## Files (all inside repo-root `frag20/`)
- prepare_frag20_scratch.py (builds data/frag20_full.hdf5 + labels + report)
- pretrain_stage1_frag20.py (stage 1 vacuum, gas energy)
- pretrain_stage2_frag20.py (stage 2 correction, dG = wat-gas)
- finetune_freesolv.py (fold-0 fine-tune + single/TTA eval)
- common_scratch.py (self-contained copies of pipeline helpers, adapted)
- element_vocab.py / energy_reference.py / DimeModels.py (verbatim copies)
- RUN_PIPELINE.md (full nohup run pipeline: clone -> download -> stage1/2 -> finetune -> tail -f)
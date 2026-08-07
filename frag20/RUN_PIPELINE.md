# Frag20 from-scratch pretraining -> FreeSolv fine-tune — FULL RUN PIPELINE

Everything lives in this folder (`frag20/`) — self-contained, nothing imports
from the pipeline. Full runs go on a GPU box (Vast) under nohup per AGENTS.md.
The repo IS on GitHub, so you clone it; the Frag20 dataset and the FreeSolv
label JSON are NOT in the repo and are downloaded by `run_all.sh`/the scripts.

VOCAB DECISION: 17-element AQM vocab already includes B (idx 9) -> B rows are
KEPT. I (Z=53) has ZERO Frag20 instances (dataset fact), reported, not hidden.

## QUICKSTART — ONE bash command runs the WHOLE pipeline

```bash
git clone https://github.com/NghiemNgocDuc/training.git
cd training
nohup bash frag20/run_all.sh > frag20/pipeline.log 2>&1 &
tail -f frag20/pipeline.log
```

`run_all.sh` chains every step (0: FreeSolv labels curl if missing;
1: prepare + dataset download; 2: stage1 vacuum; 3: stage2 correction;
4: fold-0 fine-tune + TTA). It is resume-safe (skips any step whose output
already exists), auto-detects CUDA (override: `DEVICE=cpu bash frag20/run_all.sh`),
and writes individual step logs to `frag20/logs/{prepare,stage1,stage2,finetune}.log`.

## What `run_all.sh` does (step-by-step)

0. **FreeSolv labels** — if `Data/FreeSolv/database.json` is missing (NOT in
   git), curls it from `raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json`.
1. **Prepare** — `prepare_frag20_scratch.py --geom qm`; auto-downloads
   Frag20-Aqsol-100K.tar.bz2 (88.9 MB, MIT, NYU IMA) + split CSVs into
   `frag20/data/`, then writes `data/frag20_full.hdf5` (atNUM/atXYZ +
   gas_eV/wat_eV), `data/frag20_full_labels.json`, `data/frag20_filter_report.json`.
2. **Stage 1 (vacuum)** — `pretrain_stage1_frag20.py --device $DEVICE --output_dir output`
   -> `output/stage1_scratch.pt` + `output/stage1_scratch_refs.json`.
3. **Stage 2 (solvation)** — `pretrain_stage2_frag20.py` with frozen vacuum
   ckpt -> `output/stage2_scratch.pt`.
4. **Fine-tune** — `finetune_freesolv.py --init_ckpt output/stage2_scratch.pt
   --output_dir output_finetune --n_conformers 5` -> `output_finetune/metrics.json`
   + `predictions.csv`.

The FreeSolv conformers (`freesolv_conformers.hdf5`) and the frozen fold-0
split arrive via git. Dependencies: `pip install torch torch-geometric h5py numpy rdkit`.

## Debug a specific step

Each step is a plain python call; logs land in `frag20/logs/`. To re-run one
step only (e.g. after a fix), resume from it:
```bash
bash frag20/run_all.sh stage2      # runs stage2 + finetune only
tail -f frag20/logs/stage2.log
```

## 5) Results vs baselines (fold-0 test, kcal/mol)

| init | single-conf MAE | 5-conf TTA MAE |
|---|---|---|
| verified seed-42 AQM scratch (baseline) | 0.5313 | 0.5048 |
| Frag20 scratch (this run, in finetune_out/) | ? | ? |

Timing (single GPU, seed 42): prepare 5-15 min, stage1 1.5-3 h, stage2 1-2.5 h,
finetune 15-40 min -> total ~3-6 h. For the 5-seed ensemble multiply by 5
(optional; run seed 42 first).

## Smoke test (local CPU, tiny)

```bash
python prepare_frag20_scratch.py --limit 300 --no_download
python pretrain_stage1_frag20.py --device cpu --max_structures 64 --epochs 3
python pretrain_stage2_frag20.py --device cpu --max_structures 64 --epochs 3 \
      --stage1_ckpt output_smoke/stage1_scratch.pt \
      --stage1_refs output_smoke/stage1_scratch_refs.json
python finetune_freesolv.py --device cpu --quick_test --n_conformers 1 \
      --init_ckpt output_smoke/stage2_scratch.pt
```

> NOTE: `--no_download` only works when the CSVs/tar already exist (e.g. the
> smoke data). On a fresh clone omit it so downloads happen automatically.
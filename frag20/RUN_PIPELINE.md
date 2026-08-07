# Frag20 from-scratch pretraining -> FreeSolv fine-tune — FULL RUN PIPELINE

Everything lives in this folder (`frag20/`) — self-contained, nothing imports
from the pipeline. Full runs go on a GPU box (Vast) under nohup per AGENTS.md.
The repo IS on GitHub, so you clone it; the Frag20 dataset and the FreeSolv
label JSON are NOT in the repo and must be downloaded by the scripts (they
auto-download if missing — no manual step required).

VOCAB DECISION: 17-element AQM vocab already includes B (idx 9) -> B rows are
KEPT. I (Z=53) has ZERO Frag20 instances (dataset fact), reported, not hidden.

---

## 0) Fresh machine: clone the repo (Linux/Vast)

```bash
git clone https://github.com/NghiemNgocDuc/training.git
cd training/frag20
```

If the code is NOT on GitHub for your copy, copy the folder over and jump to
step 1. Dependencies: `pip install torch torch-geometric h5py numpy rdkit`.

## 1) Build the dataset (auto-downloads if missing)

Downloads Frag20-Aqsol-100K.tar.bz2 (88.9 MB, MIT, NYU IMA) + split CSVs from
whoYouWith91. NOTE: pick `--csv_dir`/`--tar_path` INSIDE this folder so the
downloads are local (defaults are `frag20/data/`).

```bash
# full dataset (~100k mols); first run downloads + parses the tar (few min)
nohup python prepare_frag20_scratch.py --geom qm \
      > frag00_prepare.log 2>&1 &

tail -f frag00_prepare.log
```

Wait until you see `DONE` and:
- `data/frag20_full.hdf5` (~.hdf5 with atNUM/atXYZ + gas_eV/wat_eV)
- `data/frag20_full_labels.json`
- `data/frag20_filter_report.json`

The FreeSolv labels (`Data/FreeSolv/database.json`) and conformers
(`freesolv_conformers.hdf5`) + the frozen fold-0 split arrive via the repo
(git). If `database.json` is missing, fetch:
```bash
mkdir -p ../../Data/FreeSolv
curl -L -o ../../Data/FreeSolv/database.json \
  https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.json
```

## 2) Stage 1 — vacuum (gas energy), from scratch

```bash
nohup python pretrain_stage1_frag20.py --device cuda \
      --output_dir stage1_out \
      > frag01_stage1.log 2>&1 &

tail -f frag01_stage1.log
```

Wait for `DONE` -> `stage1_out/stage1_scratch.pt` + `stage1_scratch_refs.json`.

## 3) Stage 2 — solvation correction (watEnergy - gasEnergy), frozen vacuum

```bash
nohup python pretrain_stage2_frag20.py --device cuda \
      --stage1_ckpt stage1_out/stage1_scratch.pt \
      --stage1_refs stage1_out/stage1_scratch_refs.json \
      --output_dir stage2_out \
      > frag02_stage2.log 2>&1 &

tail -f frag02_stage2.log
```

-> `stage2_out/stage2_scratch.pt`.

## 4) Fine-tune on the frozen FreeSolv fold-0 split + TTA eval

```bash
nohup python finetune_freesolv.py --device cuda \
      --init_ckpt stage2_out/stage2_scratch.pt \
      --output_dir finetune_out \
      --n_conformers 5 \
      > frag03_finetune.log 2>&1 &

tail -f frag03_finetune.log
```

-> `finetune_out/metrics.json`, `predictions.csv`.

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
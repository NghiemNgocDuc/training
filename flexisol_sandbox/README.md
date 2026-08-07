# flexisol_sandbox (experimental, safe to delete)

Scratch pipeline for testing whether the uncertainty-refinement experiment
works on the **FlexiSol-water** subset instead of FreeSolv. **Nothing here
touches the existing `experimental_uncertainty_refine/` code** — delete this
folder entirely to roll back.

## Rationale

FreeSolv (642 mols, 5-seed disagreement) gave NO-GO for both uncertainty
approaches (soft weighting AND hard masking; α=0.0 control 0.5507 vs frozen
baseline 0.5048). FlexiSol (Berciu Sci 2025, DOI 10.1039/D5SC06406F) is a
larger, structurally harder set (824 values, drug-like, H/N/O/F/Cl/Br/I/S/P/Si).
The hypothesis worth testing: with more rows + more diverse chemistry, the
ensemble-disagreement weighting may finally separate signal from noise.
297 **water, neutral, 3D-conf** molecules are usable without extra machinery
(only water solvent matched to our water-phase DimeNet+).

## Files

- `fetch_flexisol.py` — shallow-clones `grimme-lab/flexisol` (MIT) into
  `data/flexisol_repo/`. The repo ships structures + references.
- `build_hdf5.py` — reads water rows from `dgsolv-references.csv`,
  joins `structures.csv`, takes primary conformer `_t0_c0`,
  writes `flexisol_water.hdf5` (same `atNUM`/`atXYZ` schema FreeSolv uses) +
  `labels.json` (`{mol_id: {expt kcal/mol, smiles, name}}`) + a frozen
  ~80/10/10 split in `split/`.
- `inspect_data.py` — validates counts, 17-elem vocab coverage, split
  disjointness (no torch needed).

## Usage

```bash
python flexisol_sandbox/fetch_flexisol.py
python flexisol_sandbox/build_hdf5.py --repo data/flexisol_repo --out flexisol_sandbox/out
python flexisol_sandbox/inspect_data.py --out flexisol_sandbox/out
```

## Vast (GPU): train + coverage in one shot

```bash
cd flexisol_sandbox
nohup bash run_vast.sh > all.log 2>&1 &
tail -f all.log
```

`run_vast.sh` trains the 5-seed ensemble (`--device cuda`, `--out out/ensemble_full`,
skips if the aggregate already exists) then runs the Approach-3 coverage
diagnostic. ~30 min – 1.5 h total on a mid-tier GPU.

Observed build (2026-08-07): 297 molecules, expt [-23.63, +4.86] kcal/mol,
11 distinct elements all inside the 17-element vocab, no geometry skips,
train/val/test = 239/29/29.

## Results (Approach 3 coverage diagnostic, FlexiSol-water)

Run 2026-08-07 on Vast (5-seed ensemble, flexisol_water, n_test=29):
**NO-GO — coverage does not explain ensemble disagreement.**

| metric | value | p |
|---|---|---|
| Spearman(ensemble_std, 1 − max_tanimoto) | **−0.172** | 0.37 |
| Spearman(ensemble_std, abs_error) | 0.116 | 0.55 |

The sign of the first correlation is *negative* — high-uncertainty molecules
do not lack structural coverage (Tanimoto NN vs training pool). Decisive
counter-examples, fully covered (max_sim = 1.000, ≥1 neighbor ≥0.7) yet
high disagreement and/or high error:

- `dibenzo-24-crown-8` — std=2.05, err=7.80, max_sim=**1.000**, top3=1.000, 3 neighbors ≥0.7
- `bitertanol-b` — std=1.03, max_sim=1.000; `kappa-bifenthrin` — std=0.96, max_sim=1.000
- Conversely `fructose` (std=3.58, err=2.32) at only max_sim=0.50 — a sugar,
  hard chemistry despite *not* being an outlier in coverage.

Conclusion: on FlexiSol the 5-seed disagreement tracks neither structural
coverage nor |error| — consistent with the FreeSolv NO-GO (α-sweep all ≥
control). Disagreement appears dominated by hard-chemistry regions even when
well-covered, so targeted coverage-based augmentation of the training pool is
**not predicted to reduce uncertainty**. Recommendation: drop Approach-3
(Frag20/SPICE supplementation is not a targeted fix here).

## Status

- [x] fetch + build + inspect (verified locally)
- [x] retrain 5-seed ensemble std on FlexiSol-water (Vast, run_vast.sh)
- [x] Approach-3 coverage diagnostic (approach3_coverage.py) — NO-GO
- [ ] port into the approach-1 runner (`--dataset flexisol` option)

## Notes / warnings

- Labels are experimental solvation free energies (kcal/mol) in water,
  298.15 K — same units FreeSolv uses.
- Only `chrg0` neutral, `t0` tautomer, first conformer per molecule is used
  (single-conf baseline; conformer ensembles could be added later).
- Does **not** depend on torch_geometric / RDKit; numpy + h5py only.
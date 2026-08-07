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

Observed build (2026-08-07): 297 molecules, expt [-23.63, +4.86] kcal/mol,
11 distinct elements all inside the 17-element vocab, no geometry skips,
train/val/test = 239/29/29.

## Status

- [x] fetch + build + inspect (verified locally)
- [ ] port into the approach-1 runner (`--dataset flexisol` option)
- [ ] retrain 5-seed ensemble std + alpha/hardmask sweep on FlexiSol-water
- [ ] compare vs FreeSolv (r² of `ensemble_std` vs │error│, subset tables)

## Notes / warnings

- Labels are experimental solvation free energies (kcal/mol) in water,
  298.15 K — same units FreeSolv uses.
- Only `chrg0` neutral, `t0` tautomer, first conformer per molecule is used
  (single-conf baseline; conformer ensembles could be added later).
- Does **not** depend on torch_geometric / RDKit; numpy + h5py only.
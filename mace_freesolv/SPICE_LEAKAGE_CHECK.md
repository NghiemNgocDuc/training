# SPICE Leakage Check — is MACE's training data overlapping our test data?

Date: 2026-09-18. Method: connectivity (stereo-stripped SMILES) matching, no
RDKit. Artifacts: `mace_freesolv/spice_overlap_*.csv`.

## What MACE-OFF23 trained on

Gas-phase QM energies+forces (wB97M-D3(BJ)/def2-TZVPPD) from SPICE (subset:
10 neutral elements incl. QMugs 50–90-atom molecules) — NEVER hydration free
energies. Papers: Kovacs et al. arXiv:2312.15211; Eastman et al. Sci Data 2023
(SPICE v2: 113,999 mols / 2M conformers across PubChem 28,039 + solvated
PubChem + DES370K + dipeptides + ion pairs + water).

## What was checked

1. Local `SPICE-2.0.1.hdf5` (374 mols = DES370K-monomer sample): **0 overlap**
   with FreeSolv-642, FlexiSol-water-297, and Guthrie-663.
2. SPICE PubChem screening pool (`openmm/spice-dataset` `pubchem/sorted.txt`,
   fetched 2026-09-18, 456,253 unique CIDs with SMILES) vs our sets by
   connectivity (strip `@/\`):
   - FreeSolv 642: **173 hits** — train 119 / val 26 / **test 28 of 129 (22%)**.
     Typical hits: toluene, ethanol, pyridine, propane (common small organics).
   - FlexiSol-water 297: **9 hits**.
   - Guthrie 663: **28 entries (27 unique connectivities)**.
   - Exact-CID matching gives 0 everywhere (stereo/isomer CID splits), so the
     connectivity numbers above are the operative ones.
3. The 456k pool is the screening pool the 28k-mol PubChem training subset
   derives from — so 173/28/9/28 are UPPER bounds on pretraining exposure,
   not confirmed training membership (exact OFF23 molecule list unpublished).
4. QMugs-subset (50–90 atoms): unchecked at molecule level; only 22 FlexiSol
   mols are size-eligible (none in FreeSolv/Guthrie range). Residual risk.

## Reading for the paper

- No label leakage is possible: SPICE has no hydration free energies, so no
  test label was ever seen in pretraining. Worst case is geometric/chemical
  exposure of common small molecules — the standard foundation-model caveat.
- Our protocol additionally never trains on test labels (frozen splits,
  train-only calibration), identical to the DimeNet arms.
- Suggested disclosure (one sentence, Limitations): "MACE-OFF23 pretraining
  (SPICE gas-phase QM) shares connectivity with 28/129 fold-0 test molecules;
  no hydration labels were seen in pretraining, and all calibration/evaluation
  uses frozen splits disjoint from fine-tuning data."
- For completeness, the 28 test molecules are listed in
  `spice_overlap_freesolv.csv` (column `fold0_split`).

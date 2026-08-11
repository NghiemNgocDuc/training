# Dataset Audit: SPICE and Halo8 for AQM pretraining gaps (Br / I / P)

Status: research-only audit. No pipeline code changed. Scope for adapter design only.
Date: 2026-08-05

## 1. Problem statement (verified against local data)

Current AQM pretraining data covers 17 elements in `element_vocab.py` (`P` idx 5, `Br` idx 15, `I` idx 16), but the actual AQM HDF5 files contain:

| Dataset | conformers | P atoms | Br atoms | I atoms |
|---|---|---|---|---|
| `AQM-sol.hdf5` (CV split, 4k) | 4,000 | 0 | 0 | 0 |
| `AQM-sol-full.hdf5` | 59,783 | 1,258 | 0 | 0 |
| `AQM-gas-full.hdf5` | 59,783 | 1,258 | 0 | 0 |

- `Br` and `I` embeddings (element_vocab idx 15/16, MACE 10-element vocab includes 35/53) are NEVER trained.
- `P` exists but is thin (1,258 atom instances).
- `fit_atomic_references` (`energy_reference.py`) uses `present_mask`: absent elements get reference energy 0.0 — so `Br`/`I` currently have a junk-reference baseline for Stage 1.

User's operating assumption (confirmed below): include **plain SPICE (NOT SPICE2)**; Halo8 presumably has **no iodine**.

## 2. SPICE 1.1.2 (plain SPICE, NOT SPICE2)

### Access (confirmed)
- Paper: Eastman et al., *Scientific Data* 10:11 (2023), DOI [10.1038/s41597-022-01882-6](https://doi.org/10.1038/s41597-022-01882-6) (open access; PMC full text: PMCID PMC9813265).
- Zenodo record: **10.5281/zenodo.7338495** — single file **`SPICE-1.1.2.hdf5`**, **10,437,210,062 bytes (~10.4 GB)**, CC-BY-4.0, open access. Verified via Zenodo API (recid 7338495) — no login/token needed.
- DOI disambiguation: `docs/setup/download_data.txt` lists `10975225` = `SPICE-2.0.1.hdf5`. That is the **SPICE2** record (explicit-water / solvated-PubChem content), which is ruled out per prior decision. Plain SPICE = record `7338495`. The GitHub `openmm/spice-dataset` README now points at the newer series DOI and reports a larger release (113,999 mol / 2,008,628 conformers) — that is a later version, NOT what we want. **Pin `7338495`.**
- Schema (verified from the Zenodo record description, which is the authoritative data-records section): HDF5, one top-level group per molecule/cluster (name = PubChem SID / AA sequence / SMILES). Fields per group:
  - `atomic_numbers` (N), `conformations` (M,N,3) — coordinates in **bohr**
  - `formation_energy` (M) = DFT total minus isolated-atom reference energies
  - `dft_total_energy` (M) — total energies in **hartree**
  - `dft_total_gradient` (M,N,3) — gradient (+dE/dr), i.e., the NEGATIVE of forces
  - `mbis_charges/dipoles/quadrupoles/octupoles`, `scf_dipoles/quadrupole`, `mayer_indices`, `wiberg_lowdin_indices`, `subset`, `smiles`.

### Size / composition (paper Table 1, 1.1.2-era)

| Subset | molecules | conformers | elements |
|---|---|---|---|
| Dipeptides | 677 | 33,850 | H,C,N,O,S |
| Solvated amino acids (20 explicit TIP3P-FB waters each) | 26 | 1,300 | H,C,N,O,S |
| DES370K dimers | 3,490 | 345,676 | H,Li,C,N,O,F,Na,Mg,P,S,Cl,K,Ca,**Br,I** |
| DES370K monomers | 374 | 18,700 | H,C,N,O,F,P,S,Cl,**Br,I** |
| PubChem (drug-like) | 14,643 | 731,856 | H,C,N,O,F,P,S,Cl,**Br,I** |
| Ion pairs (monatomic) | 28 | 1,426 | Li,F,Na,Cl,K,**Br,I** |
| **Total** | **19,238** | **1,132,808** | 15 elements incl. Br, I, P |

### Element coverage (paper Table 2 — the key audit numbers)
Counts are "instances" = number of atoms of that type across the whole dataset (element + formal charge):

| Element | charge | instances |
|---|---|---|
| Br | 0 | 87,927 |
| Br | −1 | 4,276 |
| **Br total** | | **92,203** |
| I | 0 | 21,908 |
| I | −1 | 4,344 |
| **I total** | | **26,252** |
| P | 0 | 41,528 |
| P | 1 | 750 |
| **P total** | | **42,278** |

- vs AQM baseline (P=1,258, Br=0, I=0): SPICE provides **~34× more P**, **92k Br**, **26k I** atom instances. This directly closes the AQM gap.
- Note: Table 2's "H 0 = 1,594" row is internally inconsistent (a dataset of drug-like molecules + dipeptides cannot have fewer H atoms than Br); both Nature and PMC render the same value, so it is a data-entry artifact in the published table. Irrelevant to Br/I/P conclusions, but flag it: re-derive true H counts from the file if needed.
- Br/I/P are concentrated in the **PubChem** and **DES370K** subsets (and small ion-pair counts — ion pairs should be excluded anyway; see below).

### Data-shape audit (the critical question) → **NO solvated/vacuum pairing**
- SPICE energies are **single-environment**: per-conformer `dft_total_energy` / `formation_energy`. There is **no solvent energy field and no gas/solvated pair** — nothing analogous to AQM's `eSOLV` vs `ePBE0+MBD` pairing.
- The subset literally named "solvated amino acids" is **explicit water clusters** (solute + 20 TIP3P-FB water molecules, 79–96 atoms) — i.e., just more atoms in one cluster, not an implicit-solvation energy. It is useless for a solvation-difference target as-is (energies are for the whole cluster) and should be excluded.
- **Verdict: SPICE is a vacuum-stage + atomic-reference-fitting dataset, NOT a Stage-2 solvation-correction target.** Per the audit brief, reframe the plan accordingly.
- Theory: ωB97M-D3(BJ)/def2-TZVPPD via Psi4/QCEngine. This differs from AQM's PBE0+MBD. `formation_energy` already has isolated-atom refs subtracted (at ωB97M level). Mixing theories in one energy target requires per-dataset reference handling (adapter design note below).

### Usability notes
- 28 element+charge atom types (charge-aware). Our vocab is element-only → recommend **neutral-only filter** (drop charged records) for the first pass; precedent: MACE-OFF used a neutral-only, 10-element SPICE subset ≈ 85% of SPICE 1.
- **Exclude: Ion Pairs** (monatomic gas-phase ion pairs, non-AQM-like) and **Solvated Amino Acids** (explicit-water clusters). Keep PubChem + DES370K dimers + DES370K monomers + dipeptides = ~1.03M conformers.
- Strain filter: paper discards conformers with any |F_component| > 1 hartree/bohr = 51.42 eV/Å ≈ our existing `FORCE_THRESHOLD = 52.0 eV/Å`. Consistent — reuse our threshold.
- Unit conversions required by an adapter: bohr→Å (geom), hartree→eV (energies/forces), **negate `dft_total_gradient`** → forces (AQM stores forces, `totFOR`).

## 3. Halo8

### Access (confirmed)
- Paper: "A dataset of chemical reaction pathways incorporating halogen chemistry", *Scientific Data* (2026-01-15), DOI [10.1038/s41597-025-05944-3](https://doi.org/10.1038/s41597-025-05944-3) (full text: PMC12537968).
- Zenodo record: **10.5281/zenodo.16737590** (v1, 2025-08-04), CC-BY-4.0, open. **10 ASE `.db` files** (`Halo_1.db` … `Halo_10.db`), total **47,025,844,224 bytes (~43.8 GiB / ~47 GB)** for 20,116,288 structures. `ase.db` is SQLite-backed; readable with the standard `ase` package (pure-Python, works on this box).

### Composition / shape
- **20,116,288 structures** from **~19,000 unique reaction pathways** (19,176 total: 9,341 halogen + 9,835 recalculated Transition1x). Each pathway = **10 final MEP images**: 1 reactant (Y=0), 8 NEB intermediates, 1 product (Y=9). The highest-energy MEP image = transition state.
- Halogen part: **~10.7M structures** (3.8M F, 3.7M Cl, **3.1M Br**) from 9,341 reactions. Recalculated Transition1x part: 9.4M structures (C/N/O only, no halogens).
- Elements: **H, C, N, O, F, Cl, Br only. Confirmed: NO iodine, NO phosphorus, NO sulfur.**
- Theory: ωB97X-3c (composite), energies in **eV**, forces included, plus Mulliken/Löwdin charges, dipole, HOMO/LUMO, and energy-decomposition terms. Molecules 3–8 heavy atoms.
- **Geometry caveat (explicit in the record): all geometries are GFN2-xTB optimized, NOT DFT stationary points.**

### Data-shape audit → also NO solvation pairing
- Single-environment, off-equilibrium reaction-pathway energies (reactants/intermediates/TS/products). No solvent field, no gas/sol chain. **Not a Stage-2 target.** Relevant to Stage-1 robustness for halogens, and covers equilibrium-like endpoint geometries (Y=0 / Y=9).

### Proposed filtered subset (order of magnitude)
Do NOT take the full 20.1M. For the AQM solvation program the useful slice is Br-containing, equilibrium-adjacent:

| Filter | order of magnitude |
|---|---|
| Br-containing endpoints only (Y∈{0,9} on halogen paths) | ~2×10^4 |
| Br-containing converged-MEP images (10/reaction) | ~1×10^5 |
| Downsampled Br slice of the full 3.1M Br structures | ~10^4–10^5 selected |

Recommend starting order **10^4–10^5** (e.g., MEP endpoints of the 9,341 halogen reactions ≈ 2×10^4), plus the standard force filter (|F| < 52 eV/Å). Skip the T1x portion (no halogens, redundant for our gap).

## 4. Comparison / synthesis

| | AQM (ours) | SPICE 1.1.2 | Halo8 |
|---|---|---|---|
| Access | local | Zenodo 7338495, 10.4 GB, open | Zenodo 16737590, ~47 GB, open |
| Conformers | 59,783 (full) | 1,132,808 (1.03M usable) | 20.1M (3.1M Br) |
| P atoms | 1,258 | **42,278** | 0 |
| Br atoms | 0 | **92,203** | present (3.1M structs) |
| I atoms | 0 | **26,252** | 0 |
| Theory | PBE0+MBD | ωB97M-D3(BJ)/def2-TZVPPD | ωB97X-3c |
| Geometry level | DFT | DFT (MD-sampled confs) | GFN2-xTB (semi-empirical) |
| Solvent pairing | YES (eSOLV vs gas → dG) | **NO** | **NO** |
| Forces | yes | yes (gradient, needs negation) | yes |

## 5. Recommendation (honest, scoped — nothing integrated)

1. **SPICE 1.1.2 is the worthwhile add.** It is the only one of the two that covers **Br AND I AND P** together, at DFT quality, with energies+forces, and it has a direct precedent (MACE-OFF trained on a filtered SPICE subset). Use it for:
   - **Stage-1 (vacuum) pretraining robustness** for Br/I/P chemistry (element vocab already includes all three),
   - **atomic-reference fitting** so Br/I get non-zero reference energies (existing `fit_atomic_references`/`present_mask` handles missing elements — after adding SPICE, Br/I are no longer missing; embed in stage-1 flow without touching `train_stage1_vacuum.py`).
   - Use the neutral-only subset: PubChem + DES dimers + DES monomers + dipeptides ≈ 1.0M conformers; exclude Ion Pairs and Solvated Amino Acids.
2. **SPICE cannot fix the Stage-2 solvation gap.** Neither dataset provides solvated/vacuum energy pairs, so **the Stage-2 dG target remains AQM-only**. If solvated Br/I/P dG data is ever needed, that requires a different source (e.g., PCM-level recomputations à la QMugs/SPACIER, or FreeSolv's experimental Br/I entries) — outside this audit.
3. **Halo8 = optional, lower priority.** Confirms NO iodine; Br-only, no P/S; semi-empirical geometries; off-equilibrium distribution. Only if we want broader Br chemistry robustness at small cost: order-10^4–10^5 Br-endpoint subset, Stage-1 only.
4. **Adapter scope (design only, not written):** one new dataset module per source emitting the AQM `InMemoryDataset` fields (`atNUM`, `atXYZ`, energies, `totFOR`; `eSOLV=None` for gas-only sources):
   - SPICE: `h5py` on `SPICE-1.1.2.hdf5`; convert bohr→Å, hartree→eV, negate gradient→forces; apply 52 eV/Å force filter; map `atomic_numbers` via existing 17-element vocab (unchanged); strip charge types / neutralize, or record charge for later.
   - Halo8: `ase.db` reader (SQLite); select Br paths; endpoint/MEP filter; eV energies already ML-ready.
   - **Energy-scale handling:** per-dataset atomic-reference normalization (fit refs separately per theory; the model sees shifted targets). This is the one nontrivial piece — it implies training one Stage-1 model on a mixed-theory union, which is fine with per-dataset shifts, but needs an explicit decision. Cleanest integration would be: fit refs on AQM alone (unchanged behavior), precompute SPICE's effective element offsets vs AQM per-element means, and store them in the dataset adapter.

## 6. Sources (all verified during this audit)
- SPICE paper (open access): https://www.nature.com/articles/s41597-022-01882-6 ; PMC full text https://pmc.ncbi.nlm.nih.gov/articles/PMC9813265/
- SPICE 1.1.2 Zenodo record (API): https://zenodo.org/api/records/7338495 (file, size, schema, citation)
- SPICE atom-type counts: paper Table 2 (Nature and PMC identical values)
- spice-dataset repo (scripts, latest-release summary): https://github.com/openmm/spice-dataset
- Halo8 paper (open access): https://www.nature.com/articles/s41597-025-05944-3 ; PMC12537968
- Halo8 Zenodo record (API): https://zenodo.org/api/records/16737590 (files, sizes, `dand_id`, MEP/endpoint semantics, GFN2-xTB caveat)
- MACE-OFF precedent (SPICE neutral-only 10-element subset ≈ 85% of SPICE 1): Kovács et al. JACS 147:17598 (2025)

## 7. Files touched
- **None** (research-only audit). Report only.
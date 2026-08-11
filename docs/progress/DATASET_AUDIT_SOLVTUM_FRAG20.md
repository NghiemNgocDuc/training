# Dataset Audit: Solv@TUM (mediatum 1452571) and Frag20-Aqsol-100K

Part C of the Br/I/P coverage audit. Research-only; no pipeline code touched.
Verdict per role framework: (1) Stage 1/2 pretraining supplement/replacement, (2) second fine-tuning dataset (FreeSolv-like), (3) reference-energy-fitting patch.

Date: 2026-08-05. Audit files staged in `C:\Users\User\AppData\Local\Temp\opencode\` (solvatum.sdf, frag20_*.csv).

---

## TL;DR verdicts

| Dataset | Access | Size | Br | I | P | Roles | Verdict |
|---|---|---|---|---|---|---|---|
| **Solv@TUM v1.0** (Hille 2018, JCP 150:041710) | GitHub mirror `hille721/solvatum`, CC BY-SA 4.0 | 658 solutes, 5,877 pts, 144 non-aqueous solvents; SDF 1.4 MB | **28 solutes / 116 pts** | **14 solutes / 84 pts** | **6 solutes / 10 pts** | 2 (weak) | ⚠️ **Reject for Br/I/P fix** — transport/partition (logP-like), not solvation; target = −RT lnK at 298.15 K transfer free energy; no QM energies at all → role 3 impossible. Only iodine source found, but far too thin (14 solutes). |
| **Frag20-Aqsol-100K** (Zhang 2022, JCIM 62:1840) | NYU IMA tar (~400K SDF/XYZ files) + split CSVs, MIT | 100,000 mols (80K/10K/10K); SMD/B3LYP-6-31G* aqueous ΔG | **4,064 mols** | **0 mols** | **5,288 mols** | 2 | ✅ **Recommended role (2)** — large, diverse, gives real (if SMD/B3LYP, not experimental) Br+P coverage; does not touch iodine; B in vocab beyond current 10-element MACE set. Level of theory mismatched → roles 1/3 excluded. |

**Honest bottom line:** Neither dataset fixes the **iodine** zero-coverage in a role-(2)-usable way (Solv@TUM: 84 pts across 14 I-solutes; Frag20: 0 I). Br and P are well covered by Frag20 for fine-tuning-style training. No public dataset surveyed provides PBE0+MBD-consistent reference energies at all → role (3) remains **unresolved** and would require computing a small reference dataset ourselves (DFT PBE0+MBD on a handful of Br/I/P molecules) — see Next steps.

---

## 1. Solv@TUM

### Identity (verified)
- Official record: `mediatum.ub.tum.de/1452571` (Anubis bot-blocked). 2018; Hille, Ringe, Deimel, Kunkel, Acree, Reuter, Oberhofer, "Generalized molecular solvation in non-aqueous solutions by a single parameter implicit solvation scheme", J. Chem. Phys. 150:041710, DOI 10.1063/1.5050938. DOI of record 10.14459/2018mp1452571.001. Also listed on govdata.de "Solv@TUM v 1.0".
- GitHub mirror `hille721/solvatum` (master): repo = README, `solvatum/ui.py`, `data/solvatum.sdf` (1,417,654 B), `data/solvatum_references.bib`, license.
- **License: CC BY-SA 4.0** (share-alike — a derived/copied dataset must inherit BY-SA; MIT objects in Frag20 path are fine, but this one forces SA propagation if redistributed).
- **Contents actually parsed and counted** (SDF downloaded, plain-Python parse, no RDKit needed):
  - 658 unique solute entries, 634 with 3D coords, 11,189 atoms total. Entry fields: Charge, InChI, Molecular Formula, Name, SMILES, dipole moment, mean polarizability, plus per-solvent `logK (<SOLVENT>)` properties.
  - **5,877 logK data points; 144 unique non-aqueous solvents** (matches the ~5,952 published figure within ±1%; difference = solvent entries without geometry or unparseable rows — I should re-tally to reconcile if this matters).
  - logK range −2.19 … 14.17, mean 2.64.
  - Ref: Abraham & Acree compilations; per-solvent bib references in `solvatum_references.bib`.
- **Target quantity:** `deltaG_solv = − ln(10)·RT·logK` @ 298.15 K (default; unit switchable eV/kcal/mol/J). This is the **partition (transfer) free energy between two condensed phases** — NOT a gas→solution solvation free energy, NOT an electronic-energy difference. No QM gas/solvated energies, no forces, no implicit-solvent model. This makes it useless for roles 1/3 regardless of chemistry.

### Br/I/P coverage (counted from SDF)
- Br-containing solutes: **28** (e.g. id 109, 127, 135, 160, 189), **116 data points**.
- I-containing solutes: **14** (e.g. 043, 044, 110, 244, 283), **84 data points**.
- P-containing solutes: **6** (e.g. 332–335, 495), **10 data points**.
- Also present as **solvents**: BROMOBENZENE (61), DIODOMETHANE (37), IODOBENZENE (35), CHLOROBENZENE (97), 1-CHLOROBUTANE (45), FLUOROBENZENE (13), CARBON TETRACHLORIDE (219), CHLOROFORM (224) → halogen/other chemistry is heavily represented in the *solvent* dimension, not the solute dimension.
- Beyond target-vocab elements appear as solutes: Ge, Sn, Pb, Hg, Fe, Si, noble gases.

### Role verdict
- **(1) Pretraining supplement/replacement:** ❌ No (no QM energies/forces, transfer not solvation).
- **(2) Second fine-tuning set like FreeSolv:** ⚠️ Weak. It would only magnify a mismatch: FreeSolv-style experimental ΔG_solv (aqueous) vs Solv@TUM transfer free energies (non-aqueous, still at 298.15 K). 84 I-points is far too small to meaningfully train the I-channel; Br/P better served by Frag20. Marginal value as a *benchmark/test* set for non-aqueous transfer only.
- **(3) Reference-energy fitting patch:** ❌ No reference data.

---

## 2. Frag20-Aqsol-100K

### Identity (verified)
- Paper: Zhang, Xia, Zhang, "Accurate prediction of aqueous free solvation energies using 3D atomic feature-based GNN with transfer learning", J. Chem. Inf. Model. 62:1840–1848 (2022), DOI 10.1021/acs.jcim.2c00260 (PMC9038704).
- **What it is:** 100,000 diverse molecules (Frag20 + CSD20, ≤20 heavy atoms) sampled for **aqueous solvation free energy**, computed by the paper's SMD-B3LYP protocol (continuum solvent DFT). Fixed split **80K/10K/10K**.
- **Host:** NYU IMA `yzhang.hpc.nyu.edu/IMA` — `Datasets/Frag20-Aqsol-100K.tar.bz2` (~400K SDF/XYZ files). Code/datasets/splits in `whoyouwith91/solvation_energy_prediction` (MIT).
- **What I parsed (the repo split CSVs, 21.5 MB total, no full-data download needed):**
  - train.csv 80,000 rows, valid.csv 10,000, test.csv 10,000 (headers: QM_SMILES, QM_InChI, ID, gasEnergy, watEnergy, octEnergy, CalcSol, CalcOct, calcLogP, SourceFile).
  - **Verified relationship:** `CalcSol = (watEnergy − gasEnergy) × 627.509` — confirms entries carry paired QM gas/solvated **electronic energies** (Hartree, B3LYP/6-31G* + SMD), consistent with a ΔG = E_sol − E_gas structure in *shape*, but at the **wrong level of theory vs AQM (PBE0+MBD)**. QM-optimized *and* MMFF-optimized 3D geoms both shipped (single conformer per molecule).

### Br/I/P coverage (counted from SMILES across all 100K)
- Molecules containing **Br: 4,064** (5,521 Br atoms).
- Molecules containing **P: 5,288** (5,423 P atoms; incl. `[PH]`, `[P@]` variants).
- Molecules containing **I: 0** (0 atoms). **No iodine anywhere in the set.**
- Element inventory (atom-instance top-10): C 583,573; O 160,863; N 95,949; S 30,005; F 17,027; Cl 12,607; H (implicit); Br 5,521; P 5,423; B 860 (boron!). **B is in Frag20's vocab but not in the MACE/Fairchem 10-element `{1,6,7,8,9,15,16,17,35,53}` set** → any fine-tune on Frag20 must either drop B-containing rows or extend vocab (and 17-element AQM vocab has no B index either).

### Role verdict
- **(1) Stage 1/2 pretraining supplement/replacement:** ❌ PBE0+MBD (AQM) vs B3LYP/6-31G*-SMD (Frag20) is a **level-of-theory mismatch so severe it would corrupt stage-1 energy/force learning and any reference fitting**. Also: single conformer, no forces.
- **(2) Second fine-tuning set (FreeSolv-like):** ✅ **This is the role.** Frag20 is precisely the pretraining/transfer set the authors used to reach state-of-the-art FreeSolv RMSE 0.719 / MAE 0.417 kcal/mol. As a *fine-tuning*target for our corrected-PBE0+MBD MACE, its SMD/B3LYP values are a calculated proxy (not experimental) with known ~1.3 kcal/mol MAE vs experiment — usable for learning Br/P chemistry at scale, but a poor surrogate for experimental ΔG calibration. Provides genuine, large Br (4,064) and P (5,288) solute coverage.
- **(3) Reference-energy fitting patch:** ❌ Fitting references to B3LYP/6-31G* energies would bake in the wrong total-energy offset; refs must match PBE0+MBD.

---

## 3. Bottom-line recommendation

1. **Iodine: unaddressed by any surveyed public option.** AQM = 0 I; Frag20 = 0 I; Solv@TUM = 14 I-solutes / 84 pts (transfer free energies, useless for ref-fit). If the product needs iodine, the only sound path within this pipeline is **computing our own small PBE0+MBD reference set** (a few dozen Br/I/P molecules; approach already used for AQM's ref energies) — i.e., role (3) becomes "write a generator", not "adopt a dataset".
2. **Br and P: Frag20 is the recommended role-(2) addition** if the goal is more diversity of rare-element solutes for fine-tuning. Requires: (a) level-of-theory caveat documented, (b) filter out B rows or extend vocab, (c) conformer risk (single conformer) checked vs our stage-2 pairing (each Frag20 row = one geometry pair, not AQM's multi-conformer 1,258-mol gas/solvated scheme).
3. **Solv@TUM: reject** for Br/I/P training. It remains internally useful only as a non-aqueous *benchmark* (experimental transfer energies for 658 solutes) — not part of the fix.
4. Even with Frag20, the I-channel stays at zero until we compute refs/energies ourselves. Honest framing for stakeholders: **public datasets cannot close the iodine gap; a ~30–60 molecule PBE0+MBD computation can.**

---

## Next steps (await go-ahead — no further downloads)
- Write a PBE0+MBD reference/energy **generator** for a curated Br/I/P(V) probe set (reuse existing solvent/ref-fitting code paths), sized to fit reference uncertainties, or
- Take the Frag20 role-(2) integration discussions (10-element vs 17-element vocab handling, B-row filtering, level-of-theory caveat doc) into the pipeline design slate.
- Reconcile the 5,877-vs-5,952 count if a Solv@TUM benchmark role ever becomes desirable (likely unnecessary).
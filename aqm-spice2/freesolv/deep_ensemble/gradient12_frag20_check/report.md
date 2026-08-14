# Frag20 coverage check for gradient-12 / certain-47 (fold-0 test)

Status: DONE 2026-08-14. **Verdict: gradient-12 is NOT isolated from
Frag20.** Every gradient-12 molecule has a strong neighbor in the full
Frag20-Aqsol-100K set (best Tanimoto >= 0.38; three are exact matches,
Tanimoto = 1.0), and a non-trivial neighbor in the trainable Br/P
supplement (>= 0.27). Frag20 augmentation is plausible for this subgroup.

## Step 1 - data usability

* `experimental_frag20/data/frag20_brp.hdf5` = the **Br/P-filtered
  supplement only**: 9,260 molecules (pubchem 3,983; numbered sources 10-20;
  CCDC 638; zinc 9), each group `frag20_{source}_{id}` with `atNUM` (int32),
  `atXYZ` (float64) and attrs `smiles`, `calc_sol_kcal`, `has_br`, `has_p`,
  `source`. All 9,260 have SMILES attrs. **Not** the full 100K set - the
  "brp" name is literal (see `prepare_frag20.py`: keep Br OR P, drop B,
  drop out-of-vocab).
* The **full Frag20-Aqsol-100K population** is present in
  `data/split/frag20_{train,valid,test}.csv` = 80,000 + 10,000 + 10,000
  rows with `QM_SMILES` (plus InChI, energies, CalcSol). SMILES are directly
  available for both populations - no geometry-to-structure reconstruction
  needed (point 3 of the brief is moot). The raw geometry tar
  (Frag20-Aqsol-100K.tar.bz2, ~89 MB) was intentionally deleted during
  cleanup; the CSVs + Br/P hdf5 survive.

## Step 2 - Tanimoto check (Morgan r=2, 2048 bits; identical settings to
graph.py / the earlier isolation check)

Queries: the 12 gradient-12 + 47 certain-47 fold-0 test molecules from
`gradient12_descriptor_check/descriptors_all_129.csv`. Parse failures: 0 on
both sides (100,000 + 9,260 Frag20 SMILES, 59 queries).

### Per gradient-12 molecule - best similarity vs Frag20

| mol_id | full100K best | top5 mean | Br/P best | within-FreeSolv best_sim* |
|---|---|---|---|---|
| mobley_3682850 | **1.000** (CCDC_783014) | 0.838 | 0.350 | 0.273 |
| mobley_8449031 | **1.000** (pubchem_1624) | 0.616 | 0.412 | 0.348 |
| mobley_3269565 | **1.000** (pubchem_962) | 0.560 | 0.429 | 0.375 |
| mobley_5052949 | 0.714 (pubchem_5752) | 0.646 | 0.600 | 0.462 |
| mobley_1563176 | 0.714 (19_25660) | 0.671 | 0.389 | 0.302 |
| mobley_4620651 | 0.524 (15_21657) | 0.442 | 0.343 | 0.286 |
| mobley_4252724 | 0.500 (CCDC_320710) | 0.496 | 0.321 | 0.412 |
| mobley_4639255 | 0.500 (pubchem_6171) | 0.451 | 0.308 | 0.375 |
| mobley_4483973 | 0.467 (CCDC_844066) | 0.433 | 0.391 | 0.500 |
| mobley_6257907 | 0.400 (CCDC_165170) | 0.360 | 0.268 | 0.348 |
| mobley_4883284 | 0.400 (CCDC_740699) | 0.400 | 0.368 | 0.412 |
| mobley_1449384 | 0.381 (CCDC_339968) | 0.364 | 0.367 | 0.364 |

*within-FreeSolv best_sim from the earlier neighbor-isolation check
(`rmse_analysis/neighbor_isolation_check/neighbor_similarity_results.csv`).

### Group summaries (max Tanimoto to Frag20)

| group | n | full100K median | full100K mean | full100K min | Br/P median | Br/P min | avg n(>0.3) in 100K |
|---|---|---|---|---|---|---|---|
| gradient12 | 12 | 0.512 | 0.633 | 0.381 | 0.368 | 0.268 | 102 |
| certain47 (control) | 47 | 0.615 | 0.655 | 0.333 | 0.450 | 0.225 | 493 |

## Reads

1. **Exact matches exist**: 3/12 gradient-12 (mobley_3682850, _8449031,
   _3269565) are literally present in Frag20 (Tanimoto 1.0); 7/47 certain-47
   also hit 1.0 (FreeSolv/Frag20 compound overlap is common, as expected for
   common organic molecules). For those molecules Frag20 carries the same
   compound's QM-optimized geometry and QM solvation energy.
2. **Coverage vs control**: gradient-12 is covered somewhat less than
   certain-47 (median 0.51 vs 0.62) but is far from isolated - the weakest
   gradient-12 molecule still finds 0.38, and every one of them has ~100
   Frag20 neighbors above 0.3 on average.
3. **Trainable supplement is the weaker half**: the Br/P hdf5 (the only
   Frag20 data with geometry+labels on hand) covers gradient-12 at median
   0.37 (min 0.27) vs certain-47 0.45. Non-trivial but modest; the full 100K
   coverage is much stronger (median 0.51) - though the full set's
   geometries were deleted with the tar and would need a deliberate ~89 MB
   re-download to become trainable.
4. **This overturns the earlier "probably not"**: the gradient-12 anomaly
   was not explained by isolation from the FreeSolv pool, but Frag20 holds
   real neighbors (incl. identical compounds) for all 12 - augmentation is
   worth a real look, with the caveat that the Br/P-only hdf5 is the
   currently-trainable form and its coverage is modest.

## Artifacts

* `frag20_similarity_check.py` - the check (queries, both Frag20
  populations, per-molecule rows + group summaries)
* `frag20_similarity_results.csv` - per-molecule: full100K and Br/P
  best_sim / best_neighbor / top5 mean+median / count(>0.3)
* `frag20_group_summary.json` - medians/means per group
# Frag20 coverage check for gradient-12 / certain-47 (fold-0 test)

Status: DONE 2026-08-14. **Verdict: gradient-12 is NOT isolated from
Frag20.** Every gradient-12 molecule has a strong neighbor in the full
Frag20-Aqsol-100K set (best Tanimoto >= 0.38; 2 are literally the same
compound - triethylamine, isopropyl methyl ether - and a 3rd "1.0" is a
Morgan r=2 fingerprint collision, see correction below), and a non-trivial
neighbor in the trainable Br/P supplement (>= 0.27). Frag20 augmentation is
plausible for this subgroup - **but only via the full set: the Br/P hdf5
contains none of the exact matches and retains only ~7-9% of gradient-12's
neighbor coverage (see "Br/P subset" section)**.

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
| mobley_3682850 | **1.000** (CCDC_783014)**\*** | 0.838 | 0.350 | 0.273 |
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
\*collision, not identity: mobley_3682850 = cyclohexanone; the 1.0 twins
are C11/C18 monocyclic ketones with identical Morgan r=2 neighborhoods
(see correction in Reads).

### Group summaries (max Tanimoto to Frag20)

| group | n | full100K median | full100K mean | full100K min | Br/P median | Br/P min | avg n(>0.3) in 100K |
|---|---|---|---|---|---|---|---|
| gradient12 | 12 | 0.512 | 0.633 | 0.381 | 0.368 | 0.268 | 102 |
| certain47 (control) | 47 | 0.615 | 0.655 | 0.333 | 0.450 | 0.225 | 493 |

## Reads

1. **Near/identical compounds exist - but one "1.0" is a collision**: 2/12
   gradient-12 are literally in Frag20 (mobley_8449031 triethylamine =
   pubchem_1624; mobley_3269565 isopropyl methyl ether = pubchem_962;
   identical SMILES). mobley_3682850 (cyclohexanone) does NOT occur - its
   reported 1.0 vs CCDC_783014 (cycloundecanone) and 19_18981
   (cyclooctadecanone) is a Morgan r=2 artifact: a monocyclic ketone's
   radius-2 neighborhood is ring-size-invariant, so C6/C11/C18 ring ketones
   share one fingerprint. **Moral: Tanimoto 1.0 at Morgan r=2 is a
   necessary-not-sufficient identity test for rings; SMILES/InChI must be
   checked** (done here). 7/47 certain-47 "1.0"s were not re-verified;
   some may be the same artifact.
2. **Coverage vs control**: gradient-12 is covered somewhat less than
   certain-47 (median 0.51 vs 0.62) but is far from isolated - the weakest
   gradient-12 molecule still finds 0.38, and every one of them has ~100
   Frag20 neighbors above 0.3 on average.
3. **Trainable supplement is the weaker half - much weaker than thought**:
   the Br/P hdf5 (the only Frag20 data with geometry+labels on hand) covers
   gradient-12 at median 0.37 (min 0.27) vs certain-47 0.45, **and retains
   only 6.9% median / 8.7% mean of the >0.3 neighbor counts of the full
   set** (mobley_6257907: 0/15; mobley_4883284: 2/80). ~93% of the useful
   coverage exists only in the full set.
4. **This overturns the earlier "probably not"**: the gradient-12 anomaly
   was not explained by isolation from the FreeSolv pool, but Frag20 holds
   real neighbors (incl. two identical compounds) for all 12 - augmentation
   is worth a real look. The full 100K geometries (the only trainable form
   containing that signal) would require re-downloading the ~89 MB tar.

## Br/P subset: are the exact matches there? (2026-08-14, brp_exact_match_check.py)

Queries: mobley_3682850 (cyclohexanone, C6H10O), mobley_8449031
(triethylamine, C6H15N), mobley_3269565 (isopropyl methyl ether, C4H10O).

* **All 3 are ABSENT from frag20_brp.hdf5 (9,260)** - no Tanimoto 1.0 hit;
  best available there: 0.350 (pubchem_59558), 0.412 (pubchem_73358),
  0.429 (10_60997).
* **Why**: the filter in `prepare_frag20.py` keeps only molecules
  containing Br or P; all three compounds are C/H/O/N-only, so they (and
  their exact twins) were dropped from the subset by design. They survive
  only in the full 100K CSVs.
* **Coverage retention (all 12 gradient-12, sim > 0.3)**: median 6.9%,
  mean 8.7% of full-set neighbor counts survive in the Br/P subset:
  mobley_4639255 24.0%, _1563176 13.8%, _3269565 13.3%, _5052949 12.6%,
  _1449384 12.5%, _8449031 9.1%, _4620651 4.8%, _4252724 4.8%, _3682850
  3.7%, _4483973 3.8%, _4883284 2.5%, _6257907 0.0% (0 of 15).

**Decision input**: the strongest possible Frag20 signal for gradient-12
(the identical compounds + ~93% of >0.3 neighbors) exists ONLY in the full
100K. The Br/P hdf5 alone is effectively a different population for this
subgroup. If Frag20 augmentation is pursued, the ~89 MB tar re-download is
justified (and sufficient); without it, Br/P-only training would add
negligible gradient-12 coverage.

## Artifacts

* `frag20_similarity_check.py` - the check (queries, both Frag20
  populations, per-molecule rows + group summaries)
* `frag20_similarity_results.csv` - per-molecule: full100K and Br/P
  best_sim / best_neighbor / top5 mean+median / count(>0.3)
* `frag20_group_summary.json` - medians/means per group
* `brp_exact_match_check.py` - exact-match presence in Br/P hdf5 +
  per-molecule coverage-retention (full vs Br/P >0.3 counts)
* `brp_exact_match_output.txt` - full run log (twins, element content,
  retention table)
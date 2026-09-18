# Guthrie-Novel External Transfer — Results

Date: 2026-09-18. Scripts: `guthrie_novel/run_guthrie_transfer_step*.py`
(steps 1/1b CID resolution, 2 SDF download, 3 inference, 4 subsets).
Runtime: CID resolution ~15 min (PUG-REST throttled), SDF ~1 min,
inference ~4 min CPU (445 mols x 3 seeds, DimeNet++ frozen checkpoints).
All outputs in `guthrie_novel/`.

## Source and curation

- Upstream: `https://github.com/mobleylab/GuthrieSolv` (depth-1 clone
  2026-09-18), file `guthrie_database.csv` (53,895 rows, 5,309 unique SMILES).
  README warns the sheet mixes units/processes and is largely uncurated.
- Slice used: rows with `dimension3 == 'kcal/mol'` ("final" converted values)
  → 717 rows / 663 unique SMILES. Processes: AQSOL 490, KWG 187, VP 25,
  DGT 8, KGW 7 (Guthrie's own solubility/Henry-to-dG conversions).
- CID resolution via PUG-REST `fastidentity/smiles` (630/663 resolved).
  Exact-CID dedup against FreeSolv PubChem CIDs (`Data/FreeSolv/database.json`,
  all 642 have integer CIDs): **178 overlapping**, leaving **451 novel**.
  (Naive string-match dedup found only 25 — CID step is load-bearing.)
- Overlap-value check (178 mols, Guthrie final vs FreeSolv expt):
  MAE 0.276, max 12.58 (one large outlier). Conversions trusted with that
  caveat; the outlier is retained, not trimmed.
- Geometries: PubChem PUG-REST 3D SDF, chunked download, pure-python V2000
  parse. 445/451 novel CIDs have 3D records; all parsed mols carry explicit H
  (442/445 with H; 3 without). Elements observed: C/H/N/O/S/P/Cl/Br/F/Si —
  all inside the DimeNet 17-element vocab, zero skips at inference.

## Transfer protocol (frozen, same as FlexiSol paper arm)

- Model: frozen FreeSolv 3-seed DimeNet++ (`finetuned_seed42/123/999`).
- Settings reused with NO recalibration: `TAU_STAR = 4.725394227550238e-04`,
  `mu_T` per-seed from the 411 FreeSolv train molecules
  (-0.2150/-0.2112/-0.2106), uniform `lambda_bar = 0.8014`.
- Single PubChem 3D conformer per molecule (NOT TTA-5 — disclosed difference
  vs the FreeSolv in-domain protocol).
- CIs: 10,000 paired bootstrap draws (seed 7), same discipline as GIMS audit.

## Headline (n=445)

| Method | MAE | Delta vs raw | 95% CI |
|---|---|---:|---|
| Raw ensemble | 3.994 | 0 | — |
| GIMS (ours) | 3.126 | -0.868 | [-1.118,-0.615] |
| Variance-weighted (VW) | 3.123 | -0.872 | [-1.119,-0.616] |
| Uniform (matched) | 3.147 | -0.848 | [-1.056,-0.631] |

Paired: GIMS-VW +0.004 [-0.002,+0.009]; GIMS-Uniform -0.020 [-0.066,+0.025].

## Subsets (post-hoc, no family-wise correction)

| Subset | n | raw | GIMS d | VW d | Uni d | G-VW | G-Uni |
|---|---:|---:|---:|---:|---:|---:|---:|
| heavy N>31 | 210 | 4.800 | -1.734 | -1.745 | -1.622 | +0.011 | -0.111 |
| light N<=31 | 235 | 3.275 | -0.094 | -0.091 | -0.155 | -0.003 | +0.061 |
| N>=25 | 328 | 4.170 | -1.113 | -1.121 | -1.081 | +0.008 | -0.032 |
| N>=30 | 239 | 4.532 | -1.544 | -1.556 | -1.461 | +0.012 | -0.083 |
| Lambda>0.9 | 437 | 4.011 | -0.881 | -0.884 | -0.860 | +0.003 | -0.021 |
| exp<-8 (hydrophilic) | 186 | 2.992 | +0.957 | +0.961 | +0.631 | -0.004 | +0.326 |
| exp>-4 (hydrophobic) | 91 | 4.784 | -1.733 | -1.744 | -1.486 | +0.010 | -0.248 |

No slice shows GIMS ahead of VW beyond noise. Note the hydrophilic slice:
ALL shrinkage arms HURT there (+0.63..+0.96) — shrinking very-negative dG
toward the pool mean pulls the wrong way. Expected failure mode, kept on record.

## Verdict

- Transfer replicates: all three flat methods beat raw on a second new set
  (FlexiSol 297 was the first), CIs exclude zero.
- Ranking here is VW ~= GIMS > uniform (VW wins by 0.004, NS) — NOT a second
  GIMS>VW win. The uniform separation is directional only (NS).
- Paper framing: "flat shrinkage transfers 2/2 (FlexiSol, Guthrie-novel)";
  do NOT claim GIMS>VW on Guthrie. Hydrophilic-hurt slice is a disclosed
  limitation of pool-mean shrinkage, not of invariance.

## Artifacts (this directory)

- `guthrie_all_methods.csv` — per-molecule cid/exp/raw/gims/uniform/vw/Lambda/N
- `guthrie_final.json` — novel CID -> value (451)
- `guthrie_cids.json` — all resolved SMILES -> CID/value (provenance)
- `guthrie_novel_approx.json` — pre-CID string-match novel set (provenance)
- `guthrie_overlap_check.json` — overlap agreement stats
- `guthrie_3d.pkl` — parsed 3D coords + values (rerun without re-download)
- `run_guthrie_transfer_step*.py` — steps 1/1b/2/3/4

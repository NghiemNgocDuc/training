# Page Budget Plan — SIMBIOCHEM 2026 submission

Working title: *Solvation Free-Energy Ensembles: When Confidence and Uncertainty Disagree*
Format: NeurIPS 2026 template, `dblblindworkshop` mode (no `final`, no `preprint`), US Letter.
Limit: **5–8 pages main body** (references, appendices, data-availability unlimited).
Target: **~7.5 pages main body** (≈4,200 words + 2–3 figures + 2–3 tables), leaving headroom to 8.

## Chapter → placement map (from PAPER_SKELETON.md)

| Skeleton chapter | Status | Placement | Notes |
|---|---|---|---|
| Ch 1.1–1.2 molecule-level smoothing | negative (wrong granularity) | Main §2 (motivation, condensed) | 1 table (best-λ rows only), 1 sentence on latent_normalized harm. Full 17-run table → Appendix A |
| Ch 1.3 broad-uncertainty reanalysis | finding | Main §2 | Q_std/Q_nll disjointness (Jaccard 0.21), gradient-12 disjoint — one paragraph + small table |
| Ch 1.4 converging-on-each-other diagnostic | audit | Appendix B (condensed) | 3-proxy dispersion check; reusable audit idea gets 2 sentences in §5 |
| Ch 2.2–2.6 molecule-level repair | intermediate baseline | Appendix A (table) + 1 paragraph in §4.1 | Stepping stone; head-to-head vs node-level in appendix table |
| Ch 2.5 node-level refinement (trust) | run, mechanism falsified | Main §3–§4 | Trust = one arm in the main table, not the method |
| Ch 2.5 λ-calibrated shrinkage | candidate method → superseded | Main §3 (constant-λ = special case of vw) | λ*=1.0 result folded into §4.1 as ablation arm |
| Part C variance-weighted shrinkage | refinement (NOT the engine) | Main §3–§4 | §7 verdict (2026-08-18): headline mechanism = uniform moderate-strength shrinkage of all atoms (interior τ²* protects confident atoms); vw adds unanimous-but-non-significant increment (0.04–0.85 kcal/mol); uniform@0.8014 arm = strongest competing explanation, in main table |
| Ch 2.5 mechanism controls (randAll, pool-mean, intra-molecular) | verified | Main §4.2 | The honest contribution: gating, not similarity transfer |
| Ch 2.5 H1/H2 holdout (Part B) | verified | Main §4.1 (1 row) + Appendix B | λ*_H1 = 1.0; shrink−trust sig on 5/5 H2 pops |
| Ch 2.5 seed resolution (3-seed) | audit | Main §3 (method note) + Appendix B | 3-seed 42/123/999; seeds 7/2024 pathology documented |
| Ch 2.5 gradient-12 mechanism (Parts 8–9) | case study, non-generalizing | Main §4.2 (short) + Appendix B | Case-study failure mode; ratio NOT a predictor |
| Ch 2.5 arithmetic audit (Part 9) | audit | Appendix B (2 sentences) | All headline numbers reproduced |
| Ch 3 Frag20 augmentation | declined | **Excluded** | 1 clause in related-work (no experiments shown) |
| Ch 4 charges ablation | closed negative | **Excluded** | Optionally 1 sentence in Appendix A footnote; default omit |
| Ch 5 GBn2 baseline | baseline | Main §4.3 (benchmark table) + Appendix A (per-fold) | Physics floor; fold-0 in-sample scaled 2.13 vs model fold-0 0.5059 (4.21×); 5-fold CV 0.549±0.024 = separate headline |
| Leakage reconciliation (664→623, 36/9/12 pairs) | audit | Main §2 (hygiene paragraph) + Appendix C | Benchmark-hygiene claim; details in appendix |
| Appendix: venue plan | PAUSED | — | Not paper content |

## Proposed paper structure (main body)

1. **Introduction** (~1.0 p): ML solvation FEs; ensemble uncertainty; the confidently-wrong anomaly; contributions: (i) moderate-strength shrinkage of essentially all atoms as the mechanism (interior optimum protects confident atoms from full replacement), (ii) per-atom variance weighting as a unanimous-but-non-significant refinement (matched-strength uniform control is the strongest competing explanation), (iii) rigorous control-arm/audit methodology (random-placebo, H1/H2 holdout), (iv) physics benchmark vs GBn2.
2. **Setup & motivation** (~1.0 p): dataset (SPICE2+FreeSolv, 642 mols, 411/102/129 frozen split, leakage-audited — 36/9/12 cross-set pairs removed), DimeNet++ 3-seed ensemble {42,123,999}, populations (Q_std n=33, Q_nll n=33, UNION n=50, all129, gradient-12 n=12), molecule-level smoothing failure as wrong-granularity motivation (condensed table).
3. **Method** (~1.5 p): per-node contributions P_i, node uncertainty u_i, cross-molecular pool, gate = top-quartile u3; **moderate-strength shrinkage**: P'_i = (1−λ_i)P_i + λ_i·μ_pool, with (a) uniform λ (all atoms, strength matched to vw: λ=0.8014) and (b) variance-weighted λ_i = u_i²/(u_i²+τ²), τ² calibrated on val (grid 1e-8·var..100·var(σ²); τ²* = 4.725e-4 = interior optimum); constant-λ (λ*=1.0) and trust-weighting as ablation arms; Mode A per-member + re-ensemble; controls (naive, random-shift, randAll_equal placebo); 10k paired-bootstrap CIs; seed-resolution note (1 sentence, detail in appendix).
4. **Results** (~2.5 p):
   - §4.1 main table: arms (uniform@0.8014 / vw / constant-λ / trust / no-correction baseline) × populations, ΔMAE + CI; uniform@0.8014 captures ~87–89% of vw's gains (strongest competing explanation); vw best on all 5 (Q_std −6.72, Q_nll −6.73, UNION −4.61, all129 −1.97, gradient12 −0.51), paired vw−uniform CIs all include 0 (refinement, not mechanism); H1/H2 holdout row.
   - §4.2 mechanism: random-neighbor/pool-mean ≥ trust ⇒ shrinkage not transfer; intra-molecular ablation null ⇒ cross-molecular pool needed; gradient-12 case study (gated sums small, trust uniquely harmful, ratio not a predictor).
   - §4.3 benchmark: GBn2 table (fold-0 in-sample scaled 2.13, τ 0.54; model fold-0 0.5059 → **4.21×**; raw fold-0 36.1×; 5-fold CV 0.549±0.024 quoted separately; slope 0.11–0.16 caveat; GBn2 scaled = in-sample affine-calibrated, must be labeled in caption).
   - §4.4 robustness: false-consensus spread diagnostic; 3-seed reproducibility.
5. **Discussion & limitations** (~0.75 p): James–Stein reading; gating as real contribution; limitations (single dataset/backbone, retrospective H1/H2 split, gradient-12 H2 n=5 underpowered, absolute MAEs regime-bound).
6. **Related work** (~0.5 p): equivariant GNNs, uncertainty in MLFF, classical GB solvation, test-time refinement, James–Stein shrinkage.
7. **Conclusion** (~0.25 p).

Figures: (1) method schematic (node pool → gate → variance-weighted shrink), (2) forest plot of ΔMAE arms × populations with CIs, (3) GBn2 vs model parity/rank plot.
Tables (main): T1 setup/populations; T2 main results arms × populations; T3 GBn2 benchmark.

## Appendices (unlimited)
- **A**: full Ch1 17-run sweep table; molecule-level repair head-to-head; GBn2 per-fold table; charges footnote (optional).
- **B**: validation & diagnostics (6-part audit summary, H1/H2 protocol + all 25 cells, seed-resolution evidence, arithmetic audit, Part 8/9 details, false-consensus numbers).
- **C**: data & experimental details (leakage reconciliation summary: 57 pairs → 36/9/12, 664→623 exclusions; conformer/MMFF/GFN2 protocol; training hyperparameters; val-calibration grids).

## Open items before drafting
1. **DONE (2026-08-18): PAPER_SKELETON.md Ch 2.5 + this plan synced to §7 verdict** — headline mechanism = uniform moderate-strength shrinkage of all atoms (interior τ²* protects confident atoms); variance-weighted shrinkage = unanimous-but-non-significant refinement (0.04–0.85 kcal/mol, paired CIs include 0); uniform@0.8014 arm in main table as strongest competing explanation. Skeleton + plan now predate Part C correctly.
2. Anonymization: no author/affiliation/ack/GitHub links anywhere in the LaTeX tree (audit of referenced docs passed 2026-08-18); code-link placeholder = "Code will be released upon acceptance."
3. First draft uses figures-as-placeholders; regenerate publication-quality figures from verified CSVs only.
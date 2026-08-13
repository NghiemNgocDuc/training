# Neighbor-Consistency Regularization — v2 Design

Status: plan reviewed; implemented + CPU-validated locally (2026-08-13).
Awaiting box smoke run.

## 0. Cross-fold pooling check (item 6) — DONE, committed `70ce974`

`crossfold_isolation_check.py` + `neighbor_isolation_check/crossfold_isolation/`:

| definition | fold-0 only | pooled folds 0-4 |
|---|---|---|
| isolated vs own-fold train pool (best_sim <= 0.22) | 2 | 14 / 642 test |
| isolated vs own-fold train pool (<= 0.30) | 8 | 50 / 642 |
| structurally isolated vs 642 universe (<= 0.22) | — | 9 |

Pooling 5 folds more than doubles n (14 vs 6 fold-0 isolated-6). Conclusion:
fold-0-only (n=6) is enough for the smoke/early read; plan a multi-fold version
later only if the isolated-6 read justifies it.

## 1. Exact loss formula (v2)

Per molecule i (all 642 fold-0 universe; latent/trust/uncertainty signals are
STATIC, computed once — only p_i are current-model predictions, as in v1):

```
w_ij    = latent cosine similarity  (PRIMARY; Tanimoto kept for reporting/Jaccard)
t_j     = 1[ mean_nll_j <= Q_tau(mean_nll) ]     # trust gate, tau default 0.5
u_i     = rank(mean_nll_i)/N, rank in [0,1]      # uncertainty weight (rank-based)
A       = { i : n_trusted_i >= 1 AND sum_j w_ij*t_j >= 1e-6 }   # active set
denom_i = sum_{j in N_i} w_ij * t_j

L_neighbor = (1/|A|) * sum_{i in A} u_i * [ sum_{j in N_i} w_ij*t_j*(p_i - p_j)^2 / denom_i ]

L_used = L_neighbor / var(p)        # only with --normalize_nbr (as v1)
Total  = task_loss + lambda_nbr * L_used
```

- `N_i` = top-5 neighbors by latent cosine (k=5, min_sim default 0.5).
- Neighbor pull only toward TRUSTWORTHY neighbors (t_j); untrustworthy
  neighbors are excluded from both numerator and denominator.
- Each molecule's own pull strength scaled by its uncertainty u_i.
- Isolated/low-coverage molecules fall out of active set A (see §3).

Implementation notes (divergence from plan §1-§2): trust threshold defaults to
the certain-47 group MEDIAN NLL (policy `certain47_median`), NOT a quantile of
the universe; empirical result with this policy: 182/642 trusted, 302/642 in
fallback, and 4/6 of isolated-6 have NO trusted neighbor (no gradient from the
graph — the documented structural ceiling, §4). `--coverage_floor` defaults to
1e-6 (equivalent to "at least one trusted neighbor").

## 2. Where each signal comes from (reuse, not recompute)

| signal | source | reuse? |
|---|---|---|
| Tanimoto graph | `graph.py` (Morgan r=2, 2048-bit) + cached `graph_cache/graph_k5_sim0.1.json` (642 nodes, top-5, min_sim 0.1) | fully cached (reporting only) |
| latent extraction | `gmm_uncertainty_check.py::extract_latents` (hooks on `output_blocks[b].lin`, 4x256=1024-dim per-atom), fine-tuned seed-42 ckpt sha 7994ef92 | reuse code + cached `z_train.npz` (411 mol) / `z_test.npz` (129); NEW one-time extraction ONLY for the 102 val molecules |
| molecule latent | mean-pool per-atom 1024-dim -> 1024-dim per molecule -> L2-normalize -> cosine | derived from above |
| latent graph | top-5 cosine neighbors (min_sim 0.5) -> `graph_cache/latent_k5_sim0.5.json` (+ .meta.json) | new cache (one-time) |
| GMM-NLL | cached `z_train.npz` -> REFIT StandardScaler + PCA(k=13) + GMM(n_components=10, reg_covar=1e-2, n_init=5) on train atoms only (7389); score all 642 atoms -> per-molecule mean NLL | reuses cached z/code; GMM was never saved, refit ~8 s |
| validation | refit NLL for the 129 test mols: Spearman = 1.0000 vs cached `validation/per_molecule_gmm_nll_refit.csv` (protocol reproduced exactly) | cached |
| trust threshold | certain-47 group median NLL (=18.30, 182/642 trusted) | derived |
| uncertainty | u_i = rank(mean_nll)/642 (rank-based, robust to heavy tails) | derived |
| Jaccard overlap | latent-vs-Tanimoto top-5: mean 0.110, 268/642 zero-overlap (the two graphs are nearly orthogonal views) | derived |

Uncertainty signal choice: **GMM-NLL** (validated n_components=10, held-out
protocol) over ensemble-std (would need 5-seed forwards over 642; only test-129
cached; GMM-NLL is the signal the group definitions were already built around).

## 3. Zero/low-neighbor safeguard (item 4) — pseudocode

```
N_i      = top-5 latent neighbors of i (w_ij, min_sim filter)
t_j      = trust gate                              # precomputed per j
S_i      = sum_j w_ij * t_j                        # static, computed at build

if S_i < coverage_floor (default 1e-6):
    contribution_i = 0.0          # NOT NaN, NOT inf, no gradient explosion
    logged per run to epoch_fallback.csv (mol_id, S_i)
else:
    contribution_i = u_i * sum_j w_ij*t_j*(p_i-p_j)^2 / S_i

L_neighbor = mean over ALL i of contribution_i     # fallback nodes = 0
```

- `epoch_fallback.csv` written once per run (static graph -> static fallback
  set; per-epoch count reported via epoch_history.csv `n_fallback` column).
- With default policy: 302/642 nodes in fallback; 4/6 of isolated-6 have no
  trusted neighbor at all -> the loss CANNOT help them (structural ceiling).

## 4. Documented ceiling (item 5)

Docstring + README + smoke report note: the method redistributes information
already present among the ~500 training molecules; molecules with genuinely no
trustworthy neighbors (isolated-6 structurally) cannot be helped — safeguard in
§3 makes this visible rather than hidden. Absence of isolated-6 improvement =
expected structural limit.

## 5. Added compute vs v1

| phase | added cost |
|---|---|
| one-time init | ~1-2 min CPU: 102-val latent forward (~30 s), GMM refit (~8 s), cosine top-k 642x642x1024 (~1 s), trust/rank precompute (ms) |
| per epoch | ~0 — all weights static; same single full-graph pass + backward as v1; only per-epoch bookkeeping (fallback count) |
| total | v2 ~= v1 + ~2 min one-time |

## 6. Smoke test (v2)

Same grid as v1: seed 42, 15 epochs, raw {0, 0.001, 0.01}, normalized
{0, 0.1, 1.0} on GPU box, run via `run_smoke_box.sh` with `--neighbor_source
latent --k_nbr 5 --min_sim 0.5` added to each run. Stability checks same
(NaN/Inf, task loss decreasing, L_neighbor scale-stable for normalized) PLUS:

- per-run `epoch_fallback.csv` (static set, 302 molecules) + `n_fallback` per
  epoch in epoch_history.csv
- Jaccard latent-vs-Tanimoto overlap in the latent meta (static, 0.110 mean)
- Spearman(refit NLL, cached NLL) = 1.0000 baked into config.json provenance
- early isolated-6 trajectory read via --track_groups (with the caveat that
  4/6 of isolated-6 are structurally unreachable — see §4)

# Neighbor-Consistency Regularization (Stage-3 fine-tuning, fold-0)

Goal: during FreeSolv fine-tuning, pull each molecule's prediction toward a
weighted mean of its top-5 similar molecules' predictions (static Morgan
r=2/2048 Tanimoto similarity graph), so uncertain/isolated molecules learn from
their well-predicted neighbors **during training**, not just post-hoc.

## Design decision for the loss (step 3)

Implementation: **one full-graph differentiable forward pass per epoch**.

- `L_neighbor = mean_i [ sum_j w_ij (p_i - p_j)^2 / sum_j w_ij ]`, edges =
  top-5 Tanimoto neighbors with `w >= 0.1`, self edges excluded (exact user
  formula).
- `p_i` and `p_j` are the model's **current** predictions computed in ONE
  forward pass over all 642 molecules, so both sides receive gradient (the
  objective is exact; no stop-gradient approximation needed).
- The existing per-batch task loop (MSE-in-eV, shuffle, scheduler, grad clip)
  is byte-identical to `deep_ensemble.train_member`; the graph pass is one
  extra `backward + optimizer.step()` per epoch (or per `--graph_every N`).
- Why not neighbor-clustered batching: it restructures the DataLoader, blows up
  batch size (molecule + 5 neighbors), duplicates molecules across batches, and
  breaks shuffle semantics for a purely additive loss. The full-graph pass is
  ~2-4 s/epoch on GPU (642 vs 411 forwards), keeps the harness untouched, and
  is trivially transductive.

## Transductive, no label leak

- Graph nodes = ALL 642 fold-0 molecules (train+val+test structures; also the
  full FreeSolv universe). A label-free `GraphDataset` (z/pos/mid only - no
  `y_dG`) feeds the graph pass; `expt` labels are read only for train batches
  and test evaluation. Verified: all 129 test nodes are active (have >= 1
  eligible neighbor).

## Scale finding (CPU smoke, seed 42, 2 epochs, device cpu)

| run | L_nbr @ ep1 -> ep2 (eV^2) | val MAE ep2 | test MAE ep2 (TTA) |
|---|---|---|---|
| lambda=0 (baseline, no graph pass) | - | 7.5 | 7.0 |
| lambda=0.001 | 1.636 -> 1.115 | 13.3 | 14.9 |
| lambda=0.1 | 1.636 -> 1.882 | 13.2 | 13.8 |

Measured magnitudes at init: train task MSE ~ **320 eV^2** (dominated by the
same 6-10 eV outliers as the val ~19 kcal/mol start), L_nbr ~ 1.64 eV^2,
var(p) ~ 3.56 eV^2. So by VALUE the raw consistency term is only ~0.5% of the
task loss, yet it measurably perturbs the trajectory even at lambda=0.001 ->
the effect is per-step GRADIENT CURVATURE (one graph pass = one full-model
optimizer step over 642 structures), not the loss-value ratio. The normalized
variant (L_nbr / var(p), verified: 1.636/3.557 = 0.460 at init) removes the
arbitrary eV^2 scale; whether that makes lambda ~0.1-1.0 behave sanely is the
empirical question the 15-epoch smoke answers. Early epochs are heavy-tail
dominated (stage-2 init has 80-150 kcal/mol outliers predating the fine-tune),
so short smokes pin scale and stability, not final numbers.

Options on the table (both will be swept):
  A) raw formula; lambdas {0.001, 0.003, 0.01, 0.03}
  B) normalized L_nbr / var(p); lambdas {0.05, 0.1, 0.3, 1.0}
  shared lambda=0 baseline for both; every run logs task loss and L_nbr
  separately per epoch (epoch_history.csv) plus NaN/Inf checks.

## Files

- `graph.py`            - similarity graph build/cache (reused by the
                          isolation check; single source of truth)
- `finetune_nbr.py`     - trainer (args: --lambda_nbr --k_nbr --min_sim
                          --graph_every --seed --out ...)
- `report_results.py`   - aggregates runs; subgroups all129 / wrong18 /
                          certain47 / isolated6 / gradient12
- `run_sweep.sh`        - detached GPU-box launcher for the sweep

## Run

```
python finetune_nbr.py --lambda_nbr 0.1 --out results_lambda0.1 --epochs 200
bash run_sweep.sh   # launches lambda in (0 0.05 0.1 0.3) x seeds, detached
python report_results.py --runs results_lambda0_seed42 results_lambda0.05_seed42 ...
```

Expectation to confirm/refute: isolated-6 (best_sim <= 0.22 vs the certain
pool) see little benefit (their edge weights are tiny); gradient-12 the
majority of any improvement; wrong18 as a whole improve without degrading
certain47.
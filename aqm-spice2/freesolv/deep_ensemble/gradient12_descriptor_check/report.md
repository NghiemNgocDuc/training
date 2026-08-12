# Gradient-12 signed-error & physicochemical descriptor check

p-values are **NOT multiple-testing corrected** (9 descriptors + 2 controls); treat everything here as hypothesis-generating, consistent with prior analyses.

Signed error convention: `ensemble_mean - true_value` (kcal/mol).
`signed_error < 0` = prediction MORE negative than experiment = OVER-prediction (model says solvation is more favorable than measured).
`signed_error > 0` = UNDER-prediction.

## Groups
- wrong18 (low_std_high_rmse): 18 | gradient12 = wrong18 - isolated6: 12 | isolated6: 6 | certain47: 47

## Part A - sign of errors

| group | n | over | under | %over | %under | majority frac | binomial p | mean signed | med signed |
|---|---|---|---|---|---|---|---|---|---|
| gradient12 | 12 | 5 | 7 | 41.7 | 58.3 | 0.583 | 0.7744 | 0.089 | 0.380 |
| isolated6 | 6 | 4 | 2 | 66.7 | 33.3 | 0.667 | 0.6875 | -0.253 | -0.426 |
| wrong18 | 18 | 9 | 9 | 50.0 | 50.0 | 0.5 | 1.0000 | -0.025 | -0.029 |
| certain47 | 47 | 24 | 23 | 51.1 | 48.9 | 0.511 | 1.0000 | -0.001 | -0.005 |

### Per-molecule (gradient-12 & isolated-6) with per-seed sign stability

| mol_id | group | signed err | direction | seed signs (42,123,7,2024,999) |
|---|---|---|---|---|
| mobley_4639255 | gradient12 | -0.841 | over | ----- |
| mobley_6257907 | gradient12 | -0.618 | over | ----- |
| mobley_4883284 | gradient12 | -0.565 | over | ----- |
| mobley_3682850 | gradient12 | +0.426 | under | +++++ |
| mobley_4620651 | gradient12 | +0.805 | under | +++++ |
| mobley_8449031 | gradient12 | +1.012 | under | +++++ |
| mobley_3269565 | gradient12 | +0.455 | under | +++++ |
| mobley_5052949 | gradient12 | +0.862 | under | +++++ |
| mobley_1563176 | gradient12 | +0.344 | under | +++++ |
| mobley_4483973 | gradient12 | -0.822 | over | ----- |
| mobley_1449384 | gradient12 | +0.417 | under | +++++ |
| mobley_4252724 | gradient12 | -0.402 | over | ----- |
| mobley_3359593 | isolated6 | -0.715 | over | ----- |
| mobley_7150646 | isolated6 | +0.546 | under | +++++ |
| mobley_7690440 | isolated6 | +0.474 | under | +++++ |
| mobley_9913368 | isolated6 | -0.974 | over | ----- |
| mobley_766666 | isolated6 | -0.410 | over | ----- |
| mobley_8885088 | isolated6 | -0.442 | over | ----- |

## Part B - descriptors (Mann-Whitney gradient12 vs certain47)

| descriptor | med g12 | med c47 | mean g12 | mean c47 | U | p |
|---|---|---|---|---|---|---|
| logp_crippen | 1.618 | 1.766 | 1.560 | 1.904 | 242.0 | 0.4570 |
| tpsa | 6.235 | 12.890 | 12.362 | 12.603 | 274.5 | 0.8910 |
| rotatable_bonds | 1.000 | 1.000 | 1.250 | 1.872 | 236.0 | 0.3781 |
| h_bond_donors | 0.000 | 0.000 | 0.417 | 0.149 | 337.5 | 0.1254 |
| h_bond_acceptors | 1.000 | 1.000 | 0.917 | 0.830 | 307.5 | 0.6129 |
| molecular_weight | 99.131 | 107.156 | 102.444 | 111.902 | 228.5 | 0.3182 |
| num_rings | 1.000 | 0.000 | 0.583 | 0.362 | 349.5 | 0.1356 |
| fraction_csp3 | 0.817 | 0.833 | 0.619 | 0.685 | 263.0 | 0.7178 |
| formal_charge | 0.000 | 0.000 | 0.000 | 0.000 | 282.0 | nan |

## Part B - Spearman across ALL 129 test molecules (continuous check)

| variable | n | rho vs signed | p vs signed | rho vs |err| | p vs |err| |
|---|---|---|---|---|---|
| logp_crippen | 129 | 0.098 | 0.2675 | -0.163 | 0.0655 |
| tpsa | 129 | -0.052 | 0.5551 | 0.268 | 0.0021 |
| rotatable_bonds | 129 | -0.020 | 0.8222 | -0.132 | 0.1359 |
| h_bond_donors | 129 | 0.066 | 0.4597 | 0.235 | 0.0073 |
| h_bond_acceptors | 129 | -0.002 | 0.9796 | 0.242 | 0.0058 |
| molecular_weight | 129 | 0.146 | 0.0990 | 0.164 | 0.0639 |
| num_rings | 129 | 0.188 | 0.0328 ** | 0.274 | 0.0017 |
| fraction_csp3 | 129 | -0.029 | 0.7430 | -0.170 | 0.0540 |
| formal_charge | 129 | constant (no variance) | - | - | - |
| mean_nll | 129 | -0.066 | 0.4574 | 0.397 | 0.0000 |
| best_sim | 36 | 0.108 | 0.5325 | -0.392 | 0.0181 |

## Part C - headline
- Most-correlated continuous descriptor: **num_rings** (see scatter_signed_error_vs_descriptor.png).
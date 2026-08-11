# GMM uncertainty analysis — validation folder

## Contents

- `heldout_ll_curve.png` / `heldout_ll_curve.json` — molecule-level 80/20
  (329 fold-A / 82 held-out molecules) held-out log-likelihood over
  n_components = {1, 5, 10, 20, 50}. Clear interior maximum at **n=10**;
  the BIC choice (n=50) overfits held-out atoms by ~25 nats/atom.
- `per_molecule_gmm_nll_refit.csv` — per-molecule mean/max NLL for the 129
  test molecules at the validated n=10 (and correlational inputs).
- `gmm_refit_report.json` — test-set statistics (Spearman vs ensemble_std /
  abs_error / seed_rmse; Mann-Whitney isolated-6 vs gradient-12 and
  wrong-18 vs certain-47) at the validated n=10 and, for comparison, at the
  old BIC choice n=50.

## Statistical caveats (READ THIS before citing any p-value)

**None of the p-values in this analysis are corrected for multiple testing.**
This applies to every p-value produced in this project, including:

- ensemble-std vs RMSE correlations (rmse_analysis)
- GMM-NLL vs ensemble_std / abs_error / seed_rmse Spearman correlations
  (rho = 0.25–0.46, p from ~5e-09 to ~5e-03 across six tests)
- Mann-Whitney tests: isolated-6 vs gradient-12, wrong-18 vs certain-47,
  computed separately on mean_NLL and max_NLL (four group tests)

Multiple comparisons across ~10+ hypothesis tests inflate the family-wise
error rate; a nominal p = 0.02–0.05 should not be treated as strong evidence.

**Status: exploratory / hypothesis-generating, not confirmatory.** These
analyses were mined on a fixed 129-molecule test set after inspecting the
same data (quadrant definitions, group membership, and the NLL scoring model
were all chosen post-hoc). Any p-value here should be interpreted as a
screening signal, and headline claims need pre-registered replication on a
fresh split or dataset before being reported as confirmatory.

Known analysis choices that interact with the statistics:

- n_components=10 was selected by held-out LL on the *training* molecules
  (see Part 1) — the test-set statistics themselves are not selection-free
  w.r.t. the GMM scoring choice (n=10 is preferred but the pattern of
  correlations is robust across n=10 and n=50).
- Group definitions (isolated-6, gradient-12, wrong-18, certain-47) come
  from a neighbouring analysis (rmse_analysis) and were not independent of
  the test labels.
- Per-molecule NLLs are aggregated from atom-level NLLs; atoms within a
  molecule are correlated, so molecule-level tests treat each molecule as
  one sample (n = 12–47 per group), which is the conservative choice here.

See `gmm_uncertainty_check/validation/heldout_validation.py` for the
validation protocol.
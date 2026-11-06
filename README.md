# GIS-MCDA for humanitarian demining prioritization

This repository contains the synthetic dataset, code, and results supporting the revised manuscript. The dataset and simulations were generated for the revision; they do not reconstruct the original submission's experiment.

## Run

Python 3.10 or later with NumPy and pandas: `python experiment.py`. Install dependencies with `python -m pip install -r requirements.txt`. Outputs are written to `results/`. Exact repeatability requires the same Python and NumPy versions recorded in `results/metadata.json`.

## Data generation

All generator parameters and category counts are in `SPECS` in the script. Each continuous variable is an independently sampled normal value clipped to the listed lower and upper bounds (not a mathematically truncated normal). The integer population and incident counts are rounded. The five ordinal criteria are independently sampled from normal distributions with category-specific means and SD 0.7, rounded and clipped to 1–5. Infrastructure score C4 is criticality × exp(−distance/4000), rounded to four decimals. The dataset includes the generated distance and criticality values. These are illustrative assumptions, not observed mine-action data or expert assignments.

AHP ratios use the importance profile from the article, rounded to the nearest integer on the reciprocal Saaty scale, then the principal eigenvector. Entropy weights are recalculated after each perturbation. Constant criteria receive zero entropy divergence. Ranking ties are resolved by original row order.

The same perturbed matrix is used for all four methods in each run, enabling paired comparison. Each is compared against its own unperturbed ranking. Perturbations match the assumptions reported in the article; ±1 changes are equiprobable and clipped at ordinal boundaries. The paired bootstrap resamples simulation runs and estimates 95% percentile intervals for differences in mean robustness metrics. These intervals are conditional on this one synthetic dataset and its assumed perturbation model; they are not field-validation intervals. No significance claim is made about real-world performance.

Outputs include the dataset, AHP matrix, weights, full baseline scores and rankings, sensitivity analysis, each run's four robustness metrics, summary statistics, paired bootstrap intervals and environment metadata. They do not include 30 individual scores for every perturbed run; rerun the script to reproduce the reported run metrics.

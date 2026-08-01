# First-passage distribution interpretation

Real VGG/layer-to-time evidence percentiles used to calibrate synthetic magnitudes:

`p1=-1.115`, `p5=-0.882`, `p25=-0.580`, `p50=-0.327`, `p75=0.449`, `p95=1.683`, `p99=2.371`.

The distribution summary is saved in `05_first_passage_distribution_summary.csv`.

Key cautious interpretation:

- R5 final RT is right-skewed mostly after non-decision-time variability is added.
- The hard WW decision-time distribution is bounded below by 0 and often compressed near early simulation steps under the current deterministic evidence/readout settings.
- Approximate normality of `S_i(t)` at a fixed time does not imply normality of first-passage time. These are different random variables.
- The canonical DDM benchmark shows a first-passage distribution can be bounded and right-skewed without assuming normal RT.

Synthetic checks are summarized in `04_synthetic_accumulator_simulation_summary.csv`; they are diagnostic controls, not retrained model results.

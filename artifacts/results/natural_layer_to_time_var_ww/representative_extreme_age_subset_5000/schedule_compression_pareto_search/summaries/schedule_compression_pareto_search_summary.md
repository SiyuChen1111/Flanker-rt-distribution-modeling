# Schedule compression Pareto search summary

- Run mode: `coarse`.
- Schedule candidates tested: 108.
- Schedule × noise candidates tested: 2988.
- Pareto-optimal candidates found: 20.
- Best incongruent-repair candidate: `c0.40_ls-50_tw1.10_ep30 + sb0.0000_st0.0000_sg0.0000_gs0.03`.
- Best fast-error-preservation candidate: `c0.40_ls-10_tw1.10_ep50 + sb0.0005_st0.0120_sg0.0000_gs0.03`.
- Best balanced candidate: `c0.40_ls-50_tw1.10_ep30 + sb0.0020_st0.0120_sg0.0000_gs0.05`.

## Interpretation

- This search treats incongruent repair, congruent fast-error preservation, and RT/dynamics preservation as separate objectives rather than collapsing everything into one scalar target.
- A strong Pareto front means the remaining trade-off is real, not just a ranking artifact.
- Any age-specific noise improvement should be treated as exploratory unless it clearly generalizes across both groups and preserves the broader dynamics profile.

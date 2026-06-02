# Schedule compression Pareto search repaired summary

- Original coarse ranking size: 2988; repaired ranking size: 31.
- Original Pareto count: 20; repaired Pareto count: 12.
- Original best balanced candidate: c0.40_ls-50_tw1.10_ep30 + sb0.0020_st0.0120_sg0.0000_gs0.05
- Repaired best balanced candidate: none

## Core interpretation

- The coarse search conclusions are only trustworthy after trial-level, trajectory, flag, and RT-bin metrics are derived from the same reconstructed candidate outputs.
- If the repaired Pareto front still shows a broad trade-off, that trade-off is real rather than an artifact of missing fields or approximate scoring.
- Fine search should only proceed on repaired Pareto candidates and only if at least one candidate remains plausible on both incongruent repair and congruent fast-error preservation.

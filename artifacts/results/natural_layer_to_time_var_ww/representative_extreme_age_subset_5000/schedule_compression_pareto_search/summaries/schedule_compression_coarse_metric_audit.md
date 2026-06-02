# Schedule compression coarse metric audit

- `schedule_compression_top_candidates_trial_level.csv` is not true trial-level output; it contains summary rows.
- Stochastic candidates in the coarse summary are missing reconstructed target-recovery and trajectory metrics.
- `flag_no_congruent_fast_error` in the original ranking can be contaminated by NaN or condition-inappropriate rows.
- `flag_lost_conflict_dynamics` can be mis-triggered when stochastic rows do not carry `early_flanker_dominance`.
- `error_rate_by_rt_bin_rmse` in the original script is an overall-error shortcut, not a real RT-bin RMSE.
- The original Pareto front is therefore potentially affected by metric-construction artifacts and needs a repaired re-export.

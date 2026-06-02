# Mechanism redesign summary

- Run mode: small.
- Total candidates tested: 16.
- Candidate count by mechanism family: {"CA1_trialwise_conflict_adaptive_schedule": 2, "CA2_online_conflict_schedule_acceleration": 2, "CA3_uncertainty_adaptive_schedule": 2, "COMBO_CA_only": 1, "COMBO_CA_retuned_noise": 1, "COMBO_conflict_adaptive_schedule_plus_bounded_lapse": 1, "L1_bounded_lapse_only": 1, "L2_bounded_lapse_only": 1, "R0_original_time_gap": 1, "R1_best_incongruent_repair_global_schedule": 1, "R2_best_fast_error_global_schedule": 1, "R3_best_conflict_dynamics_reference": 1, "R4_near_balanced_tradeoff_reference": 1}.
- Lenient survivors: 0.
- Main survivors: 0.
- Strict survivors: 0.

## Required answers
1. Tested candidates: 16.
2. Per-family counts: {"CA1_trialwise_conflict_adaptive_schedule": 2, "CA2_online_conflict_schedule_acceleration": 2, "CA3_uncertainty_adaptive_schedule": 2, "COMBO_CA_only": 1, "COMBO_CA_retuned_noise": 1, "COMBO_conflict_adaptive_schedule_plus_bounded_lapse": 1, "L1_bounded_lapse_only": 1, "L2_bounded_lapse_only": 1, "R0_original_time_gap": 1, "R1_best_incongruent_repair_global_schedule": 1, "R2_best_fast_error_global_schedule": 1, "R3_best_conflict_dynamics_reference": 1, "R4_near_balanced_tradeoff_reference": 1}.
3. Conflict-adaptive schedule better than global schedule: partially / exploratory only.
4. Conflict-adaptive schedule preserves congruent uncertainty: yes.
5. Conflict-adaptive schedule repairs incongruent flanker over-selection: yes.
6. Bounded lapse restores older congruent errors: yes.
7. Bounded lapse breaks incongruent repair: yes.
8. CA + lapse better than CA only or lapse only: no clear advantage.
9. Survivors: lenient=0, main=0, strict=0.
10. Fine-search or formal-fitting seed exists: no.
11. Most recommended seed: `CA1_flanker_dominance_0_100_cl0.70_ch0.45_q0.50_t0.25_ls-50_tw0.90_ep30`.
12. Main failure reason: no_congruent_fast_error.
13. The current result remains a multiple-objective trade-off unless main survivors are nonzero.
14. Most natural mechanism to continue: `CA1_flanker_dominance_0_100_cl0.70_ch0.45_q0.50_t0.25_ls-50_tw0.90_ep30`.
15. Most patch-like mechanism: `COMBO_conflict_adaptive_schedule_plus_bounded_lapse_3`.
16. Formal conclusion: this round can support mechanism comparison and negative-result reporting, but not a final balanced model unless main survivors are nonzero.
17. Exploratory conclusion: any apparent improvement from bounded lapse remains exploratory because it is a rare downstream uncertainty component, not the main mechanism.
18. Best advisor figures: mechanism_redesign_ranking_overview, constraint_survival_by_mechanism_family, conflict_score_vs_compression, older_congruent_error_recovery, mechanism_tradeoff_summary_dashboard.
19. Next step: if main survivors remain zero, prefer a smaller mechanism search around the best conflict-adaptive family or package the current negative result rather than launch formal fitting.

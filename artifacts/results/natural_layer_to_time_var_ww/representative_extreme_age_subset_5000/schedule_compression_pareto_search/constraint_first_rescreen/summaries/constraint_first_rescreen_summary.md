# Constraint-first rescreen summary

## Formal results
- Repaired candidate pool: 31 model_config_id.
- Repaired Pareto candidates: 12.
- Lenient survivors: 0.
- Main survivors: 0.
- Strict survivors: 0.
- Fine-search seed under main constraints: no.
- Most common failure category: no_older_congruent_errors.
- Most informative one-constraint sensitivity result: Removing any single main constraint still leaves zero survivors, so the failure is a coupled trade-off rather than a one-threshold problem.

## Required answers
1. Pool size: 31.
2. Pareto count: 12.
3. Survivors: lenient=0, main=0, strict=0.
4. Candidate usable as fine-search seed: no.
5. Main limiting constraint category: no_older_congruent_errors; one-at-a-time sensitivity result: no_single_constraint_sufficient.
6. Older congruent errors / fast-error evidence are more central than early conflict dynamics in the failure categories, but the one-at-a-time sensitivity shows no single constraint relaxation is enough.
7. Models with young fast-error evidence but insufficient older evidence: 11.
8. Models that repair incongruent errors but wash out congruent errors: 20.
9. Models preserving early conflict dynamics while failing incongruent thresholds: 11.
10. Closest near-balanced candidate: `c0.40_ls-50_tw1.10_ep50__sb0.0010_st0.0120_sg0.0000_gs0.03`.
11. Near-balanced failures: main_older_congruent_error_rate_ge_0.002, main_older_congruent_fast_error_not_absent.
12. Fine search recommendation: do not enter fine search from this pool.
13. If no main survivor exists, next work should adjust the mechanism/objective to preserve older congruent errors and fast-error evaluability while keeping incongruent repair.
14. If exploratory fine search is still forced, start only around the lowest fail-count repaired Pareto candidates listed in the representative table.
15. Best advisor figures: constraint_survival_flow, main_constraint_sensitivity, representative_models_error_rate_by_condition, representative_models_fast_error, constraint_tradeoff_map.
16. Formal conclusion: the repaired pool has no final balanced model unless main survivors are nonzero.
17. Exploratory conclusion: representative models are useful only as trade-off examples.

## Representative models
- best_incongruent_repair: `c0.40_ls-50_tw1.10_ep30__sb0.0000_st0.0000_sg0.0000_gs0.03` - It repairs incongruent errors best, but this can wash out the congruent-error evidence.
- best_fast_error: `c0.40_ls-10_tw1.10_ep50__sb0.0005_st0.0120_sg0.0000_gs0.03` - It best preserves fast-error timing, but still fails other acceptability constraints.
- best_conflict_dynamics: `c0.70_ls-50_tw0.70_ep0__baseline` - It preserves conflict-like dynamics best, but its behavior remains too far from the target profile.
- best_near_balanced: `c0.40_ls-50_tw1.10_ep50__sb0.0010_st0.0120_sg0.0000_gs0.03` - It is the closest current trade-off candidate, but it still fails the main standard.
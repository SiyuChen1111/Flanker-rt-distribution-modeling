# R5 Supervisor Follow-Up Diagnostic

## Purpose

This diagnostic answers three supervisor-facing questions about the current VGG/layer-to-time → Wong-Wang → R5 readout model:

1. whether CAF and CRF should use actual RT coordinates rather than quantile indices;
2. whether `S(t)` or the R5 readout compresses the RT distribution;
3. why the model produces excessive incongruent errors.

The diagnostic reuses the retained R5 package and does not retrain the full VGG model.

## How to rerun

From the repository root:

```bash
python code/scripts/run_r5_supervisor_followup.py
python code/scripts/run_ww_diffdecision_core_audit.py --mode full
python code/scripts/run_r5_supervisor_round2.py
python code/scripts/run_real_vgg_target_flanker_dynamics_audit.py
python code/scripts/run_r5_choice_rule_alignment_audit.py
python code/scripts/run_r5_choice_coupled_schedule_optimization.py
python code/scripts/plot_r5_rt_distribution_kde.py
python code/scripts/plot_r5_caf_and_delta_curves.py
python code/scripts/run_all_age_group_extension.py
python code/scripts/run_corrected_model_all_age_groups.py
python code/scripts/run_all_age_time_scale_refinement.py
python code/scripts/plot_all_age_group_rt_distributions.py
```

The first script writes its outputs to:

`artifacts/results/r5_supervisor_followup/`

The additional scripts write new bundles rather than overwriting the original follow-up:

- `artifacts/results/ww_diffdecision_core_audit_20260802/`
- `artifacts/results/r5_supervisor_round2_20260802/`
- `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/`
- `artifacts/results/r5_choice_rule_alignment_audit_20260803/`
- `artifacts/results/r5_choice_coupled_refit_20260803/`
- `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`
- `artifacts/results/r5_rt_distribution_kde_20260803/`
- `artifacts/results/r5_caf_delta_curves_20260803/`
- `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/`

For the broader generated-results tree, use `artifacts/results/ARTIFACT_DOCS_INDEX.md` and `artifacts/results/artifact_docs_inventory.csv` before treating any older artifact Markdown as current evidence.

## Primary outputs

| File | Purpose |
|---|---|
| `01_reproducibility_and_active_config_audit.md` | R5 package, parameter, config, and code-path audit |
| `02_CAF_actual_RT_values.csv` | CAF recomputed from raw trial-level rows with actual RT-bin medians |
| `02_CAF_actual_RT.png` / `.pdf` | Supervisor-facing CAF figure using actual RT coordinates |
| `03_CRF_actual_RT_values.csv` | CRF recomputed from raw target/flanker/response labels |
| `03_CRF_validation_report.md` | CRF probability and target-accuracy checks |
| `03_CRF_actual_RT.png` / `.pdf` | CRF figure produced only after validation passes |
| `04_state_and_readout_implementation.md` | Wong-Wang and readout equation audit |
| `04_state_distribution_statistics.csv` | Fixed-time `S_i(t)` summary statistics |
| `04_synthetic_accumulator_simulation_summary.csv` | Controlled evidence-input sanity checks |
| `05_first_passage_distribution_summary.csv` | Human RT, R5 RT, WW decision time, t0, and DDM benchmark summaries |
| `06_incongruent_error_decomposition.csv` | Incongruent model-error source classification |
| `07_ranked_bottleneck_assessment.md` | Ranked hypothesis assessment |
| `10_supervisor_response_summary_chinese.md` | Short Chinese response organized around the supervisor's three questions |
| `11_supervisor_response_summary_english.md` | Short English response |

## Second-round supporting outputs

| File | Purpose |
|---|---|
| `artifacts/results/r5_supervisor_round2_20260802/01_CAF_explicit_quantile_RT_ticks.pdf` | Human and model CAF panels with their own actual median-RT ticks |
| `artifacts/results/r5_supervisor_round2_20260802/02_CRF_explicit_quantile_RT_ticks.pdf` | Human and model CRF panels with their own actual median-RT ticks |
| `artifacts/results/r5_supervisor_round2_20260802/03_time_scaling_preserves_shape.pdf` | Demonstrates shape-preserving multiplicative RT rescaling |
| `artifacts/results/r5_supervisor_round2_20260802/04_improved_model_speed_accuracy.pdf` | Compares static and target-recovery synthetic evidence schedules |
| `artifacts/results/r5_supervisor_round2_20260802/summary.md` | Interpretation and limits of the second-round checks |
| `artifacts/results/ww_diffdecision_core_audit_20260802/summary.md` | Isolated two-/four-choice core audit and no-crossing correction |
| `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/summary.md` | Separates real VGG layer evidence, scheduled WW input, state recovery, and RT/choice readout |
| `artifacts/results/r5_choice_rule_alignment_audit_20260803/summary.md` | Quantifies disagreement between whole-trajectory choice and the winner at the RT readout step |
| `artifacts/results/r5_choice_coupled_refit_20260803/summary.md` | Shows what happens when the corrected rule is imposed before retiming the evidence schedule |
| `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/summary.md` | Earlier two-group exploratory choice-coupled timing result |
| `artifacts/results/r5_rt_distribution_kde_20260803/observed_vs_model_rt_kde.pdf` | Human/model RT densities by congruency and age group |
| `artifacts/results/r5_caf_delta_curves_20260803/current_model_delta_rt_human_vs_model.pdf` | Correct-trial congruency RT cost across participant-level RT bins |
| `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_caf_updated_model.pdf` | Seven-group CAF small multiples from updated trial-level predictions |
| `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/figures_publication/all_age_rt_distribution_updated_model.pdf` | Seven-group human/model RT distributions |
| `artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/results/updated_model_parameters_by_age.csv` | Age-specific timing and fit diagnostics for the all-age update |

## Verified checks

- CAF uses median RT within each quantile bin, not relabelled bin numbers.
- Human and Model keep separate RT-bin coordinates.
- CRF is recomputed from trial-level target, flanker, and observed response labels.
- CRF row probabilities sum to 1 within numerical tolerance.
- Incongruent CRF `p_target` matches the corresponding incongruent CAF accuracy.
- Fixed-time `S_i(t)` distributions are analyzed separately from first-passage RT distributions.
- Never-crossed channels are no longer recorded as zero-time responses; the stored deadline value is a censoring sentinel and is separated by a crossing flag.
- Separate human/model panels make each distribution's actual RT-bin coordinates readable without overlapping tick labels.

## Current interpretation

The follow-up sequence now supports a more specific diagnosis. The retained baseline's RT and choice used different time windows: 26.5% of all trials receive different choices when the winner is evaluated at the stated RT step, and every disagreement is incongruent. Directly correcting that rule exposes a timing bottleneck because late target recovery arrives too close to the simulation deadline.

The earlier two-group exploratory solution keeps the real VGG evidence and retained Wong-Wang parameters, couples choice to RT, and compresses the evidence schedule. All 10,000 trials then cross, and overall accuracy, incongruent accuracy, mean RT, and incongruent CAF are close to the representative human subset. This shows that the existing VGG target-recovery signal is computationally sufficient after retiming; no additional handcrafted conflict-control module is required for this result.

The improvement is incomplete. KDE plots show model RT tails that are too short, and delta curves show an incongruent RT cost several times larger than the human effect. The current schedule was selected and assessed on the same trials, with only four older participants. The result is therefore an in-sample mechanism diagnostic, not a validated age model.

The seven-group update extends the same corrected choice/readout audit to `20-29` through `80-89`. It uses a shared decision-time scale of `0.27` and age-specific `t0`; the 70–79 group contains one no-crossing censored model trial. Its integrated CAF and RT figures are descriptive diagnostics, not evidence of a causal age mechanism.

## Important unresolved issue

The retained R5 package does not include a standalone neural-network checkpoint or complete active training-loss configuration. Therefore, training-objective conclusions remain unresolved until the exact active loss weights and checkpoint provenance are fully traced.

The illustrative RT scale factor in the second-round figure is not an estimated parameter. Likewise, the normalized four-choice competition ablation is not adopted because it improves crossing at the cost of target accuracy. The next priority is held-out, participant-balanced evaluation of the fixed choice-coupled procedure, including CAF, KDE/tails, delta curves, crossing coverage, and four-direction errors.

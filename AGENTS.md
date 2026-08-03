# VAM-studying Agent Guide

## Project Role

This is a research repository for LIM / Flanker reaction-time modeling. The current main line is a VGG/layer-to-time evidence pipeline feeding a four-choice Wong-Wang accumulator and R5 readout diagnostics for young and older participant groups.

## Current Result Spine

- Retained R5 baseline package: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- Current choice-coupled schedule result: `artifacts/results/r5_choice_coupled_schedule_optimization_20260803/`
- Choice/readout alignment audit: `artifacts/results/r5_choice_rule_alignment_audit_20260803/`
- Supervisor follow-up diagnostics: `artifacts/results/r5_supervisor_followup/`
- Second-round supervisor checks: `artifacts/results/r5_supervisor_round2_20260802/`
- Wong-Wang/DiffDecision core audit: `artifacts/results/ww_diffdecision_core_audit_20260802/`
- Real VGG target/flanker dynamics audit: `artifacts/results/r5_real_vgg_target_flanker_audit_20260803/`
- Current RT-distribution and CAF/delta figures: `artifacts/results/r5_rt_distribution_kde_20260803/`, `artifacts/results/r5_caf_delta_curves_20260803/`
- Artifact documentation index: `artifacts/results/ARTIFACT_DOCS_INDEX.md`
- Reproducible follow-up script: `code/scripts/run_r5_supervisor_followup.py`
- Core and second-round scripts: `code/scripts/run_ww_diffdecision_core_audit.py`, `code/scripts/run_r5_supervisor_round2.py`
- Real-evidence dynamics script: `code/scripts/run_real_vgg_target_flanker_dynamics_audit.py`
- Long-form follow-up note: `docs/r5-supervisor-followup.md`

## Common Commands

Run from the repository root unless a script says otherwise.

```bash
python code/scripts/run_r5_supervisor_followup.py
python code/scripts/run_ww_diffdecision_core_audit.py --mode full
python code/scripts/run_r5_supervisor_round2.py
python code/scripts/run_real_vgg_target_flanker_dynamics_audit.py
python code/scripts/run_r5_choice_rule_alignment_audit.py
python code/scripts/run_r5_choice_coupled_schedule_optimization.py
python code/scripts/plot_r5_rt_distribution_kde.py
python code/scripts/plot_r5_caf_and_delta_curves.py
pytest tests/test_r5_choice_rule_alignment_audit.py tests/test_r5_choice_coupled_schedule_optimization.py
```

## Working Rules

- Treat `best_model_R5_combined_best` as a diagnostic model-development package, not a final full-cohort fit.
- Distinguish the retained legacy baseline from the exploratory choice-coupled schedule result; neither is a final validated model.
- Do not overwrite existing result bundles. Put new supervisor or mechanism follow-ups in a new directory under `artifacts/results/`.
- Before using old generated Markdown under `artifacts/results/`, check `artifacts/results/ARTIFACT_DOCS_INDEX.md` for its category and trust level.
- Validate derived CAF/CRF tables against raw trial-level rows before plotting or interpreting them.
- Use actual RT coordinates for CAF/CRF figures; do not relabel quantile indices as RT.
- Keep random seeds fixed and recorded for simulations.
- Do not claim the accumulator works perfectly. Use cautious terms such as "passes a minimal sanity check", "partially supported", or "unresolved".
- Treat a no-crossing deadline value as a censoring sentinel; check the crossing flag and never interpret the sentinel as an observed RT.
- In current fits, choice must be the winner at the sustained-crossing readout step; whole-trajectory choice is legacy reproduction only.
- Build delta curves from correct trials within participant, then aggregate participants; report the small older-group sample explicitly.
- Keep user-facing summaries concise and in Chinese; keep technical work rigorous and verified.

## Key Code Paths

- Evidence cache and layer-to-time inputs: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`
- Wong-Wang state update and differentiable crossing: `code/scripts/vgg_wongwang_lim.py`
- R5 fitting and readout utilities: `code/scripts/run_representative_extreme_age_subset_fitting.py`, `code/scripts/optimize_natural_layer_to_time_rt_shape.py`
- R5 supervisor diagnostics: `code/scripts/run_r5_supervisor_followup.py`
- Choice-coupled timing and figures: `code/scripts/run_r5_choice_coupled_schedule_optimization.py`, `code/scripts/plot_r5_rt_distribution_kde.py`, `code/scripts/plot_r5_caf_and_delta_curves.py`

## Documentation Map

- `README.md` gives the human-facing overview and current reading order.
- `docs/model_framework_summary.md` explains the model flow.
- `docs/current_results_and_limitations.md` summarizes the current interpretation and limitations.
- `docs/r5-supervisor-followup.md` records the latest supervisor-question diagnostic and where to find each output.
- `docs/r5-supervisor-systematic-report-20260803.md` organizes the recent checks into a discussion-ready narrative.
- `artifacts/results/ARTIFACT_DOCS_INDEX.md` classifies generated result Markdown into current, supporting, exploratory, and legacy tiers.

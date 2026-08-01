# Artifact Results Documentation Index

This is the top-level navigation layer for Markdown documents under `artifacts/results/`. It classifies generated reports without rewriting their historical contents.

## Reading Order

1. Current R5 supervisor follow-up: `r5_supervisor_followup/10_supervisor_response_summary_chinese.md` and `r5_supervisor_followup/09_full_technical_report.md`.
2. Current retained R5 package: `natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/README_best_model_R5_combined_best.md`.
3. Current R5 result tables and figures inside `natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/results/` and `figures_publication/`.
4. Supporting representative-subset diagnostics under `natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`.
5. Historical or exploratory materials only when tracing provenance.

## Classification Summary

| Category | Count | Meaning |
|---|---:|---|
| `current_spine` | 19 | Current promoted or supervisor-facing R5 documentation. |
| `current_supporting` | 115 | Current representative-subset context that supports the R5 branch. |
| `exploratory_or_side_diagnostic` | 118 | Side diagnostics and ablations; useful context, not promoted final evidence. |
| `legacy_or_historical_reference` | 103 | Historical outputs retained for provenance; do not treat as current truth. |
| `unclassified_reference` | 4 | Older or root-level references that require reader caution. |

## Top-Level Folder Counts

| Folder | Markdown files |
|---|---:|
| `natural_layer_to_time_var_ww` | 124 |
| `diagnostics` | 101 |
| `organized` | 56 |
| `repro_legacy_interim` | 34 |
| `r5_supervisor_followup` | 10 |
| `premature_readout_evidence_schedule_optimization` | 8 |
| `flanker_suppression_full_ww_validation` | 7 |
| `rtshape_experiment` | 4 |
| `age_groups_interim` | 3 |
| `rt_model_dmc_var_ww` | 3 |
| `proposal_aligned_behavior` | 2 |
| `age_groups` | 1 |
| `age_groups_full_matched_compare` | 1 |
| `age_groups_response_supervision_frozen` | 1 |
| `age_groups_response_supervision_interim` | 1 |
| `.` | 1 |
| `model_aligned_20_29` | 1 |
| `rt_model_variational_ww_synthesis` | 1 |

## Duplicate-Suffix Copies

The following files look like duplicate Finder/script copies because their filenames contain ` 2.md`. They were not deleted because they may still be useful for provenance; treat them as cleanup candidates only after confirming contents.

- `repro_legacy_interim/README 2.md`
- `repro_legacy_interim/flanker_mechanism_evaluation_framework 2.md`
- `repro_legacy_interim/hybrid_legacy_parameter_notes 2.md`
- `repro_legacy_interim/legacy_reference_comparison 2.md`
- `repro_legacy_interim/postpatch_canonical_gate_clarification 2.md`
- `repro_legacy_interim/true_single_subject_feasibility/true_single_subject_feasibility_summary 2.md`
- `repro_legacy_interim/urgency_readout_decision_memo 2.md`
- `rtshape_experiment/rtshape_experiment_summary 2.md`
- `rtshape_experiment/rtshape_tail_loss_summary 2.md`

## Machine-Readable Inventory

See `artifact_docs_inventory.csv` for the complete list of Markdown files, category, trust level, title, size, and duplicate-suffix flag.

## Guardrails

- Do not use older diagnostic summaries as current R5 evidence unless they are explicitly linked by the current README or R5 package.
- Do not delete historical result docs without a separate provenance check.
- When a new result becomes current, update this index, `README.md`, and `docs/current_results_and_limitations.md` together.

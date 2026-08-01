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
```

The script writes all outputs to:

`artifacts/results/r5_supervisor_followup/`

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

## Verified checks

- CAF uses median RT within each quantile bin, not relabelled bin numbers.
- Human and Model keep separate RT-bin coordinates.
- CRF is recomputed from trial-level target, flanker, and observed response labels.
- CRF row probabilities sum to 1 within numerical tolerance.
- Incongruent CRF `p_target` matches the corresponding incongruent CAF accuracy.
- Fixed-time `S_i(t)` distributions are analyzed separately from first-passage RT distributions.

## Current interpretation

The current results suggest that the R5 final RT shape depends strongly on non-decision-time variability. The hard Wong-Wang first-crossing times are more compressed than final RT, so the main bottleneck is likely in the division of work among evidence scaling, readout threshold/margin, accumulator noise, and t0 variability.

The excessive incongruent-error pattern is not currently best explained by a global response-label mapping failure. The recomputed CRF passes the basic mapping checks, while the error decomposition points more toward evidence-origin and early readout/accumulation issues.

## Important unresolved issue

The retained R5 package does not include a standalone neural-network checkpoint or complete active training-loss configuration. Therefore, training-objective conclusions remain unresolved until the exact active loss weights and checkpoint provenance are fully traced.

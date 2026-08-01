# Current Results and Limitations

## Current main result

The current main line is the representative extreme-age R5 workflow:

- young group: `young_20_29`, 5,000 representative trials
- older group: `older_80_89`, 5,000 representative trials
- retained candidate: `R5_combined_best`
- main result bundle: `artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/best_model_R5_combined_best/`
- supervisor follow-up bundle: `artifacts/results/r5_supervisor_followup/`
- artifact documentation index: `artifacts/results/ARTIFACT_DOCS_INDEX.md`

This remains a model-development and diagnostic result, not a final full-cohort age-group fit.

## What R5 currently supports

- Mean RT is close to human behavior on the representative subset:
  - young 20-29: human/model mean RT is about 0.603 / 0.612 seconds
  - older 80-89: human/model mean RT is about 0.941 / 0.919 seconds
- The model keeps the layer-to-time visual-evidence route and group-specific readout/t0 settings.
- The supervisor follow-up recomputes CAF and CRF from raw trial-level rows. CAF and CRF now use actual median RT within each quantile bin rather than generic bin indices.
- CRF validation passes the basic checks: response proportions sum to 1, and incongruent `p_target` matches the corresponding CAF accuracy.

## Main limitations

- R5 still overestimates model accuracy and does not fully match human correct/error behavior.
- The model produces excessive incongruent errors, mostly tied to evidence/readout timing in the current diagnostic decomposition.
- The hard Wong-Wang first-crossing decision time is more compressed than the final RT distribution.
- The final RT shape is materially supported by group-specific non-decision-time variability, so t0 may be masking an insufficiently expressive decision-time process.
- No standalone R5 neural-network checkpoint or complete active training-loss configuration was found inside the retained R5 package. Training-objective conclusions remain unresolved.
- Many older Markdown reports remain under `artifacts/results/` for provenance. Use `artifacts/results/ARTIFACT_DOCS_INDEX.md` and `artifacts/results/artifact_docs_inventory.csv` to distinguish current evidence from historical or exploratory outputs.

## Interpretation guardrails

- Do not treat RT normality as the success target. Threshold-crossing RT is a first-passage-time variable and is expected to be bounded below and often right-skewed.
- Do not infer that approximate fixed-time normality of `S_i(t)` implies normal first-passage RT.
- Do not claim that the accumulator works perfectly. The current accumulator/readout path passes only targeted sanity checks and still has unresolved bottlenecks.

## Recommended next step

Run small, single-factor ablations rather than adding several new mechanisms at once:

1. restrict or remove t0 variability to measure how much RT fit depends on non-decision time;
2. test calibrated accumulator noise separately;
3. test threshold/margin/readout timing separately;
4. only revisit training-objective weights after a complete active-loss audit.

# R5 choice–RT coupling correction and representative refit

## What was corrected

Fresh model fits now choose the winning response at the same sustained-crossing step used for decision time. The historical whole-trajectory maximum remains available only as an explicit legacy option for reproducing retained R5 artifacts. The readout table now records the exact readout step and whether a real crossing occurred.

## Root cause

The mismatch entered in two stages. The April 2026 core helper defined the legacy choice from the maximum state over the complete simulated trajectory. In July 2026, the RT-shape wrapper added sustained-crossing RT rules but deliberately kept that legacy choice as its default; `winner_at_readout` existed only as an optional branch. The representative fitting script then called the wrapper without specifying a choice rule, so it optimized RT and choice from different time windows. Because accuracy and CAF contributed to model selection, post-RT target recovery improved the score and allowed the mismatch to survive.

This was not caused by the no-crossing sentinel or by RTify's threshold-crossing definition. It was introduced by the local integration of a legacy choice helper with a newer RT-only readout wrapper.

## Corrected representative refit

The corrected finite search used `winner_at_readout` for all 10,000 trials and imposed a minimum 95% real-crossing gate. Under that constraint it selected `R5_combined_best`.

| Group | Human accuracy | Model accuracy | Human incongruent error | Model incongruent error | CAF RMSE | Crossing rate |
|---|---:|---:|---:|---:|---:|---:|
| Older 80–89 | 0.976 | 0.687 | 0.039 | 0.629 | 0.303 | 0.951 |
| Young 20–29 | 0.949 | 0.633 | 0.083 | 0.734 | 0.333 | 0.975 |

The selected group parameters used evidence gain 0.60, threshold 0.12, and four sustained steps for older adults; young adults used evidence gain 0.80, threshold 0.14, and two sustained steps.

## Interpretation boundary

Without a crossing gate, the search can recover high accuracy and a low CAF error, but only by allowing roughly 36% of trials to fall on the deadline sentinel. Once the crossing gate is enforced, coverage improves to 95–97%, while incongruent errors become far too frequent. The current model family therefore cannot simultaneously reproduce behavior, couple RT and choice, and maintain acceptable crossing coverage within the tested grid.

This gate-passing candidate must not replace the retained R5 as a final behavioral model. The next optimization should decouple the duration of the VGG layer-to-time schedule from the total WW simulation window. The current schedule spreads the arrival of late target evidence across almost the entire 0.8-second window, leaving too little time for a theoretically coupled decision after target recovery.

# Epoch Or Parameter Sensitivity

- Checkpoint files found: no.
- No trainable epoch checkpoint found; ACC mismatch is more likely related to WW/readout/noise parameters than training epoch.

## Existing model comparison

- Best score model: `R5_combined_best` with total score 0.404.
- Worst score model: `R0_fixed_current` with total score 1.370.
- Accuracy score range across existing models: 0.134 to 0.136.
- Mean CAF RMSE range across existing models: 0.137 to 0.169.
- Error-minus-correct RT range across existing models: 0.005 to 0.017.
- If accuracy mismatch barely moves while RT-shape metrics move, the remaining issue is more consistent with the choice/readout mechanism than with undertraining.

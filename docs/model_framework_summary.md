# Model Framework Summary

## Modules

- Visual encoding
- Variational evidence sampling
- DMC-style conflict modulation
- Wong-Wang decision dynamics
- Response and RT readout

## Data flow

1. An image or trial is encoded into class evidence.
2. The evidence is sampled over time to create a noisy sequence.
3. The sequence is optionally transformed and resampled to the decision grid.
4. Conflict modulation changes the time course for congruent vs incongruent trials.
5. Wong-Wang dynamics integrate the input until a threshold is crossed.
6. The crossing time and trajectory are converted into response and RT.

## Key implementation points

- `stage1_semisup_evidence_sampler.py` creates deterministic, variational, or dropout-based evidence samples.
- `vgg_wongwang_lim.py` converts evidence into decision dynamics and readout.
- `train_variational_ww_smoke.py` bridges sampled evidence into Wong-Wang training and evaluation.
- `run_subject_level_dmc_var_ww.py` runs the subject-level DMC + variational workflow.
- `analyze_subject_level_dmc_var_ww.py` summarizes panel results.

## Main difference from the ideal target

The current implementation captures the mechanism of fast errors, but it is still a mechanism test rather than a final human RT fit. It does not fully match human reaction-time scale, tail shape, and choice consistency.

For the public release, the retained evidence is limited to the current DMC + variational evidence + Wong-Wang branch and its minimal supporting summaries.

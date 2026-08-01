# State and readout implementation audit

Code path checked:

- Visual evidence cache: `/Users/siyu/Documents/GitHub/VAM-studying/artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/evidence_cache/representative_subset_layerwise_evidence.npz`
- Layer-to-time mapping: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`, `natural_smooth_5stage` schedule.
- Wong-Wang update: `code/scripts/vgg_wongwang_lim.py`, `WongWangMultiClassDecision`.
- R5 readout wrapper: `code/scripts/optimize_natural_layer_to_time_rt_shape.py`, `apply_readout`.

Implemented update per channel:

`s(0)=0.1`.

At each 10 ms step, for trial `b`, channel `i`:

`I_i(t) = J_ext * evidence_i(t)` while stimulus is present.

`x_i(t) = sum_j S_j(t) * J_ji + I_0 + I_i(t) + I_noise_i(t)`.

`H_i(t) = relu((a*x_i(t)-b) / (1 - exp(-d*(a*x_i(t)-b)) + 1e-6))`.

`dS_i/dt = -S_i/tau_s + (1-S_i) * H_i * gamma / 1000`.

`S_i(t+dt) = S_i(t) + dS_i/dt * dt`.

Noise is an Ornstein-Uhlenbeck-like AMPA term, but R5 sets `noise_ampa=0.0` during reconstruction and output generation, so this diagnostic treats the real R5 accumulator as deterministic given the evidence.

Fixed WW constants in code: `a=270.0`, `b=108.0`, `d=0.1540`, `gamma=0.641`, `tau_s=100.0 ms`, `J_self=0.2609`, `J_cross=-0.0497`, `J_ext=0.0156`, `I_0=0.3255`, default `tau_ampa=2.0 ms`.

R5 group parameters:

- Young: `{'evidence_gain': 0.8, 'threshold': 0.12, 'sustained_k': 2, 'margin': 0.0, 'min_decision_time': 0.0, 't0_mean': 0.55, 't0_sd': 0.12}`
- Older: `{'evidence_gain': 0.8, 'threshold': 0.14, 'sustained_k': 2, 'margin': 0.02, 'min_decision_time': 0.0, 't0_mean': 0.75, 't0_sd': 0.2}`

Readout:

- Hard crossing uses winner state > threshold plus margin, sustained for `sustained_k=2` consecutive steps.
- If no sustained crossing occurs, the readout step falls back to the final simulation step.
- The final choice in this R5 path remains `trajectory_max_choice`: class with maximum over-time threshold-relative strength, not necessarily winner exactly at the sustained crossing step.
- Final RT = decision time + sampled/clipped t0. Decision time and t0 are seconds; WW dt is 10 ms; threshold-crossing index is a 0-based simulation step.

Cautious conclusion:

The implementation passes a minimal reproducibility sanity check against the saved R5 trial table, but the available R5 package is a finite diagnostic model package rather than a separately archived end-to-end training checkpoint.

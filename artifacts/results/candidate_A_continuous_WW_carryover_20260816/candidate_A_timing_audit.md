# Candidate A trial-boundary timing audit

## Outcome

**TIMING UNRESOLVED — CANDIDATE A NOT RUN.**

The repository does not contain enough information to determine the elapsed time from commitment/response on trial *n* to sensory-evidence onset on trial *n+1*. In particular, it does not contain stimulus-onset, response-end, feedback, trial-end, or next-stimulus-onset timestamps, and it does not contain the LIM task-presentation implementation. An ITI therefore cannot be reconstructed without introducing an unsupported timing assumption.

## Sources inspected

The following frozen/current sources were inspected directly:

- `configs/canonical_baseline_manifest.json`
- `code/scripts/vgg_wongwang_lim.py`, especially `WongWangMultiClassDecision.forward` and `.inference`
- `code/scripts/analyze_layerwise_evidence_ww.py`, especially `make_ww` and `run_ww`
- `code/scripts/run_c0_canonical_h1_h6_audit.py`, especially `simulate_group`
- `code/scripts/run_c0v2_canonical_h1_h6_audit.py`, especially the C0v2 replay path
- `code/scripts/run_c0v2_causal_commitment_audit.py`, especially `identity_gate`
- `code/scripts/run_human_signature_audit.py`, especially the required human-data fields and within-session adjacency definition
- `artifacts/results/human_condition_specific_error_transitions_20260815/human_condition_specific_transition_report.md`
- All 75 participant trial files matching `data/vam_data/user*df.csv`
- `data/vam_data/metadata.csv`
- The repository file inventory for task-presentation sources (`.js`, `.ts`, `.tsx`, `.jsx`, and `.html`) and timing-related fields/terms

## Timing represented in the frozen model

| Quantity | Repository value | Interpretation |
|---|---:|---|
| Accumulator step | 0.01 s | Numerical model step, not a measured task interval |
| Accumulator steps | 80 | One independently simulated model trial |
| Model time coordinates | 0.00–0.79 s | The 80 recorded samples; nominal integration duration is 0.80 s |
| Sensory-input duration in the replay | all 80 steps | `t_stimulus` is set equal to `time_steps`; this is a modeling schedule, not verified screen timing |
| Young nondecision-time mean | 0.4466653891 s | Additive RT component; no repository evidence identifies it as post-response/ITI time |
| Older nondecision-time mean | 0.6701965077 s | Additive RT component; no repository evidence identifies it as post-response/ITI time |
| Real post-response display | unknown | No timestamp or task-presentation source |
| Real feedback duration | unknown | No timestamp, flag, or task-presentation source |
| Real ITI | unknown | Fixed versus variable cannot be established |
| Commitment-to-next-stimulus interval | unknown | Cannot be calculated |

The C0v2 trajectory is integrated through the full 80-step model horizon even when commitment occurs earlier. Sensory evidence remains active throughout that horizon. Later states cannot change the already recorded choice or RT, but this post-commitment numerical suffix is not documented as the real post-response display or ITI.

## Human trial records

All 75 participant files have exactly the same ten columns:

`anon_id`, `nth_play`, `trial`, `xpos`, `ypos`, `flanker_direction`, `response_direction`, `response_time`, `stimulus_layout`, `target_direction`.

The only time-like field is `response_time`, which is a within-trial RT in milliseconds. There is no absolute timestamp or separate onset/end time from which the interval to the next trial can be derived. `nth_play` and `trial` establish session membership and ordinal adjacency, but not elapsed time. Consequently:

- fixed versus variable ITI is unresolved;
- participant- or trial-specific timing is unresolved;
- feedback/post-response display timing is unresolved;
- removed trials can be recognized as ordinal gaps, but their elapsed timing cannot be reconstructed.

The completed human transition audit correctly excludes session starts, cleaning gaps, and nonconsecutive trial numbers. That preserves behavioral adjacency but does not recover physical time between adjacent trials.

## Current C0v2 reset location

Each invocation of the Wong–Wang accumulator creates a fresh four-channel state with `S = 0.1` for every trial. The reset occurs inside both accumulator execution paths before the first numerical update. The canonical replay batches trials, but each batch row is initialized independently to the same state. Thus the effective reset is at the start of every modeled trial (the model's time zero / evidence-onset boundary), not at a recorded real task boundary.

The accumulator wrapper also sets `t_stimulus = time_steps`, so the canonical replay has no separately represented evidence-off, post-response, feedback, or ITI segment.

## Why the 0.80 s horizon and nondecision time cannot substitute for ITI

Neither value specifies the real time between a completed commitment and the next stimulus onset. Treating the unused portion of the 0.80 s numerical horizon, the additive nondecision-time term, or any chosen constant as an ITI would invent trial-boundary dynamics. It would also leave unresolved when current-trial visual evidence should be removed, which is required for both A1 and A2.

## Stop decision

The critical stop rule is met. No arbitrary ITI was introduced. Candidate A A0/A1/A2 was not implemented or simulated; no inhibition grid was declared; no behavioral targets were consulted for parameter selection; and no Stage 1 or Stage 2 evaluation was run.

**A-S0 — TIMING UNRESOLVED**

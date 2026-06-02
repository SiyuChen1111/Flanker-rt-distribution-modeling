# Natural evidence dynamics optimization summary

- Tested candidate models: 26.
- Best natural mechanism by combined score: `M1_schedule_compression` / `c0.4_ls-5_tw0.7_ep5`.
- Baseline combined score: 2.4551; best natural combined score: 0.8862.

## Interpretation

- This round keeps time+gap readout-choice uncertainty as the response-mapping mechanism and moves the optimization target upstream into evidence/input dynamics.
- Any candidate that reduces incongruent flanker over-selection by simply erasing early flanker dominance is treated as less natural, even if its accuracy improves.
- Formal fitting should only be considered after a natural dynamics family shows a credible joint improvement in incongruent error, congruent fast errors, and RT shape.

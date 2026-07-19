# WR2 uncertainty schedule fine search summary

- Run mode: full.
- Candidates tested: 240.
- Main survivors: 31.
- Strict survivors: 0.
- Candidates improving young incongruent error below 0.1966: 70.
- Improved main survivors: 26.
- Best score-only candidate: `WR2_fine_7ce51497`.
- Recommended candidate: `WR2_fine_3744359a`.

## Current seed reference

- Young incongruent error: 0.2038.
- Young congruent error: 0.0200.
- Older incongruent error: 0.0578.
- Pass main / strict: False / False.

## Recommended candidate metrics

- Young overall accuracy: 0.9049.
- Young congruent error rate: 0.0192.
- Young incongruent error rate: 0.1709.
- Young congruent error RT minus correct RT: -0.0043.
- Older overall accuracy: 0.9697.
- Older congruent error rate: 0.0064.
- Older incongruent error rate: 0.0542.
- Older congruent error RT minus correct RT: -0.0358.

## Interpretation

- 找到优于当前 seed 的 main survivor，但仍未完全解决 strict 层面的匹配。
- Negative error-minus-correct RT is treated as a plausible fast-error / premature readout signature, not as an automatic failure.
- This search keeps the Word-compatible backbone: no VGG retraining, no rhythmic attention branch, no lapse-based explanation.

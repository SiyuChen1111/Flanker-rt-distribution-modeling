# C0v2 commitment diagnostic

All behavioral labels use the completed commitment event. Post-commitment states never alter recorded choice or RT.

## Congruent trials

- Any wrong leader before commitment: 0.0000
- First meaningful leader wrong: 0.0000
- Wrong-state peak: 0.0965
- Median distance to threshold: 0.0428
- Target-minus-wrong margin at commitment: 0.0605
- Diagnosis: **CASE A**

## Incongruent trials

- Early wrong leader: 0.9689
- `pC_pre`: 0.9381
- Wrong commitment: 0.0599
- Mean commitment time: 0.3252 s
- Post-commitment internal recovery after wrong commitment: 1.0000
- Mean recovery delay: 0.1721 s

## Fast errors and corrected trials

FAST ERROR is an incongruent error at or below the fixed-subset median incongruent RT; FAST CORRECT and SLOW CORRECT use the same median split. This definition is reproducible and does not use trajectory outcomes to set the boundary.

| Group | Commitment time | Wrong peak | Target-wrong margin |
|---|---:|---:|---:|
| FAST ERROR | 0.1221 | 0.1255 | -0.0293 |
| FAST CORRECT | 0.2732 | 0.1057 | 0.0365 |
| SLOW CORRECT | 0.3971 | 0.1120 | 0.0546 |

Associations in the companion CSV are descriptive correlations, not causal effects.

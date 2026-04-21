# Mechanism extension decision memo

## Bottom line

No — adding this minimal DMC-like automatic/control extension does **not** improve the primary flanker-specific diagnostics enough to justify promotion.

## Phase 1 comparison

- Urgency baseline earliest incongruent CAF accuracy: `0.9837`
- Dynamic-selection earliest incongruent CAF accuracy: `0.9834`
- Dynamic-selection + DMC-like extension earliest incongruent CAF accuracy: `0.9967`
- Human earliest incongruent CAF accuracy: `0.7651`

- Urgency baseline first delta quantile: `-0.1881` s
- Dynamic-selection first delta quantile: `-0.0183` s
- Dynamic-selection + DMC-like extension first delta quantile: `0.1199` s
- Human first delta quantile: `0.0171` s

- Urgency baseline incongruent error-minus-correct RT: `-0.3818` s
- Dynamic-selection incongruent error-minus-correct RT: `-0.7086` s
- Dynamic-selection + DMC-like extension incongruent error-minus-correct RT: `-0.5749` s
- Human incongruent error-minus-correct RT: `-0.0999` s

## Mechanism-oriented reading

- Preserved gains:
  - no ceiling artifact returned (`frac_at_ceiling = 0.0000`)
  - congruency RT gap stayed in the correct direction (`0.1357`)
  - incongruent conditional error RT became less pathological than phase 1 (`-0.5749` vs `-0.7086`)
- Remaining / worsened failures:
  - earliest incongruent CAF moved **farther** from human, not closer (`0.9967` vs phase-1 `0.9834`, human `0.7651`)
  - early delta overshot human strongly (`0.1199` vs human `0.0171`), so delta realism regressed relative to phase 1
  - conditional tail structure for incongruent errors is unstable (`n=2` errors; `q95=0.5520` vs human `0.7620`)
- Extension selection mode: `dynamic_flanker_dmc_like`
- Extension q95 / q99: `1.5700` / `1.7300`

## Decision logic

This branch fails the stated stop condition for the DMC-like extension cycle:

- **CAF did not improve at all** on the primary target signature.

Although the extension partially improved incongruent conditional error RT and preserved the correct congruency-gap direction, it did so while making the earliest incongruent CAF even more ceiling-like and pushing the early delta well past the human regime.

## Recommendation

Do **not** promote this DMC-like extension as the new active branch.

Keep `dynamic_selection_phase1` as the active branch and stop tuning this exact extension form. The next mechanism attempt should target early automatic capture more directly without producing a broad late incongruent slowing / overcorrection pattern.

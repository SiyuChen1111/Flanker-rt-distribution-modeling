# Mechanism pivot decision memo

## Bottom line

Yes — the minimal dynamic-selection mechanism improves flanker-specific diagnostics over the current urgency baseline in smoke testing.

## Phase 1 comparison

- Urgency baseline earliest incongruent CAF accuracy: `0.9837`
- Dynamic-selection earliest incongruent CAF accuracy: `0.9834`
- Human earliest incongruent CAF accuracy: `0.7651`

- Urgency baseline first delta quantile: `-0.1881` s
- Dynamic-selection first delta quantile: `-0.0183` s
- Human first delta quantile: `0.0171` s

- Urgency baseline incongruent error-minus-correct RT: `-0.3818` s
- Dynamic-selection incongruent error-minus-correct RT: `-0.7086` s
- Human incongruent error-minus-correct RT: `-0.0999` s

## Mechanism-oriented reading

- Improvements observed: `['earliest incongruent CAF accuracy moved closer to human', 'early delta-plot direction/magnitude moved closer to human']`
- Dynamic-selection ceiling fraction: `0.0000`
- Dynamic-selection q95 / q99: `1.6200` / `1.7300`
- Dynamic-selection selection mode: `dynamic_flanker_suppression`

## Recommendation

Promote dynamic attentional selection to the next stage before attempting a DMC-like prototype.

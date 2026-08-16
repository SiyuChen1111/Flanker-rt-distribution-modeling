# Candidate B final report

## Primary classification

**B-S1 — STARTING-STATE LOCUS FAILURE**

## Frozen parent and sequence validity

C0v2 parent files remained byte-for-byte unchanged: **True**. Candidate B uses 12 age-stratified participants and 3,072 trials in exact human chronological order. State resets at participant, session, and every cleaning-induced nonconsecutive gap. Model history uses only the model's own prior choice/outcome; no human response, correctness, RT, or future trial updates the state.

## Human directional audit

**H-D2 — MIXED DIRECTIONAL STRUCTURE**. The strongest alignment contrast was after a previous error: **−2.63 pp** (nonmatch minus match), meaning that matching was associated with 2.63 pp higher current error risk—the opposite of the hypothesized protective pattern. This remained positive on the log-odds scale after participant, congruency, previous-error, and target-repetition controls. Among human errors, the previous response repeated on **28.79%**, versus conditional chance **24.31%**.

## Predeclared mechanism

Amplitudes were 0%, 10%, 25%, and 50% of each age group's neutral-to-threshold distance. Sign families were choice repetition (+/+), win-stay/lose-shift (+/−), and error-only carryover (0/+). Beta was 0 for B1 and 0.25, 0.50, 0.75 for B2. No behavioral target was used to choose this grid.

## Stage results

- Dynamically valid: 35/37 conditions.
- Conditions with any congruent errors: 0.
- Stage-1 survivors preserving the declared H3 guardrails: **None**.
- The two invalid settings were the 50%-amplitude, beta=0.75 choice-repetition and win-stay/lose-shift conditions; they started above threshold on 2.47% and 1.46% of trials, respectively.
- B0 exactly reproduced the frozen implementation on all 3,072 trials: choice and commitment-step agreement were both 100% in every age group.
- Because no setting produced congruent errors, the formal stop rule ended the experiment at Stage 1. Later transition and lag tables are descriptive only; there are no surviving B1/B2 conditions and no confirmatory Stage 2/3 or full H1–H6 evaluation.
- No identical rendered stimulus recurred in this small chronological subset, so same-stimulus response inconsistency is unestimable rather than zero.
- In B0 incongruent trials, model errors committed 194.9 ms faster than correct responses; congruent error RT is undefined because there were no congruent errors.

This result tests, but does not establish, the causal claim that an error creates the next vulnerable state. Human sequential association alone is not causal evidence. Candidate C was not implemented.

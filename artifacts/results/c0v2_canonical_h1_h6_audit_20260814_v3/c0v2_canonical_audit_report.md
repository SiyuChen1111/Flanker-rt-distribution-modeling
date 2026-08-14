# C0v2 canonical Human H1-H6 and trajectory audit

## Exact identity

`C0v2_causal_commitment_baseline` passed the strict identity gate on 10,000 fixed core trials. Zero-based `commitment_step = window_start_step + sustained_k - 1`; choice is the winner at that completed event and decision time is `commitment_step × 0.01 s`. Post-commitment mutation and prefix-only replay both pass on every trial. No whole-trajectory choice rule entered this audit.

## Human-C0v2 scorecard

| Signature   | Full Human                                                     | Matched Human                                                  | C0v2                                                                     | Distance                                    | Objective             | Status        |
|:------------|:---------------------------------------------------------------|:---------------------------------------------------------------|:-------------------------------------------------------------------------|:--------------------------------------------|:----------------------|:--------------|
| H1          | participant mean error=0.019; repeat inconsistency=0.057       | error=0.0131; repeat=NOT ESTIMABLE (1 pairs)                   | error=0.0000; repeat=NOT ESTIMABLE (1 pairs)                             | error difference=0.0131                     | REPAIR / MATCH        | FAIL          |
| H2          | RT=0.075 s; accuracy=0.057                                     | RT=0.0648s; accuracy=0.0571                                    | RT=0.2514s; accuracy=0.0652                                              | RT=0.1867s; accuracy=0.0081                 | PRESERVE              | PARTIAL       |
| H3          | slope=0.449; slow-fast=0.160                                   | fast=0.8423; slow=0.9777; slow-fast=0.1353; slope=0.3915       | fast=0.8324; slow=0.9975; slow-fast=0.1651; slope=0.3545                 | curve RMSE=0.0123; slope difference=0.0370  | IMPROVE / PRIMARY     | PARTIAL       |
| H4          | slope=0.192; late-early=0.056 s                                | slope=0.1816; late-early=0.0571s                               | slope=0.2002; late-early=0.0811s                                         | curve RMSE=0.1870s; slope difference=0.0185 | PRESERVE              | FAIL          |
| H5          | skew congruent=1.590; incongruent=1.089                        | SD=0.142/0.163s; skew=2.759/2.626                              | SD=0.135/0.163s; skew=-0.084/0.088                                       | Wasserstein=0.1226/0.1069s                  | QUANTIFY / IMPROVE    | FAIL          |
| H6          | congruent=-0.022 s; incongruent=-0.091 s; interaction=-0.069 s | congruent=-0.1608s; incongruent=-0.1808s; interaction=-0.0200s | congruent=NOT ESTIMABLE; incongruent=-0.2383s; interaction=NOT ESTIMABLE | incongruent difference=0.0575s              | POTENTIAL IMPROVEMENT | NOT ESTIMABLE |

No arbitrary numerical pass tolerances were introduced; statuses use the frozen empirical directions and whether the requested metric is estimable.

## What C0v2 already does well

- It preserves the primary H3 fast-error direction and substantial pre-commitment correction (`pC_pre=0.938`).
- It retains overall congruency costs and strong age-related mean-RT/accuracy patterns across seven groups.
- Behavioral choice and RT have a single causal commitment definition.

## What C0v2 still fails to explain

- H1: congruent error rate is 0.0000, versus 0.0131 in the matched human subset.
- H4 remains quantitatively mismatched despite retaining a structured delta curve.
- H5 is not uniformly narrower by every width metric. Participant-mean SD is 0.135/0.163 s versus human 0.142/0.163 s: C0v2 is 5.1% narrower for congruent trials and 0.4% wider for incongruent trials. Its near-zero skew nevertheless misses the strong human long-tail shape.
- H6 congruent and interaction terms are not estimable because C0v2 has no congruent errors.

## Mechanistic diagnosis

Congruent trials are **CASE A**: wrong channels do not become meaningfully competitive before commitment. Fast errors differ from corrected captures mainly by earlier wrong commitment, a stronger wrong state, and a negative target-minus-wrong margin. Later target dominance after a wrong commitment is reported only as **POST-COMMITMENT INTERNAL RECOVERY**, never behavioral correction.

## Age-group preservation

All seven manifest groups contribute 5,000 selected trials; one no-crossing trial in 70-79 is censored, leaving 7 model age-summary rows. The age CSV contains human/model mean RT, accuracy, H1-H6 metrics where estimable, CAFs, and distances. This remains descriptive in-sample validation, not a refit or held-out result.

## Recommended first M1 experiment — not implemented

The first single-factor candidate is low-amplitude sensory/evidence variability. The strongest rationale is CASE A plus the complete absence of congruent errors and the severe tail-shape mismatch; the audit does **not** claim uniform RT-width compression. Starting-state variability is less directly indicated because wrong states are not already entering congruent competition; recurrent/commitment changes are also premature because correction is already a major C0v2 strength.

Risks are excess incongruent errors, loss of the H3 CAF shape, distortion of age RTs, or invalid crossings. Guardrails are: preserve H3 as primary; protect age mean RT, overall accuracy, H2, H4, H5, crossing validity, and exact commitment semantics; rerun all H1-H6 after one change and retain/reject without a weighted composite score.

## Outputs and invariants

The result directory contains the required scorecards, trial and summary trajectory tables, association table, eight requested figure families, QA, and this report. C0v2 was not modified, no parameter was fitted or optimized, no noise was added in this audit, and Model M1 was not created.

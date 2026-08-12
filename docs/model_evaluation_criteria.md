# Human Signatures and Evaluation Framework for the Dynamic Flanker Model

**Evaluation Framework v1.0 — FROZEN FOR POST-M0 MODEL DEVELOPMENT**

This document is the baseline evaluation framework to use before further model optimization. It defines behavioral targets from human Lost in Migration (LIM) data, not from features of the current model. It is an evaluation plan and status audit, not a claim that every candidate signature has been confirmed or reproduced.

Evaluation Framework v1.0 is prospective with respect to all model modifications after the frozen M0 baseline. Freezing v1.0 fixes the evaluation principles and workflow; the human-only audit will subsequently freeze the exact quantitative targets and pass metrics. M0 has already been inspected and therefore serves as the baseline audit model. Exact targets and metrics must be frozen before any post-M0 model variant is inspected.

## 1. Evaluation Philosophy

Human behavior, rather than model architecture, defines the primary evaluation criteria. Following the methodological logic of Rafiei et al. (2024), evaluation should proceed in this order:

1. establish the empirical or theoretical basis for a candidate human signature;
2. determine whether that signature is present in the current human LIM data;
3. define a quantitative operationalization before optimizing the model against it;
4. evaluate Jaffe et al. (2025)'s VAM and the current model with the same metric;
5. compare qualitative reproduction and quantitative correspondence; and
6. only then test which computational mechanism explains success or failure.

Rafiei et al. first identified established properties of human perceptual decisions, confirmed them in their digit-discrimination task, and then compared neural-network models. The present project adopts that reasoning order, not their task-specific list of signatures. LIM is a four-choice conflict task and therefore requires its own human behavioral targets.

VGG, temporal evidence mapping, Wong-Wang recurrence, and noise are model components, not evaluation criteria. They become scientifically relevant only if they explain a confirmed human signature and survive a matched mechanistic test.

The required logic is:

> Human signature -> confirmation in human LIM data -> matched model comparison -> mechanistic explanation -> ablation -> generalization

## 2. Central Scientific Question

> How do dynamic transformations of target and flanker information across the visual hierarchy interact with recurrent competitive decision dynamics to generate the temporal signatures of human conflict decisions?

The conceptual comparison is:

- **Jaffe et al. (2025) / VAM:** primarily explains what determines the magnitude of distractor interference.
- **Current project:** asks when distractor interference dominates the decision and when later target information can overcome or correct it.

This is the question being tested, not an established conclusion. The current scope is Flanker-specific. Cross-task claims belong to a later stage.

## 3. Core Human Behavioral Signatures

H1-H6 answer the foundational question: what behavioral phenomena should a valid human-like Flanker decision model reproduce? Age, participant, and stimulus prediction remain scientifically important, but they are evaluated separately as stronger predictive and individual-difference targets.

The status labels in this section distinguish evidence about people from evidence about models:

- Human data: `CONFIRMED IN HUMAN DATA`, `LIKELY / NEEDS FORMAL CONFIRMATION`, or `NOT YET TESTED IN HUMAN DATA`.
- Models: `MODEL PASS`, `MODEL PARTIAL`, `MODEL FAIL`, or `NOT YET TESTED`.

### Signature Hierarchy

The six core signatures fall into two conceptual groups without changing their established H1-H6 numbering or analysis order.

| Conceptual group | Signatures | Scope |
|---|---|---|
| General human decision signatures | H1, H5 | Trial-to-trial variability, non-deterministic responses, and RT-distribution shape that are not unique to conflict tasks |
| Flanker / conflict-specific signatures | H2, H3, H4, H6 | Aggregate conflict costs, conditional accuracy dynamics, RT-dependent conflict costs, and condition-dependent correct-error timing |

The exact LIM targets for both groups must be established from the human-only audit rather than assumed from a model, a generic parametric family, or another task.

### H1. Non-deterministic Decision Behavior

**1. Established human phenomenon.** Human decisions are not perfectly deterministic, including under relatively easy or congruent conditions.

**2. Prior evidence.** Rafiei et al. quantified stochasticity through different responses to repeated presentations of the same image. Rare errors establish non-deterministic performance more weakly and are not equivalent to that repeated-stimulus measure.

**3. Manifestation in human LIM data.** Human congruent accuracy is near ceiling but below 100%. The repository has not yet established whether identical stimuli repeat within participant often enough for a direct stochasticity analysis.

**4. Operational definition.** The current minimum target is participant-level congruent error rate. If repeat structure is verified, the preferred stronger target is `P(response_1 != response_2 | same stimulus)` or an equivalent response-consistency measure.

**5. Required human-only analysis.** Estimate congruent error rates with uncertainty, then audit stimulus identifiers and within-participant repeat counts before deciding whether repeated-stimulus stochasticity is estimable.

**6. Human status.** `CONFIRMED IN HUMAN DATA` for non-ceiling congruent accuracy; `NOT YET TESTED IN HUMAN DATA` for repeated-stimulus stochasticity.

**7. VAM status.** `MODEL PASS` for the weaker aggregate non-ceiling target; `NOT YET TESTED` for repeated-stimulus consistency in this framework.

**8. M0 status.** `MODEL FAIL` for the weaker target because the promoted seven-group diagnostic produces no congruent errors.

**9. Final quantitative metric.** Congruent error-rate difference with participant uncertainty; response inconsistency rate if repeat analysis is feasible.

**10. Planned figure.** Human-only congruent error rates by participant and age group, followed by a Human/VAM/M0 comparison; add a repeat-consistency panel only if supported by the trial structure.

#### Candidate extension: Trial-history / Serial-dependence Effects

Serial dependence is a candidate human diagnostic, not a new confirmed core signature. A human-only analysis should test whether current-trial RT or error probability depends on previous-trial accuracy, previous-trial congruency, previous response, response repetition versus alternation, previous target/flanker relation, previous-trial RT, or interactions between previous and current congruency. These analyses may reveal sequential conflict adaptation, post-error effects, or more general serial dependence, but the framework does not assume that any one effect explains congruent errors.

The required logic is:

> human trial-history effect -> confirm in LIM -> only then test whether the model should reproduce it

If confirmed, serial dependence becomes one candidate explanation under M3 for the non-deterministic behavior captured by H1. It remains exploratory until replicated and quantified in the human LIM data.

H2-H4 capture complementary levels of conflict-related behavior rather than redundant versions of the same effect. H2 measures the overall aggregate congruency cost, H3 measures the within-incongruent conditional-accuracy dynamics that underlie fast errors, and H4 measures how the RT congruency cost evolves across the RT distribution.

| Signature | Main question |
|---|---|
| H2 | Does conflict impair performance overall? |
| H3 | When does conflict produce errors? |
| H4 | How does congruency-related slowing evolve across RT? |

### H2. Overall Congruency Cost

**1. Established human phenomenon.** This signature asks whether conflict impairs performance overall. At the aggregate or mean level, incongruent trials are slower and less accurate than congruent trials.

**2. Prior evidence.** RT and accuracy congruency effects are foundational findings in Flanker and related conflict tasks, and Jaffe et al. evaluated both in LIM.

**3. Manifestation in human LIM data.** Both effects are present in the published LIM analysis and in the current representative human summaries.

**4. Operational definition.** Compute the overall RT congruency cost, `RT_incongruent - RT_congruent`, on correct trials and the overall accuracy congruency cost, `Accuracy_congruent - Accuracy_incongruent`, both overall and by participant. H2 is a mean-level signature and does not depend on RT bins or distributional dynamics.

**5. Required human-only analysis.** Report participant-level distributions and uncertainty for the core H2 target. Age-stratified differences may be described at this stage, but they become evaluation targets only under P1.

**6. Human status.** `CONFIRMED IN HUMAN DATA`.

**7. VAM status.** `MODEL PASS` for both published effects.

**8. M0 status.** `MODEL PARTIAL`: both overall congruency effects are present in the expected direction on representative subsets, but their magnitudes have not yet been established under matched quantitative evaluation.

**9. Final quantitative metric.** Effect-size difference, MAE, or RMSE for both RT and accuracy congruency effects.

**10. Planned figure.** Human-only participant effect distributions, followed by matched Human/VAM/M0 effect estimates with uncertainty.

### H3. Fast-Error / Conditional Accuracy Signature

**1. Established human phenomenon.** This signature asks when conflict produces errors. Within incongruent trials, fast responses are disproportionately error-prone, and accuracy tends to recover across longer RTs. H3 is therefore a within-incongruent conditional-accuracy phenomenon, not an overall congruency effect. This is the **PRIMARY BEHAVIORAL BENCHMARK**.

**2. Prior evidence.** Jaffe et al. confirmed this pattern in LIM and reported that VAM produced the opposite conditional accuracy function (CAF) trend.

**3. Manifestation in human LIM data.** The published LIM result and current human CAF diagnostics show lower accuracy for faster incongruent responses.

**4. Operational definition.** Construct the incongruent conditional accuracy function (CAF) on actual RT coordinates and quantify fastest-quintile accuracy, the slowest-minus-fastest accuracy contrast, CAF slope, and full curve shape or error.

**5. Required human-only analysis.** Recompute participant-level CAFs from raw trials, validate derived tables, preserve actual RT coordinates, and report participant bootstrap uncertainty. The qualitative direction is already confirmed; this audit freezes magnitude, uncertainty, and pass thresholds.

**6. Human status.** `CONFIRMED IN HUMAN DATA`.

**7. VAM status.** `MODEL FAIL` qualitatively because it shows the opposite trend.

**8. M0 status.** `MODEL PARTIAL`: it recovers the human direction on representative diagnostic subsets but has not passed independent evaluation.

**9. Final quantitative metric.** CAF slope difference, fastest-to-slowest bin contrast, and full-curve RMSE.

**10. Planned figure.** Human-only participant CAF summary, followed by matched Human/VAM/M0 CAF panels and held-out quantitative error.

### H4. RT-Dependent Congruency Cost

**1. Established human phenomenon.** This signature asks how congruency-related slowing evolves across the RT distribution. Unlike the aggregate H2 effect, H4 is distribution-dependent: the RT congruency cost can change as responses become slower.

**2. Prior evidence.** Delta plots are established analyses of conflict-task dynamics. Jaffe et al. reported an increasing human LIM delta pattern that VAM reproduced reasonably well.

**3. Manifestation in human LIM data.** The increasing delta pattern is present in the published LIM result and current human diagnostics.

**4. Operational definition.** Build participant-level delta plots from correct trials, bin congruent and incongruent RTs separately, and express the congruency cost as a function of actual RT coordinates.

**5. Required human-only analysis.** Confirm slope magnitude and uncertainty across participants before freezing tolerances. The qualitative direction is already confirmed. Age-stratified estimates may be described here, but their predictive evaluation belongs to P1.

**6. Human status.** `CONFIRMED IN HUMAN DATA`.

**7. VAM status.** `MODEL PASS` for the published qualitative pattern.

**8. M0 status.** `MODEL PARTIAL`: it produces the direction but overstates the distributional RT congruency cost.

**9. Final quantitative metric.** Delta-curve slope difference, binwise effect error, and full-curve RMSE.

**10. Planned figure.** Human-only participant delta summary, followed by matched Human/VAM/M0 delta curves on actual RT coordinates.

### H5. RT-Distribution Shape

**1. Established human phenomenon.** Human RT distributions are typically non-Gaussian and often positively skewed, but the exact skewness and slow-tail structure in LIM must be quantified in the human-only audit.

**2. Prior evidence.** Distributional shape is a standard property of human decision times. Published LIM figures and current diagnostics indicate broader human than M0 distributions.

**3. Manifestation in human LIM data.** Existing LIM summaries motivate a formal distribution audit, but they do not yet establish that every condition, accuracy class, or age group has the same skewness or slow-tail structure.

**4. Operational definition.** Report mean, median, SD, skewness, q10/q25/q50/q75/q90/q95, tail ratios, quantile error, and Wasserstein distance by relevant condition.

**5. Required human-only analysis.** Produce a human-only distribution audit before fixing condition directions, age patterns, thresholds, or acceptable errors.

**6. Human status.** `LIKELY / NEEDS FORMAL CONFIRMATION` in the current analysis pipeline.

**7. VAM status.** `MODEL PARTIAL`: it captured important distributional properties, but no matched H5 score is currently available.

**8. M0 status.** `MODEL FAIL` for the slow-tail component and `MODEL PARTIAL` overall because promoted RT distributions are too narrow.

**9. Final quantitative metric.** Quantile error, tail-ratio error, skewness difference, and Wasserstein distance.

**10. Planned figure.** Human-only quantile and tail panels, followed by condition-specific Human/VAM/M0 densities and quantile-error plots.

### H6. Correct-Error RT Relationship

**1. Established human phenomenon.** Correct and error decisions exhibit a systematic RT relationship, but its direction is task- and condition-dependent.

**2. Prior evidence.** Jaffe et al. reported qualitatively that LIM incongruent error RTs are typically faster than congruent error RTs and correct-trial RTs. The exact correct-trial comparison must be defined in the local audit. Rafiei et al.'s digit task showed a different relationship, so its direction cannot be copied as a universal criterion.

**3. Manifestation in human LIM data.** Published LIM evidence supports a fast incongruent-error pattern, but its magnitude and condition-by-accuracy structure have not been frozen in the exact current analysis sample.

**4. Operational definition.** Compare incongruent error versus incongruent correct RT, congruent error versus congruent correct RT, and the condition-by-accuracy interaction at participant and distribution levels.

**5. Required human-only analysis.** Verify direction, magnitude, inclusion rules, and uncertainty in the current LIM sample before setting a pass criterion; handle sparse congruent errors explicitly.

**6. Human status.** `LIKELY / NEEDS FORMAL CONFIRMATION` locally.

**7. VAM status.** `MODEL FAIL` for the published dynamic error pattern.

**8. M0 status.** `NOT YET TESTED` under a complete matched correct-error analysis.

**9. Final quantitative metric.** Participant-level correct-error RT contrasts, interaction effect difference, and distributional distance.

**10. Planned figure.** Human-only correct/error RT distributions by congruency, followed by matched Human/VAM/M0 contrasts.

## 4. Extended Predictive and Individual-Difference Targets

P1 and P2 test whether a model explains systematic variation beyond the core H1-H6 phenomena. They are not less scientifically valuable; they represent a stronger level of predictive and individual-difference validation.

### P1. Age-Related Behavioral Variation

Human mean RT slowing with age is confirmed, while age patterns across the full H1-H6 set still need formal human-only analysis. Quantify age variation in RT distributions, accuracy, congruency effects, CAFs, fast-error magnitude, and correct-error timing without assuming a model-parameter explanation. VAM passes the published mean-RT component. M0 is partial because its seven-group mean alignment is descriptive, in-sample, and uses age-specific timing calibration. Use participant-balanced age-profile errors and uncertainty as the main metrics.

### P2. Stimulus-Specific Behavioral Prediction

Jaffe et al. confirmed human layout and position effects and showed corresponding VAM predictions. The stronger target is held-out prediction of which stimuli or stimulus features are relatively fast, slow, easy, difficult, or error-prone. The current pipeline has not established a fine-grained human target or tested M0 against it. First define reliable stimulus units, repeat counts, split rules, and noise ceilings; then use held-out RT and accuracy correlations and feature-effect errors.

## 5. Establishing Human Signatures Before Model Evaluation

Before further optimization, a human-only LIM audit should freeze the exact form of H1-H6. Each audit entry must use this template:

### Hx. Signature Audit Template

- **Prior evidence:** literature or previous LIM result motivating the signature.
- **Human operationalization:** exact inclusion rules, grouping variables, equation, and uncertainty method.
- **Human dataset status:** confirmed, needs analysis, or uncertain.
- **Required figure:** a human-only visualization that establishes the target without model predictions.
- **Frozen pass criterion:** set only after the human result is known.

The audit should validate any derived CAF or delta table against raw trial rows, use actual RT coordinates, preserve participant-level aggregation, record exclusions, and treat no-crossing deadline values only as censoring sentinels. The model must not define its target behavior after its own output has been inspected.

## 6. Model Evaluation Against Human Signatures

Every confirmed signature should be evaluated for Human, VAM, and M0 under the same preprocessing and summary rules.

Current VAM status labels summarize published findings and have not all been recomputed with the final frozen v1.0 metrics. Final comparisons require a matched re-evaluation; published qualitative status must not be mistaken for completion of that analysis.

**Level 1 - Qualitative reproduction.** Does the model reproduce the direction or qualitative phenomenon? Examples include `RT_incongruent > RT_congruent`, `Accuracy_incongruent < Accuracy_congruent`, and `Accuracy_fast_incongruent < Accuracy_slow_incongruent`.

**Level 2 - Quantitative correspondence.** How close is the model to the human result? Depending on the signature, use RMSE, MAE, correlation, slope difference, effect-size difference, Wasserstein distance, or quantile error, with uncertainty where possible.

The comparison should test whether the current model improves the theoretically important signatures without degrading signatures already captured by VAM. Model superiority is not defined by a simple count of passed signatures.

## 7. Core Human-Signature Matrix

This matrix records current evidence, not final judgments. Mixed signatures remain partial until their untested components are resolved.

| Signature | Human LIM status | VAM | M0 | Main metric | Benchmark objective | Priority |
|---|---|---|---|---|---|---|
| H1 Non-deterministic decision behavior | Congruent non-ceiling confirmed; repeated-stimulus test absent | PASS for aggregate non-ceiling behavior; repeat metric untested | FAIL for non-ceiling target; repeat metric untested | Congruent error rate; repeat consistency if feasible | REPAIR / MATCH | KEY SECONDARY |
| H2 Overall congruency cost | CONFIRMED | PASS | PARTIAL | RT and accuracy effect error | PRESERVE | FOUNDATIONAL / BENCHMARK-PRESERVATION |
| H3 Fast-error / conditional accuracy signature | CONFIRMED | FAIL | PARTIAL on diagnostic subsets | CAF slope, bin contrast, curve RMSE | IMPROVE | PRIMARY |
| H4 RT-dependent congruency cost | CONFIRMED | PASS | PARTIAL: cost overstated | Delta slope and curve RMSE | PRESERVE | FOUNDATIONAL / BENCHMARK-PRESERVATION |
| H5 RT-distribution shape | Broad tail evident; unified audit needed | PARTIAL in published/descriptive evidence; matched score absent | FAIL on slow tail; otherwise PARTIAL | Quantiles and Wasserstein distance | QUANTIFY / IMPROVE M0 | KEY SECONDARY |
| H6 Correct-error RT relationship | Prior LIM result; local audit needed | FAIL for dynamic error pattern | NOT YET TESTED | Condition-by-accuracy RT contrasts | POTENTIAL IMPROVEMENT | KEY SECONDARY |

### Extended Predictive Targets Matrix

| Target | Human evidence | VAM | M0 | Main metric | Benchmark objective | Role |
|---|---|---|---|---|---|---|
| P1 Age-related behavioral variation | Mean RT confirmed; broader H1-H6 profile incomplete | PASS for mean RT component | PARTIAL and in-sample | Participant-balanced age-profile error | EXTEND / GENERALIZE | INDIVIDUAL-DIFFERENCE VALIDITY / GENERALIZATION |
| P2 Stimulus-specific behavioral prediction | Layout/position confirmed; fine-grained target absent | PASS for reported feature effects | NOT YET TESTED | Held-out stimulus correlations | EXTEND / GENERALIZE | PREDICTIVE VALIDITY / GENERALIZATION |

### Benchmark decision principle

A successor model is not required to outperform VAM on every metric. It should preserve VAM's established successes, improve theoretically important failures, repair its own new failures, and extend predictive validity.

## 8. Mechanistic Tests: Only After Behavior

Mechanistic hypotheses explain why a model passes or fails confirmed behavioral criteria. They are not themselves human signatures.

In this section, `M1-M4` are mechanism labels. To avoid ambiguity, post-baseline model versions should always be written in full as `Model M1`, `Model M2`, and so forth.

### M1. Effective Visual Evidence Dynamics

What effective target and flanker evidence is delivered to the decision system across processing stages? Analyze target evidence across layers/time, flanker evidence across layers/time, target-minus-flanker evidence, reversal or crossover timing, and whether those quantities predict RT, fast errors, slow correct responses, or fast correct responses. The repository partially supports an early-flanker/later-target reversal in the current model, but this is not proof of a human mechanism. This evidence/readout-level mechanism primarily addresses H3 and H6.

### M2. Recurrent Competitive Decision Dynamics

How does Wong-Wang competition transform transient conflict into candidate fast-error, slow-correct, and fast-correct trajectories? These classes must be shown to be empirically distinguishable and behaviorally predictive. The presence of recurrence alone is not evidence. This mechanism primarily addresses H3 and H6.

### M3. Sources of Decision Variability and Congruent Errors

What process generates the non-deterministic human behavior captured by H1, especially occasional congruent errors? Candidate explanations include sensory noise, accumulator noise, starting-state variability, attention/arousal variability, trial-history or serial-dependence effects, and lapse-like processes. No source should be selected in advance. If serial dependence is confirmed in the human audit, it becomes a stronger candidate explanation, but it remains exploratory until then.

### M4. Representational Basis of Evidence Dynamics — Exploratory

**EXPLORATORY MECHANISTIC EXTENSION.** What internal representational transformations give rise to the effective evidence dynamics observed in M1? Candidate analyses include subspace alignment, principal angles, decoding, representational similarity, and target/flanker contamination. M4 is a representation-level analysis, not another version of M1. The conceptual relationship is internal representational geometry -> effective target/flanker evidence -> recurrent decision dynamics -> behavior. Do not assume that representations must evolve from overlap to orthogonalization; the direction of change must be measured. M4 remains exploratory unless supported empirically.

## 9. Mechanistic Necessity and Ablations

**Level 3 - Mechanistic necessity.** A strong explanation requires a matched intervention that selectively degrades the behavioral signature it is proposed to explain.

- Freeze or remove temporal visual dynamics, then re-evaluate H3 and H6.
- Remove recurrent competition or replace it with a simpler accumulator, then re-evaluate capture/correction behavior and all core signatures.
- Reduce or remove the relevant variability source, then re-evaluate H1 and the full matrix.

Ablations must use the same trial inclusion, readout, censoring, binning, and evaluation metrics as the full model. No ablation should be judged from one preferred figure; the complete H1-H6 matrix must be recomputed.

## 10. Calibration Versus Evaluation

Every reported metric must be assigned one role before analysis:

| Role | Purpose | Interpretation |
|---|---|---|
| `CALIBRATION` | Directly sets parameters, such as mean RT through timing or non-decision time | Not an independent prediction |
| `MODEL SELECTION` | Chooses among candidate schedules or models | In-sample selection evidence |
| `PRIMARY EVALUATION` | Tests a frozen primary signature such as held-out H3 | Independent behavioral evidence |
| `MECHANISTIC TEST` | Tests trajectory predictions or ablation consequences | Explanatory evidence if behavior is preserved |
| `GENERALIZATION` | Tests held-out participants, stimuli, ages, or datasets | Predictive evidence |

In current promoted results, mean RT and parts of the choice-coupled schedule were calibration or model-selection targets on the evaluated representative trials. They must not be presented as fully independent predictions. A metric may serve only one primary role within a given analysis split.

## 11. Generalization and Prediction

Signature reproduction is necessary but insufficient for a strong image-computable model.

**Current-paper internal validation:** P1 across all available age groups, P2 on held-out stimuli, held-out participants where feasible, and stimulus-specific prediction. These follow once the core H1-H6 framework is sufficiently stable.

**High-priority external validation:** replication in at least one independent Flanker dataset, if feasible. This is the preferred external-validation stage before any cross-task extension.

**Future Stage 2:** only after the Flanker account is stable, test cross-task generalization to Stroop, Simon, and other conflict tasks. The current paper should not claim cross-task generality.

External data and cross-task transfer are not mandatory current-paper claims unless the project scope changes.

## 12. Proposed Paper Logic

The following is a proposed manuscript structure, not a statement that all results currently exist:

1. **Core human behavioral signatures of four-choice Flanker decision-making.** Establish H1-H6 with human-only analyses before discussing model performance.
2. **Which human signatures are reproduced by VAM and M0?** Use the common matrix to distinguish shared successes, VAM successes M0 must preserve, VAM failures M0 improves, and failures not yet explained by either model.
3. **Hierarchical visual dynamics and the emergence of distractor and target evidence.** Test mechanisms relevant primarily to H3 and possibly H6.
4. **Recurrent competition and fast error versus successful correction.** Analyze fast-error, slow-correct, and fast-correct trajectories.
5. **General decision variability and non-deterministic congruent behavior.** Address H1 without prescribing the final mechanism.
6. **Full-model behavioral adequacy across H1-H6.** Recompute the complete core matrix after targeted improvements, including distributions rather than only means.
7. **Predictive, individual-difference, and external Flanker validation.** Evaluate P1 age-related behavioral variation, P2 stimulus-specific prediction, held-out participants and stimuli, and replication in an independent Flanker dataset if feasible.
8. **Mechanistic necessity.** Test whether ablations selectively degrade the hypothesized human signature while preserving the broader core matrix.

## 13. Central Contribution

The contribution should be assessed only after the human-signature comparison:

1. **Behavioral contribution:** the model improves reproduction of dynamic conflict behavior, particularly H3.
2. **Mechanistic contribution:** the model links changing hierarchical visual evidence to recurrent competitive decision dynamics.
3. **Theoretical contribution:** decision errors may depend not only on distractor strength, but also on how target and distractor information evolve relative to decision commitment.

The theoretical claim remains conditional until supported by the human audit, matched model comparison, trajectory analyses, ablations, and held-out tests.

## 14. Current Status and Roadmap

The current result is a diagnostic model-development package, not a final validated full-cohort or causal aging model.

- **H1 - CURRENT MODEL FAIL:** M0 does not reproduce the non-deterministic behavior observed in human congruent trials.
- **H3 - MODEL PARTIAL:** M0 reproduces the incongruent CAF recovery direction on representative subsets, but not yet under held-out evaluation.
- **H4 - MODEL PARTIAL:** M0 overstates the RT congruency cost across the distribution.
- **H5 - CURRENT MODEL FAIL/PARTIAL:** M0 does not reproduce the human slow RT tail.
- **H6 - NOT YET TESTED:** matched correct-error timing remains incomplete.
- **P1 - MODEL PARTIAL:** seven-group mean RT alignment is in-sample, with uneven participant counts and calibrated timing.
- **P2 - NOT YET TESTED:** promoted stimulus-specific prediction is absent.
- Dynamic representational geometry and mechanism-specific ablations are not yet tested.

Prioritized roadmap:

1. freeze Evaluation Framework v1.0;
2. perform the human-only H1-H6 audit;
3. determine whether a direct repeated-stimulus H1 analysis is feasible;
4. freeze exact operational definitions and pass metrics;
5. audit frozen M0 without changing model parameters;
6. produce a Human/VAM/M0 core-signature scorecard;
7. identify the highest-priority failed core signature;
8. only then begin minimal single-factor model optimization;
9. re-evaluate all H1-H6 after every modification;
10. evaluate P1 and P2 after core adequacy is sufficiently stable;
11. run mechanistic trajectory analyses and ablations;
12. after internal validation, prioritize replication in an independent Flanker dataset;
13. only after the Flanker account is stable, consider cross-task generalization.

## 15. Baseline Freeze

The current fast-error-capable model should be frozen as **M0 - Fast-error baseline**. It should not be overwritten during optimization. Preserve its configuration, seed, inputs, trial predictions, and evaluation outputs.

M0 has already been observed and is the baseline audit model. Future `Model M1`, `Model M2`, and later variants must be judged prospectively against human targets and metrics frozen before those variants are inspected. The next workflow is:

> Human Framework v1.0 -> human-only H1-H6 audit -> freeze exact targets and metrics -> M0 audit -> PASS / PARTIAL / FAIL / NOT TESTED -> select one failed core signature -> diagnose a candidate mechanism -> minimal Model M1 modification -> re-evaluate all H1-H6 -> retain only if the target improves without unacceptable degradation elsewhere -> repeat prospectively

P1 and P2 follow after the core framework is sufficiently stable. This workflow prevents improvement of one behavioral effect from silently degrading another and prevents later model performance from moving the evaluation target.

## References and Scope Notes

- Jaffe et al. (2025), *An image-computable model of speeded decision-making*. VAM's reported qualitative failure concerns the **incongruent** CAF / fast-error pattern.
- Rafiei et al. (2024), *The neural network RTNet exhibits the signatures of human perceptual decision-making*. This framework adopts its human-signature-first reasoning, not its task-specific signature list.
- Current evidence boundaries are summarized in [Current Results and Limitations](current_results_and_limitations.md), [R5 Supervisor Follow-Up](r5-supervisor-followup.md), and the [Artifact Results Documentation Index](../artifacts/results/ARTIFACT_DOCS_INDEX.md).

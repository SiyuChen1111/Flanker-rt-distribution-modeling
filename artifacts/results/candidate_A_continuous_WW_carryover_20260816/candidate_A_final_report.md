# Candidate A final report

## Decision

**TIMING UNRESOLVED — CANDIDATE A NOT RUN.**

Candidate A cannot be validly evaluated from the frozen repository because the real time from trial-*n* commitment/response to trial-*n+1* sensory onset is unavailable. The participant data provide response time and ordinal trial/session identifiers, but no boundary timestamps. The repository also contains no LIM task-presentation implementation from which post-response display, feedback, or ITI duration could be recovered.

## Requested outcome summary

1. **Real trial-boundary timing:** unresolved. Fixed versus variable ITI, feedback duration, post-response duration, and commitment-to-next-stimulus time cannot be established.
2. **A1 attractor trapping:** not evaluated; A1 was not run.
3. **Predeclared A2 grid:** not created, because any simulation would first require an invented ITI/evidence-off time.
4. **Partial-relaxation settings:** not evaluated.
5. **Stage-1 `Delta_generic`:** not evaluated for A0, A1, or A2.
6. **Post-error direction:** not evaluated.
7. **Congruent errors:** not evaluated.
8. **H3:** not evaluated and therefore not put at risk.
9. **Stage 2:** not triggered.
10. **Transition and lag effects:** not evaluated.
11. **Directional carryover predictions:** not evaluated.
12. **Final classification:** A-S0 — TIMING UNRESOLVED.
13. **Outputs:** this report and `candidate_A_timing_audit.md` in the same directory. Later-stage CSV files and figures were intentionally not fabricated.
14. **Canonical C0v2:** not modified. No canonical code, configuration, evidence, parameters, seeds, results, or human-matching procedures were changed.

## What is known from the model

The frozen C0v2 uses 80 steps of 10 ms and initializes every four-channel trial at `S = 0.1`. Sensory input is active throughout this independently simulated 0.80 s numerical horizon. This establishes the current hard reset at modeled trial start, but it does not establish the real task's post-response or inter-trial timing. The additive nondecision-time values are RT components, not documented ITIs.

The detailed evidence and inspected-source list are recorded in `candidate_A_timing_audit.md`.

**A-S0 — TIMING UNRESOLVED**

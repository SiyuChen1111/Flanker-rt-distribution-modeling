# Full technical report

1. CAF/CRF now use actual median RT coordinates. Human and Model are binned separately, so the x positions differ when their RT distributions differ.
2. The previous CRF should not be trusted. The recomputed CRF validation status is `True` and the validation report is `03_CRF_validation_report.md`.
3. Fixed-time `S_i(t)` distributions were summarized separately from first-passage times in `04_state_distribution_statistics.csv`.
4. Approximate normality of `S_i(t)` at a fixed time does not imply normal first-passage RT.
5. R5 hard WW decision times are more compressed than final RT, while final RT inherits substantial spread from t0.
6. Synthetic controls partially pass minimal sanity checks but do not establish that the accumulator fully explains human RT shape.
7. Excessive incongruent errors are most consistent with a mixture of evidence-origin and premature/low-variability readout mechanisms; ambiguous trials are left ambiguous.
8. Young and Older differ mainly in t0 and threshold/margin settings in the active R5 package.
9. The apparent RT fit is materially supported by group-specific t0 variability.
10. Highest-priority modification: 下一步优先做单因素小消融：限制 t0 波动，同时测试校准后的 accumulator noise 或 threshold/margin，不要一次加入多种新机制。

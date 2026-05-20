# Reaggregated Public Release Review

path | file_size | file_type | purpose | public_decision | reason
--- | --- | --- | --- | --- | ---
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/condition_level_error_summary.csv` | 2235 bytes | csv | aggregated condition-level summary | keep_public | small aggregated table, no raw trial dump
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/feasibility_scorecard.csv` | 2111 bytes | csv | aggregated scorecard | keep_public | compact evidence table supporting public summary
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/feasibility_verdict.json` | 1130 bytes | json | verdict summary | keep_public | compact machine-readable verdict, not raw data
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/human_pooled_metrics.csv` | 2360 bytes | csv | pooled human metrics | keep_public | aggregated metrics only
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/human_subject_level_metrics.csv` | 1559 bytes | csv | aggregated subject-level human metrics | human_confirm | subject-level table is small and aggregated, but you may want to decide whether subject-level release is acceptable
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/mixture_reconstruction_summary.csv` | 725 bytes | csv | reconstruction summary | keep_public | small derived summary
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/model_pooled_metrics.csv` | 4408 bytes | csv | pooled model metrics | keep_public | aggregated model summary
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/model_subject_level_metrics.csv` | 1729 bytes | csv | aggregated subject-level model metrics | human_confirm | subject-level table is small and aggregated, but still a subject-level release decision
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/single_subject_skew_verdict.json` | 356 bytes | json | compact skew verdict | keep_public | tiny verdict artifact
`artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/subject_level_comparison.csv` | 1138 bytes | csv | aggregated subject comparison | human_confirm | comparison is compact, but still subject-level rather than pooled-only

# Public Release Rename Plan

current_path | proposed_path | file_type | reason | risk | action
--- | --- | --- | --- | --- | ---
`code/scripts/train_variational_ww_smoke.py` | `code/scripts/train_variational_ww_smoke.py` | python | already snake_case and clearly names the variational Wong-Wang smoke bridge; many references and imports | medium | keep_current_name
`code/scripts/train_dmc_var_ww_smoke.py` | `code/scripts/train_dmc_var_ww_smoke.py` | python | already snake_case and clearly names the DMC+Var→WW training script; referenced by docs and imports | medium | keep_current_name
`code/scripts/run_subject_level_dmc_var_ww.py` | `code/scripts/run_subject_level_dmc_var_ww.py` | python | already snake_case and clearly names the subject-level DMC+Var workflow; imported by analyzer | low | keep_current_name
`code/scripts/analyze_subject_level_dmc_var_ww.py` | `code/scripts/analyze_subject_level_dmc_var_ww.py` | python | already snake_case and clearly names the analyzer; imported by docs only lightly | low | keep_current_name
`code/scripts/stage1_semisup_evidence_sampler.py` | `code/scripts/stage1_semisup_evidence_sampler.py` | python | already snake_case and descriptive; imported in multiple scripts | medium | keep_current_name
`code/scripts/vgg_wongwang_lim.py` | `code/scripts/vgg_wongwang_lim.py` | python | established core model file; rename would require broad import updates across legacy and public scripts | high | needs_human_confirmation
`code/scripts/cache_vgg_stage2_features.py` | `code/scripts/cache_vgg_stage2_features.py` | python | already descriptive snake_case; imported by public-core scripts | low | keep_current_name
`code/scripts/project_paths.py` | `code/scripts/project_paths.py` | python | already concise utility name and stable import target | low | keep_current_name
`code/scripts/train_age_groups_efficient.py` | `code/scripts/train_age_groups_efficient.py` | python | supporting utility with many imports across current scripts; rename risk is very high and not needed for public hygiene | high | needs_human_confirmation
`README.md` | `README.md` | markdown | standard repository root doc name | low | keep_current_name
`docs/current_results_and_limitations.md` | `docs/current_results_and_limitations.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`docs/model_framework_summary.md` | `docs/model_framework_summary.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`docs/public_update_notes.md` | `docs/public_update_notes.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/public_readiness_review_final.md` | `artifacts/public_readiness_review_final.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/git_status_release_plan.md` | `artifacts/git_status_release_plan.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/cleanup_manifest_public_update.md` | `artifacts/cleanup_manifest_public_update.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/reaggregated_public_release_review.md` | `artifacts/reaggregated_public_release_review.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/local_path_cleanup_review.md` | `artifacts/local_path_cleanup_review.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/environment_cleanup_review.md` | `artifacts/environment_cleanup_review.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name
`artifacts/large_file_public_release_review.md` | `artifacts/large_file_public_release_review.md` | markdown | already lower_snake_case and descriptive | low | keep_current_name

## Notes

- No low-risk rename was necessary in this round.
- Public-core Python filenames are already snake_case and sufficiently descriptive.
- `vgg_wongwang_lim.py` and `train_age_groups_efficient.py` remain the only naming candidates with meaningful rename risk, so they were not renamed.
- `python -m py_compile` passed for the checked public-core/supporting scripts.

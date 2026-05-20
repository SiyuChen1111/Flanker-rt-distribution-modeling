# Local Path Cleanup Review

path | category | action | reason
--- | --- | --- | ---
`README.md` | A. Public docs/code that must be fixed | already fixed where needed | public-facing narrative already cleaned of local absolute paths
`docs/current_results_and_limitations.md` | A. Public docs/code that must be fixed | no `/Users/siyu` found | safe in current public form
`docs/model_framework_summary.md` | A. Public docs/code that must be fixed | no `/Users/siyu` found | safe in current public form
`docs/public_update_notes.md` | A. Public docs/code that must be fixed | no `/Users/siyu` found | safe in current public form
`code/scripts/train_dmc_var_ww_smoke.py` | A. Public docs/code that must be fixed | fixed | removed local absolute path example
`code/scripts/train_variational_ww_smoke.py` | A. Public docs/code that must be fixed | fixed | removed local absolute path example
`CLAUDE.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | local operator instructions with machine-specific paths
`docs/project/AGENTS.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | internal workflow doc with local paths
`docs/project/REPRODUCE_VAM.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | internal reproduction notes with local paths
`docs/project/REPRODUCTION_GUIDE.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | internal reproduction notes with local paths
`docs/notes/siyu_study.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | personal notes with local paths
`docs/notes/ideas.md` | B. Non-public docs that should be archived or ignored | exclude from public commit | personal notes with local paths
`artifacts/archive_legacy_not_for_public/**` | C. Archive files | leave in archive and ignore | already outside public repo boundary
`archive/*.py` with local paths | D. Legacy scripts | do not include in public commit | legacy helper scripts outside current public core
`code/scripts/add_visualization.py` | D. Legacy scripts | do not include in public commit | legacy notebook helper with local paths
`code/scripts/fix_drift_rate_output.py` | D. Legacy scripts | do not include in public commit | legacy notebook helper with local paths
`code/scripts/rt_distribution_shape_analysis.py` | D. Legacy scripts | do not include in public commit | not public core and contains local path example
`code/scripts/run_var_ww_param_scan.py` | D. Legacy scripts | do not include in public commit | not public core and contains local path example
`code/scripts/train_mc_dropout_ww_smoke.py` | D. Legacy scripts | do not include in public commit | non-public branch with local path example
`code/scripts/compare_age_evidence.py` | D. Legacy scripts | do not include in public commit | non-public helper with local path example
`code/scripts/cross_age_behavioral_analysis.py` | D. Legacy scripts | do not include in public commit | non-public helper with local path example

# Git Status Release Plan

## A. Stage for public commit

- `.gitignore`
- `README.md`
- `docs/current_results_and_limitations.md`
- `docs/model_framework_summary.md`
- `docs/public_update_notes.md`
- `code/scripts/train_dmc_var_ww_smoke.py`
- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/vgg_wongwang_lim.py`
- `artifacts/cleanup_manifest_public_update.md`
- `artifacts/reaggregated_public_release_review.md`
- `artifacts/local_path_cleanup_review.md`
- `artifacts/environment_cleanup_review.md`
- `artifacts/large_file_public_release_review.md`
- `artifacts/git_status_release_plan.md`
- `artifacts/public_readiness_review_final.md`
- `artifacts/public_release_rename_plan.md`
- tracked deletions that correspond to archive moves already accepted for the public release boundary

## B. Do not stage / ignored local only

- `.venv/`
- `artifacts/archive_legacy_not_for_public/`
- local private workspace already moved into archive
- local cache / processed cache under `data/vam_data/processed_cache/`
- local checkpoints under `artifacts/checkpoints/`

## C. Remove from Git index only

- none detected for `.venv/`, `venv/`, or `env/` in this round
- if any tracked checkpoint/cache path later appears in `git status`, use `git rm --cached <path>` without deleting local files

## D. Restore or revert

- non-public docs with local machine paths: `CLAUDE.md`, `docs/project/*`, `docs/notes/*`
- legacy scripts with local machine paths that are outside the public core
- unrelated experiment code or result changes not needed for the mechanism-test release

## Notes

- Current `git status` is still broad because the repo contains many historical tracked deletions and unrelated code changes.
- The public commit should be deliberately staged from the small set in section A rather than by adding everything.
- No low-risk `git mv` rename was needed in this round; naming normalization concluded that current public-core filenames can remain as-is.

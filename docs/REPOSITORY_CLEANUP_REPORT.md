# Repository Cleanup Report

## Repository purpose

The retained project is the VGG16 -> layerwise time-varying evidence -> four-choice Wong-Wang R5 Flanker diagnostic used for the presentation. The exact identity is recorded in `docs/PRESENTATION_MODEL_MANIFEST.md`.

## Added canonical infrastructure

- `configs/presentation_model.json` records the model fingerprint, layers, schedules, parameters, seeds, and readout rules.
- `code/scripts/reproduce_presentation_model.py` validates the original and corrected-equivalent two-group result tables and runs a small accumulator smoke test.
- `docs/REPRODUCING_RESULTS.md` gives non-destructive verification and figure commands.
- `docs/PRESENTATION_MODEL_MANIFEST.md` traces the mechanism figure back to its implementation and data.

## Historical material

No historical files were removed from the current repository. At the user's request, the existing history, experiments, notebooks, reports, and result directories remain in place. An additional copy was created at `../Flanker-rt-distribution-modeling_local_archive_20260810/` with `ARCHIVE_MANIFEST.md`, `PRE_CLEANUP_GIT_STATUS.txt`, and `PRE_CLEANUP_TRACKED_FILES.txt`.

## Verification

- Model imports and Wong-Wang smoke test: passed.
- Original and corrected-equivalent tables: 10,000 rows each; both age groups contain 5,000 rows.
- Focused tests: 11 passed.
- Python compilation and whitespace checks for changed Python files: passed; pre-existing generated SVG whitespace is unchanged.
- Full VGG extraction and full schedule refit: not rerun; this task did not change scientific calculations.

## Git behavior

No commit, push, history rewrite, or deletion of historical repository content was performed. The current GitHub-visible tree therefore still contains the historical material. The external archive is outside Git tracking and serves only as a recoverable duplicate.

Suggested future commit message for the non-destructive documentation/infrastructure changes:

`docs: document canonical VGG-Wong-Wang presentation model`

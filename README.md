# VAM-studying

This repository studies **human reaction-time (RT) behavior** in the LIM / Flanker task using a **VGG-based visual front-end** and **decision-dynamics models**.

The main modeling line is:

- shared Stage-1 visual evidence model
- age-specific Stage-2 Wong-Wang (WW) decision dynamics
- group-level comparisons between `20-29` and `80-89`

The broader scientific goal is to understand whether the model can reproduce key human RT-distribution signatures, including:

- plausible RT scale
- right-skew / heavy tails
- congruency effects
- error-conditioned RT structure
- age-related differences in decision dynamics

---

## Current status

The project has already explored:

1. initial VGG drift-rate prototyping
2. formal VGG + WW age-group modeling
3. response-supervision correction
4. readout redesign experiments
5. behavior-balanced selector / eval redesign
6. lightweight WW objective tweaks
7. structured accumulator-RNN prototype
8. RT-distribution-shape losses
9. WW error-regime experiments

### Current bottom line

The strongest current result is that the WW line can be pushed into an **error regime**, but the resulting regime is still **not yet human-like**.

The repository is therefore in a **consolidation phase**, not an open-ended local patching phase.

---

## Recommended entry points

If you are new to this repository, start here:

1. `docs/history/logs.md`
   - now moved to `docs/history/logs.md`
   - dated project timeline from the first VGG work through the current error-regime chain

2. `artifacts/results/organized/handoff/supervisor_update_2026-04-08.md`
   - concise supervisor-facing summary

3. `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md`
   - most important technical memo for the latest WW branch

4. `artifacts/results/organized/README.md`
   - guide to current result folders and evidence levels

---

## Top-level structure

### Core docs
- `docs/history/logs.md` — dated research timeline
- `docs/project/research_plan.md` — age-group execution plan
- `docs/project/research_proposal_v4.md` — broader research proposal
- `docs/project/AGENTS.md` — project-specific agent guidance

### Main code
- canonical scripts now live under `code/scripts/`
- root-level `scripts/` is retained as a compatibility symlink
- key entry points include:
  - `code/scripts/vgg_wongwang_lim.py`
  - `code/scripts/train_age_groups_efficient.py`
  - `code/scripts/vgg_accumulator_rnn.py`
  - `code/scripts/vgg_accumulator_rnn_v2.py`
  - `code/scripts/train_age_group_accumrnn.py`
  - `code/scripts/run_matched_full_age_group_analysis.py`

### Path abstraction status
- Stage 2 path abstraction has started via:
  - `code/scripts/project_paths.py`
- The most active age-group WW scripts now use this layer for canonical roots.
- The new grouped layout exists, while root-level names are retained as compatibility symlinks.

### Data / checkpoints
- canonical grouped layout:
  - `data/age_groups/`
  - `data/age_groups_matched/`
  - `data/vam_data/`
  - `artifacts/checkpoints/age_groups/`
  - `artifacts/checkpoints/age_groups_matched/`
  - `artifacts/checkpoints/test/`
- compatibility symlinks retained at root:
  - `data_age_groups/`
  - `data_age_groups_matched/`
  - `vam_data/`
  - `checkpoints_age_groups/`
  - `checkpoints_age_groups_matched/`
  - `checkpoints_test/`

### Results
- canonical location:
  - `artifacts/results/`
- root-level `results/` is retained as a compatibility symlink
- curated navigation layer remains under:
  - `artifacts/results/organized/`

### Supporting material
- `docs/papers/` — reference PDFs
- `docs/notes/` — personal / informal notes
- `docs/project/` — project-level plans, guides, and repo notes
- `docs/history/` — historical timeline / logs
- `notebooks/legacy_root/` — older root-level notebooks moved out of the main path
- `logs/runtime_archive/` — archived runtime `.log` / `.pid` files
- `config/` — configuration-style root files such as `requirements.txt`
- `code/vam/` — canonical VAM mixed code/assets directory
- root-level `vam/` and `Kar/` are retained as compatibility symlinks during migration and should not be treated as the canonical code location

---

## Reproducibility note

This repository follows a script-based workflow and keeps:

- code
- data preparation
- analysis scripts
- result summaries
- handoff memos

so that the research trail remains interpretable.

For the current best interpretation of file structure and evidence levels, see:

- `artifacts/results/organized/README.md`
- `artifacts/results/organized/FILE_MAPPING.md`

For a root-file relocation map, also see:

- `docs/project/ROOT_LEVEL_PY_CLASSIFICATION.md`

For the large-directory migration strategy, see:

- `docs/project/REPO_RESTRUCTURING_PLAN.md`

---

## Current scientific takeaway

The strongest current finding is:

> A VGG + WW model can be pushed from a no-error regime into a measurable error regime, but the resulting regime is still too fast and directionally wrong relative to current human error-conditioned RT behavior.

This means the next useful step is likely to be either:

- a stronger next-line structured model
- or a reframing of the behavioral target itself

rather than more local WW patching.

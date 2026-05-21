# Public Readiness Review Final

## A. Current readiness status

**Ready for final human review before public commit**

Reason:

- 第一轮清理已经完成：私人 workspace、申请计划、notebooks、草稿、日志和大量旧结果已删除或归档。
- 当前公开叙事已经收敛到 `variational evidence sampling + DMC-style conflict signal + Wong-Wang decision model as a mechanism test`。
- 仍建议你在最终提交前做一次人工确认，但两个最后 blocker 已经完成收口。

## B. What has already been cleaned

- 已删除：`.DS_Store`、`draft.md`、本地运行日志、临时 prompt 文件。
- 已归档：`brain_storm/`、`notebooks/`、`phd-application-plan/`。
- 已归档的旧结果类别：
  - response supervision / cross-age / aligned behavior / model-aligned / per-subject plan
  - HSFA / MC-dropout / semisup calibration 旧分支
  - variational synthesis 的旧 smoke 和 scan 扫描结果
- archive 目录已被 `.gitignore` 排除，不会进入公开提交。

## C. Public core to keep

### Core docs

- `README.md`
- `docs/current_results_and_limitations.md`
- `docs/public_update_notes.md`
- `docs/model_framework_summary.md`

### Core code

- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/stage1_evidence_sampler.py`
- `code/scripts/analyze_subject_level_dmc_var_ww.py`
- `code/scripts/cache_vgg_stage2_features.py`
- `code/scripts/project_paths.py`
- `code/scripts/train_dmc_var_ww_smoke.py`

### Public evidence

- `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md`
- `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`
- `artifacts/results/rt_model_variational_ww_synthesis/var_ww_mechanism_memo.md`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`

### Keep public but polish

- `README.md`
- `docs/public_update_notes.md`
- `docs/model_framework_summary.md`
- `docs/current_results_and_limitations.md`
- `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.png`
- `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.pdf`

Reason:

- 这些文件应该公开，但需要继续保持谨慎表述，避免把当前结果写成 final human RT fit。

## D. Files moved or kept outside public repo

- `artifacts/archive_legacy_not_for_public/private_workspace/brain_storm/`
- `artifacts/archive_legacy_not_for_public/private_workspace/notebooks/`
- `artifacts/archive_legacy_not_for_public/private_workspace/phd-application-plan/`
- `artifacts/archive_legacy_not_for_public/results/age_groups_response_supervision_interim/`
- `artifacts/archive_legacy_not_for_public/results/cross_age_behavioral/`
- `artifacts/archive_legacy_not_for_public/results/model_aligned_20_29/`
- `artifacts/archive_legacy_not_for_public/results/per_subject_age_comparison_plan/`
- `artifacts/archive_legacy_not_for_public/results/proposal_aligned_behavior/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_hsfa_v3_1/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_mc_dropout_ww/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_semisup_spea_v1/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_1024_n008_thr022_t025/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_1024_noise006_thr025/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_1024_t020_thr030/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_noise004_thr035/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_noise006_thr025/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/calib_noise006_thr035/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/scan_batch1/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/smoke/`
- `artifacts/archive_legacy_not_for_public/results/rt_model_variational_ww_synthesis/smoke_v2/`

## E. Resolved manual review decisions

### Results table

| path | final_decision | reason | action |
| --- | --- | --- | --- |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/` | `archive_not_public` | older feasibility branch; not directly cited by current README/docs | move to archive |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/` | `keep_public_but_polish` | only `reaggregated/` is part of the public evidence spine; sibling raw content has now been archived | keep only `reaggregated/` public |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/` | `keep_public` | directly cited by README and public notes | keep public |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only_noise05/` | `archive_not_public` | noise comparison branch; diagnostic, not part of current core story | move to archive |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_accumrnn_aligned/` | `archive_not_public` | accumrnn comparison is outside current Var+DMC+WW story | move to archive |
| `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/` | `archive_not_public` | historical diagnostic comparison | move to archive |
| `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/` | `archive_not_public` | export comparison support material, not current public evidence | move to archive |
| `artifacts/results/rt_model_dmc_var_ww/summary_smoke.md` | `keep_public` | current main summary | keep public |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/` | `keep_public` | directly cited public run | keep public |
| `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.png` | `keep_public_but_polish` | public figure supporting the summary; should remain framed as mechanism illustration | keep public |
| `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.pdf` | `keep_public_but_polish` | same as above | keep public |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a3_s4/` | `archive_not_public` | older smoke variant not cited by current docs | archive |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a3_s4_all/` | `archive_not_public` | older smoke variant not cited by current docs | archive |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a3_s4_delayed/` | `archive_not_public` | older smoke variant not cited by current docs | archive |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3/` | `archive_not_public` | intermediate sibling of kept run; not the retained public evidence run | archive |
| `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_gated/` | `archive_not_public` | intermediate/gated sibling; not cited in README | archive |

### Script table

| script | role | final_decision | reason | action |
| --- | --- | --- | --- | --- |
| `code/scripts/train_variational_ww_smoke.py` | `public_core` | `keep_public` | cited by README/docs and mechanism memo; core bridge into Wong-Wang | keep |
| `code/scripts/run_subject_level_dmc_var_ww.py` | `public_core` | `keep_public` | cited by README/docs; runs subject-level DMC+Var workflow | keep |
| `code/scripts/vgg_wongwang_lim.py` | `public_core` | `keep_public_but_polish` | core model file; contains a local test block with print output only | keep, optionally trim demo block later |
| `code/scripts/stage1_evidence_sampler.py` | `public_core` | `keep_public` | core evidence sampler dependency; Stage-1 deterministic, variational, and MC-dropout evidence sampling | keep |
| `code/scripts/analyze_subject_level_dmc_var_ww.py` | `public_core` | `keep_public` | public panel summarizer cited by docs | keep |
| `code/scripts/cache_vgg_stage2_features.py` | `public_core` | `keep_public` | core support dependency for staged workflow | keep |
| `code/scripts/train_dmc_var_ww_smoke.py` | `public_core` | `keep_public_but_polish` | core training script, but file header includes a local absolute path example | keep and polish header comments |
| `code/scripts/run_true_single_subject_feasibility.py` | `supporting_repro` | `keep_public_but_polish` | supports the retained `reaggregated/` evidence chain | keep if single-subject reaggregated evidence remains public |
| `code/scripts/analyze_true_single_subject_feasibility.py` | `supporting_repro` | `keep_public_but_polish` | supports retained single-subject reaggregated evidence | keep if same evidence remains public |
| `code/scripts/run_true_single_subject_feasibility_accumrnn.py` | `legacy_experiment` | `archive_not_public` | accumrnn is outside current story | archive or exclude from public core |
| `code/scripts/analyze_true_single_subject_feasibility_accumrnn.py` | `legacy_experiment` | `archive_not_public` | accumrnn is outside current story | archive or exclude from public core |
| `code/scripts/train_age_groups_efficient.py` | `supporting_repro` | `keep_supporting_utility` | not part of the public story, but still imported by current public-core scripts | keep as supporting utility, not public core |

## F. Still requiring human confirmation

- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/reaggregated/`
  - 为什么不能自动决定：它是保留的 public evidence，本身不是风险项，但最终是否要公开整组 reaggregated 表格仍取决于你希望开放到什么粒度。
  - 它可能支持什么：README 中保留的单被试可读证据链。
  - 建议你怎么判断：如果你希望公开最小可复查证据，就保留；如果你只想公开 narrative summary，则可再缩到更少文件。
  - 如果确认不需要：将该目录一起归档。

## G. Code change review

### Should keep

- `code/scripts/train_dmc_var_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/stage1_evidence_sampler.py`
- `code/scripts/analyze_subject_level_dmc_var_ww.py`
- `code/scripts/cache_vgg_stage2_features.py`
- `code/scripts/train_variational_ww_smoke.py`

### Should keep but polish

- `code/scripts/train_dmc_var_ww_smoke.py`
  - local absolute path example appears in header comments
- `code/scripts/train_variational_ww_smoke.py`
  - local absolute path example appears in header comments
- `code/scripts/vgg_wongwang_lim.py`
  - contains a local test/demo block with verbose printing
- `code/scripts/run_true_single_subject_feasibility.py`
- `code/scripts/analyze_true_single_subject_feasibility.py`
  - keep only if the single-subject `reaggregated/` evidence remains public

### Should revert or move out of public core

- `code/scripts/run_true_single_subject_feasibility_accumrnn.py`
- `code/scripts/analyze_true_single_subject_feasibility_accumrnn.py`

### Needs human confirmation

- none required for blocker resolution; `train_age_groups_efficient.py` has been resolved as supporting utility

## J. Final blocker resolution

1. `true_single_subject_feasibility_rt_response_only/`
   - 最终保留公开：`reaggregated/`
   - 已归档：`20-29/`, `80-89/`, `panel_manifest.json`, `panel_splits.csv`, `single_subject_skew_decision_memo.md`, `subject_panel.csv`, `true_single_subject_feasibility_summary.md`
   - 当前公开目录只剩最小单被试证据

2. `train_age_groups_efficient.py`
   - 最终状态：`keep_supporting_utility`
   - 原因：它不是公开叙事主线，但被 `train_dmc_var_ww_smoke.py`、`train_variational_ww_smoke.py`、`run_subject_level_dmc_var_ww.py`、`run_true_single_subject_feasibility.py` 等当前公开相关脚本直接 import
   - 处理：保留文件，但不在 README 中作为主线脚本强调

3. 本轮移动到 archive
   - `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/` 中除 `reaggregated/` 之外的同级内容

4. 当前是否还需要 human confirmation
   - 没有新的 blocker 级 human confirmation
   - 只剩发布粒度上的最终人工判断：是否保留 `reaggregated/` 全目录公开

5. 当前状态
   - **Ready for final human review before public commit**

## J. Final release hygiene review

1. `reaggregated/` 公开性
   - `artifacts/reaggregated_release_review.md` 已生成
   - 当前目录中文件都很小，且都是聚合表、聚合指标或 verdict 文件
   - 默认判断：除 subject-level 聚合表外，其余 reaggregated 文件可公开
   - 你已确认以下文件不公开，因此已移出公开 evidence 目录：
     - `human_subject_level_metrics.csv`
     - `model_subject_level_metrics.csv`
     - `subject_level_comparison.csv`

2. `/Users/siyu` 本地路径
   - 当前 public-facing README/docs 与已保留主线脚本中的已知本地绝对路径已处理
   - 剩余本地路径主要出现在：
     - `CLAUDE.md`
     - `docs/project/*`
     - `docs/notes/*`
     - `archive/*`
     - 若干非主线 legacy scripts
   - 这些内容不应进入公开提交主线

3. `.venv` 状态
   - `.venv/` 存在，但未显示为当前 tracked 变更
   - `.gitignore` 已覆盖 `.venv/`, `venv/`, `env/`
   - 当前不需要执行 `git rm --cached .venv`

4. 大文件
   - 大文件 review 已写入 `artifacts/large_file_release_review.md`
   - 需要避免提交的主要对象：
     - `artifacts/checkpoints/test/stage1/*.pth`
     - `archive/model_assets/vgg16-397923af.pth`
     - `data/vam_data/processed_cache/*.npy`
   - `data/age_groups/20-29/train_data.csv` 已确认不公开，不应进入 public release

5. `git status` 收敛情况
   - 还没有完全收成干净工作区
   - 但已经可以围绕一小组明确文件做“final human review before public commit”
   - 应采用选择性 `git add`，不要整体提交当前工作区

6. 剩余需要 human confirmation 的文件
   - none from the previously flagged subject-level tables or `train_data.csv`; both have now been explicitly excluded from public release

7. 当前状态
   - **Ready for final human review before public commit**

8. naming normalization
   - 本轮没有执行任何 `git mv`
   - 原因：当前 public-core Python 和 Markdown 文件名已经符合 snake_case 或标准命名，且改名风险高于收益

## H. Remaining risks before GitHub public release

- Local absolute paths still appear in comments/example usage:
  - `code/scripts/train_dmc_var_ww_smoke.py`
  - `code/scripts/train_variational_ww_smoke.py`
- A local test/demo block remains in `code/scripts/vgg_wongwang_lim.py`
- `.sisyphus/`, `.trae/`, `.claude/`, `.venv/` still exist locally; they are not part of the public story and should remain ignored
- `artifacts/archive_legacy_not_for_public/` is ignored correctly, but many tracked deletions still exist in the working tree and need a final commit decision
- Remaining old result folders under `artifacts/results/repro_legacy_interim/` still need one final inclusion/exclusion decision
- There are still many unstaged and unconfirmed code/result changes in `git status`, so the repo is not yet in a clean public-release state

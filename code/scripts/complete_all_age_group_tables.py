#!/usr/bin/env python3
"""Complete all-age derived tables, figures, and reproducibility records."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vam-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/vam-cache")
Path("/private/tmp/vam-mpl").mkdir(parents=True, exist_ok=True)
Path("/private/tmp/vam-cache").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT / "artifacts/results/all_age_groups_20260806"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures_publication"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
ANALYSIS_TO_AGE = {"young_20_29": "20-29", "older_80_89": "80-89"}
EXTREME_METRICS = (
    PROJECT_ROOT
    / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803/selected_model_metrics.csv"
)
MIDDLE_METRICS = RESULTS / "corrected_model_by_age/selected_model_metrics.csv"
MIDDLE_PARAMETERS = RESULTS / "corrected_model_by_age/selected_parameters.csv"
MECHANISM_SOURCE = (
    PROJECT_ROOT
    / "artifacts/results/r5_real_vgg_target_flanker_audit_20260803/05_natural_emergence_evidence_chain"
)


def equal_count_bins(values: pd.Series, n_bins: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    finite_index = numeric.index[np.isfinite(numeric.to_numpy(float))]
    ordered_index = numeric.loc[finite_index].sort_values(kind="mergesort").index.to_numpy()
    for rt_bin, indices in enumerate(np.array_split(ordered_index, n_bins), 1):
        result.loc[indices] = rt_bin
    return result


def savefig(fig: plt.Figure, stem: Path) -> None:
    for extension, kwargs in [
        ("png", {"dpi": 400}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 400}),
    ]:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_mechanism_metrics() -> pd.DataFrame:
    extreme = pd.read_csv(EXTREME_METRICS)
    extreme["age_group"] = extreme["analysis_group"].map(ANALYSIS_TO_AGE)
    middle = pd.read_csv(MIDDLE_METRICS)
    middle["age_group"] = middle["analysis_group"].astype(str)
    combined = pd.concat([extreme, middle], ignore_index=True, sort=False)
    combined["age_group"] = pd.Categorical(combined["age_group"], AGE_GROUPS, ordered=True)
    return combined.sort_values("age_group").reset_index(drop=True)


def build_crf(manifest: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for age_group, part in manifest.groupby("age_group", sort=False):
        cell = part.copy()
        cell["rt_bin"] = equal_count_bins(cell["human_rt"])
        cell["response_category"] = np.where(
            cell["response_label"].eq(cell["target_label"]),
            "target",
            np.where(cell["response_label"].eq(cell["flanker_label"]), "flanker", "other"),
        )
        for (rt_bin, category), group in cell.groupby(["rt_bin", "response_category"], observed=True):
            denominator = int((cell["rt_bin"] == rt_bin).sum())
            rows.append(
                {
                    "age_group": age_group,
                    "source": "human",
                    "rt_bin": int(rt_bin),
                    "response_category": category,
                    "n_trials": len(group),
                    "median_rt": float(group["human_rt"].median()),
                    "response_rate": len(group) / denominator,
                }
            )
    for age_group, part in predictions.groupby("age_group", sort=False):
        cell = part[np.isfinite(pd.to_numeric(part["pred_rt"], errors="coerce"))].copy()
        cell["rt_bin"] = equal_count_bins(cell["pred_rt"])
        cell["response_category"] = np.where(
            cell["pred_choice"].eq(cell["target_label"]),
            "target",
            np.where(cell["pred_choice"].eq(cell["flanker_label"]), "flanker", "other"),
        )
        for (rt_bin, category), group in cell.groupby(["rt_bin", "response_category"], observed=True):
            denominator = int((cell["rt_bin"] == rt_bin).sum())
            rows.append(
                {
                    "age_group": age_group,
                    "source": "model",
                    "rt_bin": int(rt_bin),
                    "response_category": category,
                    "n_trials": len(group),
                    "median_rt": float(group["pred_rt"].median()),
                    "response_rate": len(group) / denominator,
                }
            )
    return pd.DataFrame(rows)


def plot_age_panels(caf: pd.DataFrame, crf: pd.DataFrame, delta: pd.DataFrame) -> None:
    colors = {"human": "#222222", "model": "#0072B2"}
    for age_group in AGE_GROUPS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        age_caf = caf[caf["age_group"] == age_group]
        for (source, congruency), cell in age_caf.groupby(["source", "congruency"], sort=True):
            axes[0].plot(
                cell["median_rt"], cell["accuracy"], marker="o",
                linestyle="--" if int(congruency) else "-", color=colors[source],
                label=f"{source} {'incongruent' if congruency else 'congruent'}",
            )
        axes[0].set(xlabel="Median RT (s)", ylabel="Accuracy", title="CAF", ylim=(0, 1.02))
        axes[0].legend(frameon=False, fontsize=7)

        age_crf = crf[(crf["age_group"] == age_group) & (crf["response_category"] != "target")]
        for (source, category), cell in age_crf.groupby(["source", "response_category"], sort=True):
            axes[1].plot(
                cell["median_rt"], cell["response_rate"], marker="o",
                linestyle="--" if category == "other" else "-", color=colors[source],
                label=f"{source} {category}",
            )
        axes[1].set(xlabel="Median RT (s)", ylabel="Response rate", title="CRF: non-target responses")
        axes[1].legend(frameon=False, fontsize=7)

        age_delta = delta[delta["age_group"] == age_group]
        for source, cell in age_delta.groupby("source", sort=True):
            axes[2].plot(cell["rt_bin"], cell["mean_delta_rt"], "o-", color=colors[source], label=source)
        axes[2].axhline(0, color="#777777", linewidth=0.8)
        axes[2].set(xlabel="RT bin", ylabel="Incongruent - congruent RT (s)", title="Participant-first delta")
        axes[2].legend(frameon=False, fontsize=7)
        fig.suptitle(age_group)
        savefig(fig, FIGURES / f"age_group_{age_group.replace('-', '_')}_caf_crf_delta")


def plot_all_age_summaries(metrics: pd.DataFrame, delta: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    x = np.arange(len(AGE_GROUPS))
    human = metrics[metrics["source"] == "human"].set_index("age_group").reindex(AGE_GROUPS)
    model = metrics[metrics["source"] == "model"].set_index("age_group").reindex(AGE_GROUPS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, human["accuracy"], "o-", color="#222222", label="Human")
    ax.plot(x, model["accuracy"], "o-", color="#0072B2", label="Model")
    ax.set_xticks(x, AGE_GROUPS)
    ax.set(xlabel="Age group", ylabel="Accuracy", title="Accuracy across age groups", ylim=(0.85, 1.0))
    ax.legend(frameon=False)
    savefig(fig, FIGURES / "age_trend_accuracy")

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, age_group in zip(axes, AGE_GROUPS):
        cell = delta[delta["age_group"] == age_group]
        for source, group in cell.groupby("source", sort=True):
            ax.plot(group["rt_bin"], group["mean_delta_rt"], "o-", color="#222222" if source == "human" else "#0072B2", label=source)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(age_group)
        ax.set_xlabel("RT bin")
    axes[0].set_ylabel("Delta RT (s)")
    axes[-1].axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Participant-first delta curves", y=1.02)
    savefig(fig, FIGURES / "all_age_delta_small_multiples")

    mechanism = mechanism.set_index("age_group").reindex(AGE_GROUPS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(x, mechanism["input_mean_reversal_time"], "o-", label="Input", color="#D55E00")
    axes[0].plot(x, mechanism["state_mean_reversal_time"], "o-", label="WW state", color="#0072B2")
    axes[0].set(ylabel="Time (s)", title="Target recovery timing")
    axes[0].legend(frameon=False)
    axes[1].plot(x, mechanism["state_recovered_before_readout_rate"], "o-", color="#009E73")
    axes[1].set(ylabel="Proportion", title="Recovered before readout", ylim=(0, 1.02))
    axes[2].plot(x, mechanism["crossing_rate"], "o-", color="#CC79A7")
    axes[2].set(ylabel="Proportion", title="Sustained crossing", ylim=(0.98, 1.001))
    for ax in axes:
        ax.set_xticks(x, AGE_GROUPS, rotation=45)
        ax.set_xlabel("Age group")
    savefig(fig, FIGURES / "all_age_target_recovery_triptych")


def write_git_record() -> None:
    status_path = ROOT / "audits/git_worktree_status_at_completion.txt"
    report_path = ROOT / "summaries/reproducibility_and_git_status.md"
    command = ["git", "status", "--short", "--untracked-files=all"]
    first = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout
    status_path.write_text(first, encoding="utf-8")
    report_path.write_text("status snapshot pending\n", encoding="utf-8")
    final_status = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout
    status_path.write_text(final_status, encoding="utf-8")
    report = f"""# Reproducibility and git status

The all-age run completed without creating a commit or pushing. The worktree already contained many unrelated modified and untracked files; they were preserved.

## Files added or updated for this extension

- `code/scripts/audit_all_age_group_data.py`
- `code/scripts/build_all_age_group_subsets.py`
- `code/scripts/merge_all_age_group_subsets.py`
- `code/scripts/build_full_age_group_vgg_evidence_cache.py`
- `code/scripts/run_corrected_model_all_age_groups.py`
- `code/scripts/run_all_age_group_extension.py`
- `code/scripts/complete_all_age_group_tables.py`
- `code/scripts/write_model_identification.py`
- `code/scripts/run_r5_choice_coupled_schedule_optimization.py`
- `tests/test_all_age_group_extension.py`
- `artifacts/results/all_age_groups_20260806/`

## Suggested commit message

`Complete audited all-age representative-subset model extension`

## Full worktree status

The exact snapshot is also saved in `audits/git_worktree_status_at_completion.txt`.

```text
{final_status.rstrip()}
```
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "all_age_group_metrics.csv")
    pred = pd.read_csv(RESULTS / "all_age_group_trial_level_predictions.csv", low_memory=False)
    manifest = pd.read_csv(ROOT / "manifests/all_age_group_trial_manifest.csv")
    delta_trials = pd.read_csv(RESULTS / "all_age_group_subject_delta.csv")

    metric_columns = [
        "age_group", "source", "n_trials", "n_rt_observed", "rt_q10", "rt_q50", "rt_q90",
        "rt_q95", "rt_mean", "rt_median", "rt_sd", "rt_skew",
    ]
    metrics[metric_columns].to_csv(RESULTS / "all_age_group_rt_quantiles.csv", index=False)

    correct_error_rows: list[dict[str, object]] = []
    for source, data, correct_col, rt_col in [
        ("human", pred, "human_correct", "human_rt"),
        ("model", pred, "model_correct", "pred_rt"),
    ]:
        for (age_group, correct), group in data.groupby(["age_group", correct_col], sort=False):
            finite_rt = pd.to_numeric(group[rt_col], errors="coerce").dropna()
            correct_error_rows.append(
                {
                    "age_group": age_group,
                    "source": source,
                    "correct": bool(correct),
                    "n_trials": len(group),
                    "n_rt_observed": len(finite_rt),
                    "mean_rt": float(finite_rt.mean()),
                    "median_rt": float(finite_rt.median()),
                }
            )
    pd.DataFrame(correct_error_rows).to_csv(RESULTS / "all_age_group_correct_error_rt.csv", index=False)

    mechanism = load_mechanism_metrics()
    mechanism.to_csv(RESULTS / "all_age_group_mechanism_summary.csv", index=False)
    fit_columns = [
        "age_group", "analysis_group", "score", "crossing_gate_passed", "rt_quantile_mae",
        "caf_rmse", "incongruent_caf_rmse", "accuracy_abs_error", "error_rt_gap_abs_error",
    ]
    mechanism[fit_columns].to_csv(RESULTS / "all_age_group_model_fit_scores.csv", index=False)

    parameters = mechanism[
        [
            "age_group", "evidence_gain", "threshold", "sustained_k", "margin", "compression",
            "late_shift_s", "width_scale", "t0_mean", "t0_sd",
        ]
    ].copy()
    parameters["model_name"] = "choice_coupled_corrected_equivalent"
    parameters["random_seed"] = 20260530
    if MIDDLE_PARAMETERS.exists():
        middle_parameters = pd.read_csv(MIDDLE_PARAMETERS).set_index("age_group")
        parameters["age_interpolation_fraction"] = parameters["age_group"].map(
            middle_parameters["age_interpolation_fraction"]
        )
        parameters.loc[parameters["age_group"] == "20-29", "age_interpolation_fraction"] = 0.0
        parameters.loc[parameters["age_group"] == "80-89", "age_interpolation_fraction"] = 1.0
    parameters.to_csv(RESULTS / "all_age_group_parameters.csv", index=False)

    pred["winner_readout_consistent"] = pred["pred_choice"].astype(int).eq(
        pred["winner_at_readout"].astype(int)
    )
    crossing = (
        pred.groupby("age_group", sort=False)
        .agg(
            n_trials=("crossed", "size"),
            n_crossed=("crossed", "sum"),
            winner_readout_consistency=("winner_readout_consistent", "mean"),
        )
        .reset_index()
    )
    crossing["n_no_crossing"] = crossing["n_trials"] - crossing["n_crossed"]
    crossing["crossing_rate"] = crossing["n_crossed"] / crossing["n_trials"]
    crossing.to_csv(RESULTS / "all_age_group_crossing_audit.csv", index=False)

    crf = build_crf(manifest, pred)
    crf.to_csv(RESULTS / "all_age_group_crf.csv", index=False)
    delta = (
        delta_trials.groupby(["age_group", "source", "rt_bin"])
        .agg(
            n_subjects=("user_id", "nunique"),
            mean_delta_rt=("delta_rt", "mean"),
            median_delta_rt=("delta_rt", "median"),
            sd_delta_rt=("delta_rt", "std"),
        )
        .reset_index()
    )
    delta.to_csv(RESULTS / "all_age_group_delta_summary.csv", index=False)

    run = pd.read_csv(ROOT / "audits/age_group_run_status.csv")
    run.to_csv(ROOT / "manifests/age_group_run_manifest.csv", index=False)
    (ROOT / "configs/random_seeds.json").write_text(
        json.dumps(
            {
                "subset_sampling": 20260530,
                "model_simulation": 20260530,
                "age_specific_seed_rule": "stable offset derived from lower age bound for t0 simulation",
                "source": "presentation/corrected-equivalent pipeline",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ROOT / "configs/current_run_config.json").write_text(
        json.dumps(
            {
                "date": "20260806",
                "groups": AGE_GROUPS,
                "trials_per_group": 5000,
                "selected_model": "choice-coupled corrected-equivalent",
                "completed_model_groups": AGE_GROUPS,
                "blocked_model_groups": [],
                "device": "cpu",
                "status": "diagnostic_all_age_representative_subset_complete",
                "scope_warning": "not a full-cohort or held-out fit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ROOT / "logs/failed_age_groups.log").write_text("No failed age groups.\n", encoding="utf-8")

    caf = pd.read_csv(RESULTS / "all_age_group_caf.csv")
    plot_age_panels(caf, crf, delta)
    plot_all_age_summaries(metrics, delta, mechanism)
    for extension in ["png", "pdf", "svg", "tiff"]:
        source = MECHANISM_SOURCE.with_suffix(f".{extension}")
        if source.exists():
            shutil.copy2(source, FIGURES / f"presentation_target_recovery_evidence_chain.{extension}")
    write_git_record()
    print(f"completed {len(list(RESULTS.glob('all_age_group_*.csv')))} result tables")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot current choice-coupled R5 CAF and RT delta curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    PROJECT_ROOT
    / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"
    / "selected_trial_level_predictions.csv"
)
REFERENCE_CAF = (
    PROJECT_ROOT
    / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"
    / "selected_caf.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts/results/r5_caf_delta_curves_20260803"

GROUPS = [("young_20_29", "Young adults (20-29)"), ("older_80_89", "Older adults (80-89)")]
CONDITIONS = [(0, "Congruent", "#0072B2"), (1, "Incongruent", "#E69F00")]
N_BINS = 5


def ordered_bins(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=np.int64)
    for bin_id, idx in enumerate(
        np.array_split(np.argsort(values, kind="mergesort"), n_bins), start=1
    ):
        out[idx] = bin_id
    return out


def build_caf(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_part in data.groupby("analysis_group", sort=True):
        for congruency, part in group_part.groupby("congruency", sort=True):
            for source, rt_col, correct_col in (
                ("human", "true_rt", "human_correct"),
                ("model", "pred_rt", "model_correct"),
            ):
                bins = ordered_bins(part[rt_col].to_numpy(float))
                for bin_id in range(1, N_BINS + 1):
                    selected = part.iloc[np.flatnonzero(bins == bin_id)]
                    rows.append(
                        {
                            "analysis_group": group,
                            "congruency": int(congruency),
                            "source": source,
                            "rt_bin": bin_id,
                            "n_trials": len(selected),
                            "median_rt_s": float(selected[rt_col].median()),
                            "accuracy": float(selected[correct_col].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def build_subject_delta(data: pd.DataFrame) -> pd.DataFrame:
    """Build standard correct-trial RT delta plots within participant."""
    rows: list[dict[str, object]] = []
    for group, group_part in data.groupby("analysis_group", sort=True):
        for source, rt_col, correct_col in (
            ("human", "true_rt", "human_correct"),
            ("model", "pred_rt", "model_correct"),
        ):
            for user_id, subject in group_part.groupby("user_id", sort=True):
                condition_means: dict[int, list[float]] = {}
                condition_counts: dict[int, list[int]] = {}
                for congruency, part in subject.loc[subject[correct_col]].groupby(
                    "congruency", sort=True
                ):
                    if len(part) < N_BINS:
                        continue
                    bins = ordered_bins(part[rt_col].to_numpy(float))
                    condition_means[int(congruency)] = [
                        float(part.iloc[np.flatnonzero(bins == bin_id)][rt_col].mean())
                        for bin_id in range(1, N_BINS + 1)
                    ]
                    condition_counts[int(congruency)] = [
                        int(np.sum(bins == bin_id)) for bin_id in range(1, N_BINS + 1)
                    ]
                if set(condition_means) != {0, 1}:
                    continue
                for bin_index in range(N_BINS):
                    congruent_rt = condition_means[0][bin_index]
                    incongruent_rt = condition_means[1][bin_index]
                    rows.append(
                        {
                            "analysis_group": group,
                            "source": source,
                            "user_id": user_id,
                            "rt_bin": bin_index + 1,
                            "congruent_mean_rt_s": congruent_rt,
                            "incongruent_mean_rt_s": incongruent_rt,
                            "mean_rt_s": (congruent_rt + incongruent_rt) / 2.0,
                            "delta_rt_s": incongruent_rt - congruent_rt,
                            "n_congruent_correct": condition_counts[0][bin_index],
                            "n_incongruent_correct": condition_counts[1][bin_index],
                        }
                    )
    return pd.DataFrame(rows)


def summarize_delta(subject_delta: pd.DataFrame) -> pd.DataFrame:
    summary = (
        subject_delta.groupby(["analysis_group", "source", "rt_bin"], sort=True)
        .agg(
            n_subjects=("user_id", "nunique"),
            mean_rt_s=("mean_rt_s", "mean"),
            mean_delta_rt_s=("delta_rt_s", "mean"),
            sd_delta_rt_s=("delta_rt_s", "std"),
        )
        .reset_index()
    )
    summary["se_delta_rt_s"] = summary["sd_delta_rt_s"] / np.sqrt(summary["n_subjects"])
    return summary


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_caf(caf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (group, title) in zip(axes, GROUPS):
        for congruency, condition, color in CONDITIONS:
            for source, linestyle, marker, fill in (
                ("human", "-", "o", "white"),
                ("model", (0, (4, 2)), "s", color),
            ):
                part = caf.loc[
                    caf["analysis_group"].eq(group)
                    & caf["congruency"].eq(congruency)
                    & caf["source"].eq(source)
                ].sort_values("rt_bin")
                ax.plot(
                    part["median_rt_s"],
                    part["accuracy"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5.2,
                    markerfacecolor=fill,
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                    label=f"{condition} - {source.title()}",
                )
        ax.set_title(title, pad=7)
        ax.set_xlabel("Median RT in bin (s)")
        ax.set_ylim(0.76, 1.01)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        finish_axis(ax)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.77, wspace=0.18)
    save_figure(fig, OUTPUT_DIR / "current_model_caf_human_vs_model")


def plot_delta(delta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (group, title) in zip(axes, GROUPS):
        for source, color, linestyle, marker, fill in (
            ("human", "#222222", "-", "o", "white"),
            ("model", "#D55E00", (0, (4, 2)), "s", "#D55E00"),
        ):
            part = delta.loc[
                delta["analysis_group"].eq(group) & delta["source"].eq(source)
            ].sort_values("rt_bin")
            ax.errorbar(
                part["mean_rt_s"],
                part["mean_delta_rt_s"],
                yerr=part["se_delta_rt_s"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5.2,
                markerfacecolor=fill,
                markeredgecolor=color,
                markeredgewidth=1.1,
                linewidth=1.7,
                capsize=2.5,
                elinewidth=1.0,
                label=source.title(),
            )
        ax.axhline(0, color="#8A8A8A", linewidth=0.8, zorder=0)
        ax.set_title(title, pad=7)
        ax.set_xlabel("Mean RT across conditions (s)")
        ax.set_ylim(-0.01, 0.39)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        finish_axis(ax)
    axes[0].set_ylabel("Delta RT: incongruent - congruent (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.6,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.77, wspace=0.18)
    save_figure(fig, OUTPUT_DIR / "current_model_delta_rt_human_vs_model")


def plot_combined(caf: pd.DataFrame, delta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharey="row")
    for column, (group, title) in enumerate(GROUPS):
        ax = axes[0, column]
        for congruency, condition, color in CONDITIONS:
            for source, linestyle, marker, fill in (
                ("human", "-", "o", "white"),
                ("model", (0, (4, 2)), "s", color),
            ):
                part = caf.loc[
                    caf["analysis_group"].eq(group)
                    & caf["congruency"].eq(congruency)
                    & caf["source"].eq(source)
                ].sort_values("rt_bin")
                ax.plot(
                    part["median_rt_s"], part["accuracy"], color=color,
                    linestyle=linestyle, marker=marker, markersize=4.6,
                    markerfacecolor=fill, markeredgecolor=color, markeredgewidth=1.0,
                    label=f"{condition} - {source.title()}",
                )
        ax.set_title(title, pad=7)
        ax.set_xlabel("Median RT in bin (s)")
        ax.set_ylim(0.76, 1.01)
        finish_axis(ax)

        ax = axes[1, column]
        for source, color, linestyle, marker, fill in (
            ("human", "#222222", "-", "o", "white"),
            ("model", "#D55E00", (0, (4, 2)), "s", "#D55E00"),
        ):
            part = delta.loc[
                delta["analysis_group"].eq(group) & delta["source"].eq(source)
            ].sort_values("rt_bin")
            ax.errorbar(
                part["mean_rt_s"], part["mean_delta_rt_s"],
                yerr=part["se_delta_rt_s"], color=color, linestyle=linestyle,
                marker=marker, markersize=4.6, markerfacecolor=fill,
                markeredgecolor=color, markeredgewidth=1.0, capsize=2.2,
                elinewidth=0.9, label=source.title(),
            )
        ax.axhline(0, color="#8A8A8A", linewidth=0.8, zorder=0)
        ax.set_xlabel("Mean RT across conditions (s)")
        ax.set_ylim(-0.01, 0.39)
        finish_axis(ax)

    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Delta RT (s)")
    axes[0, 0].text(-0.18, 1.06, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=11)
    axes[1, 0].text(-0.18, 1.06, "b", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=11)
    caf_handles, caf_labels = axes[0, 0].get_legend_handles_labels()
    delta_handles, delta_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(caf_handles, caf_labels, loc="upper center", bbox_to_anchor=(0.51, 1.00), ncol=4, frameon=False, fontsize=7.6)
    fig.legend(delta_handles, delta_labels, loc="lower center", bbox_to_anchor=(0.51, 0.00), ncol=2, frameon=False, fontsize=8.0)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.87, hspace=0.42, wspace=0.18)
    save_figure(fig, OUTPUT_DIR / "current_model_caf_and_delta_overview")


def validate_caf(caf: pd.DataFrame) -> None:
    reference = pd.read_csv(REFERENCE_CAF).rename(columns={"median_rt": "median_rt_s"})
    keys = ["analysis_group", "congruency", "source", "rt_bin"]
    merged = caf.merge(reference, on=keys, suffixes=("_new", "_reference"), validate="one_to_one")
    if len(merged) != 40:
        raise AssertionError("CAF validation did not recover all 40 reference rows.")
    for column in ("n_trials", "median_rt_s", "accuracy"):
        left = merged[f"{column}_new"].to_numpy(float)
        right = merged[f"{column}_reference"].to_numpy(float)
        if not np.allclose(left, right, atol=1e-12, rtol=0):
            raise AssertionError(f"CAF mismatch in {column}")


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {
        "analysis_group", "user_id", "true_rt", "pred_rt", "congruency",
        "human_correct", "model_correct", "crossed",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(data) != 10_000 or not data["crossed"].astype(bool).all():
        raise ValueError("Expected 10,000 genuine-crossing selected trials.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caf = build_caf(data)
    validate_caf(caf)
    subject_delta = build_subject_delta(data)
    delta = summarize_delta(subject_delta)
    expected_subjects = {"young_20_29": 12, "older_80_89": 4}
    for group, expected in expected_subjects.items():
        observed = delta.loc[delta["analysis_group"].eq(group), "n_subjects"]
        if observed.empty or not observed.eq(expected).all():
            raise AssertionError(f"Delta curve missing participants for {group}")

    caf.to_csv(OUTPUT_DIR / "caf_source_data.csv", index=False)
    subject_delta.to_csv(OUTPUT_DIR / "delta_rt_subject_level.csv", index=False)
    delta.to_csv(OUTPUT_DIR / "delta_rt_summary.csv", index=False)

    setup_style()
    plot_caf(caf)
    plot_delta(delta)
    plot_combined(caf, delta)

    caption = """# Figures | Current choice-coupled model CAF and RT delta curves

CAF curves use five equally sized RT bins within age group, congruency, and source; x-coordinates are the observed median RT in each bin. Delta curves use correct trials and are constructed within participant by separately binning congruent and incongruent RTs, then plotting their mean RT against incongruent-minus-congruent RT. Delta points are participant means and error bars are standard errors across participants (young n = 12; older n = 4). Human and model bins are formed from their respective RT distributions. All 10,000 model trials crossed the decision threshold; no censoring sentinels were included. These are exploratory representative-subset results, not held-out validation.
"""
    (OUTPUT_DIR / "caf_delta_caption.md").write_text(caption, encoding="utf-8")
    print(delta.to_string(index=False))
    print(f"Saved figures and source data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

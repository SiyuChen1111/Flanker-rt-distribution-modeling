#!/usr/bin/env python3
"""Plot integrated human/model RT distributions for all seven age groups."""
from __future__ import annotations

import argparse
import os
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
INPUT = ROOT / "results/all_age_group_trial_level_predictions.csv"
FIGURE_DIR = ROOT / "figures_publication"
RESULT_DIR = ROOT / "results"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
CONDITIONS = [(0, "Congruent", "#0072B2"), (1, "Incongruent", "#E69F00")]
SOURCES = [("human_rt", "Human", "-"), ("pred_rt", "Model", (0, (4, 2)))]
X_MIN = 0.10
X_MAX = 2.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--stem", default="all_age_rt_distribution_small_multiples")
    parser.add_argument("--source-name", default="all_age_group_rt_distribution_kde_source.csv")
    parser.add_argument("--summary-name", default="all_age_group_rt_distribution_summary.csv")
    parser.add_argument("--title", default="Reaction-time distributions by age group")
    return parser.parse_args()


def robust_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    scale = min(float(np.std(values, ddof=1)), float(iqr / 1.34))
    return max(0.9 * scale * values.size ** (-0.2), 0.012)


def kde(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))


def savefig(fig: plt.Figure, stem: Path) -> None:
    for extension, kwargs in [
        ("png", {"dpi": 400}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 400}),
    ]:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    figure_dir = Path(args.figure_dir)
    result_dir = Path(args.result_dir)
    data = pd.read_csv(input_path, low_memory=False)
    required = {"age_group", "human_rt", "pred_rt", "congruency", "crossed"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(data["age_group"].astype(str)) != set(AGE_GROUPS):
        raise ValueError("The unified prediction file does not contain all seven age groups")
    if data.loc[~data["crossed"].astype(bool), "pred_rt"].notna().any():
        raise ValueError("No-crossing trials must have missing model RT before plotting")

    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(X_MIN, X_MAX, 700)
    source_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, int, str], np.ndarray] = {}

    for age_group in AGE_GROUPS:
        age_data = data[data["age_group"].astype(str) == age_group]
        pooled = np.concatenate(
            [
                pd.to_numeric(age_data["human_rt"], errors="coerce").dropna().to_numpy(float),
                pd.to_numeric(age_data["pred_rt"], errors="coerce").dropna().to_numpy(float),
            ]
        )
        bandwidth = robust_bandwidth(pooled)
        for congruency, condition, _ in CONDITIONS:
            condition_data = age_data[age_data["congruency"].astype(int) == congruency]
            for column, source, _ in SOURCES:
                values = pd.to_numeric(condition_data[column], errors="coerce").dropna().to_numpy(float)
                if not len(values) or (values <= 0).any():
                    raise ValueError(f"Invalid RT values for {age_group}, {condition}, {source}")
                density = kde(values, grid, bandwidth)
                curves[(age_group, congruency, source)] = density
                source_rows.extend(
                    {
                        "age_group": age_group,
                        "congruency": congruency,
                        "condition": condition,
                        "source": source,
                        "rt_s": float(rt),
                        "density": float(value),
                        "bandwidth_s": bandwidth,
                    }
                    for rt, value in zip(grid, density)
                )
                summary_rows.append(
                    {
                        "age_group": age_group,
                        "congruency": congruency,
                        "condition": condition,
                        "source": source,
                        "n_trials": len(values),
                        "mean_rt_s": float(np.mean(values)),
                        "median_rt_s": float(np.median(values)),
                        "sd_rt_s": float(np.std(values, ddof=1)),
                        "bandwidth_s": bandwidth,
                        "fraction_within_display": float(np.mean((values >= X_MIN) & (values <= X_MAX))),
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.8), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    y_max = max(float(density.max()) for density in curves.values()) * 1.06
    legend_handles = []
    legend_labels = []
    for ax, age_group in zip(axes, AGE_GROUPS):
        for congruency, condition, color in CONDITIONS:
            for _, source, linestyle in SOURCES:
                line = ax.plot(
                    grid,
                    curves[(age_group, congruency, source)],
                    color=color,
                    linestyle=linestyle,
                    label=f"{condition} - {source}",
                )[0]
                if age_group == AGE_GROUPS[0]:
                    legend_handles.append(line)
                    legend_labels.append(f"{condition} - {source}")
        ax.set_title(age_group, pad=5)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8])
        ax.tick_params(direction="in", length=3, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[4:7]:
        ax.set_xlabel("Reaction time (s)")
    axes[0].set_ylabel("Density")
    axes[4].set_ylabel("Density")
    axes[-1].axis("off")
    axes[-1].legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=False,
        handlelength=3.0,
        title="Condition and source",
    )
    fig.suptitle(args.title, fontsize=13)
    savefig(fig, figure_dir / args.stem)

    source = pd.DataFrame(source_rows)
    summary = pd.DataFrame(summary_rows)
    source.to_csv(result_dir / args.source_name, index=False)
    summary.to_csv(result_dir / args.summary_name, index=False)
    min_coverage = float(summary["fraction_within_display"].min())
    caption = f"""# Figure | Reaction-time distributions by age group

Kernel density estimates compare human reaction times (solid) with model predictions (dashed) for congruent (blue) and incongruent (orange) trials. Every panel uses the same axes and an age-specific bandwidth shared across all four curves. Densities use every finite reaction time; the visible range is {X_MIN:.1f}-{X_MAX:.1f} s and contains at least {min_coverage:.1%} of every series. The single no-crossing model trial has no reaction time and is excluded.
"""
    (figure_dir / f"{args.stem}_caption.md").write_text(
        caption, encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved integrated figure to {figure_dir}")


if __name__ == "__main__":
    main()

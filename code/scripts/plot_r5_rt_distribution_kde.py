#!/usr/bin/env python3
"""Plot observed and choice-coupled R5 reaction-time distributions."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"
    / "selected_trial_level_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/results/r5_rt_distribution_kde_20260803"

GROUPS = [("young_20_29", "Young adults (20–29)"), ("older_80_89", "Older adults (80–89)")]
CONDITIONS = [(0, "Congruent", "#0072B2"), (1, "Incongruent", "#E69F00")]
X_MIN, X_MAX = 0.10, 2.00


def robust_bandwidth(values: np.ndarray) -> float:
    """Return a robust Silverman bandwidth in seconds."""
    values = np.asarray(values, dtype=float)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    scale = min(float(np.std(values, ddof=1)), float(iqr / 1.34))
    return max(0.9 * scale * values.size ** (-0.2), 0.012)


def kde(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Evaluate a Gaussian KDE with an absolute bandwidth in seconds."""
    values = np.asarray(values, dtype=float)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    data = pd.read_csv(input_path)
    required = {"analysis_group", "true_rt", "pred_rt", "congruency", "crossed"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not data["crossed"].astype(bool).all():
        raise ValueError("KDE input contains no-crossing trials; censoring sentinels must not be plotted.")
    if not np.isfinite(data[["true_rt", "pred_rt"]].to_numpy()).all():
        raise ValueError("RT columns contain missing or non-finite values.")
    if (data[["true_rt", "pred_rt"]] <= 0).any().any():
        raise ValueError("RT columns contain non-positive values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(X_MIN, X_MAX, 900)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharex=True, sharey=False)
    source_rows: list[dict[str, float | str | int]] = []
    summary_rows: list[dict[str, float | str | int]] = []

    for ax, (group, title) in zip(axes, GROUPS):
        group_data = data.loc[data["analysis_group"].eq(group)].copy()
        bandwidth = robust_bandwidth(
            np.concatenate([group_data["true_rt"].to_numpy(), group_data["pred_rt"].to_numpy()])
        )

        for congruency, condition, color in CONDITIONS:
            part = group_data.loc[group_data["congruency"].eq(congruency)]
            for column, source, linestyle in (
                ("true_rt", "Human", "-"),
                ("pred_rt", "Model", (0, (4, 2))),
            ):
                values = part[column].to_numpy(float)
                density = kde(values, grid, bandwidth)
                ax.plot(
                    grid,
                    density,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{condition} — {source}",
                )
                source_rows.extend(
                    {
                        "analysis_group": group,
                        "congruency": congruency,
                        "condition": condition,
                        "source": source,
                        "rt_s": float(x),
                        "density": float(y),
                        "bandwidth_s": bandwidth,
                    }
                    for x, y in zip(grid, density)
                )
                summary_rows.append(
                    {
                        "analysis_group": group,
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

        ax.set_title(title, pad=7)
        ax.set_xlabel("Reaction time (s)")
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8])
        ax.tick_params(direction="in", length=3, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.78, wspace=0.27)

    stem = output_dir / "observed_vs_model_rt_kde"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(source_rows).to_csv(output_dir / "observed_vs_model_rt_kde_source_data.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "observed_vs_model_rt_summary.csv", index=False)

    min_coverage = summary["fraction_within_display"].min()
    caption = (
        "# Figure | Observed and model-predicted RT distributions\n\n"
        "Kernel density estimates compare observed human RTs (solid) with choice-coupled "
        "model predictions (dashed) for congruent (blue) and incongruent (orange) trials. "
        "Panels show young and older adults. KDE bandwidths were estimated robustly within "
        "each age group and held constant across conditions and sources. All trials crossed "
        "the decision threshold; no censoring sentinels were included. Densities were computed "
        f"from all RTs, while the display is limited to {X_MIN:.1f}–{X_MAX:.1f} s; every plotted "
        f"series retains at least {min_coverage:.1%} of observations within this window."
    )
    (output_dir / "observed_vs_model_rt_kde_caption.md").write_text(caption, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Saved figure and source data to {output_dir}")


if __name__ == "__main__":
    main()

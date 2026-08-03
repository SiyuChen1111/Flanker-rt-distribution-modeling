#!/usr/bin/env python3
"""Audit target-versus-flanker dynamics in the retained R5 model.

This analysis separates three stages that are easy to conflate:
1. VGG layerwise logits;
2. the normalized layer-to-time evidence actually sent to Wong-Wang;
3. the resulting Wong-Wang state trajectory and R5 readout.

The primary question is whether real incongruent stimuli already contain an
early-flanker / late-target reversal, and whether model errors reflect failure
to preserve that recovery through accumulation and readout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vam-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import normalize_layers  # noqa: E402
from run_r5_supervisor_followup import reconstruct_r5  # noqa: E402


ROOT = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
R5_RESULTS = ROOT / "best_model_R5_combined_best/results"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/results/r5_real_vgg_target_flanker_audit_20260803"
LAYERS = ["conv3", "conv4", "conv5", "pooled", "final"]
EVIDENCE_KEYS = {layer: f"evidence_{layer}" for layer in LAYERS}
DT_MS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="Allow replacing files in an existing output directory.")
    return parser.parse_args()


def channel_gap(values: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    """Return target-minus-flanker values for [trial,class] or [trial,time,class]."""
    values = np.asarray(values)
    target = np.asarray(target, dtype=np.int64)
    flanker = np.asarray(flanker, dtype=np.int64)
    rows = np.arange(values.shape[0])
    if values.ndim == 2:
        return values[rows, target] - values[rows, flanker]
    if values.ndim == 3:
        times = np.arange(values.shape[1])
        return (
            values[rows[:, None], times[None, :], target[:, None]]
            - values[rows[:, None], times[None, :], flanker[:, None]]
        )
    raise ValueError(f"Expected 2D or 3D values, got {values.shape}")


def first_stable_recovery(gap: np.ndarray, sustained_k: int = 2) -> np.ndarray:
    """First sustained positive step after the trajectory has previously favored flanker."""
    gap = np.asarray(gap, dtype=np.float64)
    if gap.ndim != 2:
        raise ValueError("gap must have shape [trial,time]")
    n, t_steps = gap.shape
    first = np.full(n, np.nan, dtype=np.float64)
    seen_negative = np.zeros(n, dtype=bool)
    for t in range(t_steps):
        seen_negative |= gap[:, t] < 0
        if t + sustained_k > t_steps:
            continue
        stable = np.all(gap[:, t : t + sustained_k] > 0, axis=1)
        take = np.isnan(first) & seen_negative & stable
        first[take] = float(t)
    return first


def temporal_pattern(early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """Classify the sign pattern across early and late windows."""
    early = np.asarray(early)
    late = np.asarray(late)
    out = np.full(early.shape, "mixed_or_neutral", dtype=object)
    out[(early < 0) & (late > 0)] = "early_flanker_late_target"
    out[(early >= 0) & (late >= 0)] = "target_throughout"
    out[(early <= 0) & (late <= 0)] = "flanker_throughout"
    out[(early > 0) & (late < 0)] = "early_target_late_flanker"
    return out.astype(str)


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    ci = stats.binomtest(int(successes), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan
    sem = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else math.nan
    return float(x.mean()), sem


def add_groupwise_quintile(df: pd.DataFrame, value: str, output: str) -> None:
    df[output] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for _, idx in df.groupby("analysis_group", sort=True).groups.items():
        idx = pd.Index(idx)
        ranked = df.loc[idx, value].rank(method="first")
        df.loc[idx, output] = pd.qcut(ranked, 5, labels=False).astype(int).to_numpy() + 1


def load_raw_layers(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    evidence_path = ROOT / "evidence_cache/representative_subset_layerwise_evidence.npz"
    z = np.load(evidence_path, allow_pickle=True)
    positions = {int(sid): i for i, sid in enumerate(z["subset_stimulus_id"].astype(np.int64))}
    idx = np.asarray([positions[int(sid)] for sid in df["subset_stimulus_id"].to_numpy()], dtype=np.int64)
    raw = {layer: np.asarray(z[key][idx], dtype=np.float32) for layer, key in EVIDENCE_KEYS.items()}

    normalized = {layer: np.empty_like(raw[layer]) for layer in LAYERS}
    for age, age_idx in df.groupby("analysis_group", sort=True).groups.items():
        age_idx = np.asarray(list(age_idx), dtype=np.int64)
        part = normalize_layers({layer: raw[layer][age_idx] for layer in LAYERS}, "per_layer_gap_scale")
        for layer in LAYERS:
            normalized[layer][age_idx] = part[layer]
    return raw, normalized


def outcome_labels(choice: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    return np.where(choice == target, "target", np.where(choice == flanker, "flanker", "other"))


def layer_summary(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {"all": np.ones(len(trial), dtype=bool)}
    for outcome in ["target", "flanker", "other"]:
        groups[f"model_{outcome}"] = trial["model_outcome"].eq(outcome).to_numpy()
    for group, mask in groups.items():
        for layer in LAYERS:
            raw = trial.loc[mask, f"raw_gap_{layer}"].to_numpy(float)
            norm = trial.loc[mask, f"normalized_gap_{layer}"].to_numpy(float)
            mean_raw, sem_raw = mean_sem(raw)
            mean_norm, sem_norm = mean_sem(norm)
            rows.append(
                {
                    "group": group,
                    "layer": layer,
                    "n_trials": int(mask.sum()),
                    "raw_gap_mean": mean_raw,
                    "raw_gap_sem": sem_raw,
                    "normalized_gap_mean": mean_norm,
                    "normalized_gap_sem": sem_norm,
                    "target_gt_flanker_rate": float(np.mean(raw > 0)) if raw.size else math.nan,
                    "flanker_gt_target_rate": float(np.mean(raw < 0)) if raw.size else math.nan,
                }
            )
    return pd.DataFrame(rows)


def outcome_summary(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = [("all", "all", trial.index)]
    dimensions += [("model_outcome", str(k), idx) for k, idx in trial.groupby("model_outcome").groups.items()]
    dimensions += [("analysis_group", str(k), idx) for k, idx in trial.groupby("analysis_group").groups.items()]
    for dimension, group, idx in dimensions:
        part = trial.loc[idx]
        n = len(part)
        for metric in ["raw_reversal", "input_reversal", "state_reversal", "state_recovered_before_readout", "readout_before_state_recovery"]:
            success = int(part[metric].sum())
            lo, hi = wilson_interval(success, n)
            rate = success / n if n else math.nan
            rows.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n_trials": n,
                    "metric": metric,
                    "rate": rate,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "input_early_gap_mean": float(part["input_early_gap"].mean()),
                    "input_late_gap_mean": float(part["input_late_gap"].mean()),
                    "state_early_gap_mean": float(part["state_early_gap"].mean()),
                    "state_late_gap_mean": float(part["state_late_gap"].mean()),
                    "state_recovery_time_mean_s": float(part["state_recovery_time_s"].mean()),
                    "readout_time_mean_s": float(part["readout_time_s"].mean()),
                }
            )
    return pd.DataFrame(rows)


def timecourse_summary(trial: pd.DataFrame, input_gap: np.ndarray, state_gap: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, np.ndarray]] = [("all", "all", np.ones(len(trial), dtype=bool))]
    groups += [
        ("model_outcome", outcome, trial["model_outcome"].eq(outcome).to_numpy())
        for outcome in ["target", "flanker", "other"]
    ]
    groups += [
        ("model_rt_bin", f"Q{q}", trial["model_rt_bin"].eq(q).to_numpy())
        for q in range(1, 6)
    ]
    groups += [
        ("human_outcome", outcome, trial["human_outcome"].eq(outcome).to_numpy())
        for outcome in ["target", "flanker", "other"]
    ]
    groups += [
        ("human_rt_bin", f"Q{q}", trial["human_rt_bin"].eq(q).to_numpy())
        for q in range(1, 6)
    ]
    for source, gap in [("effective_input", input_gap), ("ww_state", state_gap)]:
        for dimension, group, mask in groups:
            if not mask.any():
                continue
            mean = gap[mask].mean(axis=0)
            sem = gap[mask].std(axis=0, ddof=1) / np.sqrt(mask.sum()) if mask.sum() > 1 else np.full(gap.shape[1], np.nan)
            for t, (m, se) in enumerate(zip(mean, sem)):
                rows.append(
                    {
                        "source": source,
                        "dimension": dimension,
                        "group": group,
                        "n_trials": int(mask.sum()),
                        "time_step": t,
                        "time_s": t * DT_MS / 1000.0,
                        "target_minus_flanker_mean": float(m),
                        "target_minus_flanker_sem": float(se),
                    }
                )
    return pd.DataFrame(rows)


def rt_bin_summary(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for age, age_part in trial.groupby("analysis_group", sort=True):
        for q, part in age_part.groupby("model_rt_bin", sort=True):
            rows.append(
                {
                    "analysis_group": age,
                    "model_rt_bin": int(q),
                    "n_trials": len(part),
                    "median_model_rt_s": float(part["model_rt"].median()),
                    "model_accuracy": float(part["model_correct"].mean()),
                    "flanker_error_rate": float(part["model_outcome"].eq("flanker").mean()),
                    "input_early_gap_mean": float(part["input_early_gap"].mean()),
                    "input_late_gap_mean": float(part["input_late_gap"].mean()),
                    "state_early_gap_mean": float(part["state_early_gap"].mean()),
                    "state_late_gap_mean": float(part["state_late_gap"].mean()),
                    "state_recovery_time_mean_s": float(part["state_recovery_time_s"].mean()),
                    "readout_before_state_recovery_rate": float(part["readout_before_state_recovery"].mean()),
                }
            )
    return pd.DataFrame(rows)


def human_rt_bin_summary(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for age, age_part in trial.groupby("analysis_group", sort=True):
        for q, part in age_part.groupby("human_rt_bin", sort=True):
            rows.append(
                {
                    "analysis_group": age,
                    "human_rt_bin": int(q),
                    "n_trials": len(part),
                    "median_human_rt_s": float(part["human_rt"].median()),
                    "human_accuracy": float(part["human_correct"].mean()),
                    "human_flanker_error_rate": float(part["human_outcome"].eq("flanker").mean()),
                    "input_early_gap_mean": float(part["input_early_gap"].mean()),
                    "input_late_gap_mean": float(part["input_late_gap"].mean()),
                    "state_early_gap_mean": float(part["state_early_gap"].mean()),
                    "state_late_gap_mean": float(part["state_late_gap"].mean()),
                    "state_recovery_time_mean_s": float(part["state_recovery_time_s"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_layer_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    specs = [("all", "All incongruent", "#333333"), ("model_target", "Model target", "#0072B2"), ("model_flanker", "Model flanker error", "#D55E00")]
    for group, label, color in specs:
        part = summary[summary["group"].eq(group)].set_index("layer").reindex(LAYERS)
        if part["n_trials"].fillna(0).max() <= 0:
            continue
        axes[0].plot(LAYERS, part["target_gt_flanker_rate"], marker="o", color=color, label=label)
        axes[1].plot(LAYERS, part["normalized_gap_mean"], marker="o", color=color, label=label)
    axes[0].axhline(0.5, color="0.6", lw=0.8, ls="--")
    axes[0].set(title="Layerwise target advantage rate", xlabel="VGG layer", ylabel="P(target evidence > flanker)", ylim=(-0.03, 1.03))
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set(title="Normalized layerwise evidence gap", xlabel="VGG layer", ylabel="Target - flanker evidence")
    axes[0].legend(frameon=False)
    save_figure(fig, output_dir, "01_real_vgg_layerwise_target_flanker")


def plot_timecourses(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    specs = [("all", "All incongruent", "#333333"), ("target", "Model target", "#0072B2"), ("flanker", "Model flanker error", "#D55E00")]
    for ax, source, title in zip(axes, ["effective_input", "ww_state"], ["Evidence sent to Wong-Wang", "Wong-Wang state"]):
        for group, label, color in specs:
            dimension = "all" if group == "all" else "model_outcome"
            part = summary[(summary["source"].eq(source)) & (summary["dimension"].eq(dimension)) & (summary["group"].eq(group))]
            if part.empty:
                continue
            ax.plot(part["time_s"], part["target_minus_flanker_mean"], color=color, label=label)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(title=title, xlabel="Decision time (s)", ylabel="Target - flanker")
    axes[0].legend(frameon=False)
    save_figure(fig, output_dir, "02_effective_evidence_and_state_dynamics")


def plot_natural_emergence_chain(trial: pd.DataFrame, timecourse: pd.DataFrame, output_dir: Path) -> None:
    """Publication-ready evidence chain from VGG layers to WW state."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.35], wspace=0.42)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_input = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[0, 2])

    layer_cols = [f"normalized_gap_{layer}" for layer in LAYERS]
    matrix = trial[layer_cols].to_numpy(float)
    sort_score = matrix[:, 1:4].mean(axis=1)
    matrix = matrix[np.argsort(sort_score, kind="mergesort")]
    limit = float(np.quantile(np.abs(matrix), 0.995))
    image = ax_heat.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True,
    )
    ax_heat.set_xticks(np.arange(len(LAYERS)), LAYERS, rotation=35, ha="right")
    ax_heat.set_yticks([0, len(matrix) - 1], ["1", f"{len(matrix):,}"])
    ax_heat.set(ylabel="Incongruent trials", title="Layerwise evidence")
    ax_heat.spines["right"].set_visible(True)
    ax_heat.spines["top"].set_visible(True)
    cbar = fig.colorbar(image, ax=ax_heat, orientation="horizontal", pad=0.23, fraction=0.07)
    cbar.set_label("Target − flanker", labelpad=2)
    cbar.ax.tick_params(labelsize=6)

    colors = {"effective_input": "#3B78A8", "ww_state": "#C45A32"}
    zero_times: dict[str, float] = {}
    for ax, source, title in [
        (ax_input, "effective_input", "Evidence sent to WW"),
        (ax_state, "ww_state", "WW state"),
    ]:
        part = timecourse[
            timecourse["source"].eq(source)
            & timecourse["dimension"].eq("all")
            & timecourse["group"].eq("all")
        ].sort_values("time_s")
        x = part["time_s"].to_numpy(float)
        mean = part["target_minus_flanker_mean"].to_numpy(float)
        sem = part["target_minus_flanker_sem"].to_numpy(float)
        color = colors[source]
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.6)
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        crossing = np.flatnonzero((mean[:-1] <= 0) & (mean[1:] > 0))
        if crossing.size:
            i = int(crossing[0])
            fraction = -mean[i] / (mean[i + 1] - mean[i])
            zero_time = float(x[i] + fraction * (x[i + 1] - x[i]))
            zero_times[source] = zero_time
            ax.axvline(zero_time, color=color, lw=0.9, ls=":")
            ax.text(zero_time + 0.015, ax.get_ylim()[0] * 0.82, f"reversal\n{zero_time:.2f} s", color=color, fontsize=6)
        ax.text(0.02, 0.95, "target-favored", transform=ax.transAxes, va="top", color="#3B78A8", fontsize=6)
        ax.text(0.02, 0.05, "flanker-favored", transform=ax.transAxes, va="bottom", color="#9B3D26", fontsize=6)
        ax.set(title=title, xlabel="Decision time (s)", ylabel="Target − flanker")

    for label, ax in zip(["a", "b", "c"], [ax_heat, ax_input, ax_state]):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    fig.suptitle("Target recovery emerges across VGG layers and survives temporal mapping", y=0.98, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.80)

    stem = output_dir / "05_natural_emergence_evidence_chain"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    layer_source = trial[["row_index", "analysis_group", *layer_cols]].copy()
    layer_source.to_csv(output_dir / "05_natural_emergence_layer_source_data.csv", index=False)
    time_source = timecourse[
        timecourse["dimension"].eq("all") & timecourse["group"].eq("all")
    ].copy()
    time_source.to_csv(output_dir / "05_natural_emergence_timecourse_source_data.csv", index=False)
    caption = f"""# Figure 05 | Natural emergence of target recovery

**Conclusion.** In incongruent trials, real VGG representations change from flanker-favored at early layers to target-favored at later layers. The layer-to-time mapping preserves this reversal, while recurrent Wong-Wang dynamics delay its expression.

**Panels.** a, Trial-level normalized target-minus-flanker evidence across five VGG stages, with trials sorted only for display. b, Mean target-minus-flanker evidence delivered to Wong-Wang. c, Mean Wong-Wang state difference. Shading in b–c shows 95% normal intervals across `{len(trial):,}` incongruent trials; dotted vertical lines mark the mean zero crossing.

Input reversal: `{zero_times.get('effective_input', math.nan):.3f}` s. WW-state reversal: `{zero_times.get('ww_state', math.nan):.3f}` s. These panels establish an upstream representational pattern, not by themselves a validated cognitive mechanism or a complete behavioral fit.
"""
    (output_dir / "05_natural_emergence_evidence_chain_caption.md").write_text(caption, encoding="utf-8")


def plot_rt_bins(summary: pd.DataFrame, timecourse: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    for source, ax, title in [("effective_input", axes[0], "Input gap by model RT bin"), ("ww_state", axes[1], "State gap by model RT bin")]:
        for q, color in zip(range(1, 6), colors):
            part = timecourse[(timecourse["source"].eq(source)) & (timecourse["dimension"].eq("model_rt_bin")) & (timecourse["group"].eq(f"Q{q}"))]
            ax.plot(part["time_s"], part["target_minus_flanker_mean"], color=color, label=f"Q{q}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(title=title, xlabel="Decision time (s)", ylabel="Target - flanker")
    agg = summary.groupby("model_rt_bin", as_index=False).agg(median_rt=("median_model_rt_s", "mean"), accuracy=("model_accuracy", "mean"))
    axes[2].plot(agg["median_rt"], agg["accuracy"], marker="o", color="#0072B2")
    for _, row in agg.iterrows():
        offset = (-18, 4) if int(row["model_rt_bin"]) == 5 else (3, 4)
        axes[2].annotate(f"Q{int(row['model_rt_bin'])}", (row["median_rt"], row["accuracy"]), xytext=offset, textcoords="offset points")
    axes[2].set(title="Current R5 speed-accuracy pattern", xlabel="Median model RT (s)", ylabel="Accuracy", ylim=(0, 1.03))
    axes[2].margins(x=0.08)
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, output_dir, "03_dynamics_by_model_rt_quantile")


def plot_human_groups(summary: pd.DataFrame, human_bins: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    specs = [("target", "Human target", "#0072B2"), ("flanker", "Human flanker error", "#D55E00"), ("other", "Human other error", "#CC79A7")]
    for source, ax, title in [("effective_input", axes[0], "Input gap by human response"), ("ww_state", axes[1], "State gap by human response")]:
        for group, label, color in specs:
            part = summary[(summary["source"].eq(source)) & (summary["dimension"].eq("human_outcome")) & (summary["group"].eq(group))]
            if not part.empty:
                ax.plot(part["time_s"], part["target_minus_flanker_mean"], color=color, label=label)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(title=title, xlabel="Decision time (s)", ylabel="Target - flanker")
    agg = human_bins.groupby("human_rt_bin", as_index=False).agg(rt=("median_human_rt_s", "mean"), accuracy=("human_accuracy", "mean"))
    axes[2].plot(agg["rt"], agg["accuracy"], marker="o", color="#009E73")
    for _, row in agg.iterrows():
        offset = (-18, -12) if int(row["human_rt_bin"]) == 5 else (3, 4)
        axes[2].annotate(f"Q{int(row['human_rt_bin'])}", (row["rt"], row["accuracy"]), xytext=offset, textcoords="offset points")
    axes[2].set(title="Human speed-accuracy pattern", xlabel="Median human RT (s)", ylabel="Accuracy", ylim=(0, 1.03))
    axes[2].margins(x=0.08)
    axes[0].legend(frameon=False)
    save_figure(fig, output_dir, "04_dynamics_by_human_response_and_rt")


def write_summary(
    output_dir: Path,
    trial: pd.DataFrame,
    outcomes: pd.DataFrame,
    rt_bins: pd.DataFrame,
    human_bins: pd.DataFrame,
) -> None:
    n = len(trial)

    def rate(metric: str, group: str = "all") -> float:
        row = outcomes[(outcomes["dimension"].eq("all" if group == "all" else "model_outcome")) & (outcomes["group"].eq(group)) & (outcomes["metric"].eq(metric))]
        return float(row.iloc[0]["rate"]) if not row.empty else math.nan

    target = trial[trial["model_outcome"].eq("target")]
    flanker = trial[trial["model_outcome"].eq("flanker")]
    caf = rt_bins.groupby("model_rt_bin", as_index=False).agg(rt=("median_model_rt_s", "mean"), accuracy=("model_accuracy", "mean"))
    caf_slope = float(np.polyfit(caf["rt"], caf["accuracy"], 1)[0])
    human_caf = human_bins.groupby("human_rt_bin", as_index=False).agg(rt=("median_human_rt_s", "mean"), accuracy=("human_accuracy", "mean"))
    human_caf_slope = float(np.polyfit(human_caf["rt"], human_caf["accuracy"], 1)[0])
    text = f"""# 真实 VGG target-flanker 时间动力学检查

## 检查问题

这次检查把三个阶段分开：VGG 各层 logits、实际送入 Wong-Wang 的时间证据、Wong-Wang 状态与 R5 读出。分析只聚焦 target 与 flanker 不同的 `{n}` 个不一致试次。

## 主要结果

- 原始 VGG 层级证据已经存在非常稳定的“早期 flanker、后期 target”结构：conv3 偏向 flanker 而 final 偏向 target 的试次比例为 `{rate('raw_reversal'):.3f}`。
- 经过 `per_layer_gap_scale` 和 `natural_smooth_5stage` 映射后，实际 WW 输入仍保留这种反转，比例为 `{rate('input_reversal'):.3f}`。
- 经过循环积累后，WW 状态保留早负晚正反转的总体比例为 `{rate('state_reversal'):.3f}`；模型正确试次为 `{rate('state_reversal', 'target'):.3f}`，flanker 错误试次为 `{rate('state_reversal', 'flanker'):.3f}`。
- 模型产生 `{len(target)}` 个 target 反应和 `{len(flanker)}` 个 flanker 错误，没有 other-error。在最终发生持续恢复的试次中，正确试次的平均状态恢复时间为 `{target['state_recovery_time_s'].mean():.3f}` 秒，flanker 错误为 `{flanker['state_recovery_time_s'].mean():.3f}` 秒。
- flanker 错误中，RT 所对应的读出时点早于状态恢复（或整个窗口都未恢复）的比例为 `{rate('readout_before_state_recovery', 'flanker'):.3f}`。这说明错误更接近“target 恢复没有及时传递到状态/RT读出”，而不是上游 VGG 完全缺少 target-recovery 信息。
- 在这些不一致试次中，人类 CAF 线性斜率为 `{human_caf_slope:.3f}`，当前 R5 为 `{caf_slope:.3f}`。两者都随 RT 上升而更准确，但斜率大小和分箱曲线仍需一起比较。真实证据已经有目标恢复结构，并不自动保证最终 speed-accuracy 模式完全匹配；积累惯性、恢复速度与读出时机仍然关键。

## 对手工机制检查的定位

此前的 target-recovery 合成输入是 hand-crafted positive control。它证明该时间结构在四选项 Wong-Wang 中具有“产生目标模式的充分可能性”，但不能单独证明人类一定依赖这项机制。现在的真实 VGG 检查进一步表明，该结构确实存在于当前模型的上游表征中；不过，要把它上升为对认知机制的经验验证，还需要在真实数据上进行留出检验、单因素消融，并排除其他机制同样解释 CAF 的可能性。

## 读出解释边界

当前 R5 的 RT 来自 sustained-crossing 时点，但反应类别来自整段轨迹中每个通道的最大强度。因此，“RT 时点早于 target 恢复”不是严格的在线决策截止：模型仍会利用该时点之后的状态决定最终类别。这暴露出 RT 与 choice 使用不同时间信息的问题，应该作为下一轮单因素对照，而不能直接解释为人在该时点已经作出不可逆选择。

RT 分箱使用保留结果包中的最终 RT。重建的 WW 决策时间与保留结果一致，但重新抽取的 t0 样本不逐试次一致；这不影响 VGG 输入和 WW 状态结论。

## 当前最合理的下一步

不需要先新增一个 target-vs-flanker 控制模块。更直接的路径是保持真实 VGG 证据不变，分别调整其进入 WW 后的恢复速度、累积惯性和读出时机，并比较“整段轨迹最大值选类”与“读出时点选类”，检验能否减少 flanker 错误并改善 CAF，同时不破坏 RT 分布和四方向选择。
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RuntimeError(f"Output directory is not empty: {output_dir}. Use a new directory or pass --force.")
    output_dir.mkdir(parents=True, exist_ok=True)

    df, outputs, _ = reconstruct_r5()
    retained = pd.read_csv(R5_RESULTS / "best_model_trial_level_predictions.csv")
    if len(df) != len(retained):
        raise RuntimeError("Reconstructed and retained trial tables differ in length.")
    alignment_columns = ["analysis_group", "row_index", "user_id", "subset_stimulus_id", "target_label", "flanker_label"]
    order_match = all(
        np.array_equal(df[column].astype(str).to_numpy(), retained[column].astype(str).to_numpy())
        for column in alignment_columns
    )
    if not order_match:
        raise RuntimeError("Reconstructed trials are not in the same order as the retained R5 table.")
    reconstructed_rt_difference = float(np.max(np.abs(df["pred_rt"].to_numpy(float) - retained["pred_rt"].to_numpy(float))))
    reconstructed_decision_difference = float(
        np.max(np.abs(df["decision_time"].to_numpy(float) - retained["decision_time"].to_numpy(float)))
    )
    reconstructed_choice_match = float((df["pred_choice"].to_numpy(int) == retained["pred_choice"].to_numpy(int)).mean())
    # Use the retained package's final RT/t0 samples for RT-bin analyses. The
    # reconstructed accumulator trajectory and decision time are exact matches.
    df["pred_rt"] = retained["pred_rt"].to_numpy(float)
    df["decision_time"] = retained["decision_time"].to_numpy(float)
    df["pred_choice"] = retained["pred_choice"].to_numpy(int)
    df["model_correct"] = retained["model_correct"].astype(bool).to_numpy()
    raw, normalized = load_raw_layers(df)

    target = df["target_label"].to_numpy(dtype=np.int64)
    flanker = df["flanker_label"].to_numpy(dtype=np.int64)
    input_gap_all = channel_gap(outputs["ww_input"], target, flanker)
    state_gap_all = channel_gap(outputs["trajectory"], target, flanker)
    incongruent = df["congruency"].to_numpy(dtype=np.int64) == 1
    inc_idx = np.where(incongruent)[0]

    trial = df.loc[incongruent].copy().reset_index(drop=True)
    input_gap = input_gap_all[inc_idx]
    state_gap = state_gap_all[inc_idx]
    early_steps = max(1, int(round(0.20 * input_gap.shape[1])))
    late_steps = early_steps
    # reconstruct_r5 stores the R5 decision time after applying the sustained
    # crossing rule. Convert that exact 10 ms grid value back to its readout step.
    readout_step = np.rint(trial["decision_time"].to_numpy(dtype=float) / (DT_MS / 1000.0)).astype(np.int64)
    readout_step = np.clip(readout_step, 0, state_gap.shape[1] - 1)
    trial["readout_step"] = readout_step

    trial["model_outcome"] = outcome_labels(trial["pred_choice"].to_numpy(int), trial["target_label"].to_numpy(int), trial["flanker_label"].to_numpy(int))
    trial["human_outcome"] = outcome_labels(trial["response_label"].to_numpy(int), trial["target_label"].to_numpy(int), trial["flanker_label"].to_numpy(int))
    trial["model_rt"] = trial["pred_rt"].astype(float)
    trial["human_rt"] = trial["true_rt"].astype(float)
    add_groupwise_quintile(trial, "model_rt", "model_rt_bin")
    add_groupwise_quintile(trial, "human_rt", "human_rt_bin")

    for layer in LAYERS:
        trial[f"raw_gap_{layer}"] = channel_gap(raw[layer][inc_idx], target[inc_idx], flanker[inc_idx])
        trial[f"normalized_gap_{layer}"] = channel_gap(normalized[layer][inc_idx], target[inc_idx], flanker[inc_idx])

    trial["raw_reversal"] = (trial["raw_gap_conv3"] < 0) & (trial["raw_gap_final"] > 0)
    trial["input_early_gap"] = input_gap[:, :early_steps].mean(axis=1)
    trial["input_late_gap"] = input_gap[:, -late_steps:].mean(axis=1)
    trial["state_early_gap"] = state_gap[:, :early_steps].mean(axis=1)
    trial["state_late_gap"] = state_gap[:, -late_steps:].mean(axis=1)
    trial["input_pattern"] = temporal_pattern(trial["input_early_gap"], trial["input_late_gap"])
    trial["state_pattern"] = temporal_pattern(trial["state_early_gap"], trial["state_late_gap"])
    trial["input_reversal"] = trial["input_pattern"].eq("early_flanker_late_target")
    trial["state_reversal"] = trial["state_pattern"].eq("early_flanker_late_target")

    input_recovery = first_stable_recovery(input_gap, sustained_k=2)
    state_recovery = first_stable_recovery(state_gap, sustained_k=2)
    trial["input_recovery_step"] = pd.array(input_recovery, dtype="Float64")
    trial["input_recovery_time_s"] = input_recovery * DT_MS / 1000.0
    trial["state_recovery_step"] = pd.array(state_recovery, dtype="Float64")
    trial["state_recovery_time_s"] = state_recovery * DT_MS / 1000.0
    trial["readout_time_s"] = readout_step * DT_MS / 1000.0
    trial["state_gap_at_readout"] = state_gap[np.arange(len(trial)), readout_step]
    trial["state_recovered_before_readout"] = np.isfinite(state_recovery) & (state_recovery <= readout_step)
    trial["readout_before_state_recovery"] = ~trial["state_recovered_before_readout"]

    layers = layer_summary(trial)
    outcomes = outcome_summary(trial)
    timecourse = timecourse_summary(trial, input_gap, state_gap)
    bins = rt_bin_summary(trial)
    human_bins = human_rt_bin_summary(trial)

    trial.to_csv(output_dir / "trial_level_target_flanker_dynamics.csv", index=False)
    layers.to_csv(output_dir / "layerwise_evidence_summary.csv", index=False)
    outcomes.to_csv(output_dir / "outcome_reversal_summary.csv", index=False)
    timecourse.to_csv(output_dir / "timecourse_target_flanker_summary.csv", index=False)
    bins.to_csv(output_dir / "model_rt_bin_dynamics_summary.csv", index=False)
    human_bins.to_csv(output_dir / "human_rt_bin_dynamics_summary.csv", index=False)
    plot_layer_summary(layers, output_dir)
    plot_timecourses(timecourse, output_dir)
    plot_natural_emergence_chain(trial, timecourse, output_dir)
    plot_rt_bins(bins, timecourse, output_dir)
    plot_human_groups(timecourse, human_bins, output_dir)
    write_summary(output_dir, trial, outcomes, bins, human_bins)

    qa = {
        "n_all_trials": int(len(df)),
        "n_incongruent_trials": int(len(trial)),
        "all_incongruent_labels_distinct": bool(np.all(target[inc_idx] != flanker[inc_idx])),
        "all_raw_values_finite": bool(all(np.isfinite(raw[layer]).all() for layer in LAYERS)),
        "all_effective_input_finite": bool(np.isfinite(outputs["ww_input"]).all()),
        "all_state_values_finite": bool(np.isfinite(outputs["trajectory"]).all()),
        "retained_trial_order_match": order_match,
        "retained_choice_match_rate": reconstructed_choice_match,
        "retained_decision_time_max_abs_difference": reconstructed_decision_difference,
        "reconstructed_vs_retained_final_rt_max_abs_difference": reconstructed_rt_difference,
        "rt_bins_use_retained_final_rt": True,
        "raw_conv3_to_final_reversal_rate": float(trial["raw_reversal"].mean()),
        "effective_input_reversal_rate": float(trial["input_reversal"].mean()),
        "ww_state_reversal_rate": float(trial["state_reversal"].mean()),
    }
    qa["passed"] = bool(
        qa["n_all_trials"] == 10000
        and qa["n_incongruent_trials"] > 0
        and qa["all_incongruent_labels_distinct"]
        and qa["all_raw_values_finite"]
        and qa["all_effective_input_finite"]
        and qa["all_state_values_finite"]
        and qa["retained_trial_order_match"]
        and qa["retained_choice_match_rate"] > 0.999
        and qa["retained_decision_time_max_abs_difference"] < 1e-6
    )
    (output_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if not qa["passed"]:
        raise RuntimeError(f"QA failed: {qa}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

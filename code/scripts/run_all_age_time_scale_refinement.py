#!/usr/bin/env python3
"""Refine a shared decision-time scale and age-specific non-decision times."""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
from run_r5_choice_coupled_schedule_optimization import SEED, accuracy_curve, safe_gap


BASE = PROJECT_ROOT / "artifacts/results/all_age_groups_20260806"
INPUT = BASE / "results/all_age_group_trial_level_predictions.csv"
OUTPUT = BASE / "all_age_model_update_20260807"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
SCALE_GRID = np.round(np.arange(0.15, 0.401, 0.01), 2)
T0_SD_GRID = np.round(np.arange(0.00, 0.241, 0.03), 2)
QUANTILES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def seed_offset(age_group: str) -> int:
    if age_group == "20-29":
        return 1
    if age_group == "80-89":
        return 0
    return int(age_group.split("-")[0])


def fit_group(
    group: pd.DataFrame, age_group: str, decision_time_scale: float, t0_sd: float
) -> tuple[dict[str, float | str], np.ndarray]:
    crossed = group["crossed"].astype(bool).to_numpy()
    human_rt = group["human_rt"].to_numpy(float)
    raw_decision = group["decision_time"].to_numpy(float)
    model_correct = group["model_correct"].astype(bool).to_numpy()
    human_correct = group["human_correct"].astype(bool).to_numpy()
    incongruent = group["congruency"].astype(int).eq(1).to_numpy()
    z = np.clip(
        np.random.default_rng(SEED + seed_offset(age_group)).normal(size=len(group)), -2.5, 2.5
    )
    base_rt = decision_time_scale * raw_decision + z * t0_sd
    human_condition_gap = float(human_rt[crossed & incongruent].mean() - human_rt[crossed & ~incongruent].mean())
    human_quantiles = np.quantile(human_rt[crossed], QUANTILES)
    base_quantiles = np.quantile(base_rt[crossed], QUANTILES)
    condition_masks = [crossed & ~incongruent, crossed & incongruent]
    human_condition_quantiles = [np.quantile(human_rt[mask], QUANTILES) for mask in condition_masks]
    base_condition_quantiles = [np.quantile(base_rt[mask], QUANTILES) for mask in condition_masks]
    human_caf = accuracy_curve(human_rt[crossed], human_correct[crossed])
    model_caf = accuracy_curve(base_rt[crossed], model_correct[crossed])
    caf_rmse = float(np.sqrt(np.mean((model_caf - human_caf) ** 2)))
    human_incongruent_caf = accuracy_curve(
        human_rt[crossed & incongruent], human_correct[crossed & incongruent]
    )
    model_incongruent_caf = accuracy_curve(
        base_rt[crossed & incongruent], model_correct[crossed & incongruent]
    )
    incongruent_caf_rmse = float(
        np.sqrt(np.mean((model_incongruent_caf - human_incongruent_caf) ** 2))
    )
    human_error_gap = safe_gap(
        human_rt[crossed & incongruent], human_correct[crossed & incongruent]
    )
    t0_base = float(np.clip(np.median(human_rt[crossed]) - np.median(base_rt[crossed]), 0.05, 1.20))
    best: tuple[float, dict[str, float | str]] | None = None
    t0_candidates = np.round(np.arange(max(0.05, t0_base - 0.12), min(1.20, t0_base + 0.121), 0.005), 3)
    for t0_mean in t0_candidates:
        pred_rt = np.maximum(base_rt + t0_mean, 0.05)
        model_quantiles = np.maximum(base_quantiles + t0_mean, 0.05)
        overall_quantile_mae = float(np.mean(np.abs(model_quantiles - human_quantiles)))
        condition_quantile_errors = []
        condition_mean_errors = []
        for index, mask in enumerate(condition_masks):
            condition_quantile_errors.append(
                float(
                    np.mean(
                        np.abs(
                            np.maximum(base_condition_quantiles[index] + t0_mean, 0.05)
                            - human_condition_quantiles[index]
                        )
                    )
                )
            )
            condition_mean_errors.append(abs(float(pred_rt[mask].mean() - human_rt[mask].mean())))
        condition_quantile_mae = float(np.mean(condition_quantile_errors))
        condition_mean_mae = float(np.mean(condition_mean_errors))
        model_condition_gap = float(pred_rt[crossed & incongruent].mean() - pred_rt[crossed & ~incongruent].mean())
        condition_gap_abs_error = abs(model_condition_gap - human_condition_gap)
        model_error_gap = safe_gap(pred_rt[crossed & incongruent], model_correct[crossed & incongruent])
        error_gap_abs_error = (
            abs(model_error_gap - human_error_gap)
            if np.isfinite(model_error_gap) and np.isfinite(human_error_gap)
            else 0.25
        )
        score = (
            2.0 * condition_quantile_mae
            + overall_quantile_mae
            + 2.0 * condition_mean_mae
            + 2.0 * condition_gap_abs_error
            + 0.5 * caf_rmse
            + 0.75 * incongruent_caf_rmse
            + 0.25 * error_gap_abs_error
        )
        record: dict[str, float | str] = {
            "age_group": age_group,
            "decision_time_scale": float(decision_time_scale),
            "t0_mean": float(t0_mean),
            "t0_sd": float(t0_sd),
            "score": float(score),
            "condition_quantile_mae": condition_quantile_mae,
            "condition_mean_mae": condition_mean_mae,
            "overall_quantile_mae": overall_quantile_mae,
            "human_condition_gap": human_condition_gap,
            "model_condition_gap": model_condition_gap,
            "condition_gap_abs_error": condition_gap_abs_error,
            "caf_rmse": caf_rmse,
            "incongruent_caf_rmse": incongruent_caf_rmse,
            "human_incongruent_error_minus_correct_rt": float(human_error_gap),
            "model_incongruent_error_minus_correct_rt": float(model_error_gap),
            "error_gap_abs_error": float(error_gap_abs_error),
        }
        if best is None or score < best[0]:
            best = (score, record)
    assert best is not None
    selected_rt = np.maximum(base_rt + float(best[1]["t0_mean"]), 0.05)
    return best[1], selected_rt


def savefig(fig: plt.Figure, stem: Path) -> None:
    for extension, kwargs in [
        ("png", {"dpi": 400}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 400}),
    ]:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def build_caf(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for age_group in AGE_GROUPS:
        age_data = data[data["age_group"] == age_group]
        for congruency, part in age_data.groupby("congruency", sort=True):
            for source, rt_col, correct_col in [
                ("human", "human_rt", "human_correct"),
                ("model", "pred_rt", "model_correct"),
            ]:
                finite = part[np.isfinite(pd.to_numeric(part[rt_col], errors="coerce"))]
                ordered = finite.sort_values(rt_col, kind="mergesort")
                for rt_bin, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), 1):
                    cell = ordered.iloc[indices]
                    rows.append(
                        {
                            "age_group": age_group,
                            "congruency": int(congruency),
                            "source": source,
                            "rt_bin": rt_bin,
                            "n_trials": len(cell),
                            "median_rt": float(cell[rt_col].median()),
                            "accuracy": float(cell[correct_col].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_caf(caf: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    colors = {"human": "#222222", "model": "#0072B2"}
    handles = []
    labels = []
    for ax, age_group in zip(axes, AGE_GROUPS):
        part = caf[caf["age_group"] == age_group]
        for (source, congruency), cell in part.groupby(["source", "congruency"], sort=True):
            label = f"{source} {'incongruent' if congruency else 'congruent'}"
            line = ax.plot(
                cell["median_rt"],
                cell["accuracy"],
                marker="o",
                linestyle="--" if int(congruency) else "-",
                color=colors[source],
                label=label,
            )[0]
            if age_group == AGE_GROUPS[0]:
                handles.append(line)
                labels.append(label)
        ax.set_title(age_group)
        ax.set_xlabel("Median RT in bin (s)")
        ax.set_ylim(0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    axes[4].set_ylabel("Accuracy")
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("CAF by age group", fontsize=13)
    savefig(fig, figure_dir / "all_age_caf_updated_model")


def condition_alignment(data: pd.DataFrame, refined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in [("human", data), ("current_model", data), ("refined_model", refined)]:
        rt_col = "human_rt" if label == "human" else "pred_rt"
        for (age_group, congruency), part in frame.groupby(["age_group", "congruency"], sort=False):
            values = pd.to_numeric(part[rt_col], errors="coerce").dropna()
            rows.append(
                {
                    "age_group": age_group,
                    "congruency": int(congruency),
                    "source": label,
                    "n_trials": len(values),
                    "mean_rt": float(values.mean()),
                    "median_rt": float(values.median()),
                }
            )
    result = pd.DataFrame(rows)
    human = result[result["source"] == "human"].set_index(["age_group", "congruency"])["mean_rt"]
    result["mean_error_vs_human"] = [
        row.mean_rt - human.loc[(row.age_group, row.congruency)] if row.source != "human" else 0.0
        for row in result.itertuples()
    ]
    return result


def plot_search_and_alignment(
    aggregate: pd.DataFrame, alignment: pd.DataFrame, figure_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    best = aggregate.iloc[0]
    ordered = aggregate.sort_values("decision_time_scale")
    ax.plot(ordered["decision_time_scale"], ordered["mean_score"], "o-", color="#0072B2")
    ax.scatter([best["decision_time_scale"]], [best["mean_score"]], s=70, facecolors="white", edgecolors="#D55E00", linewidths=1.5, zorder=5)
    ax.set(xlabel="Shared decision-time scale", ylabel="Mean condition-aware score", title="Timing calibration selection")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, figure_dir / "shared_timing_calibration_selection")

    comparison = alignment[alignment["source"].isin(["current_model", "refined_model"])].copy()
    comparison["condition"] = comparison["congruency"].map({0: "Congruent", 1: "Incongruent"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    x = np.arange(len(AGE_GROUPS))
    for ax, source, title, color in [
        (axes[0], "current_model", "Before refinement", "#999999"),
        (axes[1], "refined_model", "After refinement", "#0072B2"),
    ]:
        part = comparison[comparison["source"] == source]
        for congruency, marker, label in [(0, "o", "Congruent"), (1, "s", "Incongruent")]:
            cell = part[part["congruency"] == congruency].set_index("age_group").reindex(AGE_GROUPS)
            ax.plot(x, 1000 * cell["mean_error_vs_human"], marker=marker, color=color, linestyle="-" if congruency == 0 else "--", label=label)
        ax.axhline(0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, AGE_GROUPS, rotation=45)
        ax.set(xlabel="Age group", title=title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Model - human mean RT (ms)")
    savefig(fig, figure_dir / "condition_rt_alignment_before_after")


def write_summary(
    selected_scale: float,
    selected_metrics: pd.DataFrame,
    alignment: pd.DataFrame,
    output: Path,
) -> None:
    comparison = alignment[alignment["source"].isin(["current_model", "refined_model"])]
    before = float(comparison[comparison["source"] == "current_model"]["mean_error_vs_human"].abs().mean())
    after = float(comparison[comparison["source"] == "refined_model"]["mean_error_vs_human"].abs().mean())
    lines = [
        "# 全年龄段时间尺度修正摘要",
        "",
        f"共享决策时间尺度选择为 `{selected_scale:.2f}`。VGG 证据、模型选择、正式读取时刻和到达阈值状态均未改变；本次只重新映射模型内部决策时间，并按年龄段重估非决策时间。",
        "",
        f"一致/不一致条件的平均绝对反应时误差由 {before * 1000:.1f} 毫秒降至 {after * 1000:.1f} 毫秒。",
        "",
        "| 年龄段 | 新 t0 | 新 t0 离散程度 | 人类条件差（毫秒） | 模型条件差（毫秒） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected_metrics.itertuples():
        lines.append(
            f"| {row.age_group} | {row.t0_mean:.3f} | {row.t0_sd:.3f} | "
            f"{row.human_condition_gap * 1000:.1f} | {row.model_condition_gap * 1000:.1f} |"
        )
    lines.extend(
        [
            "",
            "该修正明显改善了反应时条件结构，但参数仍在同一批代表性试次上选择，尚未经过独立被试或刺激验证。因此它是更合理的诊断性时间映射，不应表述为最终确认的年龄机制。",
        ]
    )
    (output / "summaries/updated_model_summary_chinese.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    results = OUTPUT / "results"
    figures = OUTPUT / "figures_publication"
    summaries = OUTPUT / "summaries"
    configs = OUTPUT / "configs"
    for directory in [results, figures, summaries, configs]:
        directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT, low_memory=False)
    if data.groupby("age_group").size().reindex(AGE_GROUPS).ne(5000).any():
        raise RuntimeError("Expected exactly 5,000 trials in each age group")

    candidate_rows: list[dict[str, object]] = []
    best_by_scale: list[dict[str, object]] = []
    for scale in SCALE_GRID:
        scale_best: list[dict[str, object]] = []
        for age_group in AGE_GROUPS:
            group = data[data["age_group"] == age_group]
            candidates = []
            for t0_sd in T0_SD_GRID:
                record, _ = fit_group(group, age_group, float(scale), float(t0_sd))
                candidate_rows.append(record)
                candidates.append(record)
            best = min(candidates, key=lambda row: float(row["score"]))
            scale_best.append(best)
        best_by_scale.extend(scale_best)

    candidate_table = pd.DataFrame(candidate_rows)
    best_scale_groups = pd.DataFrame(best_by_scale)
    aggregate = (
        best_scale_groups.groupby("decision_time_scale", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            maximum_group_score=("score", "max"),
            mean_condition_quantile_mae=("condition_quantile_mae", "mean"),
            mean_condition_mean_mae=("condition_mean_mae", "mean"),
            mean_condition_gap_abs_error=("condition_gap_abs_error", "mean"),
            mean_caf_rmse=("caf_rmse", "mean"),
            mean_incongruent_caf_rmse=("incongruent_caf_rmse", "mean"),
        )
        .sort_values(["mean_score", "maximum_group_score"], kind="mergesort")
        .reset_index(drop=True)
    )
    selected_scale = float(aggregate.iloc[0]["decision_time_scale"])
    selected_metrics = best_scale_groups[
        best_scale_groups["decision_time_scale"].eq(selected_scale)
    ].copy()
    selected_metrics["age_group"] = pd.Categorical(selected_metrics["age_group"], AGE_GROUPS, ordered=True)
    selected_metrics = selected_metrics.sort_values("age_group").reset_index(drop=True)

    refined_parts: list[pd.DataFrame] = []
    for row in selected_metrics.itertuples():
        group = data[data["age_group"] == str(row.age_group)].copy()
        _, pred_rt = fit_group(group, str(row.age_group), selected_scale, float(row.t0_sd))
        group["raw_decision_time"] = group["decision_time"]
        group["decision_time_scale"] = selected_scale
        group["decision_time"] = group["raw_decision_time"] * selected_scale
        group["pred_rt"] = pred_rt
        group.loc[~group["crossed"].astype(bool), "pred_rt"] = np.nan
        group["t0_mean"] = float(row.t0_mean)
        group["t0_sd"] = float(row.t0_sd)
        for column in ["target_recovery_time", "reversal_time"]:
            if column in group.columns:
                group[column] = pd.to_numeric(group[column], errors="coerce") * selected_scale
        group["model_name"] = "choice_coupled_corrected_equivalent_updated_timing"
        refined_parts.append(group)
    refined = pd.concat(refined_parts, ignore_index=True)
    if not (refined["pred_choice"].astype(int) == refined["winner_at_readout"].astype(int)).all():
        raise RuntimeError("Choice/readout alignment changed during RT-only refinement")
    if refined.loc[~refined["crossed"].astype(bool), "pred_rt"].notna().any():
        raise RuntimeError("No-crossing trial received an observed RT")

    candidate_table.to_csv(results / "shared_timing_candidate_metrics.csv", index=False)
    aggregate.to_csv(results / "shared_timing_metrics.csv", index=False)
    selected_metrics.to_csv(results / "updated_model_parameters_by_age.csv", index=False)
    refined.to_csv(results / "updated_model_trial_level_predictions.csv", index=False)
    caf = build_caf(refined)
    caf.to_csv(results / "updated_model_caf.csv", index=False)
    alignment = condition_alignment(data, refined)
    alignment.to_csv(results / "condition_rt_alignment_before_after.csv", index=False)

    plot_caf(caf, figures)
    plot_search_and_alignment(aggregate, alignment, figures)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code/scripts/plot_all_age_group_rt_distributions.py"),
            "--input",
            str(results / "updated_model_trial_level_predictions.csv"),
            "--figure-dir",
            str(figures),
            "--result-dir",
            str(results),
            "--stem",
            "all_age_rt_distribution_updated_model",
            "--source-name",
            "updated_model_rt_distribution_kde_source.csv",
            "--summary-name",
            "updated_model_rt_distribution_summary.csv",
            "--title",
            "RT distributions by age group",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    write_summary(selected_scale, selected_metrics, alignment, OUTPUT)
    config = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "seed": SEED,
        "decision_time_scale_grid": SCALE_GRID.tolist(),
        "t0_sd_grid": T0_SD_GRID.tolist(),
        "selected_shared_decision_time_scale": selected_scale,
        "score": "2*condition_quantile_mae + overall_quantile_mae + 2*condition_mean_mae + 2*condition_gap_abs_error + 0.5*caf_rmse + 0.75*incongruent_caf_rmse + 0.25*error_gap_abs_error",
        "unchanged": ["VGG evidence", "WW trajectory", "choice", "readout_step", "crossing"],
        "scope": "same-data diagnostic calibration; not held-out validation",
    }
    (configs / "updated_model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    qa = {
        "n_trials": len(refined),
        "n_age_groups": refined["age_group"].nunique(),
        "selected_shared_decision_time_scale": selected_scale,
        "choice_readout_consistency": float(
            (refined["pred_choice"] == refined["winner_at_readout"]).mean()
        ),
        "crossing_rate": float(refined["crossed"].mean()),
        "n_no_crossing": int((~refined["crossed"].astype(bool)).sum()),
        "no_crossing_rt_is_missing": bool(
            refined.loc[~refined["crossed"].astype(bool), "pred_rt"].isna().all()
        ),
    }
    (OUTPUT / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"Selected shared decision-time scale: {selected_scale:.2f}")
    print(selected_metrics.to_string(index=False))


if __name__ == "__main__":
    main()

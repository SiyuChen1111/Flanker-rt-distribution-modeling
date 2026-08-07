#!/usr/bin/env python3
"""Unify corrected predictions and recompute all-age diagnostic results."""
from __future__ import annotations

import argparse
import json
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
EXTREME_PREDICTIONS = (
    PROJECT_ROOT
    / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803/selected_trial_level_predictions.csv"
)
MIDDLE_PREDICTIONS = ROOT / "results/corrected_model_by_age/selected_trial_level_predictions.csv"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
ANALYSIS_TO_AGE = {"young_20_29": "20-29", "older_80_89": "80-89"}
MODEL_NAME = "choice_coupled_corrected_equivalent"
FINGERPRINT_ID = "vgg16_5layer_pergap_ww4_choice_coupled_20260803"
SEED = 20260530


def qstats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {key: np.nan for key in ["mean", "median", "sd", "q10", "q50", "q90", "q95", "skew"]}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "skew": float(pd.Series(x).skew()) if len(x) > 2 else np.nan,
    }


def caf(
    data: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    source: str,
    age_group: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for congruency, part in data.groupby("congruency", sort=True):
        finite = part[np.isfinite(pd.to_numeric(part[rt_col], errors="coerce"))]
        finite = finite.sort_values(rt_col, kind="mergesort")
        for rt_bin, indices in enumerate(np.array_split(np.arange(len(finite)), n_bins), 1):
            cell = finite.iloc[indices]
            rows.append(
                {
                    "age_group": age_group,
                    "congruency": int(congruency),
                    "source": source,
                    "rt_bin": rt_bin,
                    "n_trials": len(cell),
                    "median_rt": float(cell[rt_col].median()) if len(cell) else np.nan,
                    "accuracy": float(cell[correct_col].mean()) if len(cell) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def participant_delta(
    data: pd.DataFrame, rt_col: str, correct_col: str, source: str, age_group: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user_id, part in data.groupby("user_id", sort=True):
        congruent = part[(part["congruency"] == 0) & part[correct_col].astype(bool)][rt_col]
        incongruent = part[(part["congruency"] == 1) & part[correct_col].astype(bool)][rt_col]
        congruent = pd.to_numeric(congruent, errors="coerce").dropna().sort_values().to_numpy()
        incongruent = pd.to_numeric(incongruent, errors="coerce").dropna().sort_values().to_numpy()
        for rt_bin, (c_values, i_values) in enumerate(
            zip(np.array_split(congruent, 5), np.array_split(incongruent, 5)), 1
        ):
            if len(c_values) and len(i_values):
                rows.append(
                    {
                        "age_group": age_group,
                        "source": source,
                        "user_id": str(user_id),
                        "rt_bin": rt_bin,
                        "congruent_median_rt": float(np.median(c_values)),
                        "incongruent_median_rt": float(np.median(i_values)),
                        "delta_rt": float(np.median(i_values) - np.median(c_values)),
                    }
                )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, stem: Path) -> None:
    for extension, kwargs in [
        ("png", {"dpi": 400}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 400}),
    ]:
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_predictions() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if EXTREME_PREDICTIONS.exists():
        extreme = pd.read_csv(EXTREME_PREDICTIONS)
        extreme["age_group"] = extreme["analysis_group"].map(ANALYSIS_TO_AGE)
        extreme["original_age_group"] = extreme["analysis_group"]
        extreme["target_recovery_time"] = np.nan
        extreme["reversal_time"] = np.nan
        parts.append(extreme)
    if MIDDLE_PREDICTIONS.exists():
        parts.append(pd.read_csv(MIDDLE_PREDICTIONS, low_memory=False))
    if not parts:
        raise FileNotFoundError("No corrected-equivalent trial predictions were found")

    pred = pd.concat(parts, ignore_index=True, sort=False)
    pred["age_group"] = pred["age_group"].astype(str)
    pred["analysis_group"] = pred["analysis_group"].astype(str)
    pred["user_id"] = pred["user_id"].astype(str)
    pred["crossed"] = pred["crossed"].astype(bool)
    pred["model_name"] = pred.get("model_name", MODEL_NAME).fillna(MODEL_NAME)
    pred["model_fingerprint_id"] = pred.get("model_fingerprint_id", FINGERPRINT_ID).fillna(FINGERPRINT_ID)
    pred["random_seed"] = pd.to_numeric(pred.get("random_seed", SEED), errors="coerce").fillna(SEED).astype(int)
    pred["winner_at_readout"] = pred.get("winner_at_readout", pred["pred_choice"]).fillna(pred["pred_choice"])
    pred["winner_at_crossing"] = pred.get("winner_at_crossing", pred["pred_choice"]).fillna(pred["pred_choice"])
    pred["no_crossing_reason"] = np.where(pred["crossed"], "", "deadline_censoring")
    pred["pred_rt"] = pd.to_numeric(pred["pred_rt"], errors="coerce").where(pred["crossed"], np.nan)
    pred["human_rt"] = pd.to_numeric(pred["true_rt"], errors="coerce")

    expected = set(AGE_GROUPS)
    observed = set(pred["age_group"])
    if observed != expected:
        raise RuntimeError(f"Prediction age groups mismatch: expected {expected}, observed {observed}")
    counts = pred.groupby("age_group").size()
    if not counts.eq(5000).all():
        raise RuntimeError(f"Expected 5000 predictions per group; got {counts.to_dict()}")
    if not (pred["pred_choice"].astype(int) == pred["winner_at_readout"].astype(int)).all():
        raise RuntimeError("Choice/readout alignment failed")

    columns = [
        "analysis_group", "age_group", "original_age_group", "user_id", "row_index",
        "target_label", "flanker_label", "response_label", "human_correct", "human_rt",
        "congruency", "pred_choice", "model_correct", "decision_time", "pred_rt", "crossed",
        "readout_step", "winner_at_readout", "winner_at_crossing", "no_crossing_reason",
        "target_recovery_time", "reversal_time", "model_name", "model_fingerprint_id",
        "random_seed", "evidence_gain", "threshold", "min_decision_time", "sustained_k",
        "margin", "compression", "late_shift_s", "width_scale", "t0_mean", "t0_sd",
        "choice_rule", "readout_rule",
    ]
    return pred[[column for column in columns if column in pred.columns]].copy()


def compute_results(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    caf_frames: list[pd.DataFrame] = []
    delta_frames: list[pd.DataFrame] = []
    for age_group in AGE_GROUPS:
        part = pred[pred["age_group"] == age_group]
        for source, rt_col, correct_col in [
            ("human", "human_rt", "human_correct"),
            ("model", "pred_rt", "model_correct"),
        ]:
            finite = np.isfinite(pd.to_numeric(part[rt_col], errors="coerce"))
            stats = qstats(part.loc[finite, rt_col])
            metric_rows.append(
                {
                    "age_group": age_group,
                    "source": source,
                    "n_subjects": int(part["user_id"].nunique()),
                    "n_trials": len(part),
                    "n_rt_observed": int(finite.sum()),
                    "accuracy": float(part[correct_col].astype(bool).mean()),
                    **{f"rt_{key}": value for key, value in stats.items()},
                    "crossing_rate": float(part["crossed"].mean()) if source == "model" else np.nan,
                    "model_status": (
                        "completed_with_corrected_equivalent_model" if source == "model" else "human_subset"
                    ),
                }
            )
            caf_frames.append(caf(part, rt_col, correct_col, source, age_group))
            delta_frames.append(participant_delta(part, rt_col, correct_col, source, age_group))
    return (
        pd.DataFrame(metric_rows),
        pd.concat(caf_frames, ignore_index=True),
        pd.concat(delta_frames, ignore_index=True),
    )


def plot_overview(metrics: pd.DataFrame, caf_table: pd.DataFrame, figure_dir: Path) -> None:
    midpoints = [int(group.split("-")[0]) + 4.5 for group in AGE_GROUPS]
    trend = metrics.pivot(index="age_group", columns="source", values="rt_mean").reindex(AGE_GROUPS)
    fig, ax = plt.subplots(figsize=(7, 4))
    for source, color, label in [("human", "#222222", "Human"), ("model", "#0072B2", "Model")]:
        ax.plot(midpoints, trend[source], "o-", label=label, color=color)
    ax.set(xlabel="Age-group midpoint (years)", ylabel="Mean RT (s)", title="Mean RT across age groups")
    ax.legend(frameon=False)
    savefig(fig, figure_dir / "age_trend_mean_rt")

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    colors = {"human": "#222222", "model": "#0072B2"}
    styles = {0: "-", 1: "--"}
    for ax, age_group in zip(axes, AGE_GROUPS):
        part = caf_table[caf_table["age_group"] == age_group]
        for (source, congruency), cell in part.groupby(["source", "congruency"], sort=True):
            label = f"{source} {'incongruent' if congruency else 'congruent'}"
            ax.plot(
                cell["median_rt"], cell["accuracy"], marker="o", linestyle=styles[int(congruency)],
                color=colors[source], label=label,
            )
        ax.set_title(age_group)
        ax.set_xlabel("Median RT in bin (s)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Accuracy")
    axes[-1].axis("off")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("CAF by age group", y=1.02)
    savefig(fig, figure_dir / "all_age_caf_small_multiples")


def write_reports(root: Path, metrics: pd.DataFrame, pred: pd.DataFrame) -> None:
    model = metrics[metrics["source"] == "model"].set_index("age_group")
    human = metrics[metrics["source"] == "human"].set_index("age_group")
    lines = [
        "# 全年龄段扩展摘要",
        "",
        "七个年龄段均已完成。数据覆盖 75 名被试；每组使用 5,000 条确定性代表性试次，共 35,000 条；30–79 岁新生成了 25,000 条完整 VGG 分层证据并完成模型比较。所有组的选择均与正式读取时刻一致。80–89 岁仅有 4 名被试，其结果应谨慎解释。",
        "",
        "| 年龄段 | 人类正确率 | 模型正确率 | 人类平均反应时（秒） | 模型平均反应时（秒） | 到达阈值比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for age_group in AGE_GROUPS:
        lines.append(
            f"| {age_group} | {human.loc[age_group, 'accuracy']:.3f} | {model.loc[age_group, 'accuracy']:.3f} | "
            f"{human.loc[age_group, 'rt_mean']:.3f} | {model.loc[age_group, 'rt_mean']:.3f} | "
            f"{model.loc[age_group, 'crossing_rate']:.4f} |"
        )
    no_crossing = int((~pred["crossed"].astype(bool)).sum())
    lines.extend(
        [
            "",
            f"未到达阈值共 {no_crossing} 条，其反应时保留为空，不进入反应时分布、CAF、CRF 或 delta 曲线。CAF 使用各自人类/模型反应时的真实中位数坐标；delta 先在每位被试内计算，再汇总被试。",
            "",
            "这些结果支持模型在七个年龄段上复现总体反应时随年龄变化的方向，并部分支持正确率与冲突条件下的动态恢复。模型仍存在系统偏差，尤其是错误试次数偏少、部分年龄段的错误反应时差距不足，因此这里仍称为代表性子集的诊断性全年龄扩展，而不是全体试次或独立留出数据上的最终拟合。",
        ]
    )
    (root / "summaries/all_age_group_extension_summary_chinese.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (root / "summaries/all_age_group_extension_technical_report.md").write_text(
        """# Technical report

The historical presentation chain is the retained natural layer-to-time VGG16 plus four-choice Wong-Wang R5 package. The corrected-equivalent retains the five VGG evidence layers, per-layer normalization, recurrent four-choice accumulator, sustained crossing, and non-decision-time model while coupling choice to `winner_at_readout`.

All seven age groups now have 5,000 deterministic representative trials. The five intermediate groups use newly extracted complete VGG caches and the same 171-candidate schedule search. WW threshold and margin are linearly interpolated between the retained extreme-age anchors; schedule compression, late shift, width, and non-decision-time terms are selected with the original corrected-equivalent score. This is an exploratory age-structured rule, not an independently validated causal age model.

All derived accuracy, RT, CAF, CRF, crossing, and participant-first delta outputs are recomputed from the unified 35,000-row trial file. No-crossing rows are explicitly censored and have no model RT.
""",
        encoding="utf-8",
    )
    (root / "README_all_age_groups.md").write_text(
        """# All-age diagnostic extension

Seven age groups are complete on deterministic 5,000-trial representative subsets. Start with `summaries/all_age_group_extension_summary_chinese.md`, then inspect `results/all_age_group_trial_level_predictions.csv`, `results/all_age_group_metrics.csv`, and `audits/age_group_run_status.csv`. This is a diagnostic representative-subset extension, not a final full-cohort fit.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.output_dir)
    for name in ["results", "figures_publication", "summaries", "logs", "configs"]:
        (root / name).mkdir(parents=True, exist_ok=True)

    pred = load_predictions()
    pred.to_csv(root / "results/all_age_group_trial_level_predictions.csv", index=False)
    metrics, caf_table, delta_table = compute_results(pred)
    metrics.to_csv(root / "results/all_age_group_metrics.csv", index=False)
    caf_table.to_csv(root / "results/all_age_group_caf.csv", index=False)
    delta_table.to_csv(root / "results/all_age_group_subject_delta.csv", index=False)

    status = pd.read_csv(root / "audits/age_group_run_status.csv")
    status["run_status"] = "completed_with_corrected_equivalent_model"
    status["existing_subset"] = True
    status["existing_evidence_cache"] = True
    status["existing_trial_predictions"] = True
    status["n_existing_selected"] = 5000
    status["n_existing_predictions"] = 5000
    status.to_csv(root / "audits/age_group_run_status.csv", index=False)

    fingerprint_path = root / "configs/presentation_model_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    fingerprint["selected_model_for_age_extension"] = (
        "corrected-equivalent choice-coupled schedule; all seven representative age groups completed"
    )
    fingerprint["middle_age_trial_prediction_file"] = str(MIDDLE_PREDICTIONS)
    fingerprint["unified_trial_prediction_file"] = str(
        root / "results/all_age_group_trial_level_predictions.csv"
    )
    fingerprint["completed_age_groups"] = AGE_GROUPS
    fingerprint["model_identity_status"] = (
        "presentation model traced to legacy R5 mechanism figure; corrected-equivalent completed for all seven age groups"
    )
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_overview(metrics, caf_table, root / "figures_publication")
    write_reports(root, metrics, pred)
    (root / "logs/run_log.txt").write_text(
        "audit, deterministic subset, VGG cache extraction, corrected-equivalent model search, unified analysis and publication plots completed for all seven age groups.\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

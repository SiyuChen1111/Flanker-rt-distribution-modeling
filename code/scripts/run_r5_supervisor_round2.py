#!/usr/bin/env python3
"""Second supervisor follow-up: explicit RT ticks, time scaling, and SAT probes."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vam_studying_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from vgg_wongwang_lim import WongWangMultiClassDecision  # noqa: E402


FOLLOWUP = PROJECT_ROOT / "artifacts/results/r5_supervisor_followup"
CORE = PROJECT_ROOT / "artifacts/results/ww_diffdecision_core_audit_20260802"
OUT = PROJECT_ROOT / "artifacts/results/r5_supervisor_round2_20260802"
SEEDS = [20260820, 20260821, 20260822, 20260823, 20260824]
DT_MS = 10
HORIZON_MS = 2000
TIME_STEPS = HORIZON_MS // DT_MS
SOURCE_STYLE = {
    "Human": dict(color="black", linestyle="-", marker="o", markerfacecolor="white"),
    "Model": dict(color="#D55E00", linestyle="--", marker="s", markerfacecolor="#D55E00"),
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def explicit_tick_labels(values: pd.Series) -> list[str]:
    return [f"{v:.2f}" for v in values.to_numpy(float)]


def plot_explicit_rt_ticks() -> None:
    caf = pd.read_csv(FOLLOWUP / "02_CAF_actual_RT_values.csv")
    crf = pd.read_csv(FOLLOWUP / "03_CRF_actual_RT_values.csv")

    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), sharey=True)
    columns = [("Congruent", "Human"), ("Congruent", "Model"), ("Incongruent", "Human"), ("Incongruent", "Model")]
    for r, age in enumerate(["Young", "Older"]):
        for c, (cong, source) in enumerate(columns):
            ax = axes[r, c]
            p = caf[
                caf["age_group"].eq(age)
                & caf["congruency"].eq(cong)
                & caf["source"].eq(source)
            ].sort_values("quantile_bin")
            st = SOURCE_STYLE[source]
            ax.errorbar(
                p["median_rt"],
                p["accuracy"],
                yerr=[p["accuracy"] - p["ci95_low"], p["ci95_high"] - p["accuracy"]],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markerfacecolor=st["markerfacecolor"],
                markeredgecolor=st["color"],
                linewidth=1.5,
                capsize=2,
            )
            ax.set_title(f"{age}, {cong}, {source}")
            ax.set_xticks(p["median_rt"], explicit_tick_labels(p["median_rt"]), rotation=40, ha="right")
            ax.set_xlabel("Median RT in bin (s)")
            if c == 0:
                ax.set_ylabel("Accuracy")
            ax.set_ylim(0.50, 1.02)
    fig.subplots_adjust(bottom=0.15, hspace=0.42, wspace=0.28)
    save(fig, "01_CAF_explicit_quantile_RT_ticks")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharey=True)
    response_style = {
        "target": ("black", "-", "o"),
        "flanker": ("#D55E00", "--", "s"),
        "other": ("#0072B2", ":", "^")
    }
    for r, age in enumerate(["Young", "Older"]):
        for c, source in enumerate(["Human", "Model"]):
            ax = axes[r, c]
            tick_part = crf[crf["age_group"].eq(age) & crf["source"].eq(source) & crf["response_type"].eq("target")].sort_values("quantile_bin")
            for response_type in ["target", "flanker", "other"]:
                p = crf[
                    crf["age_group"].eq(age)
                    & crf["source"].eq(source)
                    & crf["response_type"].eq(response_type)
                ].sort_values("quantile_bin")
                color, linestyle, marker = response_style[response_type]
                ax.errorbar(
                    p["median_rt"],
                    p["proportion"],
                    yerr=[p["proportion"] - p["ci95_low"], p["ci95_high"] - p["proportion"]],
                    label=response_type,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor="white" if response_type != "flanker" else color,
                    linewidth=1.5,
                    capsize=2,
                )
            ax.set_title(f"{age}, {source}")
            ax.set_xticks(tick_part["median_rt"], explicit_tick_labels(tick_part["median_rt"]), rotation=40, ha="right")
            ax.set_xlabel("Median RT in bin (s)")
            if c == 0:
                ax.set_ylabel("Response proportion")
            ax.set_ylim(0.0, 1.02)
    axes[0, 0].legend(frameon=False)
    fig.subplots_adjust(bottom=0.15, hspace=0.42, wspace=0.22)
    save(fig, "02_CRF_explicit_quantile_RT_ticks")


def scale_decision_time_shape(target_median_s: float = 0.60) -> pd.DataFrame:
    raw = pd.read_csv(CORE / "representative_decision_times.csv")
    vals = raw[
        raw["regime"].eq("rtify_4choice_calibrated_noise_dt10")
        & np.isclose(raw["target_gap"], 0.1)
        & np.isclose(raw["noise_ampa"], 0.006)
    ]["decision_time_s"].to_numpy(float)
    scale = float(target_median_s / np.median(vals))
    scaled = vals * scale
    rows = []
    for label, x in [("raw", vals), ("scaled", scaled)]:
        rows.append(
            {
                "version": label,
                "n": len(x),
                "scale_factor": 1.0 if label == "raw" else scale,
                "mean_s": float(x.mean()),
                "sd_s": float(x.std(ddof=1)),
                "median_s": float(np.median(x)),
                "q10_s": float(np.quantile(x, 0.10)),
                "q90_s": float(np.quantile(x, 0.90)),
                "skewness": float(stats.skew(x)),
                "coefficient_of_variation": float(x.std(ddof=1) / x.mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "03_scaled_decision_time_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    axes[0].hist(vals, bins=35, density=True, color="#D55E00", alpha=0.82)
    axes[0].set(title="Original decision time", xlabel="Decision time (s)", ylabel="Density")
    axes[1].hist(scaled, bins=35, density=True, color="#0072B2", alpha=0.82)
    axes[1].set(title=f"Scaled time (× {scale:.3f})", xlabel="Scaled decision time (s)", ylabel="Density")
    bins = np.linspace(0, 2.2, 35)
    axes[2].hist(vals / np.median(vals), bins=bins, density=True, histtype="step", linewidth=2, color="#D55E00", label="Original")
    axes[2].hist(scaled / np.median(scaled), bins=bins, density=True, histtype="step", linewidth=2, color="#0072B2", linestyle="--", label="Scaled")
    axes[2].set(title="Shape after median normalization", xlabel="Time / median", ylabel="Density")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "03_time_scaling_preserves_shape")
    return summary


def make_model(threshold: float, noise: float) -> WongWangMultiClassDecision:
    model = WongWangMultiClassDecision(n_classes=4, dt=DT_MS, time_steps=TIME_STEPS, t_stimulus=TIME_STEPS)
    model.eval()
    with torch.no_grad():
        model.threshold.fill_(threshold)
        model.noise_ampa.fill_(noise)
    return model


def evidence_schedule(n: int, condition: str) -> torch.Tensor:
    x = torch.ones(n, TIME_STEPS, 4, dtype=torch.float32)
    if condition == "static_competition":
        x[:, :, 0] += 0.10
        x[:, :, 1] += 0.04
        x[:, :, 2:] -= 0.07
    elif condition == "target_recovery":
        # A bounded conflict schedule: target and flanker both receive early
        # support, with the flanker temporarily stronger; after 300 ms the
        # target becomes dominant. This can create fast flanker errors without
        # forcing every noisy early fluctuation to win.
        switch = 30
        x[:, :switch, 0] += 0.10
        x[:, :switch, 1] += 0.40
        x[:, :switch, 2:] -= 0.125
        x[:, switch:, 0] += 0.25
        x[:, switch:, 1] -= 0.0625
        x[:, switch:, 2:] -= 0.09375
    else:
        raise ValueError(condition)
    return x


def first_crossing(trajectory: torch.Tensor, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crossed = trajectory > threshold
    any_class = crossed.any(dim=2)
    trial_crossed = any_class.any(dim=1)
    steps = torch.arange(trajectory.shape[1], device=trajectory.device).view(1, -1)
    first_step = torch.where(any_class, steps, trajectory.shape[1]).amin(dim=1)
    safe_step = first_step.clamp_max(trajectory.shape[1] - 1)
    state = trajectory[torch.arange(trajectory.shape[0]), safe_step]
    choice = state.argmax(dim=1)
    rt = first_step.float() * DT_MS / 1000.0
    rt[~trial_crossed] = torch.nan
    return rt.cpu().numpy(), choice.cpu().numpy(), trial_crossed.cpu().numpy()


def simulate_speed_accuracy(n_per_seed: int = 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = [0.30, 0.40, 0.50, 0.60]
    rows = []
    trials = []
    for condition in ["static_competition", "target_recovery"]:
        for threshold in thresholds:
            model = make_model(threshold, noise=0.012)
            for seed in SEEDS:
                x = evidence_schedule(n_per_seed, condition)
                generator = torch.Generator().manual_seed(seed)
                with torch.no_grad():
                    _, trajectory, _ = model.inference(x, generator=generator)
                rt, choice, crossed = first_crossing(trajectory, threshold)
                valid = crossed & np.isfinite(rt)
                rows.append(
                    {
                        "condition": condition,
                        "threshold": threshold,
                        "seed": seed,
                        "n_trials": n_per_seed,
                        "crossing_rate": float(crossed.mean()),
                        "accuracy_crossed": float((choice[valid] == 0).mean()) if valid.any() else math.nan,
                        "flanker_choice_rate_crossed": float((choice[valid] == 1).mean()) if valid.any() else math.nan,
                        "median_decision_time_s": float(np.median(rt[valid])) if valid.any() else math.nan,
                    }
                )
                if math.isclose(threshold, 0.30):
                    for trial_rt, trial_choice in zip(rt[valid], choice[valid]):
                        trials.append(
                            {
                                "condition": condition,
                                "threshold": threshold,
                                "seed": seed,
                                "decision_time_s": float(trial_rt),
                                "choice": int(trial_choice),
                                "correct": int(trial_choice == 0),
                            }
                        )
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(OUT / "04_speed_accuracy_seed_metrics.csv", index=False)
    agg = seed_df.groupby(["condition", "threshold"], as_index=False).agg(
        crossing_rate=("crossing_rate", "mean"),
        accuracy=("accuracy_crossed", "mean"),
        flanker_choice_rate=("flanker_choice_rate_crossed", "mean"),
        median_decision_time_s=("median_decision_time_s", "mean"),
    )
    agg.to_csv(OUT / "04_speed_accuracy_threshold_curve.csv", index=False)

    trial_df = pd.DataFrame(trials)
    caf_rows = []
    for condition, part in trial_df.groupby("condition"):
        part = part.sort_values(["decision_time_s", "seed"]).copy()
        part["rt_bin"] = pd.qcut(part["decision_time_s"].rank(method="first"), 5, labels=False) + 1
        for rt_bin, bp in part.groupby("rt_bin"):
            caf_rows.append(
                {
                    "condition": condition,
                    "rt_bin": int(rt_bin),
                    "median_decision_time_s": float(bp["decision_time_s"].median()),
                    "accuracy": float(bp["correct"].mean()),
                    "flanker_choice_rate": float((bp["choice"] == 1).mean()),
                    "n_trials": int(len(bp)),
                }
            )
    caf = pd.DataFrame(caf_rows)
    caf.to_csv(OUT / "05_speed_accuracy_CAF.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    styles = {"static_competition": ("#777777", "o", "Static competition"), "target_recovery": ("#0072B2", "s", "Early flanker, late target recovery")}
    for condition, part in agg.groupby("condition"):
        color, marker, label = styles[condition]
        part = part.sort_values("threshold")
        axes[0].plot(part["median_decision_time_s"], part["accuracy"], color=color, marker=marker, label=label)
        axes[1].plot(part["threshold"], part["crossing_rate"], color=color, marker=marker, label=label)
    for condition, part in caf.groupby("condition"):
        color, marker, label = styles[condition]
        axes[2].plot(part["median_decision_time_s"], part["accuracy"], color=color, marker=marker, label=label)
    axes[0].set(xlabel="Median decision time (s)", ylabel="Accuracy", title="Threshold speed–accuracy curve", ylim=(0, 1.03))
    axes[1].set(xlabel="Threshold", ylabel="Crossing rate", title="Decisions before deadline", ylim=(0, 1.03))
    axes[2].set(xlabel="Median decision time in bin (s)", ylabel="Accuracy", title="Conditional accuracy (threshold = 0.30)", ylim=(0, 1.03))
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "04_improved_model_speed_accuracy")
    return agg, caf


def write_summary(scale: pd.DataFrame, sat: pd.DataFrame, caf: pd.DataFrame) -> dict[str, bool]:
    raw = scale.loc[scale["version"].eq("raw")].iloc[0]
    scaled = scale.loc[scale["version"].eq("scaled")].iloc[0]
    recovery = caf[caf["condition"].eq("target_recovery")].sort_values("rt_bin")
    static = caf[caf["condition"].eq("static_competition")].sort_values("rt_bin")
    recovery_slope = float(np.polyfit(recovery["median_decision_time_s"], recovery["accuracy"], 1)[0])
    static_slope = float(np.polyfit(static["median_decision_time_s"], static["accuracy"], 1)[0])
    qa = {
        "caf_and_crf_sources_exist": (FOLLOWUP / "02_CAF_actual_RT_values.csv").exists() and (FOLLOWUP / "03_CRF_actual_RT_values.csv").exists(),
        "time_scaling_preserves_skewness": abs(float(raw["skewness"] - scaled["skewness"])) < 1e-10,
        "time_scaling_preserves_cv": abs(float(raw["coefficient_of_variation"] - scaled["coefficient_of_variation"])) < 1e-10,
        "both_speed_accuracy_conditions_present": set(sat["condition"]) == {"static_competition", "target_recovery"},
        "recovery_caf_has_five_bins": len(recovery) == 5,
        "recovery_caf_slope_is_positive": recovery_slope > 0.0,
        "recovery_improves_caf_slope": recovery_slope > static_slope,
        "recovery_crossing_rate_exceeds_90_percent": float(sat[sat["condition"].eq("target_recovery")]["crossing_rate"].min()) > 0.90,
    }
    if not all(qa.values()):
        raise RuntimeError(qa)
    text = f"""# 第二轮导师问题检查

## CAF/CRF 横轴

- 已生成分面版本，每个分位箱的中位 RT 都直接显示为横轴刻度。
- 人类与模型分开成图，因此各自不同的 RT 刻度不会重叠或造成误读。

## 时间尺度

- 四选项校准模拟的原中位决策时间为 {raw['median_s']:.3f} 秒。
- 为只比较形状，将其乘以 {scaled['scale_factor']:.3f}，中位数变为 {scaled['median_s']:.3f} 秒。
- 缩放前后偏度均为 {raw['skewness']:.3f}，变异系数均为 {raw['coefficient_of_variation']:.3f}；因此纯乘法不改变相对分布形状。
- 该 0.60 秒目标只是可视化示例，不是重新拟合出的参数。

## Speed–accuracy 检查

- 比较了静态四选项竞争和“早期 flanker 占优、随后 target 恢复”的四选项输入时序。
- 静态条件的 CAF 斜率为 {static_slope:.3f}，target-recovery 条件为 {recovery_slope:.3f}（正值表示慢反应更准确）。
- 这是合成机制检查，不是使用真实 VGG 证据重新训练后的完整模型结果。

## 二选项建议

- 不建议把最终任务直接改成“一致/不一致”二选项，因为一致性是刺激条件，不是被试要作出的方向反应。
- 可以把 target-vs-flanker 两累积器作为不一致试次的诊断对照，但它无法表示一致试次，也无法保留 left/right/up/down 四种反应和 other-error。
"""
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> None:
    setup()
    plot_explicit_rt_ticks()
    scale = scale_decision_time_shape()
    sat, caf = simulate_speed_accuracy()
    qa = write_summary(scale, sat, caf)
    print(json.dumps({"output": str(OUT), "qa": qa}, indent=2))


if __name__ == "__main__":
    main()

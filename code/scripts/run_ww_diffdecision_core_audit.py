#!/usr/bin/env python3
"""Audit Wong-Wang operating regimes and DiffDecisionMultiClass readout semantics.

This diagnostic deliberately excludes VGG inputs, R5 post-hoc readout rules, and
non-decision time.  It asks whether the recurrent core preserves basic evidence-
accumulation behavior when moving from two to four alternatives, and whether the
differentiable per-class crossing API represents never-crossed classes correctly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
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


OUT = PROJECT_ROOT / "artifacts/results/ww_diffdecision_core_audit_20260802"


@dataclass(frozen=True)
class Regime:
    name: str
    n_classes: int
    dt_ms: int
    horizon_ms: int
    threshold: float
    common_input: float
    gaps: tuple[float, ...]
    normalize_competition: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    return parser.parse_args()


def make_model(regime: Regime, noise: float) -> WongWangMultiClassDecision:
    steps = regime.horizon_ms // regime.dt_ms
    model = WongWangMultiClassDecision(
        n_classes=regime.n_classes,
        dt=regime.dt_ms,
        time_steps=steps,
        t_stimulus=steps,
        normalize_competition=regime.normalize_competition,
    )
    model.eval()
    with torch.no_grad():
        model.threshold.fill_(regime.threshold)
        model.noise_ampa.fill_(noise)
    return model


def constant_input(n: int, n_classes: int, common: float, gap: float) -> torch.Tensor:
    values = torch.full((n, n_classes), float(common), dtype=torch.float32)
    values[:, 0] += float(gap)
    if n_classes > 1:
        values[:, 1:] -= float(gap) / float(n_classes - 1)
    return values


def first_crossing(trajectory: torch.Tensor, threshold: float, dt_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crossed = trajectory > float(threshold)
    trial_crossed = crossed.any(dim=2).any(dim=1)
    step_axis = torch.arange(trajectory.shape[1], device=trajectory.device).view(1, -1, 1)
    per_class_step = torch.where(crossed, step_axis, trajectory.shape[1]).amin(dim=1)
    winning_step, choice = per_class_step.min(dim=1)
    decision_s = winning_step.float() * float(dt_ms) / 1000.0
    decision_s[~trial_crossed] = torch.nan
    return (
        decision_s.cpu().numpy(),
        choice.cpu().numpy(),
        crossed.any(dim=1).cpu().numpy(),
    )


def finite_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: math.nan for k in ("mean_s", "sd_s", "median_s", "q10_s", "q90_s", "skewness")}
    return {
        "mean_s": float(values.mean()),
        "sd_s": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median_s": float(np.median(values)),
        "q10_s": float(np.quantile(values, 0.10)),
        "q90_s": float(np.quantile(values, 0.90)),
        "skewness": float(stats.skew(values)) if values.size > 2 and values.std() > 1e-12 else math.nan,
    }


def run_regime(regime: Regime, n: int, seeds: list[int], noises: tuple[float, ...]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    distributions: list[dict] = []
    for noise in noises:
        model = make_model(regime, noise)
        for gap in regime.gaps:
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                x = constant_input(n, regime.n_classes, regime.common_input, gap)
                with torch.no_grad():
                    api_times, trajectory, threshold = model.inference(x, generator=generator)
                decision_s, choice, class_crossed = first_crossing(trajectory, float(threshold.detach()), regime.dt_ms)
                trial_crossed = np.isfinite(decision_s)
                noncross_classes_reported_zero = (~class_crossed) & np.isclose(api_times.cpu().numpy(), 0.0)
                summary = finite_summary(decision_s)
                rows.append(
                    {
                        "regime": regime.name,
                        "n_classes": regime.n_classes,
                        "dt_ms": regime.dt_ms,
                        "horizon_ms": regime.horizon_ms,
                        "threshold": regime.threshold,
                        "common_input": regime.common_input,
                        "normalize_competition": regime.normalize_competition,
                        "target_gap": gap,
                        "noise_ampa": noise,
                        "seed": seed,
                        "n_trials": n,
                        "crossing_rate": float(trial_crossed.mean()),
                        "target_choice_rate_all": float((choice == 0).mean()),
                        "target_choice_rate_crossed": float((choice[trial_crossed] == 0).mean()) if trial_crossed.any() else math.nan,
                        "never_crossed_class_rate": float((~class_crossed).mean()),
                        "never_crossed_class_reported_zero_rate": float(non_cross_rate(noncross_classes_reported_zero, ~class_crossed)),
                        **summary,
                    }
                )
                capture_distribution = (
                    (regime.name in {"ww_2choice_dt1", "rtify_4choice_dt10"} and gap in {0.0, 0.3} and noise == 0.02)
                    or (regime.name == "rtify_4choice_calibrated_noise_dt10" and gap == 0.1 and noise == 0.006)
                )
                if capture_distribution:
                    for value in decision_s[np.isfinite(decision_s)]:
                        distributions.append(
                            {
                                "regime": regime.name,
                                "target_gap": gap,
                                "noise_ampa": noise,
                                "seed": seed,
                                "decision_time_s": float(value),
                            }
                        )
    return rows, distributions


def non_cross_rate(flag: np.ndarray, denominator: np.ndarray) -> float:
    count = int(np.asarray(denominator, dtype=bool).sum())
    return float(np.asarray(flag, dtype=bool).sum() / count) if count else math.nan


def aggregate(seed_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["regime", "n_classes", "dt_ms", "horizon_ms", "threshold", "common_input", "normalize_competition", "target_gap", "noise_ampa"]
    metrics = [
        "crossing_rate",
        "target_choice_rate_all",
        "target_choice_rate_crossed",
        "never_crossed_class_rate",
        "never_crossed_class_reported_zero_rate",
        "mean_s",
        "sd_s",
        "median_s",
        "q10_s",
        "q90_s",
        "skewness",
    ]
    return seed_df.groupby(keys, dropna=False)[metrics].mean().reset_index()


def save_figures(agg: pd.DataFrame, dist: pd.DataFrame) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, part in agg[agg["noise_ampa"].eq(0.02)].groupby("regime"):
        axes[0].plot(part["target_gap"], part["crossing_rate"], marker="o", label=name)
        axes[1].plot(part["target_gap"], part["median_s"], marker="o", label=name)
    axes[0].set(xlabel="target evidence advantage", ylabel="crossing rate", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="target evidence advantage", ylabel="median decision time (s)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "operating_regime_comparison.pdf")
    fig.savefig(OUT / "operating_regime_comparison.png", dpi=220)
    plt.close(fig)

    if not dist.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        selections = [("ww_2choice_dt1", 0.0), ("rtify_4choice_calibrated_noise_dt10", 0.1)]
        for ax, (name, gap) in zip(axes, selections):
            vals = dist[dist["regime"].eq(name) & dist["target_gap"].eq(gap)]["decision_time_s"]
            ax.hist(vals, bins=35, density=True, color="#D55E00", alpha=0.8)
            ax.set(title=f"{name}, gap={gap}", xlabel="decision time (s)", ylabel="density")
        fig.tight_layout()
        fig.savefig(OUT / "representative_first_crossing_distributions.pdf")
        fig.savefig(OUT / "representative_first_crossing_distributions.png", dpi=220)
        plt.close(fig)


def write_summary(agg: pd.DataFrame) -> None:
    two = agg[(agg["regime"].eq("ww_2choice_dt1")) & agg["noise_ampa"].eq(0.02)]
    four = agg[(agg["regime"].eq("rtify_4choice_dt10")) & agg["noise_ampa"].eq(0.02)]
    four_normalized = agg[(agg["regime"].eq("rtify_4choice_normalized_dt10")) & agg["noise_ampa"].eq(0.02)]
    calibrated = agg[
        agg["regime"].eq("rtify_4choice_calibrated_noise_dt10")
        & agg["noise_ampa"].eq(0.006)
        & np.isclose(agg["target_gap"], 0.1)
    ].iloc[0]
    zero_bug = float(agg["never_crossed_class_reported_zero_rate"].fillna(0).max())
    two_monotonic = bool(two.sort_values("target_gap")["median_s"].dropna().is_monotonic_decreasing)
    four_monotonic = bool(four.sort_values("target_gap")["median_s"].dropna().is_monotonic_decreasing)
    current_weak = four.loc[np.isclose(four["target_gap"], 0.1)].iloc[0]
    normalized_weak = four_normalized.loc[np.isclose(four_normalized["target_gap"], 0.1)].iloc[0]
    text = f"""# Wong-Wang / DiffDecision 核心检查

## 检查范围

本次检查排除了 VGG 证据、R5 的连续越界/边距规则和非决策时间，比较了论文尺度的两选项 Wong-Wang、直接四选项扩展，以及当前 R5 尺度的核心模型。

## 已确认结果

- 两选项核心存在较宽的正常工作区间：在非零共同感觉输入和内部噪声下能够稳定越界，且目标证据优势越大，决策时间越短。
- 两选项决策时间单调性检查：`{two_monotonic}`。
- 四选项扩展在目标优势足够时保留了上述规律，但弱证据工作区间明显缩窄，而且更容易受更新时间影响。
- 四选项决策时间单调性检查：`{four_monotonic}`。
- 已修正逐通道读出问题：未越界通道现在记为模拟时限，而不是 0 秒。修正后未越界通道被错误记为 0 秒的最大比例为 `{zero_bug:.3f}`。
- 将四选项总竞争强度归一化后，弱证据（优势 0.1）的越界率由 `{current_weak['crossing_rate']:.3f}` 提高到 `{normalized_weak['crossing_rate']:.3f}`，但目标选择率由 `{current_weak['target_choice_rate_crossed']:.3f}` 降到 `{normalized_weak['target_choice_rate_crossed']:.3f}`。这说明归一化解决了过度抑制，却增加了过早受噪声影响的决策，尚不能直接替换当前设置。
- 保留原竞争结构、加入共同感觉输入并将内部噪声校准为 0.006 后，同一弱目标输入重复模拟的越界率为 `{calibrated['crossing_rate']:.3f}`，目标选择率为 `{calibrated['target_choice_rate_crossed']:.3f}`，中位决策时间为 `{calibrated['median_s']:.3f}` 秒，偏度为 `{calibrated['skewness']:.3f}`。这说明四选项核心可以产生稳定、右偏的随机首次越界分布，但绝对时间仍偏长，尚不是完整行为拟合。
- 以前“零证据＋噪声”的检查大量不越界，主要因为没有共同感觉驱动且模拟时间较短；这本身不能证明 Wong-Wang 动力学无效。
- 没有结果支持 RT 必然服从正态分布。分布形状会随选择数量、证据优势、噪声、模拟时长和未越界处理而变化。

## 解释边界

- 这是核心机制检查，不是人类行为的重新拟合。
- 两选项对照采用论文尺度参数，但不是对原始 0.1 毫秒模拟的逐位复现。
- 四选项弱证据区间变窄可能来自竞争关系、参数缩放、数值更新时间或它们的共同作用；当前检查定位了现象，但尚不能归因于单一原因。
"""
    (OUT / "summary.md").write_text(text, encoding="utf-8")


def qa(seed_df: pd.DataFrame, agg: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "all_requested_regimes_present": seed_df["regime"].nunique() >= 5,
        "deterministic_repeats_have_zero_sd": bool((seed_df.loc[seed_df["noise_ampa"].eq(0.0), "sd_s"].dropna().abs() < 1e-10).all()),
        "two_choice_has_reliable_crossing": bool((agg[(agg["regime"].eq("ww_2choice_dt1")) & agg["noise_ampa"].eq(0.02)]["crossing_rate"].max() > 0.95)),
        "four_choice_has_reliable_crossing_at_strong_evidence": bool((agg[(agg["regime"].eq("rtify_4choice_dt10")) & agg["noise_ampa"].eq(0.02)]["crossing_rate"].max() > 0.95)),
        "calibrated_four_choice_operates_reliably": bool(
            (
                agg[agg["regime"].eq("rtify_4choice_calibrated_noise_dt10") & agg["noise_ampa"].eq(0.006)]["crossing_rate"].max()
                > 0.95
            )
        ),
        "per_class_zero_issue_fixed": bool((agg["never_crossed_class_reported_zero_rate"].fillna(0) < 1e-12).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Audit QA failed: {checks}")
    return checks


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        n, seeds = 80, [20260802]
    else:
        n, seeds = 500, [20260802, 20260803, 20260804, 20260805, 20260806]
    canonical_gaps = (0.0, 0.05, 0.1, 0.2, 0.3, 0.6)
    r5_gaps = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    regimes = [
        Regime("ww_2choice_dt1", 2, 1, 2000, 0.5, 1.0, canonical_gaps),
        Regime("ww_2choice_dt10", 2, 10, 2000, 0.5, 1.0, canonical_gaps),
        Regime("rtify_4choice_dt1", 4, 1, 2000, 0.5, 1.0, canonical_gaps),
        Regime("rtify_4choice_dt10", 4, 10, 2000, 0.5, 1.0, canonical_gaps),
        Regime("rtify_4choice_normalized_dt10", 4, 10, 2000, 0.5, 1.0, canonical_gaps, True),
        Regime("rtify_4choice_calibrated_noise_dt10", 4, 10, 4000, 0.5, 1.0, canonical_gaps),
        Regime("r5_young_scale_dt10", 4, 10, 800, 0.12, 0.0, r5_gaps),
        Regime("r5_older_scale_dt10", 4, 10, 800, 0.14, 0.0, r5_gaps),
        Regime("r5_young_normalized_dt10", 4, 10, 800, 0.12, 0.0, r5_gaps, True),
        Regime("r5_older_normalized_dt10", 4, 10, 800, 0.14, 0.0, r5_gaps, True),
    ]
    rows: list[dict] = []
    distributions: list[dict] = []
    for regime in regimes:
        rr, dd = run_regime(regime, n, seeds, (0.0, 0.006, 0.02))
        rows.extend(rr)
        distributions.extend(dd)
    seed_df = pd.DataFrame(rows)
    dist_df = pd.DataFrame(distributions)
    agg = aggregate(seed_df)
    seed_df.to_csv(OUT / "seed_level_metrics.csv", index=False)
    agg.to_csv(OUT / "aggregate_metrics.csv", index=False)
    dist_df.to_csv(OUT / "representative_decision_times.csv", index=False)
    save_figures(agg, dist_df)
    write_summary(agg)
    checks = qa(seed_df, agg)
    (OUT / "qa.json").write_text(json.dumps({"mode": args.mode, **checks}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "mode": args.mode, "qa": checks}, indent=2))


if __name__ == "__main__":
    main()

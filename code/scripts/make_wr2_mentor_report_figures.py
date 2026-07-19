#!/usr/bin/env python3
"""Create focused, report-ready WR2 figures without overwriting prior outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde

from project_paths import PROJECT_ROOT


BASE = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
WR2 = BASE / "wr2_uncertainty_schedule_fine_search"
METRICS = WR2 / "metrics"
DEFAULT_OUT = BASE / "wr2_mentor_report_figures_20260719_v2"
REC_ID = "WR2_fine_3744359a"
FINE_SCRIPT = WR2 / "scripts/run_wr2_uncertainty_schedule_fine_search.py"
GROUPS = ("young_20_29", "older_80_89")
GROUP_LABELS = {"young_20_29": "Young adults", "older_80_89": "Older adults"}
CONG_LABELS = {"congruent": "Congruent", "incongruent": "Incongruent"}
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#9A9A9A"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_recommended_trials() -> pd.DataFrame:
    path = METRICS / "wr2_fine_search_top_candidates_trial_level.csv"
    use = [
        "trial_id", "analysis_group", "congruency", "target_label", "flanker_label",
        "human_correct", "true_rt", "model_config_id", "model_correct", "model_rt",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=use, chunksize=100_000):
        q = chunk[chunk.model_config_id.eq(REC_ID)]
        if len(q):
            parts.append(q)
    out = pd.concat(parts, ignore_index=True)
    if len(out) != 10_000:
        raise RuntimeError(f"Expected 10,000 representative rows, found {len(out)}")
    # The representative subset is sampled with replacement, so original
    # trial_id values can legitimately recur. sample_row_id is the unique
    # identity of each sampled occurrence used by the saved WR2 evaluation.
    out.insert(0, "sample_row_id", np.arange(len(out), dtype=int))
    return out


def kde_line(values: pd.Series, grid: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < 5 or np.std(x) < 1e-8:
        return np.full_like(grid, np.nan)
    return gaussian_kde(x, bw_method="scott")(grid)


def plot_rt_kde(trial: pd.DataFrame, out: Path) -> pd.DataFrame:
    source_rows = []
    for source, rt_col in (("Human", "true_rt"), ("WR2", "model_rt")):
        q = trial[["sample_row_id", "trial_id", "analysis_group", "congruency", rt_col]].rename(columns={rt_col: "rt"})
        q["source"] = source
        source_rows.append(q)
    long = pd.concat(source_rows, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex="row", sharey="row")
    for i, group in enumerate(GROUPS):
        group_values = long[long.analysis_group.eq(group)].rt
        lo = max(0.20, float(group_values.quantile(0.002)) - 0.05)
        hi = min(2.50, float(group_values.quantile(0.998)) + 0.08)
        grid = np.linspace(lo, hi, 350)
        for j, source in enumerate(("Human", "WR2")):
            ax = axes[i, j]
            for cong, color, ls in (
                ("congruent", BLUE, "-"),
                ("incongruent", ORANGE, "--"),
            ):
                values = long[
                    long.analysis_group.eq(group)
                    & long.source.eq(source)
                    & long.congruency.eq(cong)
                ].rt
                ax.plot(grid, kde_line(values, grid), color=color, linestyle=ls, linewidth=2.2, label=CONG_LABELS[cong])
            if i == 0:
                ax.set_title(source)
            ax.text(0.02, 0.96, GROUP_LABELS[group], transform=ax.transAxes, ha="left", va="top", fontsize=10)
            ax.set_xlabel("Reaction time (s)")
            if j == 0:
                ax.set_ylabel("Density")
            ax.tick_params(direction="in")
    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle("Observed and WR2 reaction-time distributions", y=1.01)
    fig.tight_layout()
    save(fig, out, "fig01_wr2_rt_kde_human_vs_model")
    long.to_csv(out / "fig01_source_data.csv", index=False)
    return long


def load_recommended_uncertainty() -> pd.DataFrame:
    path = METRICS / "wr2_fine_search_uncertainty_schedule_diagnostics.csv"
    parts = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        q = chunk[chunk.model_config_id.eq(REC_ID)]
        if len(q):
            parts.append(q)
    return pd.concat(parts, ignore_index=True)


def plot_compression_bars(data: pd.DataFrame, out: Path) -> None:
    rng = np.random.default_rng(20260719)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True)
    summary = []
    for ax, group in zip(axes, GROUPS):
        q = data[data.analysis_group.eq(group)]
        for x, cong in enumerate(("congruent", "incongruent")):
            values = q[q.congruency.eq(cong)].compression_trial.to_numpy(float)
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            color = BLUE if cong == "congruent" else ORANGE
            ax.bar(x, mean, width=0.62, color=color, alpha=0.34, edgecolor=color, linewidth=1.0)
            ax.errorbar(x, mean, yerr=1.96 * se, color="black", capsize=4, linewidth=1.0, fmt="none")
            show = rng.choice(values, size=min(180, len(values)), replace=False)
            jitter = rng.normal(0, 0.055, len(show))
            ax.scatter(np.full(len(show), x) + jitter, show, s=10, color=color, alpha=0.28, edgecolors="none")
            summary.append({"analysis_group": group, "congruency": cong, "n": len(values), "mean": mean, "se": se})
        ax.set_xticks([0, 1], ["Congruent", "Incongruent"])
        ax.set_title(GROUP_LABELS[group])
        ax.set_xlabel("")
        ax.tick_params(direction="in")
    axes[0].set_ylabel("Trial-specific compression")
    fig.suptitle("WR2 timing adjustment by condition", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig02_wr2_compression_blank_background")
    pd.DataFrame(summary).to_csv(out / "fig02_source_data.csv", index=False)


def plot_candidate_landscape(out: Path) -> None:
    candidates = pd.read_csv(METRICS / "wr2_fine_search_all_candidates.csv")
    human = pd.read_csv(METRICS / "wr2_recommended_signature_summary.csv")
    human = human[human.source.eq("human")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    rows = []
    for ax, group in zip(axes, GROUPS):
        xcol = f"{group}_congruent_error_rate"
        ycol = f"{group}_incongruent_error_rate"
        ax.scatter(candidates[xcol], candidates[ycol], s=16, color=GRAY, alpha=0.34, edgecolors="none", label="WR2 candidates")
        rec = candidates[candidates.model_config_id.eq(REC_ID)].iloc[0]
        hcon = float(human[(human.analysis_group.eq(group)) & human.congruency.eq("congruent")].error_rate.iloc[0])
        hinc = float(human[(human.analysis_group.eq(group)) & human.congruency.eq("incongruent")].error_rate.iloc[0])
        ax.axvline(hcon, color="black", linewidth=0.8, linestyle=":")
        ax.axhline(hinc, color="black", linewidth=0.8, linestyle=":")
        ax.scatter([hcon], [hinc], marker="*", s=150, color="black", label="Human")
        ax.scatter([rec[xcol]], [rec[ycol]], marker="D", s=60, color=VERMILLION, label="Recommended WR2")
        ax.annotate("Recommended", (rec[xcol], rec[ycol]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=VERMILLION)
        ax.set_xlabel("Congruent error rate")
        ax.set_ylabel("Incongruent error rate")
        ax.set_title(GROUP_LABELS[group])
        ax.tick_params(direction="in")
        xvals = np.r_[candidates[xcol].to_numpy(float), hcon]
        yvals = np.r_[candidates[ycol].to_numpy(float), hinc]
        xpad = max(0.001, 0.08 * (xvals.max() - xvals.min()))
        ypad = max(0.005, 0.08 * (yvals.max() - yvals.min()))
        ax.set_xlim(xvals.min() - xpad, xvals.max() + xpad)
        ax.set_ylim(max(0, yvals.min() - ypad), yvals.max() + ypad)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        rows.append({"analysis_group": group, "source": "human", "congruent_error_rate": hcon, "incongruent_error_rate": hinc})
        rows.append({"analysis_group": group, "source": REC_ID, "congruent_error_rate": rec[xcol], "incongruent_error_rate": rec[ycol]})
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Behavioral trade-off across the WR2 candidate search", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig03_wr2_candidate_landscape_focused")
    pd.DataFrame(rows).to_csv(out / "fig03_highlighted_source_data.csv", index=False)


def load_fine_module():
    spec = importlib.util.spec_from_file_location("wr2_mentor_fine", FINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plot_noisy_ww_sensitivity(out: Path) -> None:
    fine = load_fine_module()
    model = fine.load_base_module()
    fine.install_wr2_override(model)
    data = model.load_inputs()
    row = pd.read_csv(METRICS / "wr2_fine_search_all_candidates.csv")
    params = json.loads(row[row.model_config_id.eq(REC_ID)].parameter_setting.iloc[0])
    spec = fine.make_wr2_spec(
        {
            **params["adaptive"],
            "late_shift_ms": params["schedule"]["late_shift_ms"],
            "early_phase_shortening_ms": params["schedule"]["early_phase_shortening_ms"],
            "transition_width": params["schedule"]["transition_width"],
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex="col", sharey="row", height_ratios=[2.0, 1.0])
    source_rows = []
    for column, group in enumerate(GROUPS):
        ax = axes[0, column]
        residual_ax = axes[1, column]
        mask = data["cache"]["analysis_group"].astype(str) == group
        layers = {k: v[mask] for k, v in data["norm_layers"].items()}
        target = data["cache"]["target_labels"][mask].astype(int)
        flanker = data["cache"]["flanker_labels"][mask].astype(int)
        mu, score, _, _ = model.build_candidate_mu(layers, target, flanker, float(data["group_params"][group]["evidence_gain"]), spec)
        incongruent = np.flatnonzero(target != flanker)
        chosen = incongruent[np.argmin(np.abs(score[incongruent] - np.median(score[incongruent])))]
        repeats = 12
        repeated = np.repeat(mu[chosen : chosen + 1], repeats, axis=0)
        noise = 0.003 if group == "young_20_29" else 0.006
        sim = model.run_ww(
            torch.as_tensor(repeated, dtype=torch.float32),
            time_steps=model.TIME_STEPS,
            dt_ms=int(model.DT * 1000),
            threshold=float(data["group_params"][group]["threshold"]),
            noise_ampa=noise,
            device="cpu",
            seed=20260719,
            readout_mode="baseline",
            t0_seconds=0.25,
            choice_temperature=0.01,
        )
        traj = np.asarray(sim["trajectory"], np.float32)
        times = np.arange(traj.shape[1]) * model.DT
        target_state = traj[:, :, target[chosen]]
        flanker_state = traj[:, :, flanker[chosen]]
        for k in range(repeats):
            ax.plot(times, target_state[k], color=BLUE, alpha=0.32, linewidth=0.8)
            ax.plot(times, flanker_state[k], color=ORANGE, alpha=0.32, linewidth=0.8)
            residual_ax.plot(times, target_state[k] - target_state.mean(axis=0), color=BLUE, alpha=0.34, linewidth=0.75)
            residual_ax.plot(times, flanker_state[k] - flanker_state.mean(axis=0), color=ORANGE, alpha=0.34, linewidth=0.75)
            for t, st, sf in zip(times, target_state[k], flanker_state[k]):
                source_rows.append({"analysis_group": group, "replicate": k, "time": t, "target_state": st, "flanker_state": sf, "noise_ampa": noise})
        ax.plot(times, target_state.mean(axis=0), color=BLUE, linewidth=2.2, label="Target (mean)")
        ax.plot(times, flanker_state.mean(axis=0), color=ORANGE, linewidth=2.2, linestyle="--", label="Flanker (mean)")
        ax.axhline(float(data["group_params"][group]["threshold"]), color="black", linewidth=0.9, linestyle=":", label="Boundary")
        ax.set_title(GROUP_LABELS[group])
        ax.tick_params(direction="in")
        residual_ax.axhline(0, color="black", linewidth=0.7)
        residual_ax.set_xlabel("Decision time (s)")
        residual_ax.tick_params(direction="in")
    axes[0, 0].set_ylabel("Wong–Wang state")
    axes[1, 0].set_ylabel("Deviation from mean")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Single-trial Wong–Wang variability: sensitivity analysis", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig04_ww_single_trial_noise_sensitivity")
    pd.DataFrame(source_rows).to_csv(out / "fig04_source_data.csv", index=False)


def write_readme(out: Path) -> None:
    text = """# WR2 导师汇报重绘图

- `fig01`: 人类与推荐 WR2 的 KDE 反应时分布；按年龄分行，Human/WR2 分列。
- `fig02`: 空白底的条件时间压缩图；柱为均值，误差线为 95% 正态近似区间，散点为抽样试次。
- `fig03`: 240 个候选的聚焦行为权衡图；黑星为人类，红菱形为推荐 WR2。
- `fig04`: 单试次 WW 内部噪声敏感性分析。注意：原始推荐 WR2 的 WW 内部 `noise_ampa=0`，因此该图不是原始拟合结果，而是加入后来诊断所用内部噪声后的机制敏感性图。细线为单次模拟，粗线为均值。

所有图均为白色背景，并输出 PNG、PDF、SVG 和对应源数据。旧结果图未被覆盖。
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    setup_style()
    trial = load_recommended_trials()
    plot_rt_kde(trial, args.output_dir)
    uncertainty = load_recommended_uncertainty()
    plot_compression_bars(uncertainty, args.output_dir)
    plot_candidate_landscape(args.output_dir)
    plot_noisy_ww_sensitivity(args.output_dir)
    write_readme(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()

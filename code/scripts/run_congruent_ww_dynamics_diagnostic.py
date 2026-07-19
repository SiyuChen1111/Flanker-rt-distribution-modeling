#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, q, safe_mean  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_representative_extreme_age_subset_fitting import (  # noqa: E402
    add_meta,
    apply_group_t0,
    load_trial_cache,
    run_base,
    subset_cache,
)


BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
BEST_DIR = BASE_DIR / "best_model_R5_combined_best/results"
OUT_DIR = BASE_DIR / "congruent_ww_dynamics_diagnostic"
GROUP_ORDER = ["young_20_29", "older_80_89"]
GROUP_LABEL = {"young_20_29": "Young 20-29", "older_80_89": "Older 80-89"}
COND_LABEL = {0: "congruent", 1: "incongruent"}
DT_MS = 10
TIME_STEPS = 80
SEED = 20260530


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose congruent-trial WW dynamics in the representative subset.")
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "figures": root / "figures",
        "figures_publication": root / "figures_publication",
        "summaries": root / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_md(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_group_params(param_path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    params = pd.read_csv(param_path)
    best = params[params["model_name"].eq("R5_combined_best")].copy()
    if best.empty:
        raise RuntimeError("R5_combined_best parameters were not found.")
    details = json.loads(str(best.iloc[0]["parameter_details"]))
    group_params = json.loads(details["group_params"]) if isinstance(details.get("group_params"), str) else details["group_params"]
    t0_mean = {str(r["analysis_group"]): float(r["t0_mean"]) for _, r in best.iterrows()}
    t0_sd = {str(r["analysis_group"]): float(r["t0_sd"]) for _, r in best.iterrows()}
    return group_params, t0_mean, t0_sd


def condition_name(congruency: int, correct: bool) -> str:
    return f"{COND_LABEL[int(congruency)]} {'correct' if bool(correct) else 'error'}"


def target_flanker_other(traj: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(traj.shape[0])
    s_target = traj[rows[:, None], np.arange(traj.shape[1])[None, :], target[:, None]]
    s_flanker = traj[rows[:, None], np.arange(traj.shape[1])[None, :], flanker[:, None]]
    other = traj.copy()
    other[rows, :, target] = -np.inf
    other[rows, :, flanker] = -np.inf
    s_other = np.max(other, axis=2)
    s_other[~np.isfinite(s_other)] = np.nan
    return s_target, s_flanker, s_other


def correct_competitor_other(traj: np.ndarray, correct_label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(traj.shape[0])
    s_correct = traj[rows[:, None], np.arange(traj.shape[1])[None, :], correct_label[:, None]]
    masked = traj.copy()
    masked[rows, :, correct_label] = -np.inf
    competitor_label = np.argmax(masked, axis=2)
    s_competitor = np.max(masked, axis=2)
    s_other = np.full_like(s_competitor, np.nan, dtype=np.float32)
    for i in range(traj.shape[0]):
        for t in range(traj.shape[1]):
            labels = [0, 1, 2, 3]
            labels.remove(int(correct_label[i]))
            labels.remove(int(competitor_label[i, t]))
            s_other[i, t] = float(np.max(traj[i, t, labels]))
    return s_correct, s_competitor, s_other, competitor_label


def readout_steps_from_df(df: pd.DataFrame, time_steps: int = TIME_STEPS) -> np.ndarray:
    if "decision_time" in df:
        steps = np.rint(df["decision_time"].to_numpy(float) / (DT_MS / 1000.0)).astype(int)
    else:
        steps = np.rint((df["pred_rt"].to_numpy(float) - df["t0_mean"].to_numpy(float)) / (DT_MS / 1000.0)).astype(int)
    return np.clip(steps, 0, time_steps - 1)


def first_crossing_steps(
    traj: np.ndarray,
    threshold: float,
    sustained_k: int,
    margin: float,
    min_decision_time: float,
    sustained: bool = True,
) -> np.ndarray:
    n, time_steps, _ = traj.shape
    top2 = np.sort(traj, axis=2)[:, :, -2:]
    runner = top2[:, :, 0]
    winner_state = top2[:, :, 1]
    winner = np.argmax(traj, axis=2)
    pass_mask = (winner_state > float(threshold)) & ((winner_state - runner) >= float(margin))
    min_step = int(round(float(min_decision_time) / (DT_MS / 1000.0)))
    if min_step > 0:
        pass_mask[:, :min_step] = False
    if sustained and sustained_k > 1:
        sustained_mask = np.zeros_like(pass_mask)
        for t in range(time_steps - int(sustained_k) + 1):
            sl = slice(t, t + int(sustained_k))
            sustained_mask[:, t] = np.all(pass_mask[:, sl], axis=1) & np.all(winner[:, sl] == winner[:, t : t + 1], axis=1)
        pass_mask = sustained_mask
    steps = np.argmax(pass_mask, axis=1).astype(int)
    steps[~pass_mask.any(axis=1)] = time_steps - 1
    return steps


def state_at_steps(arr: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return arr[np.arange(arr.shape[0]), steps]


def gap_rows(df: pd.DataFrame, traj: np.ndarray, steps: np.ndarray, threshold_label: str = "current") -> pd.DataFrame:
    target = df["target_label"].to_numpy(int)
    top2 = np.sort(traj[np.arange(traj.shape[0]), steps, :], axis=1)[:, -2:]
    winner = np.argmax(traj[np.arange(traj.shape[0]), steps, :], axis=1)
    s_correct, s_competitor, _, _ = correct_competitor_other(traj, target)
    out = pd.DataFrame(
        {
            "trial_id": df["row_index"].to_numpy(int),
            "analysis_group": df["analysis_group"].to_numpy(str),
            "congruency": [COND_LABEL[int(x)] for x in df["congruency"].to_numpy(int)],
            "model_correct": df["model_correct"].to_numpy(bool),
            "condition": [condition_name(c, ok) for c, ok in zip(df["congruency"], df["model_correct"])],
            "readout_step": steps,
            "readout_time": steps * (DT_MS / 1000.0),
            "model_rt": df["pred_rt"].to_numpy(float),
            "winner_label": winner,
            "target_label": target,
            "gap_at_readout": top2[:, 1] - top2[:, 0],
            "correct_competitor_gap_at_readout": state_at_steps(s_correct - s_competitor, steps),
            "threshold_level": threshold_label,
        }
    )
    return out


def summarize_gap(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in gaps.groupby(["analysis_group", "condition"], sort=False):
        rows.append(
            {
                "analysis_group": keys[0],
                "condition": keys[1],
                "n_trials": int(len(part)),
                "mean_gap_at_readout": safe_mean(part["gap_at_readout"]),
                "median_gap_at_readout": q(part["gap_at_readout"].to_numpy(), 0.5),
                "mean_correct_competitor_gap_at_readout": safe_mean(part["correct_competitor_gap_at_readout"]),
                "median_correct_competitor_gap_at_readout": q(part["correct_competitor_gap_at_readout"].to_numpy(), 0.5),
                "mean_readout_time": safe_mean(part["readout_time"]),
                "mean_model_rt": safe_mean(part["model_rt"]),
            }
        )
    return pd.DataFrame(rows)


def trajectory_tables(df: pd.DataFrame, traj: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = df["target_label"].to_numpy(int)
    flanker = df["flanker_label"].to_numpy(int)
    s_target, s_flanker, s_other = target_flanker_other(traj, target, flanker)
    s_correct, s_competitor, s_other_resp, _ = correct_competitor_other(traj, target)
    times = np.arange(traj.shape[1]) * (DT_MS / 1000.0)
    rows1: List[Dict[str, Any]] = []
    rows2: List[Dict[str, Any]] = []
    keep_conditions = ["congruent correct", "congruent error", "incongruent correct", "incongruent error"]
    for group in GROUP_ORDER:
        gmask = df["analysis_group"].eq(group).to_numpy()
        for cname in keep_conditions:
            cong = 0 if cname.startswith("congruent") else 1
            corr = cname.endswith("correct")
            mask = gmask & df["congruency"].eq(cong).to_numpy() & (df["model_correct"].to_numpy(bool) == corr)
            n = int(mask.sum())
            for i, t in enumerate(times):
                rows1.append(
                    {
                        "analysis_group": group,
                        "condition": cname,
                        "time": t,
                        "n_trials": n,
                        "s_target_mean": safe_mean(s_target[mask, i]) if n else math.nan,
                        "s_flanker_mean": safe_mean(s_flanker[mask, i]) if n else math.nan,
                        "s_other_max_mean": safe_mean(s_other[mask, i]) if n else math.nan,
                    }
                )
            if cname == "congruent correct" or cname == "congruent error":
                for i, t in enumerate(times):
                    gap = s_correct[:, i] - s_competitor[:, i]
                    rows2.append(
                        {
                            "analysis_group": group,
                            "condition": cname,
                            "time": t,
                            "n_trials": n,
                            "s_correct_response_mean": safe_mean(s_correct[mask, i]) if n else math.nan,
                            "s_competitor_response_mean": safe_mean(s_competitor[mask, i]) if n else math.nan,
                            "s_other_max_mean": safe_mean(s_other_resp[mask, i]) if n else math.nan,
                            "correct_competitor_gap_mean": safe_mean(gap[mask]) if n else math.nan,
                        }
                    )
    return pd.DataFrame(rows1), pd.DataFrame(rows2)


def threshold_counterfactual(
    df: pd.DataFrame,
    traj: np.ndarray,
    group_params: Dict[str, Dict[str, float]],
    t0_mean: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    levels = [1.0, 0.9, 0.8, 0.7, 0.6]
    target = df["target_label"].to_numpy(int)
    for group in GROUP_ORDER:
        gmask = df["analysis_group"].eq(group).to_numpy()
        idx = np.where(gmask)[0]
        gp = group_params[group]
        for factor in levels:
            th = float(gp["threshold"]) * factor
            steps = first_crossing_steps(
                traj[idx],
                threshold=th,
                sustained_k=int(gp["sustained_k"]),
                margin=float(gp["margin"]),
                min_decision_time=float(gp["min_decision_time"]),
                sustained=True,
            )
            states = traj[idx, steps, :]
            choice = np.argmax(states, axis=1)
            correct = choice == target[idx]
            top2 = np.sort(states, axis=1)[:, -2:]
            s_correct, s_competitor, _, _ = correct_competitor_other(traj[idx], target[idx])
            readout_time = steps * (DT_MS / 1000.0)
            model_rt = readout_time + float(t0_mean[group])
            for j, original_i in enumerate(idx):
                all_rows.append(
                    {
                        "trial_id": int(df.iloc[original_i]["row_index"]),
                        "analysis_group": group,
                        "threshold_factor": factor,
                        "threshold": th,
                        "threshold_level": "current_threshold" if factor == 1.0 else f"current_threshold_x_{factor:.1f}",
                        "congruency": COND_LABEL[int(df.iloc[original_i]["congruency"])],
                        "readout_step": int(steps[j]),
                        "readout_time": float(readout_time[j]),
                        "model_rt": float(model_rt[j]),
                        "model_choice": int(choice[j]),
                        "target_label": int(target[original_i]),
                        "model_correct": bool(correct[j]),
                        "gap_at_readout": float(top2[j, 1] - top2[j, 0]),
                        "correct_competitor_gap_at_readout": float((s_correct - s_competitor)[j, steps[j]]),
                        "s0_at_readout": float(states[j, 0]),
                        "s1_at_readout": float(states[j, 1]),
                        "s2_at_readout": float(states[j, 2]),
                        "s3_at_readout": float(states[j, 3]),
                    }
                )
    trial = pd.DataFrame(all_rows)
    summary_rows = []
    fast_rows = []
    for (factor, group, cond), part in trial.groupby(["threshold_factor", "analysis_group", "congruency"], sort=True):
        correct = part["model_correct"].to_numpy(bool)
        summary_rows.append(
            {
                "threshold_level": "current_threshold" if factor == 1.0 else f"current_threshold_x_{factor:.1f}",
                "threshold_factor": factor,
                "analysis_group": group,
                "condition": cond,
                "n_trials": int(len(part)),
                "mean_rt": safe_mean(part["model_rt"]),
                "accuracy": safe_mean(correct.astype(float)),
                "congruent_error_rate": float((~correct).mean()) if cond == "congruent" else math.nan,
                "incongruent_error_rate": float((~correct).mean()) if cond == "incongruent" else math.nan,
                "mean_gap_at_readout": safe_mean(part["gap_at_readout"]),
            }
        )
    for (factor, group), part in trial.groupby(["threshold_factor", "analysis_group"], sort=True):
        correct = part["model_correct"].to_numpy(bool)
        incong = part["congruency"].eq("incongruent").to_numpy()
        cong = part["congruency"].eq("congruent").to_numpy()
        fast_rows.append(
            {
                "threshold_level": "current_threshold" if factor == 1.0 else f"current_threshold_x_{factor:.1f}",
                "threshold_factor": factor,
                "analysis_group": group,
                "condition": "all",
                "mean_rt": safe_mean(part["model_rt"]),
                "accuracy": safe_mean(correct.astype(float)),
                "congruent_error_rate": safe_mean((~correct[cong]).astype(float)),
                "incongruent_error_rate": safe_mean((~correct[incong]).astype(float)),
                "overall_error_rt_minus_correct_rt": safe_mean(part.loc[~correct, "model_rt"]) - safe_mean(part.loc[correct, "model_rt"]),
                "within_incongruent_error_rt_minus_correct_rt": safe_mean(part.loc[incong & ~correct, "model_rt"]) - safe_mean(part.loc[incong & correct, "model_rt"]),
                "mean_gap_at_readout": safe_mean(part["gap_at_readout"]),
            }
        )
    summary = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(fast_rows)], ignore_index=True)
    fast_summary = []
    for (factor, group), part in trial.groupby(["threshold_factor", "analysis_group"], sort=True):
        cong = part[part["congruency"].eq("congruent")]
        ce = cong[~cong["model_correct"]]
        cc = cong[cong["model_correct"]]
        fast_summary.append(
            {
                "threshold_level": "current_threshold" if factor == 1.0 else f"current_threshold_x_{factor:.1f}",
                "threshold_factor": factor,
                "analysis_group": group,
                "n_congruent_trials": int(len(cong)),
                "n_congruent_errors": int(len(ce)),
                "congruent_error_rate": float(len(ce) / len(cong)) if len(cong) else math.nan,
                "congruent_error_mean_rt": safe_mean(ce["model_rt"]),
                "congruent_correct_mean_rt": safe_mean(cc["model_rt"]),
                "congruent_error_rt_minus_correct_rt": safe_mean(ce["model_rt"]) - safe_mean(cc["model_rt"]),
                "congruent_error_mean_gap": safe_mean(ce["gap_at_readout"]),
                "congruent_correct_mean_gap": safe_mean(cc["gap_at_readout"]),
                "early_readout_alone_produces_congruent_fast_errors": bool(len(ce) > 0 and safe_mean(ce["model_rt"]) < safe_mean(cc["model_rt"])),
            }
        )
    return trial, summary, pd.DataFrame(fast_summary)


def noise_pilot(counter_trial: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260601)
    base = counter_trial[counter_trial["threshold_factor"].eq(1.0)].copy()
    if base.empty:
        base = counter_trial.copy()
    rows = []
    rules = [
        ("Model 0 deterministic argmax", "det", 0.0, 0.0, 0.0, 0.05),
        ("Model 1 constant noisy readout", "normal", 0.002, 0.0, 0.0, 0.05),
        ("Model 1 constant noisy readout", "normal", 0.005, 0.0, 0.0, 0.05),
        ("Model 1 constant noisy readout", "normal", 0.010, 0.0, 0.0, 0.05),
        ("Model 2 uncertainty-dependent noisy readout", "normal", 0.001, 0.003, 0.004, 0.04),
        ("Model 2 uncertainty-dependent noisy readout", "normal", 0.001, 0.005, 0.008, 0.05),
        ("Model 3 uncertainty-dependent softmax", "softmax", 0.002, 0.004, 0.006, 0.04),
        ("Model 3 uncertainty-dependent softmax", "softmax", 0.003, 0.006, 0.010, 0.05),
    ]
    for name, mode, base_sigma, time_sigma, gap_sigma, gap_scale in rules:
        for group, part in base.groupby("analysis_group", sort=False):
            states = part[["s0_at_readout", "s1_at_readout", "s2_at_readout", "s3_at_readout"]].to_numpy(float)
            max_time = max(float(part["readout_time"].max()), 1e-6)
            earlyness = 1.0 - part["readout_time"].to_numpy(float) / max_time
            gap = part["gap_at_readout"].to_numpy(float)
            sigma = base_sigma + time_sigma * earlyness + gap_sigma * np.exp(-np.maximum(gap, 0.0) / gap_scale)
            target = part["target_label"].to_numpy(int)
            if mode == "det":
                choice = np.argmax(states, axis=1)
            elif mode == "normal":
                noisy = states + rng.normal(0.0, sigma[:, None], size=states.shape)
                choice = np.argmax(noisy, axis=1)
            else:
                temp = np.maximum(sigma, 1e-4)[:, None]
                logits = states / temp
                logits = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs = probs / probs.sum(axis=1, keepdims=True)
                cdf = np.cumsum(probs, axis=1)
                draws = rng.random(len(part))[:, None]
                choice = (draws > cdf).sum(axis=1)
            correct = choice == target
            incong = part["congruency"].eq("incongruent").to_numpy()
            cong = part["congruency"].eq("congruent").to_numpy()
            rows.append(
                {
                    "choice_rule": name,
                    "analysis_group": group,
                    "sigma_base_or_temp_base": base_sigma,
                    "sigma_time_or_temp_time": time_sigma,
                    "sigma_gap_or_temp_gap": gap_sigma,
                    "gap_scale": gap_scale,
                    "accuracy": safe_mean(correct.astype(float)),
                    "congruent_error_rate": safe_mean((~correct[cong]).astype(float)),
                    "incongruent_error_rate": safe_mean((~correct[incong]).astype(float)),
                    "congruent_error_rt_minus_correct_rt": safe_mean(part.loc[cong & ~correct, "model_rt"]) - safe_mean(part.loc[cong & correct, "model_rt"]),
                    "within_incongruent_error_rt_minus_correct_rt": safe_mean(part.loc[incong & ~correct, "model_rt"]) - safe_mean(part.loc[incong & correct, "model_rt"]),
                    "mean_gap_at_readout": safe_mean(part["gap_at_readout"]),
                    "n_trials": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def human_model_congruent_error(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in df[df["congruency"].eq(0)].groupby("analysis_group", sort=False):
        rows.append(
            {
                "analysis_group": group,
                "n_congruent_trials": int(len(part)),
                "human_congruent_error_rate": safe_mean((~part["human_correct"].to_numpy(bool)).astype(float)),
                "model_congruent_error_rate": safe_mean((~part["model_correct"].to_numpy(bool)).astype(float)),
                "human_congruent_error_n": int((~part["human_correct"].to_numpy(bool)).sum()),
                "model_congruent_error_n": int((~part["model_correct"].to_numpy(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def apa_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.grid(False)


def save_multi(fig: plt.Figure, path_no_ext: Path) -> None:
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(path_no_ext.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(summary: pd.DataFrame, out: Path) -> None:
    colors = {"S_target": "#0072B2", "S_flanker": "#D55E00", "S_other_max": "#666666"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
    conditions = ["congruent correct", "incongruent correct", "incongruent error"]
    for r, group in enumerate(GROUP_ORDER):
        for c, cond in enumerate(conditions):
            ax = axes[r, c]
            part = summary[summary["analysis_group"].eq(group) & summary["condition"].eq(cond)]
            for col, label in [("s_target_mean", "S_target"), ("s_flanker_mean", "S_flanker"), ("s_other_max_mean", "S_other_max")]:
                ax.plot(part["time"], part[col], label=label, color=colors[label], linewidth=1.8)
            n = int(part["n_trials"].max()) if len(part) else 0
            ax.set_title(f"{GROUP_LABEL[group]}: {cond} (n={n})", fontsize=10)
            ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_ylabel("WW state")
            apa_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    save_multi(fig, out)


def plot_correct_competitor(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True)
    for r, group in enumerate(GROUP_ORDER):
        part = summary[summary["analysis_group"].eq(group) & summary["condition"].eq("congruent correct")]
        axes[r, 0].plot(part["time"], part["s_correct_response_mean"], label="S_correct_response", color="#0072B2", linewidth=1.8)
        axes[r, 0].plot(part["time"], part["s_competitor_response_mean"], label="S_competitor_response", color="#D55E00", linewidth=1.8)
        axes[r, 0].plot(part["time"], part["s_other_max_mean"], label="S_other_max", color="#666666", linewidth=1.3, linestyle="--")
        axes[r, 1].plot(part["time"], part["correct_competitor_gap_mean"], color="#009E73", linewidth=1.8)
        axes[r, 0].set_title(f"{GROUP_LABEL[group]} channel states", fontsize=10)
        axes[r, 1].set_title(f"{GROUP_LABEL[group]} correct - competitor", fontsize=10)
        for ax in axes[r]:
            ax.set_xlabel("Time (s)")
            apa_axes(ax)
        axes[r, 0].set_ylabel("WW state")
        axes[r, 1].set_ylabel("Gap")
    axes[0, 0].legend(frameon=False, fontsize=9)
    save_multi(fig, out)


def plot_gap_distribution(gaps: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    order = ["congruent correct", "incongruent correct", "incongruent error"]
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for ax, group in zip(axes, GROUP_ORDER):
        data = [gaps.loc[gaps["analysis_group"].eq(group) & gaps["condition"].eq(cond), "gap_at_readout"].dropna().to_numpy() for cond in order]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        parts["cmeans"].set_color("black")
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_title(GROUP_LABEL[group], fontsize=10)
        ax.set_ylabel("Winner - competitor gap")
        apa_axes(ax)
    save_multi(fig, out)


def plot_threshold(summary: pd.DataFrame, out: Path) -> None:
    all_rows = summary[summary["condition"].eq("all")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for group in GROUP_ORDER:
        part = all_rows[all_rows["analysis_group"].eq(group)].sort_values("threshold_factor")
        label = GROUP_LABEL[group]
        axes[0].plot(part["threshold_factor"], part["mean_rt"], marker="o", label=label)
        axes[1].plot(part["threshold_factor"], part["congruent_error_rate"], marker="o", label=label)
        axes[2].plot(part["threshold_factor"], part["incongruent_error_rate"], marker="o", label=label)
    axes[0].set_ylabel("Mean RT (s)")
    axes[1].set_ylabel("Congruent error rate")
    axes[2].set_ylabel("Incongruent error rate")
    for ax in axes:
        ax.set_xlabel("Threshold factor")
        ax.invert_xaxis()
        apa_axes(ax)
    axes[0].legend(frameon=False, fontsize=9)
    save_multi(fig, out)


def plot_noise(noise: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        part = noise[noise["analysis_group"].eq(group)].copy()
        x = np.arange(len(part))
        ax.bar(x, part["congruent_error_rate"], color="#0072B2", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"M{i}" for i in range(len(part))], rotation=0)
        ax.set_title(GROUP_LABEL[group], fontsize=10)
        ax.set_ylabel("Congruent error rate")
        ax.set_xlabel("Noise pilot setting")
        apa_axes(ax)
    save_multi(fig, out)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    best_dir = base_dir / "best_model_R5_combined_best/results"
    dirs = ensure_dirs(base_dir / "congruent_ww_dynamics_diagnostic")

    group_params, t0_mean, t0_sd = parse_group_params(best_dir / "best_model_parameter_estimates.csv")
    cache = load_trial_cache(base_dir)
    all_df = []
    all_traj = []
    for group in GROUP_ORDER:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        gp = group_params[group]
        ns = argparse.Namespace(device=args.device, time_steps=TIME_STEPS, dt_ms=DT_MS, seed=SEED)
        cfg = ReadoutConfig(
            "sustained_crossing",
            min_decision_time=float(gp["min_decision_time"]),
            sustained_k=int(gp["sustained_k"]),
            margin=float(gp["margin"]),
        )
        df0, out0 = run_base(
            gc,
            ns,
            model_name="R5_combined_best",
            evidence_gain=float(gp["evidence_gain"]),
            threshold=float(gp["threshold"]),
            cfg=cfg,
        )
        df0 = apply_group_t0(df0, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)
        df0["analysis_group"] = group
        all_df.append(df0)
        all_traj.append(np.asarray(out0["trajectory"], dtype=np.float32))
    df = pd.concat(all_df, ignore_index=True)
    traj = np.concatenate(all_traj, axis=0)

    traj_summary, comp_summary = trajectory_tables(df, traj)
    traj_summary.to_csv(dirs["metrics"] / "congruent_vs_incongruent_trajectory_summary.csv", index=False)
    comp_summary.to_csv(dirs["metrics"] / "congruent_correct_competitor_gap_over_time.csv", index=False)

    steps = readout_steps_from_df(df)
    gaps = gap_rows(df, traj, steps)
    gaps.to_csv(dirs["metrics"] / "readout_gap_trial_level.csv", index=False)
    gap_summary = summarize_gap(gaps)
    gap_summary.to_csv(dirs["metrics"] / "readout_gap_by_group_condition.csv", index=False)

    cf_trial, cf_summary, fast_summary = threshold_counterfactual(df, traj, group_params, t0_mean)
    cf_trial.to_csv(dirs["metrics"] / "threshold_lowering_counterfactual_trial_level.csv", index=False)
    cf_summary.to_csv(dirs["metrics"] / "threshold_lowering_counterfactual_summary.csv", index=False)
    fast_summary.to_csv(dirs["metrics"] / "congruent_fast_error_counterfactual_summary.csv", index=False)

    hm = human_model_congruent_error(df)
    hm.to_csv(dirs["metrics"] / "human_model_congruent_error_rate.csv", index=False)

    noise = noise_pilot(cf_trial)
    noise.to_csv(dirs["metrics"] / "readout_choice_noise_pilot_summary.csv", index=False)

    plot_trajectories(traj_summary, dirs["figures_publication"] / "congruent_vs_incongruent_ww_trajectories")
    plot_correct_competitor(comp_summary, dirs["figures_publication"] / "congruent_correct_vs_competitor_trajectory")
    plot_gap_distribution(gaps, dirs["figures_publication"] / "readout_gap_distribution_by_condition")
    plot_threshold(cf_summary, dirs["figures_publication"] / "threshold_lowering_effect_on_congruent_errors")
    plot_noise(noise, dirs["figures_publication"] / "readout_choice_noise_pilot_summary")

    congruent_gap = gap_summary[gap_summary["condition"].eq("congruent correct")].set_index("analysis_group")
    incong_gap = gap_summary[gap_summary["condition"].eq("incongruent correct")].set_index("analysis_group")
    current_all = cf_summary[cf_summary["condition"].eq("all") & cf_summary["threshold_factor"].eq(1.0)].set_index("analysis_group")
    low_all = cf_summary[cf_summary["condition"].eq("all") & cf_summary["threshold_factor"].eq(0.6)].set_index("analysis_group")
    low_fast = fast_summary[fast_summary["threshold_factor"].eq(0.6)].set_index("analysis_group")
    noise_best = noise.sort_values(["congruent_error_rate", "accuracy"], ascending=[False, False]).head(1)

    summary_lines = [
        "# Congruent WW Dynamics Diagnostic Summary",
        "",
        "## 1. Goal",
        "This analysis diagnoses why the current representative-subset model almost never makes congruent-trial errors, and tests whether lowering the WW readout threshold alone can create congruent fast errors. It reuses existing trial-level predictions, cached layerwise evidence, and the fitted R5 group parameters; it does not re-extract VGG evidence or retrain CNN components.",
        "",
        "## 2. Congruent WW trajectories",
    ]
    for group in GROUP_ORDER:
        n_cong_err = int(((df["analysis_group"].eq(group)) & (df["congruency"].eq(0)) & (~df["model_correct"])).sum())
        cg = comp_summary[(comp_summary["analysis_group"].eq(group)) & (comp_summary["condition"].eq("congruent correct"))]
        early = cg[cg["time"].le(0.10)]["correct_competitor_gap_mean"].mean()
        late = cg[cg["time"].ge(0.50)]["correct_competitor_gap_mean"].mean()
        summary_lines.append(f"- {group}: model congruent error n = {n_cong_err}; mean correct-competitor gap is {early:.3f} in the first 100 ms and {late:.3f} after 500 ms.")
    summary_lines += [
        "",
        "## 3. Readout-time gap",
    ]
    for group in GROUP_ORDER:
        cg = float(congruent_gap.loc[group, "mean_gap_at_readout"])
        ig = float(incong_gap.loc[group, "mean_gap_at_readout"])
        summary_lines.append(f"- {group}: congruent correct winner-competitor gap at readout = {cg:.3f}; incongruent correct gap = {ig:.3f}.")
    summary_lines += [
        "",
        "## 4. Threshold-lowering counterfactual",
    ]
    for group in GROUP_ORDER:
        now_rt = float(current_all.loc[group, "mean_rt"])
        low_rt = float(low_all.loc[group, "mean_rt"])
        now_ce = float(current_all.loc[group, "congruent_error_rate"])
        low_ce = float(low_all.loc[group, "congruent_error_rate"])
        now_ie = float(current_all.loc[group, "incongruent_error_rate"])
        low_ie = float(low_all.loc[group, "incongruent_error_rate"])
        summary_lines.append(f"- {group}: lowering threshold to 0.6x changes mean RT from {now_rt:.3f}s to {low_rt:.3f}s, congruent error rate from {now_ce:.4f} to {low_ce:.4f}, and incongruent error rate from {now_ie:.4f} to {low_ie:.4f}.")
    summary_lines += [
        "",
        "## 5. Early readout and fast errors",
    ]
    for group in GROUP_ORDER:
        row = low_fast.loc[group]
        summary_lines.append(f"- {group}: at 0.6x threshold, new congruent errors = {int(row['n_congruent_errors'])}; early readout alone produces congruent fast errors = {bool(row['early_readout_alone_produces_congruent_fast_errors'])}.")
    summary_lines += [
        "",
        "## 6. Need for readout/choice noise",
        "Threshold lowering mainly moves readout earlier. When congruent evidence is aligned with the correct response, deterministic argmax still selects the correct channel. A conflict-independent or uncertainty-dependent choice/readout noise source is therefore the more plausible missing mechanism for human-like congruent errors.",
    ]
    if not noise_best.empty:
        row = noise_best.iloc[0]
        summary_lines.append(f"- The largest pilot congruent-error setting was `{row['choice_rule']}` in {row['analysis_group']}, with congruent error rate {row['congruent_error_rate']:.4f} and overall accuracy {row['accuracy']:.4f}. This is only a small proof-of-concept, not a refit.")
    summary_lines += [
        "",
        "## 7. Recommended interpretation",
        "之前主要检查的是 incongruent trials 的动态；congruent 条件需要单独看 correct response channel 和 competitor channel。现在的诊断显示，在 congruent trials 里，target 和 flanker 支持同一个正确反应，correct channel 很早就压过 competitor，readout 时 winner-competitor gap 也很稳定。因此模型几乎不犯 congruent 错误，不是因为读出太晚，而是因为证据对模型来说太一致。单纯降低 threshold 主要会让模型更早做对，不能自然地产生 congruent fast errors。如果要接近人类那种少量 congruent 错误，更合理的下一步是在 readout/choice 端加入和早读出、小 gap、不确定性绑定的噪声，而不是随机硬加错误。",
    ]
    write_md(dirs["summaries"] / "congruent_ww_dynamics_diagnostic_summary.md", summary_lines)

    print(f"Wrote diagnostic outputs to {dirs['root']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from run_gated_readout_simulation import GROUP_LABEL, GROUPS, OUT_DIR, reconstruct, state_metrics  # noqa: E402

DT = 0.01
LATE_POINTS = [0.00, 0.05, 0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trajectory viability diagnostic on reconstructed trial-level trajectories.")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def ensure_dirs() -> Dict[str, Path]:
    dirs = {
        "metrics": OUT_DIR / "metrics",
        "figures": OUT_DIR / "figures_publication",
        "summaries": OUT_DIR / "summaries",
        "logs": OUT_DIR / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_mean(x: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def safe_q(x: Iterable[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else math.nan


def corr_safe(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return math.nan
    if np.isclose(a[mask].std(), 0.0) or np.isclose(b[mask].std(), 0.0):
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT_DIR / "figures_publication" / f"{name}.{ext}", dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax.tick_params(labelsize=9)


def choice_type(choice: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    return np.where(choice == target, "target", np.where(choice == flanker, "flanker", "other"))


def deterministic_rows(base: pd.DataFrame, traj: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = base["target_label"].to_numpy(int)
    flanker = base["flanker_label"].to_numpy(int)
    original_steps = np.rint(base["decision_time"].to_numpy(float) / DT).astype(int)
    original_steps = np.clip(original_steps, 0, traj.shape[1] - 1)
    rows: List[pd.DataFrame] = []
    label_to_steps = {f"original_plus_{int(round(delay * 1000)):03d}ms": original_steps + int(round(delay / DT)) for delay in LATE_POINTS}
    label_to_steps["final_time_point"] = np.full(len(base), traj.shape[1] - 1, dtype=int)

    post_steps = np.arange(traj.shape[1])[None, :]
    post_mask = post_steps >= original_steps[:, None]
    post_states = np.where(post_mask[:, :, None], traj, -np.inf)
    target_vals = post_states[np.arange(len(base))[:, None], np.arange(traj.shape[1])[None, :], target[:, None]]
    masked = post_states.copy()
    masked[np.arange(len(base))[:, None], np.arange(traj.shape[1])[None, :], target[:, None]] = -np.inf
    other_max = masked.max(axis=2)
    margins = np.where(post_mask, target_vals - other_max, -np.inf)
    best_steps = margins.argmax(axis=1).astype(int)
    label_to_steps["best_possible_post_readout"] = best_steps

    for label, steps in label_to_steps.items():
        steps = np.clip(steps, 0, traj.shape[1] - 1)
        states = traj[np.arange(len(base)), steps, :]
        met = state_metrics(states, target, flanker)
        det = states.argmax(axis=1)
        part = pd.DataFrame(
            {
                "trial_id": base["row_index"].to_numpy(int),
                "analysis_group": base["analysis_group"].to_numpy(str),
                "congruency": base["congruency_label"].to_numpy(str),
                "human_correct": base["human_correct"].to_numpy(bool),
                "human_rt": base["true_rt"].to_numpy(float),
                "model_rt": base["pred_rt"].to_numpy(float),
                "target_label": target,
                "flanker_label": flanker,
                "readout_variant": label,
                "readout_time": steps * DT,
                "deterministic_choice": det,
                "deterministic_correct": det == target,
                "deterministic_choice_type": choice_type(det, target, flanker),
                "s_target_minus_flanker": met["s_target"] - met["s_flanker"],
                "s_target_minus_max_other": met["signed_target_margin"],
                "target_rank": met["target_rank"],
                "step_index": steps,
            }
        )
        rows.append(part)

    trial = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for (variant, group, cong), part in trial.groupby(["readout_variant", "analysis_group", "congruency"], sort=False):
        err = ~part["deterministic_correct"].to_numpy(bool)
        prop = part["deterministic_choice_type"].value_counts(normalize=True)
        summary_rows.append(
            {
                "readout_variant": variant,
                "analysis_group": group,
                "congruency": cong,
                "n_trials": int(len(part)),
                "deterministic_accuracy": safe_mean(part["deterministic_correct"].astype(float)),
                "congruent_error_rate": safe_mean(err.astype(float)) if cong == "congruent" else math.nan,
                "incongruent_error_rate": safe_mean(err.astype(float)) if cong == "incongruent" else math.nan,
                "choice_type_target_proportion": float(prop.get("target", 0.0)),
                "choice_type_flanker_proportion": float(prop.get("flanker", 0.0)),
                "choice_type_other_proportion": float(prop.get("other", 0.0)),
                "s_target_minus_flanker_mean": safe_mean(part["s_target_minus_flanker"]),
                "s_target_minus_max_other_mean": safe_mean(part["s_target_minus_max_other"]),
                "target_rank_mean": safe_mean(part["target_rank"]),
                "target_rank_1_proportion": safe_mean(part["target_rank"].eq(1).astype(float)),
            }
        )
    return trial, pd.DataFrame(summary_rows)


def first_true_index(mask: np.ndarray) -> np.ndarray:
    idx = mask.argmax(axis=1)
    any_true = mask.any(axis=1)
    out = np.full(mask.shape[0], np.nan, dtype=float)
    out[any_true] = idx[any_true]
    return out


def crossing_diagnostic(base: pd.DataFrame, traj: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = base["target_label"].to_numpy(int)
    flanker = base["flanker_label"].to_numpy(int)
    original_steps = np.rint(base["decision_time"].to_numpy(float) / DT).astype(int)
    original_steps = np.clip(original_steps, 0, traj.shape[1] - 1)
    n, time_steps, _ = traj.shape
    rows = np.arange(n)[:, None]
    times = np.arange(time_steps)[None, :]
    post_mask = times >= original_steps[:, None]
    target_vals = traj[rows, times, target[:, None]]
    flanker_vals = traj[rows, times, flanker[:, None]]
    masked = traj.copy()
    masked[np.arange(n)[:, None], np.arange(time_steps)[None, :], target[:, None]] = -np.inf
    other_max = masked.max(axis=2)
    order = np.argsort(-traj, axis=2, kind="mergesort")
    target_rank = np.empty((n, time_steps), dtype=int)
    for i in range(n):
        target_rank[i] = np.where(order[i] == target[i])[1] + 1

    post_rank1 = post_mask & (target_rank == 1)
    post_gt_flanker = post_mask & (target_vals > flanker_vals)
    post_gt_other = post_mask & (target_vals > other_max)
    first_rank1 = first_true_index(post_rank1)
    first_gt_flanker = first_true_index(post_gt_flanker)
    first_gt_other = first_true_index(post_gt_other)
    post_margin = np.where(post_mask, target_vals - other_max, -np.inf)
    post_tf = np.where(post_mask, target_vals - flanker_vals, -np.inf)
    best_margin_step = post_margin.argmax(axis=1).astype(int)

    trial = pd.DataFrame(
        {
            "trial_id": base["row_index"].to_numpy(int),
            "analysis_group": base["analysis_group"].to_numpy(str),
            "congruency": base["congruency_label"].to_numpy(str),
            "human_correct": base["human_correct"].to_numpy(bool),
            "human_rt": base["true_rt"].to_numpy(float),
            "model_rt": base["pred_rt"].to_numpy(float),
            "original_readout_time": original_steps * DT,
            "target_ever_rank1_after_readout": np.isfinite(first_rank1),
            "target_first_rank1_time": first_rank1 * DT,
            "target_ever_gt_flanker_after_readout": np.isfinite(first_gt_flanker),
            "target_first_gt_flanker_time": first_gt_flanker * DT,
            "target_ever_gt_max_other_after_readout": np.isfinite(first_gt_other),
            "target_first_gt_max_other_time": first_gt_other * DT,
            "maximum_post_readout_target_margin": np.where(np.isfinite(post_margin).any(axis=1), np.max(post_margin, axis=1), np.nan),
            "maximum_post_readout_target_minus_flanker": np.where(np.isfinite(post_tf).any(axis=1), np.max(post_tf, axis=1), np.nan),
            "best_possible_post_readout_time": best_margin_step * DT,
        }
    )
    for col in ["target_first_rank1_time", "target_first_gt_flanker_time", "target_first_gt_max_other_time"]:
        trial[f"{col}_before_human_rt"] = trial[col] <= trial["human_rt"]
        trial[f"{col}_before_model_rt"] = trial[col] <= trial["model_rt"]
        trial[f"{col}_delay_from_readout"] = trial[col] - trial["original_readout_time"]
    summary_rows = []
    for (group, cong), part in trial.groupby(["analysis_group", "congruency"], sort=False):
        summary_rows.append(
            {
                "analysis_group": group,
                "congruency": cong,
                "n_trials": int(len(part)),
                "target_ever_rank1_after_readout_proportion": safe_mean(part["target_ever_rank1_after_readout"].astype(float)),
                "target_first_rank1_time_mean": safe_mean(part["target_first_rank1_time"]),
                "target_ever_gt_flanker_after_readout_proportion": safe_mean(part["target_ever_gt_flanker_after_readout"].astype(float)),
                "target_first_gt_flanker_time_mean": safe_mean(part["target_first_gt_flanker_time"]),
                "target_ever_gt_max_other_after_readout_proportion": safe_mean(part["target_ever_gt_max_other_after_readout"].astype(float)),
                "target_first_gt_max_other_time_mean": safe_mean(part["target_first_gt_max_other_time"]),
                "maximum_post_readout_target_margin_mean": safe_mean(part["maximum_post_readout_target_margin"]),
                "maximum_post_readout_target_minus_flanker_mean": safe_mean(part["maximum_post_readout_target_minus_flanker"]),
                "crossing_rank1_before_human_rt_proportion": safe_mean(part["target_first_rank1_time_before_human_rt"].astype(float)),
                "crossing_rank1_before_model_rt_proportion": safe_mean(part["target_first_rank1_time_before_model_rt"].astype(float)),
                "crossing_gt_other_before_human_rt_proportion": safe_mean(part["target_first_gt_max_other_time_before_human_rt"].astype(float)),
                "crossing_gt_other_before_model_rt_proportion": safe_mean(part["target_first_gt_max_other_time_before_model_rt"].astype(float)),
            }
        )
    return trial, pd.DataFrame(summary_rows)


def human_alignment(cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in cross.groupby("analysis_group", sort=False):
        incong = part["congruency"].eq("incongruent")
        for scope_name, mask in [("all", np.ones(len(part), dtype=bool)), ("incongruent", incong.to_numpy())]:
            scoped = part.loc[mask].copy()
            hc = scoped["human_correct"].to_numpy(bool)
            rank1 = scoped["target_ever_rank1_after_readout"].to_numpy(bool)
            gt_other = scoped["target_ever_gt_max_other_after_readout"].to_numpy(bool)
            rows.append(
                {
                    "analysis_group": group,
                    "scope": scope_name,
                    "n_trials": int(len(scoped)),
                    "human_correct_target_rank1_proportion": safe_mean(rank1[hc].astype(float)),
                    "human_error_target_rank1_proportion": safe_mean(rank1[~hc].astype(float)),
                    "human_correct_target_gt_other_proportion": safe_mean(gt_other[hc].astype(float)),
                    "human_error_target_gt_other_proportion": safe_mean(gt_other[~hc].astype(float)),
                    "target_rank1_status_predicts_human_correctness": safe_mean(hc[rank1].astype(float)) - safe_mean(hc[~rank1].astype(float)),
                    "target_gt_other_status_predicts_human_correctness": safe_mean(hc[gt_other].astype(float)) - safe_mean(hc[~gt_other].astype(float)),
                    "target_first_rank1_time_vs_human_rt_corr": corr_safe(scoped["target_first_rank1_time"], scoped["human_rt"]),
                    "target_first_gt_max_other_time_vs_human_rt_corr": corr_safe(scoped["target_first_gt_max_other_time"], scoped["human_rt"]),
                    "target_crossing_delay_vs_human_correct_corr": corr_safe(scoped["target_first_gt_max_other_time_delay_from_readout"], scoped["human_correct"].astype(float)),
                    "incongruent_human_correct_cross_before_human_rt_proportion": safe_mean(scoped.loc[scoped["congruency"].eq("incongruent") & scoped["human_correct"], "target_first_gt_max_other_time_before_human_rt"].astype(float)),
                }
            )
    return pd.DataFrame(rows)


def make_over_time_margin(base: pd.DataFrame, traj: np.ndarray) -> pd.DataFrame:
    target = base["target_label"].to_numpy(int)
    rows = np.arange(traj.shape[0])[:, None]
    times = np.arange(traj.shape[1])[None, :]
    target_vals = traj[rows, times, target[:, None]]
    masked = traj.copy()
    masked[np.arange(traj.shape[0])[:, None], np.arange(traj.shape[1])[None, :], target[:, None]] = -np.inf
    other_max = masked.max(axis=2)
    margins = target_vals - other_max
    out_rows = []
    for group in GROUPS:
        gmask = base["analysis_group"].eq(group).to_numpy()
        for correctness in [True, False]:
            cmask = base["human_correct"].to_numpy(bool) == correctness
            mask = gmask & cmask
            for t in range(traj.shape[1]):
                out_rows.append(
                    {
                        "analysis_group": group,
                        "human_correct": correctness,
                        "time": t * DT,
                        "signed_target_margin_mean": safe_mean(margins[mask, t]),
                    }
                )
    return pd.DataFrame(out_rows)


def plot_late_readout(summary: pd.DataFrame) -> None:
    order = [f"original_plus_{int(round(x * 1000)):03d}ms" for x in LATE_POINTS] + ["final_time_point", "best_possible_post_readout"]
    label = {name: ("best" if name == "best_possible_post_readout" else "final" if name == "final_time_point" else name.replace("original_plus_", "+").replace("ms", " ms")) for name in order}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cong in zip(axes, ["congruent", "incongruent"]):
        for group in GROUPS:
            part = summary[(summary["analysis_group"].eq(group)) & (summary["congruency"].eq(cong))].copy()
            part["x"] = part["readout_variant"].map({k: i for i, k in enumerate(order)})
            part = part.sort_values("x")
            ax.plot(part["x"], part["deterministic_accuracy"], marker="o", label=GROUP_LABEL[group])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([label[x] for x in order], rotation=35, ha="right")
        ax.set_title(cong.capitalize())
        ax.set_ylabel("Deterministic accuracy")
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "late_readout_accuracy_curve_by_condition")

    fig, ax = plt.subplots(figsize=(8, 4))
    for group in GROUPS:
        part = summary[(summary["analysis_group"].eq(group)) & (summary["congruency"].eq("incongruent"))].copy()
        part["x"] = part["readout_variant"].map({k: i for i, k in enumerate(order)})
        part = part.sort_values("x")
        ax.plot(part["x"], part["incongruent_error_rate"], marker="o", label=GROUP_LABEL[group])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[x] for x in order], rotation=35, ha="right")
    ax.set_ylabel("Incongruent error rate")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, "late_readout_incongruent_error_reduction_curve")


def plot_crossings(cross_summary: pd.DataFrame, cross_trial: pd.DataFrame, margin_time: pd.DataFrame, late_trial: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    part = cross_summary[cross_summary["congruency"].eq("incongruent")]
    x = np.arange(len(GROUPS))
    vals = [part[part["analysis_group"].eq(g)]["target_ever_rank1_after_readout_proportion"].mean() for g in GROUPS]
    ax.bar(x, vals, color=["#4C78A8", "#F58518"])
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    ax.set_ylabel("Proportion ever rank 1")
    style_ax(ax)
    save_fig(fig, "target_ever_rank1_proportion_by_condition")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        part = cross_trial[(cross_trial["analysis_group"].eq(group)) & (cross_trial["congruency"].eq("incongruent"))]
        ax.hist(part["target_first_gt_max_other_time"].dropna(), bins=25, color="#4C78A8", alpha=0.75)
        ax.set_title(GROUP_LABEL[group])
        ax.set_xlabel("First crossing time (s)")
        style_ax(ax)
    axes[0].set_ylabel("Trials")
    save_fig(fig, "target_crossing_time_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        for correctness, color, label in [(True, "#4C78A8", "Human correct"), (False, "#F58518", "Human error")]:
            part = margin_time[(margin_time["analysis_group"].eq(group)) & (margin_time["human_correct"].eq(correctness))]
            ax.plot(part["time"], part["signed_target_margin_mean"], label=label, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(GROUP_LABEL[group])
        ax.set_xlabel("Time (s)")
        style_ax(ax)
    axes[0].set_ylabel("Mean signed target margin")
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "target_margin_over_time_human_correct_vs_error")

    best = late_trial[late_trial["readout_variant"].eq("best_possible_post_readout")]
    orig = late_trial[late_trial["readout_variant"].eq("original_plus_000ms")][["trial_id", "s_target_minus_max_other"]].rename(columns={"s_target_minus_max_other": "original_margin"})
    merged = best.merge(orig, on="trial_id", how="left")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(merged["original_margin"], merged["s_target_minus_max_other"], s=4, alpha=0.15, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Original signed margin")
    ax.set_ylabel("Best post-readout signed margin")
    style_ax(ax)
    save_fig(fig, "target_margin_original_vs_best_possible_post_readout")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    order = ["original_plus_000ms", "original_plus_050ms", "original_plus_100ms", "original_plus_150ms", "original_plus_200ms", "final_time_point", "best_possible_post_readout"]
    for ax, group in zip(axes, GROUPS):
        sub = late_trial[(late_trial["analysis_group"].eq(group)) & (late_trial["congruency"].eq("incongruent"))]
        props = sub.groupby(["readout_variant", "deterministic_choice_type"]).size().rename("n").reset_index()
        props["proportion"] = props["n"] / props.groupby("readout_variant")["n"].transform("sum")
        pivot = props.pivot(index="readout_variant", columns="deterministic_choice_type", values="proportion").reindex(order).fillna(0)
        for col in ["target", "flanker", "other"]:
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot[["target", "flanker", "other"]].plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518", "#9E9E9E"])
        ax.set_title(GROUP_LABEL[group])
        ax.set_xticklabels([x.replace("original_plus_", "+").replace("ms", "ms").replace("best_possible_post_readout", "best").replace("final_time_point", "final") for x in order], rotation=35, ha="right")
        style_ax(ax)
    axes[0].set_ylabel("Choice proportion")
    save_fig(fig, "choice_type_shift_with_late_readout")

    fig, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for group in GROUPS:
        sub = cross_trial[cross_trial["analysis_group"].eq(group)]
        for status in [True, False]:
            part = sub[sub["target_ever_gt_max_other_after_readout"].eq(status)]
            rows.append({"analysis_group": group, "status": "crossed" if status else "never crossed", "human_correct_rate": safe_mean(part["human_correct"].astype(float))})
    plot = pd.DataFrame(rows)
    x = np.arange(len(GROUPS))
    width = 0.35
    ax.bar(x - width / 2, plot[plot["status"].eq("crossed")]["human_correct_rate"], width=width, label="Crossed")
    ax.bar(x + width / 2, plot[plot["status"].eq("never crossed")]["human_correct_rate"], width=width, label="Never crossed")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    ax.set_ylabel("Human correct rate")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    save_fig(fig, "human_correctness_by_target_crossing_status")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        part = cross_trial[(cross_trial["analysis_group"].eq(group)) & (cross_trial["congruency"].eq("incongruent"))]
        ax.scatter(part["target_first_gt_max_other_time_delay_from_readout"], part["human_rt"], s=5, alpha=0.2, color="#4C78A8")
        ax.set_title(GROUP_LABEL[group])
        ax.set_xlabel("Crossing delay from readout (s)")
        style_ax(ax)
    axes[0].set_ylabel("Human RT (s)")
    save_fig(fig, "target_crossing_delay_vs_human_rt")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    inc = cross_summary[cross_summary["congruency"].eq("incongruent")]
    axes[0, 0].bar(np.arange(len(GROUPS)), [inc[inc["analysis_group"].eq(g)]["target_ever_rank1_after_readout_proportion"].mean() for g in GROUPS], color="#4C78A8")
    axes[0, 0].set_xticks(np.arange(len(GROUPS)))
    axes[0, 0].set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    axes[0, 0].set_title("Ever rank 1 after readout")
    style_ax(axes[0, 0])
    late_best = late_trial[late_trial["readout_variant"].eq("best_possible_post_readout") & late_trial["congruency"].eq("incongruent")]
    acc = late_best.groupby("analysis_group")["deterministic_correct"].mean()
    axes[0, 1].bar(np.arange(len(GROUPS)), [acc[g] for g in GROUPS], color="#F58518")
    axes[0, 1].set_xticks(np.arange(len(GROUPS)))
    axes[0, 1].set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    axes[0, 1].set_title("Best possible post-readout accuracy")
    style_ax(axes[0, 1])
    for group in GROUPS:
        part = late_trial[(late_trial["analysis_group"].eq(group)) & (late_trial["congruency"].eq("incongruent"))]
        curve = part.groupby("readout_variant")["deterministic_correct"].mean().reindex(order)
        axes[1, 0].plot(range(len(order)), curve.values, marker="o", label=GROUP_LABEL[group])
    axes[1, 0].set_xticks(range(len(order)))
    axes[1, 0].set_xticklabels(["0", "50", "100", "150", "200", "final", "best"], rotation=30)
    axes[1, 0].set_title("Late-readout recovery")
    style_ax(axes[1, 0])
    axes[1, 0].legend(frameon=False, fontsize=8)
    rows2 = []
    for group in GROUPS:
        part = cross_trial[(cross_trial["analysis_group"].eq(group)) & (cross_trial["congruency"].eq("incongruent"))]
        rows2.append(safe_mean(part["target_first_gt_max_other_time_before_human_rt"].astype(float)))
    axes[1, 1].bar(np.arange(len(GROUPS)), rows2, color="#54A24B")
    axes[1, 1].set_xticks(np.arange(len(GROUPS)))
    axes[1, 1].set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    axes[1, 1].set_title("Crossing before human RT")
    style_ax(axes[1, 1])
    save_fig(fig, "target_recovery_viability_dashboard")


def write_summary(late_summary: pd.DataFrame, cross_summary: pd.DataFrame, align: pd.DataFrame) -> None:
    inc = late_summary[late_summary["congruency"].eq("incongruent")].copy()
    cross_inc = cross_summary[cross_summary["congruency"].eq("incongruent")].copy()
    orig = inc[inc["readout_variant"].eq("original_plus_000ms")]
    plus200 = inc[inc["readout_variant"].eq("original_plus_200ms")]
    finalp = inc[inc["readout_variant"].eq("final_time_point")]
    best = inc[inc["readout_variant"].eq("best_possible_post_readout")]
    align_inc = align[align["scope"].eq("incongruent")]
    lines = [
        "# Trajectory viability diagnostic summary",
        "",
        "## Core answer",
        "",
        f"- Late readout does not fully repair the incongruent failure. Relative to original readout, +200 ms improves deterministic incongruent accuracy from {1 - orig['incongruent_error_rate'].mean():.4f} to {1 - plus200['incongruent_error_rate'].mean():.4f}, but the failure remains large.",
        f"- Even at the final available time point, deterministic incongruent accuracy is {1 - finalp['incongruent_error_rate'].mean():.4f}.",
        f"- The best possible post-readout upper bound reaches {1 - best['incongruent_error_rate'].mean():.4f} deterministic incongruent accuracy. This is the ceiling available inside the current trajectory family if commitment timing were idealized.",
        "",
        "## Recovery viability",
        "",
        f"- Mean proportion of incongruent trials where target ever becomes rank 1 after original readout: {cross_inc['target_ever_rank1_after_readout_proportion'].mean():.4f}.",
        f"- Mean proportion of incongruent trials where target ever exceeds flanker after original readout: {cross_inc['target_ever_gt_flanker_after_readout_proportion'].mean():.4f}.",
        f"- Mean proportion of incongruent trials where target ever exceeds max other after original readout: {cross_inc['target_ever_gt_max_other_after_readout_proportion'].mean():.4f}.",
        "",
        "## Human alignment",
        "",
        f"- In incongruent trials, target-crossing status predicts human correctness by {align_inc['target_gt_other_status_predicts_human_correctness'].mean():.4f} on average.",
        f"- Correlation between target crossing time and human RT in incongruent trials: {align_inc['target_first_gt_max_other_time_vs_human_rt_corr'].mean():.4f}.",
        f"- In human-correct incongruent trials, target crossing occurs before human RT with mean proportion {align_inc['incongruent_human_correct_cross_before_human_rt_proportion'].mean():.4f}.",
        "",
        "## Interpretation",
        "",
        "- If late readout had strongly repaired incongruent accuracy, commitment timing would still be the main lever. That is not what the current trajectories show.",
        "- Because even final-time and best-post-readout analyses leave a large residual failure, the bottleneck is not just premature commitment. The underlying task-relevant evidence recovery is often too weak or too late.",
        "- This points more toward evidence mapping / WW target-recovery dynamics than toward a pure gating-fit next step.",
        "- The most defensible next model direction is to modify evidence dynamics or the target-recovery mechanism first, then revisit commitment timing after the trajectories themselves are more viable.",
    ]
    (OUT_DIR / "summaries/trajectory_viability_diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base, traj = reconstruct()
    late_trial, late_summary = deterministic_rows(base, traj)
    late_trial.to_csv(OUT_DIR / "metrics/trajectory_late_readout_upper_bound_trial_level.csv", index=False)
    late_summary.to_csv(OUT_DIR / "metrics/trajectory_late_readout_upper_bound_summary.csv", index=False)

    cross_trial, cross_summary = crossing_diagnostic(base, traj)
    cross_trial.to_csv(OUT_DIR / "metrics/trajectory_target_crossing_diagnostic_trial_level.csv", index=False)
    cross_summary.to_csv(OUT_DIR / "metrics/trajectory_target_crossing_diagnostic_summary.csv", index=False)

    align = human_alignment(cross_trial)
    align.to_csv(OUT_DIR / "metrics/trajectory_target_crossing_human_alignment.csv", index=False)

    margin_time = make_over_time_margin(base, traj)
    plot_late_readout(late_summary)
    plot_crossings(cross_summary, cross_trial, margin_time, late_trial)
    write_summary(late_summary, cross_summary, align)


if __name__ == "__main__":
    main()

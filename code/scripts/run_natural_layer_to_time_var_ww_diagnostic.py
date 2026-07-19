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
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww, target_flanker_trajectory
from compare_same_subset_layerwise_vs_dmc import (
    evidence_sources,
    make_trial_df,
    no_crossing_rate,
    summarize_trials,
)
from complete_layerwise_dmc_remaining_diagnostics import build_hand_dmc_input
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import to_jsonable


LAYER_ORDER = ["conv3", "conv4", "conv5", "pooled", "final"]
BASELINE_LABELS = {
    "final_only": "final_logits_ww",
    "middle_only": "mid_layer_ww",
    "pooled_only": "pooled_ww",
    "refined_layer_time_gate": "refined_best_layerwise_gate",
    "handcrafted_dmc_positive_control": "handcrafted_dmc_final_ww",
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cache(path: Path, max_trials: int) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    n = min(int(max_trials), len(data["target_labels"]))
    return {key: data[key][:n] for key in data.files}


def raw_layer_arrays(cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "conv3": np.asarray(cache["evidence_conv3"], dtype=np.float32),
        "conv4": np.asarray(cache["evidence_conv4"], dtype=np.float32),
        "conv5": np.asarray(cache["evidence_conv5"], dtype=np.float32),
        "pooled": np.asarray(cache["evidence_pooled"], dtype=np.float32),
        "final": np.asarray(cache["evidence_final"], dtype=np.float32),
    }


def normalize_layers(raw: Dict[str, np.ndarray], method: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for layer, arr in raw.items():
        x = np.asarray(arr, dtype=np.float32)
        if method == "no_norm":
            out[layer] = x.copy()
        elif method == "per_layer_zscore":
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            out[layer] = (x - mean) / std
        elif method == "per_layer_gap_scale":
            centered = x - x.mean(axis=1, keepdims=True)
            class_std = centered.std(axis=1, keepdims=True)
            class_std[class_std < 1e-6] = 1.0
            scale = float(np.mean(class_std))
            scale = 1.0 if scale < 1e-6 else scale
            out[layer] = centered / scale
        else:
            raise ValueError(f"Unknown normalization: {method}")
    return out


def schedule_weights(schedule_type: str, time_steps: int) -> pd.DataFrame:
    t = np.linspace(0.0, 1.0, time_steps, endpoint=False, dtype=np.float32)
    weights = np.zeros((time_steps, len(LAYER_ORDER)), dtype=np.float32)

    if schedule_type == "natural_hard_5stage":
        bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for idx in range(len(LAYER_ORDER)):
            mask = (t >= bounds[idx]) & (t < bounds[idx + 1])
            weights[mask, idx] = 1.0
        weights[-1, -1] = 1.0
    elif schedule_type == "natural_smooth_5stage":
        centers = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32)
        sigma = 0.12
        basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / sigma) ** 2)
        basis_sum = basis.sum(axis=1, keepdims=True)
        basis_sum[basis_sum < 1e-6] = 1.0
        weights = basis / basis_sum
    elif schedule_type == "natural_refined_3stage":
        early = t < (1.0 / 3.0)
        middle = (t >= (1.0 / 3.0)) & (t < (2.0 / 3.0))
        late = t >= (2.0 / 3.0)
        weights[early, 0] = 1.0
        weights[middle, 1] = 0.5
        weights[middle, 2] = 0.5
        weights[late, 3] = 0.5
        weights[late, 4] = 0.5
        weights[-1, 4] = 0.5
        weights[-1, 3] = 0.5
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return pd.DataFrame(weights, columns=LAYER_ORDER)


def build_mu_schedule(
    normalized_layers: Dict[str, np.ndarray],
    schedule_df: pd.DataFrame,
    evidence_gain: float,
) -> torch.Tensor:
    n_trials = next(iter(normalized_layers.values())).shape[0]
    n_classes = next(iter(normalized_layers.values())).shape[1]
    time_steps = len(schedule_df)
    mu = np.zeros((n_trials, time_steps, n_classes), dtype=np.float32)
    for layer in LAYER_ORDER:
        w = schedule_df[layer].to_numpy(dtype=np.float32)
        mu += normalized_layers[layer][:, None, :] * w[None, :, None]
    mu *= float(evidence_gain)
    return torch.as_tensor(mu, dtype=torch.float32)


def build_sigma(
    mu: np.ndarray,
    schedule_df: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    sigma_type: str,
    sigma_base: float,
    sigma_middle: float,
    sigma_conflict: float,
) -> np.ndarray:
    sigma = np.full_like(mu, float(sigma_base), dtype=np.float32)
    if sigma_type == "fixed_sigma":
        return sigma
    if sigma_type == "layer_weighted_sigma":
        early_weight = schedule_df["conv3"].to_numpy(dtype=np.float32) + 0.5 * schedule_df["conv4"].to_numpy(dtype=np.float32)
        sigma += float(sigma_middle) * early_weight[None, :, None]
        return sigma
    if sigma_type == "conflict_dependent_sigma":
        targets = cache["target_labels"].astype(np.int64)
        flankers = cache["flanker_labels"].astype(np.int64)
        rows = np.arange(mu.shape[0])[:, None]
        times = np.arange(mu.shape[1])[None, :]
        target_vals = mu[rows, times, targets[:, None]]
        flanker_vals = mu[rows, times, flankers[:, None]]
        conflict = np.maximum(flanker_vals - target_vals, 0.0)
        denom = np.quantile(conflict, 0.95) if np.any(conflict > 0.0) else 1.0
        denom = 1.0 if denom < 1e-6 else float(denom)
        sigma += float(sigma_conflict) * np.clip(conflict / denom, 0.0, 1.0)[:, :, None]
        return sigma
    raise ValueError(f"Unknown sigma_type: {sigma_type}")


def sample_mu_sigma(mu: torch.Tensor, sigma: np.ndarray, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
    sigma_t = torch.as_tensor(sigma, dtype=mu.dtype)
    return mu + sigma_t * eps


def summarize_condition_extended(
    *,
    condition: str,
    family: str,
    variant_type: str,
    schedule_type: str,
    normalization: str,
    sigma_type: str,
    sigma_base: float,
    sigma_middle: float,
    sigma_conflict: float,
    seed: int | str,
    n_seed_repeats: int,
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    time_steps: int,
    dt_ms: int,
    t0_seconds: float,
) -> Dict[str, Any]:
    row = summarize_trials(
        condition,
        df,
        family,
        {
            "variant_type": variant_type,
            "schedule_type": schedule_type,
            "normalization": normalization,
            "sigma_type": sigma_type,
            "sigma_base": sigma_base,
            "sigma_middle": sigma_middle,
            "sigma_conflict": sigma_conflict,
            "seed": seed,
            "n_seed_repeats": n_seed_repeats,
        },
    )
    max_rt = (time_steps - 1) * (dt_ms / 1000.0) + float(t0_seconds)
    row["q95_capped"] = float(min(float(row["q95"]), max_rt))
    row["q99_capped"] = float(min(float(row["q99"]), max_rt))
    row["no_crossing_rate"] = no_crossing_rate(outputs, dt_ms, time_steps, float(t0_seconds))
    row["summary_level"] = "per_seed"
    return row


def trajectory_summary_rows_extended(
    *,
    condition: str,
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    dt_ms: int,
    extra: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    target = cache["target_labels"].astype(np.int64)
    flanker = cache["flanker_labels"].astype(np.int64)
    s_target, s_flanker, s_other = target_flanker_trajectory(outputs["trajectory"], target, flanker)
    early_steps = max(1, int(round(0.20 * s_target.shape[1])))
    late_steps = early_steps
    groups = {
        "congruent_correct": df["congruency"].eq(0) & df["model_correct"],
        "congruent_error": df["congruency"].eq(0) & ~df["model_correct"],
        "incongruent_correct": df["congruency"].eq(1) & df["model_correct"],
        "incongruent_error": df["congruency"].eq(1) & ~df["model_correct"],
        "human_fast_rt": df["true_rt"] <= df["true_rt"].median(),
        "human_slow_rt": df["true_rt"] > df["true_rt"].median(),
        "model_fast_rt": df["pred_rt"] <= df["pred_rt"].median(),
        "model_slow_rt": df["pred_rt"] > df["pred_rt"].median(),
    }
    rows: List[Dict[str, Any]] = []
    curves: Dict[str, np.ndarray] = {}
    for group_name, mask_series in groups.items():
        mask = mask_series.to_numpy(dtype=bool)
        base = {"condition": condition, "group": group_name, "n_trials": int(mask.sum()), **extra}
        if not mask.any():
            rows.append(base)
            continue
        early_target = s_target[mask, :early_steps].mean(axis=1)
        early_flanker = s_flanker[mask, :early_steps].mean(axis=1)
        late_target = s_target[mask, -late_steps:].mean(axis=1)
        late_flanker = s_flanker[mask, -late_steps:].mean(axis=1)
        rows.append(
            {
                **base,
                "early_s_target_mean": float(early_target.mean()),
                "early_s_flanker_mean": float(early_flanker.mean()),
                "early_s_other_max_mean": float(s_other[mask, :early_steps].mean()),
                "early_s_target_minus_flanker_mean": float((early_target - early_flanker).mean()),
                "early_flanker_ge_target_rate": float((early_flanker >= early_target).mean()),
                "late_s_target_mean": float(late_target.mean()),
                "late_s_flanker_mean": float(late_flanker.mean()),
                "late_s_other_max_mean": float(s_other[mask, -late_steps:].mean()),
                "late_s_target_minus_flanker_mean": float((late_target - late_flanker).mean()),
                "late_target_ge_flanker_rate": float((late_target >= late_flanker).mean()),
                "final_s_target_minus_flanker": float((s_target[mask, -1] - s_flanker[mask, -1]).mean()),
                "peak_target": float(s_target[mask].max(axis=1).mean()),
                "peak_flanker": float(s_flanker[mask].max(axis=1).mean()),
            }
        )
        curves[f"{condition}:{group_name}:target"] = s_target[mask].mean(axis=0)
        curves[f"{condition}:{group_name}:flanker"] = s_flanker[mask].mean(axis=0)
        curves[f"{condition}:{group_name}:other"] = s_other[mask].mean(axis=0)
    return rows, curves


def build_baseline_inputs(
    cache: Dict[str, np.ndarray],
    time_steps: int,
    dt_ms: int,
) -> Dict[str, torch.Tensor]:
    ev = evidence_sources(cache, gain=0.8)
    baseline: Dict[str, torch.Tensor] = {
        "final_only": ev["final"].unsqueeze(1).repeat(1, time_steps, 1),
        "middle_only": ev["mid"].unsqueeze(1).repeat(1, time_steps, 1),
        "pooled_only": ev["pooled"].unsqueeze(1).repeat(1, time_steps, 1),
    }
    dmc_input, _ = build_hand_dmc_input(
        ev["final"],
        cache["target_labels"],
        cache["flanker_labels"],
        time_steps=time_steps,
        dt_ms=dt_ms,
        auto_strength=0.30,
        selection_strength=0.40,
        target_boost=0.30,
        auto_peak_s=0.06,
        selection_midpoint_s=0.18,
        selection_tau_s=0.06,
    )
    baseline["handcrafted_dmc_positive_control"] = dmc_input
    return baseline


def load_existing_baseline_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    same = pd.read_csv(
        resolve_path("artifacts/results/diagnostics/same_subset_layerwise_vs_dmc/same_subset_model_summary.csv")
    )
    refined = pd.read_csv(
        resolve_path("artifacts/results/diagnostics/refined_layerwise_vs_dmc_same_subset/refined_vs_dmc_summary.csv")
    )
    return same, refined


def baseline_row_from_existing(input_condition: str, same: pd.DataFrame, refined: pd.DataFrame) -> pd.Series:
    if input_condition == "final_only":
        return refined[refined["condition"].eq("final_logits_ww")].iloc[0]
    if input_condition == "middle_only":
        return refined[refined["condition"].eq("mid_layer_ww")].iloc[0]
    if input_condition == "pooled_only":
        return refined[refined["condition"].eq("pooled_ww")].iloc[0]
    if input_condition == "refined_layer_time_gate":
        return refined[refined["condition"].eq("refined_best_layerwise_gate")].iloc[0]
    if input_condition == "handcrafted_dmc_positive_control":
        return refined[refined["condition"].eq("handcrafted_dmc_positive_control")].iloc[0]
    raise KeyError(input_condition)


def aggregate_seed_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        "n_trials",
        "accuracy",
        "human_accuracy",
        "model_human_choice_agreement",
        "mean_rt",
        "median_rt",
        "q90",
        "q95",
        "q99",
        "skewness",
        "q95_capped",
        "q99_capped",
        "no_crossing_rate",
        "congruent_rt",
        "incongruent_rt",
        "congruency_rt_gap",
        "correct_rt",
        "error_rt",
        "error_minus_correct_rt",
        "incongruent_correct_rt",
        "incongruent_error_rt",
        "incongruent_error_minus_correct_rt",
        "incongruent_error_rate",
        "fastest_bin_accuracy",
        "fastest_incongruent_bin_accuracy",
        "human_congruent_rt",
        "human_incongruent_rt",
        "human_congruency_rt_gap",
        "human_error_minus_correct_rt",
    ]
    records: List[Dict[str, Any]] = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for col in numeric_cols:
            if col in part.columns:
                row[col] = float(pd.to_numeric(part[col], errors="coerce").mean())
        row["seed"] = "avg"
        row["summary_level"] = "seed_avg"
        row["n_seed_repeats"] = int(part["seed"].nunique()) if "seed" in part.columns else 1
        records.append(row)
    return pd.DataFrame(records)


def choose_top_candidates(
    natural_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    max_rt: float,
) -> pd.DataFrame:
    middle_acc = float(
        baseline_summary.loc[baseline_summary["input_condition"].eq("middle_only"), "accuracy"].iloc[0]
    )
    final_incong_err = float(
        baseline_summary.loc[baseline_summary["input_condition"].eq("final_only"), "incongruent_error_rate"].iloc[0]
    )
    refined_rt = float(
        baseline_summary.loc[baseline_summary["input_condition"].eq("refined_layer_time_gate"), "mean_rt"].iloc[0]
    )
    candidates = natural_summary.copy()
    candidates["passes_min_accuracy"] = candidates["accuracy"] > middle_acc
    candidates["passes_conflict"] = candidates["incongruent_error_rate"] > final_incong_err
    candidates["not_extremely_slow"] = candidates["mean_rt"] < max(1.25, refined_rt * 1.8)
    candidates["q95_not_capped"] = candidates["q95"] < max_rt - 1e-3
    candidates["q99_not_capped"] = candidates["q99"] < max_rt - 1e-3
    candidates["acceptable"] = (
        candidates["passes_min_accuracy"]
        & candidates["passes_conflict"]
        & candidates["not_extremely_slow"]
        & (candidates["q95_not_capped"] | candidates["q99_not_capped"])
    )
    candidates["selection_score"] = (
        1.2 * candidates["accuracy"]
        + 0.7 * candidates["model_human_choice_agreement"]
        + 0.4 * candidates["incongruent_error_rate"]
        - 0.25 * candidates["mean_rt"]
        - 0.20 * candidates["q95_capped"]
        - 0.15 * candidates["q99_capped"]
        - 0.10 * candidates["no_crossing_rate"].fillna(0.0)
        + 0.15 * (-candidates["fastest_incongruent_bin_accuracy"])
    )
    top = candidates.sort_values(
        ["acceptable", "selection_score"],
        ascending=[False, False],
    ).copy()
    return top.head(12)


def canonical_baseline_summary(same_existing: pd.DataFrame, refined_existing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [
        ("final_only", refined_existing[refined_existing["condition"].eq("final_logits_ww")].iloc[0]),
        ("middle_only", refined_existing[refined_existing["condition"].eq("mid_layer_ww")].iloc[0]),
        ("pooled_only", refined_existing[refined_existing["condition"].eq("pooled_ww")].iloc[0]),
        ("refined_layer_time_gate", refined_existing[refined_existing["condition"].eq("refined_best_layerwise_gate")].iloc[0]),
        ("handcrafted_dmc_positive_control", refined_existing[refined_existing["condition"].eq("handcrafted_dmc_positive_control")].iloc[0]),
    ]
    for input_condition, row in mapping:
        item = row.to_dict()
        item["input_condition"] = input_condition
        rows.append(item)
    return pd.DataFrame(rows)


def plot_schedule_weights(schedule_map: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    for schedule_type, df in schedule_map.items():
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        t = np.arange(len(df))
        for layer in LAYER_ORDER:
            ax.plot(t, df[layer], linewidth=2, label=layer)
        ax.set_xlabel("time step")
        ax.set_ylabel("weight")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(schedule_type)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        safe = schedule_type.replace("natural_", "schedule_weights_")
        fig.savefig(output_dir / f"{safe}.png", dpi=220)
        plt.close(fig)


def plot_behavior_scatter(comparison: pd.DataFrame, output_dir: Path) -> None:
    plot_specs = [
        ("accuracy", "incongruent_error_rate", "behavior_accuracy_vs_incongruent_error_rate.png"),
        ("mean_rt", "accuracy", "behavior_mean_rt_vs_accuracy.png"),
        ("mean_rt", "human_choice_agreement", "behavior_human_agreement_vs_rt.png"),
    ]
    colors = {"baseline": "#F58518", "natural_det": "#4C78A8", "natural_var": "#54A24B"}
    for x, y, fname in plot_specs:
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        for model_family, part in comparison.groupby("model_family"):
            ax.scatter(part[x], part[y], s=50, alpha=0.85, color=colors.get(model_family, "#999999"), label=model_family)
            for _, row in part.iterrows():
                ax.annotate(str(row["input_condition"]), (row[x], row[y]), fontsize=7, alpha=0.75)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=220)
        plt.close(fig)


def plot_rt_distribution(trial_df: pd.DataFrame, conditions: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for condition in conditions:
        part = trial_df[trial_df["condition"].eq(condition)]
        if part.empty:
            continue
        ax.hist(part["pred_rt"], bins=28, density=True, histtype="step", linewidth=1.6, label=condition)
    ax.set_xlabel("Predicted RT (s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rt_quantiles(summary_df: pd.DataFrame, conditions: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    q_labels = ["q90", "q95", "q99"]
    x = np.arange(len(conditions))
    width = 0.22
    for idx, q in enumerate(q_labels):
        vals = []
        for condition in conditions:
            part = summary_df[summary_df["condition"].eq(condition)]
            vals.append(float(part.iloc[0][q]) if not part.empty else np.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, label=q)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("RT (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectory(curves: Dict[str, np.ndarray], condition: str, group: str, dt_ms: int, output_path: Path) -> None:
    target_key = f"{condition}:{group}:target"
    flanker_key = f"{condition}:{group}:flanker"
    if target_key not in curves or flanker_key not in curves:
        return
    t = np.arange(len(curves[target_key])) * (dt_ms / 1000.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t, curves[target_key], linewidth=1.8, label="target")
    ax.plot(t, curves[flanker_key], linewidth=1.8, label="flanker")
    other_key = f"{condition}:{group}:other"
    if other_key in curves:
        ax.plot(t, curves[other_key], linewidth=1.2, linestyle="--", label="other max")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wong-Wang state")
    ax.set_title(f"{condition}: {group}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_target_minus_flanker(curves: Dict[str, np.ndarray], condition: str, groups: Iterable[str], dt_ms: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for group in groups:
        target_key = f"{condition}:{group}:target"
        flanker_key = f"{condition}:{group}:flanker"
        if target_key not in curves or flanker_key not in curves:
            continue
        diff = curves[target_key] - curves[flanker_key]
        t = np.arange(len(diff)) * (dt_ms / 1000.0)
        ax.plot(t, diff, linewidth=1.8, label=group)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("s_target - s_flanker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_baseline_comparison(
    baseline_reference: pd.DataFrame,
    top_candidates: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    output_path: Path,
    same_existing: pd.DataFrame,
    refined_existing: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    traj_lookup = trajectory_summary.set_index(["condition", "group"])
    for input_condition in baseline_reference["input_condition"].tolist():
        existing = baseline_row_from_existing(str(input_condition), same_existing, refined_existing)
        cond = str(existing["condition"])
        rows.append(
            {
                "input_condition": str(input_condition),
                "model_family": "baseline",
                "accuracy": existing["accuracy"],
                "human_choice_agreement": existing["model_human_choice_agreement"],
                "mean_rt": existing["mean_rt"],
                "q95": existing["q95"],
                "q99": existing["q99"],
                "q95_capped": existing.get("q95_capped", existing["q95"]),
                "q99_capped": existing.get("q99_capped", existing["q99"]),
                "incongruent_error_rate": existing["incongruent_error_rate"],
                "error_minus_correct_rt": existing["error_minus_correct_rt"],
                "incongruent_error_minus_correct_rt": existing["incongruent_error_minus_correct_rt"],
                "fastest_incongruent_bin_accuracy": existing["fastest_incongruent_bin_accuracy"],
                "early_flanker_ge_target_rate_incongruent_error": traj_lookup.loc[(cond, "incongruent_error"), "early_flanker_ge_target_rate"] if (cond, "incongruent_error") in traj_lookup.index else np.nan,
                "late_target_ge_flanker_rate_incongruent_correct": traj_lookup.loc[(cond, "incongruent_correct"), "late_target_ge_flanker_rate"] if (cond, "incongruent_correct") in traj_lookup.index else np.nan,
                "condition": cond,
            }
        )
    for schedule_type, label in [
        ("natural_hard_5stage", "natural_hard_5stage"),
        ("natural_smooth_5stage", "natural_smooth_5stage"),
        ("natural_refined_3stage", "natural_refined_3stage"),
    ]:
        det = top_candidates[
            (top_candidates["schedule_type"].eq(schedule_type)) & (top_candidates["variant_type"].eq("deterministic"))
        ]
        var = top_candidates[
            (top_candidates["schedule_type"].eq(schedule_type)) & (top_candidates["variant_type"].eq("variational"))
        ]
        for part, suffix, family in [(det, "_det", "natural_det"), (var, "_var", "natural_var")]:
            if part.empty:
                continue
            row = part.iloc[0]
            cond = str(row["condition"])
            rows.append(
                {
                    "input_condition": f"{label}{suffix}",
                    "model_family": family,
                    "accuracy": row["accuracy"],
                    "human_choice_agreement": row["model_human_choice_agreement"],
                    "mean_rt": row["mean_rt"],
                    "q95": row["q95"],
                    "q99": row["q99"],
                    "q95_capped": row["q95_capped"],
                    "q99_capped": row["q99_capped"],
                    "incongruent_error_rate": row["incongruent_error_rate"],
                    "error_minus_correct_rt": row["error_minus_correct_rt"],
                    "incongruent_error_minus_correct_rt": row["incongruent_error_minus_correct_rt"],
                    "fastest_incongruent_bin_accuracy": row["fastest_incongruent_bin_accuracy"],
                    "early_flanker_ge_target_rate_incongruent_error": traj_lookup.loc[(cond, "incongruent_error"), "early_flanker_ge_target_rate"] if (cond, "incongruent_error") in traj_lookup.index else np.nan,
                    "late_target_ge_flanker_rate_incongruent_correct": traj_lookup.loc[(cond, "incongruent_correct"), "late_target_ge_flanker_rate"] if (cond, "incongruent_correct") in traj_lookup.index else np.nan,
                    "condition": cond,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def write_summary_md(
    path: Path,
    deterministic_best: pd.DataFrame,
    variational_best: pd.DataFrame,
    comparison: pd.DataFrame,
    top_candidates: pd.DataFrame,
    max_trials: int,
    max_rt: float,
) -> None:
    det_best = deterministic_best.iloc[0] if not deterministic_best.empty else pd.Series(dtype=object)
    var_best = variational_best.iloc[0] if not variational_best.empty else pd.Series(dtype=object)
    final_row = comparison[comparison["input_condition"].eq("final_only")].iloc[0]
    middle_row = comparison[comparison["input_condition"].eq("middle_only")].iloc[0]
    refined_row = comparison[comparison["input_condition"].eq("refined_layer_time_gate")].iloc[0]
    dmc_row = comparison[comparison["input_condition"].eq("handcrafted_dmc_positive_control")].iloc[0]

    def fmt(row: pd.Series, key: str) -> str:
        if row.empty or key not in row or pd.isna(row[key]):
            return "not available in current cache"
        if isinstance(row[key], str):
            return str(row[key])
        return f"{float(row[key]):.3f}"

    text = f"""# Natural Layer-to-Time Variational Wong-Wang Diagnostic

## 1. Goal

This diagnostic tests whether CNN layer-wise logits/evidence can be mapped directly onto decision time in Wong-Wang, without hand-written DMC `flanker_mult(t)` or `target_mult(t)`.

## 2. Motivation

Existing layer-wise evidence audit already showed:

- `conv3` is strongly flanker-dominant.
- `final` is strongly target-dominant.
- `final_only` produces almost no conflict.
- `middle_only` produces strong conflict but unstable behavior.
- the refined layer-time gate is currently the best compromise.

That makes natural layer-to-time mapping the next controlled test.

## 3. Methods

- Layer-to-time schedules:
  - `natural_hard_5stage`
  - `natural_smooth_5stage`
  - `natural_refined_3stage`
- Normalization:
  - `no_norm`
  - `per_layer_zscore`
  - `per_layer_gap_scale`
- Deterministic version:
  - `mu_t` from layer schedule only.
- Variational version:
  - `sampled_E_t = mu_t + sigma_t * epsilon_t`
  - `sigma_type` in `fixed_sigma`, `layer_weighted_sigma`, `conflict_dependent_sigma`
- WW sweep:
  - evidence_gain = `0.5, 1.0, 1.5, 2.0`
  - threshold = `0.12, 0.14, 0.16`
- Variational seeds:
  - 5 seeds per kept condition
- Baselines:
  - `final_only`
  - `middle_only`
  - `pooled_only`
  - `refined_layer_time_gate`
  - `handcrafted_dmc_positive_control`

## 4. Deterministic Results

- Best deterministic condition: `{fmt(det_best, "condition")}`
- Best deterministic schedule: `{fmt(det_best, "schedule_type")}`
- Best deterministic normalization: `{fmt(det_best, "normalization")}`
- Accuracy / human-choice agreement / mean RT / incongruent error rate:
  - `{fmt(det_best, "accuracy")} / {fmt(det_best, "model_human_choice_agreement")} / {fmt(det_best, "mean_rt")} / {fmt(det_best, "incongruent_error_rate")}`

Interpretation:

- Deterministic natural layer-to-time mapping does create conflict if incongruent error rate rises above `final_only = {fmt(final_row, "incongruent_error_rate")}`.
- It is behaviorally useful only if it stays above `middle_only accuracy = {fmt(middle_row, "accuracy")}`.
- The main question is whether it preserves early flanker pull and late target recovery without collapsing into either final-only or middle-only extremes.

## 5. Variational Results

- Best variational condition: `{fmt(var_best, "condition")}`
- Best variational schedule / normalization / sigma design:
  - `{fmt(var_best, "schedule_type")} / {fmt(var_best, "normalization")} / {fmt(var_best, "sigma_type")}`
- Accuracy / human-choice agreement / mean RT / incongruent error rate:
  - `{fmt(var_best, "accuracy")} / {fmt(var_best, "model_human_choice_agreement")} / {fmt(var_best, "mean_rt")} / {fmt(var_best, "incongruent_error_rate")}`

Interpretation:

- Variational sampling adds trial-to-trial evidence variability on top of the same `mu_t`.
- The key question is whether it improves RT tail, fast-error direction, or agreement without collapsing accuracy.
- `fixed_sigma`, `layer_weighted_sigma`, and `conflict_dependent_sigma` should be judged by that trade-off, not by noise size alone.

## 6. Comparison with Baselines

- `final_only`: accuracy / mean RT / incongruent error rate = `{fmt(final_row, "accuracy")} / {fmt(final_row, "mean_rt")} / {fmt(final_row, "incongruent_error_rate")}`
- `middle_only`: accuracy / mean RT / incongruent error rate = `{fmt(middle_row, "accuracy")} / {fmt(middle_row, "mean_rt")} / {fmt(middle_row, "incongruent_error_rate")}`
- `refined_layer_time_gate`: accuracy / mean RT / incongruent error rate = `{fmt(refined_row, "accuracy")} / {fmt(refined_row, "mean_rt")} / {fmt(refined_row, "incongruent_error_rate")}`
- `handcrafted_dmc_positive_control`: accuracy / mean RT / incongruent error rate = `{fmt(dmc_row, "accuracy")} / {fmt(dmc_row, "mean_rt")} / {fmt(dmc_row, "incongruent_error_rate")}`
- Best natural deterministic: `{fmt(det_best, "accuracy")} / {fmt(det_best, "mean_rt")} / {fmt(det_best, "incongruent_error_rate")}`
- Best natural variational: `{fmt(var_best, "accuracy")} / {fmt(var_best, "mean_rt")} / {fmt(var_best, "incongruent_error_rate")}`

## 7. Interpretation

Current diagnostic should be read in the following way:

- `mu_t` is the time-varying subjective evidence mean produced by layer-to-time mapping.
- `sigma_t` is the time-varying subjective evidence uncertainty.
- Variational sampling changes trial-to-trial evidence variability, not the basic visual hierarchy itself.
- If the best natural conditions produce conflict beyond `final_only`, stay more stable than `middle_only`, and preserve early flanker / late target structure, that supports CNN visual hierarchy as a natural DMC-replacement candidate.
- This is still a candidate route, not a completed replacement.

## 8. Noise / Prior Caution

There is no explicit cue or prior manipulation here, so this should not be described as a prior effect.

Safer wording:

- flanker context / congruency / conflict strength modulates subjective evidence uncertainty
- variational sampling should be interpreted as subjective evidence uncertainty, not as a generic random perturbation

## 9. Limitations

- This is a {max_trials}-row diagnostic cache study.
- Subject-level validation is still not completed.
- Image-identity validation is still not completed.
- Different layer raw scales required explicit normalization.
- If `q95` or `q99` remains near the simulation limit `{max_rt:.3f}`, that cap must be treated as a real limitation.
- Raw trajectory arrays are not exported separately here; the saved output is trajectory summary.
- These results cannot be used to claim hand-crafted DMC is already fully replaced.

## 10. Recommended Next Steps

1. If natural deterministic mapping is already effective, move to subject-level validation.
2. If variational sampling improves RT or fast-error structure without accuracy collapse, keep the variational layer-to-time route.
3. If variational sampling hurts accuracy, reduce `sigma` or prefer conflict-dependent `sigma_t`.
4. If natural mapping degenerates into either final-only or middle-only behavior, revisit schedule shape and normalization first.
5. Do not activate stochastic stopping yet unless early flanker competition is already stable but fast errors are still missing.

## 11. Short Chinese Summary for Discussion

这次分析测试的是：能否把 CNN 不同 layer 的 evidence 按时间自然输入 Wong-Wang，而不是手写 DMC pulse。deterministic 版本检验的是 layer-wise evidence 的均值结构本身，是否已经足够产生 early flanker / late target；variational 版本检验的是，在这个均值结构上加入 subjective evidence uncertainty 后，是否能进一步改善 RT 分布、fast-error 模式和选择一致性。这里不把它叫作 prior effect，因为当前没有显式 cue 或 prior 操作；更稳妥的说法是 conflict-dependent subjective evidence uncertainty。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz")
    parser.add_argument("--output_dir", default="artifacts/results/diagnostics/natural_layer_to_time_var_ww")
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--time_steps", type=int, default=160)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--t0_seconds", type=float, default=0.25)
    parser.add_argument("--noise_ampa", type=float, default=0.02)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--readout_mode", default="baseline")
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(resolve_path(args.cache_path), args.max_trials)
    raw_layers = raw_layer_arrays(cache)
    same_existing, refined_existing = load_existing_baseline_tables()
    baseline_reference = canonical_baseline_summary(same_existing, refined_existing)

    time_steps = int(args.time_steps)
    dt_ms = int(args.dt_ms)
    max_rt = (time_steps - 1) * (dt_ms / 1000.0) + float(args.t0_seconds)

    schedules = {
        "natural_hard_5stage": schedule_weights("natural_hard_5stage", time_steps),
        "natural_smooth_5stage": schedule_weights("natural_smooth_5stage", time_steps),
        "natural_refined_3stage": schedule_weights("natural_refined_3stage", time_steps),
    }
    plot_schedule_weights(schedules, figure_dir)

    baseline_inputs = build_baseline_inputs(cache, time_steps, dt_ms)
    baseline_rows: List[Dict[str, Any]] = []
    baseline_trials: List[pd.DataFrame] = []
    all_curves: Dict[str, np.ndarray] = {}
    trajectory_rows_all: List[Dict[str, Any]] = []

    for input_condition, ww_input in baseline_inputs.items():
        outputs = run_ww(
            ww_input,
            time_steps=time_steps,
            dt_ms=dt_ms,
            threshold=0.22,
            noise_ampa=float(args.noise_ampa),
            device="cpu",
            seed=int(args.seed),
            readout_mode=str(args.readout_mode),
            t0_seconds=float(args.t0_seconds),
            choice_temperature=float(args.choice_temperature),
        )
        condition = BASELINE_LABELS[input_condition]
        df = make_trial_df(cache, condition, outputs)
        row = summarize_condition_extended(
            condition=condition,
            family="baseline",
            variant_type="baseline",
            schedule_type=input_condition,
            normalization="existing",
            sigma_type="none",
            sigma_base=0.0,
            sigma_middle=0.0,
            sigma_conflict=0.0,
            seed="baseline",
            n_seed_repeats=1,
            df=df,
            outputs=outputs,
            time_steps=time_steps,
            dt_ms=dt_ms,
            t0_seconds=float(args.t0_seconds),
        )
        row["input_condition"] = input_condition
        baseline_rows.append(row)
        baseline_trials.append(df.assign(input_condition=input_condition))
        traj_rows, curves = trajectory_summary_rows_extended(
            condition=condition,
            df=df,
            outputs=outputs,
            cache=cache,
            dt_ms=dt_ms,
            extra={
                "variant_type": "baseline",
                "schedule_type": input_condition,
                "normalization": "existing",
                "sigma_type": "none",
                "seed": "baseline",
            },
        )
        trajectory_rows_all.extend(traj_rows)
        all_curves.update(curves)

    det_rows: List[Dict[str, Any]] = []
    det_trials: List[pd.DataFrame] = []
    det_condition_meta: List[Dict[str, Any]] = []

    condition_counter = 0
    for schedule_type, schedule_df in schedules.items():
        for normalization in ["no_norm", "per_layer_zscore", "per_layer_gap_scale"]:
            normalized_layers = normalize_layers(raw_layers, normalization)
            for evidence_gain in [0.5, 1.0, 1.5, 2.0]:
                mu = build_mu_schedule(normalized_layers, schedule_df, evidence_gain)
                for threshold in [0.12, 0.14, 0.16]:
                    condition_counter += 1
                    condition = f"natural_det_{schedule_type}_norm-{normalization}_g{evidence_gain:.2f}_th{threshold:.2f}"
                    outputs = run_ww(
                        mu,
                        time_steps=time_steps,
                        dt_ms=dt_ms,
                        threshold=float(threshold),
                        noise_ampa=float(args.noise_ampa),
                        device="cpu",
                        seed=int(args.seed),
                        readout_mode=str(args.readout_mode),
                        t0_seconds=float(args.t0_seconds),
                        choice_temperature=float(args.choice_temperature),
                    )
                    df = make_trial_df(cache, condition, outputs)
                    df = df.assign(
                        variant_type="deterministic",
                        schedule_type=schedule_type,
                        normalization=normalization,
                        sigma_type="none",
                        sigma_base=0.0,
                        sigma_middle=0.0,
                        sigma_conflict=0.0,
                        evidence_gain=evidence_gain,
                        threshold=threshold,
                        seed=int(args.seed),
                    )
                    row = summarize_condition_extended(
                        condition=condition,
                        family="natural_det",
                        variant_type="deterministic",
                        schedule_type=schedule_type,
                        normalization=normalization,
                        sigma_type="none",
                        sigma_base=0.0,
                        sigma_middle=0.0,
                        sigma_conflict=0.0,
                        seed=int(args.seed),
                        n_seed_repeats=1,
                        df=df,
                        outputs=outputs,
                        time_steps=time_steps,
                        dt_ms=dt_ms,
                        t0_seconds=float(args.t0_seconds),
                    )
                    row["evidence_gain"] = float(evidence_gain)
                    row["threshold"] = float(threshold)
                    det_rows.append(row)
                    det_trials.append(df)
                    det_condition_meta.append(row)
                    traj_rows, curves = trajectory_summary_rows_extended(
                        condition=condition,
                        df=df,
                        outputs=outputs,
                        cache=cache,
                        dt_ms=dt_ms,
                        extra={
                            "variant_type": "deterministic",
                            "schedule_type": schedule_type,
                            "normalization": normalization,
                            "sigma_type": "none",
                            "seed": int(args.seed),
                        },
                    )
                    trajectory_rows_all.extend(traj_rows)
                    all_curves.update(curves)

    det_summary = pd.DataFrame(det_rows)
    det_summary["selection_score"] = (
        1.2 * det_summary["accuracy"]
        + 0.7 * det_summary["model_human_choice_agreement"]
        + 0.4 * det_summary["incongruent_error_rate"]
        - 0.25 * det_summary["mean_rt"]
        - 0.10 * det_summary["q95_capped"]
    )
    top_det_by_schedule = (
        det_summary.sort_values("selection_score", ascending=False)
        .groupby("schedule_type", as_index=False)
        .head(1)
        .copy()
    )

    var_rows: List[Dict[str, Any]] = []
    var_trials: List[pd.DataFrame] = []
    n_seeds = 5
    seeds = [int(args.seed) + i for i in range(n_seeds)]
    sigma_specs = [
        ("fixed_sigma", 0.02, 0.00, 0.00),
        ("fixed_sigma", 0.05, 0.00, 0.00),
        ("fixed_sigma", 0.10, 0.00, 0.00),
        ("layer_weighted_sigma", 0.02, 0.05, 0.00),
        ("layer_weighted_sigma", 0.02, 0.10, 0.00),
        ("layer_weighted_sigma", 0.05, 0.05, 0.00),
        ("layer_weighted_sigma", 0.05, 0.10, 0.00),
        ("conflict_dependent_sigma", 0.02, 0.00, 0.05),
        ("conflict_dependent_sigma", 0.02, 0.00, 0.10),
        ("conflict_dependent_sigma", 0.05, 0.00, 0.05),
        ("conflict_dependent_sigma", 0.05, 0.00, 0.10),
    ]

    for _, det_best in top_det_by_schedule.iterrows():
        schedule_type = str(det_best["schedule_type"])
        normalization = str(det_best["normalization"])
        evidence_gain = float(det_best["evidence_gain"])
        threshold = float(det_best["threshold"])
        normalized_layers = normalize_layers(raw_layers, normalization)
        mu = build_mu_schedule(normalized_layers, schedules[schedule_type], evidence_gain)
        mu_np = mu.detach().cpu().numpy().astype(np.float32)
        for sigma_type, sigma_base, sigma_middle, sigma_conflict in sigma_specs:
            sigma = build_sigma(
                mu_np,
                schedules[schedule_type],
                cache,
                sigma_type,
                sigma_base,
                sigma_middle,
                sigma_conflict,
            )
            for seed in seeds:
                sampled = sample_mu_sigma(mu, sigma, seed)
                condition = (
                    f"natural_var_{schedule_type}_norm-{normalization}_g{evidence_gain:.2f}_th{threshold:.2f}"
                    f"_{sigma_type}_sb{sigma_base:.2f}_sm{sigma_middle:.2f}_sc{sigma_conflict:.2f}_seed{seed}"
                )
                outputs = run_ww(
                    sampled,
                    time_steps=time_steps,
                    dt_ms=dt_ms,
                    threshold=float(threshold),
                    noise_ampa=float(args.noise_ampa),
                    device="cpu",
                    seed=int(seed),
                    readout_mode=str(args.readout_mode),
                    t0_seconds=float(args.t0_seconds),
                    choice_temperature=float(args.choice_temperature),
                )
                df = make_trial_df(cache, condition, outputs)
                df = df.assign(
                    variant_type="variational",
                    schedule_type=schedule_type,
                    normalization=normalization,
                    sigma_type=sigma_type,
                    sigma_base=sigma_base,
                    sigma_middle=sigma_middle,
                    sigma_conflict=sigma_conflict,
                    evidence_gain=evidence_gain,
                    threshold=threshold,
                    seed=seed,
                )
                row = summarize_condition_extended(
                    condition=condition,
                    family="natural_var",
                    variant_type="variational",
                    schedule_type=schedule_type,
                    normalization=normalization,
                    sigma_type=sigma_type,
                    sigma_base=sigma_base,
                    sigma_middle=sigma_middle,
                    sigma_conflict=sigma_conflict,
                    seed=seed,
                    n_seed_repeats=n_seeds,
                    df=df,
                    outputs=outputs,
                    time_steps=time_steps,
                    dt_ms=dt_ms,
                    t0_seconds=float(args.t0_seconds),
                )
                row["evidence_gain"] = float(evidence_gain)
                row["threshold"] = float(threshold)
                var_rows.append(row)
                var_trials.append(df)
                traj_rows, curves = trajectory_summary_rows_extended(
                    condition=condition,
                    df=df,
                    outputs=outputs,
                    cache=cache,
                    dt_ms=dt_ms,
                    extra={
                        "variant_type": "variational",
                        "schedule_type": schedule_type,
                        "normalization": normalization,
                        "sigma_type": sigma_type,
                        "sigma_base": sigma_base,
                        "sigma_middle": sigma_middle,
                        "sigma_conflict": sigma_conflict,
                        "seed": seed,
                    },
                )
                trajectory_rows_all.extend(traj_rows)
                all_curves.update(curves)

    var_summary = pd.DataFrame(var_rows)
    var_avg = aggregate_seed_rows(
        var_summary,
        [
            "variant_type",
            "schedule_type",
            "normalization",
            "sigma_type",
            "sigma_base",
            "sigma_middle",
            "sigma_conflict",
            "evidence_gain",
            "threshold",
        ],
    )
    if not var_avg.empty:
        var_avg["condition"] = (
            "natural_var_"
            + var_avg["schedule_type"].astype(str)
            + "_norm-"
            + var_avg["normalization"].astype(str)
            + "_g"
            + var_avg["evidence_gain"].map(lambda x: f"{float(x):.2f}")
            + "_th"
            + var_avg["threshold"].map(lambda x: f"{float(x):.2f}")
            + "_"
            + var_avg["sigma_type"].astype(str)
            + "_sb"
            + var_avg["sigma_base"].map(lambda x: f"{float(x):.2f}")
            + "_sm"
            + var_avg["sigma_middle"].map(lambda x: f"{float(x):.2f}")
            + "_sc"
            + var_avg["sigma_conflict"].map(lambda x: f"{float(x):.2f}")
            + "_seedavg"
        )
        var_avg["family"] = "natural_var"

    summary = pd.concat([pd.DataFrame(baseline_rows), det_summary, var_summary, var_avg], ignore_index=True, sort=False)
    trial_level = pd.concat(baseline_trials + det_trials + var_trials, ignore_index=True, sort=False)
    trajectory_summary = pd.DataFrame(trajectory_rows_all)

    top_natural_source = pd.concat([det_summary, var_avg], ignore_index=True, sort=False)
    top_candidates = choose_top_candidates(top_natural_source, baseline_reference, max_rt)

    comparison = build_baseline_comparison(
        baseline_reference,
        top_candidates,
        trajectory_summary,
        output_dir / "natural_layer_to_time_var_ww_baseline_comparison.csv",
        same_existing,
        refined_existing,
    )

    deterministic_best = top_candidates[top_candidates["variant_type"].eq("deterministic")]
    variational_best = top_candidates[top_candidates["variant_type"].eq("variational")]

    top_det_condition = str(deterministic_best.iloc[0]["condition"]) if not deterministic_best.empty else None
    top_var_seedavg = str(variational_best.iloc[0]["condition"]) if not variational_best.empty else None
    top_var_schedule = None
    top_var_trial_condition = None
    if not variational_best.empty:
        top_var_schedule = variational_best.iloc[0]
        matching_trials = var_summary[
            (var_summary["schedule_type"].eq(top_var_schedule["schedule_type"]))
            & (var_summary["normalization"].eq(top_var_schedule["normalization"]))
            & (var_summary["sigma_type"].eq(top_var_schedule["sigma_type"]))
            & (var_summary["sigma_base"].eq(top_var_schedule["sigma_base"]))
            & (var_summary["sigma_middle"].eq(top_var_schedule["sigma_middle"]))
            & (var_summary["sigma_conflict"].eq(top_var_schedule["sigma_conflict"]))
            & (var_summary["evidence_gain"].eq(top_var_schedule["evidence_gain"]))
            & (var_summary["threshold"].eq(top_var_schedule["threshold"]))
        ]
        if not matching_trials.empty:
            top_var_trial_condition = str(matching_trials.sort_values("model_human_choice_agreement", ascending=False).iloc[0]["condition"])

    plot_behavior_scatter(comparison, figure_dir)
    rt_plot_conditions = [c for c in [top_det_condition, top_var_trial_condition, BASELINE_LABELS["refined_layer_time_gate"]] if c]
    plot_rt_distribution(trial_level, rt_plot_conditions, figure_dir / "rt_distribution_top_candidates.png")
    plot_rt_quantiles(summary, rt_plot_conditions, figure_dir / "rt_quantiles_top_candidates.png")

    overall_top = top_candidates.iloc[0] if not top_candidates.empty else pd.Series(dtype=object)
    traj_condition = str(overall_top["condition"]) if not overall_top.empty else None
    if traj_condition:
        plot_trajectory(all_curves, traj_condition, "incongruent_correct", dt_ms, figure_dir / "trajectory_incongruent_correct_top_candidate.png")
        plot_trajectory(all_curves, traj_condition, "incongruent_error", dt_ms, figure_dir / "trajectory_incongruent_error_top_candidate.png")
        plot_target_minus_flanker(all_curves, traj_condition, ["incongruent_correct", "incongruent_error"], dt_ms, figure_dir / "trajectory_target_minus_flanker_top_candidate.png")

    summary.to_csv(output_dir / "natural_layer_to_time_var_ww_summary.csv", index=False)
    trial_level.to_csv(output_dir / "natural_layer_to_time_var_ww_trial_level.csv", index=False)
    trajectory_summary.to_csv(output_dir / "natural_layer_to_time_var_ww_trajectory_summary.csv", index=False)
    top_candidates.to_csv(output_dir / "natural_layer_to_time_var_ww_top_candidates.csv", index=False)

    write_summary_md(
        output_dir / "natural_layer_to_time_var_ww_summary.md",
        deterministic_best,
        variational_best,
        comparison,
        top_candidates,
        int(args.max_trials),
        float(max_rt),
    )

    metadata = {
        "cache_path": str(resolve_path(args.cache_path)),
        "max_trials": int(args.max_trials),
        "time_steps": time_steps,
        "dt_ms": dt_ms,
        "t0_seconds": float(args.t0_seconds),
        "noise_ampa": float(args.noise_ampa),
        "choice_temperature": float(args.choice_temperature),
        "readout_mode": str(args.readout_mode),
        "seed": int(args.seed),
        "n_variational_seeds": n_seeds,
        "outputs": {
            "summary": str(output_dir / "natural_layer_to_time_var_ww_summary.csv"),
            "trial_level": str(output_dir / "natural_layer_to_time_var_ww_trial_level.csv"),
            "trajectory_summary": str(output_dir / "natural_layer_to_time_var_ww_trajectory_summary.csv"),
            "baseline_comparison": str(output_dir / "natural_layer_to_time_var_ww_baseline_comparison.csv"),
            "top_candidates": str(output_dir / "natural_layer_to_time_var_ww_top_candidates.csv"),
            "summary_md": str(output_dir / "natural_layer_to_time_var_ww_summary.md"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")

    print("Generated:")
    for name in [
        "natural_layer_to_time_var_ww_summary.csv",
        "natural_layer_to_time_var_ww_trial_level.csv",
        "natural_layer_to_time_var_ww_trajectory_summary.csv",
        "natural_layer_to_time_var_ww_baseline_comparison.csv",
        "natural_layer_to_time_var_ww_top_candidates.csv",
        "natural_layer_to_time_var_ww_summary.md",
    ]:
        print((output_dir / name).relative_to(PROJECT_ROOT), (output_dir / name).exists())
    print(figure_dir.relative_to(PROJECT_ROOT), figure_dir.exists())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import make_trial_df, run_ww  # noqa: E402
from compare_same_subset_layerwise_vs_dmc import evidence_sources  # noqa: E402
from complete_layerwise_dmc_remaining_diagnostics import build_hand_dmc_input  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import (  # noqa: E402
    build_mu_schedule,
    build_sigma,
    normalize_layers,
    raw_layer_arrays,
    sample_mu_sigma,
    schedule_weights,
)
from train_age_groups_efficient import to_jsonable  # noqa: E402


BEST_DET = "natural_det_natural_smooth_5stage_norm-per_layer_gap_scale_g2.00_th0.12"
BEST_VAR_SEEDAVG = (
    "natural_var_natural_smooth_5stage_norm-per_layer_gap_scale_g2.00_th0.12_"
    "fixed_sigma_sb0.05_sm0.00_sc0.00_seedavg"
)
MODEL_ALIASES = {
    BEST_DET: "best_natural_deterministic",
    BEST_VAR_SEEDAVG: "best_natural_variational_seedavg",
    "final_logits_ww": "final_only",
    "mid_layer_ww": "middle_only",
    "refined_best_layerwise_gate": "refined_layer_time_gate",
    "handcrafted_dmc_final_ww": "handcrafted_dmc_positive_control",
    "handcrafted_dmc_positive_control": "handcrafted_dmc_positive_control",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cache(path: Path, max_trials: int) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    n = min(int(max_trials), len(data["target_labels"]))
    return {key: data[key][:n] for key in data.files}


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def q(values: np.ndarray, prob: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, prob)) if values.size else float("nan")


def add_common_metadata(
    df: pd.DataFrame,
    *,
    condition_name: str,
    variant_type: str,
    schedule_type: str,
    normalization: str,
    sigma_type: str,
    seed: int | str,
    evidence_gain: float,
    threshold: float,
    sigma_base: float = 0.0,
    sigma_middle: float = 0.0,
    sigma_conflict: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    out["condition_name"] = condition_name
    out["condition"] = condition_name
    out["variant_type"] = variant_type
    out["schedule_type"] = schedule_type
    out["normalization"] = normalization
    out["sigma_type"] = sigma_type
    out["seed"] = seed
    out["evidence_gain"] = evidence_gain
    out["threshold"] = threshold
    out["sigma_base"] = sigma_base
    out["sigma_middle"] = sigma_middle
    out["sigma_conflict"] = sigma_conflict
    return out


def run_natural_condition(
    cache: Dict[str, np.ndarray],
    *,
    condition_name: str,
    variant_type: str,
    schedule_type: str,
    normalization: str,
    evidence_gain: float,
    threshold: float,
    sigma_type: str = "none",
    sigma_base: float = 0.0,
    sigma_middle: float = 0.0,
    sigma_conflict: float = 0.0,
    seed: int = 20260527,
    time_steps: int = 160,
    dt_ms: int = 10,
    t0_seconds: float = 0.25,
    noise_ampa: float = 0.02,
    readout_mode: str = "baseline",
    choice_temperature: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    raw_layers = raw_layer_arrays(cache)
    normalized = normalize_layers(raw_layers, normalization)
    schedule_df = schedule_weights(schedule_type, time_steps)
    mu = build_mu_schedule(normalized, schedule_df, evidence_gain)
    ww_input = mu
    if variant_type == "variational":
        sigma = build_sigma(
            mu.detach().cpu().numpy(),
            schedule_df,
            cache,
            sigma_type,
            sigma_base,
            sigma_middle,
            sigma_conflict,
        )
        ww_input = sample_mu_sigma(mu, sigma, seed)
    outputs = run_ww(
        ww_input,
        time_steps=time_steps,
        dt_ms=dt_ms,
        threshold=threshold,
        noise_ampa=noise_ampa,
        device="cpu",
        seed=seed,
        readout_mode=readout_mode,
        t0_seconds=t0_seconds,
        choice_temperature=choice_temperature,
    )
    df = make_trial_df(cache, condition_name, outputs)
    df = add_common_metadata(
        df,
        condition_name=condition_name,
        variant_type=variant_type,
        schedule_type=schedule_type,
        normalization=normalization,
        sigma_type=sigma_type,
        seed=seed,
        evidence_gain=evidence_gain,
        threshold=threshold,
        sigma_base=sigma_base,
        sigma_middle=sigma_middle,
        sigma_conflict=sigma_conflict,
    )
    return df, outputs


def run_baseline_condition(
    cache: Dict[str, np.ndarray],
    input_condition: str,
    *,
    time_steps: int,
    dt_ms: int,
    t0_seconds: float,
    threshold: float,
    noise_ampa: float,
    readout_mode: str,
    choice_temperature: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    ev = evidence_sources(cache, gain=0.8)
    if input_condition == "final_only":
        ww_input = ev["final"].unsqueeze(1).repeat(1, time_steps, 1)
        condition_name = "final_logits_ww"
    elif input_condition == "middle_only":
        ww_input = ev["mid"].unsqueeze(1).repeat(1, time_steps, 1)
        condition_name = "mid_layer_ww"
    elif input_condition == "handcrafted_dmc_positive_control":
        ww_input, _ = build_hand_dmc_input(
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
        condition_name = "handcrafted_dmc_final_ww"
    else:
        raise ValueError(f"Unsupported baseline input_condition: {input_condition}")
    outputs = run_ww(
        ww_input,
        time_steps=time_steps,
        dt_ms=dt_ms,
        threshold=threshold,
        noise_ampa=noise_ampa,
        device="cpu",
        seed=seed,
        readout_mode=readout_mode,
        t0_seconds=t0_seconds,
        choice_temperature=choice_temperature,
    )
    df = make_trial_df(cache, condition_name, outputs)
    df = add_common_metadata(
        df,
        condition_name=condition_name,
        variant_type="baseline",
        schedule_type=input_condition,
        normalization="existing",
        sigma_type="none",
        seed=seed,
        evidence_gain=float("nan"),
        threshold=threshold,
    )
    return df, outputs


def readout_step_from_rt(pred_rt: np.ndarray, *, t0_seconds: float, dt_ms: int, time_steps: int) -> np.ndarray:
    decision_time = np.asarray(pred_rt, dtype=np.float64) - float(t0_seconds)
    step = np.rint(decision_time / (dt_ms / 1000.0)).astype(np.int64)
    return np.clip(step, 0, time_steps - 1)


def extract_readout_timing(
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    *,
    t0_seconds: float,
    dt_ms: int,
) -> pd.DataFrame:
    trajectory = np.asarray(outputs["trajectory"], dtype=np.float32)
    n, time_steps, n_classes = trajectory.shape
    targets = df["target_label"].to_numpy(dtype=np.int64)
    flankers = df["flanker_label"].to_numpy(dtype=np.int64)
    rows = np.arange(n)
    s_target = trajectory[rows[:, None], np.arange(time_steps)[None, :], targets[:, None]]
    s_flanker = trajectory[rows[:, None], np.arange(time_steps)[None, :], flankers[:, None]]
    other = trajectory.copy()
    other[rows, :, targets] = -np.inf
    other[rows, :, flankers] = -np.inf
    s_other_max = np.max(other, axis=2)
    finite_other = np.isfinite(s_other_max)
    s_other_max[~finite_other] = np.nan
    diff = s_target - s_flanker
    readout_step = readout_step_from_rt(df["pred_rt"].to_numpy(), t0_seconds=t0_seconds, dt_ms=dt_ms, time_steps=time_steps)
    early_steps = max(1, int(round(0.20 * time_steps)))
    late_steps = early_steps
    recovery_mask = diff > 0
    any_recovery = recovery_mask.any(axis=1)
    recovery_step = np.argmax(recovery_mask, axis=1).astype(float)
    recovery_step[~any_recovery] = np.nan
    recovery_time = recovery_step * (dt_ms / 1000.0)
    readout_time = readout_step.astype(float) * (dt_ms / 1000.0)
    recovered_before = any_recovery & (recovery_time <= readout_time)
    readout_before = (~any_recovery) | (readout_time < recovery_time)
    pre_min, pre_max_flanker = [], []
    for i, step in enumerate(readout_step):
        segment = diff[i, : step + 1]
        pre_min.append(float(np.min(segment)))
        pre_max_flanker.append(float(np.max(-segment)))

    out = pd.DataFrame(
        {
            "trial_id": df["row_index"].to_numpy(),
            "condition_name": df["condition_name"].to_numpy(),
            "variant_type": df["variant_type"].to_numpy(),
            "schedule_type": df["schedule_type"].to_numpy(),
            "normalization": df["normalization"].to_numpy(),
            "sigma_type": df["sigma_type"].to_numpy(),
            "seed": df["seed"].to_numpy(),
            "congruency": df["congruency"].to_numpy(dtype=np.int64),
            "target_label": targets,
            "flanker_label": flankers,
            "human_response": df["response_label"].to_numpy(dtype=np.int64),
            "human_correct": df["human_correct"].to_numpy(dtype=bool),
            "human_rt": df["true_rt"].to_numpy(dtype=np.float32),
            "model_response": df["pred_choice"].to_numpy(dtype=np.int64),
            "model_correct": df["model_correct"].to_numpy(dtype=bool),
            "model_rt": df["pred_rt"].to_numpy(dtype=np.float32),
            "decision_time": df["pred_rt"].to_numpy(dtype=np.float32) - float(t0_seconds),
            "readout_step": readout_step,
            "readout_time_from_decision_onset": readout_time,
            "s_target_at_readout": s_target[rows, readout_step],
            "s_flanker_at_readout": s_flanker[rows, readout_step],
            "s_other_max_at_readout": s_other_max[rows, readout_step],
            "s_target_minus_flanker_at_readout": diff[rows, readout_step],
            "s_flanker_minus_target_at_readout": -diff[rows, readout_step],
            "early_s_target_minus_flanker_mean": diff[:, :early_steps].mean(axis=1),
            "early_s_flanker_ge_target": (s_flanker[:, :early_steps].mean(axis=1) >= s_target[:, :early_steps].mean(axis=1)),
            "late_s_target_minus_flanker_mean": diff[:, -late_steps:].mean(axis=1),
            "late_s_target_ge_flanker": (s_target[:, -late_steps:].mean(axis=1) >= s_flanker[:, -late_steps:].mean(axis=1)),
            "target_recovery_step": recovery_step,
            "target_recovery_time": recovery_time,
            "target_recovered_before_readout": recovered_before,
            "readout_before_target_recovery": readout_before,
            "min_target_minus_flanker_pre_readout": pre_min,
            "max_flanker_minus_target_pre_readout": pre_max_flanker,
            "final_target_minus_flanker": diff[:, -1],
        }
    )
    return out


def summarize_readout_group(part: pd.DataFrame, group_name: str) -> Dict[str, Any]:
    diff = part["s_target_minus_flanker_at_readout"].to_numpy(dtype=np.float64)
    return {
        "trial_group": group_name,
        "n_trials": int(len(part)),
        "model_rt_mean": safe_mean(part["model_rt"]),
        "model_rt_median": q(part["model_rt"].to_numpy(), 0.5),
        "decision_time_mean": safe_mean(part["decision_time"]),
        "decision_time_median": q(part["decision_time"].to_numpy(), 0.5),
        "human_rt_mean": safe_mean(part["human_rt"]),
        "human_rt_median": q(part["human_rt"].to_numpy(), 0.5),
        "s_target_minus_flanker_at_readout_mean": safe_mean(diff),
        "s_target_minus_flanker_at_readout_sem": sem(diff),
        "flanker_dominant_at_readout_rate": float((diff < 0).mean()) if len(part) else float("nan"),
        "target_dominant_at_readout_rate": float((diff > 0).mean()) if len(part) else float("nan"),
        "target_recovery_time_mean": safe_mean(part["target_recovery_time"]),
        "target_recovery_time_median": q(part["target_recovery_time"].to_numpy(), 0.5),
        "target_recovered_before_readout_rate": safe_mean(part["target_recovered_before_readout"].astype(float)),
        "readout_before_target_recovery_rate": safe_mean(part["readout_before_target_recovery"].astype(float)),
        "early_s_target_minus_flanker_mean": safe_mean(part["early_s_target_minus_flanker_mean"]),
        "late_s_target_minus_flanker_mean": safe_mean(part["late_s_target_minus_flanker_mean"]),
        "final_target_minus_flanker_mean": safe_mean(part["final_target_minus_flanker"]),
        "model_accuracy": safe_mean(part["model_correct"].astype(float)),
        "human_accuracy": safe_mean(part["human_correct"].astype(float)),
    }


def readout_timing_summary(readout_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for condition_name, cond in readout_df.groupby("condition_name"):
        base = {
            "condition_name": condition_name,
            "variant_type": cond["variant_type"].iloc[0],
        }
        groups: Dict[str, pd.Series] = {
            "congruent_correct": cond["congruency"].eq(0) & cond["model_correct"],
            "congruent_error": cond["congruency"].eq(0) & ~cond["model_correct"],
            "incongruent_correct": cond["congruency"].eq(1) & cond["model_correct"],
            "incongruent_error": cond["congruency"].eq(1) & ~cond["model_correct"],
            "model_fast": cond["model_rt"] <= cond["model_rt"].median(),
            "model_slow": cond["model_rt"] > cond["model_rt"].median(),
            "human_fast": cond["human_rt"] <= cond["human_rt"].median(),
            "human_slow": cond["human_rt"] > cond["human_rt"].median(),
        }
        for name, mask in groups.items():
            part = cond[mask].copy()
            row = {
                **base,
                "congruency": "mixed",
                "model_correct": "mixed",
                "human_correct": "mixed",
            }
            if name.startswith("congruent"):
                row["congruency"] = 0
                row["model_correct"] = bool(name.endswith("correct"))
            elif name.startswith("incongruent"):
                row["congruency"] = 1
                row["model_correct"] = bool(name.endswith("correct"))
            row.update(summarize_readout_group(part, name))
            rows.append(row)
    return pd.DataFrame(rows)


def metric_summary(condition_name: str, df: pd.DataFrame, *, variant_type: str) -> Dict[str, Any]:
    pred_rt = df["pred_rt"].to_numpy(dtype=np.float64)
    true_rt = df["true_rt"].to_numpy(dtype=np.float64)
    correct = df["model_correct"].to_numpy(dtype=bool)
    human_correct = df["human_correct"].to_numpy(dtype=bool)
    incong = df["congruency"].to_numpy(dtype=np.int64) == 1
    err = ~correct
    order = np.argsort(pred_rt)
    bins = np.array_split(order, 10) if len(order) >= 10 else []
    fastest = df.iloc[bins[0]] if bins else df.iloc[[]]
    slowest = df.iloc[bins[-1]] if bins else df.iloc[[]]
    incong_df = df[incong]
    inc_order = np.argsort(incong_df["pred_rt"].to_numpy(dtype=np.float64))
    inc_bins = np.array_split(inc_order, 10) if len(inc_order) >= 10 else []
    inc_fastest = incong_df.iloc[inc_bins[0]] if inc_bins else incong_df.iloc[[]]
    fastest_acc = safe_mean(fastest["model_correct"].astype(float)) if len(fastest) else float("nan")
    slowest_acc = safe_mean(slowest["model_correct"].astype(float)) if len(slowest) else float("nan")
    return {
        "condition_name": condition_name,
        "model_label": MODEL_ALIASES.get(condition_name, condition_name),
        "variant_type": variant_type,
        "n_trials": int(len(df)),
        "accuracy": safe_mean(correct.astype(float)),
        "human_choice_agreement": safe_mean((df["pred_choice"].to_numpy() == df["response_label"].to_numpy()).astype(float)),
        "incongruent_error_rate": safe_mean((~df.loc[incong, "model_correct"]).astype(float)) if incong.any() else float("nan"),
        "mean_rt": safe_mean(pred_rt),
        "median_rt": q(pred_rt, 0.5),
        "rt_sd": float(np.std(pred_rt, ddof=1)) if len(pred_rt) > 1 else float("nan"),
        "rt_iqr": q(pred_rt, 0.75) - q(pred_rt, 0.25),
        "q10": q(pred_rt, 0.10),
        "q25": q(pred_rt, 0.25),
        "q50": q(pred_rt, 0.50),
        "q75": q(pred_rt, 0.75),
        "q90": q(pred_rt, 0.90),
        "q95": q(pred_rt, 0.95),
        "q99": q(pred_rt, 0.99),
        "q90_minus_q10": q(pred_rt, 0.90) - q(pred_rt, 0.10),
        "q95_minus_median": q(pred_rt, 0.95) - q(pred_rt, 0.50),
        "skewness": float(stats.skew(pred_rt)) if len(pred_rt) > 2 else float("nan"),
        "correct_rt_mean": safe_mean(pred_rt[correct]),
        "error_rt_mean": safe_mean(pred_rt[err]),
        "error_minus_correct_rt": safe_mean(pred_rt[err]) - safe_mean(pred_rt[correct]),
        "correct_rt_median": q(pred_rt[correct], 0.50),
        "error_rt_median": q(pred_rt[err], 0.50),
        "error_minus_correct_rt_median": q(pred_rt[err], 0.50) - q(pred_rt[correct], 0.50),
        "incongruent_correct_rt_mean": safe_mean(pred_rt[incong & correct]),
        "incongruent_error_rt_mean": safe_mean(pred_rt[incong & err]),
        "incongruent_error_minus_correct_rt": safe_mean(pred_rt[incong & err]) - safe_mean(pred_rt[incong & correct]),
        "incongruent_correct_rt_median": q(pred_rt[incong & correct], 0.50),
        "incongruent_error_rt_median": q(pred_rt[incong & err], 0.50),
        "incongruent_error_minus_correct_rt_median": q(pred_rt[incong & err], 0.50) - q(pred_rt[incong & correct], 0.50),
        "fastest_bin_accuracy": fastest_acc,
        "fastest_incongruent_bin_accuracy": safe_mean(inc_fastest["model_correct"].astype(float)) if len(inc_fastest) else float("nan"),
        "caf_slope_proxy": slowest_acc - fastest_acc,
        "human_mean_rt": safe_mean(true_rt),
        "human_median_rt": q(true_rt, 0.50),
        "human_rt_sd": float(np.std(true_rt, ddof=1)) if len(true_rt) > 1 else float("nan"),
        "human_q90": q(true_rt, 0.90),
        "human_q95": q(true_rt, 0.95),
        "human_q90_minus_q10": q(true_rt, 0.90) - q(true_rt, 0.10),
        "human_q95_minus_median": q(true_rt, 0.95) - q(true_rt, 0.50),
        "human_error_minus_correct_rt": safe_mean(true_rt[~human_correct]) - safe_mean(true_rt[human_correct]),
    }


def t0_shift_summary(
    df: pd.DataFrame,
    *,
    condition_name: str,
    variant_type: str,
    current_t0: float,
    t0_values: List[float],
) -> pd.DataFrame:
    pred_rt = df["pred_rt"].to_numpy(dtype=np.float64)
    decision = pred_rt - current_t0
    human_rt = df["true_rt"].to_numpy(dtype=np.float64)
    human_mean = safe_mean(human_rt)
    t0_align = current_t0 + (human_mean - safe_mean(pred_rt))
    all_t0 = list(t0_values) + [float(t0_align)]
    rows = []
    for t0_new in all_t0:
        shifted = decision + t0_new
        tmp = df.copy()
        tmp["pred_rt"] = shifted
        row = metric_summary(condition_name, tmp, variant_type=variant_type)
        rows.append(
            {
                "condition_name": condition_name,
                "variant_type": variant_type,
                "t0_new": float(t0_new),
                "t0_label": "align_human_mean" if abs(float(t0_new) - float(t0_align)) < 1e-9 else f"{t0_new:.2f}",
                "shifted_mean_rt": safe_mean(shifted),
                "shifted_median_rt": q(shifted, 0.50),
                "shifted_q10": q(shifted, 0.10),
                "shifted_q50": q(shifted, 0.50),
                "shifted_q90": q(shifted, 0.90),
                "shifted_q95": q(shifted, 0.95),
                "shifted_q99": q(shifted, 0.99),
                "shifted_rt_sd": float(np.std(shifted, ddof=1)),
                "shifted_rt_iqr": q(shifted, 0.75) - q(shifted, 0.25),
                "shifted_skewness": float(stats.skew(shifted)) if len(shifted) > 2 else float("nan"),
                "human_mean_rt": human_mean,
                "human_median_rt": q(human_rt, 0.50),
                "human_q90": q(human_rt, 0.90),
                "human_q95": q(human_rt, 0.95),
                "mean_rt_gap_to_human": safe_mean(shifted) - human_mean,
                "q90_gap_to_human": q(shifted, 0.90) - q(human_rt, 0.90),
                "q95_gap_to_human": q(shifted, 0.95) - q(human_rt, 0.95),
                "error_minus_correct_rt": row["error_minus_correct_rt"],
                "incongruent_error_minus_correct_rt": row["incongruent_error_minus_correct_rt"],
                "fastest_bin_accuracy": row["fastest_bin_accuracy"],
                "fastest_incongruent_bin_accuracy": row["fastest_incongruent_bin_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_best_variational_specs(summary: pd.DataFrame) -> pd.DataFrame:
    var = summary[(summary["variant_type"].eq("variational")) & (summary["summary_level"].eq("seed_avg"))].copy()
    specs = []
    for sigma_type in ["fixed_sigma", "layer_weighted_sigma", "conflict_dependent_sigma"]:
        part = var[var["sigma_type"].eq(sigma_type)].copy()
        if part.empty:
            continue
        part = part.sort_values("selection_score", ascending=False)
        specs.append(part.iloc[0])
    return pd.DataFrame(specs)


def aggregate_variational_value(rows: List[Dict[str, Any]], group_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_rows = []
    metric_cols = [
        "accuracy",
        "human_choice_agreement",
        "incongruent_error_rate",
        "rt_sd",
        "rt_iqr",
        "q90_minus_q10",
        "q95_minus_median",
        "skewness",
        "error_minus_correct_rt",
        "incongruent_error_minus_correct_rt",
        "fastest_bin_accuracy",
        "fastest_incongruent_bin_accuracy",
    ]
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_seeds"] = int(part["seed"].nunique()) if "seed" in part else int(len(part))
        for col in metric_cols:
            row[f"{col}_mean"] = safe_mean(part[col])
            row[f"{col}_sd"] = float(part[col].std(ddof=1)) if len(part[col].dropna()) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def custom_readout_df(
    base_df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    *,
    readout_rule: str,
    min_decision_time: float,
    sustained_k: int,
    margin: float,
    threshold: float,
    dt_ms: int,
    t0_seconds: float,
) -> pd.DataFrame:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    n, time_steps, _ = traj.shape
    min_step = int(round(min_decision_time / (dt_ms / 1000.0)))
    min_step = max(0, min(min_step, time_steps - 1))
    winner = traj.argmax(axis=2)
    top2 = np.sort(traj, axis=2)[:, :, -2:]
    top = top2[:, :, 1]
    runner = top2[:, :, 0]
    pass_mask = (top > threshold) & ((top - runner) >= margin)
    if min_step > 0:
        pass_mask[:, :min_step] = False
    if sustained_k > 1:
        sustained = np.zeros_like(pass_mask)
        for t in range(time_steps):
            end = min(time_steps, t + sustained_k)
            if end - t == sustained_k:
                same_winner = np.all(winner[:, t:end] == winner[:, t : t + 1], axis=1)
                all_pass = np.all(pass_mask[:, t:end], axis=1)
                sustained[:, t] = same_winner & all_pass
        pass_mask = sustained
    readout_step = np.argmax(pass_mask, axis=1)
    no_cross = ~pass_mask.any(axis=1)
    readout_step[no_cross] = time_steps - 1
    pred_choice = winner[np.arange(n), readout_step].astype(np.int64)
    pred_rt = readout_step.astype(np.float32) * (dt_ms / 1000.0) + float(t0_seconds)
    out = base_df.copy()
    out["condition"] = readout_rule
    out["condition_name"] = readout_rule
    out["pred_rt"] = pred_rt
    out["pred_choice"] = pred_choice
    out["model_correct"] = out["pred_choice"].to_numpy(dtype=np.int64) == out["target_label"].to_numpy(dtype=np.int64)
    return out


def readout_rule_diagnostic(
    base_df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    readout_df: pd.DataFrame,
    *,
    threshold: float,
    dt_ms: int,
    t0_seconds: float,
) -> pd.DataFrame:
    rows = []
    current_metrics = metric_summary(
        "baseline_threshold_crossing_current_decoupled_choice",
        base_df,
        variant_type="readout_rule",
    )
    inc_err_current = readout_df[readout_df["congruency"].eq(1) & ~readout_df["model_correct"]]
    inc_cor_current = readout_df[readout_df["congruency"].eq(1) & readout_df["model_correct"]]
    rows.append(
        {
            "readout_rule": "baseline_threshold_crossing",
            "condition_name": "baseline_threshold_crossing_current_decoupled_choice",
            "choice_rule": "existing_trajectory_max_choice",
            "min_decision_time": 0.0,
            "sustained_k": 1,
            "margin": 0.0,
            "accuracy": current_metrics["accuracy"],
            "human_choice_agreement": current_metrics["human_choice_agreement"],
            "mean_rt": current_metrics["mean_rt"],
            "rt_sd": current_metrics["rt_sd"],
            "q90": current_metrics["q90"],
            "q95": current_metrics["q95"],
            "incongruent_error_rate": current_metrics["incongruent_error_rate"],
            "error_minus_correct_rt": current_metrics["error_minus_correct_rt"],
            "incongruent_error_minus_correct_rt": current_metrics["incongruent_error_minus_correct_rt"],
            "early_flanker_dominance_in_errors": safe_mean(inc_err_current["early_s_flanker_ge_target"].astype(float)),
            "late_target_recovery_in_correct": safe_mean(inc_cor_current["late_s_target_ge_flanker"].astype(float)),
            "target_recovered_before_readout_rate": safe_mean(readout_df["target_recovered_before_readout"].astype(float)),
        }
    )
    configs: List[Tuple[str, float, int, float]] = [("baseline_threshold_crossing_coupled_winner", 0.0, 1, 0.0)]
    for min_time in [0.0, 0.05, 0.10, 0.15]:
        configs.append(("minimum_decision_time", min_time, 1, 0.0))
    for k in [1, 3, 5]:
        configs.append(("sustained_crossing", 0.0, k, 0.0))
    for margin in [0.00, 0.02, 0.05]:
        configs.append(("margin_threshold", 0.0, 1, margin))

    for rule, min_time, k, margin in configs:
        name = f"{rule}_min{min_time:.2f}_k{k}_m{margin:.2f}"
        df = custom_readout_df(
            base_df,
            outputs,
            readout_rule=name,
            min_decision_time=min_time,
            sustained_k=k,
            margin=margin,
            threshold=threshold,
            dt_ms=dt_ms,
            t0_seconds=t0_seconds,
        )
        metrics = metric_summary(name, df, variant_type="readout_rule")
        ro = extract_readout_timing(df, outputs, t0_seconds=t0_seconds, dt_ms=dt_ms)
        inc_err = ro[ro["congruency"].eq(1) & ~ro["model_correct"]]
        inc_cor = ro[ro["congruency"].eq(1) & ro["model_correct"]]
        rows.append(
            {
                "readout_rule": rule,
                "condition_name": name,
                "choice_rule": "winner_at_readout",
                "min_decision_time": min_time,
                "sustained_k": k,
                "margin": margin,
                "accuracy": metrics["accuracy"],
                "human_choice_agreement": metrics["human_choice_agreement"],
                "mean_rt": metrics["mean_rt"],
                "rt_sd": metrics["rt_sd"],
                "q90": metrics["q90"],
                "q95": metrics["q95"],
                "incongruent_error_rate": metrics["incongruent_error_rate"],
                "error_minus_correct_rt": metrics["error_minus_correct_rt"],
                "incongruent_error_minus_correct_rt": metrics["incongruent_error_minus_correct_rt"],
                "early_flanker_dominance_in_errors": safe_mean(inc_err["early_s_flanker_ge_target"].astype(float)),
                "late_target_recovery_in_correct": safe_mean(inc_cor["late_s_target_ge_flanker"].astype(float)),
                "target_recovered_before_readout_rate": safe_mean(ro["target_recovered_before_readout"].astype(float)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["condition_name"])


def plot_outputs(
    *,
    figure_dir: Path,
    readout_df: pd.DataFrame,
    rt_conditions: Dict[str, pd.DataFrame],
    t0_summary: pd.DataFrame,
    var_value: pd.DataFrame,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    det_ro = readout_df[readout_df["condition_name"].eq(BEST_DET)]
    inc = det_ro[det_ro["congruency"].eq(1)].copy()
    inc["group"] = np.where(inc["model_correct"], "incongruent correct", "incongruent error")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    data = [inc.loc[inc["group"].eq(g), "s_target_minus_flanker_at_readout"].dropna() for g in ["incongruent correct", "incongruent error"]]
    ax.boxplot(data, labels=["correct", "error"], showfliers=False)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("s_target - s_flanker at readout")
    ax.set_title("Readout state by correctness")
    fig.tight_layout()
    fig.savefig(figure_dir / "readout_s_target_minus_flanker_by_group.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    data = [inc.loc[inc["group"].eq(g), "target_recovery_time"].dropna() for g in ["incongruent correct", "incongruent error"]]
    ax.boxplot(data, labels=["correct", "error"], showfliers=False)
    ax.set_ylabel("target recovery time (s)")
    ax.set_title("Target recovery time by correctness")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_recovery_time_correct_vs_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for ok, label, color in [(True, "correct", "#2F6B4F"), (False, "error", "#B24A3B")]:
        part = inc[inc["model_correct"].eq(ok)]
        ax.scatter(part["target_recovery_time"], part["decision_time"], s=24, alpha=0.70, label=label, color=color)
    ax.set_xlabel("target recovery time (s)")
    ax.set_ylabel("decision time (s)")
    ax.set_title("RT vs target recovery time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "rt_vs_target_recovery_time.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label in ["best_natural_deterministic", "best_natural_variational_seedavg"]:
        df = rt_conditions[label]
        ax.hist(df["pred_rt"], bins=28, density=True, histtype="step", linewidth=1.8, label=label)
    human_df = rt_conditions["best_natural_deterministic"]
    ax.hist(human_df["true_rt"], bins=28, density=True, histtype="step", linewidth=1.8, label="human", color="#333333")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("density")
    ax.set_title("RT distribution: model vs human")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "rt_distribution_model_vs_human_top_conditions.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, label in zip(axes, ["best_natural_deterministic", "best_natural_variational_seedavg"]):
        df = rt_conditions[label]
        ax.hist(df.loc[df["model_correct"], "pred_rt"], bins=22, density=True, histtype="step", linewidth=1.7, label="correct")
        ax.hist(df.loc[~df["model_correct"], "pred_rt"], bins=22, density=True, histtype="step", linewidth=1.7, label="error")
        ax.set_title(label)
        ax.set_xlabel("model RT (s)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(figure_dir / "correct_vs_error_rt_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for condition_name, part in t0_summary.groupby("condition_name"):
        part = part.sort_values("t0_new")
        ax.plot(part["t0_new"], part["shifted_mean_rt"], marker="o", label=f"{MODEL_ALIASES.get(condition_name, condition_name)} mean")
        ax.plot(part["t0_new"], part["shifted_q90"], marker="s", linestyle="--", label=f"{MODEL_ALIASES.get(condition_name, condition_name)} q90")
        ax.plot(part["t0_new"], part["shifted_q95"], marker="^", linestyle=":", label=f"{MODEL_ALIASES.get(condition_name, condition_name)} q95")
    ax.axhline(float(t0_summary["human_mean_rt"].iloc[0]), color="#333333", linewidth=1.2, label="human mean")
    ax.set_xlabel("t0 (s)")
    ax.set_ylabel("shifted RT quantile (s)")
    ax.set_title("T0 shift moves location, not spread")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "t0_shift_rt_quantiles.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for _, row in var_value.iterrows():
        label = str(row.get("sigma_type", row.get("variant_type", "")))
        color = "#4C78A8" if row.get("variant_type") == "deterministic" else "#54A24B"
        ax.scatter(row["accuracy_mean"], row["q90_minus_q10_mean"], s=75, alpha=0.85, color=color)
        ax.annotate(label, (row["accuracy_mean"], row["q90_minus_q10_mean"]), fontsize=8, alpha=0.85)
    ax.set_xlabel("accuracy")
    ax.set_ylabel("RT spread (q90 - q10)")
    ax.set_title("Variational value: spread vs accuracy")
    fig.tight_layout()
    fig.savefig(figure_dir / "variational_value_rt_spread_vs_accuracy.png", dpi=220)
    plt.close(fig)


def row_lookup(df: pd.DataFrame, group: str, condition: str = BEST_DET) -> pd.Series:
    part = df[df["condition_name"].eq(condition) & df["trial_group"].eq(group)]
    return part.iloc[0] if not part.empty else pd.Series(dtype=object)


def fnum(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def write_markdown_summary(
    path: Path,
    *,
    readout_summary: pd.DataFrame,
    rt_shape: pd.DataFrame,
    t0_summary: pd.DataFrame,
    var_value: pd.DataFrame,
    rule_summary: Optional[pd.DataFrame],
) -> None:
    inc_cor = row_lookup(readout_summary, "incongruent_correct")
    inc_err = row_lookup(readout_summary, "incongruent_error")
    det_rt = rt_shape[rt_shape["model_label"].eq("best_natural_deterministic")].iloc[0]
    var_rt = rt_shape[rt_shape["model_label"].eq("best_natural_variational_seedavg")].iloc[0]
    human_spread = det_rt["human_q90_minus_q10"]
    align = t0_summary[t0_summary["t0_label"].eq("align_human_mean")]
    align_det = align[align["condition_name"].eq(BEST_DET)].iloc[0]
    best_var = var_value[var_value["variant_type"].eq("variational")].sort_values(
        ["human_choice_agreement_mean", "accuracy_mean"], ascending=False
    ).iloc[0]
    rule_text = "本次执行了小范围 readout-rule diagnostic，因为 P0 显示大量试次在 target recovery 前读出。"
    if rule_summary is None or rule_summary.empty:
        rule_text = "本次跳过 readout-rule diagnostic，因为 P0 没有显示明显过早读出。"
    elif not rule_summary.empty:
        current = rule_summary[rule_summary["condition_name"].eq("baseline_threshold_crossing_current_decoupled_choice")]
        coupled = rule_summary[rule_summary["choice_rule"].eq("winner_at_readout")]
        best_coupled = coupled.sort_values(["accuracy", "human_choice_agreement"], ascending=False).iloc[0]
        current_text = ""
        if not current.empty:
            cur = current.iloc[0]
            current_text = (
                f" 当前原始规则保留 trajectory-max choice，accuracy={fnum(cur['accuracy'])}, "
                f"mean RT={fnum(cur['mean_rt'])}。"
            )
        rule_text += (
            current_text
            + f" 但如果强制用 RT 时刻 winner 作答，最好的小规则也只有 `{best_coupled['condition_name']}`，"
            + f"accuracy={fnum(best_coupled['accuracy'])}, mean RT={fnum(best_coupled['mean_rt'])}, "
            + f"incongruent error rate={fnum(best_coupled['incongruent_error_rate'])}。"
        )

    text = f"""# Readout Timing and RT Pattern Audit for Natural Layer-to-Time WW

## 1. Goal

This analysis checks whether natural layer-to-time evidence, after already producing DMC-like Wong-Wang internal dynamics, also has aligned readout timing, RT distribution shape, correct/error RT pattern, and meaningful added value from variational sampling.

## 2. Why Mean RT Is Not the Main Target

Current mean RT is short. A non-decision time shift `t0` can move the whole RT distribution later, so mean RT alone is not the key diagnostic. The important checks are RT spread, skewness, error-minus-correct RT, incongruent error-minus-correct RT, and fastest-bin accuracy, because `t0` does not change these shape and ordering measures.

## 3. Readout Timing Findings

- In incongruent-correct trials, mean `s_target - s_flanker` at readout is `{fnum(inc_cor.get('s_target_minus_flanker_at_readout_mean'))}`, with target-dominant readout rate `{fnum(inc_cor.get('target_dominant_at_readout_rate'))}`.
- In incongruent-error trials, mean `s_target - s_flanker` at readout is `{fnum(inc_err.get('s_target_minus_flanker_at_readout_mean'))}`, with flanker-dominant readout rate `{fnum(inc_err.get('flanker_dominant_at_readout_rate'))}`.
- Target recovered before readout in incongruent-correct trials at rate `{fnum(inc_cor.get('target_recovered_before_readout_rate'))}`; this is higher than errors but still low in absolute terms.
- Readout happened before target recovery in incongruent-error trials at rate `{fnum(inc_err.get('readout_before_target_recovery_rate'))}`.
- Mean target recovery time is `{fnum(inc_cor.get('target_recovery_time_mean'))}` for incongruent-correct and `{fnum(inc_err.get('target_recovery_time_mean'))}` for incongruent-error.

Interpretation: errors are strongly tied to readout occurring while the flanker state is still dominant or before target recovery. Correct trials show earlier/stronger later recovery than errors, but the actual RT readout still usually occurs before recovery. This means the current internal DMC-like trajectory is real, but the saved model choice is not fully explained by the state at the RT instant; it is partly supported by later trajectory evidence.

## 4. RT Distribution Shape Findings

- Best deterministic mean RT is `{fnum(det_rt['mean_rt'])}`, q90-q10 is `{fnum(det_rt['q90_minus_q10'])}`, q95-median is `{fnum(det_rt['q95_minus_median'])}`, skewness is `{fnum(det_rt['skewness'])}`.
- Human q90-q10 on the same rows is `{fnum(human_spread)}`, and human q95-median is `{fnum(det_rt['human_q95_minus_median'])}`.
- Deterministic error-minus-correct RT is `{fnum(det_rt['error_minus_correct_rt'])}`; incongruent error-minus-correct RT is `{fnum(det_rt['incongruent_error_minus_correct_rt'])}`.
- Variational mean RT is `{fnum(var_rt['mean_rt'])}`, q90-q10 is `{fnum(var_rt['q90_minus_q10'])}`, and incongruent error-minus-correct RT is `{fnum(var_rt['incongruent_error_minus_correct_rt'])}`.

Interpretation: natural layer-to-time is clearly more conflict-like than final-only because it produces incongruent errors, and it is more stable than middle-only because accuracy does not collapse. However, the model RT distribution remains far narrower than human RT. The fast-error direction exists only weakly.

## 5. T0-shift Findings

- For deterministic, the t0 that aligns mean RT to human mean is `{fnum(align_det['t0_new'])}`.
- After mean alignment, deterministic q90 gap to human is `{fnum(align_det['q90_gap_to_human'])}` and q95 gap to human is `{fnum(align_det['q95_gap_to_human'])}`.
- The shifted RT SD stays `{fnum(align_det['shifted_rt_sd'])}` because t0 only shifts all RTs by the same amount.

Interpretation: t0 solves the mean-level offset but cannot solve narrow spread, skewness, or weak error/correct RT separation.

## 6. Variational Value Findings

- Best listed variational design by human agreement is `{best_var['sigma_type']}` with accuracy mean `{fnum(best_var['accuracy_mean'])}` and human-choice agreement mean `{fnum(best_var['human_choice_agreement_mean'])}`.
- Its q90-q10 mean is `{fnum(best_var['q90_minus_q10_mean'])}`.
- Its incongruent error rate mean is `{fnum(best_var['incongruent_error_rate_mean'])}`.
- Compared with deterministic, variational sampling gives a small positive change in accuracy/agreement but does not materially widen RT spread.

Interpretation: variational sampling is worth keeping as subjective evidence uncertainty, but current results do not show it as the source of the DMC-like structure. The structure comes from deterministic `mu_t`.

## 7. Optional Readout-rule Findings

{rule_text}

## 8. Interpretation

Natural layer-to-time logits/evidence can produce DMC-like internal dynamics in Wong-Wang: early flanker pull and late target recovery are visible in the accumulated decision states. The current critical issue is aligning those internal trajectories with actual response/readout, RT shape, and error patterns. In particular, RT is often assigned before target recovery, while final choice can still reflect later trajectory evidence. t0 can fix mean RT but cannot replace RT-shape analysis. Variational sampling should be treated as subjective evidence uncertainty on top of deterministic `mu_t`, not as the mechanism that creates DMC-like dynamics. AR(1) and stochastic stopping should remain paused as main-line explanations for now.

## 9. Recommended Next Steps

1. Do subject-level and image-identity validation next, because the current analysis is still a 500-row diagnostic.
2. Keep a small readout-rule follow-up if early readout remains the main failure mode.
3. Keep variational sampling in the next stage as a secondary uncertainty mechanism, but do not optimize it before validating deterministic `mu_t`.
4. Continue pausing AR(1) as a main-line addition.
5. Continue pausing stochastic stopping as a main-line addition.
6. Export compressed or raw trajectories for future strict readout audits.

## 10. Short Chinese Summary for Discussion

目前结果说明，CNN layer-to-time evidence 已经能让 WW 内部自然表现出类似 DMC 的 early flanker / late target 动态。下一步不应该只追求 mean RT，因为 mean RT 可以通过 t0 整体平移；关键是检查 readout 是否真的发生在 target/flanker 竞争的关键阶段，以及 RT 分布形态和 error/correct 模式是否像人类。本次结果显示，错误试次更常在 target recovery 前读出，正确试次更常在 target recovery 后读出，所以内部竞争轨迹确实能解释一部分正确/错误差异。但模型 RT 分布仍明显比人类窄，t0 只能修正平均水平，不能修正分布形状。variational sampling 有小幅正向价值，可以继续作为主观证据不确定性机制保留；但 DMC-like 结构主要来自 deterministic μ_t。AR(1) 和 stochastic stopping 目前仍不应优先引入。
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
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures_readout_rt"

    cache = load_cache(resolve_path(args.cache_path), args.max_trials)
    summary = pd.read_csv(output_dir / "natural_layer_to_time_var_ww_summary.csv")

    det_df, det_outputs = run_natural_condition(
        cache,
        condition_name=BEST_DET,
        variant_type="deterministic",
        schedule_type="natural_smooth_5stage",
        normalization="per_layer_gap_scale",
        evidence_gain=2.0,
        threshold=0.12,
        seed=args.seed,
        time_steps=args.time_steps,
        dt_ms=args.dt_ms,
        t0_seconds=args.t0_seconds,
        noise_ampa=args.noise_ampa,
        readout_mode=args.readout_mode,
        choice_temperature=args.choice_temperature,
    )
    readout_trial = extract_readout_timing(det_df, det_outputs, t0_seconds=args.t0_seconds, dt_ms=args.dt_ms)
    readout_trial.to_csv(output_dir / "readout_timing_trial_level.csv", index=False)
    ro_summary = readout_timing_summary(readout_trial)
    ro_summary.to_csv(output_dir / "readout_timing_summary.csv", index=False)

    best_var_specs = select_best_variational_specs(summary)
    best_fixed = best_var_specs[best_var_specs["sigma_type"].eq("fixed_sigma")].iloc[0]
    var_seed_dfs: List[pd.DataFrame] = []
    var_metric_rows: List[Dict[str, Any]] = []
    variational_seedavg_df: Optional[pd.DataFrame] = None
    seeds = [args.seed + i for i in range(5)]
    all_value_rows: List[Dict[str, Any]] = []
    det_metric = metric_summary(BEST_DET, det_df, variant_type="deterministic")
    det_metric["seed"] = args.seed
    det_metric["sigma_type"] = "none"
    all_value_rows.append(det_metric)

    for _, spec in best_var_specs.iterrows():
        sigma_type = str(spec["sigma_type"])
        sigma_base = float(spec["sigma_base"])
        sigma_middle = float(spec["sigma_middle"])
        sigma_conflict = float(spec["sigma_conflict"])
        for seed in seeds:
            condition_name = (
                f"natural_var_natural_smooth_5stage_norm-per_layer_gap_scale_g2.00_th0.12_"
                f"{sigma_type}_sb{sigma_base:.2f}_sm{sigma_middle:.2f}_sc{sigma_conflict:.2f}_seed{seed}"
            )
            vdf, _ = run_natural_condition(
                cache,
                condition_name=condition_name,
                variant_type="variational",
                schedule_type="natural_smooth_5stage",
                normalization="per_layer_gap_scale",
                evidence_gain=2.0,
                threshold=0.12,
                sigma_type=sigma_type,
                sigma_base=sigma_base,
                sigma_middle=sigma_middle,
                sigma_conflict=sigma_conflict,
                seed=seed,
                time_steps=args.time_steps,
                dt_ms=args.dt_ms,
                t0_seconds=args.t0_seconds,
                noise_ampa=args.noise_ampa,
                readout_mode=args.readout_mode,
                choice_temperature=args.choice_temperature,
            )
            metric = metric_summary(condition_name, vdf, variant_type="variational")
            metric["seed"] = seed
            metric["sigma_type"] = sigma_type
            metric["sigma_base"] = sigma_base
            metric["sigma_middle"] = sigma_middle
            metric["sigma_conflict"] = sigma_conflict
            all_value_rows.append(metric)
            if sigma_type == str(best_fixed["sigma_type"]) and sigma_base == float(best_fixed["sigma_base"]):
                var_seed_dfs.append(vdf)
                var_metric_rows.append(metric)

    # Build a seed-averaged behavioral table by averaging per-trial RT and choosing the modal response.
    merged = var_seed_dfs[0].copy()
    rt_stack = np.vstack([df["pred_rt"].to_numpy(dtype=np.float64) for df in var_seed_dfs])
    choice_stack = np.vstack([df["pred_choice"].to_numpy(dtype=np.int64) for df in var_seed_dfs])
    modal_choice = []
    for col in choice_stack.T:
        vals, counts = np.unique(col, return_counts=True)
        modal_choice.append(vals[np.argmax(counts)])
    merged["pred_rt"] = rt_stack.mean(axis=0)
    merged["pred_choice"] = np.asarray(modal_choice, dtype=np.int64)
    merged["model_correct"] = merged["pred_choice"].to_numpy(dtype=np.int64) == merged["target_label"].to_numpy(dtype=np.int64)
    merged["condition"] = BEST_VAR_SEEDAVG
    merged["condition_name"] = BEST_VAR_SEEDAVG
    merged["seed"] = "avg"
    variational_seedavg_df = merged

    rt_shape_rows = [metric_summary(BEST_DET, det_df, variant_type="deterministic")]
    rt_shape_rows.append(metric_summary(BEST_VAR_SEEDAVG, variational_seedavg_df, variant_type="variational"))

    for input_condition in ["final_only", "middle_only", "handcrafted_dmc_positive_control"]:
        bdf, _ = run_baseline_condition(
            cache,
            input_condition,
            time_steps=args.time_steps,
            dt_ms=args.dt_ms,
            t0_seconds=args.t0_seconds,
            threshold=0.22,
            noise_ampa=args.noise_ampa,
            readout_mode=args.readout_mode,
            choice_temperature=args.choice_temperature,
            seed=args.seed,
        )
        rt_shape_rows.append(metric_summary(bdf["condition_name"].iloc[0], bdf, variant_type="baseline"))

    refined_path = resolve_path("artifacts/results/diagnostics/refined_layerwise_vs_dmc_same_subset/refined_vs_dmc_trial_level.csv")
    if refined_path.exists():
        refined_all = pd.read_csv(refined_path)
        refined = refined_all[refined_all["condition"].eq("refined_best_layerwise_gate")].copy()
        if not refined.empty:
            refined["condition_name"] = "refined_best_layerwise_gate"
            refined["variant_type"] = "baseline"
            rt_shape_rows.append(metric_summary("refined_best_layerwise_gate", refined, variant_type="baseline"))

    rt_shape = pd.DataFrame(rt_shape_rows)
    rt_shape.to_csv(output_dir / "rt_shape_error_pattern_summary.csv", index=False)

    t0_parts = [
        t0_shift_summary(
            det_df,
            condition_name=BEST_DET,
            variant_type="deterministic",
            current_t0=args.t0_seconds,
            t0_values=[0.25, 0.35, 0.45, 0.55],
        ),
        t0_shift_summary(
            variational_seedavg_df,
            condition_name=BEST_VAR_SEEDAVG,
            variant_type="variational",
            current_t0=args.t0_seconds,
            t0_values=[0.25, 0.35, 0.45, 0.55],
        ),
    ]
    t0_out = pd.concat(t0_parts, ignore_index=True)
    t0_out.to_csv(output_dir / "t0_shift_sensitivity_summary.csv", index=False)

    var_value = aggregate_variational_value(
        all_value_rows,
        ["variant_type", "sigma_type", "sigma_base", "sigma_middle", "sigma_conflict"],
    )
    var_value.to_csv(output_dir / "variational_value_check.csv", index=False)

    inc_err = ro_summary[ro_summary["condition_name"].eq(BEST_DET) & ro_summary["trial_group"].eq("incongruent_error")]
    do_rule = bool(not inc_err.empty and float(inc_err.iloc[0]["readout_before_target_recovery_rate"]) > 0.50)
    rule_summary = None
    if do_rule:
        rule_summary = readout_rule_diagnostic(
            det_df,
            det_outputs,
            readout_trial,
            threshold=0.12,
            dt_ms=args.dt_ms,
            t0_seconds=args.t0_seconds,
        )
        rule_summary.to_csv(output_dir / "readout_rule_diagnostic_summary.csv", index=False)

    rt_conditions = {
        "best_natural_deterministic": det_df,
        "best_natural_variational_seedavg": variational_seedavg_df,
    }
    plot_outputs(
        figure_dir=figure_dir,
        readout_df=readout_trial,
        rt_conditions=rt_conditions,
        t0_summary=t0_out,
        var_value=var_value,
    )

    write_markdown_summary(
        output_dir / "readout_rt_pattern_next_step_summary.md",
        readout_summary=ro_summary,
        rt_shape=rt_shape,
        t0_summary=t0_out,
        var_value=var_value,
        rule_summary=rule_summary,
    )

    metadata = {
        "best_deterministic": BEST_DET,
        "best_variational_seedavg": BEST_VAR_SEEDAVG,
        "readout_rule_diagnostic_ran": do_rule,
        "outputs": {
            "readout_timing_trial_level": str(output_dir / "readout_timing_trial_level.csv"),
            "readout_timing_summary": str(output_dir / "readout_timing_summary.csv"),
            "rt_shape_error_pattern_summary": str(output_dir / "rt_shape_error_pattern_summary.csv"),
            "t0_shift_sensitivity_summary": str(output_dir / "t0_shift_sensitivity_summary.csv"),
            "variational_value_check": str(output_dir / "variational_value_check.csv"),
            "readout_rule_diagnostic_summary": str(output_dir / "readout_rule_diagnostic_summary.csv") if do_rule else None,
            "markdown_summary": str(output_dir / "readout_rt_pattern_next_step_summary.md"),
            "figure_dir": str(figure_dir),
        },
    }
    (output_dir / "readout_rt_pattern_next_step_metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

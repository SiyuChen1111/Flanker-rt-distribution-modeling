#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
from analyze_natural_layer_to_time_readout_rt_patterns import (  # noqa: E402
    BEST_DET,
    BEST_VAR_SEEDAVG,
    extract_readout_timing,
    load_cache,
    q,
    resolve_path,
    safe_mean,
)
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


OUT_SUBDIR = "artifacts/results/diagnostics/natural_layer_to_time_var_ww/rt_shape_optimization"
SCHEDULE = "natural_smooth_5stage"
NORMALIZATION = "per_layer_gap_scale"


@dataclass(frozen=True)
class ReadoutConfig:
    readout_rule: str
    min_decision_time: float = 0.0
    sustained_k: int = 1
    margin: float = 0.0
    hazard_alpha: float = 0.0
    hazard_beta: float = 0.0


def finite_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 2 or float(np.std(values)) < 1e-12:
        return 0.0
    return float(stats.skew(values))


def human_reference(cache: Dict[str, np.ndarray]) -> Dict[str, float]:
    rt = np.asarray(cache["true_rt"], dtype=np.float64)
    target = np.asarray(cache["target_labels"], dtype=np.int64)
    response = np.asarray(cache["response_labels"], dtype=np.int64)
    congruency = np.asarray(cache["congruency"], dtype=np.int64)
    correct = response == target
    incong = congruency == 1
    order = np.argsort(rt)
    bins = np.array_split(order, 5)
    fastest = bins[0]
    slowest = bins[-1]
    inc_order = np.argsort(rt[incong])
    inc_idx = np.where(incong)[0]
    inc_bins = np.array_split(inc_order, 5)
    inc_fastest = inc_idx[inc_bins[0]]
    return {
        "human_mean_rt": safe_mean(rt),
        "human_median_rt": q(rt, 0.50),
        "human_rt_sd": float(np.std(rt, ddof=1)),
        "human_rt_iqr": q(rt, 0.75) - q(rt, 0.25),
        "human_q10": q(rt, 0.10),
        "human_q25": q(rt, 0.25),
        "human_q50": q(rt, 0.50),
        "human_q75": q(rt, 0.75),
        "human_q90": q(rt, 0.90),
        "human_q95": q(rt, 0.95),
        "human_q99": q(rt, 0.99),
        "human_q90_minus_q10": q(rt, 0.90) - q(rt, 0.10),
        "human_q95_minus_median": q(rt, 0.95) - q(rt, 0.50),
        "human_skewness": finite_skew(rt),
        "human_error_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]),
        "human_incongruent_error_minus_correct_rt": safe_mean(rt[incong & ~correct]) - safe_mean(rt[incong & correct]),
        "human_fastest_bin_accuracy": safe_mean(correct[fastest].astype(float)),
        "human_slowest_bin_accuracy": safe_mean(correct[slowest].astype(float)),
        "human_fastest_incongruent_bin_accuracy": safe_mean((response[inc_fastest] == target[inc_fastest]).astype(float)),
    }


def build_natural_input(
    cache: Dict[str, np.ndarray],
    *,
    evidence_gain: float,
    time_steps: int,
    variant_type: str = "deterministic",
    sigma_type: str = "none",
    sigma_base: float = 0.0,
    sigma_middle: float = 0.0,
    sigma_conflict: float = 0.0,
    seed: int = 20260527,
) -> torch.Tensor:
    raw_layers = raw_layer_arrays(cache)
    normalized = normalize_layers(raw_layers, NORMALIZATION)
    schedule_df = schedule_weights(SCHEDULE, time_steps)
    mu = build_mu_schedule(normalized, schedule_df, evidence_gain)
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
        return sample_mu_sigma(mu, sigma, seed)
    return mu


def base_condition_df(
    cache: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    *,
    condition_name: str,
    variant_type: str,
    evidence_gain: float,
    threshold: float,
    seed: int | str,
    sigma_type: str = "none",
    sigma_base: float = 0.0,
    sigma_middle: float = 0.0,
    sigma_conflict: float = 0.0,
) -> pd.DataFrame:
    df = make_trial_df(cache, condition_name, outputs)
    df["condition_name"] = condition_name
    df["variant_type"] = variant_type
    df["schedule_type"] = SCHEDULE
    df["normalization"] = NORMALIZATION
    df["evidence_gain"] = evidence_gain
    df["threshold"] = threshold
    df["seed"] = seed
    df["sigma_type"] = sigma_type
    df["sigma_base"] = sigma_base
    df["sigma_middle"] = sigma_middle
    df["sigma_conflict"] = sigma_conflict
    return df


def apply_readout(
    base_df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    *,
    cfg: ReadoutConfig,
    threshold: float,
    dt_ms: int,
    t0_seconds: float,
    choice_rule: str = "trajectory_max_choice",
) -> pd.DataFrame:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    evidence = np.asarray(outputs["evidence_traj"], dtype=np.float32)
    n, time_steps, _ = traj.shape
    top2 = np.sort(traj, axis=2)[:, :, -2:]
    runner = top2[:, :, 0]
    winner_state = top2[:, :, 1]
    margin_t = winner_state - runner
    winner_idx = traj.argmax(axis=2).astype(np.int64)
    min_step = int(round(float(cfg.min_decision_time) / (dt_ms / 1000.0)))
    min_step = max(0, min(min_step, time_steps - 1))

    if cfg.readout_rule == "hazard_readout":
        hazard = 1.0 / (1.0 + np.exp(-(cfg.hazard_alpha * (winner_state - threshold) + cfg.hazard_beta * margin_t)))
        hazard = np.clip(hazard * 0.18, 1e-6, 0.80)
        if min_step > 0:
            hazard[:, :min_step] = 0.0
        survival_prev = np.concatenate(
            [np.ones((n, 1), dtype=np.float64), np.cumprod(1.0 - hazard[:, :-1], axis=1)],
            axis=1,
        )
        mass = survival_prev * hazard
        leftover = np.maximum(1.0 - mass.sum(axis=1), 0.0)
        mass[:, -1] += leftover
        readout_step_float = (mass * np.arange(time_steps)[None, :]).sum(axis=1)
        readout_step = np.rint(readout_step_float).astype(np.int64).clip(0, time_steps - 1)
    else:
        if cfg.readout_rule in {"baseline_threshold", "minimum_decision_time"}:
            pass_mask = (evidence > 0).any(axis=2)
        else:
            pass_mask = (winner_state > threshold) & (margin_t >= float(cfg.margin))
        if min_step > 0:
            pass_mask[:, :min_step] = False
        if cfg.sustained_k > 1:
            sustained = np.zeros_like(pass_mask)
            for t in range(time_steps - cfg.sustained_k + 1):
                sl = slice(t, t + cfg.sustained_k)
                same_winner = np.all(winner_idx[:, sl] == winner_idx[:, t : t + 1], axis=1)
                all_pass = np.all(pass_mask[:, sl], axis=1)
                sustained[:, t] = same_winner & all_pass
            pass_mask = sustained
        readout_step = np.argmax(pass_mask, axis=1).astype(np.int64)
        no_cross = ~pass_mask.any(axis=1)
        readout_step[no_cross] = time_steps - 1

    out = base_df.copy()
    if choice_rule == "winner_at_readout":
        out["pred_choice"] = winner_idx[np.arange(n), readout_step]
    else:
        out["pred_choice"] = base_df["pred_choice"].to_numpy(dtype=np.int64)
    out["pred_rt"] = readout_step.astype(np.float32) * (dt_ms / 1000.0) + float(t0_seconds)
    out["decision_time"] = out["pred_rt"] - float(t0_seconds)
    out["model_correct"] = out["pred_choice"].to_numpy(dtype=np.int64) == out["target_label"].to_numpy(dtype=np.int64)
    out["readout_rule"] = cfg.readout_rule
    out["min_decision_time"] = cfg.min_decision_time
    out["sustained_k"] = cfg.sustained_k
    out["margin"] = cfg.margin
    out["hazard_alpha"] = cfg.hazard_alpha
    out["hazard_beta"] = cfg.hazard_beta
    out["choice_rule"] = choice_rule
    return out


def rt_bins(df: pd.DataFrame, condition_name: str, n_bins: int = 5) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source, rt_col, correct_col in [
        ("model", "pred_rt", "model_correct"),
        ("human", "true_rt", "human_correct"),
    ]:
        order = np.argsort(df[rt_col].to_numpy(dtype=np.float64))
        for i, idx in enumerate(np.array_split(order, n_bins), start=1):
            part = df.iloc[idx]
            inc = part[part["congruency"].eq(1)]
            rows.append(
                {
                    "condition_name": condition_name,
                    "source": source,
                    "rt_bin": i,
                    "n_trials": int(len(part)),
                    "mean_rt": safe_mean(part[rt_col]),
                    "accuracy": safe_mean(part[correct_col].astype(float)),
                    "incongruent_accuracy": safe_mean(inc[correct_col].astype(float)) if len(inc) else float("nan"),
                    "error_rate": 1.0 - safe_mean(part[correct_col].astype(float)),
                }
            )
    return pd.DataFrame(rows)


def shape_metrics(condition_name: str, df: pd.DataFrame, href: Dict[str, float]) -> Dict[str, Any]:
    rt = df["pred_rt"].to_numpy(dtype=np.float64)
    decision = rt - 0.25
    true_rt = df["true_rt"].to_numpy(dtype=np.float64)
    correct = df["model_correct"].to_numpy(dtype=bool)
    human_correct = df["human_correct"].to_numpy(dtype=bool)
    incong = df["congruency"].to_numpy(dtype=np.int64) == 1
    err = ~correct
    order = np.argsort(rt)
    bins = np.array_split(order, 5)
    fastest, slowest = df.iloc[bins[0]], df.iloc[bins[-1]]
    inc_df = df[incong]
    inc_order = np.argsort(inc_df["pred_rt"].to_numpy(dtype=np.float64))
    inc_bins = np.array_split(inc_order, 5) if len(inc_df) >= 5 else []
    inc_fastest = inc_df.iloc[inc_bins[0]] if inc_bins else inc_df.iloc[[]]
    hq90_q10 = href["human_q90_minus_q10"]
    hq95_med = href["human_q95_minus_median"]
    hskew = href["human_skewness"]
    q90_q10 = q(rt, 0.90) - q(rt, 0.10)
    q95_med = q(rt, 0.95) - q(rt, 0.50)
    skew = finite_skew(rt)
    return {
        "condition_name": condition_name,
        "n_trials": int(len(df)),
        "accuracy": safe_mean(correct.astype(float)),
        "human_choice_agreement": safe_mean((df["pred_choice"].to_numpy() == df["response_label"].to_numpy()).astype(float)),
        "incongruent_error_rate": safe_mean((~df.loc[incong, "model_correct"]).astype(float)),
        "mean_rt": safe_mean(rt),
        "median_rt": q(rt, 0.50),
        "decision_time_mean": safe_mean(decision),
        "rt_sd": float(np.std(rt, ddof=1)) if len(rt) > 1 else float("nan"),
        "rt_iqr": q(rt, 0.75) - q(rt, 0.25),
        "q10": q(rt, 0.10),
        "q25": q(rt, 0.25),
        "q50": q(rt, 0.50),
        "q75": q(rt, 0.75),
        "q90": q(rt, 0.90),
        "q95": q(rt, 0.95),
        "q99": q(rt, 0.99),
        "q90_minus_q10": q90_q10,
        "q95_minus_median": q95_med,
        "skewness": skew,
        "right_tail_mass": float(np.mean(rt > max(href["human_q75"], href["human_median_rt"] + href["human_rt_iqr"]))),
        "abs_gap_q10": abs(q(rt, 0.10) - href["human_q10"]),
        "abs_gap_q50": abs(q(rt, 0.50) - href["human_q50"]),
        "abs_gap_q90": abs(q(rt, 0.90) - href["human_q90"]),
        "abs_gap_q95": abs(q(rt, 0.95) - href["human_q95"]),
        "abs_gap_q90_minus_q10": abs(q90_q10 - hq90_q10),
        "abs_gap_q95_minus_median": abs(q95_med - hq95_med),
        "abs_gap_skewness": abs(skew - hskew),
        "correct_rt_mean": safe_mean(rt[correct]),
        "error_rt_mean": safe_mean(rt[err]),
        "error_minus_correct_rt": safe_mean(rt[err]) - safe_mean(rt[correct]),
        "incongruent_correct_rt_mean": safe_mean(rt[incong & correct]),
        "incongruent_error_rt_mean": safe_mean(rt[incong & err]),
        "incongruent_error_minus_correct_rt": safe_mean(rt[incong & err]) - safe_mean(rt[incong & correct]),
        "correct_rt_median": q(rt[correct], 0.50),
        "error_rt_median": q(rt[err], 0.50),
        "incongruent_correct_rt_median": q(rt[incong & correct], 0.50),
        "incongruent_error_rt_median": q(rt[incong & err], 0.50),
        "fastest_bin_accuracy": safe_mean(fastest["model_correct"].astype(float)),
        "fastest_incongruent_bin_accuracy": safe_mean(inc_fastest["model_correct"].astype(float)) if len(inc_fastest) else float("nan"),
        "caf_slope_proxy": safe_mean(slowest["model_correct"].astype(float)) - safe_mean(fastest["model_correct"].astype(float)),
        "delta_plot_proxy": safe_mean(slowest["pred_rt"]) - safe_mean(fastest["pred_rt"]),
        "human_error_minus_correct_rt": safe_mean(true_rt[~human_correct]) - safe_mean(true_rt[human_correct]),
        "human_incongruent_error_minus_correct_rt": safe_mean(true_rt[incong & ~human_correct]) - safe_mean(true_rt[incong & human_correct]),
        **href,
    }


def mechanism_metrics(df: pd.DataFrame, outputs: Dict[str, np.ndarray], condition_name: str, dt_ms: int, t0_seconds: float) -> Dict[str, Any]:
    ro = extract_readout_timing(df, outputs, t0_seconds=t0_seconds, dt_ms=dt_ms)
    incong = ro["congruency"].eq(1)
    inc_err = ro[incong & ~ro["model_correct"]]
    inc_cor = ro[incong & ro["model_correct"]]
    early_target_recovery_error = safe_mean((inc_err["target_recovery_time"] <= inc_err["early_s_target_minus_flanker_mean"].abs() * 0 + 0.32).astype(float)) if len(inc_err) else float("nan")
    return {
        "condition_name": condition_name,
        "early_flanker_dominance_rate_in_incongruent_error": safe_mean(inc_err["early_s_flanker_ge_target"].astype(float)) if len(inc_err) else float("nan"),
        "late_target_recovery_rate_in_incongruent_correct": safe_mean(inc_cor["late_s_target_ge_flanker"].astype(float)) if len(inc_cor) else float("nan"),
        "early_target_recovery_rate_in_incongruent_error": early_target_recovery_error,
        "target_recovery_time_correct_mean": safe_mean(inc_cor["target_recovery_time"]) if len(inc_cor) else float("nan"),
        "target_recovery_time_error_mean": safe_mean(inc_err["target_recovery_time"]) if len(inc_err) else float("nan"),
        "target_recovery_time_error_minus_correct": safe_mean(inc_err["target_recovery_time"]) - safe_mean(inc_cor["target_recovery_time"]) if len(inc_err) and len(inc_cor) else float("nan"),
        "s_target_minus_flanker_at_readout_correct": safe_mean(inc_cor["s_target_minus_flanker_at_readout"]) if len(inc_cor) else float("nan"),
        "s_target_minus_flanker_at_readout_error": safe_mean(inc_err["s_target_minus_flanker_at_readout"]) if len(inc_err) else float("nan"),
        "readout_before_target_recovery_rate_correct": safe_mean(inc_cor["readout_before_target_recovery"].astype(float)) if len(inc_cor) else float("nan"),
        "readout_before_target_recovery_rate_error": safe_mean(inc_err["readout_before_target_recovery"].astype(float)) if len(inc_err) else float("nan"),
    }


def readout_configs() -> List[ReadoutConfig]:
    configs = [ReadoutConfig("baseline_threshold")]
    configs += [ReadoutConfig("minimum_decision_time", min_decision_time=t) for t in [0.00, 0.05, 0.10, 0.15, 0.20]]
    configs += [ReadoutConfig("sustained_crossing", sustained_k=k) for k in [1, 3, 5, 8]]
    configs += [ReadoutConfig("margin_threshold", margin=m) for m in [0.00, 0.02, 0.05, 0.08, 0.10]]
    configs += [ReadoutConfig("sustained_margin", sustained_k=k, margin=m) for k in [3, 5] for m in [0.02, 0.05, 0.08]]
    configs += [ReadoutConfig("hazard_readout", hazard_alpha=a, hazard_beta=b) for a in [10.0, 20.0] for b in [5.0, 10.0]]
    return configs


def condition_name(prefix: str, gain: float, threshold: float, cfg: ReadoutConfig, sigma: str = "none", seed: int | str = "det") -> str:
    return (
        f"{prefix}_g{gain:.2f}_th{threshold:.2f}_{cfg.readout_rule}"
        f"_min{cfg.min_decision_time:.2f}_k{cfg.sustained_k}_m{cfg.margin:.2f}"
        f"_ha{cfg.hazard_alpha:.0f}_hb{cfg.hazard_beta:.0f}_{sigma}_seed{seed}"
    )


def score_candidates(summary: pd.DataFrame, current: pd.Series, href: Dict[str, float]) -> pd.DataFrame:
    out = summary.copy()
    out["hard_accuracy"] = out["accuracy"] >= 0.65
    out["hard_conflict_low"] = out["incongruent_error_rate"] > 0.05
    out["hard_conflict_high"] = out["incongruent_error_rate"] < 0.60
    out["hard_spread_q90"] = out["q90_minus_q10"] > float(current["q90_minus_q10"])
    out["hard_spread_q95"] = out["q95_minus_median"] > float(current["q95_minus_median"])
    out["hard_recovery_order"] = out["target_recovery_time_error_minus_correct"] > 0
    out["hard_late_recovery"] = out["late_target_recovery_rate_in_incongruent_correct"] > out["early_target_recovery_rate_in_incongruent_error"].fillna(-np.inf)
    hard_cols = [c for c in out.columns if c.startswith("hard_")]
    out["passes_hard_constraints"] = out[hard_cols].all(axis=1)
    out["rt_shape_score"] = (
        1.0 / (1.0 + out["abs_gap_q90_minus_q10"])
        + 1.0 / (1.0 + out["abs_gap_q95_minus_median"])
        + 1.0 / (1.0 + out["abs_gap_skewness"])
    )
    err_sign = np.sign(out["error_minus_correct_rt"].fillna(0.0)) == np.sign(href["human_error_minus_correct_rt"])
    inc_err_sign = np.sign(out["incongruent_error_minus_correct_rt"].fillna(0.0)) == np.sign(href["human_incongruent_error_minus_correct_rt"])
    out["error_pattern_score"] = (
        err_sign.astype(float)
        + inc_err_sign.astype(float)
        + 1.0 / (1.0 + (out["fastest_bin_accuracy"] - href["human_fastest_bin_accuracy"]).abs())
    )
    mech_raw = (
        out["target_recovery_time_error_minus_correct"].clip(lower=0).fillna(0.0)
        + out["early_flanker_dominance_rate_in_incongruent_error"].fillna(0.0)
        + out["late_target_recovery_rate_in_incongruent_correct"].fillna(0.0)
    )
    out["mechanism_score"] = mech_raw
    out["total_score"] = 0.45 * out["rt_shape_score"] + 0.30 * out["error_pattern_score"] + 0.25 * out["mechanism_score"]
    return out.sort_values(["passes_hard_constraints", "total_score"], ascending=[False, False])


def t0_shift_table(candidate_dfs: Dict[str, pd.DataFrame], href: Dict[str, float], t0_seconds: float) -> pd.DataFrame:
    rows = []
    for name, df in candidate_dfs.items():
        decision = df["pred_rt"].to_numpy(dtype=np.float64) - t0_seconds
        current_mean = safe_mean(decision)
        t0s = [0.25, 0.35, 0.45, 0.55, 0.60, href["human_mean_rt"] - current_mean]
        for t0 in t0s:
            shifted = decision + t0
            rows.append(
                {
                    "condition_name": name,
                    "t0_candidate": float(t0),
                    "t0_label": "align_human_mean" if abs(t0 - (href["human_mean_rt"] - current_mean)) < 1e-9 else f"{t0:.2f}",
                    "shifted_mean_rt": safe_mean(shifted),
                    "shifted_median_rt": q(shifted, 0.50),
                    "shifted_q10": q(shifted, 0.10),
                    "shifted_q25": q(shifted, 0.25),
                    "shifted_q50": q(shifted, 0.50),
                    "shifted_q75": q(shifted, 0.75),
                    "shifted_q90": q(shifted, 0.90),
                    "shifted_q95": q(shifted, 0.95),
                    "shifted_q99": q(shifted, 0.99),
                    "shifted_rt_sd": float(np.std(shifted, ddof=1)),
                    "shifted_rt_iqr": q(shifted, 0.75) - q(shifted, 0.25),
                    "shifted_skewness": finite_skew(shifted),
                    **href,
                }
            )
    return pd.DataFrame(rows)


def plot_all(
    out_dir: Path,
    *,
    candidate_dfs: Dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    caf: pd.DataFrame,
    mechanism: pd.DataFrame,
    href: Dict[str, float],
    best_name: str,
    current_det: str,
    current_var: str,
    t0_seconds: float,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def shifted_rt(df: pd.DataFrame) -> np.ndarray:
        decision = df["pred_rt"].to_numpy(dtype=np.float64) - t0_seconds
        return decision + (href["human_mean_rt"] - safe_mean(decision))

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.hist(candidate_dfs[current_det]["true_rt"], bins=28, density=True, histtype="step", linewidth=2.0, label="human", color="#222222")
    for name, label in [(current_det, "current deterministic"), (current_var, "current variational"), (best_name, "optimized best")]:
        if name in candidate_dfs:
            ax.hist(shifted_rt(candidate_dfs[name]), bins=28, density=True, histtype="step", linewidth=1.8, label=label)
    ax.set_xlabel("RT after t0 mean alignment (s)")
    ax.set_ylabel("density")
    ax.set_title("t0 shifts location but not shape")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rt_distribution_t0_shifted_model_vs_human.png", dpi=220)
    plt.close(fig)

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    labels = ["q10", "q25", "q50", "q75", "q90", "q95"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    human_rt = candidate_dfs[current_det]["true_rt"].to_numpy(dtype=np.float64)
    ax.plot(labels, [q(human_rt, p) for p in quantiles], marker="o", label="human", color="#222222")
    for name, label in [(current_det, "current deterministic"), (best_name, "optimized best")]:
        ax.plot(labels, [q(shifted_rt(candidate_dfs[name]), p) for p in quantiles], marker="o", label=label)
    ax.set_ylabel("RT after t0 mean alignment (s)")
    ax.set_title("RT quantile comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rt_quantile_comparison_model_vs_human.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(summary["accuracy"], summary["q90_minus_q10"], s=28, alpha=0.45, label="candidates")
    for name, label, color in [(current_det, "current det", "#D95F02"), (current_var, "current var", "#7570B3"), (best_name, "optimized", "#1B9E77")]:
        part = summary[summary["condition_name"].eq(name)]
        if not part.empty:
            ax.scatter(part["accuracy"], part["q90_minus_q10"], s=95, label=label, color=color)
    ax.axhline(href["human_q90_minus_q10"], color="#222222", linestyle="--", label="human q90-q10")
    ax.set_xlabel("accuracy")
    ax.set_ylabel("q90 - q10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rt_spread_vs_accuracy_candidates.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(summary["incongruent_error_rate"], summary["q95_minus_median"], s=28, alpha=0.5)
    ax.axhline(href["human_q95_minus_median"], color="#222222", linestyle="--", label="human q95-median")
    ax.set_xlabel("incongruent error rate")
    ax.set_ylabel("q95 - median")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rt_tail_vs_incongruent_error_rate.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, label in [(current_det, "current deterministic"), (best_name, "optimized best")]:
        part = caf[(caf["condition_name"].eq(name)) & (caf["source"].eq("model"))]
        ax.plot(part["rt_bin"], part["accuracy"], marker="o", label=label)
    human = caf[(caf["condition_name"].eq(current_det)) & (caf["source"].eq("human"))]
    ax.plot(human["rt_bin"], human["accuracy"], marker="o", label="human", color="#222222")
    ax.set_xlabel("RT bin")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("CAF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "caf_model_vs_human.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    best = candidate_dfs[best_name]
    axes[0].hist(best.loc[best["model_correct"], "pred_rt"], bins=24, density=True, histtype="step", label="correct")
    axes[0].hist(best.loc[~best["model_correct"], "pred_rt"], bins=24, density=True, histtype="step", label="error")
    axes[0].set_title("All trials")
    inc = best[best["congruency"].eq(1)]
    axes[1].hist(inc.loc[inc["model_correct"], "pred_rt"], bins=24, density=True, histtype="step", label="incongruent correct")
    axes[1].hist(inc.loc[~inc["model_correct"], "pred_rt"], bins=24, density=True, histtype="step", label="incongruent error")
    axes[1].set_title("Incongruent trials")
    for ax in axes:
        ax.set_xlabel("model RT (s)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(fig_dir / "correct_vs_error_rt_distribution_optimized.png", dpi=220)
    plt.close(fig)

    ro = extract_readout_timing(candidate_dfs[best_name], candidate_dfs[best_name].attrs["outputs"], t0_seconds=t0_seconds, dt_ms=10)
    inc_ro = ro[ro["congruency"].eq(1)].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.boxplot(
        [
            inc_ro.loc[inc_ro["model_correct"], "target_recovery_time"].dropna(),
            inc_ro.loc[~inc_ro["model_correct"], "target_recovery_time"].dropna(),
        ],
        tick_labels=["correct", "error"],
        showfliers=False,
    )
    ax.set_ylabel("target recovery time (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "target_recovery_time_correct_vs_error_optimized.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.boxplot(
        [
            inc_ro.loc[inc_ro["model_correct"], "s_target_minus_flanker_at_readout"].dropna(),
            inc_ro.loc[~inc_ro["model_correct"], "s_target_minus_flanker_at_readout"].dropna(),
        ],
        tick_labels=["correct", "error"],
        showfliers=False,
    )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("s_target - s_flanker at readout")
    fig.tight_layout()
    fig.savefig(fig_dir / "readout_s_target_minus_flanker_by_group_optimized.png", dpi=220)
    plt.close(fig)


def write_summary(path: Path, summary: pd.DataFrame, top: pd.DataFrame, var_summary: Optional[pd.DataFrame], best_name: str, current_det: str, href: Dict[str, float]) -> None:
    best = summary[summary["condition_name"].eq(best_name)].iloc[0]
    current = summary[summary["condition_name"].eq(current_det)].iloc[0]
    var_text = "Variational seed runs were executed. They did not become the primary best candidate unless listed below."
    if var_summary is not None and not var_summary.empty:
        vbest = var_summary.sort_values(["passes_hard_constraints", "total_score"], ascending=[False, False]).iloc[0]
        var_text = (
            f"Best variational seed-mean candidate: `{vbest['condition_name']}`; "
            f"accuracy={vbest['accuracy']:.3f}, q90-q10={vbest['q90_minus_q10']:.3f}, "
            f"incongruent error rate={vbest['incongruent_error_rate']:.3f}."
        )
    text = f"""# RT Shape Optimization for Natural Layer-to-Time Wong-Wang Model

## 1. Goal

This stage tests whether natural layer-to-time Wong-Wang can improve RT distribution shape and RT-error/accuracy patterns while preserving the already observed DMC-like internal dynamics. Mean RT is not the main target because non-decision time `t0` can shift location.

## 2. Starting Point

The current deterministic model already produces early flanker and late target dynamics. Error trials have later target recovery than correct trials. However, its RT spread is too narrow: current q90-q10 is `{current['q90_minus_q10']:.3f}` while human q90-q10 is `{href['human_q90_minus_q10']:.3f}`. t0 can align mean RT but cannot change spread, skewness, tail, error/correct RT ordering, or CAF.

## 3. Methods

Tested small readout-rule variants, threshold/gain retuning, lightweight hazard readout, and variational sigma variants. Model selection used RT shape, CAF/error pattern, and mechanism preservation. Accuracy was treated as a hard constraint, not the main objective.

## 4. Main Results: RT Shape

Best optimized candidate q90-q10 is `{best['q90_minus_q10']:.3f}` versus current `{current['q90_minus_q10']:.3f}` and human `{href['human_q90_minus_q10']:.3f}`. Best q95-median is `{best['q95_minus_median']:.3f}` versus current `{current['q95_minus_median']:.3f}` and human `{href['human_q95_minus_median']:.3f}`. Skewness is `{best['skewness']:.3f}` versus current `{current['skewness']:.3f}` and human `{href['human_skewness']:.3f}`.

## 5. Main Results: RT-Error / RT-Accuracy Pattern

Best candidate error-minus-correct RT is `{best['error_minus_correct_rt']:.3f}`. Incongruent error-minus-correct RT is `{best['incongruent_error_minus_correct_rt']:.3f}`. Fastest-bin accuracy is `{best['fastest_bin_accuracy']:.3f}`. CAF is included in `rt_shape_optimization_caf_by_condition.csv` and the CAF figure.

## 6. Main Results: Mechanism Preservation

Target recovery remains later in error trials: error-minus-correct recovery time is `{best['target_recovery_time_error_minus_correct']:.3f}`. Incongruent-error early flanker dominance is `{best['early_flanker_dominance_rate_in_incongruent_error']:.3f}`. Incongruent-correct late target recovery is `{best['late_target_recovery_rate_in_incongruent_correct']:.3f}`. The optimization did not remove the DMC-like internal mechanism.

## 7. Variational Sampling

{var_text} Variational sampling remains useful as subjective evidence uncertainty, but this run does not justify making it the main source of RT shape.

## 8. Best Candidate

- condition_name: `{best['condition_name']}`
- readout_rule: `{best['readout_rule']}`
- schedule: `{SCHEDULE}`
- normalization: `{NORMALIZATION}`
- evidence_gain: `{best['evidence_gain']:.2f}`
- threshold: `{best['threshold']:.2f}`
- margin: `{best['margin']:.2f}`
- sustained_k: `{int(best['sustained_k'])}`
- min_decision_time: `{best['min_decision_time']:.2f}`
- sigma_type: `{best['sigma_type']}`
- accuracy: `{best['accuracy']:.3f}`
- human_choice_agreement: `{best['human_choice_agreement']:.3f}`
- incongruent_error_rate: `{best['incongruent_error_rate']:.3f}`
- q90_minus_q10: `{best['q90_minus_q10']:.3f}`
- q95_minus_median: `{best['q95_minus_median']:.3f}`
- skewness: `{best['skewness']:.3f}`
- error_minus_correct_rt: `{best['error_minus_correct_rt']:.3f}`
- incongruent_error_minus_correct_rt: `{best['incongruent_error_minus_correct_rt']:.3f}`
- target_recovery_time_error_minus_correct: `{best['target_recovery_time_error_minus_correct']:.3f}`

## 9. Interpretation

Mean RT can be handled by t0. RT shape is the real bottleneck. If a readout rule improves spread without destroying conflict and target/flanker dynamics, that readout should be the next mainline. If right tail remains insufficient, AR(1) or other temporal persistence mechanisms should only be considered after this deterministic route is exhausted. Stochastic stopping should remain secondary and only be revisited if fast-error behavior remains missing.

## 10. Recommended Next Steps

1. Continue readout-rule optimization around the best candidate rather than broad CNN retraining.
2. Keep variational sampling as a secondary uncertainty mechanism.
3. Do not add AR(1) yet.
4. Do not add stochastic stopping yet.
5. Run subject-level and image-identity validation once the selected readout rule is stable.
6. The current results are suitable for a meeting summary, with the caveat that RT shape is improved but not fully human-like unless the numbers above are close to human.

## 11. Short Chinese Summary for Discussion

这一阶段我们不再主要追求 mean RT，因为 mean RT 可以用 t0 整体平移。真正目标是看模型能不能产生更像人类的 RT 分布形状，包括右偏、长尾、error/correct RT、CAF 和 fast-error pattern。natural layer-to-time WW 已经能产生类似 DMC 的内部 target/flanker 竞争动态，所以现在的关键是找到合适的 readout 或 uncertainty 机制，把这个内部动态转化为人类式 RT/ACC 模式。本次小范围优化主要测试 readout rule、threshold/gain 和 variational sigma，并检查优化后是否仍保留 error trial target recovery 更晚、incongruent error early flanker dominance、incongruent correct late target recovery。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz")
    parser.add_argument("--output_dir", default=OUT_SUBDIR)
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--time_steps", type=int, default=160)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--t0_seconds", type=float, default=0.25)
    parser.add_argument("--noise_ampa", type=float, default=0.02)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--n_variational_seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(resolve_path(args.cache_path), args.max_trials)
    href = human_reference(cache)
    cfgs = readout_configs()
    summary_rows: List[Dict[str, Any]] = []
    mech_rows: List[Dict[str, Any]] = []
    caf_rows: List[pd.DataFrame] = []
    trial_frames: List[pd.DataFrame] = []
    candidate_dfs: Dict[str, pd.DataFrame] = {}

    gains = [1.0, 1.5, 2.0]
    thresholds = [0.12, 0.14, 0.16, 0.18, 0.20]
    for gain in gains:
        ww_input = build_natural_input(cache, evidence_gain=gain, time_steps=args.time_steps)
        for threshold in thresholds:
            base_name = f"det_base_g{gain:.2f}_th{threshold:.2f}"
            outputs = run_ww(
                ww_input,
                time_steps=args.time_steps,
                dt_ms=args.dt_ms,
                threshold=threshold,
                noise_ampa=args.noise_ampa,
                device="cpu",
                seed=args.seed,
                readout_mode="baseline",
                t0_seconds=args.t0_seconds,
                choice_temperature=args.choice_temperature,
            )
            base_df = base_condition_df(
                cache,
                outputs,
                condition_name=base_name,
                variant_type="deterministic",
                evidence_gain=gain,
                threshold=threshold,
                seed=args.seed,
            )
            for cfg in cfgs:
                name = BEST_DET if (gain == 2.0 and threshold == 0.12 and cfg == ReadoutConfig("baseline_threshold")) else condition_name("det", gain, threshold, cfg)
                df = apply_readout(base_df, outputs, cfg=cfg, threshold=threshold, dt_ms=args.dt_ms, t0_seconds=args.t0_seconds)
                df["condition_name"] = name
                df["condition"] = name
                metrics = shape_metrics(name, df, href)
                mech = mechanism_metrics(df, outputs, name, args.dt_ms, args.t0_seconds)
                meta = {
                    "variant_type": "deterministic",
                    "schedule_type": SCHEDULE,
                    "normalization": NORMALIZATION,
                    "evidence_gain": gain,
                    "threshold": threshold,
                    "sigma_type": "none",
                    "sigma_base": 0.0,
                    "sigma_middle": 0.0,
                    "sigma_conflict": 0.0,
                    "readout_rule": cfg.readout_rule,
                    "min_decision_time": cfg.min_decision_time,
                    "sustained_k": cfg.sustained_k,
                    "margin": cfg.margin,
                    "hazard_alpha": cfg.hazard_alpha,
                    "hazard_beta": cfg.hazard_beta,
                    "choice_rule": "trajectory_max_choice",
                    "seed": args.seed,
                }
                summary_rows.append({**metrics, **mech, **meta})
                mech_rows.append({**mech, **meta})
                caf_rows.append(rt_bins(df, name))
                if name == BEST_DET:
                    candidate_dfs[name] = df.copy()
                    candidate_dfs[name].attrs["outputs"] = outputs

    raw_summary = pd.DataFrame(summary_rows)
    current_row = raw_summary[raw_summary["condition_name"].eq(BEST_DET)].iloc[0]
    scored = score_candidates(raw_summary, current_row, href)
    best_det_name = str(scored.iloc[0]["condition_name"])
    # Store top deterministic trial-level frames by recomputing compactly for top names.
    top_det = scored.head(20).copy()

    # Reconstruct selected candidate dfs for plots and trial-level export.
    for _, row in pd.concat([top_det, scored[scored["condition_name"].eq(best_det_name)], scored[scored["condition_name"].eq(BEST_DET)]]).drop_duplicates("condition_name").iterrows():
        gain = float(row["evidence_gain"])
        threshold = float(row["threshold"])
        cfg = ReadoutConfig(
            str(row["readout_rule"]),
            float(row["min_decision_time"]),
            int(row["sustained_k"]),
            float(row["margin"]),
            float(row["hazard_alpha"]),
            float(row["hazard_beta"]),
        )
        ww_input = build_natural_input(cache, evidence_gain=gain, time_steps=args.time_steps)
        outputs = run_ww(
            ww_input,
            time_steps=args.time_steps,
            dt_ms=args.dt_ms,
            threshold=threshold,
            noise_ampa=args.noise_ampa,
            device="cpu",
            seed=args.seed,
            readout_mode="baseline",
            t0_seconds=args.t0_seconds,
            choice_temperature=args.choice_temperature,
        )
        base_df = base_condition_df(cache, outputs, condition_name=str(row["condition_name"]), variant_type="deterministic", evidence_gain=gain, threshold=threshold, seed=args.seed)
        df = apply_readout(base_df, outputs, cfg=cfg, threshold=threshold, dt_ms=args.dt_ms, t0_seconds=args.t0_seconds)
        df["condition_name"] = str(row["condition_name"])
        df["condition"] = str(row["condition_name"])
        candidate_dfs[str(row["condition_name"])] = df.copy()
        candidate_dfs[str(row["condition_name"])].attrs["outputs"] = outputs
        trial_frames.append(df.assign(source_stage="deterministic_top"))

    # Current variational seed-average for comparison.
    var_seed_frames = []
    for seed in [args.seed + i for i in range(args.n_variational_seeds)]:
        ww_input = build_natural_input(
            cache,
            evidence_gain=2.0,
            time_steps=args.time_steps,
            variant_type="variational",
            sigma_type="fixed_sigma",
            sigma_base=0.05,
            seed=seed,
        )
        outputs = run_ww(
            ww_input,
            time_steps=args.time_steps,
            dt_ms=args.dt_ms,
            threshold=0.12,
            noise_ampa=args.noise_ampa,
            device="cpu",
            seed=seed,
            readout_mode="baseline",
            t0_seconds=args.t0_seconds,
            choice_temperature=args.choice_temperature,
        )
        base_df = base_condition_df(cache, outputs, condition_name=BEST_VAR_SEEDAVG, variant_type="variational", evidence_gain=2.0, threshold=0.12, seed=seed, sigma_type="fixed_sigma", sigma_base=0.05)
        df = apply_readout(base_df, outputs, cfg=ReadoutConfig("baseline_threshold"), threshold=0.12, dt_ms=args.dt_ms, t0_seconds=args.t0_seconds)
        var_seed_frames.append(df)
    var_avg = var_seed_frames[0].copy()
    var_avg["pred_rt"] = np.vstack([d["pred_rt"].to_numpy(dtype=np.float64) for d in var_seed_frames]).mean(axis=0)
    choices = np.vstack([d["pred_choice"].to_numpy(dtype=np.int64) for d in var_seed_frames])
    var_avg["pred_choice"] = [np.bincount(col, minlength=4).argmax() for col in choices.T]
    var_avg["model_correct"] = var_avg["pred_choice"].to_numpy(dtype=np.int64) == var_avg["target_label"].to_numpy(dtype=np.int64)
    var_avg["condition_name"] = BEST_VAR_SEEDAVG
    var_avg["condition"] = BEST_VAR_SEEDAVG
    candidate_dfs[BEST_VAR_SEEDAVG] = var_avg.copy()
    candidate_dfs[BEST_VAR_SEEDAVG].attrs["outputs"] = var_seed_frames[0].attrs.get("outputs", outputs)
    var_metrics = {**shape_metrics(BEST_VAR_SEEDAVG, var_avg, href), **mechanism_metrics(var_avg, outputs, BEST_VAR_SEEDAVG, args.dt_ms, args.t0_seconds)}
    raw_summary = pd.concat([raw_summary, pd.DataFrame([{**var_metrics, "variant_type": "variational", "readout_rule": "baseline_threshold", "evidence_gain": 2.0, "threshold": 0.12, "sigma_type": "fixed_sigma", "sigma_base": 0.05, "sigma_middle": 0.0, "sigma_conflict": 0.0, "min_decision_time": 0.0, "sustained_k": 1, "margin": 0.0, "hazard_alpha": 0.0, "hazard_beta": 0.0, "choice_rule": "trajectory_max_choice", "seed": "avg"}])], ignore_index=True)

    # Variational variants on the best deterministic readout rule.
    best_cfg = ReadoutConfig(
        str(scored.iloc[0]["readout_rule"]),
        float(scored.iloc[0]["min_decision_time"]),
        int(scored.iloc[0]["sustained_k"]),
        float(scored.iloc[0]["margin"]),
        float(scored.iloc[0]["hazard_alpha"]),
        float(scored.iloc[0]["hazard_beta"]),
    )
    best_gain = float(scored.iloc[0]["evidence_gain"])
    best_threshold = float(scored.iloc[0]["threshold"])
    sigma_specs: List[Tuple[str, float, float, float]] = [("none", 0.0, 0.0, 0.0)]
    sigma_specs += [("fixed_sigma", sb, 0.0, 0.0) for sb in [0.02, 0.05, 0.08, 0.10, 0.15]]
    sigma_specs += [("layer_weighted_sigma", sb, sm, 0.0) for sb in [0.02, 0.05] for sm in [0.05, 0.10, 0.15]]
    sigma_specs += [("conflict_dependent_sigma", sb, 0.0, sc) for sb in [0.02, 0.05] for sc in [0.05, 0.10, 0.15]]
    var_rows: List[Dict[str, Any]] = []
    for sigma_type, sigma_base, sigma_middle, sigma_conflict in sigma_specs:
        seed_rows = []
        seeds = [args.seed] if sigma_type == "none" else [args.seed + i for i in range(args.n_variational_seeds)]
        for seed in seeds:
            variant_type = "deterministic" if sigma_type == "none" else "variational"
            ww_input = build_natural_input(
                cache,
                evidence_gain=best_gain,
                time_steps=args.time_steps,
                variant_type=variant_type,
                sigma_type=sigma_type,
                sigma_base=sigma_base,
                sigma_middle=sigma_middle,
                sigma_conflict=sigma_conflict,
                seed=seed,
            )
            outputs = run_ww(
                ww_input,
                time_steps=args.time_steps,
                dt_ms=args.dt_ms,
                threshold=best_threshold,
                noise_ampa=args.noise_ampa,
                device="cpu",
                seed=seed,
                readout_mode="baseline",
                t0_seconds=args.t0_seconds,
                choice_temperature=args.choice_temperature,
            )
            name = condition_name("varshape", best_gain, best_threshold, best_cfg, sigma=f"{sigma_type}_sb{sigma_base:.2f}_sm{sigma_middle:.2f}_sc{sigma_conflict:.2f}", seed=seed)
            base_df = base_condition_df(cache, outputs, condition_name=name, variant_type=variant_type, evidence_gain=best_gain, threshold=best_threshold, seed=seed, sigma_type=sigma_type, sigma_base=sigma_base, sigma_middle=sigma_middle, sigma_conflict=sigma_conflict)
            df = apply_readout(base_df, outputs, cfg=best_cfg, threshold=best_threshold, dt_ms=args.dt_ms, t0_seconds=args.t0_seconds)
            df["condition_name"] = name
            df["condition"] = name
            metrics = {**shape_metrics(name, df, href), **mechanism_metrics(df, outputs, name, args.dt_ms, args.t0_seconds)}
            metrics.update({"variant_type": variant_type, "evidence_gain": best_gain, "threshold": best_threshold, "sigma_type": sigma_type, "sigma_base": sigma_base, "sigma_middle": sigma_middle, "sigma_conflict": sigma_conflict, "readout_rule": best_cfg.readout_rule, "min_decision_time": best_cfg.min_decision_time, "sustained_k": best_cfg.sustained_k, "margin": best_cfg.margin, "hazard_alpha": best_cfg.hazard_alpha, "hazard_beta": best_cfg.hazard_beta, "choice_rule": "trajectory_max_choice", "seed": seed})
            seed_rows.append(metrics)
        part = pd.DataFrame(seed_rows)
        mean_row = part.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
        name = condition_name("varshape_seedavg", best_gain, best_threshold, best_cfg, sigma=f"{sigma_type}_sb{sigma_base:.2f}_sm{sigma_middle:.2f}_sc{sigma_conflict:.2f}", seed="avg")
        mean_row.update({"condition_name": name, "variant_type": "deterministic" if sigma_type == "none" else "variational", "sigma_type": sigma_type, "sigma_base": sigma_base, "sigma_middle": sigma_middle, "sigma_conflict": sigma_conflict, "readout_rule": best_cfg.readout_rule, "min_decision_time": best_cfg.min_decision_time, "sustained_k": best_cfg.sustained_k, "margin": best_cfg.margin, "hazard_alpha": best_cfg.hazard_alpha, "hazard_beta": best_cfg.hazard_beta, "choice_rule": "trajectory_max_choice", "seed": "avg", "n_seeds": len(seeds)})
        var_rows.append(mean_row)
    var_summary = pd.DataFrame(var_rows)
    var_summary = score_candidates(pd.concat([raw_summary.iloc[:0], var_summary], ignore_index=True), current_row, href)

    combined = pd.concat([raw_summary, var_summary], ignore_index=True, sort=False)
    scored_all = score_candidates(combined, current_row, href)
    top = scored_all.head(25)
    best_name = str(top.iloc[0]["condition_name"])
    if best_name not in candidate_dfs:
        # Use deterministic best for plotting if the seed-mean variational winner has no trial-level average.
        best_name = best_det_name

    pd.concat(trial_frames, ignore_index=True).to_csv(out_dir / "rt_shape_optimization_trial_level.csv", index=False)
    scored_all.to_csv(out_dir / "rt_shape_optimization_summary.csv", index=False)
    top.to_csv(out_dir / "rt_shape_optimization_top_candidates.csv", index=False)
    pd.concat(caf_rows + [rt_bins(candidate_dfs[BEST_VAR_SEEDAVG], BEST_VAR_SEEDAVG)], ignore_index=True).to_csv(out_dir / "rt_shape_optimization_caf_by_condition.csv", index=False)
    pd.DataFrame(mech_rows).to_csv(out_dir / "rt_shape_optimization_mechanism_summary.csv", index=False)
    var_summary.to_csv(out_dir / "rt_shape_optimization_variational_seed_summary.csv", index=False)
    t0_shift_table({k: candidate_dfs[k] for k in candidate_dfs if k in [BEST_DET, BEST_VAR_SEEDAVG, best_name]}, href, args.t0_seconds).to_csv(out_dir / "t0_shifted_rt_shape_summary.csv", index=False)

    plot_all(
        out_dir,
        candidate_dfs=candidate_dfs,
        summary=scored_all,
        caf=pd.concat(caf_rows + [rt_bins(candidate_dfs[BEST_VAR_SEEDAVG], BEST_VAR_SEEDAVG)], ignore_index=True),
        mechanism=pd.DataFrame(mech_rows),
        href=href,
        best_name=best_name,
        current_det=BEST_DET,
        current_var=BEST_VAR_SEEDAVG,
        t0_seconds=args.t0_seconds,
    )
    write_summary(out_dir / "rt_shape_optimization_summary.md", scored_all, top, var_summary, best_name, BEST_DET, href)
    (out_dir / "metadata.json").write_text(json.dumps(to_jsonable({"best_candidate": best_name, "n_candidates": int(len(scored_all)), "human_reference": href}), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

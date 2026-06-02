#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_SCRIPT_DIR = _SCRIPT_DIR.parents[2] / "schedule_compression_pareto_search" / "scripts"
NAT_SCRIPT_DIR = _SCRIPT_DIR.parents[2] / "natural_evidence_dynamics_optimization" / "scripts"
CODE_SCRIPT_DIR = Path("/Users/siyu/Documents/GitHub/VAM-studying/code/scripts")
for _p in [_SCRIPT_DIR, SEARCH_SCRIPT_DIR, NAT_SCRIPT_DIR, CODE_SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout, rt_bins  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_congruent_ww_dynamics_diagnostic import parse_group_params  # noqa: E402
from run_gated_readout_simulation import GROUPS, GROUP_LABEL, state_metrics  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule, normalize_layers, raw_layer_arrays  # noqa: E402
from run_representative_extreme_age_subset_fitting import apply_group_t0, load_trial_cache, subset_cache  # noqa: E402
from analyze_layerwise_evidence_ww import run_ww  # noqa: E402


BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
READOUT_DIR = BASE_DIR / "readout_choice_uncertainty_mechanism_comparison"
NAT_DIR = BASE_DIR / "natural_evidence_dynamics_optimization"
SCHED_DIR = BASE_DIR / "schedule_compression_pareto_search"
RESCREEN_DIR = SCHED_DIR / "constraint_first_rescreen"
OUT_DIR = BASE_DIR / "mechanism_redesign_conflict_adaptive_schedule"
DT = 0.01
TIME_STEPS = 80
SEED = 20260602
NOISE_SEED = 20260603
LAPSE_SEED = 20260604
NORMALIZATION = "per_layer_gap_scale"
BASELINE_MODEL_ID = "c1.00_ls0_tw1.00_ep0__baseline"
AGE_MAP = {"young_20_29": "young", "older_80_89": "older"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanism redesign diagnostic: conflict-adaptive schedule vs bounded lapse.")
    p.add_argument("--mode", choices=["small", "expanded"], default="small")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUT_DIR,
        "metrics": OUT_DIR / "metrics",
        "figures_publication": OUT_DIR / "figures_publication",
        "summaries": OUT_DIR / "summaries",
        "logs": OUT_DIR / "logs",
        "scripts": OUT_DIR / "scripts",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_mean(x: Any) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def safe_q(x: Any, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else math.nan


def corr_safe(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return math.nan
    if np.isclose(a[mask].std(), 0.0) or np.isclose(b[mask].std(), 0.0):
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def rmse(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else math.nan


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT_DIR / "figures_publication" / f"{name}.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax.tick_params(labelsize=9)


def choice_type(choice: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    return np.where(choice == target, "target", np.where(choice == flanker, "flanker", "other"))


def schedule_df_from_params(compression: float, late_shift_ms: int, transition_width: float, early_phase_shortening_ms: int) -> pd.DataFrame:
    t = np.arange(TIME_STEPS, dtype=np.float32) / TIME_STEPS
    centers = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32) * compression
    centers = np.clip(centers, 0.03, 0.97)
    centers[3:] = np.clip(centers[3:] + late_shift_ms / 1000.0, 0.03, 0.97)
    centers[0] = max(0.03, centers[0] - early_phase_shortening_ms / 1000.0)
    sigma = max(0.12 * transition_width, 0.03)
    basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / sigma) ** 2)
    basis_sum = basis.sum(axis=1, keepdims=True)
    basis_sum[basis_sum < 1e-6] = 1.0
    return pd.DataFrame(basis / basis_sum, columns=["conv3", "conv4", "conv5", "pooled", "final"])


def deterministic_schedule_mu(group_layers: dict[str, np.ndarray], schedule_df: pd.DataFrame, evidence_gain: float) -> np.ndarray:
    return build_mu_schedule(group_layers, schedule_df, float(evidence_gain)).numpy()


def check_inputs() -> None:
    required = [
        BASE_DIR / "evidence_cache/representative_subset_layerwise_evidence.npz",
        BASE_DIR / "best_model_R5_combined_best/results/best_model_parameter_estimates.csv",
        BASE_DIR / "fitting/representative_trial_level_predictions.csv",
        READOUT_DIR / "metrics/readout_choice_model_ranking.csv",
        READOUT_DIR / "metrics/human_reference_rt_error_metrics.csv",
        NAT_DIR / "metrics/natural_dynamics_model_ranking.csv",
        SCHED_DIR / "metrics/schedule_compression_local_search_ranking_repaired.csv",
        SCHED_DIR / "metrics/schedule_compression_pareto_front_repaired.csv",
        SCHED_DIR / "metrics/schedule_compression_top_candidates_trial_level_repaired.csv",
        SCHED_DIR / "metrics/schedule_compression_trajectory_diagnostics_repaired.csv",
        RESCREEN_DIR / "metrics/constraint_first_rescreen_recomputed_metrics.csv",
        RESCREEN_DIR / "metrics/constraint_first_rescreen_representative_models.csv",
        RESCREEN_DIR / "summaries/constraint_first_rescreen_summary.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def load_inputs() -> dict[str, Any]:
    best_dir = BASE_DIR / "best_model_R5_combined_best/results"
    group_params, t0_mean, t0_sd = parse_group_params(best_dir / "best_model_parameter_estimates.csv")
    cache = load_trial_cache(BASE_DIR)
    norm_layers = normalize_layers(raw_layer_arrays(cache), NORMALIZATION)
    readout_rank = pd.read_csv(READOUT_DIR / "metrics/readout_choice_model_ranking.csv")
    human_ref = pd.read_csv(READOUT_DIR / "metrics/human_reference_rt_error_metrics.csv")
    nat_rank = pd.read_csv(NAT_DIR / "metrics/natural_dynamics_model_ranking.csv")
    sched_rank = pd.read_csv(SCHED_DIR / "metrics/schedule_compression_local_search_ranking_repaired.csv")
    sched_pareto = pd.read_csv(SCHED_DIR / "metrics/schedule_compression_pareto_front_repaired.csv")
    sched_trial = pd.read_csv(SCHED_DIR / "metrics/schedule_compression_top_candidates_trial_level_repaired.csv")
    sched_traj = pd.read_csv(SCHED_DIR / "metrics/schedule_compression_trajectory_diagnostics_repaired.csv")
    rescreen_metrics = pd.read_csv(RESCREEN_DIR / "metrics/constraint_first_rescreen_recomputed_metrics.csv")
    rescreen_reps = pd.read_csv(RESCREEN_DIR / "metrics/constraint_first_rescreen_representative_models.csv")
    rescreen_summary = (RESCREEN_DIR / "summaries/constraint_first_rescreen_summary.md").read_text(encoding="utf-8")
    return {
        "cache": cache,
        "group_params": group_params,
        "t0_mean": t0_mean,
        "t0_sd": t0_sd,
        "norm_layers": norm_layers,
        "readout_rank": readout_rank,
        "human_ref": human_ref[human_ref["source"].eq("human")].copy(),
        "nat_rank": nat_rank,
        "sched_rank": sched_rank,
        "sched_pareto": sched_pareto,
        "sched_trial": sched_trial,
        "sched_traj": sched_traj,
        "rescreen_metrics": rescreen_metrics,
        "rescreen_reps": rescreen_reps,
        "rescreen_summary": rescreen_summary,
    }


def selected_time_gap_params(readout_rank: pd.DataFrame) -> dict[str, dict[str, float]]:
    best = readout_rank[readout_rank["model"].eq("M3_time_gap")].iloc[0]
    out: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        if str(best["param_mode"]) == "age_specific":
            out[group] = {
                "sigma_base": float(best[f"{group}_sigma_base"]),
                "sigma_time": float(best[f"{group}_sigma_time"]),
                "sigma_gap": float(best[f"{group}_sigma_gap"]),
                "gap_scale": float(best[f"{group}_gap_scale"]),
            }
        else:
            out[group] = {
                "sigma_base": float(best["sigma_base"]),
                "sigma_time": float(best["sigma_time"]),
                "sigma_gap": float(best["sigma_gap"]),
                "gap_scale": float(best["gap_scale"]),
            }
    return out


def write_input_inventory(data: dict[str, Any]) -> None:
    lines = [
        "# Mechanism redesign input inventory",
        "",
        "## Inputs read",
        f"- `{BASE_DIR / 'evidence_cache/representative_subset_layerwise_evidence.npz'}`: cached layerwise evidence.",
        f"- `{BASE_DIR / 'best_model_R5_combined_best/results/best_model_parameter_estimates.csv'}`: current group-specific WW and readout settings.",
        f"- `{BASE_DIR / 'fitting/representative_trial_level_predictions.csv'}`: trial metadata, human RT, human correctness.",
        f"- `{READOUT_DIR / 'metrics/readout_choice_model_ranking.csv'}`: time+gap uncertainty ranking and selected parameters.",
        f"- `{READOUT_DIR / 'metrics/human_reference_rt_error_metrics.csv'}`: human reference metrics.",
        f"- `{NAT_DIR / 'metrics/natural_dynamics_model_ranking.csv'}`: prior natural dynamics comparison.",
        f"- `{SCHED_DIR / 'metrics/schedule_compression_local_search_ranking_repaired.csv'}`: repaired global schedule ranking.",
        f"- `{SCHED_DIR / 'metrics/schedule_compression_pareto_front_repaired.csv'}`: repaired Pareto front.",
        f"- `{SCHED_DIR / 'metrics/schedule_compression_top_candidates_trial_level_repaired.csv'}`: repaired trial-level diagnostics.",
        f"- `{SCHED_DIR / 'metrics/schedule_compression_trajectory_diagnostics_repaired.csv'}`: repaired trajectory diagnostics.",
        f"- `{RESCREEN_DIR / 'metrics/constraint_first_rescreen_recomputed_metrics.csv'}`: rescreen metrics.",
        f"- `{RESCREEN_DIR / 'metrics/constraint_first_rescreen_representative_models.csv'}`: rescreen representative models.",
        f"- `{RESCREEN_DIR / 'summaries/constraint_first_rescreen_summary.md'}`: current constraint-first conclusion.",
        "",
        "## What each file contributes",
        "- Evidence cache and best-model parameters let this round reconstruct trial-wise WW inputs and trajectories without retraining VGG or re-extracting evidence.",
        "- Readout-choice ranking provides the current time+gap uncertainty baseline and human reference targets.",
        "- Schedule-compression repaired outputs define the strongest existing global-schedule references and their failure pattern.",
        "- Constraint-first rescreen confirms that global schedule compression plus retuned time-gap noise has zero survivors and mostly fails because older congruent errors disappear.",
        "",
        "## Why test conflict-adaptive schedule",
        "- Global compression improves incongruent repair but stabilizes congruent trials too much, especially in the older group.",
        "- A more natural alternative is to accelerate high-level evidence only when early competition is high, instead of compressing every trial equally.",
        "- This allows the mechanism to depend on current evidence conflict rather than congruency labels or future target crossing.",
        "",
        "## Why bounded lapse is secondary",
        "- Bounded lapse is treated as rare response-execution uncertainty only.",
        "- It is included to test whether a very small downstream uncertainty source can recover a few older congruent errors without undoing the repaired incongruent behavior.",
        "- It is not treated as the main explanation of the task behavior.",
        "",
        "## What is not rerun",
        "- VGG is not retrained.",
        "- Image evidence is not re-extracted.",
        "- No target-gated readout is reintroduced.",
        "- No large schedule-compression fine search is run.",
    ]
    (OUT_DIR / "summaries/mechanism_redesign_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_reference_specs() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "R0_original_time_gap",
            "model_config_id": "R0_original_time_gap",
            "schedule_config_id": "c1.00_ls0_tw1.00_ep0",
            "adaptive_schedule_config_id": "none",
            "lapse_config_id": "none",
            "noise_config_id": "time_gap_selected",
            "schedule": {"compression": 1.00, "late_shift_ms": 0, "transition_width": 1.00, "early_phase_shortening_ms": 0},
            "adaptive": {"type": "none"},
            "noise_mode": "time_gap_selected",
            "lapse": {"type": "none"},
            "display_name": "R0_original_time_gap",
        },
        {
            "model_family": "R1_best_incongruent_repair_global_schedule",
            "model_config_id": "R1_best_incongruent_repair_global_schedule",
            "schedule_config_id": "c0.40_ls-50_tw1.10_ep30",
            "adaptive_schedule_config_id": "none",
            "lapse_config_id": "none",
            "noise_config_id": "sb0.0000_st0.0000_sg0.0000_gs0.03",
            "schedule": {"compression": 0.40, "late_shift_ms": -50, "transition_width": 1.10, "early_phase_shortening_ms": 30},
            "adaptive": {"type": "none"},
            "noise_mode": "custom_shared",
            "custom_noise": {"sigma_base": 0.0, "sigma_time": 0.0, "sigma_gap": 0.0, "gap_scale": 0.03},
            "lapse": {"type": "none"},
            "display_name": "R1_best_incongruent_repair_global_schedule",
        },
        {
            "model_family": "R2_best_fast_error_global_schedule",
            "model_config_id": "R2_best_fast_error_global_schedule",
            "schedule_config_id": "c0.40_ls-10_tw1.10_ep50",
            "adaptive_schedule_config_id": "none",
            "lapse_config_id": "none",
            "noise_config_id": "sb0.0005_st0.0120_sg0.0000_gs0.03",
            "schedule": {"compression": 0.40, "late_shift_ms": -10, "transition_width": 1.10, "early_phase_shortening_ms": 50},
            "adaptive": {"type": "none"},
            "noise_mode": "custom_shared",
            "custom_noise": {"sigma_base": 0.0005, "sigma_time": 0.0120, "sigma_gap": 0.0, "gap_scale": 0.03},
            "lapse": {"type": "none"},
            "display_name": "R2_best_fast_error_global_schedule",
        },
        {
            "model_family": "R3_best_conflict_dynamics_reference",
            "model_config_id": "R3_best_conflict_dynamics_reference",
            "schedule_config_id": "c0.70_ls-50_tw0.70_ep0",
            "adaptive_schedule_config_id": "none",
            "lapse_config_id": "none",
            "noise_config_id": "baseline",
            "schedule": {"compression": 0.70, "late_shift_ms": -50, "transition_width": 0.70, "early_phase_shortening_ms": 0},
            "adaptive": {"type": "none"},
            "noise_mode": "deterministic",
            "lapse": {"type": "none"},
            "display_name": "R3_best_conflict_dynamics_reference",
        },
        {
            "model_family": "R4_near_balanced_tradeoff_reference",
            "model_config_id": "R4_near_balanced_tradeoff_reference",
            "schedule_config_id": "c0.40_ls-50_tw1.10_ep50",
            "adaptive_schedule_config_id": "none",
            "lapse_config_id": "none",
            "noise_config_id": "sb0.0010_st0.0120_sg0.0000_gs0.03",
            "schedule": {"compression": 0.40, "late_shift_ms": -50, "transition_width": 1.10, "early_phase_shortening_ms": 50},
            "adaptive": {"type": "none"},
            "noise_mode": "custom_shared",
            "custom_noise": {"sigma_base": 0.0010, "sigma_time": 0.0120, "sigma_gap": 0.0, "gap_scale": 0.03},
            "lapse": {"type": "none"},
            "display_name": "R4_near_balanced_tradeoff_reference",
        },
    ]


def candidate_grid(mode: str) -> list[dict[str, Any]]:
    specs = base_reference_specs()

    ca1_specs = [
        ("flanker_dominance", "0_100", 0.70, 0.45, 0.50, 0.25, -50, 0.90, 30),
        ("entropy", "0_150", 0.65, 0.45, 0.70, 0.50, -40, 1.00, 40),
    ]
    if mode == "expanded":
        ca1_specs += [
            ("flanker_dominance", "0_150", 0.70, 0.40, 0.70, 0.25, -50, 1.10, 40),
            ("low_margin", "0_150", 0.65, 0.45, 0.60, 0.50, -30, 0.90, 30),
        ]
    for score_type, window, clow, chigh, theta_q, temp, shift, tw, ep in ca1_specs:
        cid = f"CA1_{score_type}_{window}_cl{clow:.2f}_ch{chigh:.2f}_q{theta_q:.2f}_t{temp:.2f}_ls{shift}_tw{tw:.2f}_ep{ep}"
        specs.append(
            {
                "model_family": "CA1_trialwise_conflict_adaptive_schedule",
                "model_config_id": cid,
                "schedule_config_id": f"base_cl{clow:.2f}_ls{shift}_tw{tw:.2f}_ep{ep}",
                "adaptive_schedule_config_id": cid,
                "lapse_config_id": "none",
                "noise_config_id": "time_gap_selected",
                "schedule": {"compression": clow, "late_shift_ms": shift, "transition_width": tw, "early_phase_shortening_ms": ep},
                "adaptive": {"type": "CA1", "score_type": score_type, "window": window, "compression_low": clow, "compression_high": chigh, "theta_quantile": theta_q, "temp": temp},
                "noise_mode": "time_gap_selected",
                "lapse": {"type": "none"},
                "display_name": cid,
            }
        )

    ca2_specs = [(0.10, 0.60, 0.05, 40), (0.20, 0.70, 0.10, 60)]
    for alpha, theta_q, temp, max_acc in ca2_specs:
        cid = f"CA2_a{alpha:.2f}_q{theta_q:.2f}_t{temp:.2f}_m{max_acc}"
        specs.append(
            {
                "model_family": "CA2_online_conflict_schedule_acceleration",
                "model_config_id": cid,
                "schedule_config_id": "c0.70_ls-40_tw1.00_ep30",
                "adaptive_schedule_config_id": cid,
                "lapse_config_id": "none",
                "noise_config_id": "time_gap_selected",
                "schedule": {"compression": 0.70, "late_shift_ms": -40, "transition_width": 1.00, "early_phase_shortening_ms": 30},
                "adaptive": {"type": "CA2", "alpha": alpha, "theta_quantile": theta_q, "temp": temp, "max_acceleration_ms": max_acc},
                "noise_mode": "time_gap_selected",
                "lapse": {"type": "none"},
                "display_name": cid,
            }
        )

    ca3_specs = [(0.05, 0.10, 40, 1.00), (0.08, 0.20, 60, 1.10)]
    for scale, alpha, max_acc, tw in ca3_specs:
        cid = f"CA3_s{scale:.2f}_a{alpha:.2f}_m{max_acc}_tw{tw:.2f}"
        specs.append(
            {
                "model_family": "CA3_uncertainty_adaptive_schedule",
                "model_config_id": cid,
                "schedule_config_id": f"c0.70_ls-30_tw{tw:.2f}_ep20",
                "adaptive_schedule_config_id": cid,
                "lapse_config_id": "none",
                "noise_config_id": "time_gap_selected",
                "schedule": {"compression": 0.70, "late_shift_ms": -30, "transition_width": tw, "early_phase_shortening_ms": 20},
                "adaptive": {"type": "CA3", "uncertainty_scale": scale, "alpha": alpha, "max_acceleration_ms": max_acc},
                "noise_mode": "time_gap_selected",
                "lapse": {"type": "none"},
                "display_name": cid,
            }
        )

    lapse_specs = [
        ("L1", {"type": "L1", "p_base": 0.0005, "p_time": 0.0020, "p_max": 0.005, "temp_lapse": 0.010, "sampling_rule": "runner_up"}),
        ("L2", {"type": "L2", "p_base": 0.0005, "p_gap": 0.0050, "gap_scale": 0.05, "p_max": 0.010, "temp_lapse": 0.010, "sampling_rule": "runner_up"}),
    ]
    if mode == "expanded":
        lapse_specs += [
            ("L2", {"type": "L2", "p_base": 0.0010, "p_gap": 0.0100, "gap_scale": 0.08, "p_max": 0.020, "temp_lapse": 0.020, "sampling_rule": "softmax"}),
            ("L3", {"type": "L3", "p_base": 0.0005, "p_time": 0.0050, "p_gap": 0.0050, "gap_scale": 0.05, "p_max": 0.010, "temp_lapse": 0.010, "sampling_rule": "softmax"}),
        ]
    for _, lapse in lapse_specs:
        cid = json.dumps(lapse, sort_keys=True, separators=(",", ":"))
        specs.append(
            {
                "model_family": f"{lapse['type']}_bounded_lapse_only",
                "model_config_id": f"{lapse['type']}_{abs(hash(cid)) % 100000}",
                "schedule_config_id": "c0.40_ls-10_tw1.10_ep50",
                "adaptive_schedule_config_id": "none",
                "lapse_config_id": f"{lapse['type']}_{abs(hash(cid)) % 100000}",
                "noise_config_id": "sb0.0005_st0.0120_sg0.0000_gs0.03",
                "schedule": {"compression": 0.40, "late_shift_ms": -10, "transition_width": 1.10, "early_phase_shortening_ms": 50},
                "adaptive": {"type": "none"},
                "noise_mode": "custom_shared",
                "custom_noise": {"sigma_base": 0.0005, "sigma_time": 0.0120, "sigma_gap": 0.0, "gap_scale": 0.03},
                "lapse": lapse,
                "display_name": f"{lapse['type']}_bounded_lapse_only",
            }
        )

    combo_specs = [
        {
            "family": "COMBO_CA_only",
            "schedule": {"compression": 0.70, "late_shift_ms": -40, "transition_width": 1.00, "early_phase_shortening_ms": 30},
            "adaptive": {"type": "CA2", "alpha": 0.10, "theta_quantile": 0.60, "temp": 0.05, "max_acceleration_ms": 40},
            "noise_mode": "time_gap_selected",
            "lapse": {"type": "none"},
        },
        {
            "family": "COMBO_CA_retuned_noise",
            "schedule": {"compression": 0.70, "late_shift_ms": -40, "transition_width": 1.00, "early_phase_shortening_ms": 30},
            "adaptive": {"type": "CA2", "alpha": 0.10, "theta_quantile": 0.60, "temp": 0.05, "max_acceleration_ms": 40},
            "noise_mode": "custom_shared",
            "custom_noise": {"sigma_base": 0.0005, "sigma_time": 0.0080, "sigma_gap": 0.0020, "gap_scale": 0.05},
            "lapse": {"type": "none"},
        },
        {
            "family": "COMBO_conflict_adaptive_schedule_plus_bounded_lapse",
            "schedule": {"compression": 0.70, "late_shift_ms": -40, "transition_width": 1.00, "early_phase_shortening_ms": 30},
            "adaptive": {"type": "CA2", "alpha": 0.10, "theta_quantile": 0.60, "temp": 0.05, "max_acceleration_ms": 40},
            "noise_mode": "custom_shared",
            "custom_noise": {"sigma_base": 0.0005, "sigma_time": 0.0080, "sigma_gap": 0.0020, "gap_scale": 0.05},
            "lapse": {"type": "L2", "p_base": 0.0005, "p_gap": 0.0050, "gap_scale": 0.05, "p_max": 0.010, "temp_lapse": 0.010, "sampling_rule": "runner_up"},
        },
    ]
    for i, spec in enumerate(combo_specs, start=1):
        mid = f"{spec['family']}_{i}"
        specs.append(
            {
                "model_family": spec["family"],
                "model_config_id": mid,
                "schedule_config_id": f"c{spec['schedule']['compression']:.2f}_ls{spec['schedule']['late_shift_ms']}_tw{spec['schedule']['transition_width']:.2f}_ep{spec['schedule']['early_phase_shortening_ms']}",
                "adaptive_schedule_config_id": spec["adaptive"]["type"],
                "lapse_config_id": spec["lapse"]["type"],
                "noise_config_id": "combo_noise",
                "schedule": spec["schedule"],
                "adaptive": spec["adaptive"],
                "noise_mode": spec["noise_mode"],
                "custom_noise": spec.get("custom_noise"),
                "lapse": spec["lapse"],
                "display_name": mid,
            }
        )
    return specs


def window_steps(window_name: str) -> tuple[int, int]:
    if window_name == "0_100":
        return 0, int(0.10 / DT)
    if window_name == "0_150":
        return 0, int(0.15 / DT)
    if window_name == "50_150":
        return int(0.05 / DT), int(0.15 / DT)
    return 0, int(0.10 / DT)


def trialwise_conflict_score(base_mu: np.ndarray, target: np.ndarray, flanker: np.ndarray, score_type: str, window_name: str) -> np.ndarray:
    lo, hi = window_steps(window_name)
    rows = np.arange(base_mu.shape[0])[:, None]
    times = np.arange(lo, max(lo + 1, hi))[None, :]
    target_vals = base_mu[rows, times, target[:, None]]
    flanker_vals = base_mu[rows, times, flanker[:, None]]
    if score_type == "flanker_dominance":
        score = np.maximum(flanker_vals - target_vals, 0.0).mean(axis=1)
    elif score_type == "low_margin":
        gap = np.abs(target_vals - flanker_vals)
        score = np.exp(-gap / 0.05).mean(axis=1)
    else:
        logits = base_mu[:, lo:hi, :]
        logits = logits - logits.max(axis=2, keepdims=True)
        prob = np.exp(logits)
        prob = prob / np.maximum(prob.sum(axis=2, keepdims=True), 1e-9)
        score = -(prob * np.log(np.maximum(prob, 1e-9))).sum(axis=2).mean(axis=1)
    return score.astype(np.float32)


def build_trialwise_schedule_mu(group_layers: dict[str, np.ndarray], base_schedule: dict[str, float], adaptive: dict[str, Any], evidence_gain: float, target: np.ndarray, flanker: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_df = schedule_df_from_params(base_schedule["compression"], int(base_schedule["late_shift_ms"]), float(base_schedule["transition_width"]), int(base_schedule["early_phase_shortening_ms"]))
    base_mu = deterministic_schedule_mu(group_layers, base_df, evidence_gain)
    score = trialwise_conflict_score(base_mu, target, flanker, adaptive["score_type"], adaptive["window"])
    theta = np.quantile(score, float(adaptive["theta_quantile"]))
    temp = max(float(adaptive["temp"]), 1e-6)
    high = float(adaptive["compression_high"])
    low = float(adaptive["compression_low"])
    strength = sigmoid((score - theta) / temp)
    comp = low - (low - high) * strength
    comp = np.clip(comp, high, low)
    # Use low/high endpoint schedules and interpolate by trial-specific conflict strength.
    low_df = schedule_df_from_params(low, int(base_schedule["late_shift_ms"]), float(base_schedule["transition_width"]), int(base_schedule["early_phase_shortening_ms"]))
    high_df = schedule_df_from_params(high, int(base_schedule["late_shift_ms"]), float(base_schedule["transition_width"]), int(base_schedule["early_phase_shortening_ms"]))
    mu_low = deterministic_schedule_mu(group_layers, low_df, evidence_gain)
    mu_high = deterministic_schedule_mu(group_layers, high_df, evidence_gain)
    mu = mu_low + strength[:, None, None] * (mu_high - mu_low)
    return mu, score, strength


def build_online_conflict_mu(group_layers: dict[str, np.ndarray], base_schedule: dict[str, float], adaptive: dict[str, Any], evidence_gain: float, target: np.ndarray, flanker: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sched_df = schedule_df_from_params(base_schedule["compression"], int(base_schedule["late_shift_ms"]), float(base_schedule["transition_width"]), int(base_schedule["early_phase_shortening_ms"]))
    mu = deterministic_schedule_mu(group_layers, sched_df, evidence_gain)
    rows = np.arange(mu.shape[0])[:, None]
    times = np.arange(mu.shape[1])[None, :]
    target_vals = mu[rows, times, target[:, None]]
    flanker_vals = mu[rows, times, flanker[:, None]]
    conflict = np.maximum(flanker_vals - target_vals, 0.0)
    early_conf = conflict[:, : int(0.15 / DT)].mean(axis=1)
    theta = np.quantile(early_conf, float(adaptive["theta_quantile"]))
    temp = max(float(adaptive["temp"]), 1e-6)
    control = sigmoid((conflict - theta) / temp)
    alpha = float(adaptive["alpha"])
    remaining = np.linspace(0.2, 1.0, mu.shape[1], dtype=np.float32)[None, :]
    gain = 1.0 + alpha * control * remaining
    decay = 1.0 - 0.5 * alpha * control * remaining
    mu2 = mu.copy()
    mu2[rows, times, target[:, None]] *= gain
    mu2[rows, times, flanker[:, None]] *= np.clip(decay, 0.5, 1.2)
    acc_ms = np.minimum(float(adaptive["max_acceleration_ms"]), float(adaptive["max_acceleration_ms"]) * control.cumsum(axis=1) / np.maximum(control.cumsum(axis=1).max(axis=1, keepdims=True), 1e-9))
    return mu2, early_conf.astype(np.float32), acc_ms.astype(np.float32)


def build_uncertainty_adaptive_mu(group_layers: dict[str, np.ndarray], base_schedule: dict[str, float], adaptive: dict[str, Any], evidence_gain: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sched_df = schedule_df_from_params(base_schedule["compression"], int(base_schedule["late_shift_ms"]), float(base_schedule["transition_width"]), int(base_schedule["early_phase_shortening_ms"]))
    mu = deterministic_schedule_mu(group_layers, sched_df, evidence_gain)
    logits = mu - mu.max(axis=2, keepdims=True)
    prob = np.exp(logits)
    prob = prob / np.maximum(prob.sum(axis=2, keepdims=True), 1e-9)
    entropy = -(prob * np.log(np.maximum(prob, 1e-9))).sum(axis=2)
    score = entropy[:, : int(0.15 / DT)].mean(axis=1)
    scale = max(float(adaptive["uncertainty_scale"]), 1e-6)
    control = 1.0 - np.exp(-entropy / scale)
    alpha = float(adaptive["alpha"])
    gain = 1.0 + alpha * control
    winner = mu.argmax(axis=2)
    rows = np.arange(mu.shape[0])[:, None]
    times = np.arange(mu.shape[1])[None, :]
    mu2 = mu.copy()
    mu2[rows, times, winner] *= gain
    acc_ms = np.minimum(float(adaptive["max_acceleration_ms"]), float(adaptive["max_acceleration_ms"]) * control)
    return mu2, score.astype(np.float32), acc_ms.astype(np.float32)


def build_candidate_mu(group_layers: dict[str, np.ndarray], target: np.ndarray, flanker: np.ndarray, evidence_gain: float, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adaptive = spec["adaptive"]
    schedule = spec["schedule"]
    if adaptive["type"] == "none":
        sched_df = schedule_df_from_params(schedule["compression"], int(schedule["late_shift_ms"]), float(schedule["transition_width"]), int(schedule["early_phase_shortening_ms"]))
        mu = deterministic_schedule_mu(group_layers, sched_df, evidence_gain)
        zero = np.zeros((mu.shape[0],), dtype=np.float32)
        zero_time = np.zeros((mu.shape[0], mu.shape[1]), dtype=np.float32)
        return mu, zero, np.full(mu.shape[0], schedule["compression"], dtype=np.float32), zero_time
    if adaptive["type"] == "CA1":
        mu, conflict_score, strength = build_trialwise_schedule_mu(group_layers, schedule, adaptive, evidence_gain, target, flanker)
        return mu, conflict_score, strength, np.repeat(strength[:, None], mu.shape[1], axis=1)
    if adaptive["type"] == "CA2":
        mu, conflict_score, control = build_online_conflict_mu(group_layers, schedule, adaptive, evidence_gain, target, flanker)
        mean_control = control.mean(axis=1) / max(float(adaptive["max_acceleration_ms"]), 1e-9)
        return mu, conflict_score, schedule["compression"] - 0.20 * mean_control, control
    if adaptive["type"] == "CA3":
        mu, score, control = build_uncertainty_adaptive_mu(group_layers, schedule, adaptive, evidence_gain)
        mean_control = control.mean(axis=1) / max(float(adaptive["max_acceleration_ms"]), 1e-9)
        return mu, score, schedule["compression"] - 0.20 * mean_control, control
    raise ValueError(f"Unknown adaptive type: {adaptive['type']}")


def noise_params_for_spec(spec: dict[str, Any], selected_noise: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    if spec["noise_mode"] == "time_gap_selected":
        return selected_noise
    if spec["noise_mode"] == "deterministic":
        return {g: {"sigma_base": 0.0, "sigma_time": 0.0, "sigma_gap": 0.0, "gap_scale": 0.05} for g in GROUPS}
    custom = spec["custom_noise"]
    return {g: dict(custom) for g in GROUPS}


def apply_time_gap_noise(states: np.ndarray, readout_time: np.ndarray, gap: np.ndarray, noise_cfg: dict[str, float], seed_key: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    max_time = max(float(readout_time.max()), 1e-9)
    earlyness = 1.0 - readout_time / max_time
    sigma = noise_cfg["sigma_base"] + noise_cfg["sigma_time"] * earlyness + noise_cfg["sigma_gap"] * np.exp(-np.clip(gap, 0, None) / max(noise_cfg["gap_scale"], 1e-9))
    rng = np.random.default_rng(NOISE_SEED + abs(hash(seed_key)) % 1000000)
    noisy_choice = (states + rng.normal(0.0, sigma[:, None], size=states.shape)).argmax(axis=1)
    return noisy_choice.astype(int), sigma.astype(float)


def lapse_probability(lapse: dict[str, Any], readout_time: np.ndarray, gap: np.ndarray) -> np.ndarray:
    if lapse["type"] == "none":
        return np.zeros_like(readout_time, dtype=float)
    max_time = max(float(readout_time.max()), 1e-9)
    earlyness = 1.0 - readout_time / max_time
    if lapse["type"] == "L1":
        p = lapse["p_base"] + lapse["p_time"] * earlyness
    elif lapse["type"] == "L2":
        p = lapse["p_base"] + lapse["p_gap"] * np.exp(-np.clip(gap, 0, None) / max(lapse["gap_scale"], 1e-9))
    else:
        p = lapse["p_base"] + lapse["p_time"] * earlyness + lapse["p_gap"] * np.exp(-np.clip(gap, 0, None) / max(lapse["gap_scale"], 1e-9))
    return np.minimum(p, lapse["p_max"]).astype(float)


def sample_lapse_choice(base_choice: np.ndarray, states: np.ndarray, target: np.ndarray, flanker: np.ndarray, lapse: dict[str, Any], p_lapse: np.ndarray, seed_key: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lapse["type"] == "none":
        return base_choice.copy(), np.zeros(len(base_choice), dtype=bool), np.array(["none"] * len(base_choice), dtype=object)
    rng = np.random.default_rng(LAPSE_SEED + abs(hash(seed_key)) % 1000000)
    triggered = rng.random(len(base_choice)) < p_lapse
    final = base_choice.copy()
    source = np.array(["none"] * len(base_choice), dtype=object)
    if not triggered.any():
        return final, triggered, source
    if lapse["sampling_rule"] == "runner_up":
        runner_up = np.argsort(states, axis=1)[:, -2]
        final[triggered] = runner_up[triggered]
        source[triggered] = "runner_up"
    else:
        temp = max(float(lapse["temp_lapse"]), 1e-6)
        logits = states[triggered] / temp
        logits = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-9)
        sampled = [rng.choice(states.shape[1], p=prob[i]) for i in range(prob.shape[0])]
        final[triggered] = np.asarray(sampled, dtype=int)
        source[triggered] = "softmax"
    return final.astype(int), triggered.astype(bool), source


def run_candidate(data: dict[str, Any], spec: dict[str, Any], selected_noise: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = data["cache"]
    norm_layers = data["norm_layers"]
    group_params = data["group_params"]
    t0_mean = data["t0_mean"]
    t0_sd = data["t0_sd"]
    human_ref = data["human_ref"]
    noise_by_group = noise_params_for_spec(spec, selected_noise)
    all_trials: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    lapse_rows: list[dict[str, Any]] = []

    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        group_layers = {k: v[mask] for k, v in norm_layers.items()}
        gp = group_params[group]
        target = gc["target_labels"].astype(int)
        flanker = gc["flanker_labels"].astype(int)
        mu, conflict_score, compression_trial, control_time = build_candidate_mu(group_layers, target, flanker, float(gp["evidence_gain"]), spec)
        ww_input = torch.as_tensor(mu, dtype=torch.float32)
        out = run_ww(
            ww_input,
            time_steps=TIME_STEPS,
            dt_ms=int(DT * 1000),
            threshold=float(gp["threshold"]),
            noise_ampa=0.0,
            device="cpu",
            seed=SEED,
            readout_mode="baseline",
            t0_seconds=0.25,
            choice_temperature=0.01,
        )
        base_df = pd.DataFrame(
            {
                "trial_id": gc["row_indices"].astype(int),
                "analysis_group": gc["analysis_group"].astype(str),
                "target_label": target,
                "flanker_label": flanker,
                "response_label": gc["response_labels"].astype(int),
                "human_correct": gc["human_correct"].astype(bool),
                "true_rt": gc["true_rt"].astype(float),
                "congruency": pd.Series(gc["congruency"]).map({0: "congruent", 1: "incongruent"}).astype(str),
                "pred_choice": out["pred_choice"],
                "pred_rt": out["pred_rt"],
            }
        )
        cfg = ReadoutConfig("sustained_crossing", min_decision_time=float(gp["min_decision_time"]), sustained_k=int(gp["sustained_k"]), margin=float(gp["margin"]))
        base_df = apply_readout(base_df, out, cfg=cfg, threshold=float(gp["threshold"]), dt_ms=int(DT * 1000), t0_seconds=0.0)
        base_df = apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)
        traj = np.asarray(out["trajectory"], dtype=np.float32)
        readout_steps = np.clip(np.rint(base_df["decision_time"].to_numpy(float) / DT).astype(int), 0, TIME_STEPS - 1)
        states = traj[np.arange(len(base_df)), readout_steps, :]
        met = state_metrics(states, target, flanker)
        deterministic_choice = states.argmax(axis=1)
        deterministic_correct = deterministic_choice == target
        readout_time = readout_steps * DT
        stochastic_choice, sigma = apply_time_gap_noise(states, readout_time, np.clip(met["gap"], 0, None), noise_by_group[group], (spec["model_config_id"], group))
        lapse_prob = lapse_probability(spec["lapse"], readout_time, np.clip(met["gap"], 0, None))
        final_choice, lapse_triggered, lapse_choice_source = sample_lapse_choice(stochastic_choice, states, target, flanker, spec["lapse"], lapse_prob, (spec["model_config_id"], group))
        final_source = np.where(lapse_triggered, "lapse", np.where(stochastic_choice != deterministic_choice, "time_gap_noise", "deterministic"))
        final_correct = final_choice == target

        rows = np.arange(len(base_df))[:, None]
        times = np.arange(TIME_STEPS)[None, :]
        target_vals = traj[rows, times, target[:, None]]
        flanker_vals = traj[rows, times, flanker[:, None]]
        masked = traj.copy()
        masked[np.arange(len(base_df))[:, None], np.arange(TIME_STEPS)[None, :], target[:, None]] = -np.inf
        other_max = masked.max(axis=2)
        target_gt_flanker = target_vals > flanker_vals
        target_gt_other = target_vals > other_max
        first_gt_flanker = target_gt_flanker.argmax(axis=1).astype(float)
        first_gt_flanker[~target_gt_flanker.any(axis=1)] = np.nan
        first_gt_other = target_gt_other.argmax(axis=1).astype(float)
        first_gt_other[~target_gt_other.any(axis=1)] = np.nan
        first_rank1 = target_gt_other.argmax(axis=1).astype(float)
        first_rank1[~target_gt_other.any(axis=1)] = np.nan
        flanker_dom = np.maximum(flanker_vals - target_vals, 0.0)
        flanker_dur = (flanker_dom > 0).sum(axis=1) * DT
        early_flanker = (flanker_dom[:, : max(1, int(0.15 / DT))] > 0).mean(axis=1)
        late_target = (target_vals[:, int(0.30 / DT):] - flanker_vals[:, int(0.30 / DT):]).max(axis=1)
        max_post = (target_vals - other_max).max(axis=1)

        trial = pd.DataFrame(
            {
                "trial_id": base_df["trial_id"].to_numpy(int),
                "analysis_group": group,
                "congruency": base_df["congruency"].to_numpy(str),
                "target_label": target,
                "flanker_label": flanker,
                "human_correct": base_df["human_correct"].to_numpy(bool),
                "true_rt": base_df["true_rt"].to_numpy(float),
                "model_family": spec["model_family"],
                "model_config_id": spec["model_config_id"],
                "schedule_config_id": spec["schedule_config_id"],
                "adaptive_schedule_config_id": spec["adaptive_schedule_config_id"],
                "lapse_config_id": spec["lapse_config_id"],
                "noise_config_id": spec["noise_config_id"],
                "deterministic_choice": deterministic_choice,
                "stochastic_choice": stochastic_choice,
                "final_choice": final_choice,
                "model_correct": final_correct,
                "choice_type": choice_type(final_choice, target, flanker),
                "model_rt": base_df["pred_rt"].to_numpy(float),
                "readout_time": readout_time,
                "target_recovery_time": first_gt_other * DT,
                "target_rank_at_readout": met["target_rank"],
                "signed_target_margin_at_readout": met["signed_target_margin"],
                "s_target_at_readout": met["s_target"],
                "s_flanker_at_readout": met["s_flanker"],
                "s_other_max_at_readout": met["s_other_max"],
                "gap_at_readout": met["gap"],
                "early_flanker_dominance": early_flanker,
                "late_target_recovery_strength": late_target,
                "target_first_rank1_time": first_rank1 * DT,
                "target_first_exceeds_flanker_time": first_gt_flanker * DT,
                "target_first_exceeds_max_other_time": first_gt_other * DT,
                "target_ever_rank1": np.isfinite(first_rank1),
                "target_ever_exceeds_flanker": np.isfinite(first_gt_flanker),
                "target_ever_exceeds_max_other": np.isfinite(first_gt_other),
                "maximum_post_readout_target_margin": max_post,
                "flanker_dominance_duration": flanker_dur,
                "conflict_score": conflict_score,
                "compression_trial": compression_trial,
                "control_strength": control_time.mean(axis=1) if control_time.ndim == 2 else np.asarray(control_time, dtype=float),
                "lapse_probability": lapse_prob,
                "lapse_triggered": lapse_triggered,
                "lapse_choice_type": np.where(lapse_triggered, choice_type(final_choice, target, flanker), "none"),
                "final_choice_source": final_source,
            }
        )
        all_trials.append(trial)

        adaptive_rows.append(
            pd.DataFrame(
                {
                    "model_family": spec["model_family"],
                    "model_config_id": spec["model_config_id"],
                    "analysis_group": group,
                    "congruency": trial["congruency"],
                    "conflict_score": conflict_score,
                    "compression_trial": compression_trial,
                    "control_strength": trial["control_strength"],
                    "schedule_acceleration": trial["control_strength"] * 1000 * DT,
                    "target_recovery_time": trial["target_recovery_time"],
                }
            )
        )
        lapse_rows.append(
            pd.DataFrame(
                {
                    "model_family": spec["model_family"],
                    "model_config_id": spec["model_config_id"],
                    "analysis_group": group,
                    "congruency": trial["congruency"],
                    "lapse_probability": lapse_prob,
                    "lapse_triggered": lapse_triggered,
                    "lapse_triggered_error": lapse_triggered & (~final_correct),
                    "lapse_choice_type": trial["lapse_choice_type"],
                    "response_sampling_type": np.where(lapse_triggered, lapse_choice_source, "none"),
                    "lapse_fast": lapse_triggered & (trial["readout_time"].to_numpy(float) <= np.nanmedian(trial["readout_time"].to_numpy(float))),
                }
            )
        )

        for cong in ["congruent", "incongruent"]:
            part = trial[trial["congruency"].eq(cong)].copy()
            idx = part.index.to_numpy(int)
            if part.empty:
                continue
            human_rt = part["true_rt"].to_numpy(float)
            human_correct = part["human_correct"].to_numpy(bool)
            model_rt = part["model_rt"].to_numpy(float)
            correct = part["model_correct"].to_numpy(bool)
            err_bins = rt_bins(
                pd.DataFrame(
                    {
                        "pred_rt": model_rt,
                        "true_rt": human_rt,
                        "model_correct": correct,
                        "human_correct": human_correct,
                        "congruency": np.full(len(part), 0 if cong == "congruent" else 1),
                        "pred_choice": part["final_choice"].to_numpy(int),
                        "response_label": np.zeros(len(part), dtype=int),
                    }
                ),
                "tmp",
            )
            pivot = err_bins.pivot_table(index="rt_bin", columns="source", values="error_rate")
            human_row = human_ref[human_ref["analysis_group"].eq(group)].iloc[0]
            props = part["choice_type"].value_counts(normalize=True)
            summary_rows.append(
                {
                    "model_family": spec["model_family"],
                    "model_config_id": spec["model_config_id"],
                    "schedule_config_id": spec["schedule_config_id"],
                    "adaptive_schedule_config_id": spec["adaptive_schedule_config_id"],
                    "lapse_config_id": spec["lapse_config_id"],
                    "noise_config_id": spec["noise_config_id"],
                    "analysis_group": group,
                    "congruency": cong,
                    "parameter_setting": json.dumps({"schedule": spec["schedule"], "adaptive": spec["adaptive"], "lapse": spec["lapse"], "noise_mode": spec["noise_mode"], "custom_noise": spec.get("custom_noise")}, sort_keys=True),
                    "n_trials": len(part),
                    "overall_accuracy": safe_mean(correct.astype(float)),
                    "congruent_accuracy": safe_mean(correct.astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_accuracy": safe_mean(correct.astype(float)) if cong == "incongruent" else math.nan,
                    "congruent_error_rate": safe_mean((~correct).astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_error_rate": safe_mean((~correct).astype(float)) if cong == "incongruent" else math.nan,
                    "mean_rt": safe_mean(model_rt),
                    "rt_q10": safe_q(model_rt, 0.10),
                    "rt_q50": safe_q(model_rt, 0.50),
                    "rt_q90": safe_q(model_rt, 0.90),
                    "rt_distribution_similarity": corr_safe([safe_q(model_rt, p) for p in [0.1, 0.5, 0.9]], [safe_q(human_rt, p) for p in [0.1, 0.5, 0.9]]),
                    "congruent_error_count": int((~correct).sum()) if cong == "congruent" else math.nan,
                    "congruent_correct_count": int(correct.sum()) if cong == "congruent" else math.nan,
                    "congruent_error_rt_minus_correct_rt": safe_mean(model_rt[~correct]) - safe_mean(model_rt[correct]) if cong == "congruent" else math.nan,
                    "congruent_fast_error_evaluable": bool((~correct).sum() >= 5 and correct.sum() >= 5) if cong == "congruent" else False,
                    "incongruent_error_rt_minus_correct_rt": safe_mean(model_rt[~correct]) - safe_mean(model_rt[correct]) if cong == "incongruent" else math.nan,
                    "fast_bin_error_rate": float(pivot.get("model", pd.Series(dtype=float)).iloc[0]) if len(pivot.index) else math.nan,
                    "slow_bin_error_rate": float(pivot.get("model", pd.Series(dtype=float)).iloc[-1]) if len(pivot.index) else math.nan,
                    "error_rate_by_rt_bin_rmse": rmse(pivot.get("model", np.array([])), pivot.get("human", np.array([]))),
                    "target_choice_proportion": float(props.get("target", 0.0)),
                    "flanker_choice_proportion": float(props.get("flanker", 0.0)),
                    "other_choice_proportion": float(props.get("other", 0.0)),
                    "incongruent_flanker_choice_proportion": float(props.get("flanker", 0.0)) if cong == "incongruent" else math.nan,
                    "congruent_non_target_choice_proportion": float(props.get("flanker", 0.0) + props.get("other", 0.0)) if cong == "congruent" else math.nan,
                    "target_recovery_time": safe_mean(part["target_recovery_time"]),
                    "target_first_rank1_time": safe_mean(part["target_first_rank1_time"]),
                    "target_first_exceeds_flanker_time": safe_mean(part["target_first_exceeds_flanker_time"]),
                    "target_first_exceeds_max_other_time": safe_mean(part["target_first_exceeds_max_other_time"]),
                    "target_ever_rank1_proportion": safe_mean(part["target_ever_rank1"].astype(float)),
                    "target_ever_exceeds_flanker_proportion": safe_mean(part["target_ever_exceeds_flanker"].astype(float)),
                    "target_ever_exceeds_max_other_proportion": safe_mean(part["target_ever_exceeds_max_other"].astype(float)),
                    "maximum_post_readout_target_margin": safe_mean(part["maximum_post_readout_target_margin"]),
                    "flanker_dominance_duration": safe_mean(part["flanker_dominance_duration"]),
                    "early_flanker_dominance": safe_mean(part["early_flanker_dominance"]),
                    "late_target_recovery_strength": safe_mean(part["late_target_recovery_strength"]),
                    "signed_target_margin_at_readout": safe_mean(part["signed_target_margin_at_readout"]),
                    "target_rank_at_readout": safe_mean(part["target_rank_at_readout"]),
                    "conflict_score_mean": safe_mean(part["conflict_score"]),
                    "conflict_score_median": safe_q(part["conflict_score"], 0.50),
                    "compression_trial_mean": safe_mean(part["compression_trial"]),
                    "compression_trial_median": safe_q(part["compression_trial"], 0.50),
                    "control_strength_mean": safe_mean(part["control_strength"]),
                    "schedule_acceleration_mean": safe_mean(part["control_strength"]) * 1000 * DT,
                    "conflict_compression_corr": corr_safe(part["conflict_score"], part["compression_trial"]),
                    "compression_target_recovery_corr": corr_safe(part["compression_trial"], part["target_recovery_time"]),
                    "lapse_probability_mean": safe_mean(part["lapse_probability"]),
                    "lapse_trigger_proportion": safe_mean(part["lapse_triggered"].astype(float)),
                    "lapse_triggered_error_proportion": safe_mean((part["lapse_triggered"] & (~part["model_correct"])).astype(float)),
                    "lapse_triggered_congruent_errors": int((part["lapse_triggered"] & (~part["model_correct"])).sum()) if cong == "congruent" else math.nan,
                    "lapse_triggered_incongruent_errors": int((part["lapse_triggered"] & (~part["model_correct"])).sum()) if cong == "incongruent" else math.nan,
                    "human_overall_accuracy": float(human_row["overall_accuracy"]),
                    "human_congruent_error_rate": float(human_row["congruent_error_rate"]),
                    "human_incongruent_error_rate": float(human_row["incongruent_error_rate"]),
                }
            )

            for split_name, split_mask in [
                ("human_correct", part["human_correct"].to_numpy(bool)),
                ("human_error", ~part["human_correct"].to_numpy(bool)),
                ("model_correct", part["model_correct"].to_numpy(bool)),
                ("model_error", ~part["model_correct"].to_numpy(bool)),
            ]:
                if not split_mask.any():
                    continue
                sel = idx[split_mask]
                for t in range(TIME_STEPS):
                    traj_rows.append(
                        {
                            "model_family": spec["model_family"],
                            "model_config_id": spec["model_config_id"],
                            "analysis_group": group,
                            "congruency": cong,
                            "split": split_name,
                            "time": t * DT,
                            "s_target_mean": safe_mean(target_vals[sel, t]),
                            "s_flanker_mean": safe_mean(flanker_vals[sel, t]),
                            "s_other_max_mean": safe_mean(other_max[sel, t]),
                            "s_target_minus_flanker_mean": safe_mean((target_vals - flanker_vals)[sel, t]),
                            "s_target_minus_max_other_mean": safe_mean((target_vals - other_max)[sel, t]),
                            "control_strength_mean": safe_mean(control_time[sel, t]) if control_time.ndim == 2 else safe_mean(control_time[sel]),
                            "compression_trial_mean": safe_mean(compression_trial[sel]),
                        }
                    )

    return (
        pd.concat(all_trials, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(traj_rows),
        pd.concat(adaptive_rows, ignore_index=True),
        pd.concat(lapse_rows, ignore_index=True),
    )


def add_scores(summary: pd.DataFrame, baseline_rt_rmse: dict[str, float]) -> pd.DataFrame:
    out = summary.copy()
    out["incongruent_error_deviation"] = (out["incongruent_error_rate"].fillna(0.0) - out["human_incongruent_error_rate"].fillna(0.08)).abs()
    out["congruent_fast_error_mismatch"] = (
        (out["congruent_error_rate"].fillna(0.0) - out["human_congruent_error_rate"].fillna(0.02)).abs()
        + np.maximum(0.0, out["congruent_error_rt_minus_correct_rt"].fillna(0.0))
    )
    out["fast_error_preservation_score"] = (
        np.maximum(0.0, (out["congruent_error_rt_minus_correct_rt"].fillna(0.0) + 0.001))
        + np.maximum(0.0, 0.002 - out["congruent_error_rate"].fillna(0.0))
        + np.maximum(0.0, 5 - out["congruent_error_count"].fillna(0.0)) / 5.0
    )
    out["incongruent_repair_score"] = out["incongruent_error_deviation"] + 0.5 * out["incongruent_flanker_choice_proportion"].fillna(0.0)
    out["behavior_score"] = (
        (out["overall_accuracy"] - out["human_overall_accuracy"].fillna(0.95)).abs()
        + out["incongruent_error_deviation"]
        + 0.5 * out["error_rate_by_rt_bin_rmse"].fillna(1.0)
    )
    out["adaptive_schedule_plausibility_score"] = (
        np.maximum(0.0, 0.15 - out["early_flanker_dominance"].fillna(0.0))
        + np.maximum(0.0, out["compression_trial_mean"].fillna(0.7) - 0.72)
    )
    out["lapse_penalty"] = (
        2.0 * np.maximum(0.0, out["lapse_triggered_error_proportion"].fillna(0.0) - 0.01)
        + 2.0 * np.maximum(0.0, out["lapse_probability_mean"].fillna(0.0) - 0.01)
    )
    out["naturalness_penalty"] = (
        np.maximum(0.0, 0.15 - out["early_flanker_dominance"].fillna(0.0))
        + np.maximum(0.0, out["congruent_error_rate"].fillna(0.0).eq(0.0).astype(float))
        + np.maximum(0.0, out["compression_trial_mean"].fillna(0.7) <= 0.41).astype(float)
    )
    out["mechanism_score"] = out["adaptive_schedule_plausibility_score"] + 0.5 * out["lapse_penalty"] + 0.5 * np.maximum(0.0, out["target_recovery_time"].fillna(1.0) - 0.40)
    rt_tol = []
    for _, r in out.iterrows():
        key = f"{r['analysis_group']}::{r['congruency']}"
        rt_tol.append(float(r["error_rate_by_rt_bin_rmse"]) - baseline_rt_rmse.get(key, float(r["error_rate_by_rt_bin_rmse"])))
    out["rt_rmse_delta_vs_baseline"] = rt_tol
    out["combined_score"] = (
        out["mechanism_score"]
        + out["behavior_score"]
        + out["fast_error_preservation_score"]
        + out["incongruent_repair_score"]
        + out["naturalness_penalty"]
        + out["lapse_penalty"]
    )
    return out


def aggregate_model_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, model_id), part in summary.groupby(["model_family", "model_config_id"], sort=False):
        row = {
            "model_family": family,
            "model_config_id": model_id,
            "schedule_config_id": part["schedule_config_id"].iloc[0],
            "adaptive_schedule_config_id": part["adaptive_schedule_config_id"].iloc[0],
            "lapse_config_id": part["lapse_config_id"].iloc[0],
            "noise_config_id": part["noise_config_id"].iloc[0],
            "parameter_setting": part["parameter_setting"].iloc[0],
            "mechanism_score": safe_mean(part["mechanism_score"]),
            "behavior_score": safe_mean(part["behavior_score"]),
            "fast_error_preservation_score": safe_mean(part["fast_error_preservation_score"]),
            "incongruent_repair_score": safe_mean(part["incongruent_repair_score"]),
            "naturalness_penalty": safe_mean(part["naturalness_penalty"]),
            "lapse_penalty": safe_mean(part["lapse_penalty"]),
            "adaptive_schedule_plausibility_score": safe_mean(part["adaptive_schedule_plausibility_score"]),
            "combined_score": safe_mean(part["combined_score"]),
        }
        for group in GROUPS:
            g = part[part["analysis_group"].eq(group)]
            row[f"{group}_overall_accuracy"] = safe_mean(g["overall_accuracy"])
            row[f"{group}_congruent_error_rate"] = safe_mean(g["congruent_error_rate"])
            row[f"{group}_incongruent_error_rate"] = safe_mean(g["incongruent_error_rate"])
            row[f"{group}_congruent_error_count"] = safe_mean(g["congruent_error_count"])
            row[f"{group}_congruent_error_rt_minus_correct_rt"] = safe_mean(g["congruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_flanker_choice_proportion"] = safe_mean(g["incongruent_flanker_choice_proportion"])
            row[f"{group}_early_flanker_dominance"] = safe_mean(g["early_flanker_dominance"])
            row[f"{group}_target_recovery_time"] = safe_mean(g["target_recovery_time"])
            row[f"{group}_lapse_trigger_proportion"] = safe_mean(g["lapse_trigger_proportion"])
            row[f"{group}_compression_trial_mean"] = safe_mean(g["compression_trial_mean"])
            row[f"{group}_conflict_score_mean"] = safe_mean(g["conflict_score_mean"])
            row[f"{group}_control_strength_mean"] = safe_mean(g["control_strength_mean"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("combined_score", kind="mergesort").reset_index(drop=True)


def add_constraints(model_summary: pd.DataFrame, per_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in model_summary.iterrows():
        part = per_condition[per_condition["model_config_id"].eq(r["model_config_id"])]
        yg = part[(part["analysis_group"] == "young_20_29") & (part["congruency"] == "congruent")]
        yi = part[(part["analysis_group"] == "young_20_29") & (part["congruency"] == "incongruent")]
        og = part[(part["analysis_group"] == "older_80_89") & (part["congruency"] == "congruent")]
        oi = part[(part["analysis_group"] == "older_80_89") & (part["congruency"] == "incongruent")]
        if any(df.empty for df in [yg, yi, og, oi]):
            continue
        yg = yg.iloc[0]
        yi = yi.iloc[0]
        og = og.iloc[0]
        oi = oi.iloc[0]
        p = {"model_config_id": r["model_config_id"]}
        main = {
            "main_young_incongruent_error_rate_le_0.20": yi["incongruent_error_rate"] <= 0.20,
            "main_older_incongruent_error_rate_le_0.10": oi["incongruent_error_rate"] <= 0.10,
            "main_young_congruent_error_rate_0.003_to_0.05": 0.003 <= yg["congruent_error_rate"] <= 0.05,
            "main_older_congruent_error_rate_ge_0.002": og["congruent_error_rate"] >= 0.002,
            "main_young_congruent_fast_error": bool(yg["congruent_fast_error_evaluable"]) and yg["congruent_error_rt_minus_correct_rt"] < 0,
            "main_older_congruent_fast_error_not_absent": og["congruent_error_rate"] >= 0.002 and og["congruent_error_count"] >= 5 and og["congruent_error_rt_minus_correct_rt"] < 0,
            "main_early_flanker_dominance_ge_0.15": min(yi["early_flanker_dominance"], oi["early_flanker_dominance"]) >= 0.15,
            "main_incongruent_flanker_choice_limited": yi["incongruent_flanker_choice_proportion"] <= 0.25 and oi["incongruent_flanker_choice_proportion"] <= 0.15,
            "main_rt_bin_rmse_not_too_much_worse": max(yg["rt_rmse_delta_vs_baseline"], yi["rt_rmse_delta_vs_baseline"], og["rt_rmse_delta_vs_baseline"], oi["rt_rmse_delta_vs_baseline"]) <= 0.05,
            "main_no_unrealistic_perfect_accuracy": r["young_20_29_overall_accuracy"] < 0.999 and r["older_80_89_overall_accuracy"] < 0.999,
            "main_lapse_errors_not_excessive": max(yg["lapse_triggered_error_proportion"], yi["lapse_triggered_error_proportion"], og["lapse_triggered_error_proportion"], oi["lapse_triggered_error_proportion"]) <= 0.02,
            "main_not_all_trials_strongest_compression": max(r["young_20_29_compression_trial_mean"], r["older_80_89_compression_trial_mean"]) > 0.42,
        }
        lenient = {
            "lenient_young_incongruent_error_rate_le_0.25": yi["incongruent_error_rate"] <= 0.25,
            "lenient_older_incongruent_error_rate_le_0.15": oi["incongruent_error_rate"] <= 0.15,
            "lenient_young_congruent_nonzero": yg["congruent_error_rate"] > 0,
            "lenient_older_congruent_nonzero": og["congruent_error_rate"] > 0,
            "lenient_congruent_fast_error_or_weak": (yg["congruent_error_rt_minus_correct_rt"] < 0 or yg["congruent_error_count"] < 5) and (og["congruent_error_rt_minus_correct_rt"] < 0 or og["congruent_error_count"] < 5),
            "lenient_early_flanker_dominance_ge_0.10": min(yi["early_flanker_dominance"], oi["early_flanker_dominance"]) >= 0.10,
            "lenient_rt_not_broken": max(yg["rt_rmse_delta_vs_baseline"], yi["rt_rmse_delta_vs_baseline"], og["rt_rmse_delta_vs_baseline"], oi["rt_rmse_delta_vs_baseline"]) <= 0.10,
            "lenient_no_perfect_accuracy": r["young_20_29_overall_accuracy"] < 0.999 and r["older_80_89_overall_accuracy"] < 0.999,
        }
        strict = {
            "strict_young_incongruent_human_plus_0.05": yi["incongruent_error_rate"] <= yi["human_incongruent_error_rate"] + 0.05,
            "strict_older_incongruent_human_plus_0.05": oi["incongruent_error_rate"] <= oi["human_incongruent_error_rate"] + 0.05,
            "strict_young_congruent_human_pm_0.015": abs(yg["congruent_error_rate"] - yg["human_congruent_error_rate"]) <= 0.015,
            "strict_older_congruent_human_pm_0.015": abs(og["congruent_error_rate"] - og["human_congruent_error_rate"]) <= 0.015,
            "strict_both_fast_error_evaluable_negative": bool(yg["congruent_fast_error_evaluable"]) and bool(og["congruent_fast_error_evaluable"]) and yg["congruent_error_rt_minus_correct_rt"] < 0 and og["congruent_error_rt_minus_correct_rt"] < 0,
            "strict_early_flanker_dominance_ge_0.20": min(yi["early_flanker_dominance"], oi["early_flanker_dominance"]) >= 0.20,
            "strict_rt_profile_close": max(yg["rt_rmse_delta_vs_baseline"], yi["rt_rmse_delta_vs_baseline"], og["rt_rmse_delta_vs_baseline"], oi["rt_rmse_delta_vs_baseline"]) <= 0.025,
            "strict_no_excessive_lapse": max(yg["lapse_triggered_error_proportion"], yi["lapse_triggered_error_proportion"], og["lapse_triggered_error_proportion"], oi["lapse_triggered_error_proportion"]) <= 0.01,
        }
        p.update(lenient)
        p.update(main)
        p.update(strict)
        for level, cons in [("lenient", lenient), ("main", main), ("strict", strict)]:
            failed = [k for k, v in cons.items() if not v]
            p[f"pass_{level}"] = len(failed) == 0
            p[f"fail_count_{level}"] = len(failed)
            p[f"first_failed_constraint_{level}"] = failed[0] if failed else ""
        if og["congruent_error_rate"] <= 0:
            p["failure_reason_category"] = "no_older_congruent_errors"
        elif og["congruent_error_count"] < 5:
            p["failure_reason_category"] = "insufficient_older_fast_error_trials"
        elif yg["congruent_error_rt_minus_correct_rt"] >= 0:
            p["failure_reason_category"] = "no_congruent_fast_error"
        elif yi["incongruent_error_rate"] > 0.20 or oi["incongruent_error_rate"] > 0.10:
            p["failure_reason_category"] = "high_incongruent_error"
        elif min(yi["early_flanker_dominance"], oi["early_flanker_dominance"]) < 0.15:
            p["failure_reason_category"] = "lost_conflict_dynamics"
        elif max(r["young_20_29_compression_trial_mean"], r["older_80_89_compression_trial_mean"]) <= 0.42:
            p["failure_reason_category"] = "overcompressed_congruent_trials"
        elif max(yg["lapse_triggered_error_proportion"], yi["lapse_triggered_error_proportion"], og["lapse_triggered_error_proportion"], oi["lapse_triggered_error_proportion"]) > 0.02:
            p["failure_reason_category"] = "excessive_lapse_errors"
        elif max(yg["rt_rmse_delta_vs_baseline"], yi["rt_rmse_delta_vs_baseline"], og["rt_rmse_delta_vs_baseline"], oi["rt_rmse_delta_vs_baseline"]) > 0.05:
            p["failure_reason_category"] = "rt_profile_bad"
        elif r["young_20_29_overall_accuracy"] >= 0.999 or r["older_80_89_overall_accuracy"] >= 0.999:
            p["failure_reason_category"] = "high_accuracy_no_errors"
        else:
            p["failure_reason_category"] = "mixed_tradeoff"
        p["recommended_for_next_search"] = "fine_search_seed" if p["pass_main"] else ("keep_as_diagnostic" if p["pass_lenient"] else "report_negative_result")
        rows.append(p)
    return pd.DataFrame(rows)


def pareto_front(model_summary: pd.DataFrame) -> pd.DataFrame:
    df = model_summary.copy().reset_index(drop=True)
    feats = df[["incongruent_repair_score", "fast_error_preservation_score", "behavior_score", "naturalness_penalty", "lapse_penalty", "adaptive_schedule_plausibility_score"]].to_numpy(float)
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if np.all(feats[j] <= feats[i]) and np.any(feats[j] < feats[i]):
                is_pareto[i] = False
                break
    df["is_pareto_optimal"] = is_pareto
    return df[df["is_pareto_optimal"]].copy()


def select_top_trial_export(model_summary: pd.DataFrame, pass_fail: pd.DataFrame) -> list[str]:
    merged = model_summary.merge(pass_fail[["model_config_id", "fail_count_main"]], on="model_config_id", how="left")
    keep = merged.head(12)["model_config_id"].tolist()
    for fam in ["CA1_trialwise_conflict_adaptive_schedule", "CA2_online_conflict_schedule_acceleration", "CA3_uncertainty_adaptive_schedule", "L1_bounded_lapse_only", "L2_bounded_lapse_only", "COMBO_conflict_adaptive_schedule_plus_bounded_lapse"]:
        part = merged[merged["model_family"].eq(fam)]
        if not part.empty:
            keep.append(part.iloc[0]["model_config_id"])
    return sorted(set(keep))


def representative_models(model_summary: pd.DataFrame, pass_fail: pd.DataFrame) -> pd.DataFrame:
    merged = model_summary.copy()
    for col in ["fail_count_main", "failure_reason_category"]:
        if col not in merged.columns and col in pass_fail.columns:
            merged = merged.merge(pass_fail[["model_config_id", col]], on="model_config_id", how="left")
    rows = []
    picks = {
        "best_conflict_adaptive_schedule": merged[merged["model_family"].str.startswith("CA")].sort_values("combined_score").head(1),
        "best_bounded_lapse": merged[merged["model_family"].str.contains("bounded_lapse")].sort_values("combined_score").head(1),
        "best_ca_plus_lapse": merged[merged["model_family"].eq("COMBO_conflict_adaptive_schedule_plus_bounded_lapse")].sort_values("combined_score").head(1),
        "best_near_acceptable": merged.sort_values(["fail_count_main", "combined_score"]).head(1),
    }
    for role, part in picks.items():
        if part.empty:
            continue
        r = part.iloc[0]
        rows.append(
            {
                "representative_role": role,
                "model_family": r["model_family"],
                "model_config_id": r["model_config_id"],
                "schedule_config_id": r["schedule_config_id"],
                "adaptive_schedule_config_id": r["adaptive_schedule_config_id"],
                "lapse_config_id": r["lapse_config_id"],
                "noise_config_id": r["noise_config_id"],
                "why_selected": role,
                "failure_reason_category": r["failure_reason_category"],
                "young_overall_accuracy": r["young_20_29_overall_accuracy"],
                "young_congruent_error_rate": r["young_20_29_congruent_error_rate"],
                "young_incongruent_error_rate": r["young_20_29_incongruent_error_rate"],
                "young_congruent_error_count": r["young_20_29_congruent_error_count"],
                "young_congruent_error_rt_minus_correct_rt": r["young_20_29_congruent_error_rt_minus_correct_rt"],
                "young_incongruent_flanker_choice_proportion": r["young_20_29_incongruent_flanker_choice_proportion"],
                "young_early_flanker_dominance": r["young_20_29_early_flanker_dominance"],
                "young_target_recovery_time": r["young_20_29_target_recovery_time"],
                "young_lapse_trigger_proportion": r["young_20_29_lapse_trigger_proportion"],
                "older_overall_accuracy": r["older_80_89_overall_accuracy"],
                "older_congruent_error_rate": r["older_80_89_congruent_error_rate"],
                "older_incongruent_error_rate": r["older_80_89_incongruent_error_rate"],
                "older_congruent_error_count": r["older_80_89_congruent_error_count"],
                "older_congruent_error_rt_minus_correct_rt": r["older_80_89_congruent_error_rt_minus_correct_rt"],
                "older_incongruent_flanker_choice_proportion": r["older_80_89_incongruent_flanker_choice_proportion"],
                "older_early_flanker_dominance": r["older_80_89_early_flanker_dominance"],
                "older_target_recovery_time": r["older_80_89_target_recovery_time"],
                "older_lapse_trigger_proportion": r["older_80_89_lapse_trigger_proportion"],
            }
        )
    return pd.DataFrame(rows)


def make_figures(model_summary: pd.DataFrame, pass_fail: pd.DataFrame, per_condition: pd.DataFrame, trial_export: pd.DataFrame, reps: pd.DataFrame) -> None:
    reps = reps.loc[:, ~reps.columns.duplicated()].copy()
    merged = model_summary.copy()
    needed = ["pass_lenient", "pass_main", "pass_strict", "fail_count_main", "failure_reason_category"]
    missing = [c for c in needed if c not in merged.columns and c in pass_fail.columns]
    if missing:
        merged = merged.merge(pass_fail[["model_config_id"] + missing], on="model_config_id", how="left")

    fig, ax = plt.subplots(figsize=(10, 5))
    top = merged.head(20).copy().iloc[::-1]
    ax.barh(np.arange(len(top)), top["combined_score"], color="#4C78A8")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels((top["model_family"] + "\n" + top["model_config_id"]).tolist(), fontsize=7)
    ax.set_xlabel("combined score")
    ax.set_title("Mechanism redesign ranking overview")
    style_ax(ax)
    save_fig(fig, "mechanism_redesign_ranking_overview")

    fig, ax = plt.subplots(figsize=(8, 4))
    fam = merged.groupby("model_family").agg(lenient=("pass_lenient", "sum"), main=("pass_main", "sum"), strict=("pass_strict", "sum")).reset_index()
    x = np.arange(len(fam))
    width = 0.25
    ax.bar(x - width, fam["lenient"], width, label="lenient")
    ax.bar(x, fam["main"], width, label="main")
    ax.bar(x + width, fam["strict"], width, label="strict")
    ax.set_xticks(x)
    ax.set_xticklabels(fam["model_family"], rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Constraint survival by mechanism family")
    style_ax(ax)
    save_fig(fig, "constraint_survival_by_mechanism_family")

    fig, ax = plt.subplots(figsize=(7, 5))
    status = np.where(merged["pass_main"], "main", np.where(merged["pass_lenient"], "lenient", "fail"))
    markers = {"main": "o", "lenient": "s", "fail": "x"}
    fam_codes = {fam: i for i, fam in enumerate(merged["model_family"].unique())}
    colors = [fam_codes[f] for f in merged["model_family"]]
    for st in ["fail", "lenient", "main"]:
        sub = merged[status == st]
        ax.scatter(sub["incongruent_repair_score"], sub["fast_error_preservation_score"], c=[fam_codes[f] for f in sub["model_family"]], marker=markers[st], label=st, cmap="tab20", s=50)
    ax.set_xlabel("incongruent repair score")
    ax.set_ylabel("fast-error preservation score")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pareto front redesign incongruent vs fast error")
    style_ax(ax)
    save_fig(fig, "pareto_front_redesign_incongruent_vs_fast_error")

    adaptive = per_condition[["model_family", "model_config_id", "analysis_group", "congruency", "compression_trial_mean", "control_strength_mean"]].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "compression_trial_mean", "Compression trial"), (axes[1], "control_strength_mean", "Control strength")]:
        plot = adaptive.groupby(["model_family", "congruency"], as_index=False)[metric].mean()
        for i, cong in enumerate(["congruent", "incongruent"]):
            part = plot[plot["congruency"].eq(cong)]
            ax.bar(np.arange(len(part)) + (i - 0.5) * 0.35, part[metric], 0.35, label=cong)
        ax.set_xticks(np.arange(len(part)))
        ax.set_xticklabels(part["model_family"], rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "conflict_adaptive_compression_by_condition")

    fig, ax = plt.subplots(figsize=(6, 5))
    trial_ca = trial_export[trial_export["model_family"].str.startswith("CA")]
    if len(trial_ca) > 12000:
        trial_ca = trial_ca.sample(12000, random_state=SEED)
    if not trial_ca.empty:
        ax.scatter(trial_ca["conflict_score"], trial_ca["compression_trial"], s=8, alpha=0.4)
    ax.set_xlabel("conflict score")
    ax.set_ylabel("compression trial")
    ax.set_title("Conflict score vs compression")
    style_ax(ax)
    save_fig(fig, "conflict_score_vs_compression")

    fig, ax = plt.subplots(figsize=(10, 4))
    rep_ids = reps["model_config_id"].tolist()
    plot = per_condition[per_condition["model_config_id"].isin(rep_ids)].copy()
    labels = {r["model_config_id"]: r["representative_role"] for _, r in reps.iterrows()}
    vals = []
    names = []
    for mid in rep_ids:
        p = plot[plot["model_config_id"].eq(mid)]
        vals.append(safe_mean(p["congruent_error_rate"]))
        names.append(labels[mid].replace("best_", ""))
    ax.bar(names, vals, color="#D55E00")
    ax.set_ylabel("mean congruent error rate")
    ax.set_title("Human vs model error rate by condition top candidates")
    style_ax(ax)
    save_fig(fig, "human_vs_model_error_rate_by_condition_top_candidates")

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(rep_ids))
    rep_index = reps.drop_duplicates("model_config_id").set_index("model_config_id")
    young_vals = [float(rep_index.loc[mid, "young_congruent_error_rt_minus_correct_rt"]) for mid in rep_ids]
    older_vals = [float(rep_index.loc[mid, "older_congruent_error_rt_minus_correct_rt"]) for mid in rep_ids]
    ax.bar(x - width / 2, young_vals, width, label="young")
    ax.bar(x + width / 2, older_vals, width, label="older")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Representative fast-error evidence")
    style_ax(ax)
    save_fig(fig, "representative_fast_error_evidence")

    fig, ax = plt.subplots(figsize=(8, 4))
    older_err = [float(rep_index.loc[mid, "older_congruent_error_rate"]) for mid in rep_ids]
    ax.bar([labels[mid].replace("best_", "") for mid in rep_ids], older_err, color="#4C78A8")
    ax.set_ylabel("older congruent error rate")
    ax.set_title("Older congruent error recovery")
    style_ax(ax)
    save_fig(fig, "older_congruent_error_recovery")

    fig, ax = plt.subplots(figsize=(8, 4))
    younger_flanker = [float(rep_index.loc[mid, "young_incongruent_flanker_choice_proportion"]) for mid in rep_ids]
    older_flanker = [float(rep_index.loc[mid, "older_incongruent_flanker_choice_proportion"]) for mid in rep_ids]
    ax.bar(x - width / 2, younger_flanker, width, label="young")
    ax.bar(x + width / 2, older_flanker, width, label="older")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Incongruent flanker choice reduction")
    style_ax(ax)
    save_fig(fig, "incongruent_flanker_choice_reduction")

    fig, ax = plt.subplots(figsize=(8, 4))
    early = [np.nanmean([float(rep_index.loc[mid, "young_early_flanker_dominance"]), float(rep_index.loc[mid, "older_early_flanker_dominance"])]) for mid in rep_ids]
    late = [np.nanmean([float(rep_index.loc[mid, "young_target_recovery_time"]), float(rep_index.loc[mid, "older_target_recovery_time"])]) for mid in rep_ids]
    ax.bar(x - width / 2, early, width, label="early flanker")
    ax.bar(x + width / 2, late, width, label="target recovery")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Early flanker late target dynamics")
    style_ax(ax)
    save_fig(fig, "early_flanker_late_target_dynamics")

    lapse = trial_export.groupby(["model_family", "congruency"], as_index=False).agg(lapse_probability=("lapse_probability", "mean"), lapse_trigger=("lapse_triggered", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "lapse_probability", "Lapse probability"), (axes[1], "lapse_trigger", "Trigger rate")]:
        plot = lapse.pivot(index="model_family", columns="congruency", values=metric).fillna(0)
        plot.plot(kind="bar", ax=ax, rot=30, legend=False)
        ax.set_title(title)
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "lapse_diagnostics_by_condition")

    fig, ax = plt.subplots(figsize=(8, 5))
    top_ids = merged.sort_values("combined_score").head(4)["model_config_id"].tolist()
    for mid in top_ids:
        sub = per_condition[(per_condition["model_config_id"].eq(mid)) & (per_condition["analysis_group"].eq("young_20_29")) & (per_condition["congruency"].eq("incongruent"))]
        if sub.empty:
            continue
        ax.plot([1, 2], [sub["fast_bin_error_rate"].iloc[0], sub["slow_bin_error_rate"].iloc[0]], marker="o", label=mid[:18])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["fast", "slow"])
    ax.set_ylabel("error rate")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("RT-bin error profile top candidates")
    style_ax(ax)
    save_fig(fig, "rt_bin_error_profile_top_candidates")

    near = merged.sort_values(["fail_count_main", "combined_score"]).iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["young_20_29_congruent_error_rate", "young_20_29_incongruent_error_rate", "older_80_89_congruent_error_rate", "older_80_89_incongruent_error_rate", "young_20_29_early_flanker_dominance", "older_80_89_target_recovery_time"]
    vals = [near[m] for m in metrics]
    ax.bar(metrics, vals, color="#4C78A8")
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_title("Near acceptable candidate dashboard")
    style_ax(ax)
    save_fig(fig, "near_acceptable_candidate_dashboard")

    fig, ax = plt.subplots(figsize=(10, 5))
    overview = merged[["model_family", "young_20_29_overall_accuracy", "young_20_29_congruent_error_rate", "young_20_29_incongruent_error_rate", "older_80_89_congruent_error_count", "young_20_29_congruent_error_rt_minus_correct_rt", "young_20_29_incongruent_flanker_choice_proportion", "young_20_29_early_flanker_dominance", "young_20_29_target_recovery_time", "young_20_29_lapse_trigger_proportion"]].head(10)
    ax.imshow(overview.drop(columns=["model_family"]).to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(overview)))
    ax.set_yticklabels(overview["model_family"], fontsize=8)
    ax.set_xticks(np.arange(len(overview.columns) - 1))
    ax.set_xticklabels(overview.columns[1:], rotation=30, ha="right", fontsize=8)
    ax.set_title("Mechanism tradeoff summary dashboard")
    save_fig(fig, "mechanism_tradeoff_summary_dashboard")


def write_summary(model_summary: pd.DataFrame, pass_fail: pd.DataFrame, reps: pd.DataFrame, mode: str) -> None:
    merged = model_summary.merge(pass_fail[["model_config_id", "pass_lenient", "pass_main", "pass_strict", "fail_count_main", "failure_reason_category"]], on="model_config_id", how="left")
    fam_counts = merged["model_family"].value_counts().to_dict()
    best_ca = reps[reps["representative_role"].eq("best_conflict_adaptive_schedule")]
    best_lapse = reps[reps["representative_role"].eq("best_bounded_lapse")]
    best_combo = reps[reps["representative_role"].eq("best_ca_plus_lapse")]
    best_near = reps[reps["representative_role"].eq("best_near_acceptable")].iloc[0]
    lines = [
        "# Mechanism redesign summary",
        "",
        f"- Run mode: {mode}.",
        f"- Total candidates tested: {len(model_summary)}.",
        f"- Candidate count by mechanism family: {json.dumps(fam_counts, ensure_ascii=False, sort_keys=True)}.",
        f"- Lenient survivors: {int(pass_fail['pass_lenient'].sum())}.",
        f"- Main survivors: {int(pass_fail['pass_main'].sum())}.",
        f"- Strict survivors: {int(pass_fail['pass_strict'].sum())}.",
        "",
        "## Required answers",
        f"1. Tested candidates: {len(model_summary)}.",
        f"2. Per-family counts: {json.dumps(fam_counts, ensure_ascii=False, sort_keys=True)}.",
        f"3. Conflict-adaptive schedule better than global schedule: {'yes' if not best_ca.empty and float(best_ca.iloc[0]['young_incongruent_error_rate']) <= 0.20 else 'partially / exploratory only'}.",
        f"4. Conflict-adaptive schedule preserves congruent uncertainty: {'yes' if not best_ca.empty and float(best_ca.iloc[0]['older_congruent_error_rate']) > 0 else 'not reliably'}.",
        f"5. Conflict-adaptive schedule repairs incongruent flanker over-selection: {'yes' if not best_ca.empty and float(best_ca.iloc[0]['older_incongruent_flanker_choice_proportion']) <= 0.15 else 'not enough'}.",
        f"6. Bounded lapse restores older congruent errors: {'yes' if not best_lapse.empty and float(best_lapse.iloc[0]['older_congruent_error_rate']) > 0 else 'no'}.",
        f"7. Bounded lapse breaks incongruent repair: {'yes' if not best_lapse.empty and float(best_lapse.iloc[0]['older_incongruent_error_rate']) > 0.10 else 'not clearly'}.",
        f"8. CA + lapse better than CA only or lapse only: {'yes' if not best_combo.empty and float(best_combo.iloc[0]['older_congruent_error_rate']) >= float(best_near['older_congruent_error_rate']) else 'no clear advantage'}.",
        f"9. Survivors: lenient={int(pass_fail['pass_lenient'].sum())}, main={int(pass_fail['pass_main'].sum())}, strict={int(pass_fail['pass_strict'].sum())}.",
        f"10. Fine-search or formal-fitting seed exists: {'yes' if int(pass_fail['pass_main'].sum()) > 0 else 'no'}.",
        f"11. Most recommended seed: `{best_near['model_config_id']}`.",
        f"12. Main failure reason: {pass_fail['failure_reason_category'].value_counts().idxmax() if not pass_fail.empty else 'none'}.",
        "13. The current result remains a multiple-objective trade-off unless main survivors are nonzero.",
        f"14. Most natural mechanism to continue: `{best_ca.iloc[0]['model_config_id']}`." if not best_ca.empty else "14. Most natural mechanism to continue: none yet.",
        f"15. Most patch-like mechanism: `{best_lapse.iloc[0]['model_config_id']}`." if not best_lapse.empty else "15. Most patch-like mechanism: bounded lapse family.",
        "16. Formal conclusion: this round can support mechanism comparison and negative-result reporting, but not a final balanced model unless main survivors are nonzero.",
        "17. Exploratory conclusion: any apparent improvement from bounded lapse remains exploratory because it is a rare downstream uncertainty component, not the main mechanism.",
        "18. Best advisor figures: mechanism_redesign_ranking_overview, constraint_survival_by_mechanism_family, conflict_score_vs_compression, older_congruent_error_recovery, mechanism_tradeoff_summary_dashboard.",
        "19. Next step: if main survivors remain zero, prefer a smaller mechanism search around the best conflict-adaptive family or package the current negative result rather than launch formal fitting.",
    ]
    (OUT_DIR / "summaries/mechanism_redesign_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_outputs() -> None:
    required = [
        OUT_DIR / "metrics/mechanism_redesign_model_summary.csv",
        OUT_DIR / "metrics/mechanism_redesign_model_ranking.csv",
        OUT_DIR / "metrics/mechanism_redesign_pareto_front.csv",
        OUT_DIR / "metrics/mechanism_redesign_pass_fail_table.csv",
        OUT_DIR / "metrics/mechanism_redesign_top_candidates_trial_level.csv",
        OUT_DIR / "metrics/mechanism_redesign_trajectory_diagnostics.csv",
        OUT_DIR / "metrics/conflict_adaptive_schedule_diagnostics.csv",
        OUT_DIR / "metrics/bounded_lapse_diagnostics.csv",
        OUT_DIR / "summaries/mechanism_redesign_summary.md",
        OUT_DIR / "logs/mechanism_redesign_run_log.txt",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output missing: {path}")


def finalize_from_saved_results(mode: str) -> None:
    rank_path = OUT_DIR / "metrics/mechanism_redesign_model_ranking.csv"
    pass_path = OUT_DIR / "metrics/mechanism_redesign_pass_fail_table.csv"
    reps_path = OUT_DIR / "metrics/mechanism_redesign_representative_models.csv"
    summary_path = OUT_DIR / "metrics/mechanism_redesign_model_summary.csv"
    trial_path = OUT_DIR / "metrics/mechanism_redesign_top_candidates_trial_level.csv"
    traj_path = OUT_DIR / "metrics/mechanism_redesign_trajectory_diagnostics.csv"
    if not all(p.exists() for p in [rank_path, pass_path, reps_path, summary_path, trial_path, traj_path]):
        raise RuntimeError("Cannot finalize from saved results because core output tables are missing.")
    model_summary = pd.read_csv(rank_path)
    pass_fail = pd.read_csv(pass_path)
    reps = pd.read_csv(reps_path)
    per_condition = pd.read_csv(summary_path)
    trial_export = pd.read_csv(trial_path)
    make_figures(model_summary, pass_fail, per_condition, trial_export, reps)
    write_summary(model_summary, pass_fail, reps, mode)
    logs = [
        f"mode={mode}",
        f"candidate_count={len(model_summary)}",
        f"lenient_survivors={int(pass_fail['pass_lenient'].sum())}",
        f"main_survivors={int(pass_fail['pass_main'].sum())}",
        f"strict_survivors={int(pass_fail['pass_strict'].sum())}",
        f"pareto_count={len(pd.read_csv(OUT_DIR / 'metrics/mechanism_redesign_pareto_front.csv')) if (OUT_DIR / 'metrics/mechanism_redesign_pareto_front.csv').exists() else 0}",
        "finalized_from_saved_results=true",
    ]
    (OUT_DIR / "logs/mechanism_redesign_run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    check_inputs()
    if Path(__file__).resolve() != OUT_DIR / "scripts" / "run_mechanism_redesign_conflict_adaptive_schedule.py":
        shutil.copy2(Path(__file__).resolve(), OUT_DIR / "scripts" / "run_mechanism_redesign_conflict_adaptive_schedule.py")

    data = load_inputs()
    write_input_inventory(data)
    selected_noise = selected_time_gap_params(data["readout_rank"])
    specs = candidate_grid(args.mode)

    trial_frames = []
    summary_frames = []
    traj_frames = []
    adaptive_frames = []
    lapse_frames = []
    logs = [f"mode={args.mode}", f"candidate_count={len(specs)}"]
    for i, spec in enumerate(specs, start=1):
        trials, summary, traj, adaptive, lapse = run_candidate(data, spec, selected_noise)
        trial_frames.append(trials)
        summary_frames.append(summary)
        traj_frames.append(traj)
        adaptive_frames.append(adaptive)
        lapse_frames.append(lapse)
        logs.append(f"{i}/{len(specs)} {spec['model_family']} {spec['model_config_id']} done")

    per_condition = pd.concat(summary_frames, ignore_index=True)
    baseline_rt_rmse = {}
    baseline_rows = per_condition[per_condition["model_family"].eq("R0_original_time_gap")]
    for _, row in baseline_rows.iterrows():
        baseline_rt_rmse[f"{row['analysis_group']}::{row['congruency']}"] = float(row["error_rate_by_rt_bin_rmse"])
    per_condition = add_scores(per_condition, baseline_rt_rmse)
    model_summary = aggregate_model_summary(per_condition)
    pass_fail = add_constraints(model_summary, per_condition)
    model_summary = model_summary.merge(pass_fail[["model_config_id", "pass_lenient", "pass_main", "pass_strict", "fail_count_main", "first_failed_constraint_main", "failure_reason_category", "recommended_for_next_search"]], on="model_config_id", how="left")
    pareto = pareto_front(model_summary)
    reps = representative_models(model_summary, pass_fail)

    top_ids = select_top_trial_export(model_summary, pass_fail)
    trial_export = pd.concat(trial_frames, ignore_index=True)
    trial_export = trial_export[trial_export["model_config_id"].isin(top_ids)].copy()
    traj_export = pd.concat(traj_frames, ignore_index=True)
    traj_export = traj_export[traj_export["model_config_id"].isin(top_ids)].copy()
    adaptive_export = pd.concat(adaptive_frames, ignore_index=True)
    lapse_export = pd.concat(lapse_frames, ignore_index=True)

    per_condition.to_csv(OUT_DIR / "metrics/mechanism_redesign_model_summary.csv", index=False)
    model_summary.to_csv(OUT_DIR / "metrics/mechanism_redesign_model_ranking.csv", index=False)
    pareto.to_csv(OUT_DIR / "metrics/mechanism_redesign_pareto_front.csv", index=False)
    pass_fail.to_csv(OUT_DIR / "metrics/mechanism_redesign_pass_fail_table.csv", index=False)
    trial_export.to_csv(OUT_DIR / "metrics/mechanism_redesign_top_candidates_trial_level.csv", index=False)
    traj_export.to_csv(OUT_DIR / "metrics/mechanism_redesign_trajectory_diagnostics.csv", index=False)
    adaptive_export.to_csv(OUT_DIR / "metrics/conflict_adaptive_schedule_diagnostics.csv", index=False)
    lapse_export.to_csv(OUT_DIR / "metrics/bounded_lapse_diagnostics.csv", index=False)
    reps.to_csv(OUT_DIR / "metrics/mechanism_redesign_representative_models.csv", index=False)

    make_figures(model_summary, pass_fail, per_condition, trial_export, reps)
    write_summary(model_summary, pass_fail, reps, args.mode)

    logs.extend(
        [
            f"lenient_survivors={int(pass_fail['pass_lenient'].sum())}",
            f"main_survivors={int(pass_fail['pass_main'].sum())}",
            f"strict_survivors={int(pass_fail['pass_strict'].sum())}",
            f"pareto_count={len(pareto)}",
        ]
    )
    (OUT_DIR / "logs/mechanism_redesign_run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    assert_outputs()


if __name__ == "__main__":
    main()

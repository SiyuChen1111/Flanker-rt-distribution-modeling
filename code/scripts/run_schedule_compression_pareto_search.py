#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout, rt_bins  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_congruent_ww_dynamics_diagnostic import parse_group_params  # noqa: E402
from run_gated_readout_simulation import GROUPS, GROUP_LABEL, state_metrics  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule, normalize_layers, raw_layer_arrays  # noqa: E402
from run_representative_extreme_age_subset_fitting import apply_group_t0, load_trial_cache, subset_cache  # noqa: E402
from analyze_layerwise_evidence_ww import run_ww  # noqa: E402

BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
NAT_DIR = BASE_DIR / "natural_evidence_dynamics_optimization"
READOUT_DIR = BASE_DIR / "readout_choice_uncertainty_mechanism_comparison"
OUT_DIR = BASE_DIR / "schedule_compression_pareto_search"
DT = 0.01
TIME_STEPS = 80
SEED = 20260530
NOISE_SEED = 20260601
NORMALIZATION = "per_layer_gap_scale"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Schedule compression local Pareto search with time+gap noise retuning.")
    p.add_argument("--mode", choices=["coarse", "fine", "both"], default="coarse")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def ensure_dirs() -> Dict[str, Path]:
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


def rmse(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else math.nan


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


def load_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float], Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    best_dir = BASE_DIR / "best_model_R5_combined_best/results"
    group_params, t0_mean, t0_sd = parse_group_params(best_dir / "best_model_parameter_estimates.csv")
    cache = load_trial_cache(BASE_DIR)
    norm = normalize_layers(raw_layer_arrays(cache), NORMALIZATION)
    nat_rank = pd.read_csv(NAT_DIR / "metrics/natural_dynamics_model_ranking.csv")
    human = pd.read_csv(READOUT_DIR / "metrics/human_reference_rt_error_metrics.csv")
    human = human[human["source"].eq("human")].copy()
    return cache, group_params, t0_mean, t0_sd, norm, nat_rank, human


def write_inventory(nat_rank: pd.DataFrame) -> None:
    best = nat_rank.iloc[0]
    lines = [
        "# Schedule compression Pareto input inventory",
        "",
        "## Inputs used",
        "",
        "- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_model_ranking.csv`: previous round ranking and best schedule family.",
        "- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_model_comparison_summary.csv`: previous round group/condition diagnostics.",
        "- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_trial_level_predictions.csv`: previous round trial-level outputs.",
        "- `natural_evidence_dynamics_optimization/metrics/natural_dynamics_trajectory_diagnostics.csv`: previous round trajectory summaries.",
        "- `natural_evidence_dynamics_optimization/summaries/natural_evidence_dynamics_optimization_summary.md`: prior interpretation.",
        "- `evidence_cache/representative_subset_layerwise_evidence.npz`: cached layerwise evidence.",
        "- `best_model_R5_combined_best/results/best_model_parameter_estimates.csv`: group-specific WW parameters.",
        "- `fitting/representative_trial_level_predictions.csv`: human trial metadata and RTs.",
        "- `readout_choice_uncertainty_mechanism_comparison/metrics/readout_choice_model_ranking.csv`: prior time+gap uncertainty parameters.",
        "- `readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv`: human reference metrics.",
        "",
        "## Previous best schedule compression",
        "",
        f"- Previous best schedule model: `{best['model_family']} / {best['model_config_id']}`.",
        f"- Previous best parameters: {best['parameter_setting']}.",
        "- Previous failure point: incongruent repair was strong, but congruent fast errors were lost or weakened.",
        "",
        "## Why local Pareto search",
        "",
        "- The current question is no longer whether schedule compression can repair incongruent flanker over-selection. It can.",
        "- The open question is whether a less aggressive local region, combined with retuned time+gap choice noise, can recover congruent fast errors without undoing the incongruent repair.",
        "",
        "## What is not retrained",
        "",
        "- VGG is not retrained.",
        "- Image evidence is not re-extracted.",
        "- Earlier result folders are not overwritten.",
    ]
    (OUT_DIR / "summaries/schedule_compression_pareto_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_schedule_df(compression: float, late_shift_ms: int, transition_width: float, early_shorten_ms: int) -> pd.DataFrame:
    t = np.arange(TIME_STEPS, dtype=np.float32) / TIME_STEPS
    centers = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32) * compression
    centers = np.clip(centers, 0.03, 0.97)
    centers[3:] = np.clip(centers[3:] + late_shift_ms / 1000.0, 0.03, 0.97)
    centers[0] = max(0.03, centers[0] - early_shorten_ms / 1000.0)
    sigma = max(0.12 * transition_width, 0.03)
    basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / sigma) ** 2)
    basis_sum = basis.sum(axis=1, keepdims=True)
    basis_sum[basis_sum < 1e-6] = 1.0
    return pd.DataFrame(basis / basis_sum, columns=["conv3", "conv4", "conv5", "pooled", "final"])


def schedule_candidates(mode: str) -> List[Dict[str, Any]]:
    coarse = []
    for c in [0.40, 0.50, 0.60, 0.70]:
        for shift in [-50, -30, -10]:
            for tw in [0.70, 0.90, 1.10]:
                for ep in [0, 30, 50]:
                    coarse.append({"compression": c, "late_shift_ms": shift, "transition_width_scale": tw, "early_phase_shortening_ms": ep})
    fine = []
    for c in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for shift in [-60, -50, -40, -30, -20, -10, 0]:
            for tw in [0.70, 0.80, 0.90, 1.00, 1.10]:
                for ep in [0, 10, 20, 30, 40, 50]:
                    fine.append({"compression": c, "late_shift_ms": shift, "transition_width_scale": tw, "early_phase_shortening_ms": ep})
    if mode == "coarse":
        pool = coarse
    elif mode == "fine":
        pool = fine
    else:
        pool = coarse + fine
    out = []
    for item in pool:
        item = dict(item)
        item["schedule_config_id"] = f"c{item['compression']:.2f}_ls{item['late_shift_ms']}_tw{item['transition_width_scale']:.2f}_ep{item['early_phase_shortening_ms']}"
        out.append(item)
    uniq = []
    seen = set()
    for item in out:
        if item["schedule_config_id"] not in seen:
            seen.add(item["schedule_config_id"])
            uniq.append(item)
    return uniq


def shared_noise_grid() -> List[Dict[str, Any]]:
    out = []
    for sb in [0.0, 0.0005, 0.0010, 0.0020]:
        for st in [0.0, 0.0040, 0.0080, 0.0120, 0.0160]:
            for sg in [0.0, 0.0040, 0.0080, 0.0120]:
                for gs in [0.03, 0.05, 0.08]:
                    out.append({"noise_mode": "shared", "sigma_base": sb, "sigma_time": st, "sigma_gap": sg, "gap_scale": gs, "noise_config_id": f"sb{sb:.4f}_st{st:.4f}_sg{sg:.4f}_gs{gs:.2f}"})
    return out


def selected_old_age_specific_noise() -> Dict[str, Dict[str, float]]:
    rank = pd.read_csv(READOUT_DIR / "metrics/readout_choice_model_ranking.csv")
    best = rank[rank["model"].eq("M3_time_gap")].iloc[0]
    out = {}
    for group in GROUPS:
        out[group] = {
            "sigma_base": float(best[f"{group}_sigma_base"]),
            "sigma_time": float(best[f"{group}_sigma_time"]),
            "sigma_gap": float(best[f"{group}_sigma_gap"]),
            "gap_scale": float(best[f"{group}_gap_scale"]),
        }
    return out


def run_schedule_only(
    cache: Dict[str, np.ndarray],
    norm_layers: Dict[str, np.ndarray],
    group_params: Dict[str, Dict[str, float]],
    t0_mean: Dict[str, float],
    t0_sd: Dict[str, float],
    sched: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    traj_rows = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        layers = {k: v[mask] for k, v in norm_layers.items()}
        gp = group_params[group]
        schedule_df = make_schedule_df(sched["compression"], int(sched["late_shift_ms"]), float(sched["transition_width_scale"]), int(sched["early_phase_shortening_ms"]))
        mu = build_mu_schedule(layers, schedule_df, float(gp["evidence_gain"]))
        out = run_ww(
            mu,
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
                "target_label": gc["target_labels"].astype(int),
                "flanker_label": gc["flanker_labels"].astype(int),
                "response_label": gc["response_labels"].astype(int),
                "true_rt": gc["true_rt"].astype(float),
                "human_correct": gc["human_correct"].astype(bool),
                "congruency": pd.Series(gc["congruency"]).map({0: "congruent", 1: "incongruent"}).astype(str),
                "pred_choice": out["pred_choice"],
                "pred_rt": out["pred_rt"],
            }
        )
        cfg = ReadoutConfig("sustained_crossing", min_decision_time=float(gp["min_decision_time"]), sustained_k=int(gp["sustained_k"]), margin=float(gp["margin"]))
        base_df = apply_readout(base_df, out, cfg=cfg, threshold=float(gp["threshold"]), dt_ms=int(DT * 1000), t0_seconds=0.0)
        base_df = apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)
        base_df["model_correct"] = base_df["pred_choice"].to_numpy(int) == base_df["target_label"].to_numpy(int)
        traj = np.asarray(out["trajectory"], dtype=np.float32)
        target = base_df["target_label"].to_numpy(int)
        flanker = base_df["flanker_label"].to_numpy(int)
        steps = np.clip(np.rint(base_df["decision_time"].to_numpy(float) / DT).astype(int), 0, TIME_STEPS - 1)
        states = traj[np.arange(len(base_df)), steps, :]
        met = state_metrics(states, target, flanker)
        rows_idx = np.arange(len(base_df))[:, None]
        times = np.arange(TIME_STEPS)[None, :]
        target_vals = traj[rows_idx, times, target[:, None]]
        flanker_vals = traj[rows_idx, times, flanker[:, None]]
        masked = traj.copy()
        masked[np.arange(len(base_df))[:, None], np.arange(TIME_STEPS)[None, :], target[:, None]] = -np.inf
        other_max = masked.max(axis=2)
        first_gt_other = (target_vals > other_max).argmax(axis=1).astype(float)
        valid_gt = (target_vals > other_max).any(axis=1)
        first_gt_other[~valid_gt] = np.nan
        flanker_dom = np.maximum(flanker_vals - target_vals, 0.0)
        flanker_dur = (flanker_dom > 0).sum(axis=1) * DT
        early_flanker = (flanker_dom[:, : max(1, int(0.15 / DT))] > 0).mean(axis=1)
        late_target = (target_vals[:, int(0.30 / DT) :] - flanker_vals[:, int(0.30 / DT) :]).max(axis=1)
        for cong in ["congruent", "incongruent"]:
            part = base_df[base_df["congruency"].eq(cong)].copy()
            idx = part.index.to_numpy(int)
            rt = part["pred_rt"].to_numpy(float)
            correct = part["model_correct"].to_numpy(bool)
            human_rt = part["true_rt"].to_numpy(float)
            human_correct = part["human_correct"].to_numpy(bool)
            q_model = [safe_q(rt, p) for p in [0.1, 0.5, 0.9]]
            q_human = [safe_q(human_rt, p) for p in [0.1, 0.5, 0.9]]
            rows.append(
                {
                    "schedule_config_id": sched["schedule_config_id"],
                    "analysis_group": group,
                    "congruency": cong,
                    "noise_config_id": "baseline",
                    "noise_mode": "deterministic",
                    "overall_accuracy": safe_mean(correct.astype(float)),
                    "congruent_accuracy": safe_mean(correct.astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_accuracy": safe_mean(correct.astype(float)) if cong == "incongruent" else math.nan,
                    "congruent_error_rate": safe_mean((~correct).astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_error_rate": safe_mean((~correct).astype(float)) if cong == "incongruent" else math.nan,
                    "mean_rt": safe_mean(rt),
                    "rt_q10": safe_q(rt, 0.10),
                    "rt_q50": safe_q(rt, 0.50),
                    "rt_q90": safe_q(rt, 0.90),
                    "rt_distribution_similarity": corr_safe(q_model, q_human),
                    "error_rate_by_rt_bin_rmse": abs((1.0 - safe_mean(correct.astype(float))) - (1.0 - safe_mean(human_correct.astype(float)))),
                    "congruent_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]) if cong == "congruent" else math.nan,
                    "incongruent_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]) if cong == "incongruent" else math.nan,
                    "overall_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]),
                    "flanker_choice_proportion": safe_mean((part["pred_choice"].to_numpy(int) == part["flanker_label"].to_numpy(int)).astype(float)),
                    "target_recovery_time": safe_mean(first_gt_other[idx] * DT),
                    "target_rank_at_readout": safe_mean(met["target_rank"][idx]),
                    "signed_target_margin_at_readout": safe_mean(met["signed_target_margin"][idx]),
                    "s_target_at_readout": safe_mean(met["s_target"][idx]),
                    "s_flanker_at_readout": safe_mean(met["s_flanker"][idx]),
                    "s_other_max_at_readout": safe_mean(met["s_other_max"][idx]),
                    "gap_at_readout": safe_mean(met["gap"][idx]),
                    "flanker_dominance_duration": safe_mean(flanker_dur[idx]),
                    "early_flanker_dominance": safe_mean(early_flanker[idx]),
                    "late_target_recovery_strength": safe_mean(late_target[idx]),
                    "compression": sched["compression"],
                    "late_shift_ms": sched["late_shift_ms"],
                    "transition_width_scale": sched["transition_width_scale"],
                    "early_phase_shortening_ms": sched["early_phase_shortening_ms"],
                }
            )
            for split_name, split_mask in [("human_correct", part["human_correct"].to_numpy(bool)), ("human_error", ~part["human_correct"].to_numpy(bool)), ("model_correct", correct), ("model_error", ~correct)]:
                if not split_mask.any():
                    continue
                sel = idx[split_mask]
                for t in range(TIME_STEPS):
                    traj_rows.append(
                        {
                            "schedule_config_id": sched["schedule_config_id"],
                            "noise_config_id": "baseline",
                            "model_config_id": f"{sched['schedule_config_id']}__deterministic",
                            "analysis_group": group,
                            "congruency": cong,
                            "split": split_name,
                            "time": t * DT,
                            "s_target_mean": safe_mean(target_vals[sel, t]),
                            "s_flanker_mean": safe_mean(flanker_vals[sel, t]),
                            "s_other_max_mean": safe_mean(other_max[sel, t]),
                            "s_target_minus_flanker_mean": safe_mean((target_vals - flanker_vals)[sel, t]),
                            "s_target_minus_max_other_mean": safe_mean((target_vals - other_max)[sel, t]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(traj_rows)


def pick_promising_schedules(det_summary: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    agg = det_summary.groupby("schedule_config_id", as_index=False).agg(
        mean_incong=("incongruent_error_rate", "mean"),
        mean_cong_fast=("congruent_error_rt_minus_correct_rt", "mean"),
        mean_acc=("overall_accuracy", "mean"),
        mean_early=("early_flanker_dominance", "mean"),
        mean_flanker=("flanker_choice_proportion", "mean"),
        mean_tr=("target_recovery_time", "mean"),
    )
    agg["schedule_priority"] = (
        2.0 * agg["mean_incong"].fillna(1.0)
        + 0.5 * np.maximum(0.0, agg["mean_cong_fast"].fillna(0.0))
        + 0.5 * np.maximum(0.0, 0.20 - agg["mean_early"].fillna(0.0))
        + 0.25 * agg["mean_tr"].fillna(1.0)
    )
    return agg.sort_values("schedule_priority", kind="mergesort").head(top_n)


def evaluate_noise(
    cache: Dict[str, np.ndarray],
    norm_layers: Dict[str, np.ndarray],
    group_params: Dict[str, Dict[str, float]],
    t0_mean: Dict[str, float],
    t0_sd: Dict[str, float],
    sched: Dict[str, Any],
    noise: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        layers = {k: v[mask] for k, v in norm_layers.items()}
        gp = group_params[group]
        schedule_df = make_schedule_df(sched["compression"], int(sched["late_shift_ms"]), float(sched["transition_width_scale"]), int(sched["early_phase_shortening_ms"]))
        mu = build_mu_schedule(layers, schedule_df, float(gp["evidence_gain"]))
        out = run_ww(
            mu,
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
                "target_label": gc["target_labels"].astype(int),
                "flanker_label": gc["flanker_labels"].astype(int),
                "response_label": gc["response_labels"].astype(int),
                "true_rt": gc["true_rt"].astype(float),
                "human_correct": gc["human_correct"].astype(bool),
                "congruency": pd.Series(gc["congruency"]).map({0: "congruent", 1: "incongruent"}).astype(str),
                "pred_choice": out["pred_choice"],
                "pred_rt": out["pred_rt"],
            }
        )
        cfg = ReadoutConfig("sustained_crossing", min_decision_time=float(gp["min_decision_time"]), sustained_k=int(gp["sustained_k"]), margin=float(gp["margin"]))
        base_df = apply_readout(base_df, out, cfg=cfg, threshold=float(gp["threshold"]), dt_ms=int(DT * 1000), t0_seconds=0.0)
        base_df = apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)
        traj = np.asarray(out["trajectory"], dtype=np.float32)
        target = base_df["target_label"].to_numpy(int)
        flanker = base_df["flanker_label"].to_numpy(int)
        steps = np.clip(np.rint(base_df["decision_time"].to_numpy(float) / DT).astype(int), 0, TIME_STEPS - 1)
        states = traj[np.arange(len(base_df)), steps, :]
        met = state_metrics(states, target, flanker)
        earlyness = 1.0 - (steps * DT) / max(float((steps * DT).max()), 1e-9)
        sigma = noise["sigma_base"] + noise["sigma_time"] * earlyness + noise["sigma_gap"] * np.exp(-np.clip(met["gap"], 0, None) / max(noise["gap_scale"], 1e-9))
        rng = np.random.default_rng(NOISE_SEED + abs(hash((sched["schedule_config_id"], noise["noise_config_id"], group))) % 1000000)
        stoch = (states + rng.normal(0.0, sigma[:, None], size=states.shape)).argmax(axis=1)
        correct = stoch == target
        for cong in ["congruent", "incongruent"]:
            part = base_df[base_df["congruency"].eq(cong)].copy()
            idx = part.index.to_numpy(int)
            mask_c = base_df["congruency"].eq(cong).to_numpy()
            rt = part["pred_rt"].to_numpy(float)
            c = correct[mask_c]
            human_rt = part["true_rt"].to_numpy(float)
            human_correct = part["human_correct"].to_numpy(bool)
            rows.append(
                {
                    "schedule_config_id": sched["schedule_config_id"],
                    "noise_config_id": noise["noise_config_id"],
                    "noise_mode": noise["noise_mode"],
                    "analysis_group": group,
                    "congruency": cong,
                    "overall_accuracy": safe_mean(c.astype(float)),
                    "congruent_accuracy": safe_mean(c.astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_accuracy": safe_mean(c.astype(float)) if cong == "incongruent" else math.nan,
                    "congruent_error_rate": safe_mean((~c).astype(float)) if cong == "congruent" else math.nan,
                    "incongruent_error_rate": safe_mean((~c).astype(float)) if cong == "incongruent" else math.nan,
                    "mean_rt": safe_mean(rt),
                    "rt_q10": safe_q(rt, 0.10),
                    "rt_q50": safe_q(rt, 0.50),
                    "rt_q90": safe_q(rt, 0.90),
                    "rt_distribution_similarity": corr_safe([safe_q(rt, p) for p in [0.1, 0.5, 0.9]], [safe_q(human_rt, p) for p in [0.1, 0.5, 0.9]]),
                    "error_rate_by_rt_bin_rmse": abs((1.0 - safe_mean(c.astype(float))) - (1.0 - safe_mean(human_correct.astype(float)))),
                    "congruent_error_rt_minus_correct_rt": safe_mean(rt[~c]) - safe_mean(rt[c]) if cong == "congruent" else math.nan,
                    "incongruent_error_rt_minus_correct_rt": safe_mean(rt[~c]) - safe_mean(rt[c]) if cong == "incongruent" else math.nan,
                    "overall_error_rt_minus_correct_rt": safe_mean(rt[~c]) - safe_mean(rt[c]),
                    "target_choice_proportion": safe_mean((stoch[mask_c] == target[mask_c]).astype(float)),
                    "flanker_choice_proportion": safe_mean((stoch[mask_c] == flanker[mask_c]).astype(float)),
                    "other_choice_proportion": safe_mean(((stoch[mask_c] != target[mask_c]) & (stoch[mask_c] != flanker[mask_c])).astype(float)),
                    "incongruent_flanker_choice_proportion": safe_mean((stoch[mask_c] == flanker[mask_c]).astype(float)) if cong == "incongruent" else math.nan,
                    "target_recovery_time": safe_mean(np.nan * np.ones(len(idx))),
                    "target_rank_at_readout": safe_mean(met["target_rank"][idx]),
                    "signed_target_margin_at_readout": safe_mean(met["signed_target_margin"][idx]),
                    "s_target_at_readout": safe_mean(met["s_target"][idx]),
                    "s_flanker_at_readout": safe_mean(met["s_flanker"][idx]),
                    "s_other_max_at_readout": safe_mean(met["s_other_max"][idx]),
                    "gap_at_readout": safe_mean(met["gap"][idx]),
                    "compression": sched["compression"],
                    "late_shift_ms": sched["late_shift_ms"],
                    "transition_width_scale": sched["transition_width_scale"],
                    "early_phase_shortening_ms": sched["early_phase_shortening_ms"],
                    "sigma_base": noise["sigma_base"],
                    "sigma_time": noise["sigma_time"],
                    "sigma_gap": noise["sigma_gap"],
                    "gap_scale": noise["gap_scale"],
                }
            )
    return pd.DataFrame(rows)


def add_scores(summary: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    human = human.rename(
        columns={
            "overall_accuracy": "human_overall_accuracy",
            "congruent_error_rate": "human_congruent_error_rate",
            "incongruent_error_rate": "human_incongruent_error_rate",
            "congruent_error_rt_minus_correct_rt": "human_congruent_error_rt_minus_correct_rt",
            "incongruent_error_rt_minus_correct_rt": "human_incongruent_error_rt_minus_correct_rt",
        }
    )
    out = summary.merge(
        human[["analysis_group", "human_overall_accuracy", "human_congruent_error_rate", "human_incongruent_error_rate", "human_congruent_error_rt_minus_correct_rt", "human_incongruent_error_rt_minus_correct_rt"]],
        on="analysis_group",
        how="left",
    )
    out["incongruent_repair_score"] = (
        (out["incongruent_error_rate"].fillna(0.0) - out["human_incongruent_error_rate"].fillna(0.08)).abs()
        + 0.5 * out["incongruent_flanker_choice_proportion"].fillna(0.0)
        + 0.5 * (out["overall_accuracy"] - out["human_overall_accuracy"].fillna(0.95)).abs()
    )
    out["congruent_fast_error_score"] = (
        (out["congruent_error_rate"].fillna(0.0) - out["human_congruent_error_rate"].fillna(0.02)).abs()
        + (out["congruent_error_rt_minus_correct_rt"].fillna(0.0) - out["human_congruent_error_rt_minus_correct_rt"].fillna(-0.05)).abs()
        + np.maximum(0.0, out["congruent_error_rt_minus_correct_rt"].fillna(0.0))
    )
    out["rt_dynamics_preservation_score"] = (
        0.5 * (1.0 - out["rt_distribution_similarity"].fillna(0.0))
        + 0.5 * out["error_rate_by_rt_bin_rmse"].fillna(1.0)
        + np.maximum(0.0, 0.20 - out.get("early_flanker_dominance", pd.Series(0.2, index=out.index)).fillna(0.0))
    )
    out["naturalness_penalty"] = (
        np.maximum(0.0, out.get("early_flanker_dominance", pd.Series(0.2, index=out.index)).fillna(0.0) < 0.15).astype(float)
        + np.maximum(0.0, out["congruent_error_rate"].fillna(0.0).eq(0.0)).astype(float)
    )
    out["combined_score"] = out["incongruent_repair_score"] + out["congruent_fast_error_score"] + out["rt_dynamics_preservation_score"] + out["naturalness_penalty"]
    out["flag_high_incongruent_error"] = out["incongruent_error_rate"] > 0.25
    out["flag_low_accuracy"] = out["overall_accuracy"] < 0.85
    out["flag_no_congruent_errors"] = out["congruent_error_rate"].fillna(0.0) == 0.0
    out["flag_no_congruent_fast_error"] = out["congruent_error_rt_minus_correct_rt"].fillna(0.0) >= 0.0
    out["flag_congruent_too_many_errors"] = out["congruent_error_rate"].fillna(0.0) > 0.05
    out["flag_lost_conflict_dynamics"] = out.get("early_flanker_dominance", pd.Series(0.0, index=out.index)).fillna(0.0) < 0.15
    out["flag_excessive_flanker_dominance"] = out.get("flanker_dominance_duration", pd.Series(0.0, index=out.index)).fillna(0.0) > 0.35
    out["flag_rt_distribution_broken"] = out["rt_distribution_similarity"].fillna(0.0) < 0.70
    out["flag_excessive_noise"] = False
    out["flag_unrealistic_perfect_accuracy"] = out["overall_accuracy"] > 0.995
    return out


def aggregate_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in summary.groupby(["schedule_config_id", "noise_config_id", "noise_mode"], sort=False):
        row = {
            "schedule_config_id": keys[0],
            "noise_config_id": keys[1],
            "noise_mode": keys[2],
            "combined_score": safe_mean(part["combined_score"]),
            "incongruent_repair_score": safe_mean(part["incongruent_repair_score"]),
            "congruent_fast_error_score": safe_mean(part["congruent_fast_error_score"]),
            "rt_dynamics_preservation_score": safe_mean(part["rt_dynamics_preservation_score"]),
            "naturalness_penalty": safe_mean(part["naturalness_penalty"]),
            "schedule_parameters": json.dumps(
                {
                    "compression": float(part["compression"].iloc[0]),
                    "late_shift_ms": int(part["late_shift_ms"].iloc[0]),
                    "transition_width_scale": float(part["transition_width_scale"].iloc[0]),
                    "early_phase_shortening_ms": int(part["early_phase_shortening_ms"].iloc[0]),
                },
                sort_keys=True,
            ),
            "noise_parameters": json.dumps(
                {
                    "sigma_base": float(part["sigma_base"].iloc[0]) if "sigma_base" in part else None,
                    "sigma_time": float(part["sigma_time"].iloc[0]) if "sigma_time" in part else None,
                    "sigma_gap": float(part["sigma_gap"].iloc[0]) if "sigma_gap" in part else None,
                    "gap_scale": float(part["gap_scale"].iloc[0]) if "gap_scale" in part else None,
                },
                sort_keys=True,
            ),
        }
        for group in GROUPS:
            g = part[part["analysis_group"].eq(group)]
            row[f"{group}_overall_accuracy"] = safe_mean(g["overall_accuracy"])
            row[f"{group}_congruent_error_rate"] = safe_mean(g["congruent_error_rate"])
            row[f"{group}_incongruent_error_rate"] = safe_mean(g["incongruent_error_rate"])
            row[f"{group}_congruent_error_rt_minus_correct_rt"] = safe_mean(g["congruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_error_rt_minus_correct_rt"] = safe_mean(g["incongruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_flanker_choice_proportion"] = safe_mean(g["incongruent_flanker_choice_proportion"])
            row[f"{group}_target_recovery_time"] = safe_mean(g.get("target_recovery_time", pd.Series(np.nan)))
        for flag in [
            "flag_high_incongruent_error",
            "flag_low_accuracy",
            "flag_no_congruent_errors",
            "flag_no_congruent_fast_error",
            "flag_congruent_too_many_errors",
            "flag_lost_conflict_dynamics",
            "flag_excessive_flanker_dominance",
            "flag_rt_distribution_broken",
            "flag_excessive_noise",
            "flag_unrealistic_perfect_accuracy",
        ]:
            row[flag] = bool(part[flag].any())
        rows.append(row)
    rank = pd.DataFrame(rows).sort_values("combined_score", kind="mergesort")
    return rank


def pareto_front(rank: pd.DataFrame) -> pd.DataFrame:
    df = rank.copy().reset_index(drop=True)
    feats = df[["incongruent_repair_score", "congruent_fast_error_score", "rt_dynamics_preservation_score", "naturalness_penalty"]].to_numpy(float)
    is_pareto = np.ones(len(df), dtype=bool)
    pareto_rank = np.ones(len(df), dtype=int)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if np.all(feats[j] <= feats[i]) and np.any(feats[j] < feats[i]):
                is_pareto[i] = False
                break
    df["is_pareto_optimal"] = is_pareto
    df["pareto_rank"] = np.where(is_pareto, 1, 2)
    region = []
    for _, r in df.iterrows():
        if r["is_pareto_optimal"] and r["incongruent_repair_score"] == df["incongruent_repair_score"].min():
            region.append("incongruent_repair_best")
        elif r["is_pareto_optimal"] and r["congruent_fast_error_score"] == df["congruent_fast_error_score"].min():
            region.append("fast_error_best")
        elif r["is_pareto_optimal"] and r["rt_dynamics_preservation_score"] == df["rt_dynamics_preservation_score"].min():
            region.append("rt_preservation_best")
        elif r["is_pareto_optimal"]:
            region.append("balanced")
        elif r["flag_no_congruent_fast_error"]:
            region.append("high_accuracy_but_no_fast_error")
        elif r["flag_high_incongruent_error"]:
            region.append("fast_error_but_incongruent_bad")
        else:
            region.append("not_recommended")
    df["tradeoff_region"] = region
    return df


def plot_outputs(rank: pd.DataFrame, pareto: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    top = rank.head(15).iloc[::-1]
    ax.barh(np.arange(len(top)), top["combined_score"], color="#4C78A8")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels((top["schedule_config_id"] + "\n" + top["noise_config_id"]).tolist(), fontsize=7)
    ax.set_xlabel("Combined score")
    style_ax(ax)
    save_fig(fig, "schedule_local_search_ranking_overview")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rank["incongruent_repair_score"], rank["congruent_fast_error_score"], c=rank[[f"{GROUPS[0]}_target_recovery_time", f"{GROUPS[1]}_target_recovery_time"]].mean(axis=1), s=40, cmap="viridis", alpha=0.6)
    p = pareto[pareto["is_pareto_optimal"]]
    ax.scatter(p["incongruent_repair_score"], p["congruent_fast_error_score"], facecolors="none", edgecolors="black", s=90)
    ax.set_xlabel("Incongruent repair score")
    ax.set_ylabel("Congruent fast-error score")
    style_ax(ax)
    save_fig(fig, "pareto_front_incongruent_vs_fast_error")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rank[f"{GROUPS[0]}_incongruent_flanker_choice_proportion"], rank[f"{GROUPS[0]}_congruent_error_rt_minus_correct_rt"], alpha=0.5, label="Young")
    ax.scatter(rank[f"{GROUPS[1]}_incongruent_flanker_choice_proportion"], rank[f"{GROUPS[1]}_congruent_error_rt_minus_correct_rt"], alpha=0.5, label="Older")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Incongruent flanker choice proportion")
    ax.set_ylabel("Congruent error RT - correct RT")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    save_fig(fig, "tradeoff_flanker_overselection_vs_congruent_fast_error")


def write_summary(rank: pd.DataFrame, pareto: pd.DataFrame, n_sched: int, n_models: int, mode: str) -> None:
    best_incong = rank.sort_values("incongruent_repair_score", kind="mergesort").iloc[0]
    best_fast = rank.sort_values("congruent_fast_error_score", kind="mergesort").iloc[0]
    balanced = pareto[pareto["tradeoff_region"].eq("balanced")]
    best_balanced = balanced.iloc[0] if not balanced.empty else rank.iloc[0]
    lines = [
        "# Schedule compression Pareto search summary",
        "",
        f"- Run mode: `{mode}`.",
        f"- Schedule candidates tested: {n_sched}.",
        f"- Schedule × noise candidates tested: {n_models}.",
        f"- Pareto-optimal candidates found: {int(pareto['is_pareto_optimal'].sum())}.",
        f"- Best incongruent-repair candidate: `{best_incong['schedule_config_id']} + {best_incong['noise_config_id']}`.",
        f"- Best fast-error-preservation candidate: `{best_fast['schedule_config_id']} + {best_fast['noise_config_id']}`.",
        f"- Best balanced candidate: `{best_balanced['schedule_config_id']} + {best_balanced['noise_config_id']}`.",
        "",
        "## Interpretation",
        "",
        "- This search treats incongruent repair, congruent fast-error preservation, and RT/dynamics preservation as separate objectives rather than collapsing everything into one scalar target.",
        "- A strong Pareto front means the remaining trade-off is real, not just a ranking artifact.",
        "- Any age-specific noise improvement should be treated as exploratory unless it clearly generalizes across both groups and preserves the broader dynamics profile.",
    ]
    (OUT_DIR / "summaries/schedule_compression_pareto_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_script() -> None:
    src = Path(__file__).resolve()
    dst = OUT_DIR / "scripts" / "run_schedule_compression_pareto_search.py"
    if src != dst:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    copy_script()
    cache, group_params, t0_mean, t0_sd, norm_layers, nat_rank, human = load_inputs()
    write_inventory(nat_rank)
    schedules = schedule_candidates(args.mode)
    det_rows = []
    traj_rows = []
    logs = [f"mode={args.mode}", f"n_schedule_candidates={len(schedules)}"]
    for i, sched in enumerate(schedules, start=1):
        det, traj = run_schedule_only(cache, norm_layers, group_params, t0_mean, t0_sd, sched)
        det_rows.append(det)
        traj_rows.append(traj)
        logs.append(f"schedule {i}/{len(schedules)} {sched['schedule_config_id']} done")
    det_summary = pd.concat(det_rows, ignore_index=True)
    promising = pick_promising_schedules(det_summary, top_n=12)
    promising_ids = set(promising["schedule_config_id"].tolist())
    logs.append(f"promising_schedule_count={len(promising_ids)}")
    noise_rows = []
    grid = shared_noise_grid()
    for sid in promising_ids:
        sched = next(s for s in schedules if s["schedule_config_id"] == sid)
        for j, noise in enumerate(grid, start=1):
            noise_rows.append(evaluate_noise(cache, norm_layers, group_params, t0_mean, t0_sd, sched, noise))
        logs.append(f"noise_search_done {sid} n_noise={len(grid)}")
    noise_summary = pd.concat(noise_rows, ignore_index=True) if noise_rows else pd.DataFrame()
    full_summary = pd.concat([det_summary, noise_summary], ignore_index=True)
    full_summary = add_scores(full_summary, human)
    rank = aggregate_ranking(full_summary)
    pareto = pareto_front(rank)
    rank = pareto.copy()
    top_ids = set(rank.head(10)[["schedule_config_id", "noise_config_id"]].apply(tuple, axis=1).tolist())
    pareto_ids = set(rank[rank["is_pareto_optimal"]][["schedule_config_id", "noise_config_id"]].apply(tuple, axis=1).tolist())
    keep_ids = top_ids | pareto_ids | {(s, "baseline") for s in det_summary["schedule_config_id"].unique()}
    det_summary.to_csv(OUT_DIR / "metrics/schedule_compression_local_search_summary.csv", index=False)
    rank.to_csv(OUT_DIR / "metrics/schedule_compression_local_search_ranking.csv", index=False)
    rank[rank["is_pareto_optimal"]].to_csv(OUT_DIR / "metrics/schedule_compression_pareto_front.csv", index=False)
    top_det = det_summary[det_summary[["schedule_config_id", "noise_config_id"]].apply(tuple, axis=1).isin(keep_ids)].copy()
    top_det.to_csv(OUT_DIR / "metrics/schedule_compression_top_candidates_trial_level.csv", index=False)
    traj_all = pd.concat(traj_rows, ignore_index=True)
    traj_all.to_csv(OUT_DIR / "metrics/schedule_compression_trajectory_diagnostics.csv", index=False)
    plot_outputs(rank, rank[rank["is_pareto_optimal"]])
    write_summary(rank, rank[rank["is_pareto_optimal"]], len(schedules), len(rank), args.mode)
    (OUT_DIR / "logs/schedule_compression_pareto_search_run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

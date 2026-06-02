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
from run_gated_readout_simulation import GROUPS, GROUP_LABEL, state_metrics  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule, normalize_layers, raw_layer_arrays, schedule_weights  # noqa: E402
from run_representative_extreme_age_subset_fitting import apply_group_t0, load_trial_cache, subset_cache  # noqa: E402
from run_congruent_ww_dynamics_diagnostic import parse_group_params  # noqa: E402
from analyze_layerwise_evidence_ww import run_ww  # noqa: E402

BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
READOUT_DIR = BASE_DIR / "readout_choice_uncertainty_mechanism_comparison"
OUT_DIR = BASE_DIR / "natural_evidence_dynamics_optimization"
DT = 0.01
TIME_STEPS = 80
SEED = 20260530
NOISE_SEED = 20260601
NORMALIZATION = "per_layer_gap_scale"
BASE_SCHEDULE = "natural_smooth_5stage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Natural evidence-dynamics optimization on representative subset.")
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


def selected_time_gap_params() -> Dict[str, Dict[str, float]]:
    ranking = pd.read_csv(READOUT_DIR / "metrics/readout_choice_model_ranking.csv")
    best = ranking[ranking["model"].eq("M3_time_gap")].iloc[0]
    out: Dict[str, Dict[str, float]] = {}
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


def load_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
    best_dir = BASE_DIR / "best_model_R5_combined_best/results"
    group_params, t0_mean, t0_sd = parse_group_params(best_dir / "best_model_parameter_estimates.csv")
    cache = load_trial_cache(BASE_DIR)
    raw = raw_layer_arrays(cache)
    norm = normalize_layers(raw, NORMALIZATION)
    return cache, group_params, t0_mean, t0_sd, norm


def write_input_inventory(cache: Dict[str, np.ndarray]) -> None:
    lines = [
        "# Natural evidence-dynamics input inventory",
        "",
        "## Inputs used",
        "",
        "- `evidence_cache/representative_subset_layerwise_evidence.npz`: layerwise evidence arrays (`evidence_conv3`, `evidence_conv4`, `evidence_conv5`, `evidence_pooled`, `evidence_final`) and trial-to-stimulus ids.",
        "- `best_model_R5_combined_best/results/best_model_parameter_estimates.csv`: current R5 group-specific WW and readout parameters.",
        "- `fitting/representative_trial_level_predictions.csv`: trial metadata, human RT, human response, congruency, current model RT.",
        "- `fitting/representative_best_model_mechanism_trial_level.csv`: readout-stage mechanism diagnostics and target recovery fields.",
        "- `readout_choice_uncertainty_mechanism_comparison/metrics/*`: current time+gap uncertainty parameters and diagnostics.",
        "- `readout_choice_uncertainty_mechanism_comparison/summaries/*`: prior diagnostic context for gating and trajectory viability.",
        "",
        "## Key fields",
        "",
        "- Trial metadata: `row_index`, `analysis_group`, `target_label`, `flanker_label`, `response_label`, `true_rt`, `congruency`.",
        "- Layerwise evidence: one four-channel vector per layer per stimulus.",
        "- Existing behavior outputs: `pred_rt`, `decision_time`, `human_correct`, `model_correct`.",
        "",
        "## Reconstruction viability",
        "",
        "- Trial-level WW trajectories can be reconstructed from cached layerwise evidence plus saved R5 parameters.",
        "- Different layer-to-time schedules can be implemented by modifying the schedule weights before WW input is built.",
        "- Attention gain, flanker decay, and online conflict control can all be applied to the time-varying WW input without retraining VGG or re-extracting image evidence.",
        "",
        "## Limits",
        "",
        "- There is no saved per-trial full trajectory file on disk; trajectories are reconstructed rather than loaded directly.",
        "- Human correctness is available for evaluation only and is not used as model input.",
        "- This round does not retrain VGG, does not re-extract image features, and does not overwrite earlier result directories.",
    ]
    (OUT_DIR / "summaries/natural_evidence_dynamics_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_schedule_df() -> pd.DataFrame:
    return schedule_weights(BASE_SCHEDULE, TIME_STEPS)


def schedule_variant(
    compression: float = 1.0,
    late_shift_steps: int = 0,
    transition_scale: float = 1.0,
    early_shorten_steps: int = 0,
) -> pd.DataFrame:
    base = base_schedule_df().copy()
    t = np.arange(TIME_STEPS, dtype=np.float32)
    centers = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32) * compression
    centers = np.clip(centers, 0.03, 0.97)
    centers[3:] = np.clip(centers[3:] + late_shift_steps / TIME_STEPS, 0.03, 0.97)
    centers[0] = max(0.03, centers[0] - early_shorten_steps / TIME_STEPS)
    sigma = 0.12 * transition_scale
    sigma = max(sigma, 0.03)
    basis = np.exp(-0.5 * ((t[:, None] / TIME_STEPS - centers[None, :]) / sigma) ** 2)
    basis_sum = basis.sum(axis=1, keepdims=True)
    basis_sum[basis_sum < 1e-6] = 1.0
    weights = basis / basis_sum
    out = pd.DataFrame(weights, columns=base.columns)
    return out


def apply_dynamics(
    mu: np.ndarray,
    cache: Dict[str, np.ndarray],
    family: str,
    params: Dict[str, float],
) -> np.ndarray:
    out = np.array(mu, copy=True)
    target = cache["target_labels"].astype(np.int64)
    flanker = cache["flanker_labels"].astype(np.int64)
    rows = np.arange(out.shape[0])
    times = np.arange(out.shape[1], dtype=np.float32) * DT
    time_grid = times[None, :]

    if family in {"M2_attention_gain_ramp", "M4_attention_gain_plus_flanker_decay", "M6_combined_natural_dynamics"}:
        A = float(params.get("A", 0.0))
        onset = float(params.get("t_onset", 0.10))
        tau = float(params.get("tau", 0.04))
        gain = 1.0 + A * (1.0 / (1.0 + np.exp(-np.clip((time_grid - onset) / max(tau, 1e-6), -60, 60))))
        out[rows[:, None], np.arange(out.shape[1])[None, :], target[:, None]] *= gain

    if family in {"M3_flanker_decay", "M4_attention_gain_plus_flanker_decay", "M6_combined_natural_dynamics"}:
        D = float(params.get("D", 0.0))
        onset = float(params.get("t_decay_onset", params.get("t_onset", 0.10)))
        tau = float(params.get("tau", 0.04))
        decay = 1.0 - D * (1.0 / (1.0 + np.exp(-np.clip((time_grid - onset) / max(tau, 1e-6), -60, 60))))
        decay = np.clip(decay, 0.5, 1.2)
        out[rows[:, None], np.arange(out.shape[1])[None, :], flanker[:, None]] *= decay

    if family == "M5_conflict_dependent_control":
        C = float(params.get("C", 0.1))
        theta = float(params.get("theta", 0.0))
        temp = float(params.get("temp", 0.02))
        control_decay = float(params.get("control_decay", 1.0))
        target_vals = out[rows[:, None], np.arange(out.shape[1])[None, :], target[:, None]]
        flanker_vals = out[rows[:, None], np.arange(out.shape[1])[None, :], flanker[:, None]]
        conflict = np.maximum(flanker_vals - target_vals, 0.0)
        control = C * (1.0 / (1.0 + np.exp(-np.clip((conflict - theta) / max(temp, 1e-6), -60, 60))))
        out[rows[:, None], np.arange(out.shape[1])[None, :], target[:, None]] *= 1.0 + control
        out[rows[:, None], np.arange(out.shape[1])[None, :], flanker[:, None]] *= np.clip(1.0 - control_decay * control, 0.4, 1.0)

    return out


def run_candidate(
    cache: Dict[str, np.ndarray],
    norm_layers: Dict[str, np.ndarray],
    group_params: Dict[str, Dict[str, float]],
    t0_mean: Dict[str, float],
    t0_sd: Dict[str, float],
    readout_noise: Dict[str, Dict[str, float]],
    family: str,
    config_id: str,
    schedule_params: Dict[str, float],
    dynamics_params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trials = []
    all_traj_rows = []
    summary_rows = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        group_layers = {k: v[mask] for k, v in norm_layers.items()}
        gp = group_params[group]
        sched = schedule_variant(
            compression=float(schedule_params.get("compression", 1.0)),
            late_shift_steps=int(schedule_params.get("late_shift_steps", 0)),
            transition_scale=float(schedule_params.get("transition_scale", 1.0)),
            early_shorten_steps=int(schedule_params.get("early_shorten_steps", 0)),
        )
        mu = build_mu_schedule(group_layers, sched, float(gp["evidence_gain"])).numpy()
        mu = apply_dynamics(mu, gc, family, dynamics_params)
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
        trial = pd.DataFrame(
            {
                "trial_id": gc["row_indices"].astype(int),
                "analysis_group": gc["analysis_group"].astype(str),
                "congruency": pd.Series(gc["congruency"]).map({0: "congruent", 1: "incongruent"}).astype(str),
                "target_label": gc["target_labels"].astype(int),
                "flanker_label": gc["flanker_labels"].astype(int),
                "human_correct": gc["human_correct"].astype(bool),
                "true_rt": gc["true_rt"].astype(float),
                "human_rt": gc["true_rt"].astype(float),
                "model_family": family,
                "model_config_id": config_id,
                "response_label": gc["response_labels"].astype(int),
            }
        )
        base_df = trial.rename(columns={"true_rt": "true_rt"}).copy()
        base_df["pred_choice"] = out["pred_choice"]
        base_df["pred_rt"] = out["pred_rt"]
        base_df["model_correct"] = base_df["pred_choice"].to_numpy(int) == base_df["target_label"].to_numpy(int)
        readout_cfg = ReadoutConfig(
            "sustained_crossing",
            min_decision_time=float(gp["min_decision_time"]),
            sustained_k=int(gp["sustained_k"]),
            margin=float(gp["margin"]),
        )
        base_df = apply_readout(base_df, out, cfg=readout_cfg, threshold=float(gp["threshold"]), dt_ms=int(DT * 1000), t0_seconds=0.0)
        base_df = apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)

        traj = np.asarray(out["trajectory"], dtype=np.float32)
        readout_steps = np.rint(base_df["decision_time"].to_numpy(float) / DT).astype(int)
        readout_steps = np.clip(readout_steps, 0, TIME_STEPS - 1)
        states = traj[np.arange(len(base_df)), readout_steps, :]
        met = state_metrics(states, base_df["target_label"].to_numpy(int), base_df["flanker_label"].to_numpy(int))
        deterministic_choice = states.argmax(axis=1)
        deterministic_correct = deterministic_choice == base_df["target_label"].to_numpy(int)
        target = base_df["target_label"].to_numpy(int)
        flanker = base_df["flanker_label"].to_numpy(int)
        rows = np.arange(len(base_df))[:, None]
        times = np.arange(TIME_STEPS)[None, :]
        target_vals = traj[rows, times, target[:, None]]
        flanker_vals = traj[rows, times, flanker[:, None]]
        masked = traj.copy()
        masked[np.arange(len(base_df))[:, None], np.arange(TIME_STEPS)[None, :], target[:, None]] = -np.inf
        other_max = masked.max(axis=2)
        target_gt_flanker = target_vals > flanker_vals
        first_gt_flanker = target_gt_flanker.argmax(axis=1).astype(float)
        first_gt_flanker[~target_gt_flanker.any(axis=1)] = np.nan
        target_gt_other = target_vals > other_max
        first_gt_other = target_gt_other.argmax(axis=1).astype(float)
        first_gt_other[~target_gt_other.any(axis=1)] = np.nan
        target_rank1 = target_gt_other
        first_rank1 = target_rank1.argmax(axis=1).astype(float)
        first_rank1[~target_rank1.any(axis=1)] = np.nan
        flanker_dom = np.maximum(flanker_vals - target_vals, 0.0)
        flanker_dom_dur = (flanker_dom > 0).sum(axis=1) * DT
        early_flanker_dom = (flanker_dom[:, : max(1, int(0.15 / DT))] > 0).mean(axis=1)
        late_target_rec = (target_vals[:, int(0.30 / DT) :] - flanker_vals[:, int(0.30 / DT) :]).max(axis=1)
        max_post_margin = (target_vals - other_max).max(axis=1)

        deterministic_trial = pd.DataFrame(
            {
                "trial_id": base_df["trial_id"].to_numpy(int),
                "analysis_group": group,
                "congruency": base_df["congruency"].to_numpy(str),
                "target_label": target,
                "flanker_label": flanker,
                "human_correct": base_df["human_correct"].to_numpy(bool),
                "true_rt": base_df["true_rt"].to_numpy(float),
                "model_family": family,
                "model_config_id": config_id,
                "deterministic_choice": deterministic_choice,
                "deterministic_correct": deterministic_correct,
                "readout_time": readout_steps * DT,
                "model_rt": base_df["pred_rt"].to_numpy(float),
                "target_recovery_time": first_gt_other * DT,
                "target_rank_at_readout": met["target_rank"],
                "signed_target_margin_at_readout": met["signed_target_margin"],
                "s_target_at_readout": met["s_target"],
                "s_flanker_at_readout": met["s_flanker"],
                "s_other_max_at_readout": met["s_other_max"],
                "target_first_rank1_time": first_rank1 * DT,
                "target_first_exceeds_flanker_time": first_gt_flanker * DT,
                "target_first_exceeds_max_other_time": first_gt_other * DT,
                "target_ever_rank1": np.isfinite(first_rank1),
                "target_ever_exceeds_flanker": np.isfinite(first_gt_flanker),
                "target_ever_exceeds_max_other": np.isfinite(first_gt_other),
                "maximum_post_readout_target_margin": max_post_margin,
                "flanker_dominance_duration": flanker_dom_dur,
                "early_flanker_dominance": early_flanker_dom,
                "late_target_recovery_strength": late_target_rec,
            }
        )

        # Add stochastic time+gap choice uncertainty.
        p = readout_noise[group]
        max_time = max(float(deterministic_trial["readout_time"].max()), 1e-9)
        earlyness = 1.0 - deterministic_trial["readout_time"].to_numpy(float) / max_time
        gap = np.clip(met["gap"], 0, None)
        sigma = p["sigma_base"] + p["sigma_time"] * earlyness + p["sigma_gap"] * np.exp(-gap / max(p["gap_scale"], 1e-9))
        rng = np.random.default_rng(NOISE_SEED + abs(hash((family, config_id, group))) % 1000000)
        stochastic_choice = (states + rng.normal(0.0, sigma[:, None], size=states.shape)).argmax(axis=1)
        deterministic_trial["stochastic_choice"] = stochastic_choice
        deterministic_trial["stochastic_correct"] = stochastic_choice == target
        deterministic_trial["choice_type"] = choice_type(stochastic_choice, target, flanker)
        all_trials.append(deterministic_trial)

        # Trajectory summary rows.
        for cong_name in ["congruent", "incongruent"]:
            cmask = deterministic_trial["congruency"].eq(cong_name).to_numpy()
            if not cmask.any():
                continue
            for correctness_name, corr_mask in [("human_correct", deterministic_trial["human_correct"].to_numpy(bool)), ("human_error", ~deterministic_trial["human_correct"].to_numpy(bool)), ("model_correct", deterministic_trial["stochastic_correct"].to_numpy(bool)), ("model_error", ~deterministic_trial["stochastic_correct"].to_numpy(bool))]:
                final_mask = cmask & corr_mask
                if not final_mask.any():
                    continue
                for t in range(TIME_STEPS):
                    out_row = {
                        "model_family": family,
                        "model_config_id": config_id,
                        "analysis_group": group,
                        "congruency": cong_name,
                        "split": correctness_name,
                        "time": t * DT,
                        "s_target_mean": safe_mean(target_vals[final_mask, t]),
                        "s_flanker_mean": safe_mean(flanker_vals[final_mask, t]),
                        "s_other_max_mean": safe_mean(other_max[final_mask, t]),
                        "s_target_minus_flanker_mean": safe_mean((target_vals - flanker_vals)[final_mask, t]),
                        "s_target_minus_max_other_mean": safe_mean((target_vals - other_max)[final_mask, t]),
                    }
                    all_traj_rows.append(out_row)

        # Summary rows by condition.
        for cong_name in ["congruent", "incongruent"]:
            part = deterministic_trial[deterministic_trial["congruency"].eq(cong_name)].copy()
            if part.empty:
                continue
            hmask = gc["congruency"] == (0 if cong_name == "congruent" else 1)
            human_rt = gc["true_rt"][hmask].astype(float)
            human_correct = gc["human_correct"][hmask].astype(bool)
            model_rt = part["model_rt"].to_numpy(float)
            det_correct = part["deterministic_correct"].to_numpy(bool)
            stoch_correct = part["stochastic_correct"].to_numpy(bool)
            err_rate_bins_model = rt_bins(
                pd.DataFrame(
                    {
                        "pred_rt": model_rt,
                        "true_rt": human_rt,
                        "model_correct": stoch_correct,
                        "human_correct": human_correct,
                        "congruency": np.full(len(part), 0 if cong_name == "congruent" else 1),
                        "pred_choice": part["stochastic_choice"].to_numpy(int),
                        "response_label": np.zeros(len(part), dtype=int),
                    }
                ),
                "tmp",
            )
            pivot = err_rate_bins_model.pivot_table(index="rt_bin", columns="source", values="error_rate")
            human_q = [safe_q(human_rt, p) for p in [0.1, 0.5, 0.9]]
            model_q = [safe_q(model_rt, p) for p in [0.1, 0.5, 0.9]]
            props = part["choice_type"].value_counts(normalize=True)
            summary_rows.append(
                {
                    "model_family": family,
                    "model_config_id": config_id,
                    "analysis_group": group,
                    "congruency": cong_name,
                    "parameter_setting": json.dumps({"schedule": schedule_params, "dynamics": dynamics_params}, sort_keys=True),
                    "overall_accuracy": safe_mean(stoch_correct.astype(float)),
                    "deterministic_accuracy": safe_mean(det_correct.astype(float)),
                    "congruent_error_rate": safe_mean((~stoch_correct).astype(float)) if cong_name == "congruent" else math.nan,
                    "incongruent_error_rate": safe_mean((~stoch_correct).astype(float)) if cong_name == "incongruent" else math.nan,
                    "mean_rt": safe_mean(model_rt),
                    "rt_q10": safe_q(model_rt, 0.10),
                    "rt_q50": safe_q(model_rt, 0.50),
                    "rt_q90": safe_q(model_rt, 0.90),
                    "rt_distribution_similarity_with_human": corr_safe(model_q, human_q),
                    "error_rate_by_rt_bin_rmse": rmse(pivot.get("model", np.array([])), pivot.get("human", np.array([]))),
                    "caf_like_slope": safe_mean(err_rate_bins_model[err_rate_bins_model["source"].eq("model")]["error_rate"].iloc[-1:]) - safe_mean(err_rate_bins_model[err_rate_bins_model["source"].eq("model")]["error_rate"].iloc[:1]),
                    "congruent_error_rt_minus_correct_rt": safe_mean(model_rt[~stoch_correct]) - safe_mean(model_rt[stoch_correct]) if cong_name == "congruent" else math.nan,
                    "incongruent_error_rt_minus_correct_rt": safe_mean(model_rt[~stoch_correct]) - safe_mean(model_rt[stoch_correct]) if cong_name == "incongruent" else math.nan,
                    "overall_error_rt_minus_correct_rt": safe_mean(model_rt[~stoch_correct]) - safe_mean(model_rt[stoch_correct]),
                    "target_choice_proportion": float(props.get("target", 0.0)),
                    "flanker_choice_proportion": float(props.get("flanker", 0.0)),
                    "other_choice_proportion": float(props.get("other", 0.0)),
                    "incongruent_flanker_overselection_rate": float(props.get("flanker", 0.0)) if cong_name == "incongruent" else math.nan,
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
                }
            )

    return pd.concat(all_trials, ignore_index=True), pd.DataFrame(summary_rows), pd.DataFrame(all_traj_rows)


def candidate_grid() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    candidates.append({"family": "M0_original_time_gap", "config_id": "baseline", "schedule": {}, "dynamics": {}})
    for comp, shift, tscale, eshort in [(0.8, 0, 1.0, 0), (0.6, -2, 0.8, 2), (0.4, -5, 0.7, 5), (0.8, -5, 1.0, 0), (1.0, 0, 0.7, 5)]:
        candidates.append({"family": "M1_schedule_compression", "config_id": f"c{comp}_ls{shift}_tw{tscale}_ep{eshort}", "schedule": {"compression": comp, "late_shift_steps": shift, "transition_scale": tscale, "early_shorten_steps": eshort}, "dynamics": {}})
    for A, onset, tau in [(0.05, 0.05, 0.02), (0.10, 0.075, 0.04), (0.20, 0.10, 0.04), (0.30, 0.125, 0.06)]:
        candidates.append({"family": "M2_attention_gain_ramp", "config_id": f"A{A}_o{onset}_t{tau}", "schedule": {}, "dynamics": {"A": A, "t_onset": onset, "tau": tau}})
    for D, onset, tau in [(0.05, 0.05, 0.02), (0.10, 0.075, 0.04), (0.20, 0.10, 0.04), (0.30, 0.125, 0.06)]:
        candidates.append({"family": "M3_flanker_decay", "config_id": f"D{D}_o{onset}_t{tau}", "schedule": {}, "dynamics": {"D": D, "t_decay_onset": onset, "tau": tau}})
    for A, D, onset, tau in [(0.10, 0.10, 0.075, 0.04), (0.20, 0.10, 0.10, 0.04), (0.10, 0.20, 0.10, 0.06), (0.20, 0.20, 0.075, 0.06)]:
        candidates.append({"family": "M4_attention_gain_plus_flanker_decay", "config_id": f"A{A}_D{D}_o{onset}_t{tau}", "schedule": {}, "dynamics": {"A": A, "D": D, "t_onset": onset, "t_decay_onset": onset, "tau": tau}})
    for C, theta, temp, decay in [(0.05, 0.0, 0.01, 0.5), (0.10, 0.01, 0.02, 1.0), (0.20, 0.02, 0.05, 1.0), (0.10, 0.0, 0.02, 0.5)]:
        candidates.append({"family": "M5_conflict_dependent_control", "config_id": f"C{C}_th{theta}_te{temp}_d{decay}", "schedule": {}, "dynamics": {"C": C, "theta": theta, "temp": temp, "control_decay": decay}})
    for comp, A, D, onset, tau in [(0.8, 0.10, 0.10, 0.075, 0.04), (0.6, 0.10, 0.20, 0.10, 0.06), (0.8, 0.20, 0.10, 0.10, 0.04), (0.6, 0.20, 0.20, 0.075, 0.06)]:
        candidates.append({"family": "M6_combined_natural_dynamics", "config_id": f"c{comp}_A{A}_D{D}_o{onset}_t{tau}", "schedule": {"compression": comp, "late_shift_steps": -2 if comp < 1.0 else 0, "transition_scale": 0.9, "early_shorten_steps": 2}, "dynamics": {"A": A, "D": D, "t_onset": onset, "t_decay_onset": onset, "tau": tau}})
    return candidates


def add_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["behavior_fit_score"] = (
        (out["overall_accuracy"] - out["human_overall_accuracy"].fillna(0.95)).abs()
        + 2.0 * (out["incongruent_error_rate"].fillna(0.0) - out["human_incongruent_error_rate"].fillna(0.08)).abs()
        + 0.5 * (out["congruent_error_rate"].fillna(0.0) - out["human_congruent_error_rate"].fillna(0.02)).abs()
        + 0.5 * out["error_rate_by_rt_bin_rmse"].fillna(1.0)
        + 0.25 * (1.0 - out["rt_distribution_similarity_with_human"].fillna(0.0))
    )
    out["mechanism_viability_score"] = (
        out["target_first_exceeds_max_other_time"].fillna(1.0)
        + 0.5 * (1.0 - out["target_ever_exceeds_max_other_proportion"].fillna(0.0))
        + 0.5 * out["flanker_dominance_duration"].fillna(0.0)
        - 0.25 * out["late_target_recovery_strength"].fillna(0.0)
    )
    out["naturalness_penalty"] = (
        np.maximum(0.0, 0.20 - out["early_flanker_dominance"].fillna(0.0))
        + np.maximum(0.0, out["target_choice_proportion"].fillna(0.0) - 0.99)
        + np.maximum(0.0, out["congruent_error_rate"].fillna(0.0).eq(0.0).astype(float))
    )
    out["combined_score"] = out["behavior_fit_score"] + out["mechanism_viability_score"] + out["naturalness_penalty"]
    out["flag_high_incongruent_error"] = out["incongruent_error_rate"] > 0.25
    out["flag_low_accuracy"] = out["overall_accuracy"] < 0.85
    out["flag_no_congruent_errors"] = out["congruent_error_rate"].fillna(0.0) == 0.0
    out["flag_no_congruent_fast_error"] = out["congruent_error_rt_minus_correct_rt"].fillna(0.0) > 0.0
    out["flag_lost_conflict_dynamics"] = out["early_flanker_dominance"].fillna(0.0) < 0.15
    out["flag_no_target_recovery"] = out["target_ever_exceeds_max_other_proportion"].fillna(0.0) < 0.50
    out["flag_rt_distribution_broken"] = out["rt_distribution_similarity_with_human"].fillna(0.0) < 0.70
    return out


def aggregate_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, config_id), part in summary.groupby(["model_family", "model_config_id"], sort=False):
        row = {
            "model_family": family,
            "model_config_id": config_id,
            "parameter_setting": part["parameter_setting"].iloc[0],
            "combined_score": safe_mean(part["combined_score"]),
            "behavior_fit_score": safe_mean(part["behavior_fit_score"]),
            "mechanism_viability_score": safe_mean(part["mechanism_viability_score"]),
            "naturalness_penalty": safe_mean(part["naturalness_penalty"]),
        }
        for group in GROUPS:
            g = part[part["analysis_group"].eq(group)]
            row[f"{group}_overall_accuracy"] = safe_mean(g["overall_accuracy"])
            row[f"{group}_congruent_error_rate"] = safe_mean(g["congruent_error_rate"])
            row[f"{group}_incongruent_error_rate"] = safe_mean(g["incongruent_error_rate"])
            row[f"{group}_congruent_error_rt_minus_correct_rt"] = safe_mean(g["congruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_error_rt_minus_correct_rt"] = safe_mean(g["incongruent_error_rt_minus_correct_rt"])
            row[f"{group}_flanker_choice_proportion"] = safe_mean(g["flanker_choice_proportion"])
            row[f"{group}_early_flanker_dominance"] = safe_mean(g["early_flanker_dominance"])
            row[f"{group}_target_recovery_time"] = safe_mean(g["target_recovery_time"])
        for flag in [
            "flag_high_incongruent_error",
            "flag_low_accuracy",
            "flag_no_congruent_errors",
            "flag_no_congruent_fast_error",
            "flag_lost_conflict_dynamics",
            "flag_no_target_recovery",
            "flag_rt_distribution_broken",
        ]:
            row[flag] = bool(part[flag].any())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("combined_score", kind="mergesort")


def select_trial_level_export(ranking: pd.DataFrame) -> List[Tuple[str, str]]:
    keep = [("M0_original_time_gap", "baseline")]
    top10 = ranking.head(10)[["model_family", "model_config_id"]].itertuples(index=False, name=None)
    keep.extend(list(top10))
    for fam in ["M1_schedule_compression", "M2_attention_gain_ramp", "M5_conflict_dependent_control"]:
        part = ranking[ranking["model_family"].eq(fam)]
        if not part.empty:
            keep.append((part.iloc[0]["model_family"], part.iloc[0]["model_config_id"]))
    seen = []
    for k in keep:
        if k not in seen:
            seen.append(k)
    return seen


def plot_ranking(ranking: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    top = ranking.head(12).iloc[::-1]
    ax.barh(np.arange(len(top)), top["combined_score"], color="#4C78A8")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels((top["model_family"] + "\n" + top["model_config_id"]).tolist(), fontsize=7)
    ax.set_xlabel("Lower is better")
    ax.set_title("Natural dynamics model ranking")
    style_ax(ax)
    save_fig(fig, "natural_dynamics_model_ranking_overview")


def plot_summary(summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    best = ranking.iloc[0]
    best_sum = summary[(summary["model_family"].eq(best["model_family"])) & (summary["model_config_id"].eq(best["model_config_id"]))]
    baseline = summary[(summary["model_family"].eq("M0_original_time_gap")) & (summary["model_config_id"].eq("baseline"))]
    top_models = pd.concat([baseline, best_sum], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in [(axes[0], "overall_accuracy", "Accuracy"), (axes[1], "incongruent_error_rate", "Incongruent error rate")]:
        x = np.arange(len(GROUPS))
        width = 0.35
        for i, (label, part) in enumerate([("baseline", baseline), ("best", best_sum)]):
            vals = [safe_mean(part[part["analysis_group"].eq(g)][metric]) for g in GROUPS]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
        ax.set_title(title)
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "human_vs_model_accuracy_by_condition")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in [(axes[0], "congruent_error_rate", "Congruent error"), (axes[1], "incongruent_error_rate", "Incongruent error")]:
        x = np.arange(len(GROUPS))
        width = 0.35
        for i, part in enumerate([baseline, best_sum]):
            vals = [safe_mean(part[part["analysis_group"].eq(g)][metric]) for g in GROUPS]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=["baseline", "best"][i])
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
        ax.set_title(title)
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "human_vs_model_error_rate_by_condition")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot = ranking.head(20)
    ax.scatter(plot["behavior_fit_score"], 1 - plot["older_80_89_incongruent_error_rate"], c=plot["naturalness_penalty"], cmap="viridis", s=40)
    ax.set_xlabel("Behavior-fit score")
    ax.set_ylabel("Older incongruent accuracy")
    style_ax(ax)
    save_fig(fig, "tradeoff_congruent_fast_error_vs_incongruent_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        part = best_sum[best_sum["analysis_group"].eq(group)]
        base = baseline[baseline["analysis_group"].eq(group)]
        ax.bar(["baseline", "best"], [safe_mean(base["flanker_choice_proportion"]), safe_mean(part["flanker_choice_proportion"])], color=["#999999", "#D55E00"])
        ax.set_title(GROUP_LABEL[group])
        ax.set_ylabel("Flanker choice proportion")
        style_ax(ax)
    save_fig(fig, "incongruent_flanker_overselection_reduction")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        part = pd.concat([baseline[baseline["analysis_group"].eq(group)], best_sum[best_sum["analysis_group"].eq(group)]], ignore_index=True)
        ax.bar(["baseline", "best"], [safe_mean(part.iloc[[0]]["target_recovery_time"]), safe_mean(part.iloc[[1]]["target_recovery_time"])], color=["#999999", "#4C78A8"])
        ax.set_title(GROUP_LABEL[group])
        ax.set_ylabel("Target recovery time")
        style_ax(ax)
    save_fig(fig, "target_recovery_time_comparison")

    fig, ax = plt.subplots(figsize=(8, 4))
    fam = ranking.groupby("model_family", as_index=False)["combined_score"].mean().sort_values("combined_score")
    ax.bar(fam["model_family"], fam["combined_score"], color="#4C78A8")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.set_ylabel("Combined score")
    style_ax(ax)
    save_fig(fig, "naturalness_tradeoff_dashboard")


def write_summary(ranking: pd.DataFrame, summary: pd.DataFrame, n_candidates: int) -> None:
    best = ranking.iloc[0]
    best_sum = summary[(summary["model_family"].eq(best["model_family"])) & (summary["model_config_id"].eq(best["model_config_id"]))]
    baseline = ranking[(ranking["model_family"].eq("M0_original_time_gap")) & (ranking["model_config_id"].eq("baseline"))].iloc[0]
    lines = [
        "# Natural evidence dynamics optimization summary",
        "",
        f"- Tested candidate models: {n_candidates}.",
        f"- Best natural mechanism by combined score: `{best['model_family']}` / `{best['model_config_id']}`.",
        f"- Baseline combined score: {baseline['combined_score']:.4f}; best natural combined score: {best['combined_score']:.4f}.",
        "",
        "## Interpretation",
        "",
        "- This round keeps time+gap readout-choice uncertainty as the response-mapping mechanism and moves the optimization target upstream into evidence/input dynamics.",
        "- Any candidate that reduces incongruent flanker over-selection by simply erasing early flanker dominance is treated as less natural, even if its accuracy improves.",
        "- Formal fitting should only be considered after a natural dynamics family shows a credible joint improvement in incongruent error, congruent fast errors, and RT shape.",
    ]
    (OUT_DIR / "summaries/natural_evidence_dynamics_optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_script() -> None:
    src = Path(__file__).resolve()
    dst = OUT_DIR / "scripts" / "run_natural_evidence_dynamics_optimization.py"
    if src != dst:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_script()
    cache, group_params, t0_mean, t0_sd, norm_layers = load_inputs()
    write_input_inventory(cache)
    readout_noise = selected_time_gap_params()
    candidates = candidate_grid()
    all_summary = []
    all_trials = []
    all_traj = []
    log_lines = [f"candidate_count={len(candidates)}"]
    for idx, cand in enumerate(candidates, start=1):
        trials, summary, traj = run_candidate(
            cache,
            norm_layers,
            group_params,
            t0_mean,
            t0_sd,
            readout_noise,
            cand["family"],
            cand["config_id"],
            cand["schedule"],
            cand["dynamics"],
        )
        all_trials.append(trials)
        all_summary.append(summary)
        all_traj.append(traj)
        log_lines.append(f"{idx}/{len(candidates)} {cand['family']} {cand['config_id']} done")

    summary = pd.concat(all_summary, ignore_index=True)
    human_ref = pd.read_csv(READOUT_DIR / "metrics/human_reference_rt_error_metrics.csv") if (READOUT_DIR / "metrics/human_reference_rt_error_metrics.csv").exists() else pd.DataFrame()
    if not human_ref.empty and {"analysis_group", "overall_accuracy"}.issubset(human_ref.columns):
        human_ref = human_ref[human_ref["source"].eq("human")].copy()
        human_ref = human_ref.rename(
            columns={
                "overall_accuracy": "human_overall_accuracy",
                "congruent_error_rate": "human_congruent_error_rate",
                "incongruent_error_rate": "human_incongruent_error_rate",
                "congruent_error_rt_minus_correct_rt": "human_congruent_error_rt_minus_correct_rt",
                "incongruent_error_rt_minus_correct_rt": "human_incongruent_error_rt_minus_correct_rt",
            }
        )
        ref_cols = [
            "analysis_group",
            "human_overall_accuracy",
            "human_congruent_error_rate",
            "human_incongruent_error_rate",
            "human_congruent_error_rt_minus_correct_rt",
            "human_incongruent_error_rt_minus_correct_rt",
        ]
        summary = summary.merge(human_ref[ref_cols].drop_duplicates(), on="analysis_group", how="left")
    summary = add_scores(summary)
    ranking = aggregate_ranking(summary)
    selected_ids = set(select_trial_level_export(ranking))
    trials = pd.concat(all_trials, ignore_index=True)
    trials = trials[trials[["model_family", "model_config_id"]].apply(tuple, axis=1).isin(selected_ids)].copy()
    traj = pd.concat(all_traj, ignore_index=True)
    traj = traj[traj[["model_family", "model_config_id"]].apply(tuple, axis=1).isin(selected_ids)].copy()

    summary.to_csv(OUT_DIR / "metrics/natural_dynamics_model_comparison_summary.csv", index=False)
    ranking.to_csv(OUT_DIR / "metrics/natural_dynamics_model_ranking.csv", index=False)
    trials.to_csv(OUT_DIR / "metrics/natural_dynamics_trial_level_predictions.csv", index=False)
    traj.to_csv(OUT_DIR / "metrics/natural_dynamics_trajectory_diagnostics.csv", index=False)
    plot_ranking(ranking)
    plot_summary(summary, ranking)
    write_summary(ranking, summary, len(candidates))
    (OUT_DIR / "logs/natural_evidence_dynamics_optimization_run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

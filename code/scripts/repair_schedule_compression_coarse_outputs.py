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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from run_schedule_compression_pareto_search import (  # noqa: E402
    BASE_DIR,
    DT,
    GROUPS,
    GROUP_LABEL,
    OUT_DIR,
    TIME_STEPS,
    add_scores,
    aggregate_ranking,
    choice_type,
    corr_safe,
    ensure_dirs,
    load_inputs,
    make_schedule_df,
    parse_group_params,
    pareto_front,
    rmse,
    safe_mean,
    safe_q,
    style_ax,
    save_fig,
)
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule  # noqa: E402
from analyze_layerwise_evidence_ww import run_ww  # noqa: E402
from run_representative_extreme_age_subset_fitting import apply_group_t0, subset_cache  # noqa: E402
from run_gated_readout_simulation import state_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit and repair coarse schedule compression outputs.")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def schedule_from_id(schedule_id: str) -> Dict[str, Any]:
    # c0.40_ls-50_tw1.10_ep30
    parts = schedule_id.split("_")
    return {
        "schedule_config_id": schedule_id,
        "compression": float(parts[0][1:]),
        "late_shift_ms": int(parts[1][2:]),
        "transition_width_scale": float(parts[2][2:]),
        "early_phase_shortening_ms": int(parts[3][2:]),
    }


def noise_from_id(noise_id: str) -> Dict[str, Any]:
    if noise_id == "baseline":
        return {"noise_mode": "deterministic", "noise_config_id": "baseline", "sigma_base": np.nan, "sigma_time": np.nan, "sigma_gap": np.nan, "gap_scale": np.nan}
    parts = noise_id.split("_")
    return {
        "noise_mode": "shared",
        "noise_config_id": noise_id,
        "sigma_base": float(parts[0][2:]),
        "sigma_time": float(parts[1][2:]),
        "sigma_gap": float(parts[2][2:]),
        "gap_scale": float(parts[3][2:]),
    }


def first_true_time(mask: np.ndarray) -> np.ndarray:
    idx = mask.argmax(axis=1).astype(float)
    idx[~mask.any(axis=1)] = np.nan
    return idx * DT


def rebuild_candidate(
    cache: Dict[str, np.ndarray],
    norm_layers: Dict[str, np.ndarray],
    group_params: Dict[str, Dict[str, float]],
    t0_mean: Dict[str, float],
    t0_sd: Dict[str, float],
    sched: Dict[str, Any],
    noise: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    traj_rows = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        layers = {k: v[mask] for k, v in norm_layers.items()}
        gp = group_params[group]
        schedule_df = make_schedule_df(sched["compression"], sched["late_shift_ms"], sched["transition_width_scale"], sched["early_phase_shortening_ms"])
        mu = build_mu_schedule(layers, schedule_df, float(gp["evidence_gain"]))
        out = run_ww(
            mu,
            time_steps=TIME_STEPS,
            dt_ms=int(DT * 1000),
            threshold=float(gp["threshold"]),
            noise_ampa=0.0,
            device="cpu",
            seed=20260530,
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
        base_df = apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, 20260530)
        traj = np.asarray(out["trajectory"], dtype=np.float32)
        target = base_df["target_label"].to_numpy(int)
        flanker = base_df["flanker_label"].to_numpy(int)
        readout_steps = np.clip(np.rint(base_df["decision_time"].to_numpy(float) / DT).astype(int), 0, TIME_STEPS - 1)
        states = traj[np.arange(len(base_df)), readout_steps, :]
        met = state_metrics(states, target, flanker)
        det_choice = states.argmax(axis=1)
        det_correct = det_choice == target
        rows = np.arange(len(base_df))[:, None]
        times = np.arange(TIME_STEPS)[None, :]
        target_vals = traj[rows, times, target[:, None]]
        flanker_vals = traj[rows, times, flanker[:, None]]
        masked = traj.copy()
        masked[np.arange(len(base_df))[:, None], np.arange(TIME_STEPS)[None, :], target[:, None]] = -np.inf
        other_max = masked.max(axis=2)
        target_rank1 = target_vals > other_max
        target_gt_flanker = target_vals > flanker_vals
        target_gt_other = target_vals > other_max
        target_margin = target_vals - other_max
        flanker_dom = np.maximum(flanker_vals - target_vals, 0.0)
        early_flanker = (flanker_dom[:, : max(1, int(0.15 / DT))] > 0).mean(axis=1)
        flanker_duration = (flanker_dom > 0).sum(axis=1) * DT
        late_target = (target_vals[:, int(0.30 / DT) :] - flanker_vals[:, int(0.30 / DT) :]).max(axis=1)
        target_recovery_time = first_true_time(target_gt_other)
        first_rank1 = first_true_time(target_rank1)
        first_gt_flanker = first_true_time(target_gt_flanker)
        first_gt_other = first_true_time(target_gt_other)
        max_post_margin = np.max(target_margin, axis=1)

        if noise["noise_mode"] == "deterministic":
            stochastic_choice = det_choice.copy()
        else:
            earlyness = 1.0 - (readout_steps * DT) / max(float((readout_steps * DT).max()), 1e-9)
            sigma = noise["sigma_base"] + noise["sigma_time"] * earlyness + noise["sigma_gap"] * np.exp(-np.clip(met["gap"], 0, None) / max(noise["gap_scale"], 1e-9))
            rng = np.random.default_rng(20260601 + abs(hash((sched["schedule_config_id"], noise["noise_config_id"], group))) % 1000000)
            stochastic_choice = (states + rng.normal(0.0, sigma[:, None], size=states.shape)).argmax(axis=1)
        stochastic_correct = stochastic_choice == target
        ctype = choice_type(stochastic_choice, target, flanker)

        part = pd.DataFrame(
            {
                "trial_id": base_df["trial_id"].to_numpy(int),
                "analysis_group": group,
                "congruency": base_df["congruency"].to_numpy(str),
                "target_label": target,
                "flanker_label": flanker,
                "human_correct": base_df["human_correct"].to_numpy(bool),
                "true_rt": base_df["true_rt"].to_numpy(float),
                "model_config_id": f"{sched['schedule_config_id']}__{noise['noise_config_id']}",
                "schedule_config_id": sched["schedule_config_id"],
                "noise_config_id": noise["noise_config_id"],
                "deterministic_choice": det_choice,
                "deterministic_correct": det_correct,
                "stochastic_choice": stochastic_choice,
                "stochastic_correct": stochastic_correct,
                "choice_type": ctype,
                "model_rt": base_df["pred_rt"].to_numpy(float),
                "readout_time": readout_steps * DT,
                "target_recovery_time": target_recovery_time,
                "target_first_rank1_time": first_rank1,
                "target_first_exceeds_flanker_time": first_gt_flanker,
                "target_first_exceeds_max_other_time": first_gt_other,
                "target_ever_rank1": np.isfinite(first_rank1),
                "target_ever_exceeds_flanker": np.isfinite(first_gt_flanker),
                "target_ever_exceeds_max_other": np.isfinite(first_gt_other),
                "maximum_post_readout_target_margin": max_post_margin,
                "target_rank_at_readout": met["target_rank"],
                "signed_target_margin_at_readout": met["signed_target_margin"],
                "s_target_at_readout": met["s_target"],
                "s_flanker_at_readout": met["s_flanker"],
                "s_other_max_at_readout": met["s_other_max"],
                "gap_at_readout": met["gap"],
                "early_flanker_dominance": early_flanker,
                "flanker_dominance_duration": flanker_duration,
                "late_target_recovery_strength": late_target,
            }
        )
        trial_rows.append(part)

        for cong in ["congruent", "incongruent"]:
            sub = part[part["congruency"].eq(cong)]
            idx = sub.index.to_numpy(int)
            if len(idx) == 0:
                continue
            for split_name, split_mask in [
                ("human_correct", sub["human_correct"].to_numpy(bool)),
                ("human_error", ~sub["human_correct"].to_numpy(bool)),
                ("model_correct", sub["stochastic_correct"].to_numpy(bool)),
                ("model_error", ~sub["stochastic_correct"].to_numpy(bool)),
            ]:
                if not split_mask.any():
                    continue
                sel = idx[split_mask]
                for t in range(TIME_STEPS):
                    traj_rows.append(
                        {
                            "schedule_config_id": sched["schedule_config_id"],
                            "noise_config_id": noise["noise_config_id"],
                            "model_config_id": f"{sched['schedule_config_id']}__{noise['noise_config_id']}",
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
    return pd.concat(trial_rows, ignore_index=True), pd.DataFrame(traj_rows)


def rt_bin_profiles(trial_df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    rows = []
    for (model_id, group, cong), part in trial_df.groupby(["model_config_id", "analysis_group", "congruency"], sort=False):
        for source, rt_col, correct_col in [("human", "true_rt", "human_correct"), ("model", "model_rt", "stochastic_correct")]:
            order = np.argsort(part[rt_col].to_numpy(float), kind="mergesort")
            for i, idx in enumerate(np.array_split(order, bins), start=1):
                sub = part.iloc[idx]
                rows.append(
                    {
                        "model_config_id": model_id,
                        "analysis_group": group,
                        "congruency": cong,
                        "source": source,
                        "rt_bin": i,
                        "n_trials": int(len(sub)),
                        "mean_rt": safe_mean(sub[rt_col]),
                        "error_rate": 1.0 - safe_mean(sub[correct_col].astype(float)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_from_trials(
    trial_df: pd.DataFrame,
    rt_bins_df: pd.DataFrame,
    human_ref: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (schedule_id, noise_id, model_id, group, cong), part in trial_df.groupby(["schedule_config_id", "noise_config_id", "model_config_id", "analysis_group", "congruency"], sort=False):
        rt = part["model_rt"].to_numpy(float)
        correct = part["stochastic_correct"].to_numpy(bool)
        human_rt = part["true_rt"].to_numpy(float)
        human_correct = part["human_correct"].to_numpy(bool)
        bin_part = rt_bins_df[(rt_bins_df["model_config_id"].eq(model_id)) & (rt_bins_df["analysis_group"].eq(group)) & (rt_bins_df["congruency"].eq(cong))]
        pivot = bin_part.pivot(index="rt_bin", columns="source", values="error_rate")
        rows.append(
            {
                "schedule_config_id": schedule_id,
                "noise_config_id": noise_id,
                "model_config_id": model_id,
                "analysis_group": group,
                "congruency": cong,
                "overall_accuracy": safe_mean(correct.astype(float)),
                "congruent_accuracy": safe_mean(correct.astype(float)) if cong == "congruent" else math.nan,
                "incongruent_accuracy": safe_mean(correct.astype(float)) if cong == "incongruent" else math.nan,
                "congruent_error_rate": safe_mean((~correct).astype(float)) if cong == "congruent" else math.nan,
                "incongruent_error_rate": safe_mean((~correct).astype(float)) if cong == "incongruent" else math.nan,
                "mean_rt": safe_mean(rt),
                "rt_q10": safe_q(rt, 0.10),
                "rt_q50": safe_q(rt, 0.50),
                "rt_q90": safe_q(rt, 0.90),
                "rt_distribution_similarity": corr_safe([safe_q(rt, p) for p in [0.1, 0.5, 0.9]], [safe_q(human_rt, p) for p in [0.1, 0.5, 0.9]]),
                "error_rate_by_rt_bin_rmse": rmse(pivot.get("model", np.array([])), pivot.get("human", np.array([]))),
                "fast_bin_error_mismatch": abs(float(pivot.iloc[0]["model"] - pivot.iloc[0]["human"])) if len(pivot) else math.nan,
                "slow_bin_error_mismatch": abs(float(pivot.iloc[-1]["model"] - pivot.iloc[-1]["human"])) if len(pivot) else math.nan,
                "congruent_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]) if cong == "congruent" and (~correct).any() and correct.any() else math.nan,
                "incongruent_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]) if cong == "incongruent" and (~correct).any() and correct.any() else math.nan,
                "overall_error_rt_minus_correct_rt": safe_mean(rt[~correct]) - safe_mean(rt[correct]) if (~correct).any() and correct.any() else math.nan,
                "target_choice_proportion": safe_mean(part["choice_type"].eq("target").astype(float)),
                "flanker_choice_proportion": safe_mean(part["choice_type"].eq("flanker").astype(float)),
                "other_choice_proportion": safe_mean(part["choice_type"].eq("other").astype(float)),
                "incongruent_flanker_choice_proportion": safe_mean(part["choice_type"].eq("flanker").astype(float)) if cong == "incongruent" else math.nan,
                "target_recovery_time": safe_mean(part["target_recovery_time"]),
                "target_first_rank1_time": safe_mean(part["target_first_rank1_time"]),
                "target_first_exceeds_flanker_time": safe_mean(part["target_first_exceeds_flanker_time"]),
                "target_first_exceeds_max_other_time": safe_mean(part["target_first_exceeds_max_other_time"]),
                "target_ever_rank1_proportion": safe_mean(part["target_ever_rank1"].astype(float)),
                "target_ever_exceeds_flanker_proportion": safe_mean(part["target_ever_exceeds_flanker"].astype(float)),
                "target_ever_exceeds_max_other_proportion": safe_mean(part["target_ever_exceeds_max_other"].astype(float)),
                "maximum_post_readout_target_margin": safe_mean(part["maximum_post_readout_target_margin"]),
                "target_rank_at_readout": safe_mean(part["target_rank_at_readout"]),
                "signed_target_margin_at_readout": safe_mean(part["signed_target_margin_at_readout"]),
                "s_target_at_readout": safe_mean(part["s_target_at_readout"]),
                "s_flanker_at_readout": safe_mean(part["s_flanker_at_readout"]),
                "s_other_max_at_readout": safe_mean(part["s_other_max_at_readout"]),
                "gap_at_readout": safe_mean(part["gap_at_readout"]),
                "flanker_dominance_duration": safe_mean(part["flanker_dominance_duration"]),
                "early_flanker_dominance": safe_mean(part["early_flanker_dominance"]),
                "late_target_recovery_strength": safe_mean(part["late_target_recovery_strength"]),
            }
        )
    return pd.DataFrame(rows)


def compute_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    # condition-specific flags
    out["flag_no_congruent_errors_cond"] = out["congruent_error_rate"].fillna(0.0) == 0.0
    out["flag_no_congruent_fast_error_cond"] = np.where(
        out["congruency"].eq("congruent") & out["congruent_error_rate"].fillna(0.0).gt(0.0),
        out["congruent_error_rt_minus_correct_rt"].fillna(0.0) >= 0.0,
        False,
    )
    out["flag_congruent_too_many_errors_cond"] = np.where(
        out["congruency"].eq("congruent"),
        out["congruent_error_rate"].fillna(0.0) > np.maximum(0.05, out["human_congruent_error_rate"].fillna(0.02) + 0.02),
        False,
    )
    out["flag_lost_conflict_dynamics_cond"] = out["early_flanker_dominance"].fillna(0.0) < 0.15
    out["flag_excessive_flanker_dominance_cond"] = out["flanker_dominance_duration"].fillna(0.0) > 0.35
    out["flag_rt_distribution_broken_cond"] = out["rt_distribution_similarity"].fillna(0.0) < 0.70
    out["flag_unrealistic_perfect_accuracy_cond"] = (out["overall_accuracy"] > 0.995) & (out["congruent_error_rate"].fillna(0.0).eq(0.0) | out["incongruent_error_rate"].fillna(0.0).eq(0.0))
    out["flag_high_incongruent_error_cond"] = np.where(out["congruency"].eq("incongruent"), out["incongruent_error_rate"].fillna(0.0) > 0.25, False)
    out["flag_low_accuracy_cond"] = out["overall_accuracy"] < 0.85

    rows = []
    for keys, part in out.groupby(["schedule_config_id", "noise_config_id", "model_config_id", "analysis_group"], sort=False):
        cong = part[part["congruency"].eq("congruent")]
        incong = part[part["congruency"].eq("incongruent")]
        rows.append(
            {
                "schedule_config_id": keys[0],
                "noise_config_id": keys[1],
                "model_config_id": keys[2],
                "analysis_group": keys[3],
                "flag_high_incongruent_error": bool(incong["flag_high_incongruent_error_cond"].any()),
                "flag_low_accuracy": bool(part["flag_low_accuracy_cond"].any()),
                "flag_no_congruent_errors": bool(cong["flag_no_congruent_errors_cond"].any()) if not cong.empty else True,
                "flag_no_congruent_fast_error": bool(cong["flag_no_congruent_fast_error_cond"].any()) if not cong.empty else True,
                "flag_congruent_too_many_errors": bool(cong["flag_congruent_too_many_errors_cond"].any()) if not cong.empty else False,
                "flag_lost_conflict_dynamics": bool(part["flag_lost_conflict_dynamics_cond"].any()),
                "flag_excessive_flanker_dominance": bool(part["flag_excessive_flanker_dominance_cond"].any()),
                "flag_rt_distribution_broken": bool(part["flag_rt_distribution_broken_cond"].any()),
                "flag_unrealistic_perfect_accuracy": bool(part["flag_unrealistic_perfect_accuracy_cond"].any()),
            }
        )
    return pd.DataFrame(rows)


def repaired_tradeoff_labels(rank: pd.DataFrame) -> pd.DataFrame:
    out = rank.copy()
    region = []
    for _, r in out.iterrows():
        if r["is_pareto_optimal"] and r["incongruent_repair_score"] == out["incongruent_repair_score"].min():
            region.append("incongruent_repair_best")
        elif r["is_pareto_optimal"] and r["congruent_fast_error_score"] == out["congruent_fast_error_score"].min():
            region.append("fast_error_best")
        elif r["is_pareto_optimal"] and (not r["flag_no_congruent_fast_error"]) and (not r["flag_lost_conflict_dynamics"]):
            region.append("balanced")
        elif r["is_pareto_optimal"] and r["older_80_89_congruent_error_rate"] <= 0.001:
            region.append("older_fast_error_missing")
        elif r["is_pareto_optimal"] and r["young_20_29_congruent_error_rate"] > 0 and r["older_80_89_congruent_error_rate"] <= 0.001:
            region.append("young_fast_error_only")
        elif r["flag_no_congruent_fast_error"]:
            region.append("high_accuracy_but_no_fast_error")
        elif r["flag_lost_conflict_dynamics"]:
            region.append("repair_but_lost_conflict")
        else:
            region.append("not_recommended")
    out["tradeoff_region"] = region
    out["recommended_for_fine_search"] = out["is_pareto_optimal"] & (~out["flag_lost_conflict_dynamics"]) & (~out["flag_high_incongruent_error"])
    return out


def plot_repaired(rank: pd.DataFrame, trial_df: pd.DataFrame, rtbin: pd.DataFrame) -> None:
    pareto = rank[rank["is_pareto_optimal"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rank["incongruent_repair_score"], rank["congruent_fast_error_score"], c=rank[[f"{GROUPS[0]}_target_recovery_time", f"{GROUPS[1]}_target_recovery_time"]].mean(axis=1), s=40, cmap="viridis", alpha=0.55)
    ax.scatter(pareto["incongruent_repair_score"], pareto["congruent_fast_error_score"], facecolors="none", edgecolors="black", s=90)
    ax.set_xlabel("Incongruent repair score")
    ax.set_ylabel("Congruent fast-error mismatch")
    style_ax(ax)
    save_fig(fig, "pareto_front_incongruent_vs_fast_error_repaired")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rank[f"{GROUPS[0]}_incongruent_flanker_choice_proportion"], rank[f"{GROUPS[0]}_congruent_error_rt_minus_correct_rt"], alpha=0.5, label="Young")
    ax.scatter(rank[f"{GROUPS[1]}_incongruent_flanker_choice_proportion"], rank[f"{GROUPS[1]}_congruent_error_rt_minus_correct_rt"], alpha=0.5, label="Older")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Incongruent flanker choice proportion")
    ax.set_ylabel("Congruent error RT - correct RT")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    save_fig(fig, "tradeoff_flanker_overselection_vs_congruent_fast_error_repaired")


def write_audit(md_lines: List[str]) -> None:
    (OUT_DIR / "summaries/schedule_compression_coarse_metric_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_summary(rank: pd.DataFrame, orig_rank: pd.DataFrame) -> None:
    orig_bal = orig_rank[orig_rank["tradeoff_region"].eq("balanced")]
    repaired_bal = rank[rank["tradeoff_region"].eq("balanced")]
    lines = [
        "# Schedule compression Pareto search repaired summary",
        "",
        f"- Original coarse ranking size: {len(orig_rank)}; repaired ranking size: {len(rank)}.",
        f"- Original Pareto count: {int(orig_rank['is_pareto_optimal'].sum())}; repaired Pareto count: {int(rank['is_pareto_optimal'].sum())}.",
        f"- Original best balanced candidate: {orig_bal.iloc[0]['schedule_config_id']} + {orig_bal.iloc[0]['noise_config_id']}" if not orig_bal.empty else "- Original best balanced candidate: none",
        f"- Repaired best balanced candidate: {repaired_bal.iloc[0]['schedule_config_id']} + {repaired_bal.iloc[0]['noise_config_id']}" if not repaired_bal.empty else "- Repaired best balanced candidate: none",
        "",
        "## Core interpretation",
        "",
        "- The coarse search conclusions are only trustworthy after trial-level, trajectory, flag, and RT-bin metrics are derived from the same reconstructed candidate outputs.",
        "- If the repaired Pareto front still shows a broad trade-off, that trade-off is real rather than an artifact of missing fields or approximate scoring.",
        "- Fine search should only proceed on repaired Pareto candidates and only if at least one candidate remains plausible on both incongruent repair and congruent fast-error preservation.",
    ]
    (OUT_DIR / "summaries/schedule_compression_pareto_search_repaired_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cache, group_params, t0_mean, t0_sd, norm_layers, _, human = load_inputs()
    orig_rank = pd.read_csv(OUT_DIR / "metrics/schedule_compression_local_search_ranking.csv")
    orig_pareto = pd.read_csv(OUT_DIR / "metrics/schedule_compression_pareto_front.csv")
    selected = set(orig_rank.head(10)[["schedule_config_id", "noise_config_id"]].apply(tuple, axis=1).tolist())
    selected |= set(orig_pareto[["schedule_config_id", "noise_config_id"]].apply(tuple, axis=1).tolist())
    if not orig_rank.empty:
        selected.add((orig_rank.sort_values("incongruent_repair_score").iloc[0]["schedule_config_id"], orig_rank.sort_values("incongruent_repair_score").iloc[0]["noise_config_id"]))
        selected.add((orig_rank.sort_values("congruent_fast_error_score").iloc[0]["schedule_config_id"], orig_rank.sort_values("congruent_fast_error_score").iloc[0]["noise_config_id"]))
    # Previous best schedule compression and original baseline.
    selected.add(("c0.40_ls-5_tw0.7_ep5", "baseline"))
    selected.add(("baseline", "baseline"))

    audit_lines = [
        "# Schedule compression coarse metric audit",
        "",
        "- `schedule_compression_top_candidates_trial_level.csv` is not true trial-level output; it contains summary rows.",
        "- Stochastic candidates in the coarse summary are missing reconstructed target-recovery and trajectory metrics.",
        "- `flag_no_congruent_fast_error` in the original ranking can be contaminated by NaN or condition-inappropriate rows.",
        "- `flag_lost_conflict_dynamics` can be mis-triggered when stochastic rows do not carry `early_flanker_dominance`.",
        "- `error_rate_by_rt_bin_rmse` in the original script is an overall-error shortcut, not a real RT-bin RMSE.",
        "- The original Pareto front is therefore potentially affected by metric-construction artifacts and needs a repaired re-export.",
    ]

    trial_parts = []
    traj_parts = []
    for sched_id, noise_id in sorted(selected):
        if sched_id == "baseline":
            sched_id = "c1.00_ls0_tw1.00_ep0"
        sched = schedule_from_id(sched_id)
        noise = noise_from_id(noise_id)
        trial_df, traj_df = rebuild_candidate(cache, norm_layers, group_params, t0_mean, t0_sd, sched, noise)
        trial_parts.append(trial_df)
        traj_parts.append(traj_df)
    trial_all = pd.concat(trial_parts, ignore_index=True)
    traj_all = pd.concat(traj_parts, ignore_index=True)
    rtbin = rt_bin_profiles(trial_all, bins=5)
    summary = summarize_from_trials(trial_all, rtbin, human)
    summary = add_scores(summary, human)
    flags = compute_flags(summary)

    # Aggregate ranking with repaired metrics and group-level flags.
    rank_rows = []
    for (sched, noise, model), part in summary.groupby(["schedule_config_id", "noise_config_id", "model_config_id"], sort=False):
        row = {
            "schedule_config_id": sched,
            "noise_config_id": noise,
            "model_config_id": model,
            "combined_score": safe_mean(part["combined_score"]),
            "incongruent_repair_score": safe_mean(part["incongruent_repair_score"]),
            "congruent_fast_error_score": safe_mean(part["congruent_fast_error_score"]),
            "rt_dynamics_preservation_score": safe_mean(part["rt_dynamics_preservation_score"]),
            "naturalness_penalty": safe_mean(part["naturalness_penalty"]),
        }
        for group in GROUPS:
            g = part[part["analysis_group"].eq(group)]
            row[f"{group}_overall_accuracy"] = safe_mean(g["overall_accuracy"])
            row[f"{group}_congruent_error_rate"] = safe_mean(g["congruent_error_rate"])
            row[f"{group}_incongruent_error_rate"] = safe_mean(g["incongruent_error_rate"])
            row[f"{group}_congruent_error_rt_minus_correct_rt"] = safe_mean(g["congruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_error_rt_minus_correct_rt"] = safe_mean(g["incongruent_error_rt_minus_correct_rt"])
            row[f"{group}_incongruent_flanker_choice_proportion"] = safe_mean(g["incongruent_flanker_choice_proportion"])
            row[f"{group}_target_recovery_time"] = safe_mean(g["target_recovery_time"])
        f = flags[(flags["schedule_config_id"].eq(sched)) & (flags["noise_config_id"].eq(noise))]
        for col in [c for c in flags.columns if c.startswith("flag_")]:
            row[col] = bool(f[col].any()) if col in f else False
        rank_rows.append(row)
    rank = pd.DataFrame(rank_rows).sort_values("combined_score", kind="mergesort")
    rank = pareto_front(rank)
    rank = repaired_tradeoff_labels(rank)

    trial_all.to_csv(OUT_DIR / "metrics/schedule_compression_top_candidates_trial_level_repaired.csv", index=False)
    summary.to_csv(OUT_DIR / "metrics/schedule_compression_local_search_summary_repaired.csv", index=False)
    traj_all.to_csv(OUT_DIR / "metrics/schedule_compression_trajectory_diagnostics_repaired.csv", index=False)
    rtbin.to_csv(OUT_DIR / "metrics/schedule_compression_error_rate_by_rt_bin_repaired.csv", index=False)
    rank.to_csv(OUT_DIR / "metrics/schedule_compression_local_search_ranking_repaired.csv", index=False)
    rank[rank["is_pareto_optimal"]].to_csv(OUT_DIR / "metrics/schedule_compression_pareto_front_repaired.csv", index=False)
    write_audit(audit_lines)
    write_summary(rank, orig_rank)
    plot_repaired(rank, trial_all, rtbin)


if __name__ == "__main__":
    main()

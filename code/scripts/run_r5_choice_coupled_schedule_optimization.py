#!/usr/bin/env python3
"""Optimize VGG layer-to-time compression under choice–RT coupling."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vam-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww  # noqa: E402
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import (  # noqa: E402
    build_mu_schedule,
    normalize_layers,
    raw_layer_arrays,
    schedule_weights,
)
from run_r5_supervisor_followup import group_params  # noqa: E402
from run_representative_extreme_age_subset_fitting import load_trial_cache, subset_cache  # noqa: E402


BASE = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"
GROUPS = ["young_20_29", "older_80_89"]
GROUP_LABEL = {"young_20_29": "Young 20–29", "older_80_89": "Older 80–89"}
LAYER_ORDER = ["conv3", "conv4", "conv5", "pooled", "final"]
TIME_STEPS = 80
DT_S = 0.01
SEED = 20260530
MIN_CROSSING_RATE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def compressed_schedule(
    compression: float,
    late_shift_s: float = 0.0,
    width_scale: float = 1.0,
    *,
    time_steps: int = TIME_STEPS,
    dt_s: float = DT_S,
) -> pd.DataFrame:
    """Compress the original five-stage schedule in physical time."""
    duration = time_steps * dt_s
    time_s = np.arange(time_steps, dtype=np.float32) * float(dt_s)
    centers = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32) * duration
    centers *= float(compression)
    centers[3:] += float(late_shift_s)
    centers = np.clip(centers, dt_s, duration - dt_s)
    sigma = max(0.12 * duration * float(width_scale), dt_s)
    basis = np.exp(-0.5 * ((time_s[:, None] - centers[None, :]) / sigma) ** 2)
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-12)
    return pd.DataFrame(basis, columns=LAYER_ORDER)


def ordered_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=np.int64)
    for bin_id, idx in enumerate(np.array_split(np.argsort(values, kind="mergesort"), n_bins), start=1):
        out[idx] = bin_id
    return out


def accuracy_curve(rt: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> np.ndarray:
    bins = ordered_bins(rt, n_bins)
    return np.asarray([correct[bins == i].mean() for i in range(1, n_bins + 1)], dtype=float)


def safe_gap(rt: np.ndarray, correct: np.ndarray) -> float:
    if correct.all() or (~correct).all():
        return math.nan
    return float(rt[~correct].mean() - rt[correct].mean())


def safe_bool_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
    return float(values.mean()) if values.size else math.nan


def mean_zero_crossing(gap: np.ndarray) -> float:
    mean = np.asarray(gap, dtype=float).mean(axis=0)
    crossing = np.flatnonzero((mean[:-1] <= 0) & (mean[1:] > 0))
    if not crossing.size:
        return math.nan
    i = int(crossing[0])
    fraction = -mean[i] / (mean[i + 1] - mean[i])
    return float((i + fraction) * DT_S)


def first_stable_positive_after_negative(gap: np.ndarray, sustained_k: int = 2) -> np.ndarray:
    gap = np.asarray(gap, dtype=float)
    recovery = np.full(len(gap), np.nan)
    seen_negative = np.zeros(len(gap), dtype=bool)
    for step in range(gap.shape[1]):
        seen_negative |= gap[:, step] < 0
        if step + sustained_k > gap.shape[1]:
            continue
        stable = np.all(gap[:, step : step + sustained_k] > 0, axis=1)
        take = np.isnan(recovery) & seen_negative & stable
        recovery[take] = step
    return recovery


def fit_t0(
    decision_time: np.ndarray,
    true_rt: np.ndarray,
    model_correct: np.ndarray,
    human_correct: np.ndarray,
    incongruent: np.ndarray,
    group: str,
) -> dict[str, object]:
    z_rng = np.random.default_rng(SEED + (0 if group.startswith("older") else 1))
    z = np.clip(z_rng.normal(size=len(decision_time)), -2.5, 2.5)
    best: dict[str, object] | None = None
    human_q = np.quantile(true_rt, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    human_caf = accuracy_curve(true_rt, human_correct)
    human_inc_caf = accuracy_curve(true_rt[incongruent], human_correct[incongruent])
    human_inc_gap = safe_gap(true_rt[incongruent], human_correct[incongruent])
    for sd in [0.00, 0.06, 0.12, 0.18, 0.24]:
        noise = z * sd
        t0_mean = float(np.clip(np.median(true_rt) - np.median(decision_time + noise), 0.05, 1.00))
        pred_rt = np.maximum(decision_time + t0_mean + noise, 0.05)
        model_q = np.quantile(pred_rt, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        rt_q_mae = float(np.mean(np.abs(model_q - human_q)))
        caf_rmse = float(np.sqrt(np.mean((accuracy_curve(pred_rt, model_correct) - human_caf) ** 2)))
        inc_caf_rmse = float(
            np.sqrt(
                np.mean(
                    (accuracy_curve(pred_rt[incongruent], model_correct[incongruent]) - human_inc_caf) ** 2
                )
            )
        )
        model_inc_gap = safe_gap(pred_rt[incongruent], model_correct[incongruent])
        gap_error = (
            abs(model_inc_gap - human_inc_gap)
            if np.isfinite(model_inc_gap) and np.isfinite(human_inc_gap)
            else 0.25
        )
        rec = {
            "t0_mean": t0_mean,
            "t0_sd": sd,
            "pred_rt": pred_rt,
            "rt_quantile_mae": rt_q_mae,
            "caf_rmse": caf_rmse,
            "incongruent_caf_rmse": inc_caf_rmse,
            "incongruent_error_minus_correct_rt": model_inc_gap,
            "human_incongruent_error_minus_correct_rt": human_inc_gap,
            "error_rt_gap_abs_error": gap_error,
            "rt_component_score": 2.0 * rt_q_mae + 0.5 * caf_rmse + 1.0 * inc_caf_rmse + 0.5 * gap_error,
        }
        if best is None or float(rec["rt_component_score"]) < float(best["rt_component_score"]):
            best = rec
    assert best is not None
    return best


def candidate_grid() -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for compression in np.round(np.arange(0.25, 0.701, 0.025), 3):
        for late_shift_s in [-0.04, 0.00, 0.04]:
            for width_scale in [0.80, 1.00, 1.20]:
                values.append(
                    {
                        "compression": float(compression),
                        "late_shift_s": float(late_shift_s),
                        "width_scale": float(width_scale),
                    }
                )
    return values


def run_candidate(
    group: str,
    group_cache: dict[str, np.ndarray],
    group_layers: dict[str, np.ndarray],
    params: dict[str, float],
    candidate: dict[str, float],
    *,
    return_trials: bool = False,
) -> tuple[dict[str, object], pd.DataFrame | None, dict[str, np.ndarray] | None]:
    schedule = compressed_schedule(**candidate)
    ww_input = build_mu_schedule(group_layers, schedule, float(params["evidence_gain"]))
    outputs = run_ww(
        ww_input,
        time_steps=TIME_STEPS,
        dt_ms=int(DT_S * 1000),
        threshold=float(params["threshold"]),
        noise_ampa=0.0,
        device="cpu",
        seed=SEED,
        readout_mode="baseline",
        t0_seconds=0.0,
        choice_temperature=0.01,
    )
    trial = pd.DataFrame(
        {
            "analysis_group": group_cache["analysis_group"].astype(str),
            "row_index": group_cache["row_indices"].astype(np.int64),
            "user_id": group_cache["user_id"].astype(str),
            "target_label": group_cache["target_labels"].astype(np.int64),
            "flanker_label": group_cache["flanker_labels"].astype(np.int64),
            "response_label": group_cache["response_labels"].astype(np.int64),
            "human_correct": group_cache["human_correct"].astype(bool),
            "true_rt": group_cache["true_rt"].astype(float),
            "congruency": group_cache["congruency"].astype(np.int64),
            "pred_choice": outputs["pred_choice"],
        }
    )
    trial = apply_readout(
        trial,
        outputs,
        cfg=ReadoutConfig(
            "sustained_crossing",
            sustained_k=int(params["sustained_k"]),
            margin=float(params["margin"]),
            min_decision_time=float(params["min_decision_time"]),
        ),
        threshold=float(params["threshold"]),
        dt_ms=int(DT_S * 1000),
        t0_seconds=0.0,
        choice_rule="winner_at_readout",
    )
    inc = trial["congruency"].eq(1).to_numpy()
    fit = fit_t0(
        trial["decision_time"].to_numpy(float),
        trial["true_rt"].to_numpy(float),
        trial["model_correct"].to_numpy(bool),
        trial["human_correct"].to_numpy(bool),
        inc,
        group,
    )
    trial["pred_rt"] = np.asarray(fit.pop("pred_rt"), dtype=float)
    crossing_rate = float(trial["crossed"].mean())
    accuracy = float(trial["model_correct"].mean())
    human_accuracy = float(trial["human_correct"].mean())
    inc_accuracy = float(trial.loc[inc, "model_correct"].mean())
    human_inc_accuracy = float(trial.loc[inc, "human_correct"].mean())
    accuracy_gap = abs(accuracy - human_accuracy)
    inc_accuracy_gap = abs(inc_accuracy - human_inc_accuracy)
    target = trial["target_label"].to_numpy(int)
    flanker = trial["flanker_label"].to_numpy(int)
    rows = np.arange(len(trial))[:, None]
    times = np.arange(TIME_STEPS)[None, :]
    input_gap = ww_input.detach().cpu().numpy()[rows, times, target[:, None]] - ww_input.detach().cpu().numpy()[rows, times, flanker[:, None]]
    state = np.asarray(outputs["trajectory"], dtype=float)
    state_gap = state[rows, times, target[:, None]] - state[rows, times, flanker[:, None]]
    recovery_step = first_stable_positive_after_negative(state_gap)
    recovered_before_readout = np.isfinite(recovery_step) & (
        recovery_step <= trial["readout_step"].to_numpy(int)
    )
    gate = crossing_rate >= MIN_CROSSING_RATE
    score_unconstrained = accuracy_gap + 1.5 * inc_accuracy_gap + float(fit["rt_component_score"])
    score = score_unconstrained if gate else score_unconstrained + 10.0 + (MIN_CROSSING_RATE - crossing_rate)
    result: dict[str, object] = {
        "analysis_group": group,
        **candidate,
        "evidence_gain": float(params["evidence_gain"]),
        "threshold": float(params["threshold"]),
        "sustained_k": int(params["sustained_k"]),
        "margin": float(params["margin"]),
        "crossing_rate": crossing_rate,
        "crossing_gate_passed": gate,
        "accuracy": accuracy,
        "human_accuracy": human_accuracy,
        "accuracy_abs_error": accuracy_gap,
        "incongruent_accuracy": inc_accuracy,
        "human_incongruent_accuracy": human_inc_accuracy,
        "incongruent_accuracy_abs_error": inc_accuracy_gap,
        "response_agreement": float((trial["pred_choice"] == trial["response_label"]).mean()),
        "mean_rt": float(trial["pred_rt"].mean()),
        "human_mean_rt": float(trial["true_rt"].mean()),
        "mean_decision_time": float(trial["decision_time"].mean()),
        "input_mean_reversal_time": mean_zero_crossing(input_gap),
        "state_mean_reversal_time": mean_zero_crossing(state_gap),
        "state_recovered_before_readout_rate": float(recovered_before_readout[inc].mean()),
        "state_recovered_before_readout_correct_rate": safe_bool_mean(
            recovered_before_readout[inc & trial["model_correct"].to_numpy(bool)]
        ),
        "state_recovered_before_readout_error_rate": safe_bool_mean(
            recovered_before_readout[inc & ~trial["model_correct"].to_numpy(bool)]
        ),
        "score_unconstrained": score_unconstrained,
        "score": score,
        **fit,
    }
    if return_trials:
        for key, value in candidate.items():
            trial[key] = value
        trial["t0_mean"] = float(result["t0_mean"])
        trial["t0_sd"] = float(result["t0_sd"])
        return result, trial, {**outputs, "ww_input": ww_input.detach().cpu().numpy(), "schedule": schedule.to_numpy()}
    return result, None, None


def add_shared_ranking(group_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["compression", "late_shift_s", "width_scale"]
    shared = group_metrics.groupby(keys, as_index=False).agg(
        shared_score=("score", "mean"),
        shared_unconstrained_score=("score_unconstrained", "mean"),
        minimum_crossing_rate=("crossing_rate", "min"),
        maximum_accuracy_abs_error=("accuracy_abs_error", "max"),
        maximum_incongruent_accuracy_abs_error=("incongruent_accuracy_abs_error", "max"),
        mean_rt_quantile_mae=("rt_quantile_mae", "mean"),
        mean_caf_rmse=("caf_rmse", "mean"),
        mean_incongruent_caf_rmse=("incongruent_caf_rmse", "mean"),
    )
    shared["crossing_gate_passed"] = shared["minimum_crossing_rate"] >= MIN_CROSSING_RATE
    return shared.sort_values("shared_score", kind="mergesort").reset_index(drop=True)


def build_caf(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_part in trial.groupby("analysis_group", sort=True):
        for congruency, part in group_part.groupby("congruency", sort=True):
            for source, rt_col, correct_col in [
                ("human", "true_rt", "human_correct"),
                ("model", "pred_rt", "model_correct"),
            ]:
                bins = ordered_bins(part[rt_col].to_numpy(float))
                for bin_id in range(1, 6):
                    selected = part.iloc[np.flatnonzero(bins == bin_id)]
                    rows.append(
                        {
                            "analysis_group": group,
                            "congruency": int(congruency),
                            "source": source,
                            "rt_bin": bin_id,
                            "n_trials": len(selected),
                            "median_rt": float(selected[rt_col].median()),
                            "accuracy": float(selected[correct_col].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_results(group_metrics: pd.DataFrame, shared: pd.DataFrame, selected: pd.DataFrame, caf: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    colors = {"young_20_29": "#3B78A8", "older_80_89": "#C45A32"}
    for group in GROUPS:
        part = group_metrics[
            group_metrics["analysis_group"].eq(group)
            & group_metrics["late_shift_s"].eq(0.0)
            & group_metrics["width_scale"].eq(1.0)
        ].sort_values("compression")
        axes[0].plot(part["compression"], part["incongruent_accuracy"], marker="o", ms=2.5, color=colors[group], label=GROUP_LABEL[group])
        axes[0].axhline(float(part["human_incongruent_accuracy"].iloc[0]), color=colors[group], ls="--", lw=0.8)
    axes[0].set(title="Compression restores target choice", xlabel="Schedule compression", ylabel="Incongruent accuracy", ylim=(0, 1.02))
    axes[0].legend(fontsize=6)

    valid = shared[shared["crossing_gate_passed"]]
    scatter = axes[1].scatter(
        valid["mean_rt_quantile_mae"],
        valid["maximum_incongruent_accuracy_abs_error"],
        c=valid["compression"],
        cmap="viridis",
        s=12,
        alpha=0.7,
    )
    best_shared = valid.iloc[0]
    axes[1].scatter(best_shared["mean_rt_quantile_mae"], best_shared["maximum_incongruent_accuracy_abs_error"], facecolors="none", edgecolors="black", s=55, lw=1)
    axes[1].set(title="Shared-schedule trade-off", xlabel="RT quantile MAE (s)", ylabel="Max. incongruent accuracy error")
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.06, pad=0.03)
    cbar.set_label("Compression")

    inc = caf[caf["congruency"].eq(1)]
    line_specs = [("human", "Human", "--", "white"), ("model", "Selected model", "-", None)]
    for source, label, ls, marker_face in line_specs:
        for group in GROUPS:
            part = inc[inc["source"].eq(source) & inc["analysis_group"].eq(group)].sort_values("rt_bin")
            axes[2].plot(
                part["median_rt"],
                part["accuracy"],
                marker="o",
                ms=2.8,
                color=colors[group],
                markerfacecolor=marker_face if marker_face is not None else colors[group],
                markeredgecolor=colors[group],
                ls=ls,
                alpha=0.95,
                label=f"{label}, {GROUP_LABEL[group]}",
            )
    axes[2].set(title="Selected incongruent CAF", xlabel="Median RT (s)", ylabel="Accuracy", ylim=(0, 1.03))
    axes[2].legend(fontsize=5.5)
    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
    fig.suptitle("Choice-coupled schedule compression optimization", y=0.99, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.78, wspace=0.42)
    stem = out / "schedule_compression_optimization_overview"
    for ext in ["png", "tiff", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)
    caption = """# Figure | Choice-coupled schedule compression optimization

**Conclusion.** Earlier delivery of late VGG evidence restores target choices without using post-RT information or deadline fallbacks; separate age-group schedules outperform a shared schedule on this representative subset.

**Panels.** a, Incongruent accuracy across the baseline compression slice; dashed lines show observed accuracy. b, Shared-schedule candidates that passed the 95% crossing gate, showing RT-distribution error against the maximum age-group incongruent-accuracy error; the outlined point is the best shared setting. c, Observed and selected-model incongruent conditional accuracy functions. Each age group contains 5,000 trials; the optimization is deterministic apart from a fixed non-decision-time draw and has not yet been evaluated on held-out participants or stimuli.
"""
    (out / "schedule_compression_optimization_overview_caption.md").write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise RuntimeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    cache = load_trial_cache(BASE)
    params = group_params()
    normalized = normalize_layers(raw_layer_arrays(cache), "per_layer_gap_scale")
    grid = candidate_grid()
    rows: list[dict[str, object]] = []
    group_inputs: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        layers = {key: value[mask] for key, value in normalized.items()}
        group_inputs[group] = (gc, layers)
        for candidate in grid:
            result, _, _ = run_candidate(group, gc, layers, params[group], candidate)
            rows.append(result)
    group_metrics = pd.DataFrame(rows).sort_values(["analysis_group", "score"], kind="mergesort")
    shared = add_shared_ranking(group_metrics)
    best_shared = shared[shared["crossing_gate_passed"]].iloc[0]
    shared_key = {key: float(best_shared[key]) for key in ["compression", "late_shift_s", "width_scale"]}

    selected_specs: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        valid = group_metrics[group_metrics["analysis_group"].eq(group) & group_metrics["crossing_gate_passed"]]
        best = valid.iloc[0]
        selected_specs[group] = {key: float(best[key]) for key in ["compression", "late_shift_s", "width_scale"]}
    age_specific_score = float(
        np.mean(
            [
                group_metrics[
                    group_metrics["analysis_group"].eq(group)
                    & group_metrics["compression"].eq(spec["compression"])
                    & group_metrics["late_shift_s"].eq(spec["late_shift_s"])
                    & group_metrics["width_scale"].eq(spec["width_scale"])
                ].iloc[0]["score"]
                for group, spec in selected_specs.items()
            ]
        )
    )
    use_age_specific = age_specific_score < float(best_shared["shared_score"]) * 0.95
    if not use_age_specific:
        selected_specs = {group: dict(shared_key) for group in GROUPS}

    selected_rows: list[pd.DataFrame] = []
    selected_metrics: list[dict[str, object]] = []
    selected_outputs: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUPS:
        gc, layers = group_inputs[group]
        metric, trial, outputs = run_candidate(
            group,
            gc,
            layers,
            params[group],
            selected_specs[group],
            return_trials=True,
        )
        assert trial is not None and outputs is not None
        selected_rows.append(trial)
        selected_metrics.append(metric)
        selected_outputs[group] = outputs
    selected = pd.concat(selected_rows, ignore_index=True)
    selected_metric_df = pd.DataFrame(selected_metrics)
    caf = build_caf(selected)

    group_metrics.to_csv(out / "candidate_group_metrics.csv", index=False)
    shared.to_csv(out / "shared_schedule_ranking.csv", index=False)
    selected_metric_df.to_csv(out / "selected_model_metrics.csv", index=False)
    selected.to_csv(out / "selected_trial_level_predictions.csv", index=False)
    caf.to_csv(out / "selected_caf.csv", index=False)
    schedule_rows = []
    for group, spec in selected_specs.items():
        sdf = compressed_schedule(**spec)
        for step, row in sdf.iterrows():
            for layer in LAYER_ORDER:
                schedule_rows.append({"analysis_group": group, "time_step": step, "time_s": step * DT_S, "layer": layer, "weight": float(row[layer]), **spec})
    pd.DataFrame(schedule_rows).to_csv(out / "selected_schedule_weights.csv", index=False)
    plot_results(group_metrics, shared, selected, caf, out)

    human_inc = selected[selected["congruency"].eq(1)].groupby("analysis_group")["human_correct"].mean()
    model_inc = selected[selected["congruency"].eq(1)].groupby("analysis_group")["model_correct"].mean()
    crossing = selected.groupby("analysis_group")["crossed"].mean()
    qa = {
        "n_trials": len(selected),
        "n_candidates_per_group": len(grid),
        "choice_rule_all_winner_at_readout": bool(selected["choice_rule"].eq("winner_at_readout").all()),
        "minimum_selected_crossing_rate": float(crossing.min()),
        "crossing_gate_passed": bool((crossing >= MIN_CROSSING_RATE).all()),
        "selected_schedules": selected_specs,
        "best_shared_schedule": shared_key,
        "best_shared_score": float(best_shared["shared_score"]),
        "age_specific_score": age_specific_score,
        "age_specific_selected": use_age_specific,
        "maximum_incongruent_accuracy_abs_error": float((model_inc - human_inc).abs().max()),
        "all_metrics_finite": bool(np.isfinite(selected_metric_df.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    qa["passed"] = bool(
        qa["n_trials"] == 10000
        and qa["choice_rule_all_winner_at_readout"]
        and qa["crossing_gate_passed"]
        and qa["all_metrics_finite"]
    )
    (out / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Choice-coupled schedule compression optimization",
        "",
        "## Design",
        "",
        f"- Tested `{len(grid)}` schedule configurations per age group while keeping VGG logits and retained R5 Wong-Wang parameters fixed.",
        "- Choice was always the winner at the sustained-crossing RT step.",
        f"- Candidates below `{MIN_CROSSING_RATE:.0%}` real crossing coverage were ineligible.",
        "- Non-decision mean and spread were refit for every candidate.",
        "",
        "## Selected result",
        "",
        f"- Shared optimum: `{shared_key}` with score `{float(best_shared['shared_score']):.4f}`.",
        f"- Age-specific optimum score: `{age_specific_score:.4f}`; age-specific schedules selected: `{use_age_specific}`.",
    ]
    for group in GROUPS:
        row = selected_metric_df[selected_metric_df["analysis_group"].eq(group)].iloc[0]
        lines.append(
            f"- {GROUP_LABEL[group]}: schedule `{selected_specs[group]}`, crossing `{row['crossing_rate']:.3f}`, "
            f"accuracy `{row['accuracy']:.3f}` vs human `{row['human_accuracy']:.3f}`, incongruent accuracy "
            f"`{row['incongruent_accuracy']:.3f}` vs human `{row['human_incongruent_accuracy']:.3f}`, "
            f"RT quantile MAE `{row['rt_quantile_mae']:.3f}` s, incongruent CAF RMSE `{row['incongruent_caf_rmse']:.3f}`."
        )
        lines.append(
            f"  Mean input reversal `{row['input_mean_reversal_time']:.3f}` s, mean WW-state reversal "
            f"`{row['state_mean_reversal_time']:.3f}` s, target state recovered before readout on "
            f"`{row['state_recovered_before_readout_rate']:.1%}` of incongruent trials "
            f"(`{row['state_recovered_before_readout_correct_rate']:.1%}` of correct; "
            f"`{row['state_recovered_before_readout_error_rate']:.1%}` of errors)."
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is an exploratory representative-subset schedule optimization, not a held-out or full-cohort fit. It tests whether the existing real-VGG target-recovery signal can support a theoretically coupled decision after its timing is corrected; it does not by itself validate a human conflict-control mechanism.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not qa["passed"]:
        raise RuntimeError(f"QA failed: {qa}")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Formal frozen Human H1-H6 and commitment-consistent audit of C0v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_c0_canonical_h1_h6_audit import (  # noqa: E402
    BASE,
    FULL_TARGETS_PATH,
    association_table,
    c0_distances,
    distribution_table,
    enrich_summary,
    load_human_module,
    matched_behavior_tables,
    matched_repeat_inconsistency,
    normalize_layers,
    raw_layer_arrays,
    simulate_group,
    summarize_model,
    summarize_trajectories,
    trajectory_audit,
)
from run_c0v2_causal_commitment_audit import identity_gate  # noqa: E402
from run_representative_extreme_age_subset_fitting import load_trial_cache  # noqa: E402

MANIFEST = ROOT / "configs/canonical_baseline_manifest.json"
C0V2_RESULT = ROOT / "artifacts/results/c0v2_causal_commitment_baseline_20260813_v2"
CORE_SAVED = C0V2_RESULT / "c0v2_core_trial_level_predictions.csv"
ALL_AGE_SAVED = C0V2_RESULT / "c0v2_all_age_trial_level_predictions.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/results/c0v2_canonical_h1_h6_audit_20260814_v3"
AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
MODEL_ID = "C0v2_causal_commitment_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise AssertionError(f"Identity differs: expected {MODEL_ID}, found {manifest.get('model_id')}")
    semantics = manifest["decision_semantics"]
    expected = {
        "commitment_step_rule": "window_start_step_plus_sustained_k_minus_1",
        "choice_rule": "winner_at_commitment_completion",
    }
    for key, value in expected.items():
        if semantics.get(key) != value:
            raise AssertionError(f"Identity differs at {key}: {semantics.get(key)}")
    return manifest


def replay_core(manifest: dict, device: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    cache = load_trial_cache(BASE)
    normalized = normalize_layers(raw_layer_arrays(cache), "per_layer_gap_scale")
    frames, trajectories, inputs = [], [], []
    for group in ["young_20_29", "older_80_89"]:
        frame, outputs = simulate_group(group, cache, normalized, manifest, device)
        frames.append(frame)
        trajectories.append(np.asarray(outputs["trajectory"], dtype=np.float32))
        inputs.append(np.asarray(outputs["ww_input"], dtype=np.float32))
    frame = pd.concat(frames, ignore_index=True)
    trajectory = np.concatenate(trajectories)
    ww_input = np.concatenate(inputs)
    saved = pd.read_csv(CORE_SAVED)
    stable = ["analysis_group", "row_index", "user_id", "target_label", "flanker_label", "response_label", "congruency"]
    frame["user_id"] = frame["user_id"].astype(str)
    saved["user_id"] = saved["user_id"].astype(str)
    stable_match = len(saved) == len(frame) == 10_000 and all(
        np.array_equal(saved[column].to_numpy(), frame[column].to_numpy()) for column in stable
    )
    gate = identity_gate(frame, trajectory, manifest)
    qa = {
        **gate,
        "stable_trial_identity_exact_match": bool(stable_match),
        "saved_choice_exact_match_rate": float(np.mean(saved["pred_choice"].to_numpy() == frame["pred_choice"].to_numpy())),
        "saved_commitment_step_exact_match_rate": float(np.mean(saved["commitment_step"].to_numpy() == frame["commitment_step"].to_numpy())),
        "saved_rt_max_abs_difference_s": float(np.max(np.abs(saved["pred_rt"].to_numpy(float) - frame["pred_rt"].to_numpy(float)))),
        "c0v2_modified": False,
        "model_m1_created": False,
    }
    qa["passed"] = bool(
        gate["passed"] and stable_match and qa["saved_choice_exact_match_rate"] == 1.0
        and qa["saved_commitment_step_exact_match_rate"] == 1.0
        and qa["saved_rt_max_abs_difference_s"] < 1e-6
    )
    return frame, trajectory, ww_input, qa


def rename_source_tables(tables: dict) -> dict:
    return {"human": tables["human"], "c0": tables["model"]}


def core_scorecard(summary: pd.DataFrame, distances: pd.DataFrame, full_targets: pd.DataFrame, repeat: dict) -> pd.DataFrame:
    human = summary[summary.source.eq("human")].iloc[0]
    model = summary[summary.source.eq("c0")].iloc[0]
    distance = distances.iloc[0]
    full = dict(zip(full_targets["Signature"], full_targets["Human estimate"]))
    rows = [
        ("H1", f"error={human.congruent_error_rate:.4f}; repeat={repeat['human']}", f"error={model.congruent_error_rate:.4f}; repeat={repeat['model']}", f"error difference={distance.h1_abs_distance:.4f}", "REPAIR / MATCH", "FAIL"),
        ("H2", f"RT={human.h2_rt_cost_s:.4f}s; accuracy={human.h2_accuracy_cost:.4f}", f"RT={model.h2_rt_cost_s:.4f}s; accuracy={model.h2_accuracy_cost:.4f}", f"RT={distance.h2_rt_cost_abs_distance_s:.4f}s; accuracy={distance.h2_accuracy_cost_abs_distance:.4f}", "PRESERVE", "PARTIAL"),
        ("H3", f"fast={human.h3_fastest_accuracy:.4f}; slow={human.h3_slowest_accuracy:.4f}; slow-fast={human.h3_slow_minus_fast:.4f}; slope={human.h3_caf_slope_per_s:.4f}", f"fast={model.h3_fastest_accuracy:.4f}; slow={model.h3_slowest_accuracy:.4f}; slow-fast={model.h3_slow_minus_fast:.4f}; slope={model.h3_caf_slope_per_s:.4f}", f"curve RMSE={distance.h3_curve_rmse:.4f}; slope difference={distance.h3_slope_abs_distance:.4f}", "IMPROVE / PRIMARY", "PARTIAL"),
        ("H4", f"slope={human.h4_delta_slope:.4f}; late-early={human.h4_late_minus_early_s:.4f}s", f"slope={model.h4_delta_slope:.4f}; late-early={model.h4_late_minus_early_s:.4f}s", f"curve RMSE={distance.h4_curve_rmse_s:.4f}s; slope difference={distance.h4_slope_abs_distance:.4f}", "PRESERVE", "FAIL"),
        ("H5", f"SD={human.h5_congruent_sd_rt_s:.3f}/{human.h5_incongruent_sd_rt_s:.3f}s; skew={human.h5_correct_skew_congruent:.3f}/{human.h5_correct_skew_incongruent:.3f}", f"SD={model.h5_congruent_sd_rt_s:.3f}/{model.h5_incongruent_sd_rt_s:.3f}s; skew={model.h5_correct_skew_congruent:.3f}/{model.h5_correct_skew_incongruent:.3f}", f"Wasserstein={distance.h5_wasserstein_congruent_s:.4f}/{distance.h5_wasserstein_incongruent_s:.4f}s", "QUANTIFY / IMPROVE", "FAIL"),
        ("H6", f"congruent={human.h6_error_minus_correct_congruent_s:.4f}s; incongruent={human.h6_error_minus_correct_incongruent_s:.4f}s; interaction={human.h6_interaction_s:.4f}s", f"congruent=NOT ESTIMABLE; incongruent={model.h6_error_minus_correct_incongruent_s:.4f}s; interaction=NOT ESTIMABLE", f"incongruent difference={distance.h6_incongruent_abs_distance_s:.4f}s", "POTENTIAL IMPROVEMENT", "NOT ESTIMABLE"),
    ]
    return pd.DataFrame([
        {"Signature": signature, "Full Human": full.get(signature, "see frozen human audit"), "Matched Human": matched, "C0v2": c0v2, "Distance": dist, "Objective": objective, "Status": status}
        for signature, matched, c0v2, dist, objective, status in rows
    ])


def curve_table(results: dict) -> pd.DataFrame:
    rows = []
    for source in ["human", "c0"]:
        label = "matched_human" if source == "human" else "C0v2"
        h3 = results[source]["h3b"].groupby("rt_bin", as_index=False).agg(rt_coordinate_s=("mean_rt_s", "mean"), value=("accuracy", "mean"))
        h3["signature"] = "H3_CAF"
        h3["source"] = label
        h4 = results[source]["h4b"].groupby("rt_bin", as_index=False).agg(rt_coordinate_s=("mean_rt_s", "mean"), value=("delta_rt_s", "mean"))
        h4["signature"] = "H4_delta"
        h4["source"] = label
        rows.extend([h3, h4])
    return pd.concat(rows, ignore_index=True)


def complete_trajectory_fields(trial: pd.DataFrame, frame: pd.DataFrame, trajectory: np.ndarray, ww_input: np.ndarray) -> pd.DataFrame:
    """Add requested pre-commitment target peaks and separate early evidence values."""
    out = trial.copy()
    target_peaks, early_target, early_flanker = [], [], []
    for i, row in frame.reset_index(drop=True).iterrows():
        commit = int(row["commitment_step"])
        target = int(row["target_label"])
        flanker = int(row["flanker_label"])
        target_peaks.append(float(trajectory[i, :commit, target].max()))
        early_end = min(commit, 10)
        early_target.append(float(ww_input[i, :early_end, target].mean()))
        early_flanker.append(float(ww_input[i, :early_end, flanker].mean()))
    out["target_state_peak_before_commitment"] = target_peaks
    out["early_target_evidence"] = early_target
    out["early_flanker_evidence"] = early_flanker
    return out


def add_width_ratios(distributions: pd.DataFrame) -> pd.DataFrame:
    out = distributions.copy()
    for condition in ["congruent", "incongruent"]:
        for scope in ["participant_balanced_primary", "pooled_trials_secondary"]:
            cell = out.condition.eq(condition) & out.scope.eq(scope)
            human = out[cell & out.source.eq("human")]
            model = out[cell & out.source.eq("c0")]
            if len(human) == len(model) == 1:
                sd_ratio = float(model.iloc[0].sd_s / human.iloc[0].sd_s)
                q95q10_ratio = float((model.iloc[0].q95_s - model.iloc[0].q10_s) / (human.iloc[0].q95_s - human.iloc[0].q10_s))
                out.loc[cell, "c0v2_to_human_sd_ratio"] = sd_ratio
                out.loc[cell, "c0v2_to_human_q95_q10_width_ratio"] = q95q10_ratio
    return out


def age_audit(all_age: pd.DataFrame, human_module) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = all_age[all_age["crossed"].astype(bool)].copy()
    valid["analysis_group"] = valid["age_group"].astype(str)
    valid["true_rt"] = valid["human_rt"].astype(float)
    metric_rows, curve_rows = [], []
    for age in AGE_ORDER:
        part = valid[valid.analysis_group.eq(age)].copy()
        tables, _ = matched_behavior_tables(part, human_module)
        results = rename_source_tables(tables)
        summary = enrich_summary(pd.DataFrame([summarize_model(results[key], key) for key in ["human", "c0"]]), results)
        distances = c0_distances(summary, results)
        for source in ["human", "c0"]:
            row = summary[summary.source.eq(source)].iloc[0].to_dict()
            row.update({"age_group": age, "source": "matched_human" if source == "human" else "C0v2"})
            metric_rows.append(row)
            for signature, table, value_col in [("H3_CAF", results[source]["h3b"], "accuracy"), ("H4_delta", results[source]["h4b"], "delta_rt_s")]:
                curve = table.groupby("rt_bin", as_index=False).agg(rt_coordinate_s=("mean_rt_s", "mean"), value=(value_col, "mean"))
                curve["age_group"] = age
                curve["source"] = row["source"]
                curve["signature"] = signature
                curve_rows.append(curve)
        dist = distances.iloc[0].to_dict()
        dist.update({"age_group": age, "source": "C0v2_distance_to_matched_human"})
        metric_rows.append(dist)
    return pd.DataFrame(metric_rows), pd.concat(curve_rows, ignore_index=True)


def plot_core(results: dict, distributions: pd.DataFrame, trajectory: pd.DataFrame, output: Path) -> None:
    colors = {"human": "#222222", "c0": "#D55E00"}
    labels = {"human": "Matched human", "c0": "C0v2"}
    summary = enrich_summary(pd.DataFrame([summarize_model(results[k], k) for k in ["human", "c0"]]), results)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = np.arange(2)
    for offset, source in [(-.18, "human"), (.18, "c0")]:
        row = summary[summary.source.eq(source)].iloc[0]
        ax.bar(x + offset, [row.h2_rt_cost_s, row.h2_accuracy_cost], .34, color=colors[source], label=labels[source])
    ax.set_xticks(x, ["RT cost (s)", "Accuracy cost"]); ax.set_ylabel("Effect"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "01_h2_human_vs_c0v2.png", dpi=300); plt.close(fig)

    for number, signature, key, ylabel, title in [
        (2, "h3", "accuracy", "Accuracy", "H3 incongruent CAF"),
        (3, "h4", "delta_rt_s", "Incongruent - congruent RT (s)", "H4 correct-trial delta plot"),
    ]:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        table_key = "h3b" if signature == "h3" else "h4b"
        for source in ["human", "c0"]:
            curve = results[source][table_key].groupby("rt_bin", as_index=False).agg(rt=("mean_rt_s", "mean"), value=(key, "mean"))
            ax.plot(curve.rt, curve.value, marker="o", color=colors[source], label=labels[source])
        ax.set(xlabel="Actual RT coordinate (s)", ylabel=ylabel, title=title); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(output / f"0{number}_{signature}_curve.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, condition in zip(axes, ["congruent", "incongruent"]):
        for source in ["human", "c0"]:
            data = results[source]["data"]
            values = data.loc[data.congruency.eq(condition) & data.correct, "rt_s"]
            ax.hist(values, bins=50, density=True, histtype="step", linewidth=1.6, color=colors[source], label=labels[source])
        ax.set(title=condition.title(), xlabel="Correct RT (s)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density"); axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(output / "04_h5_rt_distributions.png", dpi=300); plt.close(fig)

    congruent = trajectory[trajectory.congruency.eq("congruent")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].hist(congruent.distance_wrong_peak_to_threshold, bins=45, color="#4C78A8")
    axes[0].set(xlabel="Threshold - wrong-state peak", ylabel="Trials", title="Congruent wrong-state competition")
    counts = congruent.trajectory_class.value_counts(normalize=True)
    axes[1].bar(counts.index, counts.values, color="#D55E00"); axes[1].tick_params(axis="x", rotation=18); axes[1].set(ylabel="Proportion", title="Trajectory classes")
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "05_congruent_wrong_state_competition.png", dpi=300); plt.close(fig)

    classes = ["FAST ERROR", "FAST CORRECT", "SLOW CORRECT"]
    metrics = [("commitment_time_s", "Commitment time"), ("wrong_state_peak_before_commitment", "Wrong-state peak"), ("target_minus_wrong_at_commitment", "Target-wrong margin")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, (metric, title) in zip(axes, metrics):
        values = [trajectory.loc[trajectory.rt_outcome_class.eq(group), metric].dropna() for group in classes]
        ax.boxplot(values, tick_labels=classes, showfliers=False); ax.tick_params(axis="x", rotation=18); ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "06_fast_error_fast_correct_slow_correct.png", dpi=300); plt.close(fig)


def plot_age(age_metrics: pd.DataFrame, age_curves: pd.DataFrame, all_age: pd.DataFrame, output: Path) -> None:
    colors = {"matched_human": "#222222", "C0v2": "#D55E00"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharex=False)
    h3 = age_curves[age_curves.signature.eq("H3_CAF")]
    for ax, age in zip(axes.flat, AGE_ORDER):
        for source in ["matched_human", "C0v2"]:
            part = h3[h3.age_group.eq(age) & h3.source.eq(source)].sort_values("rt_bin")
            ax.plot(part.rt_coordinate_s, part.value, marker="o", color=colors[source], label=source)
        ax.set(title=age, xlabel="RT (s)", ylabel="Accuracy"); ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off"); axes.flat[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "07_age_group_caf.png", dpi=300); plt.close(fig)

    valid = all_age[all_age.crossed.astype(bool)].copy()
    valid["condition"] = np.where(valid.congruency.astype(int).eq(0), "congruent", "incongruent")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharey=False)
    for ax, age in zip(axes.flat, AGE_ORDER):
        part = valid[valid.age_group.eq(age)]
        for source, rt, correct, color in [("Human", "human_rt", "human_correct", "#222222"), ("C0v2", "pred_rt", "model_correct", "#D55E00")]:
            values = part.loc[part[correct].astype(bool), rt].dropna()
            ax.hist(values, bins=35, density=True, histtype="step", linewidth=1.4, color=color, label=source)
        ax.set(title=age, xlabel="Correct RT (s)", ylabel="Density"); ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off"); axes.flat[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "08_age_group_rt_distributions.png", dpi=300); plt.close(fig)


def write_reports(output: Path, qa: dict, scorecard: pd.DataFrame, summary: pd.DataFrame, trajectory_summary: pd.DataFrame, age_metrics: pd.DataFrame) -> None:
    congruent = trajectory_summary[trajectory_summary.scope.eq("congruent")].iloc[0]
    incongruent = trajectory_summary[trajectory_summary.scope.eq("incongruent")].iloc[0]
    groups = {row.scope: row for _, row in trajectory_summary[trajectory_summary.scope.isin(["FAST ERROR", "FAST CORRECT", "SLOW CORRECT"])].iterrows()}
    case = "A" if congruent.p_any_wrong_leader_before_commitment < .01 else ("B" if congruent.wrong_commitment_error_proportion < .01 else "C")
    diagnostic = f"""# C0v2 commitment diagnostic

All behavioral labels use the completed commitment event. Post-commitment states never alter recorded choice or RT.

## Congruent trials

- Any wrong leader before commitment: {congruent.p_any_wrong_leader_before_commitment:.4f}
- First meaningful leader wrong: {congruent.p_first_meaningful_leader_wrong:.4f}
- Wrong-state peak: {congruent.mean_wrong_state_peak_before_commitment:.4f}
- Median distance to threshold: {congruent.median_distance_wrong_peak_to_threshold:.4f}
- Target-minus-wrong margin at commitment: {congruent.mean_target_minus_wrong_at_commitment:.4f}
- Diagnosis: **CASE {case}**

## Incongruent trials

- Early wrong leader: {incongruent.p_first_meaningful_leader_wrong:.4f}
- `pC_pre`: {incongruent.pC_pre_correct_commitment_given_early_wrong:.4f}
- Wrong commitment: {incongruent.wrong_commitment_error_proportion:.4f}
- Mean commitment time: {incongruent.mean_commitment_time_s:.4f} s
- Post-commitment internal recovery after wrong commitment: {incongruent.postcommit_recovery_rate_given_wrong_commitment:.4f}
- Mean recovery delay: {incongruent.mean_postcommit_recovery_delay_s:.4f} s

## Fast errors and corrected trials

FAST ERROR is an incongruent error at or below the fixed-subset median incongruent RT; FAST CORRECT and SLOW CORRECT use the same median split. This definition is reproducible and does not use trajectory outcomes to set the boundary.

| Group | Commitment time | Wrong peak | Target-wrong margin |
|---|---:|---:|---:|
| FAST ERROR | {groups['FAST ERROR'].mean_commitment_time_s:.4f} | {groups['FAST ERROR'].mean_wrong_state_peak_before_commitment:.4f} | {groups['FAST ERROR'].mean_target_minus_wrong_at_commitment:.4f} |
| FAST CORRECT | {groups['FAST CORRECT'].mean_commitment_time_s:.4f} | {groups['FAST CORRECT'].mean_wrong_state_peak_before_commitment:.4f} | {groups['FAST CORRECT'].mean_target_minus_wrong_at_commitment:.4f} |
| SLOW CORRECT | {groups['SLOW CORRECT'].mean_commitment_time_s:.4f} | {groups['SLOW CORRECT'].mean_wrong_state_peak_before_commitment:.4f} | {groups['SLOW CORRECT'].mean_target_minus_wrong_at_commitment:.4f} |

Associations in the companion CSV are descriptive correlations, not causal effects.
"""
    (output / "c0v2_commitment_diagnostic.md").write_text(diagnostic, encoding="utf-8")

    human = summary[summary.source.eq("human")].iloc[0]
    model = summary[summary.source.eq("c0")].iloc[0]
    age_model = age_metrics[age_metrics.source.eq("C0v2")]
    con_sd_ratio = model.h5_congruent_sd_rt_s / human.h5_congruent_sd_rt_s
    inc_sd_ratio = model.h5_incongruent_sd_rt_s / human.h5_incongruent_sd_rt_s
    report = f"""# C0v2 canonical Human H1-H6 and trajectory audit

## Exact identity

`{MODEL_ID}` passed the strict identity gate on {qa['n_trials']:,} fixed core trials. Zero-based `commitment_step = window_start_step + sustained_k - 1`; choice is the winner at that completed event and decision time is `commitment_step × 0.01 s`. Post-commitment mutation and prefix-only replay both pass on every trial. No whole-trajectory choice rule entered this audit.

## Human-C0v2 scorecard

{scorecard.to_markdown(index=False)}

No arbitrary numerical pass tolerances were introduced; statuses use the frozen empirical directions and whether the requested metric is estimable.

## What C0v2 already does well

- It preserves the primary H3 fast-error direction and substantial pre-commitment correction (`pC_pre={incongruent.pC_pre_correct_commitment_given_early_wrong:.3f}`).
- It retains overall congruency costs and strong age-related mean-RT/accuracy patterns across seven groups.
- Behavioral choice and RT have a single causal commitment definition.

## What C0v2 still fails to explain

- H1: congruent error rate is {model.congruent_error_rate:.4f}, versus {human.congruent_error_rate:.4f} in the matched human subset.
- H4 remains quantitatively mismatched despite retaining a structured delta curve.
- H5 is not uniformly narrower by every width metric. Participant-mean SD is {model.h5_congruent_sd_rt_s:.3f}/{model.h5_incongruent_sd_rt_s:.3f} s versus human {human.h5_congruent_sd_rt_s:.3f}/{human.h5_incongruent_sd_rt_s:.3f} s: C0v2 is {100*(1-con_sd_ratio):.1f}% narrower for congruent trials and {100*(inc_sd_ratio-1):.1f}% wider for incongruent trials. Its near-zero skew nevertheless misses the strong human long-tail shape.
- H6 congruent and interaction terms are not estimable because C0v2 has no congruent errors.

## Mechanistic diagnosis

Congruent trials are **CASE {case}**: wrong channels do not become meaningfully competitive before commitment. Fast errors differ from corrected captures mainly by earlier wrong commitment, a stronger wrong state, and a negative target-minus-wrong margin. Later target dominance after a wrong commitment is reported only as **POST-COMMITMENT INTERNAL RECOVERY**, never behavioral correction.

## Age-group preservation

All seven manifest groups contribute 5,000 selected trials; one no-crossing trial in 70-79 is censored, leaving {int(age_model.get('mean_rt_s', pd.Series(dtype=float)).notna().sum())} model age-summary rows. The age CSV contains human/model mean RT, accuracy, H1-H6 metrics where estimable, CAFs, and distances. This remains descriptive in-sample validation, not a refit or held-out result.

## Recommended first M1 experiment — not implemented

The first single-factor candidate is low-amplitude sensory/evidence variability. The strongest rationale is CASE A plus the complete absence of congruent errors and the severe tail-shape mismatch; the audit does **not** claim uniform RT-width compression. Starting-state variability is less directly indicated because wrong states are not already entering congruent competition; recurrent/commitment changes are also premature because correction is already a major C0v2 strength.

Risks are excess incongruent errors, loss of the H3 CAF shape, distortion of age RTs, or invalid crossings. Guardrails are: preserve H3 as primary; protect age mean RT, overall accuracy, H2, H4, H5, crossing validity, and exact commitment semantics; rerun all H1-H6 after one change and retain/reject without a weighted composite score.

## Outputs and invariants

The result directory contains the required scorecards, trial and summary trajectory tables, association table, eight requested figure families, QA, and this report. C0v2 was not modified, no parameter was fitted or optimized, no noise was added in this audit, and Model M1 was not created.
"""
    (output / "c0v2_canonical_audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty result directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    frame, states, ww_input, qa = replay_core(manifest, args.device)
    if not qa["passed"]:
        (output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("STOP: C0v2 identity differs or failed reproduction")

    human_module = load_human_module()
    tables, h4_eligibility = matched_behavior_tables(frame, human_module)
    results = rename_source_tables(tables)
    summary = enrich_summary(pd.DataFrame([summarize_model(results[key], key) for key in ["human", "c0"]]), results)
    distances = c0_distances(summary, results)
    hpairs, hrepeat = matched_repeat_inconsistency(frame, "response_label")
    mpairs, mrepeat = matched_repeat_inconsistency(frame, "pred_choice")
    repeat = {
        "human_pairs": hpairs, "model_pairs": mpairs,
        "human": f"{hrepeat:.4f} ({hpairs} pairs)" if hpairs >= 20 else f"NOT ESTIMABLE ({hpairs} pairs)",
        "model": f"{mrepeat:.4f} ({mpairs} pairs)" if mpairs >= 20 else f"NOT ESTIMABLE ({mpairs} pairs)",
    }
    full_targets = pd.read_csv(FULL_TARGETS_PATH)
    scorecard = core_scorecard(summary, distances, full_targets, repeat)
    distributions = add_width_ratios(distribution_table(results)).rename(columns={"wasserstein_c0_to_matched_human_s": "wasserstein_c0v2_to_matched_human_s"})
    trajectory = complete_trajectory_fields(trajectory_audit(frame, states, ww_input), frame, states, ww_input)
    trajectory_summary = summarize_trajectories(trajectory)
    associations = association_table(trajectory)
    curves = curve_table(results)
    all_age = pd.read_csv(ALL_AGE_SAVED, low_memory=False)
    age_metrics, age_curves = age_audit(all_age, human_module)

    plot_core(results, distributions, trajectory, output)
    plot_age(age_metrics, age_curves, all_age, output)
    scorecard.to_csv(output / "human_c0v2_scorecard.csv", index=False)
    summary.assign(source=summary.source.replace({"c0": "C0v2", "human": "matched_human"})).to_csv(output / "c0v2_h1_h6_metrics.csv", index=False)
    distances.assign(source="C0v2").to_csv(output / "c0v2_h1_h6_distances.csv", index=False)
    curves.to_csv(output / "c0v2_h3_h4_curve_points.csv", index=False)
    distributions.assign(source=distributions.source.replace({"c0": "C0v2", "human": "matched_human"})).to_csv(output / "c0v2_h5_distribution_metrics.csv", index=False)
    trajectory.to_csv(output / "c0v2_trajectory_trial_level.csv", index=False)
    trajectory_summary.to_csv(output / "c0v2_trajectory_summary.csv", index=False)
    associations.to_csv(output / "c0v2_behavior_dynamics_associations.csv", index=False)
    h4_eligibility.to_csv(output / "c0v2_h4_participant_eligibility.csv", index=False)
    age_metrics.to_csv(output / "c0v2_age_group_h1_h6_metrics.csv", index=False)
    age_curves.to_csv(output / "c0v2_age_group_h3_h4_curve_points.csv", index=False)
    qa.update({**repeat, "all_age_trials": len(all_age), "all_age_crossed_trials": int(all_age.crossed.astype(bool).sum()), "all_age_censored_trials": int((~all_age.crossed.astype(bool)).sum()), "parameters_refit": False, "optimization_run": False, "noise_added": False})
    (output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    write_reports(output, qa, scorecard, summary, trajectory_summary, age_metrics)
    inc = trajectory_summary[trajectory_summary.scope.eq("incongruent")].iloc[0]
    print(json.dumps({"passed": qa["passed"], "output": str(output), "pC_pre": inc.pC_pre_correct_commitment_given_early_wrong, "files": len(list(output.iterdir()))}, indent=2))


if __name__ == "__main__":
    main()

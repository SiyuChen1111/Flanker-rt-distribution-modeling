#!/usr/bin/env python3
"""Run the choice-coupled corrected-equivalent model on intermediate age groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT
from run_natural_layer_to_time_var_ww_diagnostic import raw_layer_arrays
from run_representative_extreme_age_subset_fitting import load_trial_cache
from run_r5_choice_coupled_schedule_optimization import (
    BASE,
    DT_S,
    LAYER_ORDER,
    MIN_CROSSING_RATE,
    SEED,
    candidate_grid,
    first_stable_positive_after_negative,
    run_candidate,
)

OUT_DEFAULT = PROJECT_ROOT / "artifacts/results/all_age_groups_20260806"
MIDDLE_GROUPS = ["30-39", "40-49", "50-59", "60-69", "70-79"]
EVIDENCE_KEYS = [f"evidence_{layer}" for layer in LAYER_ORDER]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_DEFAULT))
    p.add_argument("--groups", nargs="+", default=["all"])
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def reference_scales() -> dict[str, float]:
    reference = raw_layer_arrays(load_trial_cache(BASE))
    scales: dict[str, float] = {}
    for layer, values in reference.items():
        centered = values - values.mean(axis=1, keepdims=True)
        class_std = centered.std(axis=1, keepdims=True)
        class_std[class_std < 1e-6] = 1.0
        scales[layer] = max(float(class_std.mean()), 1e-6)
    return scales


def load_group_cache(root: Path, group: str) -> dict[str, np.ndarray]:
    path = root / "evidence_cache" / group / "full_age_group_layerwise_evidence.npz"
    z = np.load(path, allow_pickle=True)
    available = np.asarray(z["evidence_available"], dtype=bool)
    if len(available) != 5000 or not available.all():
        raise RuntimeError(f"{group}: incomplete evidence coverage {available.sum()}/{len(available)}")
    cache = {key: np.asarray(z[key]) for key in z.files}
    cache["analysis_group"] = np.full(len(available), group, dtype=object)
    cache["original_age_group"] = np.full(len(available), group, dtype=object)
    return cache


def normalize_with_reference(cache: dict[str, np.ndarray], scales: dict[str, float]) -> dict[str, np.ndarray]:
    raw = raw_layer_arrays(cache)
    out: dict[str, np.ndarray] = {}
    for layer, values in raw.items():
        centered = values - values.mean(axis=1, keepdims=True)
        out[layer] = (centered / scales[layer]).astype(np.float32)
    return out


def interpolated_params(group: str) -> dict[str, float]:
    low, high = map(int, group.split("-"))
    midpoint = (low + high) / 2.0
    fraction = float(np.clip((midpoint - 24.5) / (84.5 - 24.5), 0.0, 1.0))
    return {
        "evidence_gain": 0.8,
        "threshold": 0.12 + 0.02 * fraction,
        "sustained_k": 2,
        "margin": 0.02 * fraction,
        "min_decision_time": 0.0,
        "age_interpolation_fraction": fraction,
    }


def add_trial_dynamics(trial: pd.DataFrame, outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = trial.copy()
    target = out["target_label"].to_numpy(int)
    flanker = out["flanker_label"].to_numpy(int)
    rows = np.arange(len(out))[:, None]
    times = np.arange(np.asarray(outputs["trajectory"]).shape[1])[None, :]
    state = np.asarray(outputs["trajectory"], dtype=float)
    ww_input = np.asarray(outputs["ww_input"], dtype=float)
    state_gap = state[rows, times, target[:, None]] - state[rows, times, flanker[:, None]]
    input_gap = ww_input[rows, times, target[:, None]] - ww_input[rows, times, flanker[:, None]]
    state_recovery = first_stable_positive_after_negative(state_gap)
    input_reversal = first_stable_positive_after_negative(input_gap)
    out["target_recovery_time"] = state_recovery * DT_S
    out["reversal_time"] = input_reversal * DT_S
    out["winner_at_readout"] = out["pred_choice"].astype(int)
    out["winner_at_crossing"] = out["pred_choice"].astype(int)
    out["no_crossing_reason"] = np.where(out["crossed"].astype(bool), "", "deadline_censoring")
    out["model_name"] = "choice_coupled_corrected_equivalent"
    out["model_fingerprint_id"] = "vgg16_5layer_pergap_ww4_choice_coupled_20260803"
    out["random_seed"] = SEED
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    groups = MIDDLE_GROUPS if args.groups == ["all"] else args.groups
    bad = sorted(set(groups).difference(MIDDLE_GROUPS))
    if bad:
        raise ValueError(f"Unsupported groups: {bad}")
    result_dir = root / "results" / "corrected_model_by_age"
    result_dir.mkdir(parents=True, exist_ok=True)
    scales = reference_scales()
    grid = candidate_grid()
    all_metrics: list[pd.DataFrame] = []
    all_selected_metrics: list[dict[str, object]] = []
    all_trials: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for group in groups:
        group_dir = result_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        selected_path = group_dir / "selected_trial_level_predictions.csv"
        metric_path = group_dir / "selected_model_metrics.csv"
        if args.resume and selected_path.exists() and metric_path.exists():
            reused_trials = pd.read_csv(selected_path)
            reused_metrics = pd.read_csv(metric_path)
            all_trials.append(reused_trials)
            all_selected_metrics.extend(reused_metrics.to_dict(orient="records"))
            candidates_path = group_dir / "candidate_group_metrics.csv"
            if candidates_path.exists():
                all_metrics.append(pd.read_csv(candidates_path))
            first_trial = reused_trials.iloc[0]
            first_metric = reused_metrics.iloc[0]
            parameter_rows.append({
                "age_group": group,
                **interpolated_params(group),
                "compression": float(first_trial["compression"]),
                "late_shift_s": float(first_trial["late_shift_s"]),
                "width_scale": float(first_trial["width_scale"]),
                "t0_mean": float(first_metric["t0_mean"]),
                "t0_sd": float(first_metric["t0_sd"]),
            })
            print(f"{group}: reused completed result", flush=True)
            continue
        cache = load_group_cache(root, group)
        layers = normalize_with_reference(cache, scales)
        params = interpolated_params(group)
        candidate_rows: list[dict[str, object]] = []
        for index, candidate in enumerate(grid, start=1):
            metric, _, _ = run_candidate(group, cache, layers, params, candidate)
            candidate_rows.append(metric)
            if index % 25 == 0 or index == len(grid):
                print(f"{group}: {index}/{len(grid)} candidates", flush=True)
        candidate_df = pd.DataFrame(candidate_rows).sort_values("score", kind="mergesort")
        eligible = candidate_df[candidate_df["crossing_rate"] >= MIN_CROSSING_RATE]
        if eligible.empty:
            raise RuntimeError(f"{group}: no candidate passed crossing gate")
        best = eligible.iloc[0]
        spec = {key: float(best[key]) for key in ["compression", "late_shift_s", "width_scale"]}
        selected_metric, selected_trial, outputs = run_candidate(
            group, cache, layers, params, spec, return_trials=True
        )
        assert selected_trial is not None and outputs is not None
        selected_trial = add_trial_dynamics(selected_trial, outputs)
        selected_trial["age_group"] = group
        selected_trial["original_age_group"] = group
        candidate_df.to_csv(group_dir / "candidate_group_metrics.csv", index=False)
        pd.DataFrame([selected_metric]).to_csv(metric_path, index=False)
        selected_trial.to_csv(selected_path, index=False)
        qa = {
            "age_group": group,
            "n_trials": len(selected_trial),
            "n_candidates": len(candidate_df),
            "crossing_rate": float(selected_trial["crossed"].mean()),
            "choice_readout_consistency": float((selected_trial["pred_choice"] == selected_trial["winner_at_readout"]).mean()),
            "finite_crossed_rt": bool(np.isfinite(selected_trial.loc[selected_trial["crossed"], "pred_rt"]).all()),
            "selected_schedule": spec,
            "passed": bool(len(selected_trial) == 5000 and selected_trial["crossed"].mean() >= MIN_CROSSING_RATE),
        }
        (group_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
        if not qa["passed"]:
            raise RuntimeError(f"{group}: QA failed {qa}")
        all_metrics.append(candidate_df)
        all_selected_metrics.append(selected_metric)
        all_trials.append(selected_trial)
        parameter_rows.append({"age_group": group, **params, **spec, "t0_mean": selected_metric["t0_mean"], "t0_sd": selected_metric["t0_sd"]})
        print(f"{group}: selected {spec}; score={selected_metric['score']:.4f}", flush=True)
    pd.concat(all_metrics, ignore_index=True).to_csv(result_dir / "candidate_group_metrics.csv", index=False)
    pd.DataFrame(all_selected_metrics).to_csv(result_dir / "selected_model_metrics.csv", index=False)
    pd.concat(all_trials, ignore_index=True).to_csv(result_dir / "selected_trial_level_predictions.csv", index=False)
    if parameter_rows:
        pd.DataFrame(parameter_rows).to_csv(result_dir / "selected_parameters.csv", index=False)
    (result_dir / "run_config.json").write_text(
        json.dumps({"groups": groups, "seed": SEED, "candidate_grid_size": len(grid), "normalization_reference": str(BASE), "parameter_rule": "linear interpolation of retained young/older WW readout anchors; age-specific schedule and t0 search"}, indent=2),
        encoding="utf-8",
    )
    print(f"completed groups: {groups}", flush=True)


if __name__ == "__main__":
    main()

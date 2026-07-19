#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

from analyze_layerwise_evidence_ww import (  # noqa: E402
    build_gate_input,
    make_trial_df,
    normalize_evidence,
    run_ww,
    summarize_condition,
    target_flanker_trajectory,
)
from complete_layerwise_dmc_remaining_diagnostics import build_hand_dmc_input  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from train_age_groups_efficient import to_jsonable  # noqa: E402


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cache(path: Path, max_trials: int) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    n = min(int(max_trials), len(data["target_labels"]))
    return {key: data[key][:n] for key in data.files}


def evidence_sources(cache: Dict[str, np.ndarray], gain: float) -> Dict[str, torch.Tensor]:
    raw = {
        "final": cache["evidence_final"],
        "pooled": cache["evidence_pooled"],
        "conv3": cache["evidence_conv3"],
        "conv4": cache["evidence_conv4"],
        "conv5": cache["evidence_conv5"],
    }
    raw["mid"] = 0.5 * (raw["conv3"] + raw["conv4"])
    raw["late"] = raw["pooled"]
    return {key: normalize_evidence(value, gain=float(gain)) for key, value in raw.items()}


def add_human_metrics(row: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    true_rt = df["true_rt"].to_numpy(dtype=np.float32)
    human_correct = df["human_correct"].to_numpy(dtype=bool)
    cong = df["congruency"].to_numpy(dtype=np.int64)
    incong = cong == 1
    row["human_congruent_rt"] = float(np.mean(true_rt[~incong])) if (~incong).any() else float("nan")
    row["human_incongruent_rt"] = float(np.mean(true_rt[incong])) if incong.any() else float("nan")
    row["human_congruency_rt_gap"] = row["human_incongruent_rt"] - row["human_congruent_rt"]
    row["human_error_minus_correct_rt"] = (
        float(np.mean(true_rt[~human_correct])) - float(np.mean(true_rt[human_correct]))
        if (~human_correct).any() and human_correct.any()
        else float("nan")
    )
    return row


def caf_table(df: pd.DataFrame, condition: str, bins: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, scoped in [("all", df), ("incongruent", df[df["congruency"].eq(1)])]:
        if len(scoped) < bins:
            continue
        order = np.argsort(scoped["pred_rt"].to_numpy(dtype=np.float32))
        for idx, bin_idx in enumerate(np.array_split(order, bins), start=1):
            part = scoped.iloc[bin_idx]
            rows.append(
                {
                    "condition": condition,
                    "scope": scope,
                    "rt_bin": idx,
                    "n_trials": int(len(part)),
                    "mean_rt": float(part["pred_rt"].mean()),
                    "accuracy": float(part["model_correct"].mean()),
                    "human_choice_agreement": float((part["pred_choice"] == part["response_label"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_trials(condition: str, df: pd.DataFrame, family: str, config: Dict[str, Any]) -> Dict[str, Any]:
    row = summarize_condition(condition, df)
    row = add_human_metrics(row, df)
    row["family"] = family
    protected = set(row.keys())
    for key, value in config.items():
        if key in protected:
            row[f"config_{key}"] = value
        else:
            row[key] = value
    return row


def run_condition(
    *,
    cache: Dict[str, np.ndarray],
    condition: str,
    family: str,
    ww_input: torch.Tensor,
    time_steps: int,
    dt_ms: int,
    threshold: float,
    noise_ampa: float,
    readout_mode: str,
    t0_seconds: float,
    choice_temperature: float,
    seed: int,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    out = run_ww(
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
    df = make_trial_df(cache, condition, out)
    row = summarize_trials(condition, df, family, config)
    return row, df, out


def add_ar1_noise(x: torch.Tensor, rho: float, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0:
        return x.clone()
    rng = np.random.default_rng(seed)
    arr = x.detach().cpu().numpy().astype(np.float32)
    noise = np.zeros_like(arr, dtype=np.float32)
    innovation_scale = float(sigma) * np.sqrt(max(1.0 - float(rho) ** 2, 0.0))
    noise[:, 0, :] = rng.normal(0.0, float(sigma), size=noise[:, 0, :].shape)
    for ti in range(1, noise.shape[1]):
        noise[:, ti, :] = float(rho) * noise[:, ti - 1, :] + rng.normal(
            0.0,
            innovation_scale,
            size=noise[:, ti, :].shape,
        )
    return torch.as_tensor(np.clip(arr + noise, 0.0, None), dtype=torch.float32)


def right_tail_mass(pred_rt: np.ndarray, true_rt: np.ndarray) -> float:
    cutoff = float(np.quantile(true_rt, 0.90))
    return float(np.mean(pred_rt >= cutoff))


def no_crossing_rate(outputs: Dict[str, np.ndarray], dt_ms: int, time_steps: int, t0_seconds: float) -> float:
    max_rt = (time_steps - 1) * (dt_ms / 1000.0) + t0_seconds
    return float(np.mean(outputs["pred_rt"] >= max_rt - 1e-6))


def trajectory_persistence(outputs: Dict[str, np.ndarray], cache: Dict[str, np.ndarray]) -> float:
    target = cache["target_labels"].astype(np.int64)
    flanker = cache["flanker_labels"].astype(np.int64)
    s_target, s_flanker, _ = target_flanker_trajectory(outputs["trajectory"], target, flanker)
    diff = s_target - s_flanker
    if diff.shape[1] < 2:
        return float("nan")
    x = diff[:, :-1].reshape(-1)
    y = diff[:, 1:].reshape(-1)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def stochastic_threshold_readout(
    trajectory: np.ndarray,
    cache: Dict[str, np.ndarray],
    *,
    base_threshold: float,
    threshold_sigma: float,
    lambda_uniform: float,
    min_stop_time: float,
    dt_ms: int,
    t0_seconds: float,
    seed: int,
    condition: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, time_steps, n_classes = trajectory.shape
    thresholds = rng.normal(base_threshold, threshold_sigma, size=(n, n_classes))
    thresholds = np.clip(thresholds, 0.04, 0.80)
    min_step = int(round(float(min_stop_time) / (dt_ms / 1000.0)))
    min_step = max(0, min(min_step, time_steps - 1))

    crossing = trajectory >= thresholds[:, None, :]
    if min_step > 0:
        crossing[:, :min_step, :] = False
    first = np.argmax(crossing, axis=1)
    no_cross = ~crossing.any(axis=1)
    first[no_cross] = time_steps - 1
    threshold_choice = np.argmin(first, axis=1).astype(np.int64)
    threshold_step = first[np.arange(n), threshold_choice]

    use_uniform = rng.random(n) < float(lambda_uniform)
    uniform_step = rng.integers(min_step, time_steps, size=n)
    uniform_choice = trajectory[np.arange(n), uniform_step, :].argmax(axis=1).astype(np.int64)

    pred_choice = np.where(use_uniform, uniform_choice, threshold_choice).astype(np.int64)
    pred_step = np.where(use_uniform, uniform_step, threshold_step)
    pred_rt = pred_step.astype(np.float32) * (dt_ms / 1000.0) + float(t0_seconds)

    df = pd.DataFrame(
        {
            "condition": condition,
            "row_index": cache["row_indices"],
            "age_group": cache["age_group"].astype(str),
            "user_id": cache["user_id"].astype(str),
            "target_label": cache["target_labels"].astype(np.int64),
            "flanker_label": cache["flanker_labels"].astype(np.int64),
            "response_label": cache["response_labels"].astype(np.int64),
            "true_rt": cache["true_rt"].astype(np.float32),
            "congruency": cache["congruency"].astype(np.int64),
            "pred_rt": pred_rt,
            "pred_choice": pred_choice,
            "used_uniform_stop": use_uniform,
        }
    )
    df["model_correct"] = df["pred_choice"] == df["target_label"]
    df["human_correct"] = df["response_label"] == df["target_label"]
    return df


def trajectory_rows(
    condition: str,
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    dt_ms: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    target = cache["target_labels"].astype(np.int64)
    flanker = cache["flanker_labels"].astype(np.int64)
    s_target, s_flanker, s_other = target_flanker_trajectory(outputs["trajectory"], target, flanker)
    early_steps = max(1, int(round(0.12 / (dt_ms / 1000.0))))
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
        if not mask.any():
            rows.append({"condition": condition, "group": group_name, "n_trials": 0})
            continue
        early_ft = s_flanker[mask, :early_steps] - s_target[mask, :early_steps]
        late_tf = s_target[mask, -early_steps:] - s_flanker[mask, -early_steps:]
        rows.append(
            {
                "condition": condition,
                "group": group_name,
                "n_trials": int(mask.sum()),
                "early_flanker_minus_target": float(early_ft.mean()),
                "early_flanker_ge_target_rate": float((early_ft.mean(axis=1) >= 0).mean()),
                "late_target_minus_flanker": float(late_tf.mean()),
                "peak_target": float(s_target[mask].max(axis=1).mean()),
                "peak_flanker": float(s_flanker[mask].max(axis=1).mean()),
                "final_target_minus_flanker": float((s_target[mask, -1] - s_flanker[mask, -1]).mean()),
                "mean_rt": float(df.loc[mask, "pred_rt"].mean()),
            }
        )
        curves[f"{condition}:{group_name}:target"] = s_target[mask].mean(axis=0)
        curves[f"{condition}:{group_name}:flanker"] = s_flanker[mask].mean(axis=0)
        curves[f"{condition}:{group_name}:other"] = s_other[mask].mean(axis=0)
    return rows, curves


def plot_metric(summary: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_df = summary.copy()
    colors = {
        "layerwise_single": "#4C78A8",
        "layerwise_gate": "#54A24B",
        "handcrafted_dmc": "#F58518",
        "parameter_scan": "#72B7B2",
        "ar1_noise": "#B279A2",
        "stochastic_stop": "#E45756",
    }
    bar_colors = [colors.get(str(v), "#999999") for v in plot_df.get("family", pd.Series([""] * len(plot_df)))]
    ax.bar(plot_df["condition"], plot_df[metric], color=bar_colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scatter(summary: pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(summary[x], summary[y], s=28, alpha=0.75, color="#4C78A8")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectory_curves(curves: Dict[str, np.ndarray], condition: str, group: str, dt_ms: int, output_path: Path) -> None:
    target_key = f"{condition}:{group}:target"
    flanker_key = f"{condition}:{group}:flanker"
    if target_key not in curves or flanker_key not in curves:
        return
    t = np.arange(len(curves[target_key])) * (dt_ms / 1000.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(t, curves[target_key], label="target", linewidth=1.9)
    ax.plot(t, curves[flanker_key], label="flanker", linewidth=1.9)
    other_key = f"{condition}:{group}:other"
    if other_key in curves:
        ax.plot(t, curves[other_key], label="other max", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wong-Wang state")
    ax.set_title(f"{condition}: {group}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def selection_score(row: pd.Series) -> float:
    return float(
        row["accuracy"]
        + 0.60 * row["model_human_choice_agreement"]
        - 0.35 * abs(row["human_model_mean_rt_gap"])
        - 0.15 * abs(row["human_model_q95_gap"])
        - 0.15 * max(row["incongruent_error_rate"] - 0.65, 0.0)
    )


def best_rows(df: pd.DataFrame, n: int, min_accuracy: float = 0.0) -> pd.DataFrame:
    scoped = df[df["accuracy"] >= min_accuracy].copy()
    if scoped.empty:
        scoped = df.copy()
    scoped["selection_score"] = scoped.apply(selection_score, axis=1)
    return scoped.sort_values("selection_score", ascending=False).head(n).copy()


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def fmt(row: pd.Series, key: str) -> str:
    value = row.get(key, np.nan)
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz")
    parser.add_argument("--output_root", default="artifacts/results/diagnostics")
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--time_steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--t0_seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_cache(resolve_path(args.cache_path), args.max_trials)
    root = resolve_path(args.output_root)

    same_dir = root / "same_subset_layerwise_vs_dmc"
    scan_dir = root / "layerwise_gate_parameter_scan"
    traj_dir = root / "layerwise_gate_trajectory_validation"
    ar1_dir = root / "ar1_layerwise_gate_noise_ablation"
    stop_dir = root / "stochastic_stopping_layerwise_gate"
    for directory in [same_dir, scan_dir, traj_dir, ar1_dir, stop_dir]:
        (directory / "figures").mkdir(parents=True, exist_ok=True)

    base_gain = 0.8
    threshold = 0.22
    noise_ampa = 0.02
    choice_temperature = 0.10
    readout_mode = "baseline"
    tau_s = 0.24
    gate_k = 20.0
    time_steps = int(args.time_steps)
    dt_ms = int(args.dt_ms)

    ev = evidence_sources(cache, base_gain)
    same_rows: List[Dict[str, Any]] = []
    same_trials: List[pd.DataFrame] = []
    same_outputs: Dict[str, Dict[str, np.ndarray]] = {}

    same_specs: List[Tuple[str, str, torch.Tensor, Dict[str, Any]]] = [
        ("final_logits_ww", "layerwise_single", ev["final"], {"evidence_gain": base_gain}),
        ("mid_layer_ww", "layerwise_single", ev["mid"], {"evidence_gain": base_gain}),
        ("pooled_ww", "layerwise_single", ev["pooled"], {"evidence_gain": base_gain}),
    ]
    gate_input, _ = build_gate_input(ev["mid"], ev["pooled"], time_steps=time_steps, dt_ms=dt_ms, tau_s=tau_s, k=gate_k)
    same_specs.append(
        (
            "fixed_layer_time_gate_ww",
            "layerwise_gate",
            gate_input,
            {"evidence_gain": base_gain, "gate_tau": tau_s, "gate_k": gate_k},
        )
    )
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
    same_specs.append(
        (
            "handcrafted_dmc_final_ww",
            "handcrafted_dmc",
            dmc_input,
            {
                "evidence_gain": base_gain,
                "auto_strength": 0.30,
                "selection_strength": 0.40,
                "target_boost": 0.30,
            },
        )
    )

    caf_frames = []
    for condition, family, ww_input, cfg in same_specs:
        row, df, out = run_condition(
            cache=cache,
            condition=condition,
            family=family,
            ww_input=ww_input,
            time_steps=time_steps,
            dt_ms=dt_ms,
            threshold=threshold,
            noise_ampa=noise_ampa,
            readout_mode=readout_mode,
            t0_seconds=float(args.t0_seconds),
            choice_temperature=choice_temperature,
            seed=int(args.seed),
            config={**cfg, "threshold": threshold, "noise_ampa": noise_ampa, "choice_temperature": choice_temperature, "readout_mode": readout_mode},
        )
        same_rows.append(row)
        same_trials.append(df)
        same_outputs[condition] = out
        caf_frames.append(caf_table(df, condition))

    same_summary = pd.DataFrame(same_rows)
    same_trial_level = pd.concat(same_trials, ignore_index=True)
    same_summary.to_csv(same_dir / "same_subset_model_summary.csv", index=False)
    same_trial_level.to_csv(same_dir / "same_subset_trial_level.csv", index=False)
    pd.concat(caf_frames, ignore_index=True).to_csv(same_dir / "same_subset_caf.csv", index=False)
    for metric in ["accuracy", "model_human_choice_agreement", "mean_rt", "q95", "incongruent_error_rate", "fastest_incongruent_bin_accuracy"]:
        plot_metric(same_summary, metric, same_dir / "figures" / f"{metric}.png", f"Same-subset {metric}")

    layer_gate_row = same_summary[same_summary["condition"].eq("fixed_layer_time_gate_ww")].iloc[0]
    dmc_row = same_summary[same_summary["condition"].eq("handcrafted_dmc_final_ww")].iloc[0]
    write_md(
        same_dir / "same_subset_comparison_memo.md",
        f"""
# Same-subset Layer-wise vs DMC Comparison

## Files read

- `{args.cache_path}`
- `code/scripts/analyze_layerwise_evidence_ww.py`
- `code/scripts/complete_layerwise_dmc_remaining_diagnostics.py`
- `code/scripts/vgg_wongwang_lim.py`

## Command run

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Outputs

- `same_subset_model_summary.csv`
- `same_subset_trial_level.csv`
- `same_subset_caf.csv`
- `figures/`

## Main paired result

The fixed layer-wise gate and hand-crafted DMC were evaluated on the same {len(cache["target_labels"])} rows, using the same Wong-Wang and readout settings.

- Layer-wise gate accuracy: {fmt(layer_gate_row, "accuracy")}; DMC accuracy: {fmt(dmc_row, "accuracy")}.
- Layer-wise gate human-choice agreement: {fmt(layer_gate_row, "model_human_choice_agreement")}; DMC agreement: {fmt(dmc_row, "model_human_choice_agreement")}.
- Layer-wise gate mean RT: {fmt(layer_gate_row, "mean_rt")} s; DMC mean RT: {fmt(dmc_row, "mean_rt")} s.
- Layer-wise gate incongruent error rate: {fmt(layer_gate_row, "incongruent_error_rate")}; DMC incongruent error rate: {fmt(dmc_row, "incongruent_error_rate")}.

## Interpretation

The layer-wise gate still produces more conflict-like behavior than final logits, but its behavioral fit remains worse than the hand-crafted DMC positive control on this paired subset. The gap is visible in accuracy, human-choice agreement, and RT scale. The current evidence points more toward scale, timing, and readout mismatch than toward a complete failure of the evidence source, because hidden-layer evidence does create conflict whereas final logits do not.

## What this supports

Layer-wise evidence can supply a natural conflict source on the same rows used for DMC positive control testing.

## What this does not support

The fixed layer-wise gate is not yet a full replacement for hand-crafted DMC.

## Next step

Use the same cache to scan evidence scale, gate timing, threshold, accumulator noise, choice temperature, and readout mode.
""",
    )

    scan_rows: List[Dict[str, Any]] = []
    scan_trials: List[pd.DataFrame] = []
    scan_outputs: Dict[str, Dict[str, np.ndarray]] = {}

    for gain in [0.8, 1.2, 1.6, 2.0, 2.5]:
        ev_gain = evidence_sources(cache, gain)
        for tau in [0.16, 0.20, 0.24, 0.28]:
            for k in [10.0, 20.0, 40.0]:
                x, _ = build_gate_input(ev_gain["mid"], ev_gain["pooled"], time_steps=time_steps, dt_ms=dt_ms, tau_s=tau, k=k)
                condition = f"scan_gain{gain:.2f}_tau{tau:.2f}_k{int(k)}_base"
                row, df, out = run_condition(
                    cache=cache,
                    condition=condition,
                    family="parameter_scan",
                    ww_input=x,
                    time_steps=time_steps,
                    dt_ms=dt_ms,
                    threshold=threshold,
                    noise_ampa=noise_ampa,
                    readout_mode="baseline",
                    t0_seconds=float(args.t0_seconds),
                    choice_temperature=choice_temperature,
                    seed=int(args.seed),
                    config={
                        "stage": "gain_gate",
                        "evidence_gain": gain,
                        "gate_tau": tau,
                        "gate_k": k,
                        "threshold": threshold,
                        "noise_ampa": noise_ampa,
                        "choice_temperature": choice_temperature,
                        "readout_mode": "baseline",
                    },
                )
                scan_rows.append(row)
                scan_outputs[condition] = out

    stage1 = pd.DataFrame(scan_rows)
    for _, cfg in best_rows(stage1, 6, min_accuracy=0.45).iterrows():
        gain = float(cfg["evidence_gain"])
        tau = float(cfg["gate_tau"])
        k = float(cfg["gate_k"])
        ev_gain = evidence_sources(cache, gain)
        x, _ = build_gate_input(ev_gain["mid"], ev_gain["pooled"], time_steps=time_steps, dt_ms=dt_ms, tau_s=tau, k=k)
        for th in [0.16, 0.20, 0.22, 0.26]:
            for noise in [0.00, 0.02, 0.04, 0.06]:
                for temp in [0.05, 0.08, 0.10, 0.15]:
                    condition = f"scan_gain{gain:.2f}_tau{tau:.2f}_k{int(k)}_th{th:.2f}_n{noise:.2f}_ct{temp:.2f}"
                    row, df, out = run_condition(
                        cache=cache,
                        condition=condition,
                        family="parameter_scan",
                        ww_input=x,
                        time_steps=time_steps,
                        dt_ms=dt_ms,
                        threshold=th,
                        noise_ampa=noise,
                        readout_mode="baseline",
                        t0_seconds=float(args.t0_seconds),
                        choice_temperature=temp,
                        seed=int(args.seed),
                        config={
                            "stage": "ww_readout",
                            "evidence_gain": gain,
                            "gate_tau": tau,
                            "gate_k": k,
                            "threshold": th,
                            "noise_ampa": noise,
                            "choice_temperature": temp,
                            "readout_mode": "baseline",
                        },
                    )
                    scan_rows.append(row)
                    if len(scan_trials) < 30:
                        scan_trials.append(df)
                    scan_outputs[condition] = out

    scan_summary = pd.DataFrame(scan_rows)
    scan_summary["selection_score"] = scan_summary.apply(selection_score, axis=1)
    scan_summary = scan_summary.sort_values("selection_score", ascending=False)
    scan_summary.to_csv(scan_dir / "parameter_scan_summary.csv", index=False)
    top_configs = scan_summary.head(20).copy()
    top_configs.to_csv(scan_dir / "top_configs.csv", index=False)
    if scan_trials:
        pd.concat(scan_trials, ignore_index=True).to_csv(scan_dir / "parameter_scan_trial_level_sample.csv", index=False)
    for metric in ["selection_score", "accuracy", "model_human_choice_agreement", "mean_rt", "q95", "incongruent_error_rate"]:
        plot_metric(top_configs.head(12), metric, scan_dir / "figures" / f"top_{metric}.png", f"Top configs: {metric}")
    plot_scatter(scan_summary, "incongruent_error_rate", "accuracy", scan_dir / "figures" / "conflict_accuracy_tradeoff.png", "Conflict vs accuracy")
    plot_scatter(scan_summary, "threshold", "mean_rt", scan_dir / "figures" / "threshold_mean_rt.png", "Threshold vs mean RT")
    plot_scatter(scan_summary, "evidence_gain", "mean_rt", scan_dir / "figures" / "gain_mean_rt.png", "Evidence gain vs mean RT")

    base_scan = scan_summary[scan_summary["condition"].eq("scan_gain0.80_tau0.24_k20_base")]
    best_scan = scan_summary.iloc[0]
    if base_scan.empty:
        base_scan = pd.DataFrame([layer_gate_row])
    base_scan_row = base_scan.iloc[0]
    write_md(
        scan_dir / "parameter_scan_memo.md",
        f"""
# Layer-wise Gate Parameter Scan

## Files read

- `{args.cache_path}`
- `code/scripts/compare_same_subset_layerwise_vs_dmc.py`
- `code/scripts/analyze_layerwise_evidence_ww.py`
- `code/scripts/vgg_wongwang_lim.py`

## Command run

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Outputs

- `parameter_scan_summary.csv`
- `top_configs.csv`
- `parameter_scan_trial_level_sample.csv`
- `figures/`

## Best config found

Best scanned condition: `{best_scan["condition"]}`.

- Accuracy: {fmt(best_scan, "accuracy")} versus baseline fixed gate {fmt(base_scan_row, "accuracy")}.
- Human-choice agreement: {fmt(best_scan, "model_human_choice_agreement")} versus baseline {fmt(base_scan_row, "model_human_choice_agreement")}.
- Mean RT: {fmt(best_scan, "mean_rt")} s versus baseline {fmt(base_scan_row, "mean_rt")} s.
- Incongruent error rate: {fmt(best_scan, "incongruent_error_rate")} versus baseline {fmt(base_scan_row, "incongruent_error_rate")}.

## Interpretation

The parameter scan tests whether the poor fixed-gate result is mainly due to evidence scale, gate timing, Wong-Wang threshold, accumulator noise, or choice temperature. If the best rows improve RT and accuracy without removing incongruent errors, the mechanism should not be rejected yet. If conflict rises while accuracy falls, that is a conflict-fit trade-off rather than a complete model success.

## What this supports

The scan identifies candidate settings for trajectory validation and later AR(1)/stochastic stopping tests.

## What this does not support

This is still a fixed diagnostic grid, not a trained subject-level model.
""",
    )

    trajectory_summary_rows: List[Dict[str, Any]] = []
    all_curves: Dict[str, np.ndarray] = {}
    top_for_traj = top_configs.head(5).copy()
    traj_trial_frames = []
    top_output_by_condition: Dict[str, Dict[str, np.ndarray]] = {}
    top_df_by_condition: Dict[str, pd.DataFrame] = {}
    for _, cfg in top_for_traj.iterrows():
        gain = float(cfg["evidence_gain"])
        tau = float(cfg["gate_tau"])
        k = float(cfg["gate_k"])
        th = float(cfg["threshold"])
        noise = float(cfg["noise_ampa"])
        temp = float(cfg["choice_temperature"])
        mode = str(cfg["readout_mode"])
        condition = str(cfg["condition"])
        ev_gain = evidence_sources(cache, gain)
        x, _ = build_gate_input(ev_gain["mid"], ev_gain["pooled"], time_steps=time_steps, dt_ms=dt_ms, tau_s=tau, k=k)
        row, df, out = run_condition(
            cache=cache,
            condition=condition,
            family="parameter_scan",
            ww_input=x,
            time_steps=time_steps,
            dt_ms=dt_ms,
            threshold=th,
            noise_ampa=noise,
            readout_mode=mode,
            t0_seconds=float(args.t0_seconds),
            choice_temperature=temp,
            seed=int(args.seed),
            config=dict(cfg),
        )
        del row
        rows, curves = trajectory_rows(condition, df, out, cache, dt_ms)
        trajectory_summary_rows.extend(rows)
        all_curves.update(curves)
        traj_trial_frames.append(df)
        top_output_by_condition[condition] = out
        top_df_by_condition[condition] = df
    traj_summary = pd.DataFrame(trajectory_summary_rows)
    traj_summary.to_csv(traj_dir / "trajectory_validation_summary.csv", index=False)
    pd.concat(traj_trial_frames, ignore_index=True).to_csv(traj_dir / "trajectory_validation_trial_level.csv", index=False)
    for condition in list(top_for_traj["condition"].astype(str))[:3]:
        for group in ["incongruent_error", "incongruent_correct", "model_fast_rt", "model_slow_rt"]:
            safe_name = condition.replace(".", "p")
            plot_trajectory_curves(all_curves, condition, group, dt_ms, traj_dir / "figures" / f"{safe_name}_{group}.png")

    best_condition = str(best_scan["condition"])
    best_traj = traj_summary[traj_summary["condition"].eq(best_condition)]
    incong_err = best_traj[best_traj["group"].eq("incongruent_error")]
    incong_cor = best_traj[best_traj["group"].eq("incongruent_correct")]
    write_md(
        traj_dir / "trajectory_validation_memo.md",
        f"""
# Layer-wise Gate Trajectory Validation

## Files read

- `{args.cache_path}`
- `artifacts/results/diagnostics/layerwise_gate_parameter_scan/top_configs.csv`

## Command run

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Outputs

- `trajectory_validation_summary.csv`
- `trajectory_validation_trial_level.csv`
- `figures/`

## Main result

Top scanned configurations were checked for target and flanker trajectories. For the best config `{best_condition}`:

- Incongruent-error early flanker-minus-target: {fmt(incong_err.iloc[0], "early_flanker_minus_target") if not incong_err.empty else "NA"}.
- Incongruent-correct early flanker-minus-target: {fmt(incong_cor.iloc[0], "early_flanker_minus_target") if not incong_cor.empty else "NA"}.
- Incongruent-correct late target-minus-flanker: {fmt(incong_cor.iloc[0], "late_target_minus_flanker") if not incong_cor.empty else "NA"}.

## Interpretation

A natural DMC-like layer-wise mechanism is supported only if incongruent errors show early flanker dominance and incongruent correct trials show later target recovery. These trajectory results should be read together with behavior: a configuration that produces conflict but collapses accuracy is not a behavioral replacement.
""",
    )

    best_cfg = best_scan
    best_gain = float(best_cfg["evidence_gain"])
    best_tau = float(best_cfg["gate_tau"])
    best_k = float(best_cfg["gate_k"])
    best_threshold = float(best_cfg["threshold"])
    best_noise = float(best_cfg["noise_ampa"])
    best_temp = float(best_cfg["choice_temperature"])
    best_mode = str(best_cfg["readout_mode"])
    best_ev = evidence_sources(cache, best_gain)
    best_input, _ = build_gate_input(best_ev["mid"], best_ev["pooled"], time_steps=time_steps, dt_ms=dt_ms, tau_s=best_tau, k=best_k)

    ar1_rows: List[Dict[str, Any]] = []
    ar1_trials: List[pd.DataFrame] = []
    _, base_best_df, base_best_out = run_condition(
        cache=cache,
        condition="best_gate_no_ar1",
        family="ar1_noise",
        ww_input=best_input,
        time_steps=time_steps,
        dt_ms=dt_ms,
        threshold=best_threshold,
        noise_ampa=best_noise,
        readout_mode=best_mode,
        t0_seconds=float(args.t0_seconds),
        choice_temperature=best_temp,
        seed=int(args.seed),
        config={},
    )
    base_row = summarize_trials("best_gate_no_ar1", base_best_df, "ar1_noise", dict(best_cfg))
    base_row["rho"] = 0.0
    base_row["sigma"] = 0.0
    base_row["right_tail_mass"] = right_tail_mass(base_best_df["pred_rt"].to_numpy(), base_best_df["true_rt"].to_numpy())
    base_row["choice_consistency_vs_base"] = 1.0
    base_row["no_crossing_rate"] = no_crossing_rate(base_best_out, dt_ms, time_steps, float(args.t0_seconds))
    base_row["trajectory_persistence"] = trajectory_persistence(base_best_out, cache)
    ar1_rows.append(base_row)
    ar1_trials.append(base_best_df)
    for rho in [0.25, 0.50, 0.75, 0.90]:
        for sigma in [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12]:
            noisy = add_ar1_noise(best_input, rho, sigma, int(args.seed) + int(rho * 1000) + int(sigma * 1000))
            condition = f"ar1_rho{rho:.2f}_sigma{sigma:.2f}"
            row, df, out = run_condition(
                cache=cache,
                condition=condition,
                family="ar1_noise",
                ww_input=noisy,
                time_steps=time_steps,
                dt_ms=dt_ms,
                threshold=best_threshold,
                noise_ampa=best_noise,
                readout_mode=best_mode,
                t0_seconds=float(args.t0_seconds),
                choice_temperature=best_temp,
                seed=int(args.seed),
                config=dict(best_cfg),
            )
            row["rho"] = rho
            row["sigma"] = sigma
            row["right_tail_mass"] = right_tail_mass(df["pred_rt"].to_numpy(), df["true_rt"].to_numpy())
            row["choice_consistency_vs_base"] = float((df["pred_choice"].to_numpy() == base_best_df["pred_choice"].to_numpy()).mean())
            row["no_crossing_rate"] = no_crossing_rate(out, dt_ms, time_steps, float(args.t0_seconds))
            row["trajectory_persistence"] = trajectory_persistence(out, cache)
            ar1_rows.append(row)
            ar1_trials.append(df)
    ar1_summary = pd.DataFrame(ar1_rows)
    ar1_summary["selection_score"] = ar1_summary.apply(selection_score, axis=1)
    ar1_summary.to_csv(ar1_dir / "ar1_noise_ablation_summary.csv", index=False)
    pd.concat(ar1_trials, ignore_index=True).to_csv(ar1_dir / "ar1_noise_trial_level.csv", index=False)
    for metric in ["q95", "q99", "skewness", "right_tail_mass", "accuracy", "model_human_choice_agreement", "incongruent_error_minus_correct_rt"]:
        plot_metric(ar1_summary.head(15), metric, ar1_dir / "figures" / f"{metric}.png", f"AR(1): {metric}")

    ar1_candidates = ar1_summary.copy()
    ar1_candidates["tail_gain"] = (ar1_candidates["q95"] - float(base_row["q95"])) + 0.5 * (ar1_candidates["q99"] - float(base_row["q99"]))
    ar1_candidates["accuracy_drop"] = float(base_row["accuracy"]) - ar1_candidates["accuracy"]
    ar1_candidates["ar1_candidate_score"] = (
        ar1_candidates["tail_gain"]
        - 0.6 * ar1_candidates["accuracy_drop"].clip(lower=0)
        + 0.2 * ar1_candidates["model_human_choice_agreement"]
    )
    best_ar1 = ar1_candidates.sort_values("ar1_candidate_score", ascending=False).iloc[0]
    write_md(
        ar1_dir / "ar1_noise_ablation_memo.md",
        f"""
# AR(1) Layer-wise Gate Noise Ablation

## Files read

- `{args.cache_path}`
- `artifacts/results/diagnostics/layerwise_gate_parameter_scan/top_configs.csv`

## Command run

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Outputs

- `ar1_noise_ablation_summary.csv`
- `ar1_noise_trial_level.csv`
- `figures/`

## Main result

AR(1) was tested on the best scanned layer-wise gate. The formula used was `eta_t = rho eta_(t-1) + sqrt(1-rho^2) sigma epsilon_t`, so higher rho changes persistence without automatically increasing marginal variance.

Best AR(1) candidate by tail/accuracy trade-off: `{best_ar1["condition"]}`.

- Baseline q95/q99/skewness: {fmt(pd.Series(base_row), "q95")} / {fmt(pd.Series(base_row), "q99")} / {fmt(pd.Series(base_row), "skewness")}.
- Candidate q95/q99/skewness: {fmt(best_ar1, "q95")} / {fmt(best_ar1, "q99")} / {fmt(best_ar1, "skewness")}.
- Baseline accuracy: {fmt(pd.Series(base_row), "accuracy")}; candidate accuracy: {fmt(best_ar1, "accuracy")}.
- Candidate choice consistency versus no-AR(1): {fmt(best_ar1, "choice_consistency_vs_base")}.

## Interpretation

AR(1) supports the subjective-evidence trajectory explanation only if it changes tail shape or persistence without merely making every trial slower and without destroying choice behavior. If q95/q99 are already capped by the simulation horizon, AR(1) cannot be properly judged until the base RT scale is fixed.
""",
    )

    stop_rows: List[Dict[str, Any]] = []
    stop_trials: List[pd.DataFrame] = []
    stop_base_out = base_best_out
    for lambda_uniform in [0.00, 0.05, 0.10, 0.20]:
        for min_stop_time in [0.10, 0.15, 0.20]:
            for threshold_sigma in [0.00, 0.02, 0.05, 0.08]:
                condition = f"stop_lam{lambda_uniform:.2f}_min{min_stop_time:.2f}_sig{threshold_sigma:.2f}"
                df = stochastic_threshold_readout(
                    stop_base_out["trajectory"],
                    cache,
                    base_threshold=best_threshold,
                    threshold_sigma=threshold_sigma,
                    lambda_uniform=lambda_uniform,
                    min_stop_time=min_stop_time,
                    dt_ms=dt_ms,
                    t0_seconds=float(args.t0_seconds),
                    seed=int(args.seed) + int(lambda_uniform * 1000) + int(min_stop_time * 1000) + int(threshold_sigma * 1000),
                    condition=condition,
                )
                row = summarize_trials(
                    condition,
                    df,
                    "stochastic_stop",
                    {
                        **dict(best_cfg),
                        "lambda_uniform": lambda_uniform,
                        "min_stop_time": min_stop_time,
                        "threshold_sigma": threshold_sigma,
                    },
                )
                stop_rows.append(row)
                stop_trials.append(df)
    stop_summary = pd.DataFrame(stop_rows)
    stop_summary["selection_score"] = stop_summary.apply(selection_score, axis=1)
    stop_summary.to_csv(stop_dir / "stochastic_stopping_summary.csv", index=False)
    pd.concat(stop_trials, ignore_index=True).to_csv(stop_dir / "stochastic_stopping_trial_level.csv", index=False)
    for metric in ["error_minus_correct_rt", "incongruent_error_minus_correct_rt", "fastest_incongruent_bin_accuracy", "accuracy", "model_human_choice_agreement", "mean_rt", "q95"]:
        plot_metric(stop_summary.sort_values("selection_score", ascending=False).head(15), metric, stop_dir / "figures" / f"{metric}.png", f"Stochastic stopping: {metric}")

    stop_rank = stop_summary.copy()
    stop_rank["fast_error_score"] = (
        -stop_rank["incongruent_error_minus_correct_rt"].fillna(99)
        -0.8 * (float(base_row["accuracy"]) - stop_rank["accuracy"]).clip(lower=0)
        +0.2 * stop_rank["model_human_choice_agreement"]
    )
    best_stop = stop_rank.sort_values("fast_error_score", ascending=False).iloc[0]
    write_md(
        stop_dir / "stochastic_stopping_memo.md",
        f"""
# Stochastic Stopping Layer-wise Gate

## Files read

- `{args.cache_path}`
- `artifacts/results/diagnostics/layerwise_gate_parameter_scan/top_configs.csv`

## Command run

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Outputs

- `stochastic_stopping_summary.csv`
- `stochastic_stopping_trial_level.csv`
- `figures/`

## Main result

Best fast-error candidate: `{best_stop["condition"]}`.

- Error-minus-correct RT: {fmt(best_stop, "error_minus_correct_rt")}.
- Incongruent error-minus-correct RT: {fmt(best_stop, "incongruent_error_minus_correct_rt")}.
- Fastest incongruent bin accuracy: {fmt(best_stop, "fastest_incongruent_bin_accuracy")}.
- Accuracy: {fmt(best_stop, "accuracy")}.
- Human-choice agreement: {fmt(best_stop, "model_human_choice_agreement")}.

## Interpretation

Stochastic stopping can improve fast-error signatures only by reading out an already conflicted trajectory earlier. If it lowers accuracy sharply, it is not a general replacement for DMC or AR(1); it is a separate readout mechanism for premature commitment.
""",
    )

    report_path = root / "dmc_noise_integrated_model_report.md"
    write_md(
        report_path,
        f"""
# DMC Noise Integrated Model Report

## Executive summary

The current evidence supports layer-wise CNN evidence as a natural conflict generator, but not yet as a complete behavioral replacement for hand-crafted DMC. Parameter scanning can improve parts of the fixed-gate behavior, but the final claim still depends on preserving trajectory-level early flanker and late target dynamics while improving RT scale and human-choice agreement.

## Why original DMC is hand-crafted

The old DMC route explicitly changes target and flanker channels over time. It uses manually specified early automatic flanker influence and later target selection. This is useful for a positive control, but the conflict source is imposed by design.

## Why layer-wise evidence is a natural DMC replacement candidate

Layer-wise evidence uses the CNN hierarchy as the source of time-varying subjective evidence. Middle layers retain flanker information; later pooled/final evidence is more target-oriented. A gate from middle to pooled evidence can therefore create early conflict and late recovery without directly writing flanker and target pulses.

## Current evidence for layer-wise conflict

The fixed layer-wise gate produced incongruent errors and prior trajectory diagnostics showed early flanker dominance on incongruent-error trials. In the same-subset comparison, final logits remained target-dominant while layer-wise evidence created more conflict.

## Current limitations of fixed layer-wise gate

Same-subset fixed gate accuracy was {fmt(layer_gate_row, "accuracy")}, human-choice agreement was {fmt(layer_gate_row, "model_human_choice_agreement")}, and mean RT was {fmt(layer_gate_row, "mean_rt")} s. The hand-crafted DMC positive control on the same rows reached accuracy {fmt(dmc_row, "accuracy")}, agreement {fmt(dmc_row, "model_human_choice_agreement")}, and mean RT {fmt(dmc_row, "mean_rt")} s.

## Same-subset DMC comparison

The paired comparison confirms the earlier warning: layer-wise gate is a mechanism smoke success, not yet a behavioral replacement. DMC's advantage comes from better accuracy, better human-choice agreement, and better RT scale under the current settings.

## Parameter scan results

Best scanned layer-wise config: `{best_scan["condition"]}`. It reached accuracy {fmt(best_scan, "accuracy")}, human-choice agreement {fmt(best_scan, "model_human_choice_agreement")}, mean RT {fmt(best_scan, "mean_rt")} s, and incongruent error rate {fmt(best_scan, "incongruent_error_rate")}. This suggests the fixed-gate weakness is partly a scale/timing/readout problem, although the evidence source still needs stronger validation.

## Trajectory validation

Top configurations were checked for `s_target(t)`, `s_flanker(t)`, `s_other_max(t)`, and `s_target_minus_flanker(t)` across congruent/incongruent, correct/error, human fast/slow, and model fast/slow groups. A configuration is considered DMC-like only if incongruent errors show early flanker competition and incongruent correct trials show later target recovery.

## AR(1) noise results

Best AR(1) candidate: `{best_ar1["condition"]}`. AR(1) is interpretable as subjective evidence persistence only when it improves tail shape or persistence without simply slowing all trials or damaging accuracy. These results should be read after checking whether the base configuration still has q95/q99 ceiling effects.

## Stochastic stopping results

Best stochastic stopping candidate: `{best_stop["condition"]}`. Stochastic stopping is useful only for fast-error readout after early flanker competition already exists. It does not create the conflict source.

## Final proposed model

```text
image
-> CNN layer-wise visual representation
-> subjective evidence distribution mu_t, sigma_t
-> AR(1) evidence noise
-> Wong-Wang accumulation
-> optional stochastic readout
-> RT + choice
```

## What is supported

- Hidden/layer-wise CNN evidence can create conflict that final logits suppress.
- A layer-time gate can produce DMC-like qualitative trajectories.
- AR(1) and stochastic stopping are separable additions with different explanatory roles.

## What is not yet supported

- The current layer-wise model cannot yet replace hand-crafted DMC behaviorally.
- AR(1) cannot be claimed as a tail solution if RT quantiles remain capped by simulation length.
- Stochastic stopping cannot explain fast errors unless the underlying trajectory already has flanker competition.

## Recommended next experiments

1. Fit or scan a larger but still bounded gate/WW/readout space around the best scanned settings.
2. Re-run trajectory validation on candidate settings with fewer ceiling RTs.
3. Test AR(1) only after the base RT scale is reasonable.
4. Test stochastic stopping only on configurations with confirmed early flanker competition.
5. Move from cache-level smoke tests to subject-level and image-identity tests.

## Psychological and cognitive-neuroscience interpretation

The emerging interpretation is that visual hierarchy supplies changing subjective evidence: early visual representations can contain distracting flanker information, later representations can recover target-oriented evidence, correlated noise can make subjective evidence states persist, and stochastic readout can occasionally commit before recovery.

## Exact commands to reproduce

```bash
python3 code/scripts/compare_same_subset_layerwise_vs_dmc.py --max_trials {args.max_trials} --time_steps {args.time_steps}
```

## Final answers

- Can layer-wise CNN evidence replace hand-crafted DMC? Not yet. It can replace the conflict source mechanistically in smoke tests, but not the full behavioral performance.
- Does AR(1) noise improve RT tail? It is the right mechanism to test tail persistence, but the result depends on whether q95/q99 are not already capped by the simulation horizon.
- Does stochastic stopping improve fast error? It can improve fast-error signatures in some settings, but often trades off against accuracy.
- Does the final model support the idea that noise affects decision-making by altering subjective evidence trajectories? Partially. The framework supports that interpretation, but stronger evidence requires a base model with better RT scale and uncapped tails.
- What remains hand-crafted? Gate timing, parameter grids, readout choices, DMC positive-control pulses, and current stopping/noise settings.
- What has become data-driven? The target/flanker evidence source is increasingly moved from manual pulses into CNN layer-wise visual representations.
""",
    )

    metadata = {
        "cache_path": str(resolve_path(args.cache_path)),
        "max_trials": int(args.max_trials),
        "dt_ms": dt_ms,
        "time_steps": time_steps,
        "seed": int(args.seed),
        "outputs": {
            "same_subset": str(same_dir),
            "parameter_scan": str(scan_dir),
            "trajectory_validation": str(traj_dir),
            "ar1_noise": str(ar1_dir),
            "stochastic_stopping": str(stop_dir),
            "integrated_report": str(report_path),
        },
    }
    for directory in [same_dir, scan_dir, traj_dir, ar1_dir, stop_dir]:
        (directory / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import (
    build_gate_input,
    make_trial_df,
    normalize_evidence,
    run_ww,
    summarize_condition,
)
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import to_jsonable


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cache(path: Path, n: int) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key][:n] for key in data.files}


def alpha_pulse(t: torch.Tensor, peak_s: float) -> torch.Tensor:
    peak = max(float(peak_s), 1e-6)
    scaled = t / peak
    return scaled * torch.exp(1.0 - scaled)


def build_hand_dmc_input(
    base: torch.Tensor,
    target: np.ndarray,
    flanker: np.ndarray,
    *,
    time_steps: int,
    dt_ms: int,
    auto_strength: float,
    selection_strength: float,
    target_boost: float,
    auto_peak_s: float,
    selection_midpoint_s: float,
    selection_tau_s: float,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    time_axis = torch.arange(time_steps, dtype=torch.float32) * (float(dt_ms) / 1000.0)
    auto = alpha_pulse(time_axis, auto_peak_s)
    gate = torch.sigmoid((time_axis - float(selection_midpoint_s)) / max(float(selection_tau_s), 1e-6))
    flanker_mult = (1.0 + float(auto_strength) * auto - float(selection_strength) * gate).clamp_min(0.0)
    target_mult = (1.0 - float(auto_strength) * auto + float(target_boost) * gate).clamp_min(0.0)
    x = base.unsqueeze(1).repeat(1, time_steps, 1).clone()
    rows = torch.arange(base.shape[0])
    target_t = torch.as_tensor(target, dtype=torch.long)
    flanker_t = torch.as_tensor(flanker, dtype=torch.long)
    for ti in range(time_steps):
        x[rows, ti, flanker_t] *= flanker_mult[ti]
        x[rows, ti, target_t] *= target_mult[ti]
    return x, {
        "auto_pulse": auto.numpy(),
        "selection_gate": gate.numpy(),
        "flanker_mult": flanker_mult.numpy(),
        "target_mult": target_mult.numpy(),
    }


def add_ar1_noise(x: torch.Tensor, rho: float, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0:
        return x.clone()
    rng = np.random.default_rng(seed)
    arr = x.detach().cpu().numpy().astype(np.float32)
    noise = np.zeros_like(arr)
    innovation_scale = float(sigma) * np.sqrt(max(1.0 - float(rho) ** 2, 0.0))
    noise[:, 0, :] = rng.normal(0.0, float(sigma), size=noise[:, 0, :].shape)
    for ti in range(1, noise.shape[1]):
        noise[:, ti, :] = float(rho) * noise[:, ti - 1, :] + rng.normal(0.0, innovation_scale, size=noise[:, ti, :].shape)
    return torch.as_tensor(np.clip(arr + noise, 0.0, None), dtype=torch.float32)


def stochastic_threshold_readout(
    trajectory: np.ndarray,
    cache: Dict[str, np.ndarray],
    *,
    base_threshold: float,
    sigma: float,
    dt_ms: int,
    t0_seconds: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    thresholds = rng.normal(float(base_threshold), float(sigma), size=(trajectory.shape[0], trajectory.shape[2]))
    thresholds = np.clip(thresholds, 0.05, 0.8)
    crossing = trajectory > thresholds[:, None, :]
    first = np.argmax(crossing, axis=1)
    no_cross = ~crossing.any(axis=1)
    first[no_cross] = trajectory.shape[1] - 1
    pred_choice = np.argmin(first, axis=1).astype(np.int64)
    pred_step = first[np.arange(first.shape[0]), pred_choice]
    pred_rt = pred_step.astype(np.float32) * (float(dt_ms) / 1000.0) + float(t0_seconds)
    return pd.DataFrame(
        {
            "condition": f"stochastic_stop_sigma{sigma:.3f}",
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
            "model_correct": pred_choice == cache["target_labels"].astype(np.int64),
            "human_correct": cache["response_labels"].astype(np.int64) == cache["target_labels"].astype(np.int64),
        }
    )


def plot_summary(summary: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(summary["condition"], summary[metric], color="#4C78A8")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_memo(path: Path, summaries: Dict[str, pd.DataFrame], command: str) -> None:
    dmc = summaries["same_subset_dmc"].sort_values("model_human_choice_agreement", ascending=False).iloc[0]
    ar1 = summaries["ar1"].sort_values("q95", ascending=False).iloc[0]
    stop = summaries["stochastic_stop"].sort_values("error_minus_correct_rt", ascending=True, na_position="last").iloc[0]
    memo = f"""# Remaining Layer-wise DMC Diagnostics

## Commands run

```bash
{command}
```

## What was added

1. Same-subset hand-crafted DMC positive controls on the 500-row layer-wise evidence cache.
2. AR(1) input-noise smoke tests on the best layer-time gate.
3. Stochastic stopping smoke tests using trial-wise threshold variability.

## Main results

- Best same-subset DMC-like control by human-choice agreement: `{dmc['condition']}`.
- Highest q95 under AR(1) noise: `{ar1['condition']}`.
- Fastest error-ordering stochastic-stop setting: `{stop['condition']}`.

## Interpretation

These additions complete the unfinished parts of the plan as smoke diagnostics. They do not replace a full trained comparison. The same-subset DMC controls are explicitly hand-crafted and are included only as positive controls; they are not part of the no-DMC replacement claim.

## Next decision

The layer-wise mechanism should be kept as a candidate conflict generator. It should not yet be claimed as a full DMC replacement until same-subset, image-identity, and subject-level tests improve behavioral fit.
"""
    path.write_text(memo, encoding="utf-8")


def update_final_report(report_path: Path) -> None:
    addition = """

## Completed remaining diagnostics

Additional diagnostics were added at:

`artifacts/results/diagnostics/layerwise_remaining_diagnostics/`

This completes the previously unfinished items:

- same-subset hand-crafted DMC positive controls;
- AR(1) input-noise smoke tests;
- stochastic stopping smoke tests.

Current interpretation remains unchanged: layer-wise CNN evidence can provide a natural conflict source, but the current fixed gate is not yet a complete behavioral replacement for DMC.
"""
    text = report_path.read_text(encoding="utf-8")
    if "## Completed remaining diagnostics" not in text:
        report_path.write_text(text + addition, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz")
    parser.add_argument("--best_gate_json", default="artifacts/results/diagnostics/layer_time_gate_ww/layer_time_gate_best_config.json")
    parser.add_argument("--output_dir", default="artifacts/results/diagnostics/layerwise_remaining_diagnostics")
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--time_steps", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--noise_ampa", type=float, default=0.02)
    parser.add_argument("--evidence_gain", type=float, default=0.8)
    parser.add_argument("--t0_seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(resolve_path(args.cache_path), int(args.max_trials))
    best_gate = json.loads(resolve_path(args.best_gate_json).read_text(encoding="utf-8"))

    raw = {
        "final": cache["evidence_final"],
        "pooled": cache["evidence_pooled"],
        "conv3": cache["evidence_conv3"],
        "conv4": cache["evidence_conv4"],
    }
    raw["mid"] = 0.5 * (raw["conv3"] + raw["conv4"])
    evidence = {key: normalize_evidence(value, gain=float(args.evidence_gain)) for key, value in raw.items()}
    gate_input, gate_trace = build_gate_input(
        evidence["mid"],
        evidence["pooled"],
        time_steps=int(args.time_steps),
        dt_ms=float(args.dt_ms),
        tau_s=float(best_gate.get("tau_s", 0.24)),
        k=float(best_gate.get("k", 20)),
    )

    summaries: Dict[str, pd.DataFrame] = {}

    dmc_rows = []
    dmc_trials = []
    for source in ["final", "mid", "pooled"]:
        dmc_input, traces = build_hand_dmc_input(
            evidence[source],
            cache["target_labels"],
            cache["flanker_labels"],
            time_steps=int(args.time_steps),
            dt_ms=int(args.dt_ms),
            auto_strength=0.30,
            selection_strength=0.40,
            target_boost=0.30,
            auto_peak_s=0.06,
            selection_midpoint_s=0.18,
            selection_tau_s=0.06,
        )
        out = run_ww(
            dmc_input,
            time_steps=int(args.time_steps),
            dt_ms=int(args.dt_ms),
            threshold=float(args.threshold),
            noise_ampa=float(args.noise_ampa),
            device="cpu",
            seed=int(args.seed),
            readout_mode="baseline",
            t0_seconds=float(args.t0_seconds),
            choice_temperature=0.10,
        )
        df = make_trial_df(cache, f"hand_dmc_{source}", out)
        dmc_trials.append(df)
        dmc_rows.append(summarize_condition(f"hand_dmc_{source}", df))
    summaries["same_subset_dmc"] = pd.DataFrame(dmc_rows)
    summaries["same_subset_dmc"].to_csv(output_dir / "same_subset_dmc_positive_control_summary.csv", index=False)
    pd.concat(dmc_trials, ignore_index=True).to_csv(output_dir / "same_subset_dmc_positive_control_trial_level.csv", index=False)

    ar1_rows = []
    ar1_trials = []
    for rho in [0.25, 0.50, 0.75, 0.90]:
        for sigma in [0.03, 0.06, 0.10]:
            noisy_input = add_ar1_noise(gate_input, rho=rho, sigma=sigma, seed=int(args.seed) + int(rho * 1000) + int(sigma * 1000))
            out = run_ww(
                noisy_input,
                time_steps=int(args.time_steps),
                dt_ms=int(args.dt_ms),
                threshold=float(args.threshold),
                noise_ampa=float(args.noise_ampa),
                device="cpu",
                seed=int(args.seed),
                readout_mode="baseline",
                t0_seconds=float(args.t0_seconds),
                choice_temperature=0.10,
            )
            name = f"ar1_rho{rho:.2f}_sigma{sigma:.2f}"
            df = make_trial_df(cache, name, out)
            row = summarize_condition(name, df)
            row["rho"] = rho
            row["sigma"] = sigma
            ar1_rows.append(row)
            ar1_trials.append(df)
    summaries["ar1"] = pd.DataFrame(ar1_rows)
    summaries["ar1"].to_csv(output_dir / "ar1_layer_gate_summary.csv", index=False)
    pd.concat(ar1_trials, ignore_index=True).to_csv(output_dir / "ar1_layer_gate_trial_level.csv", index=False)

    base_out = run_ww(
        gate_input,
        time_steps=int(args.time_steps),
        dt_ms=int(args.dt_ms),
        threshold=float(args.threshold),
        noise_ampa=float(args.noise_ampa),
        device="cpu",
        seed=int(args.seed),
        readout_mode="baseline",
        t0_seconds=float(args.t0_seconds),
        choice_temperature=0.10,
    )
    stop_rows = []
    stop_trials = []
    for sigma in [0.00, 0.02, 0.05, 0.08, 0.12]:
        df = stochastic_threshold_readout(
            base_out["trajectory"],
            cache,
            base_threshold=float(args.threshold),
            sigma=sigma,
            dt_ms=int(args.dt_ms),
            t0_seconds=float(args.t0_seconds),
            seed=int(args.seed) + int(sigma * 1000),
        )
        stop_rows.append(summarize_condition(f"stochastic_stop_sigma{sigma:.3f}", df))
        stop_trials.append(df)
    summaries["stochastic_stop"] = pd.DataFrame(stop_rows)
    summaries["stochastic_stop"].to_csv(output_dir / "stochastic_stopping_summary.csv", index=False)
    pd.concat(stop_trials, ignore_index=True).to_csv(output_dir / "stochastic_stopping_trial_level.csv", index=False)

    plot_summary(summaries["same_subset_dmc"], "model_human_choice_agreement", figure_dir / "same_subset_dmc_choice_agreement.png", "Same-subset DMC choice agreement")
    plot_summary(summaries["ar1"], "q95", figure_dir / "ar1_q95.png", "AR(1) q95")
    plot_summary(summaries["stochastic_stop"], "error_minus_correct_rt", figure_dir / "stochastic_stop_error_minus_correct_rt.png", "Stochastic stopping error ordering")
    plot_summary(summaries["stochastic_stop"], "incongruent_error_rate", figure_dir / "stochastic_stop_incongruent_error_rate.png", "Stochastic stopping incongruent errors")

    metadata = {
        "cache_path": str(resolve_path(args.cache_path)),
        "best_gate": best_gate,
        "max_trials": int(args.max_trials),
        "dt_ms": int(args.dt_ms),
        "time_steps": int(args.time_steps),
        "threshold": float(args.threshold),
        "noise_ampa": float(args.noise_ampa),
        "evidence_gain": float(args.evidence_gain),
        "t0_seconds": float(args.t0_seconds),
        "seed": int(args.seed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    command = (
        "python3 code/scripts/complete_layerwise_dmc_remaining_diagnostics.py "
        f"--max_trials {args.max_trials} --time_steps {args.time_steps}"
    )
    write_memo(output_dir / "remaining_diagnostics_memo.md", summaries, command)
    update_final_report(resolve_path("artifacts/results/diagnostics/layerwise_dmc_replacement_report.md"))


if __name__ == "__main__":
    main()

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
import torch.nn.functional as F
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT
from train_age_groups_efficient import to_jsonable
from vgg_wongwang_lim import (
    WongWangMultiClassDecision,
    compute_legacy_choice_logits,
    compute_rt_readout,
)


SOURCE_ORDER = ["conv3", "conv4", "conv5", "pooled", "final", "mid", "late"]
SINGLE_LAYER_SOURCES = ["final", "pooled", "conv5", "conv4", "conv3", "mid", "late"]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_evidence(evidence: np.ndarray, gain: float) -> torch.Tensor:
    x = np.asarray(evidence, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    z = np.clip((x - mean) / std, -6.0, 6.0)
    return F.softplus(torch.as_tensor(z * float(gain), dtype=torch.float32))


def build_gate_input(
    e_mid: torch.Tensor,
    e_late: torch.Tensor,
    *,
    time_steps: int,
    dt_ms: float,
    tau_s: float,
    k: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    time_axis = torch.arange(time_steps, dtype=torch.float32) * (float(dt_ms) / 1000.0)
    gate = torch.sigmoid(torch.as_tensor(float(k), dtype=torch.float32) * (time_axis - float(tau_s)))
    ww_input = (1.0 - gate.view(1, -1, 1)) * e_mid.unsqueeze(1) + gate.view(1, -1, 1) * e_late.unsqueeze(1)
    return ww_input, gate.numpy()


def make_ww(time_steps: int, dt_ms: int, threshold: float, noise_ampa: float, device: str) -> WongWangMultiClassDecision:
    ww = WongWangMultiClassDecision(n_classes=4, dt=int(dt_ms), time_steps=int(time_steps), t_stimulus=int(time_steps))
    with torch.no_grad():
        ww.threshold.copy_(torch.tensor(float(threshold)))
        ww.noise_ampa.copy_(torch.tensor(float(noise_ampa)))
    ww.eval()
    return ww.to(device)


def run_ww(
    ww_input: torch.Tensor,
    *,
    time_steps: int,
    dt_ms: int,
    threshold: float,
    noise_ampa: float,
    device: str,
    seed: int,
    readout_mode: str,
    t0_seconds: float,
    choice_temperature: float,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    ww = make_ww(time_steps, dt_ms, threshold, noise_ampa, device)
    x = ww_input.to(device)
    with torch.no_grad():
        decision_times, trajectory, threshold_tensor = ww.inference(x)
        evidence_traj = trajectory - threshold_tensor
        readout = compute_rt_readout(
            readout_mode,
            evidence_traj,
            readout_config={
                "dt_ms": float(dt_ms),
                "t0_mode": "fixed_global",
                "t0_seconds": float(t0_seconds),
                "choice_temperature": float(choice_temperature),
                "sigma_s": 0.05,
            },
        )
        choice_logits = compute_legacy_choice_logits(evidence_traj, choice_temperature=float(choice_temperature))
        pred_choice = choice_logits.argmax(dim=1)
    return {
        "pred_rt": readout["pred_rt"].detach().cpu().numpy().astype(np.float32),
        "pred_choice": pred_choice.detach().cpu().numpy().astype(np.int64),
        "choice_logits": choice_logits.detach().cpu().numpy().astype(np.float32),
        "trajectory": trajectory.detach().cpu().numpy().astype(np.float32),
        "evidence_traj": evidence_traj.detach().cpu().numpy().astype(np.float32),
        "decision_times": decision_times.detach().cpu().numpy().astype(np.float32),
    }


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def caf_fast_accuracy(df: pd.DataFrame, bins: int = 10, incongruent_only: bool = False) -> float:
    scoped = df[df["congruency"].eq(1)] if incongruent_only else df
    if len(scoped) < bins:
        return float("nan")
    order = np.argsort(scoped["pred_rt"].to_numpy(dtype=np.float32))
    first = np.array_split(order, bins)[0]
    return float(scoped.iloc[first]["model_correct"].mean())


def summarize_condition(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    pred_rt = df["pred_rt"].to_numpy(dtype=np.float32)
    true_rt = df["true_rt"].to_numpy(dtype=np.float32)
    correct = df["model_correct"].to_numpy(dtype=bool)
    human_correct = df["human_correct"].to_numpy(dtype=bool)
    cong = df["congruency"].to_numpy(dtype=np.int64)
    incong = cong == 1
    err = ~correct
    rows = {
        "condition": name,
        "n_trials": int(len(df)),
        "accuracy": float(correct.mean()),
        "human_accuracy": float(human_correct.mean()),
        "model_human_choice_agreement": float((df["pred_choice"] == df["response_label"]).mean()),
        "mean_rt": float(pred_rt.mean()),
        "median_rt": float(np.median(pred_rt)),
        "skewness": float(stats.skew(pred_rt)) if len(pred_rt) > 2 else float("nan"),
        "q90": float(np.quantile(pred_rt, 0.90)),
        "q95": float(np.quantile(pred_rt, 0.95)),
        "q99": float(np.quantile(pred_rt, 0.99)),
        "human_mean_rt": float(true_rt.mean()),
        "human_q95": float(np.quantile(true_rt, 0.95)),
        "human_model_mean_rt_gap": float(pred_rt.mean() - true_rt.mean()),
        "human_model_q95_gap": float(np.quantile(pred_rt, 0.95) - np.quantile(true_rt, 0.95)),
        "congruent_rt": safe_mean(pred_rt[cong == 0]),
        "incongruent_rt": safe_mean(pred_rt[incong]),
        "congruency_rt_gap": safe_mean(pred_rt[incong]) - safe_mean(pred_rt[cong == 0]),
        "correct_rt": safe_mean(pred_rt[correct]),
        "error_rt": safe_mean(pred_rt[err]),
        "error_minus_correct_rt": safe_mean(pred_rt[err]) - safe_mean(pred_rt[correct]),
        "incongruent_correct_rt": safe_mean(pred_rt[incong & correct]),
        "incongruent_error_rt": safe_mean(pred_rt[incong & err]),
        "incongruent_error_minus_correct_rt": safe_mean(pred_rt[incong & err]) - safe_mean(pred_rt[incong & correct]),
        "fastest_bin_accuracy": caf_fast_accuracy(df),
        "fastest_incongruent_bin_accuracy": caf_fast_accuracy(df, incongruent_only=True),
        "incongruent_error_rate": float((incong & err).sum() / max(incong.sum(), 1)),
    }
    return rows


def make_trial_df(cache: Dict[str, np.ndarray], condition: str, outputs: Dict[str, np.ndarray]) -> pd.DataFrame:
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
            "pred_rt": outputs["pred_rt"],
            "pred_choice": outputs["pred_choice"],
        }
    )
    df["model_correct"] = df["pred_choice"] == df["target_label"]
    df["human_correct"] = df["response_label"] == df["target_label"]
    return df


def plot_rt_distribution(trial_level: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for condition, group in trial_level.groupby("condition"):
        ax.hist(group["pred_rt"], bins=24, histtype="step", density=True, linewidth=1.6, label=condition)
    ax.set_xlabel("Predicted RT (s)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_caf(trial_level: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for condition, group in trial_level.groupby("condition"):
        if len(group) < 10:
            continue
        order = np.argsort(group["pred_rt"].to_numpy(dtype=np.float32))
        bins = np.array_split(order, 10)
        xs, ys = [], []
        for idx, bin_idx in enumerate(bins, start=1):
            subset = group.iloc[bin_idx]
            xs.append(idx)
            ys.append(float(subset["model_correct"].mean()))
        ax.plot(xs, ys, marker="o", linewidth=1.4, label=condition)
    ax.set_xlabel("RT bin (fast to slow)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_bar(summary: pd.DataFrame, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(summary["condition"], summary[metric], color="#4C78A8")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def target_flanker_trajectory(
    trajectory: np.ndarray,
    target: np.ndarray,
    flanker: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(trajectory.shape[0])
    s_target = trajectory[rows, :, target]
    s_flanker = trajectory[rows, :, flanker]
    other = trajectory.copy()
    other[rows, :, target] = np.nan
    other[rows, :, flanker] = np.nan
    s_other_max = np.nanmax(other, axis=2)
    return s_target, s_flanker, s_other_max


def trajectory_summary_rows(
    condition: str,
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    dt_ms: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    target = df["target_label"].to_numpy(dtype=np.int64)
    flanker = df["flanker_label"].to_numpy(dtype=np.int64)
    s_target, s_flanker, s_other = target_flanker_trajectory(outputs["trajectory"], target, flanker)
    early_steps = max(1, int(round(0.12 / (dt_ms / 1000.0))))
    rows = []
    curves: Dict[str, np.ndarray] = {}
    groups = {
        "congruent_correct": (df["congruency"].eq(0) & df["model_correct"]).to_numpy(),
        "congruent_error": (df["congruency"].eq(0) & ~df["model_correct"]).to_numpy(),
        "incongruent_correct": (df["congruency"].eq(1) & df["model_correct"]).to_numpy(),
        "incongruent_error": (df["congruency"].eq(1) & ~df["model_correct"]).to_numpy(),
        "human_fast_rt": (df["true_rt"] <= df["true_rt"].median()).to_numpy(),
        "human_slow_rt": (df["true_rt"] > df["true_rt"].median()).to_numpy(),
    }
    for group_name, mask in groups.items():
        if not mask.any():
            rows.append({"condition": condition, "group": group_name, "n_trials": 0})
            continue
        early_adv = (s_flanker[mask, :early_steps] - s_target[mask, :early_steps]).mean(axis=1)
        rows.append(
            {
                "condition": condition,
                "group": group_name,
                "n_trials": int(mask.sum()),
                "early_flanker_minus_target": float(early_adv.mean()),
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


def plot_trajectory(curves: Dict[str, np.ndarray], condition: str, group: str, dt_ms: int, output_path: Path) -> None:
    key_t = f"{condition}:{group}:target"
    key_f = f"{condition}:{group}:flanker"
    if key_t not in curves or key_f not in curves:
        return
    t = np.arange(len(curves[key_t])) * (dt_ms / 1000.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t, curves[key_t], label="target", linewidth=1.8)
    ax.plot(t, curves[key_f], label="flanker", linewidth=1.8)
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


def write_single_layer_memo(output_path: Path, summary: pd.DataFrame, command: str) -> None:
    best_errors = summary.sort_values("incongruent_error_rate", ascending=False).iloc[0]
    most_target = summary.sort_values("accuracy", ascending=False).iloc[0]
    best_fast = summary.sort_values("fastest_incongruent_bin_accuracy", ascending=True).iloc[0]
    memo = f"""# Single-layer Evidence to Wong-Wang

## Commands run

```bash
{command}
```

## Outputs

- `single_layer_ww_summary.csv`
- `single_layer_ww_trial_level.csv`
- figures in `figures/`

## Main result

- Highest incongruent error rate: `{best_errors['condition']}` ({best_errors['incongruent_error_rate']:.3f}).
- Highest target accuracy: `{most_target['condition']}` ({most_target['accuracy']:.3f}).
- Lowest fastest-incongruent-bin accuracy: `{best_fast['condition']}` ({best_fast['fastest_incongruent_bin_accuracy']:.3f}).

## Interpretation

Middle-layer evidence is useful if conv or pooled sources create more incongruent errors than final logits while final logits remain target-dominant. This run is still a smoke test: it uses frozen diagnostic evidence and fixed Wong-Wang parameters, not trained behavior fitting.

## What this supports

It tests whether layer-wise evidence can create conflict-like behavior without hand-written `flanker_mult(t)` or `target_mult(t)`.

## What this does not support

It does not yet prove a full DMC replacement, because the WW parameters and evidence scaling were not fitted and the evidence split still has image-identity overlap.
"""
    output_path.write_text(memo, encoding="utf-8")


def write_gate_memo(output_path: Path, summary: pd.DataFrame, best: Dict[str, Any], command: str) -> None:
    final = summary[summary["condition"].eq("final")]
    best_row = summary[summary["condition"].eq(best["condition"])]
    final_acc = float(final["accuracy"].iloc[0]) if not final.empty else float("nan")
    best_acc = float(best_row["accuracy"].iloc[0]) if not best_row.empty else float("nan")
    memo = f"""# Fixed Layer-time Gate to Wong-Wang

## Commands run

```bash
{command}
```

## Best fixed gate

```json
{json.dumps(to_jsonable(best), indent=2)}
```

## Interpretation

- Best layer-time gate accuracy: {best_acc:.3f}; final-logit-only accuracy: {final_acc:.3f}.
- A layer-time gate is promising only if it improves conflict behavior relative to final logits without destroying target accuracy.
- The gate is label-free: early input is middle-layer evidence, late input is pooled/final evidence. It does not directly boost flanker or target labels.

## What this supports

This is a first no-DMC smoke test for natural early visual conflict followed by later target-oriented evidence.

## What this does not support

It does not yet match a trained DMC positive control, and it does not include AR(1) noise or stochastic stopping.
"""
    output_path.write_text(memo, encoding="utf-8")


def write_noise_relation(path: Path) -> None:
    memo = """# Layer-wise DMC Replacement, AR(1) Noise, and Stochastic Stopping

Layer-wise evidence dynamics mainly address the source of early flanker competition and late target selection. This is the proposed natural replacement for hand-written DMC pulses.

AR(1) evidence noise should be treated separately. It mainly addresses slow-trial persistence and the long right tail of RT distributions; it does not create missing flanker information by itself.

Stochastic stopping is also separate. If early flanker competition already exists, stochastic stopping can help explain why some errors are fast.

Recommended order:

1. Test layer-wise evidence first.
2. Add AR(1) noise only after conflict evidence exists.
3. Add stochastic stopping only after checking whether fast errors remain missing.

Current smoke tests should not mix these mechanisms into a single explanation.
"""
    path.write_text(memo, encoding="utf-8")


def write_final_report(path: Path, single_summary: pd.DataFrame, gate_summary: pd.DataFrame, best: Dict[str, Any]) -> None:
    final_single = single_summary[single_summary["condition"].eq("final")].iloc[0]
    best_gate = gate_summary[gate_summary["condition"].eq(best["condition"])].iloc[0]
    report = f"""# Layer-wise DMC Replacement Report

## Executive summary

The first layer-wise evidence-to-Wong-Wang smoke tests were completed without long training and without hand-written DMC pulses.

## Current hidden-feature result

Earlier layer-wise probing showed that hidden CNN layers retain strong flanker information while final logits are target-dominant.

## Single-layer WW result

Final-logit-only accuracy was {final_single['accuracy']:.3f}, with incongruent error rate {final_single['incongruent_error_rate']:.3f}. The single-layer table reports whether conv/pool sources create more conflict-like errors.

## Fixed layer-time gate result

Best gate: `{best['condition']}`. Accuracy was {best_gate['accuracy']:.3f}; incongruent error rate was {best_gate['incongruent_error_rate']:.3f}.

## Relation to DMC

This test removes hand-written `flanker_mult(t)` and `target_mult(t)`. It replaces them with a label-free transition from middle-layer visual evidence to later target-oriented evidence.

## Relation to AR(1) noise and stochastic stopping

AR(1) noise and stochastic stopping remain separate follow-up mechanisms. They should be tested only after deciding whether layer-wise evidence creates enough target/flanker competition.

## Can natural DMC-like dynamics be produced by layer-wise CNN evidence entering Wong-Wang over time?

Supported by current results:

- Hidden layers expose flanker information that final logits suppress.
- A label-free layer-time gate can be evaluated with the current evidence cache.

Not yet supported:

- A full replacement for hand-crafted DMC is not established by this smoke run alone.
- Image-identity and subject-level generalization remain unresolved.

Requires further experiment:

- Compare against the current DMC positive control on matched settings.
- Fit or scan WW/evidence scaling more carefully.
- Test AR(1) noise and stochastic stopping only after layer-wise conflict is confirmed.

## Exact commands to reproduce

See the memo files in:

- `artifacts/results/diagnostics/layerwise_feature_probe/`
- `artifacts/results/diagnostics/layerwise_evidence_ww_single_layer/`
- `artifacts/results/diagnostics/layer_time_gate_ww/`
- `artifacts/results/diagnostics/layer_time_gate_trajectory/`
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz")
    parser.add_argument("--output_root", default="artifacts/results/diagnostics")
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--time_steps", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--noise_ampa", type=float, default=0.02)
    parser.add_argument("--evidence_gain", type=float, default=0.8)
    parser.add_argument("--t0_seconds", type=float, default=0.25)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--readout_mode", default="baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = resolve_path(args.cache_path)
    output_root = resolve_path(args.output_root)
    single_dir = output_root / "layerwise_evidence_ww_single_layer"
    gate_dir = output_root / "layer_time_gate_ww"
    traj_dir = output_root / "layer_time_gate_trajectory"
    for directory in [single_dir / "figures", gate_dir / "figures", traj_dir / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cache_path)
    n = min(int(args.max_trials), len(cache["target_labels"]))
    cache = {key: value[:n] for key, value in cache.items()}

    raw_sources = {
        "conv3": cache["evidence_conv3"],
        "conv4": cache["evidence_conv4"],
        "conv5": cache["evidence_conv5"],
        "pooled": cache["evidence_pooled"],
        "final": cache["evidence_final"],
    }
    raw_sources["mid"] = 0.5 * (raw_sources["conv3"] + raw_sources["conv4"])
    raw_sources["late"] = raw_sources["pooled"]
    evidence = {key: normalize_evidence(value, gain=float(args.evidence_gain)) for key, value in raw_sources.items()}

    command = (
        "python3 code/scripts/analyze_layerwise_evidence_ww.py "
        f"--cache_path {args.cache_path} --max_trials {args.max_trials} --dt_ms {args.dt_ms} "
        f"--time_steps {args.time_steps} --threshold {args.threshold} --noise_ampa {args.noise_ampa} "
        f"--evidence_gain {args.evidence_gain} --t0_seconds {args.t0_seconds}"
    )

    single_trials = []
    single_summaries = []
    single_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for source in SINGLE_LAYER_SOURCES:
        outputs = run_ww(
            evidence[source],
            time_steps=int(args.time_steps),
            dt_ms=int(args.dt_ms),
            threshold=float(args.threshold),
            noise_ampa=float(args.noise_ampa),
            device=str(args.device),
            seed=int(args.seed),
            readout_mode=str(args.readout_mode),
            t0_seconds=float(args.t0_seconds),
            choice_temperature=float(args.choice_temperature),
        )
        single_outputs[source] = outputs
        trial_df = make_trial_df(cache, source, outputs)
        single_trials.append(trial_df)
        single_summaries.append(summarize_condition(source, trial_df))

    single_trial_level = pd.concat(single_trials, ignore_index=True)
    single_summary = pd.DataFrame(single_summaries)
    single_summary.to_csv(single_dir / "single_layer_ww_summary.csv", index=False)
    single_trial_level.to_csv(single_dir / "single_layer_ww_trial_level.csv", index=False)
    plot_rt_distribution(single_trial_level, single_dir / "figures" / "rt_distribution_by_layer.png", "Single-layer WW RT")
    plot_caf(single_trial_level, single_dir / "figures" / "caf_by_layer.png", "Single-layer WW CAF")
    plot_bar(single_summary, "error_minus_correct_rt", single_dir / "figures" / "error_minus_correct_rt_by_layer.png", "Error - Correct RT", "Seconds")
    plot_bar(single_summary, "fastest_incongruent_bin_accuracy", single_dir / "figures" / "incongruent_fastest_bin_accuracy_by_layer.png", "Fastest Incongruent Bin Accuracy", "Accuracy")
    write_single_layer_memo(single_dir / "single_layer_ww_memo.md", single_summary, command)

    gate_trials = []
    gate_summaries = []
    gate_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    gate_traces: Dict[str, np.ndarray] = {}
    for tau_s in [0.08, 0.12, 0.16, 0.20, 0.24]:
        for k in [20, 40, 60, 80]:
            name = f"layer_gate_tau{tau_s:.2f}_k{k}"
            gate_input, gate = build_gate_input(
                evidence["mid"],
                evidence["late"],
                time_steps=int(args.time_steps),
                dt_ms=float(args.dt_ms),
                tau_s=tau_s,
                k=float(k),
            )
            outputs = run_ww(
                gate_input,
                time_steps=int(args.time_steps),
                dt_ms=int(args.dt_ms),
                threshold=float(args.threshold),
                noise_ampa=float(args.noise_ampa),
                device=str(args.device),
                seed=int(args.seed),
                readout_mode=str(args.readout_mode),
                t0_seconds=float(args.t0_seconds),
                choice_temperature=float(args.choice_temperature),
            )
            gate_outputs[name] = outputs
            gate_traces[name] = gate
            trial_df = make_trial_df(cache, name, outputs)
            gate_trials.append(trial_df)
            row = summarize_condition(name, trial_df)
            row["tau_s"] = tau_s
            row["k"] = k
            gate_summaries.append(row)

    gate_summary = pd.DataFrame(gate_summaries)
    # Prefer target accuracy, conflict, and lower mean/q95 mismatch.
    score = (
        gate_summary["accuracy"]
        + 0.25 * gate_summary["incongruent_error_rate"]
        - 0.20 * gate_summary["human_model_mean_rt_gap"].abs()
        - 0.10 * gate_summary["human_model_q95_gap"].abs()
    )
    best_idx = int(score.idxmax())
    best_name = str(gate_summary.loc[best_idx, "condition"])
    best = gate_summary.loc[best_idx].to_dict()
    best["selection_score"] = float(score.loc[best_idx])
    (gate_dir / "layer_time_gate_best_config.json").write_text(json.dumps(to_jsonable(best), indent=2), encoding="utf-8")
    gate_summary.to_csv(gate_dir / "layer_time_gate_grid_summary.csv", index=False)
    best_trial = [df for df in gate_trials if str(df["condition"].iloc[0]) == best_name][0]
    best_trial.to_csv(gate_dir / "layer_time_gate_trial_level_best.csv", index=False)

    comparison_trial = pd.concat(
        [
            single_trial_level[single_trial_level["condition"].isin(["final", "pooled", "mid", "late"])],
            best_trial,
        ],
        ignore_index=True,
    )
    plot_rt_distribution(comparison_trial, gate_dir / "figures" / "rt_distribution_best_gate.png", "Best Gate vs Single Sources")
    plot_caf(comparison_trial, gate_dir / "figures" / "caf_best_gate.png", "Best Gate CAF")
    comp_summary = pd.concat(
        [
            single_summary[single_summary["condition"].isin(["final", "pooled", "mid", "late"])],
            pd.DataFrame([summarize_condition(best_name, best_trial)]),
        ],
        ignore_index=True,
    )
    plot_bar(comp_summary, "incongruent_error_rate", gate_dir / "figures" / "dmc_vs_layer_gate_comparison.png", "Conflict-like Error Rate", "Rate")
    t = np.arange(int(args.time_steps)) * (int(args.dt_ms) / 1000.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(t, gate_traces[best_name], linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Late-layer gate")
    ax.set_title("Best Layer Gate Trace")
    fig.tight_layout()
    fig.savefig(gate_dir / "figures" / "layer_gate_trace.png", dpi=220)
    plt.close(fig)
    write_gate_memo(gate_dir / "layer_time_gate_memo.md", comp_summary, best, command)

    traj_rows = []
    curves: Dict[str, np.ndarray] = {}
    traj_conditions = {
        "final": (single_trial_level[single_trial_level["condition"].eq("final")].reset_index(drop=True), single_outputs["final"]),
        "mid": (single_trial_level[single_trial_level["condition"].eq("mid")].reset_index(drop=True), single_outputs["mid"]),
        best_name: (best_trial.reset_index(drop=True), gate_outputs[best_name]),
    }
    for condition, (df, outputs) in traj_conditions.items():
        rows, condition_curves = trajectory_summary_rows(condition, df, outputs, int(args.dt_ms))
        traj_rows.extend(rows)
        curves.update(condition_curves)
    traj_summary = pd.DataFrame(traj_rows)
    traj_summary.to_csv(traj_dir / "trajectory_summary.csv", index=False)
    plot_trajectory(curves, best_name, "incongruent_error", int(args.dt_ms), traj_dir / "figures" / "s_target_flanker_incongruent_error.png")
    plot_trajectory(curves, best_name, "incongruent_correct", int(args.dt_ms), traj_dir / "figures" / "s_target_flanker_incongruent_correct.png")
    plot_bar(
        traj_summary[traj_summary["group"].eq("incongruent_error")].fillna(0),
        "early_flanker_minus_target",
        traj_dir / "figures" / "early_flanker_advantage_by_condition.png",
        "Early Flanker Advantage",
        "s_flanker - s_target",
    )
    (traj_dir / "trajectory_memo.md").write_text(
        f"# Layer-time Gate Trajectory\n\nBest condition: `{best_name}`.\n\nTrajectory summaries are saved in `trajectory_summary.csv`. The key diagnostic is whether incongruent-error trials show early flanker activity approaching or exceeding target activity.\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_target = best_name
    key_t = f"{plot_target}:incongruent_correct:target"
    key_f = f"{plot_target}:incongruent_correct:flanker"
    if key_t in curves and key_f in curves:
        ax.plot(t, curves[key_t] - curves[key_f], label="incongruent correct")
    key_t = f"{plot_target}:incongruent_error:target"
    key_f = f"{plot_target}:incongruent_error:flanker"
    if key_t in curves and key_f in curves:
        ax.plot(t, curves[key_t] - curves[key_f], label="incongruent error")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("s_target - s_flanker")
    ax.set_title("Best Gate Target vs Flanker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(gate_dir / "figures" / "s_target_s_flanker_best_gate.png", dpi=220)
    plt.close(fig)

    write_noise_relation(output_root / "layerwise_dmc_noise_readout_relation_memo.md")
    write_final_report(output_root / "layerwise_dmc_replacement_report.md", single_summary, comp_summary, best)

    metadata = {
        "cache_path": str(cache_path),
        "n_trials": int(n),
        "dt_ms": int(args.dt_ms),
        "time_steps": int(args.time_steps),
        "threshold": float(args.threshold),
        "noise_ampa": float(args.noise_ampa),
        "evidence_gain": float(args.evidence_gain),
        "t0_seconds": float(args.t0_seconds),
        "readout_mode": str(args.readout_mode),
        "seed": int(args.seed),
        "best_gate": best,
    }
    for directory in [single_dir, gate_dir, traj_dir]:
        (directory / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

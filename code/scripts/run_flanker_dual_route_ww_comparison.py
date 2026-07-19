#!/usr/bin/env python3
"""Four-fold controlled comparison of full-image and source-separated WR2/WW inputs."""
from __future__ import annotations

import argparse
import json
import math
import os
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
from scipy.stats import wasserstein_distance

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww  # noqa: E402
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_congruent_ww_dynamics_diagnostic import parse_group_params  # noqa: E402
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule  # noqa: E402
from run_representative_extreme_age_subset_fitting import apply_group_t0, load_trial_cache  # noqa: E402


BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
SOURCE_RUN = BASE_DIR / "dual_source_conflict_test/20260719_full_v1"
SPLIT_PATH = BASE_DIR / "flanker_rt_4bin_fitting/dual_track_4bin_20260716_final/split_manifest.csv"
PARAM_PATH = BASE_DIR / "best_model_R5_combined_best/results/best_model_parameter_estimates.csv"
OUTPUT_ROOT = BASE_DIR / "dual_source_conflict_test"
LAYERS = ["conv3", "conv4", "conv5", "pooled", "final"]
NONFINAL_LAYERS = ["conv3", "conv4", "conv5", "pooled"]
GROUPS = ["young_20_29", "older_80_89"]
DT = 0.01
TIME_STEPS = 80
WR2 = {"compression_low": 0.72, "compression_high": 0.42, "theta_quantile": 0.50, "temp": 0.22, "score_window_s": 0.15, "late_shift_ms": -40, "early_phase_shortening_ms": 30, "transition_width": 1.0}
NOISE = {"young_20_29": 0.003, "older_80_89": 0.006}
SEEDS = [2026071901 + i for i in range(10)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-source timing comparison using the existing WR2 and Wong-Wang backbone.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--fold-limit", type=int, default=4)
    p.add_argument("--seed-limit", type=int, default=10)
    p.add_argument("--grid-limit", type=int, default=None)
    return p.parse_args()


def schedule_df(compression: float) -> pd.DataFrame:
    t = np.arange(TIME_STEPS, dtype=np.float32) / TIME_STEPS
    centers = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32) * compression
    centers = np.clip(centers, 0.03, 0.97)
    centers[3:] = np.clip(centers[3:] + WR2["late_shift_ms"] / 1000.0, 0.03, 0.97)
    centers[0] = max(0.03, centers[0] - WR2["early_phase_shortening_ms"] / 1000.0)
    sigma = max(0.12 * WR2["transition_width"], 0.03)
    basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / sigma) ** 2)
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-9)
    return pd.DataFrame(basis, columns=LAYERS)


def normalize_layers_train(raw: dict[str, np.ndarray], train_mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for layer, value in raw.items():
        centered = np.asarray(value, np.float32) - np.asarray(value, np.float32).mean(axis=1, keepdims=True)
        row_sd = centered.std(axis=1)
        scale = float(np.mean(row_sd[train_mask]))
        out[layer] = centered / max(scale, 1e-6)
    return out


def _standardize_train(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x[train_mask]))
    sd = float(np.std(x[train_mask]))
    return (x - mean) / max(sd, 1e-6)


def wr2_full_mu(norm_layers: dict[str, np.ndarray], evidence_gain: float, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = build_mu_schedule(norm_layers, schedule_df(WR2["compression_low"]), evidence_gain).numpy()
    high = build_mu_schedule(norm_layers, schedule_df(WR2["compression_high"]), evidence_gain).numpy()
    hi = max(1, int(WR2["score_window_s"] / DT))
    logits = low[:, :hi] - low[:, :hi].max(axis=2, keepdims=True)
    prob = np.exp(logits)
    prob /= np.maximum(prob.sum(axis=2, keepdims=True), 1e-9)
    entropy = -(prob * np.log(np.maximum(prob, 1e-9))).sum(axis=2).mean(axis=1)
    sorted_mu = np.sort(low[:, :hi], axis=2)
    gap = (sorted_mu[:, :, -1] - sorted_mu[:, :, -2]).mean(axis=1)
    score = (_standardize_train(entropy, train_mask) + _standardize_train(-gap, train_mask)) / 2.0
    theta = float(np.quantile(score[train_mask], WR2["theta_quantile"]))
    strength = 1.0 / (1.0 + np.exp(-np.clip((score - theta) / WR2["temp"], -60, 60)))
    mu = low + strength[:, None, None] * (high - low)
    return mu.astype(np.float32), strength.astype(np.float32)


def temporal_envelopes(model: str, delay_ms: int = 80, decay_ms: int = 120) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(TIME_STEPS, dtype=np.float32) * DT
    delay = delay_ms / 1000.0
    decay = max(decay_ms / 1000.0, DT)
    early = np.exp(-t / decay)
    late = np.where(t >= delay, 1.0 - np.exp(-(t - delay) / 0.08), 0.0)
    if model == "M1_simultaneous":
        return np.ones_like(t), np.ones_like(t)
    if model == "M2_flanker_early_target_late":
        return late.astype(np.float32), early.astype(np.float32)
    if model == "M3_target_early_flanker_late":
        return early.astype(np.float32), late.astype(np.float32)
    if model == "M0_full_WR2":
        return np.ones_like(t), np.ones_like(t)
    raise ValueError(model)


def unit_direction(value: np.ndarray) -> np.ndarray:
    centered = value - value.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))
    return centered / np.maximum(rms, 1e-6)


def source_mu_energy_matched(full_mu: np.ndarray, target_ev: np.ndarray, flanker_ev: np.ndarray, target_env: np.ndarray, flanker_env: np.ndarray) -> np.ndarray:
    target = unit_direction(target_ev)[:, None, :] * target_env[None, :, None]
    flanker = unit_direction(flanker_ev)[:, None, :] * flanker_env[None, :, None]
    combined = target + flanker
    combined -= combined.mean(axis=2, keepdims=True)
    combined_rms = np.sqrt(np.mean(combined**2, axis=2, keepdims=True))
    full_centered = full_mu - full_mu.mean(axis=2, keepdims=True)
    full_rms = np.sqrt(np.mean(full_centered**2, axis=2, keepdims=True))
    return (combined / np.maximum(combined_rms, 1e-6) * full_rms).astype(np.float32)


def select_source_layers(source: dict[str, np.ndarray], target: np.ndarray, flanker: np.ndarray, train_mask: np.ndarray) -> tuple[str, str, pd.DataFrame]:
    rows = []
    for source_name, labels in [("target", target), ("flanker", flanker)]:
        for layer in NONFINAL_LAYERS:
            pred = source[f"{source_name}_{layer}"].argmax(axis=1)
            rows.append({"source": source_name, "layer": layer, "train_accuracy": float((pred[train_mask] == labels[train_mask]).mean())})
    table = pd.DataFrame(rows)
    selected = {}
    for name in ["target", "flanker"]:
        part = table[table["source"].eq(name)].sort_values(["train_accuracy", "layer"], ascending=[False, True])
        selected[name] = str(part.iloc[0]["layer"])
    return selected["target"], selected["flanker"], table


def equal_count_bins(rt: np.ndarray, n_bins: int = 4) -> np.ndarray:
    order = np.argsort(rt, kind="mergesort")
    bins = np.empty(len(rt), dtype=int)
    bins[order] = np.floor(np.arange(len(rt)) * n_bins / max(len(rt), 1)).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def subject_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, user, congruency), part in df.groupby(["analysis_group", "user_id", "congruency"], sort=True):
        for source_name, rt_col, correct_col in [("human", "true_rt", "human_correct"), ("model", "model_rt", "model_correct")]:
            bins = equal_count_bins(part[rt_col].to_numpy(float))
            correct = part[correct_col].to_numpy(bool)
            for b in range(4):
                mask = bins == b
                rows.append({"analysis_group": group, "user_id": user, "congruency": congruency, "source": source_name, "bin": b + 1, "n": int(mask.sum()), "error_rate": float((~correct[mask]).mean()), "mean_rt": float(part.loc[mask, rt_col].mean())})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> dict[str, float]:
    profiles = subject_profiles(df)
    agg = profiles.groupby(["analysis_group", "congruency", "source", "bin"], as_index=False)["error_rate"].mean()
    pivot = agg.pivot_table(index=["analysis_group", "congruency", "bin"], columns="source", values="error_rate").reset_index()
    curve_rmse = float(np.sqrt(np.mean((pivot["model"] - pivot["human"]) ** 2)))
    cells = []
    for (group, condition), part in df.groupby(["analysis_group", "congruency"], sort=True):
        cells.append({"group": group, "condition": condition, "human_error": float((~part.human_correct).mean()), "model_error": float((~part.model_correct).mean()), "human_median": float(part.true_rt.median()), "model_median": float(part.model_rt.median()), "wdist": float(wasserstein_distance(part.true_rt, part.model_rt))})
    cell = pd.DataFrame(cells)
    yi = pivot[(pivot.analysis_group == "young_20_29") & (pivot.congruency == 1) & (pivot.bin == 1)].iloc[0]
    yi_cell = cell[(cell.group == "young_20_29") & (cell.condition == 1)].iloc[0]
    other = cell[~((cell.group == "young_20_29") & (cell.condition == 1))]
    metrics = {
        "curve_rmse": curve_rmse,
        "error_mae": float(np.mean(np.abs(cell.model_error - cell.human_error))),
        "rt_wasserstein": float(cell.wdist.mean()),
        "young_incongruent_fast_error_gap": float(abs(yi.model - yi.human)),
        "young_incongruent_fast_model_error": float(yi.model),
        "young_incongruent_fast_human_error": float(yi.human),
        "young_incongruent_median_rt_gap": float(abs(yi_cell.model_median - yi_cell.human_median)),
        "other_cell_composite": float(np.mean(np.abs(other.model_error - other.human_error) + np.abs(other.model_median - other.human_median))),
    }
    for group in GROUPS:
        part = cell[(cell.group == group) & (cell.condition == 0)]
        metrics[f"{group}_congruent_model_error"] = float(part.model_error.iloc[0])
    metrics["selection_score"] = 4 * curve_rmse + 2 * metrics["error_mae"] + metrics["rt_wasserstein"] + 4 * metrics["young_incongruent_fast_error_gap"] + 2 * metrics["young_incongruent_median_rt_gap"]
    return metrics


def load_source_trial_arrays(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(SOURCE_RUN / "cache/source_separated_evidence.npz", allow_pickle=True) as z:
        ids = z["subset_stimulus_id"].astype(int)
        pos = {int(v): i for i, v in enumerate(ids)}
        idx = np.asarray([pos[int(v)] for v in cache["subset_stimulus_id"]], dtype=int)
        out = {}
        for source_name in ["target", "flanker"]:
            variant = f"{source_name}_only"
            for layer in LAYERS:
                out[f"{source_name}_{layer}"] = z[f"evidence_{variant}_{layer}"][idx].astype(np.float32)
    return out


def make_mu_by_group(cache: dict[str, np.ndarray], source: dict[str, np.ndarray], train_mask: np.ndarray, group_params: dict[str, dict[str, float]], model: str, delay_ms: int, decay_ms: int, target_layer: str, flanker_layer: str) -> dict[str, np.ndarray]:
    result = {}
    raw_full = {layer: cache[f"evidence_{layer}"] for layer in LAYERS}
    norm_full = normalize_layers_train(raw_full, train_mask)
    target_norm = normalize_layers_train({layer: source[f"target_{layer}"] for layer in LAYERS}, train_mask)
    flanker_norm = normalize_layers_train({layer: source[f"flanker_{layer}"] for layer in LAYERS}, train_mask)
    for group in GROUPS:
        gm = cache["analysis_group"].astype(str) == group
        group_train = train_mask[gm]
        full_mu, _ = wr2_full_mu({k: v[gm] for k, v in norm_full.items()}, float(group_params[group]["evidence_gain"]), group_train)
        if model == "M0_full_WR2":
            result[group] = full_mu
        else:
            te, fe = temporal_envelopes(model, delay_ms, decay_ms)
            result[group] = source_mu_energy_matched(full_mu, target_norm[target_layer][gm], flanker_norm[flanker_layer][gm], te, fe)
    return result


def simulate(cache: dict[str, np.ndarray], mu_by_group: dict[str, np.ndarray], group_params: dict[str, dict[str, float]], t0_mean: dict[str, float], t0_sd: dict[str, float], model: str, seed: int, device: str) -> pd.DataFrame:
    parts = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        idx = np.flatnonzero(mask)
        gp = group_params[group]
        outputs = run_ww(torch.as_tensor(mu_by_group[group]), time_steps=TIME_STEPS, dt_ms=10, threshold=float(gp["threshold"]), noise_ampa=float(NOISE[group]), device=device, seed=seed, readout_mode="baseline", t0_seconds=0.0, choice_temperature=0.01)
        base = pd.DataFrame({"target_label": cache["target_labels"][mask], "flanker_label": cache["flanker_labels"][mask], "true_rt": cache["true_rt"][mask], "human_correct": cache["human_correct"][mask], "congruency": cache["congruency"][mask], "analysis_group": group, "user_id": cache["user_id"][mask], "pred_choice": outputs["pred_choice"], "pred_rt": outputs["pred_rt"]})
        cfg = ReadoutConfig("sustained_crossing", min_decision_time=float(gp["min_decision_time"]), sustained_k=int(gp["sustained_k"]), margin=float(gp["margin"]))
        base = apply_readout(base, outputs, cfg=cfg, threshold=float(gp["threshold"]), dt_ms=10, t0_seconds=0.0, choice_rule="winner_at_readout")
        base["global_index"] = idx
        parts.append(base)
    df = pd.concat(parts, ignore_index=True).sort_values("global_index").reset_index(drop=True)
    df = apply_group_t0(df, t0_mean, t0_sd, seed)
    return df.rename(columns={"pred_rt": "model_rt"}).assign(model=model, seed=seed)


def timing_grid(limit: int | None) -> list[tuple[int, int]]:
    grid = [(d, x) for d in [40, 80, 120] for x in [80, 120, 160]]
    return grid if limit is None else grid[: max(1, limit)]


def masks_for_fold(cache: dict[str, np.ndarray], splits: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    f = splits[splits["fold"].eq(fold)].copy()
    train_users = set(f[f.role.eq("train")].user_id.astype(str))
    test_users = set(f[f.role.eq("test")].user_id.astype(str))
    users = cache["user_id"].astype(str)
    train, test = np.isin(users, list(train_users)), np.isin(users, list(test_users))
    if np.any(train & test) or not train.any() or not test.any():
        raise RuntimeError(f"Invalid split for fold {fold}")
    return train, test


def candidate_spec(model: str, delay: int = 80, decay: int = 120) -> str:
    return f"{model}__delay{delay}__decay{decay}"


def main() -> None:
    args = parse_args()
    out = Path(args.output_root) / args.run_id
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    for d in ["metrics", "figures", "summaries", "audits"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    gate = json.loads((SOURCE_RUN / "audits/representation_gate.json").read_text())
    if not gate["representation_audit_passed"]:
        raise RuntimeError("Source representation audit did not pass")
    cache = load_trial_cache(BASE_DIR)
    source = load_source_trial_arrays(cache)
    group_params, t0_mean, t0_sd = parse_group_params(PARAM_PATH)
    splits = pd.read_csv(SPLIT_PATH)
    folds = sorted(splits.fold.unique())[: args.fold_limit]
    seeds = SEEDS[: args.seed_limit]
    splits[splits.fold.isin(folds)].to_csv(out / "audits/split_manifest.csv", index=False)
    layer_rows, search_rows, trial_parts, metric_rows = [], [], [], []
    selected_rows = []
    for fold in folds:
        train, test = masks_for_fold(cache, splits, int(fold))
        target_layer, flanker_layer, layer_table = select_source_layers(source, cache["target_labels"], cache["flanker_labels"], train)
        layer_table["fold"] = fold
        layer_table["selected"] = ((layer_table.source == "target") & (layer_table.layer == target_layer)) | ((layer_table.source == "flanker") & (layer_table.layer == flanker_layer))
        layer_rows.append(layer_table)
        family_specs = {"M0_full_WR2": [(80, 120)], "M1_simultaneous": [(80, 120)], "M2_flanker_early_target_late": timing_grid(args.grid_limit), "M3_target_early_flanker_late": timing_grid(args.grid_limit)}
        selected: dict[str, tuple[int, int]] = {}
        for family, specs in family_specs.items():
            scored = []
            for delay, decay in specs:
                mu = make_mu_by_group(cache, source, train, group_params, family, delay, decay, target_layer, flanker_layer)
                df = simulate(cache, mu, group_params, t0_mean, t0_sd, family, seeds[0], args.device)
                met = evaluate(df[train].copy())
                row = {"fold": fold, "model": family, "delay_ms": delay, "decay_ms": decay, "target_layer": target_layer, "flanker_layer": flanker_layer, **met}
                search_rows.append(row)
                scored.append(row)
            best = min(scored, key=lambda x: x["selection_score"])
            selected[family] = (int(best["delay_ms"]), int(best["decay_ms"]))
            selected_rows.append({"fold": fold, **best})
        baseline_by_seed = {}
        results_by_seed: dict[int, dict[str, dict[str, float]]] = {}
        for seed in seeds:
            results_by_seed[seed] = {}
            for family, (delay, decay) in selected.items():
                mu = make_mu_by_group(cache, source, train, group_params, family, delay, decay, target_layer, flanker_layer)
                df = simulate(cache, mu, group_params, t0_mean, t0_sd, family, seed, args.device)
                test_df = df[test].copy()
                test_df["fold"] = fold
                test_df["delay_ms"] = delay
                test_df["decay_ms"] = decay
                trial_parts.append(test_df)
                met = evaluate(test_df)
                results_by_seed[seed][family] = met
                metric_rows.append({"fold": fold, "seed": seed, "model": family, "delay_ms": delay, "decay_ms": decay, **met})
        # Pass is defined relative to the same-seed M0 joint-first-passage control.
        for seed in seeds:
            base = results_by_seed[seed]["M0_full_WR2"]
            for row in metric_rows:
                if row["fold"] != fold or row["seed"] != seed:
                    continue
                if row["model"] == "M0_full_WR2":
                    row["seed_pass"] = False
                    continue
                row["seed_pass"] = bool(row["young_incongruent_fast_error_gap"] <= 0.072 and row["young_incongruent_median_rt_gap"] <= 0.1125 and row["other_cell_composite"] <= 1.10 * base["other_cell_composite"] and row["young_20_29_congruent_model_error"] > 0 and row["older_80_89_congruent_model_error"] > 0 and row["curve_rmse"] < base["curve_rmse"] and row["rt_wasserstein"] < base["rt_wasserstein"] and row["error_mae"] < base["error_mae"])
    layers = pd.concat(layer_rows, ignore_index=True)
    search = pd.DataFrame(search_rows)
    selected_df = pd.DataFrame(selected_rows)
    metrics = pd.DataFrame(metric_rows)
    trials = pd.concat(trial_parts, ignore_index=True)
    layers.to_csv(out / "metrics/source_layer_selection.csv", index=False)
    search.to_csv(out / "metrics/training_timing_grid.csv", index=False)
    selected_df.to_csv(out / "metrics/selected_timing_by_fold.csv", index=False)
    metrics.to_csv(out / "metrics/test_seed_metrics.csv", index=False)
    trials.to_csv(out / "metrics/test_trial_predictions.csv", index=False)
    summary = metrics.groupby("model", as_index=False).agg(seed_fold_runs=("seed", "size"), pass_rate=("seed_pass", "mean"), fast_error_gap=("young_incongruent_fast_error_gap", "mean"), median_rt_gap=("young_incongruent_median_rt_gap", "mean"), curve_rmse=("curve_rmse", "mean"), rt_wasserstein=("rt_wasserstein", "mean"), error_mae=("error_mae", "mean"))
    fold_pass = metrics.groupby(["model", "fold"], as_index=False)["seed_pass"].mean()
    fold_pass["fold_pass"] = fold_pass.seed_pass >= 0.8
    fold_counts = fold_pass.groupby("model")["fold_pass"].sum().rename("folds_passing")
    summary = summary.merge(fold_counts, on="model", how="left")
    summary["formal_pass"] = (summary.pass_rate >= 0.8) & (summary.folds_passing >= 3)
    summary.to_csv(out / "metrics/model_summary.csv", index=False)
    # Focused report figure.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), facecolor="white")
    order = ["M0_full_WR2", "M1_simultaneous", "M2_flanker_early_target_late", "M3_target_early_flanker_late"]
    labels = ["M0 Full", "M1 Simultaneous", "M2 Flanker→Target", "M3 Reversed"]
    colors = ["#777777", "#56B4E9", "#009E73", "#E69F00"]
    q = summary.set_index("model").reindex(order)
    for ax, col, title in zip(axes, ["fast_error_gap", "median_rt_gap", "pass_rate"], ["Young incongruent\nfast-error gap", "Young incongruent\nmedian RT gap", "Seed/fold pass rate"]):
        ax.bar(np.arange(4), q[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(np.arange(4), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in")
    axes[0].axhline(0.072, color="#D55E00", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Absolute error-rate gap")
    axes[1].axhline(0.1125, color="#D55E00", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Absolute RT gap (s)")
    axes[2].axhline(0.8, color="#D55E00", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Proportion")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"figures/dual_route_model_comparison.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    passed = summary[summary.formal_pass]
    lines = ["# Flanker 双通道 Wong–Wang 比较", "", f"- 四折运行数：{len(folds)}", f"- 随机种子数：{len(seeds)}", f"- 正式通过模型：{', '.join(passed.model) if len(passed) else '无'}", "", "## 结果", ""]
    for r in summary.itertuples():
        lines.append(f"- {r.model}: fast-error gap={r.fast_error_gap:.3f}, median RT gap={r.median_rt_gap:.3f}s, pass rate={r.pass_rate:.2f}, passing folds={int(r.folds_passing)}")
    lines += ["", "## 解释限制", "", "- 这是 16 名被试内部验证，不是独立外部验证。", "- 老年组只有 4 人，年龄机制结论仍为探索性。", "- 只有 M2 同时优于同步输入 M1 与反向对照 M3，才支持 flanker-early/target-late 的特定时序解释。"]
    (out / "summaries/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "summary": summary.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

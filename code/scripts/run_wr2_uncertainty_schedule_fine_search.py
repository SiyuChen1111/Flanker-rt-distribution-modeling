#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
BASE_SCRIPT = BASE_DIR / "mechanism_redesign_conflict_adaptive_schedule/scripts/run_mechanism_redesign_conflict_adaptive_schedule.py"
OUT_DIR = BASE_DIR / "wr2_uncertainty_schedule_fine_search"
SEED_PARAMS = {
    "compression_low": 0.72,
    "compression_high": 0.45,
    "theta_quantile": 0.50,
    "temp": 0.30,
    "score_window_s": 0.15,
    "late_shift_ms": -40,
    "early_phase_shortening_ms": 30,
    "transition_width": 1.00,
}
SEED_YOUNG_INCONGRUENT_ERROR = 0.1966


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine search around the Word-compatible WR2 uncertainty schedule seed.")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--device", choices=["cpu"], default="cpu")
    p.add_argument("--max-full-candidates", type=int, default=240)
    return p.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location("mechanism_redesign_base_for_wr2_fine_search", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base mechanism script: {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mechanism_redesign_base_for_wr2_fine_search"] = mod
    spec.loader.exec_module(mod)
    mod.OUT_DIR = OUT_DIR
    mod.SEED = 20260607
    mod.NOISE_SEED = 20260608
    mod.LAPSE_SEED = 20260609
    return mod


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / sd


def entropy_and_gap(mu: np.ndarray, hi: int) -> tuple[np.ndarray, np.ndarray]:
    logits = mu[:, :hi, :] - mu[:, :hi, :].max(axis=2, keepdims=True)
    prob = np.exp(logits)
    prob = prob / np.maximum(prob.sum(axis=2, keepdims=True), 1e-9)
    entropy = -(prob * np.log(np.maximum(prob, 1e-9))).sum(axis=2).mean(axis=1)
    sorted_mu = np.sort(mu[:, :hi, :], axis=2)
    gap = (sorted_mu[:, :, -1] - sorted_mu[:, :, -2]).mean(axis=1)
    return entropy.astype(np.float32), gap.astype(np.float32)


def general_uncertainty_score(mu: np.ndarray, hi: int) -> np.ndarray:
    entropy, gap = entropy_and_gap(mu, hi)
    return ((zscore(entropy) + zscore(-gap)) / 2.0).astype(np.float32)


def install_wr2_override(m) -> None:
    original_build_candidate_mu = m.build_candidate_mu

    def build_uncertainty_schedule_mu(
        group_layers: dict[str, np.ndarray],
        base_schedule: dict[str, float],
        adaptive: dict[str, Any],
        evidence_gain: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low = float(adaptive["compression_low"])
        high = float(adaptive["compression_high"])
        low_df = m.schedule_df_from_params(
            low,
            int(base_schedule["late_shift_ms"]),
            float(base_schedule["transition_width"]),
            int(base_schedule["early_phase_shortening_ms"]),
        )
        high_df = m.schedule_df_from_params(
            high,
            int(base_schedule["late_shift_ms"]),
            float(base_schedule["transition_width"]),
            int(base_schedule["early_phase_shortening_ms"]),
        )
        mu_low = m.deterministic_schedule_mu(group_layers, low_df, evidence_gain)
        mu_high = m.deterministic_schedule_mu(group_layers, high_df, evidence_gain)
        hi = max(1, int(float(adaptive["score_window_s"]) / m.DT))
        score = general_uncertainty_score(mu_low, hi)
        theta = np.quantile(score, float(adaptive["theta_quantile"]))
        strength = m.sigmoid((score - theta) / max(float(adaptive["temp"]), 1e-6)).astype(np.float32)
        mu = mu_low + strength[:, None, None] * (mu_high - mu_low)
        return mu.astype(np.float32), score, np.repeat(strength[:, None], m.TIME_STEPS, axis=1)

    def build_candidate_mu(group_layers, target, flanker, evidence_gain, spec):
        adaptive = spec["adaptive"]
        if adaptive["type"] == "WR2_uncertainty_schedule":
            mu, score, control = build_uncertainty_schedule_mu(group_layers, spec["schedule"], adaptive, evidence_gain)
            low = float(adaptive["compression_low"])
            high = float(adaptive["compression_high"])
            compression_proxy = low - (low - high) * control.mean(axis=1)
            return mu, score, compression_proxy.astype(np.float32), control
        return original_build_candidate_mu(group_layers, target, flanker, evidence_gain, spec)

    m.build_candidate_mu = build_candidate_mu


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def make_wr2_spec(params: dict[str, float], role: str = "fine") -> dict[str, Any]:
    schedule = {
        "compression": float(params["compression_low"]),
        "late_shift_ms": int(params["late_shift_ms"]),
        "transition_width": float(params["transition_width"]),
        "early_phase_shortening_ms": int(params["early_phase_shortening_ms"]),
    }
    adaptive = {
        "type": "WR2_uncertainty_schedule",
        "compression_low": float(params["compression_low"]),
        "compression_high": float(params["compression_high"]),
        "theta_quantile": float(params["theta_quantile"]),
        "temp": float(params["temp"]),
        "score_window_s": float(params["score_window_s"]),
    }
    payload = {"schedule": schedule, "adaptive": adaptive, "role": role}
    model_id = "WR2_seed_current" if role == "seed" else f"WR2_fine_{stable_id(payload)}"
    return {
        "model_family": "WR2_uncertainty_schedule_fine",
        "model_config_id": model_id,
        "schedule_config_id": f"c{schedule['compression']:.2f}_ls{schedule['late_shift_ms']}_tw{schedule['transition_width']:.2f}_ep{schedule['early_phase_shortening_ms']}",
        "adaptive_schedule_config_id": adaptive["type"],
        "lapse_config_id": "none",
        "noise_config_id": "time_gap_selected",
        "schedule": schedule,
        "adaptive": adaptive,
        "noise_mode": "time_gap_selected",
        "lapse": {"type": "none"},
        "display_name": model_id,
    }


def param_distance(params: dict[str, float]) -> float:
    scales = {
        "compression_low": 0.02,
        "compression_high": 0.03,
        "theta_quantile": 0.05,
        "temp": 0.08,
        "score_window_s": 0.03,
        "late_shift_ms": 10,
        "early_phase_shortening_ms": 10,
        "transition_width": 0.10,
    }
    return float(
        sum(((float(params[k]) - float(SEED_PARAMS[k])) / float(scales[k])) ** 2 for k in scales)
    )


def candidate_params(mode: str, max_full_candidates: int) -> list[dict[str, float]]:
    if mode == "smoke":
        return [
            dict(SEED_PARAMS),
            {**SEED_PARAMS, "compression_high": 0.48},
            {**SEED_PARAMS, "compression_high": 0.42},
            {**SEED_PARAMS, "compression_low": 0.70},
            {**SEED_PARAMS, "compression_low": 0.74},
            {**SEED_PARAMS, "theta_quantile": 0.55},
            {**SEED_PARAMS, "theta_quantile": 0.45},
            {**SEED_PARAMS, "temp": 0.22},
            {**SEED_PARAMS, "temp": 0.38},
            {**SEED_PARAMS, "score_window_s": 0.12},
            {**SEED_PARAMS, "late_shift_ms": -30},
            {**SEED_PARAMS, "early_phase_shortening_ms": 40},
        ]

    grid = {
        "compression_low": [0.68, 0.70, 0.72, 0.74, 0.76],
        "compression_high": [0.42, 0.45, 0.48, 0.51],
        "theta_quantile": [0.45, 0.50, 0.55, 0.60],
        "temp": [0.22, 0.30, 0.38],
        "score_window_s": [0.12, 0.15, 0.18],
        "late_shift_ms": [-50, -40, -30],
        "early_phase_shortening_ms": [20, 30, 40],
        "transition_width": [0.90, 1.00, 1.10],
    }
    keys = list(grid)
    all_params = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
    all_params = sorted(all_params, key=lambda p: (param_distance(p), json.dumps(p, sort_keys=True)))
    keep = [dict(SEED_PARAMS)]
    seen = {json.dumps(SEED_PARAMS, sort_keys=True)}
    for params in all_params:
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        keep.append(params)
        seen.add(key)
        if len(keep) >= max_full_candidates:
            break
    return keep


def wr2_candidate_grid_factory(max_full_candidates: int):
    def grid(mode: str) -> list[dict[str, Any]]:
        params = candidate_params(mode, max_full_candidates)
        specs = []
        seen_ids = set()
        for idx, p in enumerate(params):
            spec = make_wr2_spec(p, "seed" if idx == 0 else "fine")
            if spec["model_config_id"] in seen_ids:
                continue
            seen_ids.add(spec["model_config_id"])
            specs.append(spec)
        return specs

    return grid


def copy_outputs() -> None:
    mapping = {
        "mechanism_redesign_model_ranking.csv": "wr2_fine_search_all_candidates.csv",
        "mechanism_redesign_model_summary.csv": "wr2_fine_search_condition_summary.csv",
        "mechanism_redesign_pass_fail_table.csv": "wr2_fine_search_pass_fail_table.csv",
        "mechanism_redesign_top_candidates_trial_level.csv": "wr2_fine_search_top_candidates_trial_level.csv",
        "mechanism_redesign_trajectory_diagnostics.csv": "wr2_fine_search_trajectory_diagnostics.csv",
        "conflict_adaptive_schedule_diagnostics.csv": "wr2_fine_search_uncertainty_schedule_diagnostics.csv",
        "mechanism_redesign_pareto_front.csv": "wr2_fine_search_pareto_front.csv",
    }
    metrics = OUT_DIR / "metrics"
    for src, dst in mapping.items():
        if (metrics / src).exists():
            shutil.copy2(metrics / src, metrics / dst)

    rank_path = metrics / "wr2_fine_search_all_candidates.csv"
    if rank_path.exists():
        rank = pd.read_csv(rank_path)
        best = rank.sort_values(["pass_strict", "pass_main", "fail_count_main", "combined_score"], ascending=[False, False, True, True]).head(20)
        best.to_csv(metrics / "wr2_fine_search_best_models.csv", index=False)

    log_src = OUT_DIR / "logs/mechanism_redesign_run_log.txt"
    if log_src.exists():
        shutil.copy2(log_src, OUT_DIR / "logs/run_log.txt")


def write_wr2_summary(mode: str) -> None:
    metrics = OUT_DIR / "metrics"
    rank_path = metrics / "wr2_fine_search_all_candidates.csv"
    pass_path = metrics / "wr2_fine_search_pass_fail_table.csv"
    if not rank_path.exists() or not pass_path.exists():
        return
    rank = pd.read_csv(rank_path)
    passed = pd.read_csv(pass_path)
    best = rank.iloc[0]
    main_rank = rank[rank["pass_main"].astype(bool)]
    recommended = main_rank.iloc[0] if not main_rank.empty else best
    strict = int(passed["pass_strict"].sum())
    main = int(passed["pass_main"].sum())
    improved = rank[rank["young_20_29_incongruent_error_rate"] < SEED_YOUNG_INCONGRUENT_ERROR]
    improved_main = improved[improved["pass_main"].astype(bool)]
    seed = rank[rank["model_config_id"].eq("WR2_seed_current")]
    seed_row = seed.iloc[0] if not seed.empty else None

    if strict > 0:
        conclusion = "找到至少一个 strict survivor，可作为新的候选最优模型。"
    elif not improved_main.empty:
        conclusion = "找到优于当前 seed 的 main survivor，但仍未完全解决 strict 层面的匹配。"
    elif main > 0:
        conclusion = "保留 main survivor，但未明显优于当前 seed；Word 兼容路线可能已接近当前框架下较优状态。"
    else:
        conclusion = "本轮未保住 main survivor；应回退到当前 WR2 seed 或 Word 版基础模型。"

    lines = [
        "# WR2 uncertainty schedule fine search summary",
        "",
        f"- Run mode: {mode}.",
        f"- Candidates tested: {len(rank)}.",
        f"- Main survivors: {main}.",
        f"- Strict survivors: {strict}.",
        f"- Candidates improving young incongruent error below {SEED_YOUNG_INCONGRUENT_ERROR:.4f}: {len(improved)}.",
        f"- Improved main survivors: {len(improved_main)}.",
        f"- Best score-only candidate: `{best['model_config_id']}`.",
        f"- Recommended candidate: `{recommended['model_config_id']}`.",
        "",
        "## Current seed reference",
        "",
    ]
    if seed_row is not None:
        lines += [
            f"- Young incongruent error: {seed_row['young_20_29_incongruent_error_rate']:.4f}.",
            f"- Young congruent error: {seed_row['young_20_29_congruent_error_rate']:.4f}.",
            f"- Older incongruent error: {seed_row['older_80_89_incongruent_error_rate']:.4f}.",
            f"- Pass main / strict: {bool(seed_row['pass_main'])} / {bool(seed_row['pass_strict'])}.",
            "",
        ]
    lines += [
        "## Recommended candidate metrics",
        "",
        f"- Young overall accuracy: {recommended['young_20_29_overall_accuracy']:.4f}.",
        f"- Young congruent error rate: {recommended['young_20_29_congruent_error_rate']:.4f}.",
        f"- Young incongruent error rate: {recommended['young_20_29_incongruent_error_rate']:.4f}.",
        f"- Young congruent error RT minus correct RT: {recommended['young_20_29_congruent_error_rt_minus_correct_rt']:.4f}.",
        f"- Older overall accuracy: {recommended['older_80_89_overall_accuracy']:.4f}.",
        f"- Older congruent error rate: {recommended['older_80_89_congruent_error_rate']:.4f}.",
        f"- Older incongruent error rate: {recommended['older_80_89_incongruent_error_rate']:.4f}.",
        f"- Older congruent error RT minus correct RT: {recommended['older_80_89_congruent_error_rt_minus_correct_rt']:.4f}.",
        "",
        "## Interpretation",
        "",
        f"- {conclusion}",
        "- Negative error-minus-correct RT is treated as a plausible fast-error / premature readout signature, not as an automatic failure.",
        "- This search keeps the Word-compatible backbone: no VGG retraining, no rhythmic attention branch, no lapse-based explanation.",
    ]
    (OUT_DIR / "summaries/wr2_fine_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs() -> None:
    required = [
        OUT_DIR / "metrics/wr2_fine_search_all_candidates.csv",
        OUT_DIR / "metrics/wr2_fine_search_best_models.csv",
        OUT_DIR / "metrics/wr2_fine_search_pass_fail_table.csv",
        OUT_DIR / "summaries/wr2_fine_search_summary.md",
        OUT_DIR / "logs/run_log.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Missing required outputs: {missing}")
    rank = pd.read_csv(OUT_DIR / "metrics/wr2_fine_search_all_candidates.csv")
    pass_fail = pd.read_csv(OUT_DIR / "metrics/wr2_fine_search_pass_fail_table.csv")
    if rank.empty or pass_fail.empty:
        raise AssertionError("Fine-search outputs are empty.")
    needed_rank = {
        "model_config_id",
        "pass_main",
        "pass_strict",
        "young_20_29_incongruent_error_rate",
        "young_20_29_congruent_error_rate",
        "older_80_89_incongruent_error_rate",
        "failure_reason_category",
    }
    missing_cols = sorted(needed_rank - set(rank.columns))
    if missing_cols:
        raise AssertionError(f"Missing expected ranking columns: {missing_cols}")
    if rank["failure_reason_category"].isna().any():
        raise AssertionError("Some candidates are missing failure_reason_category.")


def main() -> None:
    args = parse_args()
    m = load_base_module()
    install_wr2_override(m)
    m.candidate_grid = wr2_candidate_grid_factory(args.max_full_candidates)
    m.parse_args = lambda: argparse.Namespace(mode=args.mode, device=args.device)
    m.main()

    (OUT_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), OUT_DIR / "scripts" / Path(__file__).name)
    copy_outputs()
    write_wr2_summary(args.mode)
    validate_outputs()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww  # noqa: E402
from analyze_natural_layer_to_time_readout_rt_patterns import extract_readout_timing  # noqa: E402
from optimize_natural_layer_to_time_rt_shape import (  # noqa: E402
    ReadoutConfig,
    apply_readout,
    base_condition_df,
    build_natural_input,
    q,
    rt_bins,
    safe_mean,
)
from project_paths import PROJECT_ROOT  # noqa: E402

ANALYSIS_NAME = "representative_extreme_age_subset_5000"
OUT_DIR = PROJECT_ROOT / "artifacts/results/diagnostics/natural_layer_to_time_var_ww" / ANALYSIS_NAME
EVIDENCE_KEYS = ["evidence_conv3", "evidence_conv4", "evidence_conv5", "evidence_pooled", "evidence_final"]
GROUPS = ["young_20_29", "older_80_89", "older_70_89"]
T0_GRID = np.round(np.arange(0.10, 0.801, 0.05), 2).tolist()
T0_SD_GRID = [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run representative extreme-age finite model comparison.")
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument(
        "--input-dir",
        default=None,
        help="Optional source directory for manifests and evidence cache; defaults to output-dir.",
    )
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--time-steps", type=int, default=80)
    p.add_argument("--dt-ms", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260530)
    return p.parse_args()


def require_evidence_gate(root: Path) -> None:
    path = root / "audits/representative_subset_evidence_coverage_audit.csv"
    if not path.exists():
        raise RuntimeError("Evidence coverage audit missing; fitting is blocked.")
    audit = pd.read_csv(path)
    if audit.empty or not bool(audit.iloc[0]["coverage_gate_passed"]):
        raise RuntimeError("Evidence coverage gate did not pass; fitting is blocked.")


def load_trial_cache(root: Path) -> Dict[str, np.ndarray]:
    mapping = pd.read_csv(root / "manifests/representative_subset_trial_to_stimulus_mapping.csv")
    ev = np.load(root / "evidence_cache/representative_subset_layerwise_evidence.npz", allow_pickle=True)
    pos = {int(sid): i for i, sid in enumerate(ev["subset_stimulus_id"].astype(np.int64))}
    idx = np.asarray([pos[int(x)] for x in mapping["subset_stimulus_id"].astype(int)], dtype=np.int64)
    cache: Dict[str, np.ndarray] = {
        "target_labels": mapping["target_label"].to_numpy(dtype=np.int64),
        "flanker_labels": mapping["flanker_label"].to_numpy(dtype=np.int64),
        "response_labels": mapping["response_label"].to_numpy(dtype=np.int64),
        "true_rt": mapping["human_rt"].to_numpy(dtype=np.float32),
        "human_correct": mapping["human_correct"].astype(bool).to_numpy(),
        "congruency": mapping["congruency"].to_numpy(dtype=np.int64),
        "row_indices": mapping["row_index"].to_numpy(dtype=np.int64),
        "age_group": mapping["analysis_group"].astype(str).to_numpy(),
        "analysis_group": mapping["analysis_group"].astype(str).to_numpy(),
        "original_age_group": mapping["original_age_group"].astype(str).to_numpy(),
        "user_id": mapping["subject_id"].astype(str).to_numpy(),
        "subset_stimulus_id": mapping["subset_stimulus_id"].to_numpy(dtype=np.int64),
        "sampling_stratum": mapping["sampling_stratum"].astype(str).to_numpy(),
    }
    for key in EVIDENCE_KEYS:
        cache[key] = ev[key][idx].astype(np.float32)
    return cache


def subset_cache(cache: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    n = len(cache["true_rt"])
    for k, v in cache.items():
        arr = np.asarray(v)
        out[k] = arr[mask] if arr.shape[0] == n else arr
    return out


def subset_outputs(outputs: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in outputs.items():
        arr = np.asarray(value)
        out[key] = arr[indices] if arr.shape[0] >= int(indices.max()) + 1 else arr
    return out


def add_meta(df: pd.DataFrame, cache: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    out["analysis_group"] = cache["analysis_group"].astype(str)
    out["original_age_group"] = cache["original_age_group"].astype(str)
    out["user_id"] = cache["user_id"].astype(str)
    out["subset_stimulus_id"] = cache["subset_stimulus_id"].astype(np.int64)
    out["sampling_stratum"] = cache["sampling_stratum"].astype(str)
    out["trial_output_index"] = np.arange(len(out), dtype=np.int64)
    return out


def fixed_t0_noise(n: int, sd: float, seed: int) -> np.ndarray:
    if sd <= 0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sd, size=n)
    return np.clip(x, -2.5 * sd, 2.5 * sd).astype(np.float32)


def apply_group_t0(df: pd.DataFrame, t0_by_group: Dict[str, float], sd_by_group: Dict[str, float], seed: int) -> pd.DataFrame:
    out = df.copy()
    pred = np.zeros(len(out), dtype=np.float32)
    t0_mean_col = np.zeros(len(out), dtype=np.float32)
    t0_sd_col = np.zeros(len(out), dtype=np.float32)
    for i, (group, idx) in enumerate(out.groupby("analysis_group", sort=True).groups.items()):
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        mean = float(t0_by_group.get(group, 0.25))
        sd = float(sd_by_group.get(group, 0.0))
        noise = fixed_t0_noise(len(idx_arr), sd, seed + i)
        vals = out.loc[idx_arr, "decision_time"].to_numpy(dtype=float) + mean + noise
        pred[idx_arr] = np.maximum(vals, 0.05)
        t0_mean_col[idx_arr] = mean
        t0_sd_col[idx_arr] = sd
    out["pred_rt"] = pred
    out["t0_mean"] = t0_mean_col
    out["t0_sd"] = t0_sd_col
    out["model_correct"] = out["pred_choice"].to_numpy(dtype=np.int64) == out["target_label"].to_numpy(dtype=np.int64)
    return out


def metrics_for_group(df: pd.DataFrame, outputs: Dict[str, np.ndarray], dt_ms: int, t0_for_mech: float) -> Dict[str, Any]:
    rt = df["pred_rt"].to_numpy(dtype=float)
    hrt = df["true_rt"].to_numpy(dtype=float)
    correct = df["model_correct"].to_numpy(dtype=bool)
    human_correct = df["human_correct"].to_numpy(dtype=bool)
    incong = df["congruency"].to_numpy(dtype=np.int64) == 1
    out_idx = df["trial_output_index"].to_numpy(dtype=np.int64)
    caf = rt_bins(df, str(df["model_name"].iloc[0]), n_bins=5)
    pivot = caf.pivot_table(index="rt_bin", columns="source", values="accuracy")
    caf_rmse = float(np.sqrt(np.nanmean((pivot.get("model") - pivot.get("human")) ** 2))) if {"model", "human"}.issubset(set(pivot.columns)) else float("nan")
    ro = extract_readout_timing(df, subset_outputs(outputs, out_idx), t0_seconds=t0_for_mech, dt_ms=dt_ms)
    inc_cor = ro[ro["congruency"].eq(1) & ro["model_correct"]]
    inc_err = ro[ro["congruency"].eq(1) & ~ro["model_correct"]]
    return {
        "model_name": str(df["model_name"].iloc[0]),
        "analysis_group": str(df["analysis_group"].iloc[0]),
        "n_trials": int(len(df)),
        "n_subjects": int(df["user_id"].nunique()),
        "human_accuracy": float(human_correct.mean()),
        "model_accuracy": float(correct.mean()),
        "human_choice_agreement": float((df["pred_choice"].to_numpy() == df["response_label"].to_numpy()).mean()),
        "model_crossing_rate": float(df["crossed"].astype(bool).mean()),
        "model_no_crossing_rate": float((~df["crossed"].astype(bool)).mean()),
        "human_incongruent_error_rate": float((~human_correct[incong]).mean()),
        "model_incongruent_error_rate": float((~correct[incong]).mean()),
        "human_mean_rt": float(np.mean(hrt)),
        "model_mean_rt": float(np.mean(rt)),
        "human_median_rt": float(np.median(hrt)),
        "model_median_rt": float(np.median(rt)),
        "human_q90_minus_q10": q(hrt, 0.90) - q(hrt, 0.10),
        "model_q90_minus_q10": q(rt, 0.90) - q(rt, 0.10),
        "human_q95_minus_median": q(hrt, 0.95) - q(hrt, 0.50),
        "model_q95_minus_median": q(rt, 0.95) - q(rt, 0.50),
        "human_correct_rt": safe_mean(hrt[human_correct]),
        "human_error_rt": safe_mean(hrt[~human_correct]),
        "model_correct_rt": safe_mean(rt[correct]),
        "model_error_rt": safe_mean(rt[~correct]),
        "human_incongruent_correct_rt": safe_mean(hrt[incong & human_correct]),
        "human_incongruent_error_rt": safe_mean(hrt[incong & ~human_correct]),
        "model_incongruent_correct_rt": safe_mean(rt[incong & correct]),
        "model_incongruent_error_rt": safe_mean(rt[incong & ~correct]),
        "caf_binwise_rmse": caf_rmse,
        "target_recovery_time_correct": safe_mean(inc_cor["target_recovery_time"]) if len(inc_cor) else float("nan"),
        "target_recovery_time_error": safe_mean(inc_err["target_recovery_time"]) if len(inc_err) else float("nan"),
        "target_recovery_time_error_minus_correct": (safe_mean(inc_err["target_recovery_time"]) - safe_mean(inc_cor["target_recovery_time"])) if len(inc_cor) and len(inc_err) else float("nan"),
        "early_flanker_dominance": safe_mean(inc_err["early_s_flanker_ge_target"].astype(float)) if len(inc_err) else float("nan"),
        "late_target_recovery": safe_mean(inc_cor["late_s_target_ge_flanker"].astype(float)) if len(inc_cor) else float("nan"),
        "readout_before_target_recovery_error": safe_mean(inc_err["readout_before_target_recovery"].astype(float)) if len(inc_err) else float("nan"),
    } | {f"human_q{int(p*100):02d}": q(hrt, p) for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]} | {f"model_q{int(p*100):02d}": q(rt, p) for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]}


def score_summary(summary: pd.DataFrame) -> Dict[str, float]:
    q_cols = [f"q{int(p*100):02d}" for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]]
    q_errs = []
    for qn in q_cols:
        q_errs.extend((summary[f"model_{qn}"] - summary[f"human_{qn}"]).abs().tolist())
    score_rt_quantile = float(np.nanmean(q_errs))
    score_caf = float(summary["caf_binwise_rmse"].mean())
    score_accuracy = float((summary["model_accuracy"] - summary["human_accuracy"]).abs().mean())
    score_mechanism = float(np.nanmean(np.maximum(0.0, -summary["target_recovery_time_error_minus_correct"].fillna(0).to_numpy())))
    crossing_shortfall = np.maximum(0.95 - summary["model_crossing_rate"].to_numpy(float), 0.0)
    score_crossing_coverage = float(np.mean(crossing_shortfall))
    crossing_gate_passed = bool(np.all(crossing_shortfall <= 1e-12))
    unconstrained_total = 2.0 * score_rt_quantile + 1.5 * score_caf + 1.0 * score_accuracy + 0.25 * score_mechanism
    # A deadline sentinel is censoring, not an observed RT. Prevent a candidate
    # with extensive deadline fallback from winning by fitting the RT ceiling.
    total = unconstrained_total if crossing_gate_passed else unconstrained_total + 10.0 + score_crossing_coverage
    return {
        "score_total": total,
        "score_total_unconstrained": unconstrained_total,
        "score_rt_quantile": score_rt_quantile,
        "score_caf": score_caf,
        "score_accuracy": score_accuracy,
        "score_mechanism": score_mechanism,
        "score_crossing_coverage": score_crossing_coverage,
        "crossing_gate_passed": crossing_gate_passed,
    }


def run_base(cache: Dict[str, np.ndarray], args: argparse.Namespace, *, model_name: str, evidence_gain: float, threshold: float, cfg: ReadoutConfig, variant_type: str = "deterministic", sigma_type: str = "none", sigma_base: float = 0.0, sigma_conflict: float = 0.0, seed: int | None = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    run_seed = args.seed if seed is None else int(seed)
    ww_input = build_natural_input(
        cache,
        evidence_gain=evidence_gain,
        time_steps=args.time_steps,
        variant_type=variant_type,
        sigma_type=sigma_type,
        sigma_base=sigma_base,
        sigma_conflict=sigma_conflict,
        seed=run_seed,
    )
    outputs = run_ww(ww_input, time_steps=args.time_steps, dt_ms=args.dt_ms, threshold=threshold, noise_ampa=0.0, device=args.device, seed=run_seed, readout_mode="baseline", t0_seconds=0.25, choice_temperature=0.01)
    df = base_condition_df(cache, outputs, condition_name=model_name, variant_type=variant_type, evidence_gain=evidence_gain, threshold=threshold, seed=run_seed, sigma_type=sigma_type, sigma_base=sigma_base, sigma_conflict=sigma_conflict)
    df = add_meta(df, cache)
    df = apply_readout(df, outputs, cfg=cfg, threshold=threshold, dt_ms=args.dt_ms, t0_seconds=0.0)
    df["model_name"] = model_name
    df["evidence_gain"] = evidence_gain
    df["threshold"] = threshold
    df["sustained_k"] = cfg.sustained_k
    df["margin"] = cfg.margin
    df["min_decision_time"] = cfg.min_decision_time
    df["sigma_type"] = sigma_type
    df["sigma_base"] = sigma_base
    df["sigma_conflict"] = sigma_conflict
    return df, outputs


def best_t0_for_groups(df: pd.DataFrame, sd_by_group: Dict[str, float] | None = None, seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sd_by_group = sd_by_group or {}
    best: Dict[str, float] = {}
    for group, part in df.groupby("analysis_group", sort=True):
        errs = []
        for t0 in T0_GRID:
            pred = part["decision_time"].to_numpy(float) + t0 + fixed_t0_noise(len(part), float(sd_by_group.get(group, 0.0)), seed)
            human = part["true_rt"].to_numpy(float)
            err = abs(np.median(pred) - np.median(human)) + 0.5 * abs(np.mean(pred) - np.mean(human))
            errs.append((err, t0))
        best[group] = float(min(errs)[1])
    return apply_group_t0(df, best, sd_by_group, seed), best


def evaluate_model(df: pd.DataFrame, outputs: Dict[str, np.ndarray], args: argparse.Namespace, t0_means: Dict[str, float], t0_sds: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = [metrics_for_group(part, outputs, args.dt_ms, float(t0_means.get(group, 0.25))) for group, part in df.groupby("analysis_group", sort=True)]
    summary = pd.DataFrame(rows)
    scores = score_summary(summary)
    return summary, scores


def parameter_rows(model_name: str, groups: Iterable[str], t0: Dict[str, float], t0sd: Dict[str, float], params: Dict[str, Any], scores: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for g in groups:
        rows.append({
            "model_name": model_name,
            "analysis_group": g,
            "t0_mean": float(t0.get(g, np.nan)),
            "t0_sd": float(t0sd.get(g, 0.0)),
            "evidence_gain": params.get("evidence_gain"),
            "threshold": params.get("threshold"),
            "sustained_k": params.get("sustained_k"),
            "margin": params.get("margin"),
            "min_decision_time": params.get("min_decision_time"),
            "sigma_type": params.get("sigma_type", "none"),
            "sigma_base": params.get("sigma_base", 0.0),
            "sigma_conflict": params.get("sigma_conflict", 0.0),
            "parameter_details": json.dumps(params, ensure_ascii=False, sort_keys=True),
            **scores,
        })
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    input_root = Path(args.input_dir) if args.input_dir else output_root
    fit_dir = output_root / "fitting"
    sum_dir = output_root / "summaries"
    fit_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)
    require_evidence_gate(input_root)
    cache = load_trial_cache(input_root)
    active_groups = sorted(set(cache["analysis_group"].astype(str)))

    all_trials: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    all_caf: List[pd.DataFrame] = []
    all_params: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    outputs_by_model: Dict[str, Dict[str, np.ndarray]] = {}

    # R0
    cfg0 = ReadoutConfig("sustained_crossing", min_decision_time=0.02, sustained_k=3, margin=0.01)
    df0, out0 = run_base(cache, args, model_name="R0_fixed_current", evidence_gain=0.80, threshold=0.12, cfg=cfg0)
    t0_r0 = {g: 0.25 for g in active_groups}
    t0sd_r0 = {g: 0.0 for g in active_groups}
    df0 = apply_group_t0(df0, t0_r0, t0sd_r0, args.seed)
    summ0, score0 = evaluate_model(df0, out0, args, t0_r0, t0sd_r0)
    outputs_by_model["R0_fixed_current"] = out0

    candidates: List[Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]] = []
    params0 = {"evidence_gain": 0.80, "threshold": 0.12, "sustained_k": 3, "margin": 0.01, "min_decision_time": 0.02}
    candidates.append((df0, out0, summ0, score0, t0_r0, t0sd_r0, params0))

    # R1
    df1_base, out1 = run_base(cache, args, model_name="R1_group_t0_mean", evidence_gain=0.80, threshold=0.12, cfg=cfg0)
    df1, t0_r1 = best_t0_for_groups(df1_base, seed=args.seed + 10)
    t0sd_r1 = {g: 0.0 for g in active_groups}
    summ1, score1 = evaluate_model(df1, out1, args, t0_r1, t0sd_r1)
    candidates.append((df1, out1, summ1, score1, t0_r1, t0sd_r1, params0))
    outputs_by_model["R1_group_t0_mean"] = out1

    # R2
    best_r2 = None
    for young_sd in T0_SD_GRID:
        for older_sd in T0_SD_GRID:
            sd = {g: (older_sd if g.startswith("older") else young_sd) for g in active_groups}
            dfx, t0x = best_t0_for_groups(df1_base, sd_by_group=sd, seed=args.seed + 20)
            summx, scorex = evaluate_model(dfx, out1, args, t0x, sd)
            rec = (scorex["score_total"], dfx, summx, scorex, t0x, sd)
            if best_r2 is None or rec[0] < best_r2[0]:
                best_r2 = rec
    assert best_r2 is not None
    df2, summ2, score2, t0_r2, t0sd_r2 = best_r2[1], best_r2[2], best_r2[3], best_r2[4], best_r2[5]
    df2["model_name"] = "R2_group_t0_mean_sd"
    summ2["model_name"] = "R2_group_t0_mean_sd"
    candidates.append((df2, out1, summ2, score2, t0_r2, t0sd_r2, params0))
    outputs_by_model["R2_group_t0_mean_sd"] = out1

    # R3 finite WW/readout candidates, group-specific by fitting each group separately.
    r3_param_grid = [
        (gain, thr, k, margin, mdt)
        for gain in [0.60, 0.80, 1.00]
        for thr in [0.12, 0.14, 0.16]
        for k in [2, 3, 4]
        for margin in [0.00, 0.02]
        for mdt in [0.00, 0.04]
    ]
    r3_parts = []
    r3_outputs_parts: Dict[str, List[np.ndarray]] = {}
    r3_params_by_group: Dict[str, Dict[str, Any]] = {}
    for group in active_groups:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        best_g = None
        for gain, thr, k, margin, mdt in r3_param_grid:
            cfg = ReadoutConfig("sustained_crossing", min_decision_time=mdt, sustained_k=k, margin=margin)
            dfg, outg = run_base(gc, args, model_name="R3_group_ww_readout", evidence_gain=gain, threshold=thr, cfg=cfg)
            dfg, t0g = best_t0_for_groups(dfg, seed=args.seed + 30)
            summg, scoreg = evaluate_model(dfg, outg, args, t0g, {group: 0.0})
            rec = (scoreg["score_total"], dfg, outg, summg, scoreg, t0g, {"evidence_gain": gain, "threshold": thr, "sustained_k": k, "margin": margin, "min_decision_time": mdt})
            if best_g is None or rec[0] < best_g[0]:
                best_g = rec
        assert best_g is not None
        r3_parts.append(best_g[1])
        for key, value in best_g[2].items():
            r3_outputs_parts.setdefault(key, []).append(value)
        r3_params_by_group[group] = best_g[6]
    df3 = pd.concat(r3_parts, ignore_index=True)
    df3["trial_output_index"] = np.arange(len(df3), dtype=np.int64)
    out3 = {k: np.concatenate(v, axis=0) for k, v in r3_outputs_parts.items()}
    t0_r3 = dict(zip(df3.groupby("analysis_group")["t0_mean"].first().index, df3.groupby("analysis_group")["t0_mean"].first().values))
    t0sd_r3 = {g: 0.0 for g in active_groups}
    summ3, score3 = evaluate_model(df3, out3, args, t0_r3, t0sd_r3)
    params3 = {"evidence_gain": "group_specific", "threshold": "group_specific", "sustained_k": "group_specific", "margin": "group_specific", "min_decision_time": "group_specific", "group_params": json.dumps(r3_params_by_group)}
    candidates.append((df3, out3, summ3, score3, t0_r3, t0sd_r3, params3))
    outputs_by_model["R3_group_ww_readout"] = out3

    # R4 representative variational noise, limited to requested grid with 5 seeds.
    r4_grid = [
        (stype, sb, sc)
        for stype in ["fixed_sigma", "conflict_dependent_sigma", "layer_weighted_sigma"]
        for sb in [0.00, 0.03, 0.06, 0.10, 0.15]
        for sc in [0.00, 0.05, 0.10, 0.15]
        if not (stype == "fixed_sigma" and sc != 0.0)
    ]
    best_r4 = None
    for stype, sb, sc in r4_grid:
        seed_trials = []
        seed_summ = []
        seed_scores = []
        last_out = None
        for si in range(5):
            seed = args.seed + 400 + si
            dfv, outv = run_base(cache, args, model_name="R4_variational_noise", evidence_gain=0.80, threshold=0.12, cfg=cfg0, variant_type="variational", sigma_type=stype, sigma_base=sb, sigma_conflict=sc, seed=seed)
            dfv, t0v = best_t0_for_groups(dfv, seed=seed)
            summv, scorev = evaluate_model(dfv, outv, args, t0v, {g: 0.0 for g in active_groups})
            seed_trials.append(dfv.assign(variational_seed=seed))
            seed_summ.append(summv.assign(variational_seed=seed))
            seed_scores.append(scorev["score_total"])
            last_out = outv
        mean_score = float(np.mean(seed_scores))
        rec = (mean_score, seed_trials[0], last_out, seed_summ[0], score_summary(pd.concat(seed_summ).groupby(["model_name", "analysis_group"], as_index=False).mean(numeric_only=True)), stype, sb, sc)
        if best_r4 is None or rec[0] < best_r4[0]:
            best_r4 = rec
    assert best_r4 is not None
    df4, out4, summ4, score4 = best_r4[1], best_r4[2], best_r4[3], best_r4[4]
    t0_r4 = dict(zip(df4.groupby("analysis_group")["t0_mean"].first().index, df4.groupby("analysis_group")["t0_mean"].first().values))
    t0sd_r4 = {g: 0.0 for g in active_groups}
    params4 = {"evidence_gain": 0.80, "threshold": 0.12, "sustained_k": 3, "margin": 0.01, "min_decision_time": 0.02, "sigma_type": best_r4[5], "sigma_base": best_r4[6], "sigma_conflict": best_r4[7]}
    candidates.append((df4, out4, summ4, score4, t0_r4, t0sd_r4, params4))
    outputs_by_model["R4_variational_noise"] = out4

    # R5: only three theoretically motivated combinations.
    combo_specs = [
        ("R5_combined_t0sd_ww", "R3", t0sd_r2, r3_params_by_group, None),
        ("R5_combined_t0sd_var", "R4", t0sd_r2, None, params4),
        ("R5_combined_t0_ww_var", "R4", {g: 0.0 for g in active_groups}, r3_params_by_group, params4),
    ]
    best_combo = None
    for name, base_type, sd_map, ww_map, var_params in combo_specs:
        if base_type == "R3":
            base_df = df3.copy()
            base_out = out3
        else:
            base_df, base_out = run_base(cache, args, model_name=name, evidence_gain=0.80, threshold=0.12, cfg=cfg0, variant_type="variational", sigma_type=params4["sigma_type"], sigma_base=float(params4["sigma_base"]), sigma_conflict=float(params4["sigma_conflict"]), seed=args.seed + 900)
        base_df["model_name"] = name
        dfx, t0x = best_t0_for_groups(base_df, sd_by_group=sd_map, seed=args.seed + 950)
        summx, scorex = evaluate_model(dfx, base_out, args, t0x, sd_map)
        rec = (scorex["score_total"], dfx, base_out, summx, scorex, t0x, sd_map, {"combined_from": name, **(params4 if var_params else params3)})
        if best_combo is None or rec[0] < best_combo[0]:
            best_combo = rec
    assert best_combo is not None
    df5, out5, summ5, score5, t0_r5, t0sd_r5, params5 = best_combo[1], best_combo[2], best_combo[3], best_combo[4], best_combo[5], best_combo[6], best_combo[7]
    df5["model_name"] = "R5_combined_best"
    df5["trial_output_index"] = np.arange(len(df5), dtype=np.int64)
    summ5["model_name"] = "R5_combined_best"
    candidates.append((df5, out5, summ5, score5, t0_r5, t0sd_r5, params5))
    outputs_by_model["R5_combined_best"] = out5

    for df, outputs, summ, scores, t0m, t0sd, params in candidates:
        model_name = str(df["model_name"].iloc[0])
        all_trials.append(df)
        all_summaries.append(summ)
        for group, part in df.groupby("analysis_group", sort=True):
            caf = rt_bins(part, model_name, n_bins=5)
            caf["analysis_group"] = group
            caf["model_name"] = model_name
            all_caf.append(caf)
        comparisons.append({"model_name": model_name, **scores})
        all_params.extend(parameter_rows(model_name, active_groups, t0m, t0sd, params, scores))

    trial = pd.concat(all_trials, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    comp = pd.DataFrame(comparisons).sort_values("score_total", ascending=True)
    params = pd.DataFrame(all_params)
    caf = pd.concat(all_caf, ignore_index=True)

    best_model = str(comp.iloc[0]["model_name"])
    best_trial = trial[trial["model_name"].eq(best_model)].copy()
    best_outputs = outputs_by_model.get(best_model, out5)
    if best_outputs["trajectory"].shape[0] != len(best_trial):
        best_outputs = out5

    # Best-model readout/mechanism source data.
    mech_rows = []
    for group, part in best_trial.groupby("analysis_group", sort=True):
        out_idx = part["trial_output_index"].to_numpy(dtype=np.int64)
        ro = extract_readout_timing(part, subset_outputs(best_outputs, out_idx), t0_seconds=float(part["t0_mean"].iloc[0]), dt_ms=args.dt_ms)
        ro["analysis_group"] = group
        ro["model_name"] = best_model
        mech_rows.append(ro)
    mech_trial = pd.concat(mech_rows, ignore_index=True)

    quant_rows = []
    for model, mpart in trial.groupby("model_name", sort=True):
        for group, gpart in mpart.groupby("analysis_group", sort=True):
            for source, col in [("human", "true_rt"), ("model", "pred_rt")]:
                vals = gpart[col].to_numpy(float)
                quant_rows.append({"model_name": model, "analysis_group": group, "source": source, **{f"q{int(p*100):02d}": q(vals, p) for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]}})
    quant = pd.DataFrame(quant_rows)

    summaries.to_csv(fit_dir / "representative_model_group_metrics.csv", index=False)
    comp.to_csv(fit_dir / "representative_model_comparison.csv", index=False)
    params.to_csv(fit_dir / "representative_parameter_estimates.csv", index=False)
    best_trial.to_csv(fit_dir / "representative_trial_level_predictions.csv", index=False)
    quant.to_csv(fit_dir / "representative_rt_quantiles.csv", index=False)
    caf.to_csv(fit_dir / "representative_caf.csv", index=False)
    summaries[[
        "model_name", "analysis_group", "human_correct_rt", "human_error_rt", "model_correct_rt", "model_error_rt",
        "human_incongruent_correct_rt", "human_incongruent_error_rt", "model_incongruent_correct_rt", "model_incongruent_error_rt",
    ]].to_csv(fit_dir / "representative_correct_error_rt.csv", index=False)
    summaries[[
        "model_name", "analysis_group", "target_recovery_time_correct", "target_recovery_time_error",
        "target_recovery_time_error_minus_correct", "early_flanker_dominance", "late_target_recovery",
        "readout_before_target_recovery_error",
    ]].to_csv(fit_dir / "representative_mechanism_summary.csv", index=False)
    mech_trial.to_csv(fit_dir / "representative_best_model_mechanism_trial_level.csv", index=False)

    best_params = params[params["model_name"].eq(best_model)]
    text = f"""# Representative Fitting Summary

Representative subset diagnostic / exploratory analysis. This is not a full age-group conclusion.

- models compared: R0 fixed, R1 group t0_mean, R2 group t0_mean+t0_sd, R3 group WW/readout, R4 variational evidence noise, R5 combined best
- best model by composite score: {best_model}
- score_total: {float(comp.iloc[0]['score_total']):.4f}
- t0 means by group: {json.dumps(dict(zip(best_params['analysis_group'], best_params['t0_mean'])), ensure_ascii=False)}
- t0 sds by group: {json.dumps(dict(zip(best_params['analysis_group'], best_params['t0_sd'])), ensure_ascii=False)}
- outputs are representative subset diagnostic / exploratory, not full age-group conclusions.
"""
    (sum_dir / "representative_fitting_summary.md").write_text(text, encoding="utf-8")
    print(json.dumps({"best_model": best_model, "score_total": float(comp.iloc[0]["score_total"]), "n_prediction_trials": int(len(best_trial))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

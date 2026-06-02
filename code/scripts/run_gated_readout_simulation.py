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

from optimize_natural_layer_to_time_rt_shape import ReadoutConfig  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_congruent_ww_dynamics_diagnostic import parse_group_params  # noqa: E402
from run_representative_extreme_age_subset_fitting import (  # noqa: E402
    apply_group_t0,
    load_trial_cache,
    run_base,
    subset_cache,
)

BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
OUT_DIR = BASE_DIR / "readout_choice_uncertainty_mechanism_comparison"
GROUPS = ["young_20_29", "older_80_89"]
GROUP_LABEL = {"young_20_29": "Young 20-29", "older_80_89": "Older 80-89"}
DT = 0.01
TIME_STEPS = 80
SEED = 20260530
NOISE_SEED = 20260601
MARGIN_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
MAX_DELAYS = [0.05, 0.10, 0.15, 0.20]
SOFT_THETAS = [0.0, 0.005, 0.01]
SOFT_TEMPS = [0.005, 0.01, 0.02]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run target-recovery-gated time+gap readout simulation.")
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def ensure_dirs() -> Dict[str, Path]:
    dirs = {
        "metrics": OUT_DIR / "metrics",
        "figures": OUT_DIR / "figures_publication",
        "summaries": OUT_DIR / "summaries",
        "logs": OUT_DIR / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_mean(x: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def safe_q(x: Iterable[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else math.nan


def corr_safe(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return math.nan
    if np.isclose(a[mask].std(), 0.0) or np.isclose(b[mask].std(), 0.0):
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def rmse(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else math.nan


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT_DIR / "figures_publication" / f"{name}.{ext}", dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax.tick_params(labelsize=9)


def normalize_congruency(s: pd.Series) -> pd.Series:
    return s.map({0: "congruent", 1: "incongruent"}).fillna(s.astype(str))


def selected_time_gap_params() -> Dict[str, Dict[str, float]]:
    ranking = pd.read_csv(OUT_DIR / "metrics/readout_choice_model_ranking.csv")
    best = ranking[ranking["model"].eq("M3_time_gap")].iloc[0]
    params: Dict[str, Dict[str, float]] = {}
    if str(best["param_mode"]) == "age_specific":
        for group in GROUPS:
            params[group] = {
                "sigma_base": float(best[f"{group}_sigma_base"]),
                "sigma_time": float(best[f"{group}_sigma_time"]),
                "sigma_gap": float(best[f"{group}_sigma_gap"]),
                "gap_scale": float(best[f"{group}_gap_scale"]),
            }
    else:
        for group in GROUPS:
            params[group] = {
                "sigma_base": float(best["sigma_base"]),
                "sigma_time": float(best["sigma_time"]),
                "sigma_gap": float(best["sigma_gap"]),
                "gap_scale": float(best["gap_scale"]),
            }
    return params


def inventory_report(dirs: Dict[str, Path]) -> None:
    patterns = ["*trajectory*.csv", "*trial_level*.csv", "*readout*.csv", "*mechanism*.csv", "*.npz"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(BASE_DIR.glob(f"**/{pat}"))
    files = sorted(set(files))
    lines = [
        "# Gated readout input inventory",
        "",
        "This audit checks whether the representative subset contains trial-level time-series state data for gated readout.",
        "",
        "## Files found",
    ]
    for path in files:
        rel = path.relative_to(BASE_DIR)
        try:
            if path.suffix == ".csv":
                head = pd.read_csv(path, nrows=3)
                n_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
                cols = ", ".join(head.columns.astype(str).tolist())
                lines.append(f"- `{rel}`: csv, rows={n_rows}, columns={cols}")
            elif path.suffix == ".npz":
                z = np.load(path, allow_pickle=True)
                cols = ", ".join([f"{k}{z[k].shape}" for k in z.files])
                lines.append(f"- `{rel}`: npz, arrays={cols}")
        except Exception as exc:  # pragma: no cover - inventory should not block simulation.
            lines.append(f"- `{rel}`: could not inspect ({exc})")
    lines.extend(
        [
            "",
            "## Sufficiency assessment",
            "",
            "- Saved CSV trajectory files are group/condition summaries, not full per-trial four-channel trajectories.",
            "- `evidence_cache/representative_subset_layerwise_evidence.npz` plus the saved R5 parameter table are sufficient to reconstruct the full trial-level Wong-Wang trajectory without retraining VGG or refitting the main model.",
            "- The gated simulation below therefore uses reconstructed trial-level trajectories. It is a true delayed readout simulation over future time points, not the earlier single-readout counterfactual.",
        ]
    )
    (dirs["summaries"] / "gated_readout_input_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct() -> Tuple[pd.DataFrame, np.ndarray]:
    best_dir = BASE_DIR / "best_model_R5_combined_best/results"
    group_params, t0_mean, t0_sd = parse_group_params(best_dir / "best_model_parameter_estimates.csv")
    cache = load_trial_cache(BASE_DIR)
    all_df: List[pd.DataFrame] = []
    all_traj: List[np.ndarray] = []
    for group in GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = subset_cache(cache, mask)
        gp = group_params[group]
        ns = argparse.Namespace(device="cpu", time_steps=TIME_STEPS, dt_ms=int(DT * 1000), seed=SEED)
        cfg = ReadoutConfig(
            "sustained_crossing",
            min_decision_time=float(gp["min_decision_time"]),
            sustained_k=int(gp["sustained_k"]),
            margin=float(gp["margin"]),
        )
        df, out = run_base(
            gc,
            ns,
            model_name="R5_combined_best",
            evidence_gain=float(gp["evidence_gain"]),
            threshold=float(gp["threshold"]),
            cfg=cfg,
        )
        df = apply_group_t0(df, {group: t0_mean[group]}, {group: t0_sd[group]}, SEED)
        df["analysis_group"] = group
        df["congruency_label"] = normalize_congruency(df["congruency"])
        all_df.append(df)
        all_traj.append(np.asarray(out["trajectory"], dtype=np.float32))
    base = pd.concat(all_df, ignore_index=True)
    traj = np.concatenate(all_traj, axis=0)
    return base, traj


def states_at(traj: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return traj[np.arange(traj.shape[0]), np.clip(steps, 0, traj.shape[1] - 1), :]


def state_metrics(states: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> Dict[str, np.ndarray]:
    rows = np.arange(states.shape[0])
    s_target = states[rows, target]
    s_flanker = states[rows, flanker]
    masked = states.copy()
    masked[rows, target] = -np.inf
    s_other_max = masked.max(axis=1)
    order = np.argsort(-states, axis=1, kind="mergesort")
    rank = np.empty(states.shape[0], dtype=int)
    for i, t in enumerate(target):
        rank[i] = int(np.where(order[i] == t)[0][0] + 1)
    top2 = np.sort(states, axis=1)[:, -2:]
    return {
        "s_target": s_target,
        "s_flanker": s_flanker,
        "s_other_max": s_other_max,
        "target_rank": rank,
        "signed_target_margin": s_target - s_other_max,
        "gap": top2[:, 1] - top2[:, 0],
    }


def find_gate_steps(
    traj: np.ndarray,
    original_steps: np.ndarray,
    target: np.ndarray,
    max_delay: float,
    gate: str,
    margin_threshold: float = 0.0,
    soft_theta: float = 0.0,
    soft_temperature: float = 0.01,
    incongruent_only: np.ndarray | None = None,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, time_steps, _ = traj.shape
    max_extra = int(round(max_delay / DT))
    gated = original_steps.copy()
    satisfied = np.zeros(n, dtype=bool)
    forced = np.zeros(n, dtype=bool)
    rows = np.arange(n)
    rng = np.random.default_rng(NOISE_SEED + 1777 + seed_offset)
    active = np.ones(n, dtype=bool) if incongruent_only is None else incongruent_only.copy()
    for i in range(n):
        if not active[i]:
            satisfied[i] = True
            continue
        start = int(np.clip(original_steps[i], 0, time_steps - 1))
        end = int(np.clip(start + max_extra, start, time_steps - 1))
        local = traj[i, start : end + 1, :]
        target_vals = local[:, int(target[i])]
        masked = local.copy()
        masked[:, int(target[i])] = -np.inf
        margins = target_vals - masked.max(axis=1)
        if gate == "none":
            hit = 0
        elif gate == "rank":
            hit_idx = np.flatnonzero(margins >= 0.0)
            hit = int(hit_idx[0]) if hit_idx.size else None
        elif gate == "margin":
            hit_idx = np.flatnonzero(margins > margin_threshold)
            hit = int(hit_idx[0]) if hit_idx.size else None
        elif gate == "soft":
            p = 1.0 / (1.0 + np.exp(-np.clip((margins - soft_theta) / max(soft_temperature, 1e-9), -60, 60)))
            draw = rng.random(len(p))
            hit_idx = np.flatnonzero(draw < p)
            hit = int(hit_idx[0]) if hit_idx.size else None
        else:
            raise ValueError(gate)
        if hit is None:
            gated[i] = end
            forced[i] = True
        else:
            gated[i] = start + hit
            satisfied[i] = True
    return gated, satisfied, forced


def choice_type(choice: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    return np.where(choice == target, "target", np.where(choice == flanker, "flanker", "other"))


def simulate_choice(part: pd.DataFrame, states: np.ndarray, metrics: Dict[str, np.ndarray], params: Dict[str, float], seed_offset: int) -> np.ndarray:
    max_time = max(float(part["gated_readout_time"].max()), 1e-9)
    earlyness = 1.0 - part["gated_readout_time"].to_numpy(float) / max_time
    gap = np.clip(metrics["gap"], 0, None)
    sigma = (
        params["sigma_base"]
        + params["sigma_time"] * earlyness
        + params["sigma_gap"] * np.exp(-gap / max(params["gap_scale"], 1e-9))
    )
    rng = np.random.default_rng(NOISE_SEED + seed_offset)
    return (states + rng.normal(0.0, sigma[:, None], size=states.shape)).argmax(axis=1)


def trial_rows_for_config(
    base: pd.DataFrame,
    traj: np.ndarray,
    params_by_group: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    config_id: int,
) -> pd.DataFrame:
    target = base["target_label"].to_numpy(int)
    flanker = base["flanker_label"].to_numpy(int)
    original_steps = np.rint(base["decision_time"].to_numpy(float) / DT).astype(int)
    original_steps = np.clip(original_steps, 0, traj.shape[1] - 1)
    incong = base["congruency_label"].eq("incongruent").to_numpy()
    active = incong if config["incongruent_only"] else None
    gated_steps, satisfied, forced = find_gate_steps(
        traj,
        original_steps,
        target,
        max_delay=float(config["max_delay"]),
        gate=str(config["gate"]),
        margin_threshold=float(config["margin_threshold"]),
        soft_theta=float(config["soft_theta"]),
        soft_temperature=float(config["soft_temperature"]),
        incongruent_only=active,
        seed_offset=config_id * 1009,
    )
    original_states = states_at(traj, original_steps)
    gated_states = states_at(traj, gated_steps)
    om = state_metrics(original_states, target, flanker)
    gm = state_metrics(gated_states, target, flanker)
    out = pd.DataFrame(
        {
            "trial_id": base["row_index"].to_numpy(int),
            "analysis_group": base["analysis_group"].to_numpy(str),
            "congruency": base["congruency_label"].to_numpy(str),
            "target_label": target,
            "flanker_label": flanker,
            "human_correct": base["human_correct"].to_numpy(bool),
            "true_rt": base["true_rt"].to_numpy(float),
            "original_readout_time": original_steps * DT,
            "gated_readout_time": gated_steps * DT,
            "gating_delay": (gated_steps - original_steps) * DT,
            "gating_model": config["name"],
            "margin_threshold": config["margin_threshold"],
            "max_delay": config["max_delay"],
            "soft_theta": config["soft_theta"],
            "soft_temperature": config["soft_temperature"],
            "gate_satisfied": satisfied,
            "forced_readout": forced,
            "target_rank_original": om["target_rank"],
            "target_rank_gated": gm["target_rank"],
            "signed_target_margin_original": om["signed_target_margin"],
            "signed_target_margin_gated": gm["signed_target_margin"],
            "gap_original": om["gap"],
            "gap_gated": gm["gap"],
            "human_rt": base["true_rt"].to_numpy(float),
        }
    )
    t0_component = base["pred_rt"].to_numpy(float) - base["decision_time"].to_numpy(float)
    out["model_rt"] = np.maximum(out["gated_readout_time"].to_numpy(float) + t0_component, 0.05)
    choices = np.empty(len(out), dtype=int)
    for gi, group in enumerate(GROUPS):
        mask = out["analysis_group"].eq(group).to_numpy()
        p = params_by_group[group]
        choices[mask] = simulate_choice(out.loc[mask].copy(), gated_states[mask], {k: v[mask] for k, v in gm.items()}, p, config_id * 100000 + gi * 50000)
    out["model_choice"] = choices
    out["model_correct"] = choices == target
    out["model_choice_type"] = choice_type(choices, target, flanker)
    for prefix, correct_col, rt_col in [("model", "model_correct", "model_rt"), ("human", "human_correct", "human_rt")]:
        err = out[correct_col].eq(False)
        corr = out[correct_col].eq(True)
        group_mean_err = out.groupby(["analysis_group", "congruency"])[rt_col].transform(lambda s: s[out.loc[s.index, correct_col].eq(False)].mean())
        group_mean_cor = out.groupby(["analysis_group", "congruency"])[rt_col].transform(lambda s: s[out.loc[s.index, correct_col].eq(True)].mean())
        out[f"{prefix}_condition_error_rt_minus_correct_rt"] = group_mean_err - group_mean_cor
        out[f"{prefix}_is_error"] = err
        out[f"{prefix}_is_correct"] = corr
    return out


def configs() -> List[Dict[str, Any]]:
    out = [
        {
            "name": "original_time_gap_no_gating",
            "gate": "none",
            "margin_threshold": 0.0,
            "max_delay": 0.0,
            "soft_theta": 0.0,
            "soft_temperature": 0.0,
            "incongruent_only": False,
        }
    ]
    for md in MAX_DELAYS:
        out.append({"name": f"rank_gated_maxdelay_{md:.2f}", "gate": "rank", "margin_threshold": 0.0, "max_delay": md, "soft_theta": 0.0, "soft_temperature": 0.0, "incongruent_only": False})
        out.append({"name": f"rank_gated_incongruent_only_maxdelay_{md:.2f}", "gate": "rank", "margin_threshold": 0.0, "max_delay": md, "soft_theta": 0.0, "soft_temperature": 0.0, "incongruent_only": True})
        for mt in MARGIN_THRESHOLDS:
            out.append({"name": f"margin_gated_m{mt:.3f}_maxdelay_{md:.2f}", "gate": "margin", "margin_threshold": mt, "max_delay": md, "soft_theta": 0.0, "soft_temperature": 0.0, "incongruent_only": False})
            out.append({"name": f"margin_gated_incongruent_only_m{mt:.3f}_maxdelay_{md:.2f}", "gate": "margin", "margin_threshold": mt, "max_delay": md, "soft_theta": 0.0, "soft_temperature": 0.0, "incongruent_only": True})
        for th in SOFT_THETAS:
            for temp in SOFT_TEMPS:
                out.append({"name": f"soft_gated_theta{th:.3f}_temp{temp:.3f}_maxdelay_{md:.2f}", "gate": "soft", "margin_threshold": 0.0, "max_delay": md, "soft_theta": th, "soft_temperature": temp, "incongruent_only": False})
                out.append({"name": f"soft_gated_incongruent_only_theta{th:.3f}_temp{temp:.3f}_maxdelay_{md:.2f}", "gate": "soft", "margin_threshold": 0.0, "max_delay": md, "soft_theta": th, "soft_temperature": temp, "incongruent_only": True})
    return out


def rt_bin_error(df: pd.DataFrame, correct_col: str, rt_col: str, bins: int = 5) -> pd.DataFrame:
    rows = []
    for (group, cong), part in df.groupby(["analysis_group", "congruency"], sort=False):
        order = np.argsort(part[rt_col].to_numpy(float), kind="mergesort")
        for i, idx in enumerate(np.array_split(order, bins), start=1):
            sub = part.iloc[idx]
            rows.append({"analysis_group": group, "congruency": cong, "rt_bin": i, "error_rate": float((~sub[correct_col].to_numpy(bool)).mean()), "mean_rt": safe_mean(sub[rt_col])})
    return pd.DataFrame(rows)


def metrics_for(part: pd.DataFrame) -> Dict[str, float]:
    correct = part["model_correct"].to_numpy(bool)
    h_correct = part["human_correct"].to_numpy(bool)
    rt = part["model_rt"].to_numpy(float)
    hrt = part["human_rt"].to_numpy(float)
    cong = part["congruency"].eq("congruent").to_numpy()
    incong = part["congruency"].eq("incongruent").to_numpy()
    err = ~correct
    h_err = ~h_correct
    mb = rt_bin_error(part, "model_correct", "model_rt")
    hb = rt_bin_error(part, "human_correct", "human_rt")
    b = mb.merge(hb, on=["analysis_group", "congruency", "rt_bin"], suffixes=("_model", "_human"))
    probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    mq = [safe_q(rt, p) for p in probs]
    hq = [safe_q(hrt, p) for p in probs]
    choice_props = part["model_choice_type"].value_counts(normalize=True)
    return {
        "n_trials": int(len(part)),
        "overall_accuracy": safe_mean(correct.astype(float)),
        "congruent_error_rate": safe_mean(err[cong].astype(float)),
        "incongruent_error_rate": safe_mean(err[incong].astype(float)),
        "congruent_error_rt_minus_correct_rt": safe_mean(rt[cong & err]) - safe_mean(rt[cong & correct]),
        "incongruent_error_rt_minus_correct_rt": safe_mean(rt[incong & err]) - safe_mean(rt[incong & correct]),
        "overall_error_rt_minus_correct_rt": safe_mean(rt[err]) - safe_mean(rt[correct]),
        "error_rate_by_rt_bin_rmse": rmse(b["error_rate_model"], b["error_rate_human"]),
        "rt_distribution_similarity": corr_safe(mq, hq),
        "mean_gating_delay": safe_mean(part["gating_delay"]),
        "proportion_forced_readout": safe_mean(part["forced_readout"].astype(float)),
        "proportion_gate_satisfied": safe_mean(part["gate_satisfied"].astype(float)),
        "target_recovery_preservation": safe_mean(part["target_rank_gated"].eq(1).astype(float)) - safe_mean(part["target_rank_original"].eq(1).astype(float)),
        "choice_type_proportion_target": float(choice_props.get("target", 0.0)),
        "choice_type_proportion_flanker": float(choice_props.get("flanker", 0.0)),
        "choice_type_proportion_other": float(choice_props.get("other", 0.0)),
        "human_overall_accuracy": safe_mean(h_correct.astype(float)),
        "human_congruent_error_rate": safe_mean(h_err[cong].astype(float)),
        "human_incongruent_error_rate": safe_mean(h_err[incong].astype(float)),
        "human_congruent_error_rt_minus_correct_rt": safe_mean(hrt[cong & h_err]) - safe_mean(hrt[cong & h_correct]),
        "human_incongruent_error_rt_minus_correct_rt": safe_mean(hrt[incong & h_err]) - safe_mean(hrt[incong & h_correct]),
        "human_overall_error_rt_minus_correct_rt": safe_mean(hrt[h_err]) - safe_mean(hrt[h_correct]),
    }


def summarize(trials: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (name, group), part in trials.groupby(["gating_model", "analysis_group"], sort=False):
        d = metrics_for(part)
        d.update(
            {
                "gating_model": name,
                "analysis_group": group,
                "margin_threshold": part["margin_threshold"].iloc[0],
                "max_delay": part["max_delay"].iloc[0],
                "soft_theta": part["soft_theta"].iloc[0],
                "soft_temperature": part["soft_temperature"].iloc[0],
            }
        )
        for key in [
            "overall_accuracy",
            "congruent_error_rate",
            "incongruent_error_rate",
            "congruent_error_rt_minus_correct_rt",
            "incongruent_error_rt_minus_correct_rt",
            "overall_error_rt_minus_correct_rt",
        ]:
            d[f"deviation_{key}"] = d[key] - d[f"human_{key}"]
            d[f"abs_deviation_{key}"] = abs(d[f"deviation_{key}"])
        rows.append(d)
    summary = pd.DataFrame(rows)
    rank_rows = []
    for name, part in summary.groupby("gating_model", sort=False):
        score = (
            part["abs_deviation_overall_accuracy"].mean()
            + part["abs_deviation_congruent_error_rate"].mean()
            + part["abs_deviation_incongruent_error_rate"].mean()
            + 0.5 * part["error_rate_by_rt_bin_rmse"].mean()
            + 0.25 * (1.0 - part["rt_distribution_similarity"].fillna(0)).mean()
            + 0.5 * part["proportion_forced_readout"].mean()
            + 0.5 * part["mean_gating_delay"].mean()
            + 0.25 * np.maximum(0, part["congruent_error_rt_minus_correct_rt"]).mean()
        )
        row = {"gating_model": name, "ranking_score": float(score)}
        for _, r in part.iterrows():
            g = r["analysis_group"]
            for col in [
                "overall_accuracy",
                "congruent_error_rate",
                "incongruent_error_rate",
                "congruent_error_rt_minus_correct_rt",
                "incongruent_error_rt_minus_correct_rt",
                "mean_gating_delay",
                "proportion_forced_readout",
                "choice_type_proportion_flanker",
            ]:
                row[f"{g}_{col}"] = r[col]
        rank_rows.append(row)
    ranking = pd.DataFrame(rank_rows).sort_values("ranking_score", kind="mergesort")
    return summary, ranking


def bar_by_group(ax: plt.Axes, data: pd.DataFrame, metric: str, models: List[str], ylabel: str) -> None:
    x = np.arange(len(GROUPS))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        vals = [data[(data["gating_model"].eq(model)) & (data["analysis_group"].eq(g))][metric].mean() for g in GROUPS]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, label=model.replace("_", "\n"))
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUPS])
    ax.set_ylabel(ylabel)
    style_ax(ax)


def make_figures(trials: pd.DataFrame, summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    top_models = ["original_time_gap_no_gating"] + [m for m in ranking["gating_model"].head(3).tolist() if m != "original_time_gap_no_gating"]
    top_models = top_models[:4]
    human_rows = []
    for group, part in trials[trials["gating_model"].eq("original_time_gap_no_gating")].groupby("analysis_group"):
        h = metrics_for(part)
        human_rows.append({"gating_model": "human", "analysis_group": group, "overall_accuracy": h["human_overall_accuracy"], "congruent_error_rate": h["human_congruent_error_rate"], "incongruent_error_rate": h["human_incongruent_error_rate"]})
    human = pd.DataFrame(human_rows)
    plot_data = pd.concat([summary, human], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    view = ranking.head(12).iloc[::-1]
    ax.barh(np.arange(len(view)), view["ranking_score"], color="#4C78A8")
    ax.set_yticks(np.arange(len(view)))
    ax.set_yticklabels(view["gating_model"].str.replace("_", " "), fontsize=7)
    ax.set_xlabel("Lower is better")
    ax.set_title("Gated readout model ranking")
    style_ax(ax)
    save_fig(fig, "gated_readout_model_ranking_overview")

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_by_group(ax, plot_data, "overall_accuracy", ["human"] + top_models, "Accuracy")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, "human_vs_model_accuracy_by_condition")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bar_by_group(axes[0], plot_data, "congruent_error_rate", ["human"] + top_models, "Error rate")
    axes[0].set_title("Congruent")
    bar_by_group(axes[1], plot_data, "incongruent_error_rate", ["human"] + top_models, "Error rate")
    axes[1].set_title("Incongruent")
    axes[1].legend(frameon=False, fontsize=7)
    save_fig(fig, "human_vs_model_error_rate_by_condition")

    best = ranking.iloc[0]["gating_model"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        sub = trials[trials["analysis_group"].eq(group)]
        for model in ["original_time_gap_no_gating", best]:
            b = rt_bin_error(sub[sub["gating_model"].eq(model)], "model_correct", "model_rt")
            ax.plot(b.groupby("rt_bin")["error_rate"].mean().index, b.groupby("rt_bin")["error_rate"].mean().values, marker="o", label=model)
        hb = rt_bin_error(sub[sub["gating_model"].eq("original_time_gap_no_gating")], "human_correct", "human_rt")
        ax.plot(hb.groupby("rt_bin")["error_rate"].mean().index, hb.groupby("rt_bin")["error_rate"].mean().values, marker="o", color="black", label="human")
        ax.set_title(GROUP_LABEL[group])
        ax.set_xlabel("RT bin")
        style_ax(ax)
    axes[0].set_ylabel("Error rate")
    axes[1].legend(frameon=False, fontsize=7)
    save_fig(fig, "error_rate_by_rt_bin_human_vs_gated_model")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cong in zip(axes, ["congruent", "incongruent"]):
        data = [trials[(trials["gating_model"].eq(best)) & (trials["analysis_group"].eq(g)) & (trials["congruency"].eq(cong))]["gating_delay"].to_numpy(float) for g in GROUPS]
        ax.boxplot(data, tick_labels=[GROUP_LABEL[g] for g in GROUPS], showfliers=False)
        ax.set_title(cong.capitalize())
        ax.set_ylabel("Gating delay (s)")
        style_ax(ax)
    save_fig(fig, "gating_delay_distribution_by_condition")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    best_trials = trials[trials["gating_model"].eq(best)]
    for ax, col, title in [(axes[0], "target_rank_original", "Original"), (axes[1], "target_rank_gated", "Gated")]:
        counts = best_trials.groupby(["analysis_group", col]).size().reset_index(name="n")
        for group in GROUPS:
            part = counts[counts["analysis_group"].eq(group)]
            ax.plot(part[col], part["n"] / part["n"].sum(), marker="o", label=GROUP_LABEL[group])
        ax.set_title(title)
        ax.set_xlabel("Target evidence rank")
        ax.set_ylabel("Proportion")
        style_ax(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "target_rank_original_vs_gated")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(best_trials["signed_target_margin_original"], best_trials["signed_target_margin_gated"], s=4, alpha=0.15, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Original signed margin")
    ax.set_ylabel("Gated signed margin")
    style_ax(ax)
    save_fig(fig, "signed_target_margin_original_vs_gated")

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = trials[trials["gating_model"].isin(top_models)].groupby(["gating_model", "model_choice_type"]).size().rename("n").reset_index()
    counts["proportion"] = counts["n"] / counts.groupby("gating_model")["n"].transform("sum")
    props = counts[["gating_model", "model_choice_type", "proportion"]]
    pivot = props.pivot(index="gating_model", columns="model_choice_type", values="proportion").fillna(0)
    pivot[["target", "flanker", "other"]].plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518", "#9E9E9E"])
    ax.set_ylabel("Choice proportion")
    ax.set_xlabel("")
    ax.set_xticklabels([x.replace("_", "\n") for x in pivot.index], rotation=0, fontsize=7)
    style_ax(ax)
    save_fig(fig, "choice_type_proportion_target_flanker_other")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, GROUPS):
        sub = trials[trials["analysis_group"].eq(group)]
        bins = np.linspace(sub["human_rt"].quantile(0.01), sub["human_rt"].quantile(0.99), 40)
        ax.hist(sub[sub["gating_model"].eq("original_time_gap_no_gating")]["human_rt"], bins=bins, histtype="step", density=True, color="black", label="human")
        ax.hist(sub[sub["gating_model"].eq("original_time_gap_no_gating")]["model_rt"], bins=bins, histtype="step", density=True, label="original")
        ax.hist(sub[sub["gating_model"].eq(best)]["model_rt"], bins=bins, histtype="step", density=True, label="gated")
        ax.set_title(GROUP_LABEL[group])
        ax.set_xlabel("RT (s)")
        style_ax(ax)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, "RT_distribution_human_vs_gated_model")

    fig, ax = plt.subplots(figsize=(8, 4))
    fast = summary[summary["gating_model"].isin(top_models)].copy()
    ax.scatter(fast["congruent_error_rate"], fast["congruent_error_rt_minus_correct_rt"], c=fast["analysis_group"].map({"young_20_29": "#4C78A8", "older_80_89": "#F58518"}), s=45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Congruent error rate")
    ax.set_ylabel("Congruent error RT - correct RT")
    style_ax(ax)
    save_fig(fig, "congruent_fast_error_preservation")

    fig, ax = plt.subplots(figsize=(8, 4))
    inc = summary[summary["gating_model"].isin(top_models)]
    for group in GROUPS:
        part = inc[inc["analysis_group"].eq(group)]
        ax.plot(part["gating_model"].str.replace("_", "\n"), part["incongruent_error_rate"], marker="o", label=GROUP_LABEL[group])
    ax.set_ylabel("Incongruent error rate")
    ax.tick_params(axis="x", labelrotation=0, labelsize=7)
    ax.legend(frameon=False)
    style_ax(ax)
    save_fig(fig, "incongruent_error_reduction_comparison")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(summary["congruent_error_rate"], 1 - summary["incongruent_error_rate"], s=35, alpha=0.75, c=summary["mean_gating_delay"], cmap="viridis")
    ax.set_xlabel("Congruent error rate")
    ax.set_ylabel("Incongruent accuracy")
    style_ax(ax)
    save_fig(fig, "tradeoff_congruent_error_vs_incongruent_accuracy")


def write_summary(summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    best_overall_name = ranking.iloc[0]["gating_model"]
    gated_rank = ranking[~ranking["gating_model"].eq("original_time_gap_no_gating")].copy()
    best_gated_name = gated_rank.iloc[0]["gating_model"]
    low_incong = (
        summary[~summary["gating_model"].eq("original_time_gap_no_gating")]
        .groupby("gating_model", as_index=False)
        .agg(mean_incongruent_error_rate=("incongruent_error_rate", "mean"), mean_forced=("proportion_forced_readout", "mean"), mean_delay=("mean_gating_delay", "mean"))
        .sort_values("mean_incongruent_error_rate", kind="mergesort")
        .iloc[0]
    )
    best = summary[summary["gating_model"].eq(best_gated_name)]
    orig = summary[summary["gating_model"].eq("original_time_gap_no_gating")]
    lines = [
        "# Gated readout model summary",
        "",
        "## Data status",
        "",
        "- Full per-trial trajectory CSV files were not found.",
        "- Full per-trial trajectories were reconstructed from the saved evidence cache and saved R5 parameter table. No VGG or main WW model was retrained.",
        "- The resulting simulation performs real delayed readout over future time points.",
        "",
        "## Best model",
        "",
        f"- Best overall ranked model: `{best_overall_name}`.",
        f"- Best gated candidate by the balanced ranking: `{best_gated_name}`.",
        f"- Strongest incongruent-error reduction: `{low_incong['gating_model']}` with mean incongruent error rate {low_incong['mean_incongruent_error_rate']:.4f}, but mean forced readout {low_incong['mean_forced']:.4f} and mean delay {low_incong['mean_delay']:.4f}s.",
        "- The gate is interpreted as a response commitment-stage task-relevant evidence gate, not as a mechanism that knows the correct answer.",
        "- Time+gap choice uncertainty is retained after the gated readout point, so congruent fast errors are still explained by early/low-gap readout uncertainty.",
        "",
        "## Group metrics",
        "",
    ]
    cols = ["overall_accuracy", "congruent_error_rate", "incongruent_error_rate", "congruent_error_rt_minus_correct_rt", "mean_gating_delay", "proportion_forced_readout", "choice_type_proportion_flanker"]
    for group in GROUPS:
        b = best[best["analysis_group"].eq(group)].iloc[0]
        o = orig[orig["analysis_group"].eq(group)].iloc[0]
        lines.append(f"### {GROUP_LABEL[group]}")
        for col in cols:
            lines.append(f"- {col}: best={b[col]:.4f}; original={o[col]:.4f}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The tested gates did not fully fix the incongruent flanker-over-selection problem. The balanced score still favors the no-gating baseline because gated variants either leave incongruent errors high or introduce substantial forced readout/delay.",
            "- Rank and margin gates directly test whether delaying commitment until task-relevant evidence has recovered reduces premature flanker commitments.",
            "- Soft gates are more psychologically graded, but their stochastic commitment can preserve more early errors and may leave more incongruent failures.",
            "- Incongruent-only gates are diagnostic controls. They show how much of the problem is specifically due to incongruent premature commitment and should be treated as exploratory.",
            "- Results suitable for reporting are the reconstructed-trajectory gated simulations, the ranking, and the human/model condition comparisons. The exact best parameter should still be treated as exploratory until formal parameter fitting is added.",
            "- A formal next step would fit the commitment gate parameters jointly with the time+gap uncertainty parameters instead of selecting from this diagnostic grid.",
        ]
    )
    (OUT_DIR / "summaries/gated_readout_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dirs = ensure_dirs()
    inventory_report(dirs)
    params = selected_time_gap_params()
    base, traj = reconstruct()
    all_rows = []
    cfgs = configs()
    for i, cfg in enumerate(cfgs):
        all_rows.append(trial_rows_for_config(base, traj, params, cfg, i))
    trials = pd.concat(all_rows, ignore_index=True)
    trials.to_csv(dirs["metrics"] / "gated_readout_trial_level.csv", index=False)
    summary, ranking = summarize(trials)
    summary.to_csv(dirs["metrics"] / "gated_readout_model_comparison_summary.csv", index=False)
    ranking.to_csv(dirs["metrics"] / "gated_readout_model_ranking.csv", index=False)
    make_figures(trials, summary, ranking)
    write_summary(summary, ranking)
    (dirs["logs"] / "gated_readout_run_log.txt").write_text(
        json.dumps({"n_trials": int(len(trials)), "n_configs": len(cfgs), "best_model": str(ranking.iloc[0]["gating_model"])}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

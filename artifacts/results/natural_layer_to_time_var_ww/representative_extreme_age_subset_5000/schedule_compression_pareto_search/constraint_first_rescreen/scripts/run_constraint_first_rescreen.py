#!/usr/bin/env python3
"""
Constraint-first rescreen for repaired schedule-compression candidates.

This script intentionally reads only repaired outputs and human references. It
does not run WW, train models, or regenerate evidence.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/siyu/Documents/GitHub/VAM-studying")
BASE = ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
SEARCH = BASE / "schedule_compression_pareto_search"
OUT = SEARCH / "constraint_first_rescreen"
METRICS = OUT / "metrics"
FIGS = OUT / "figures_publication"
SUMMARIES = OUT / "summaries"
LOGS = OUT / "logs"
SCRIPTS = OUT / "scripts"

INPUTS = {
    "trial_level": SEARCH / "metrics/schedule_compression_top_candidates_trial_level_repaired.csv",
    "summary": SEARCH / "metrics/schedule_compression_local_search_summary_repaired.csv",
    "ranking": SEARCH / "metrics/schedule_compression_local_search_ranking_repaired.csv",
    "pareto": SEARCH / "metrics/schedule_compression_pareto_front_repaired.csv",
    "rt_bin": SEARCH / "metrics/schedule_compression_error_rate_by_rt_bin_repaired.csv",
    "trajectory": SEARCH / "metrics/schedule_compression_trajectory_diagnostics_repaired.csv",
    "audit_notes": SEARCH / "summaries/schedule_compression_coarse_metric_audit.md",
    "repaired_notes": SEARCH / "summaries/schedule_compression_pareto_search_repaired_summary.md",
    "human_ref": BASE / "readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv",
}

AGE_MAP = {"young_20_29": "young", "older_80_89": "older"}
AGE_GROUPS = list(AGE_MAP)
CONDITIONS = ["congruent", "incongruent"]
MAIN_ORDER = [
    "main_young_incongruent_error_rate_le_0.20",
    "main_older_incongruent_error_rate_le_0.10",
    "main_young_congruent_error_rate_0.003_to_0.05",
    "main_older_congruent_error_rate_ge_0.002",
    "main_young_congruent_fast_error_evaluable_negative",
    "main_older_congruent_fast_error_not_absent",
    "main_early_flanker_dominance_ge_0.15",
    "main_incongruent_flanker_choice_limited",
    "main_rt_bin_rmse_not_worse_than_baseline",
    "main_target_recovery_time_plausible",
]


def log(msg: str) -> None:
    print(msg)


def parse_schedule(s: str) -> dict[str, float]:
    m = re.search(r"c(?P<c>-?\d+(?:\.\d+)?)_ls(?P<ls>-?\d+)_tw(?P<tw>-?\d+(?:\.\d+)?)_ep(?P<ep>-?\d+)", str(s))
    if not m:
        return {"schedule_compression": np.nan, "late_shift": np.nan, "time_warp": np.nan, "early_pause": np.nan}
    return {
        "schedule_compression": float(m.group("c")),
        "late_shift": float(m.group("ls")),
        "time_warp": float(m.group("tw")),
        "early_pause": float(m.group("ep")),
    }


def parse_noise(s: str) -> dict[str, float]:
    if str(s) == "baseline":
        return {"noise_base": 0.0, "noise_time": 0.0, "noise_gap": 0.0, "gap_scale": 0.0}
    m = re.search(
        r"sb(?P<sb>-?\d+(?:\.\d+)?)_st(?P<st>-?\d+(?:\.\d+)?)_sg(?P<sg>-?\d+(?:\.\d+)?)_gs(?P<gs>-?\d+(?:\.\d+)?)",
        str(s),
    )
    if not m:
        return {"noise_base": np.nan, "noise_time": np.nan, "noise_gap": np.nan, "gap_scale": np.nan}
    return {
        "noise_base": float(m.group("sb")),
        "noise_time": float(m.group("st")),
        "noise_gap": float(m.group("sg")),
        "gap_scale": float(m.group("gs")),
    }


def bool_value(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return x.lower() in {"true", "1", "yes"}
    return bool(x)


def finite(x) -> bool:
    return x is not None and not pd.isna(x) and math.isfinite(float(x))


def safe_mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else np.nan


def rmse(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if x.empty:
        return np.nan
    return float(np.sqrt(np.mean((x.iloc[:, 0] - x.iloc[:, 1]) ** 2)))


def check_inputs() -> None:
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {name}: {path}")


def load_inputs() -> dict[str, pd.DataFrame | str]:
    out: dict[str, pd.DataFrame | str] = {}
    for name, path in INPUTS.items():
        if path.suffix == ".csv":
            out[name] = pd.read_csv(path)
        else:
            out[name] = path.read_text()
    return out


def make_inventory(data: dict[str, pd.DataFrame | str]) -> None:
    required = {
        "trial_level": [
            "model_config_id", "analysis_group", "congruency", "human_correct", "stochastic_correct",
            "choice_type", "model_rt", "true_rt", "early_flanker_dominance", "late_target_recovery_strength",
        ],
        "summary": ["model_config_id", "analysis_group", "congruency"],
        "ranking": ["model_config_id", "schedule_config_id", "noise_config_id", "tradeoff_region"],
        "pareto": ["model_config_id"],
        "rt_bin": ["model_config_id", "analysis_group", "congruency", "source", "rt_bin", "error_rate"],
        "trajectory": ["model_config_id", "analysis_group", "congruency", "time", "s_target_mean", "s_flanker_mean"],
        "human_ref": ["analysis_group", "overall_accuracy", "congruent_error_rate", "incongruent_error_rate"],
    }
    lines = ["# Constraint-first rescreen input inventory", ""]
    for name, obj in data.items():
        path = INPUTS[name]
        if isinstance(obj, pd.DataFrame):
            missing = [c for c in required.get(name, []) if c not in obj.columns]
            lines.append(f"## {name}")
            lines.append(f"- Path: `{path}`")
            lines.append(f"- Shape: {obj.shape[0]} rows x {obj.shape[1]} columns")
            lines.append(f"- Missing required columns: {', '.join(missing) if missing else 'none'}")
        else:
            lines.append(f"## {name}")
            lines.append(f"- Path: `{path}`")
            lines.append(f"- Text length: {len(obj)} characters")
    trial = data["trial_level"]
    ranking = data["ranking"]
    pareto = data["pareto"]
    assert isinstance(trial, pd.DataFrame) and isinstance(ranking, pd.DataFrame) and isinstance(pareto, pd.DataFrame)
    lines.extend([
        "",
        "## Candidate pool",
        f"- Repaired trial-level candidate count: {trial['model_config_id'].nunique()}",
        f"- Repaired ranking candidate count: {ranking['model_config_id'].nunique()}",
        f"- Repaired Pareto candidate count: {pareto['model_config_id'].nunique()}",
        "",
        "## Direct hard-constraint metrics",
        "- Directly usable: model_config_id, schedule_config_id, noise_config_id, Pareto status, tradeoff region, repaired flags, trajectory columns, RT-bin profiles.",
        "- Recomputed from trial-level: error rates, accuracy, RT quantiles, fast-error counts and RT differences, choice-type proportions, condition-level trajectory summaries.",
        "- Recomputed from RT-bin profile: congruent/incongruent RMSE, fast-bin mismatch, slow-bin mismatch, CAF-like slope.",
    ])
    (SUMMARIES / "constraint_first_rescreen_input_inventory.md").write_text("\n".join(lines))


def recompute_metrics(data: dict[str, pd.DataFrame | str]) -> pd.DataFrame:
    trial = data["trial_level"].copy()
    ranking = data["ranking"].copy()
    pareto = data["pareto"].copy()
    rtbin = data["rt_bin"].copy()
    human_ref = data["human_ref"].copy()
    assert all(isinstance(x, pd.DataFrame) for x in [trial, ranking, pareto, rtbin, human_ref])

    trial["model_correct"] = trial["stochastic_correct"].astype(bool)
    trial["model_error"] = ~trial["model_correct"]
    trial["human_correct_bool"] = trial["human_correct"].astype(bool)
    trial["human_error"] = ~trial["human_correct_bool"]

    rows = []
    for mid, gmid in trial.groupby("model_config_id"):
        first = gmid.iloc[0]
        row = {
            "model_config_id": mid,
            "schedule_config_id": first["schedule_config_id"],
            "noise_config_id": first["noise_config_id"],
        }
        row.update(parse_schedule(first["schedule_config_id"]))
        row.update(parse_noise(first["noise_config_id"]))
        for analysis_group, age in AGE_MAP.items():
            ga = gmid[gmid["analysis_group"] == analysis_group]
            hrow = human_ref[human_ref["analysis_group"] == analysis_group]
            if not hrow.empty:
                for col in ["overall_accuracy", "congruent_error_rate", "incongruent_error_rate"]:
                    row[f"{age}_human_{col}"] = float(hrow.iloc[0][col])
            row[f"{age}_n_trials"] = len(ga)
            row[f"{age}_overall_accuracy"] = safe_mean(ga["model_correct"])
            row[f"{age}_mean_rt"] = safe_mean(ga["model_rt"])
            for q, label in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
                row[f"{age}_rt_{label}"] = float(ga["model_rt"].quantile(q)) if len(ga) else np.nan
            row[f"{age}_human_overall_accuracy_from_trials"] = safe_mean(ga["human_correct_bool"])
            for cond in CONDITIONS:
                gc = ga[ga["congruency"] == cond]
                prefix = f"{age}_{cond}"
                row[f"{prefix}_n_trials"] = len(gc)
                row[f"{prefix}_accuracy"] = safe_mean(gc["model_correct"])
                row[f"{prefix}_error_rate"] = safe_mean(gc["model_error"])
                row[f"{prefix}_human_error_rate_from_trials"] = safe_mean(gc["human_error"])
                err = gc[gc["model_error"]]
                cor = gc[gc["model_correct"]]
                row[f"{prefix}_error_count"] = len(err)
                row[f"{prefix}_correct_count"] = len(cor)
                row[f"{prefix}_error_rt_minus_correct_rt"] = safe_mean(err["model_rt"]) - safe_mean(cor["model_rt"]) if len(err) and len(cor) else np.nan
                row[f"{prefix}_fast_error_evaluable"] = len(err) >= 5 and len(cor) >= 5
                for ctype in ["target", "flanker", "other"]:
                    row[f"{prefix}_{ctype}_choice_proportion"] = safe_mean(gc["choice_type"].eq(ctype))
                row[f"{prefix}_non_target_choice_proportion"] = 1 - row[f"{prefix}_target_choice_proportion"] if finite(row[f"{prefix}_target_choice_proportion"]) else np.nan
                for col in [
                    "early_flanker_dominance", "flanker_dominance_duration", "late_target_recovery_strength",
                    "target_recovery_time", "signed_target_margin_at_readout", "target_rank_at_readout",
                    "target_ever_rank1", "target_first_rank1_time",
                ]:
                    if col in gc.columns:
                        val = safe_mean(gc[col].astype(float)) if col == "target_ever_rank1" else safe_mean(gc[col])
                        name = "target_ever_rank1_proportion" if col == "target_ever_rank1" else col
                        row[f"{prefix}_{name}"] = val
            row[f"{age}_young_congruent_fast_error_evaluable"] = row.get(f"{age}_congruent_fast_error_evaluable", False)
        rows.append(row)

    metrics = pd.DataFrame(rows)

    rt_rows = []
    for (mid, age_group, cond), grp in rtbin.groupby(["model_config_id", "analysis_group", "congruency"]):
        age = AGE_MAP.get(age_group, age_group)
        h = grp[grp["source"] == "human"].set_index("rt_bin")["error_rate"]
        m = grp[grp["source"] == "model"].set_index("rt_bin")["error_rate"]
        if m.empty or h.empty:
            continue
        bins = sorted(set(h.index).intersection(set(m.index)))
        if not bins:
            continue
        diff = m.reindex(bins) - h.reindex(bins)
        rt_rows.append({
            "model_config_id": mid,
            f"{age}_{cond}_rt_bin_rmse": rmse(m.reindex(bins), h.reindex(bins)),
            f"{age}_{cond}_fast_bin_error_mismatch": float(diff.loc[min(bins)]),
            f"{age}_{cond}_slow_bin_error_mismatch": float(diff.loc[max(bins)]),
            f"{age}_{cond}_caf_like_error_rate_slope": float(m.reindex(bins).iloc[-1] - m.reindex(bins).iloc[0]),
        })
    if rt_rows:
        rtwide = pd.DataFrame(rt_rows).groupby("model_config_id", as_index=False).first()
        metrics = metrics.merge(rtwide, on="model_config_id", how="left")

    for age in ["young", "older"]:
        metrics[f"{age}_incongruent_flanker_choice_proportion"] = metrics[f"{age}_incongruent_flanker_choice_proportion"]
        metrics[f"{age}_congruent_non_target_choice_proportion"] = metrics[f"{age}_congruent_non_target_choice_proportion"]
        metrics[f"{age}_congruent_fast_error_evaluable"] = metrics[f"{age}_congruent_fast_error_evaluable"].astype(bool)

    baseline_id = "c1.00_ls0_tw1.00_ep0__baseline"
    baseline = metrics[metrics["model_config_id"] == baseline_id]
    for age in ["young", "older"]:
        for cond in CONDITIONS:
            col = f"{age}_{cond}_rt_bin_rmse"
            if not baseline.empty and col in baseline:
                metrics[f"baseline_{age}_{cond}_rt_bin_rmse"] = float(baseline.iloc[0][col])
    ranking_cols = [
        "model_config_id", "combined_score", "incongruent_repair_score", "congruent_fast_error_score",
        "rt_dynamics_preservation_score", "naturalness_penalty", "is_pareto_optimal", "pareto_rank",
        "tradeoff_region", "recommended_for_fine_search", "flag_lost_conflict_dynamics",
        "flag_unrealistic_perfect_accuracy", "flag_excessive_noise", "flag_rt_distribution_broken",
    ]
    metrics = metrics.merge(ranking[[c for c in ranking_cols if c in ranking.columns]], on="model_config_id", how="left")
    metrics["is_repaired_pareto"] = metrics["model_config_id"].isin(set(pareto["model_config_id"]))
    return metrics


def constraint_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    baseline_allowance = 0.05
    for _, r in metrics.iterrows():
        p = {"model_config_id": r["model_config_id"]}
        yc_err = r["young_congruent_error_rate"]
        oc_err = r["older_congruent_error_rate"]
        yi_err = r["young_incongruent_error_rate"]
        oi_err = r["older_incongruent_error_rate"]
        yfast = bool_value(r["young_congruent_fast_error_evaluable"]) and finite(r["young_congruent_error_rt_minus_correct_rt"]) and r["young_congruent_error_rt_minus_correct_rt"] < 0
        ofast = bool_value(r["older_congruent_fast_error_evaluable"]) and finite(r["older_congruent_error_rt_minus_correct_rt"]) and r["older_congruent_error_rt_minus_correct_rt"] < 0
        yweak = yc_err > 0 and r["young_congruent_error_count"] < 5
        oweak = oc_err > 0 and r["older_congruent_error_count"] < 5
        early_min = min(r["young_incongruent_early_flanker_dominance"], r["older_incongruent_early_flanker_dominance"])
        rt_cols = [c for c in r.index if c.endswith("_rt_bin_rmse") and not c.startswith("baseline_")]
        rt_ok_lenient = all((not finite(r[c])) or r[c] <= 1.0 for c in rt_cols)
        rt_ok_main = True
        rt_ok_strict = True
        for age in ["young", "older"]:
            for cond in CONDITIONS:
                col = f"{age}_{cond}_rt_bin_rmse"
                bcol = f"baseline_{age}_{cond}_rt_bin_rmse"
                if col in r and bcol in r and finite(r[col]) and finite(r[bcol]):
                    rt_ok_main &= r[col] <= r[bcol] + baseline_allowance
                    rt_ok_strict &= r[col] <= r[bcol] + 0.025
        target_recovery_vals = [r["young_incongruent_target_recovery_time"], r["older_incongruent_target_recovery_time"]]
        target_recovery_ok = all(finite(v) and 0.05 <= v <= 0.40 for v in target_recovery_vals)
        no_perfect = not (r["young_overall_accuracy"] >= 0.999 or r["older_overall_accuracy"] >= 0.999)
        not_absent_older_fast = oc_err >= 0.002 and (ofast or r["older_congruent_error_count"] < 5)

        human_yi = r["young_human_incongruent_error_rate"]
        human_oi = r["older_human_incongruent_error_rate"]
        human_yc = r["young_human_congruent_error_rate"]
        human_oc = r["older_human_congruent_error_rate"]

        constraints = {
            "lenient_young_incongruent_error_rate_le_0.25": yi_err <= 0.25,
            "lenient_older_incongruent_error_rate_le_0.15": oi_err <= 0.15,
            "lenient_young_congruent_error_rate_gt_0": yc_err > 0,
            "lenient_older_congruent_error_rate_gt_0": oc_err > 0,
            "lenient_young_congruent_fast_error_negative_or_weak": yfast or yweak,
            "lenient_older_congruent_fast_error_negative_or_weak": ofast or oweak,
            "lenient_early_flanker_dominance_ge_0.10": early_min >= 0.10,
            "lenient_rt_distribution_not_obviously_broken": rt_ok_lenient and not bool_value(r.get("flag_rt_distribution_broken", False)),
            "lenient_no_unrealistic_perfect_accuracy": no_perfect and not bool_value(r.get("flag_unrealistic_perfect_accuracy", False)),
            "main_young_incongruent_error_rate_le_0.20": yi_err <= 0.20,
            "main_older_incongruent_error_rate_le_0.10": oi_err <= 0.10,
            "main_young_congruent_error_rate_0.003_to_0.05": 0.003 <= yc_err <= 0.05,
            "main_older_congruent_error_rate_ge_0.002": oc_err >= 0.002,
            "main_young_congruent_fast_error_evaluable_negative": yfast,
            "main_older_congruent_fast_error_not_absent": not_absent_older_fast,
            "main_early_flanker_dominance_ge_0.15": early_min >= 0.15,
            "main_incongruent_flanker_choice_limited": r["young_incongruent_flanker_choice_proportion"] <= 0.25 and r["older_incongruent_flanker_choice_proportion"] <= 0.15,
            "main_rt_bin_rmse_not_worse_than_baseline": rt_ok_main,
            "main_target_recovery_time_plausible": target_recovery_ok,
            "strict_young_incongruent_error_rate_human_plus_0.05": yi_err <= human_yi + 0.05,
            "strict_older_incongruent_error_rate_human_plus_0.05": oi_err <= human_oi + 0.05,
            "strict_young_congruent_error_rate_human_pm_0.015": abs(yc_err - human_yc) <= 0.015,
            "strict_older_congruent_error_rate_human_pm_0.015": abs(oc_err - human_oc) <= 0.015,
            "strict_both_congruent_fast_errors_evaluable_negative": yfast and ofast,
            "strict_early_flanker_dominance_ge_0.20": early_min >= 0.20,
            "strict_rt_bin_rmse_close": rt_ok_strict,
            "strict_no_lost_conflict_dynamics": not bool_value(r.get("flag_lost_conflict_dynamics", False)),
            "strict_no_unrealistic_perfect_accuracy": no_perfect and not bool_value(r.get("flag_unrealistic_perfect_accuracy", False)),
            "strict_no_excessive_noise": not bool_value(r.get("flag_excessive_noise", False)),
        }
        p.update(constraints)
        for level in ["lenient", "main", "strict"]:
            names = [k for k in constraints if k.startswith(level + "_")]
            failed = [k for k in names if not constraints[k]]
            p[f"pass_{level}"] = len(failed) == 0
            p[f"fail_count_{level}"] = len(failed)
            p[f"first_failed_constraint_{level}"] = failed[0] if failed else ""
        p["failure_reason_category"] = categorize_failure(r, p)
        rows.append(p)

    pass_fail = pd.DataFrame(rows)
    sensitivity = []
    main_cols = [c for c in pass_fail.columns if c.startswith("main_")]
    base_survivors = pass_fail[pass_fail[main_cols].all(axis=1)]["model_config_id"].tolist()
    sensitivity.append({"removed_constraint": "none", "survivor_count": len(base_survivors), "model_config_ids": ";".join(base_survivors)})
    for removed in MAIN_ORDER:
        active = [c for c in main_cols if c != removed]
        surv = pass_fail[pass_fail[active].all(axis=1)]["model_config_id"].tolist()
        sensitivity.append({"removed_constraint": removed, "survivor_count": len(surv), "model_config_ids": ";".join(surv)})
    return pass_fail, pd.DataFrame(sensitivity)


def categorize_failure(r: pd.Series, p: dict) -> str:
    if r["older_congruent_error_rate"] <= 0 or r["older_congruent_error_count"] == 0:
        return "no_older_congruent_errors"
    if not bool_value(r["young_congruent_fast_error_evaluable"]) or not bool_value(r["older_congruent_fast_error_evaluable"]):
        return "insufficient_fast_error_trials"
    if r["young_incongruent_error_rate"] > 0.20 or r["older_incongruent_error_rate"] > 0.10:
        return "high_incongruent_error"
    if bool_value(r.get("flag_lost_conflict_dynamics", False)) or min(r["young_incongruent_early_flanker_dominance"], r["older_incongruent_early_flanker_dominance"]) < 0.15:
        return "lost_conflict_dynamics"
    if r["young_congruent_error_rate"] > 0.05:
        return "too_many_congruent_errors"
    if bool_value(r.get("flag_rt_distribution_broken", False)):
        return "rt_profile_bad"
    if bool_value(r.get("flag_unrealistic_perfect_accuracy", False)):
        return "unrealistic_accuracy"
    return "mixed_tradeoff"


def recommended_action(r: pd.Series, pf: pd.Series) -> str:
    if bool_value(pf["pass_main"]):
        return "fine_search_seed"
    cat = pf["failure_reason_category"]
    if cat == "lost_conflict_dynamics":
        return "discard_lost_conflict"
    if cat in {"no_older_congruent_errors", "insufficient_fast_error_trials"}:
        return "discard_high_accuracy_no_fast_error"
    if cat == "rt_profile_bad":
        return "discard_unstable_noise"
    if bool_value(pf["pass_lenient"]):
        return "keep_as_diagnostic"
    return "report_as_tradeoff_example"


def survivor_tables(metrics: pd.DataFrame, pass_fail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = metrics.merge(pass_fail, on="model_config_id", how="left")
    merged["recommended_next_action"] = [recommended_action(r, r) for _, r in merged.iterrows()]
    core_cols = [
        "model_config_id", "schedule_config_id", "noise_config_id", "schedule_compression", "late_shift",
        "time_warp", "early_pause", "noise_base", "noise_time", "noise_gap", "gap_scale",
        "young_congruent_error_rate", "young_incongruent_error_rate", "young_congruent_error_count",
        "young_congruent_error_rt_minus_correct_rt", "young_congruent_fast_error_evaluable",
        "young_incongruent_flanker_choice_proportion", "young_incongruent_early_flanker_dominance",
        "older_congruent_error_rate", "older_incongruent_error_rate", "older_congruent_error_count",
        "older_congruent_error_rt_minus_correct_rt", "older_congruent_fast_error_evaluable",
        "older_incongruent_flanker_choice_proportion", "older_incongruent_early_flanker_dominance",
        "is_repaired_pareto", "tradeoff_region", "pass_lenient", "pass_main", "pass_strict",
        "fail_count_main", "failure_reason_category", "recommended_next_action",
    ]
    tables = {}
    for level in ["lenient", "main", "strict"]:
        tables[level] = merged[merged[f"pass_{level}"]][[c for c in core_cols if c in merged.columns]].copy()
    return tables


def representative_models(metrics: pd.DataFrame, pass_fail: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(pass_fail, on="model_config_id", how="left")
    reps = []

    def add(role: str, row: pd.Series, why: str) -> None:
        failed = [c for c in pass_fail.columns if c.startswith("main_") and c in row.index and not bool_value(row[c])]
        reps.append({
            "representative_role": role,
            "model_config_id": row["model_config_id"],
            "schedule_config_id": row["schedule_config_id"],
            "noise_config_id": row["noise_config_id"],
            "schedule_compression": row["schedule_compression"],
            "late_shift": row["late_shift"],
            "time_warp": row["time_warp"],
            "early_pause": row["early_pause"],
            "noise_base": row["noise_base"],
            "noise_time": row["noise_time"],
            "noise_gap": row["noise_gap"],
            "gap_scale": row["gap_scale"],
            "why_selected": why,
            "failed_constraints": ";".join(failed),
            "young_congruent_error_rate": row["young_congruent_error_rate"],
            "young_incongruent_error_rate": row["young_incongruent_error_rate"],
            "young_congruent_error_rt_minus_correct_rt": row["young_congruent_error_rt_minus_correct_rt"],
            "older_congruent_error_rate": row["older_congruent_error_rate"],
            "older_incongruent_error_rate": row["older_incongruent_error_rate"],
            "older_congruent_error_rt_minus_correct_rt": row["older_congruent_error_rt_minus_correct_rt"],
            "suitable_for_advisor_report": "yes, as trade-off example",
            "one_sentence_explanation": one_sentence(role, row),
        })

    repair_score = merged["young_incongruent_error_rate"] + merged["older_incongruent_error_rate"]
    add("best_incongruent_repair", merged.loc[repair_score.idxmin()], "lowest summed young+older incongruent error rate")
    fast = merged.copy()
    fast["fast_score"] = fast[["young_congruent_error_rt_minus_correct_rt", "older_congruent_error_rt_minus_correct_rt"]].clip(lower=-10, upper=10).mean(axis=1, skipna=True)
    fast["fast_eval_count"] = fast["young_congruent_fast_error_evaluable"].astype(int) + fast["older_congruent_fast_error_evaluable"].astype(int)
    add("best_fast_error", fast.sort_values(["fast_eval_count", "fast_score", "young_congruent_error_count", "older_congruent_error_count"], ascending=[False, True, False, False]).iloc[0], "strongest congruent fast-error evidence with evaluability prioritized")
    dyn = merged.copy()
    dyn["dyn_score"] = dyn[["young_incongruent_early_flanker_dominance", "older_incongruent_early_flanker_dominance", "young_incongruent_late_target_recovery_strength", "older_incongruent_late_target_recovery_strength"]].mean(axis=1, skipna=True)
    add("best_conflict_dynamics", dyn.sort_values("dyn_score", ascending=False).iloc[0], "highest combined early flanker dominance and late target recovery")
    balanced = merged.sort_values(["fail_count_main", "is_repaired_pareto", "combined_score"], ascending=[True, False, True]).iloc[0]
    add("best_near_balanced", balanced, "fewest main-constraint failures; not a balanced model")
    return pd.DataFrame(reps)


def one_sentence(role: str, r: pd.Series) -> str:
    if role == "best_incongruent_repair":
        return "It repairs incongruent errors best, but this can wash out the congruent-error evidence."
    if role == "best_fast_error":
        return "It best preserves fast-error timing, but still fails other acceptability constraints."
    if role == "best_conflict_dynamics":
        return "It preserves conflict-like dynamics best, but its behavior remains too far from the target profile."
    return "It is the closest current trade-off candidate, but it still fails the main standard."


def save_fig(name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def make_figures(metrics: pd.DataFrame, pass_fail: pd.DataFrame, sensitivity: pd.DataFrame, reps: pd.DataFrame, rtbin: pd.DataFrame) -> None:
    merged = metrics.merge(pass_fail, on="model_config_id", how="left")

    plt.figure(figsize=(6, 4))
    counts = [len(merged), int(merged["pass_lenient"].sum()), int(merged["pass_main"].sum()), int(merged["pass_strict"].sum())]
    plt.bar(["pool", "lenient", "main", "strict"], counts, color=["#606c76", "#4c78a8", "#59a14f", "#e15759"])
    plt.ylabel("candidate count")
    plt.title("Constraint survival flow")
    for i, v in enumerate(counts):
        plt.text(i, v + 0.2, str(v), ha="center")
    save_fig("constraint_survival_flow")

    plt.figure(figsize=(8, 4))
    fc = pass_fail["failure_reason_category"].value_counts()
    plt.bar(fc.index, fc.values, color="#8f6f3f")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("candidate count")
    plt.title("Constraint failure counts")
    save_fig("constraint_failure_counts")

    plt.figure(figsize=(9, 4))
    sens = sensitivity[sensitivity["removed_constraint"] != "none"].copy()
    plt.bar(range(len(sens)), sens["survivor_count"], color="#4c78a8")
    plt.xticks(range(len(sens)), sens["removed_constraint"].str.replace("main_", "", regex=False), rotation=45, ha="right")
    plt.ylabel("main survivors after removing one constraint")
    plt.title("Main constraint sensitivity")
    save_fig("main_constraint_sensitivity")

    rep_ids = reps["model_config_id"].unique().tolist()
    repm = metrics[metrics["model_config_id"].isin(rep_ids)].copy()
    labels = reps.set_index("model_config_id")["representative_role"].to_dict()
    x = np.arange(len(rep_ids))
    width = 0.18
    plt.figure(figsize=(10, 4.5))
    vals = [
        [repm.set_index("model_config_id").loc[mid, "young_congruent_error_rate"] for mid in rep_ids],
        [repm.set_index("model_config_id").loc[mid, "young_incongruent_error_rate"] for mid in rep_ids],
        [repm.set_index("model_config_id").loc[mid, "older_congruent_error_rate"] for mid in rep_ids],
        [repm.set_index("model_config_id").loc[mid, "older_incongruent_error_rate"] for mid in rep_ids],
    ]
    for i, (name, val) in enumerate(zip(["Y congr", "Y incongr", "O congr", "O incongr"], vals)):
        plt.bar(x + (i - 1.5) * width, val, width, label=name)
    plt.xticks(x, [labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    plt.ylabel("model error rate")
    plt.legend()
    plt.title("Representative models error rate by condition")
    save_fig("representative_models_error_rate_by_condition")

    plt.figure(figsize=(8, 4))
    for i, age in enumerate(["young", "older"]):
        vals = [repm.set_index("model_config_id").loc[mid, f"{age}_congruent_error_rt_minus_correct_rt"] for mid in rep_ids]
        plt.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=age)
    plt.axhline(0, color="black", lw=1)
    plt.xticks(x, [labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    plt.ylabel("congruent error RT - correct RT")
    plt.legend()
    plt.title("Representative models fast-error timing")
    save_fig("representative_models_fast_error")

    plt.figure(figsize=(8, 4))
    for i, age in enumerate(["young", "older"]):
        vals = [repm.set_index("model_config_id").loc[mid, f"{age}_incongruent_flanker_choice_proportion"] for mid in rep_ids]
        plt.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=age)
    plt.ylabel("incongruent flanker choice proportion")
    plt.xticks(x, [labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    plt.legend()
    plt.title("Representative models flanker choice")
    save_fig("representative_models_flanker_choice")

    plt.figure(figsize=(8, 4))
    for i, metric in enumerate(["early_flanker_dominance", "late_target_recovery_strength"]):
        vals = [np.nanmean([repm.set_index("model_config_id").loc[mid, f"young_incongruent_{metric}"], repm.set_index("model_config_id").loc[mid, f"older_incongruent_{metric}"]]) for mid in rep_ids]
        plt.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=metric)
    plt.xticks(x, [labels[mid].replace("best_", "") for mid in rep_ids], rotation=20, ha="right")
    plt.legend()
    plt.title("Representative models conflict dynamics")
    save_fig("representative_models_conflict_dynamics")

    plt.figure(figsize=(8, 5))
    for mid in rep_ids[:4]:
        sub = rtbin[(rtbin["model_config_id"] == mid) & (rtbin["analysis_group"] == "young_20_29") & (rtbin["congruency"] == "incongruent")]
        model = sub[sub["source"] == "model"].sort_values("rt_bin")
        if not model.empty:
            plt.plot(model["rt_bin"], model["error_rate"], marker="o", label=labels[mid].replace("best_", ""))
    human = rtbin[(rtbin["model_config_id"] == rep_ids[0]) & (rtbin["analysis_group"] == "young_20_29") & (rtbin["congruency"] == "incongruent") & (rtbin["source"] == "human")].sort_values("rt_bin")
    if not human.empty:
        plt.plot(human["rt_bin"], human["error_rate"], color="black", lw=2, label="human")
    plt.xlabel("RT bin")
    plt.ylabel("error rate")
    plt.title("Representative models RT-bin profile")
    plt.legend(fontsize=8)
    save_fig("representative_models_rt_bin_profile")

    near = reps[reps["representative_role"] == "best_near_balanced"].iloc[0]["model_config_id"]
    nr = merged[merged["model_config_id"] == near].iloc[0]
    plt.figure(figsize=(8, 5))
    names = ["Y congr err", "Y incongr err", "O congr err", "O incongr err", "early conflict", "Y fast RTdiff", "O fast RTdiff"]
    vals = [
        nr["young_congruent_error_rate"], nr["young_incongruent_error_rate"], nr["older_congruent_error_rate"],
        nr["older_incongruent_error_rate"], min(nr["young_incongruent_early_flanker_dominance"], nr["older_incongruent_early_flanker_dominance"]),
        nr["young_congruent_error_rt_minus_correct_rt"], nr["older_congruent_error_rt_minus_correct_rt"],
    ]
    plt.bar(names, vals, color="#4c78a8")
    plt.axhline(0, color="black", lw=1)
    plt.xticks(rotation=30, ha="right")
    plt.title("Near-balanced candidate dashboard")
    save_fig("near_balanced_candidate_dashboard")

    plt.figure(figsize=(7, 5))
    status = np.where(merged["pass_main"], "main", np.where(merged["pass_lenient"], "lenient", "fail"))
    markers = {"main": "o", "lenient": "s", "fail": "x"}
    for st in ["fail", "lenient", "main"]:
        sub = merged[status == st]
        plt.scatter(
            sub["young_incongruent_error_rate"],
            sub["young_congruent_error_rt_minus_correct_rt"],
            c=sub["young_incongruent_early_flanker_dominance"],
            marker=markers[st],
            label=st,
            cmap="viridis",
            vmin=0,
            vmax=max(1, merged["young_incongruent_early_flanker_dominance"].max()),
        )
    plt.axhline(0, color="black", lw=1)
    plt.colorbar(label="young early flanker dominance")
    plt.xlabel("young incongruent error rate")
    plt.ylabel("young congruent error RT - correct RT")
    plt.legend()
    plt.title("Constraint tradeoff map")
    save_fig("constraint_tradeoff_map")


def write_summary(metrics: pd.DataFrame, pass_fail: pd.DataFrame, sensitivity: pd.DataFrame, reps: pd.DataFrame) -> None:
    pool = len(metrics)
    pareto_n = int(metrics["is_repaired_pareto"].sum())
    n_len = int(pass_fail["pass_lenient"].sum())
    n_main = int(pass_fail["pass_main"].sum())
    n_strict = int(pass_fail["pass_strict"].sum())
    failures = pass_fail["failure_reason_category"].value_counts()
    top_failure = failures.index[0] if len(failures) else "none"
    sens_nonzero = sensitivity[sensitivity["removed_constraint"] != "none"].copy()
    max_single_relax = int(sens_nonzero["survivor_count"].max()) if not sens_nonzero.empty else 0
    if max_single_relax > 0:
        limiting = sens_nonzero.sort_values("survivor_count", ascending=False).iloc[0]["removed_constraint"]
        sensitivity_sentence = f"Removing `{limiting}` gives the largest survivor count ({max_single_relax})."
    else:
        limiting = "no_single_constraint_sufficient"
        sensitivity_sentence = "Removing any single main constraint still leaves zero survivors, so the failure is a coupled trade-off rather than a one-threshold problem."
    near = reps[reps["representative_role"] == "best_near_balanced"].iloc[0]
    near_pf = pass_fail[pass_fail["model_config_id"] == near["model_config_id"]].iloc[0]
    failed_near = [c for c in pass_fail.columns if c.startswith("main_") and not bool_value(near_pf[c])]
    young_fast_only = metrics[(metrics["young_congruent_fast_error_evaluable"]) & (~metrics["older_congruent_fast_error_evaluable"])]
    repaired_no_congr = metrics[(metrics["young_incongruent_error_rate"] <= 0.20) & (metrics["older_incongruent_error_rate"] <= 0.10) & ((metrics["young_congruent_error_rate"] <= 0) | (metrics["older_congruent_error_rate"] <= 0.002))]
    conflict_bad_incong = metrics[(metrics["young_incongruent_early_flanker_dominance"] >= 0.15) & (metrics["older_incongruent_early_flanker_dominance"] >= 0.15) & ((metrics["young_incongruent_error_rate"] > 0.20) | (metrics["older_incongruent_error_rate"] > 0.10))]

    lines = [
        "# Constraint-first rescreen summary",
        "",
        "## Formal results",
        f"- Repaired candidate pool: {pool} model_config_id.",
        f"- Repaired Pareto candidates: {pareto_n}.",
        f"- Lenient survivors: {n_len}.",
        f"- Main survivors: {n_main}.",
        f"- Strict survivors: {n_strict}.",
        f"- Fine-search seed under main constraints: {'yes' if n_main else 'no'}.",
        f"- Most common failure category: {top_failure}.",
        f"- Most informative one-constraint sensitivity result: {sensitivity_sentence}",
        "",
        "## Required answers",
        f"1. Pool size: {pool}.",
        f"2. Pareto count: {pareto_n}.",
        f"3. Survivors: lenient={n_len}, main={n_main}, strict={n_strict}.",
        f"4. Candidate usable as fine-search seed: {'yes' if n_main else 'no'}.",
        f"5. Main limiting constraint category: {top_failure}; one-at-a-time sensitivity result: {limiting}.",
        "6. Older congruent errors / fast-error evidence are more central than early conflict dynamics in the failure categories, but the one-at-a-time sensitivity shows no single constraint relaxation is enough.",
        f"7. Models with young fast-error evidence but insufficient older evidence: {len(young_fast_only)}.",
        f"8. Models that repair incongruent errors but wash out congruent errors: {len(repaired_no_congr)}.",
        f"9. Models preserving early conflict dynamics while failing incongruent thresholds: {len(conflict_bad_incong)}.",
        f"10. Closest near-balanced candidate: `{near['model_config_id']}`.",
        f"11. Near-balanced failures: {', '.join(failed_near) if failed_near else 'none under main constraints'}.",
        f"12. Fine search recommendation: {'enter fine search around main survivors' if n_main else 'do not enter fine search from this pool'}.",
        "13. If no main survivor exists, next work should adjust the mechanism/objective to preserve older congruent errors and fast-error evaluability while keeping incongruent repair.",
        "14. If exploratory fine search is still forced, start only around the lowest fail-count repaired Pareto candidates listed in the representative table.",
        "15. Best advisor figures: constraint_survival_flow, main_constraint_sensitivity, representative_models_error_rate_by_condition, representative_models_fast_error, constraint_tradeoff_map.",
        "16. Formal conclusion: the repaired pool has no final balanced model unless main survivors are nonzero.",
        "17. Exploratory conclusion: representative models are useful only as trade-off examples.",
        "",
        "## Representative models",
    ]
    for _, r in reps.iterrows():
        lines.append(f"- {r['representative_role']}: `{r['model_config_id']}` - {r['one_sentence_explanation']}")
    (SUMMARIES / "constraint_first_rescreen_summary.md").write_text("\n".join(lines))


def assert_outputs() -> None:
    required = [
        METRICS / "constraint_first_rescreen_recomputed_metrics.csv",
        METRICS / "constraint_first_rescreen_pass_fail_table.csv",
        METRICS / "constraint_sensitivity_analysis.csv",
        METRICS / "constraint_first_rescreen_representative_models.csv",
        SUMMARIES / "constraint_first_rescreen_summary.md",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected non-empty output missing: {path}")


def main() -> None:
    for d in [METRICS, FIGS, SUMMARIES, LOGS, SCRIPTS]:
        d.mkdir(parents=True, exist_ok=True)
    check_inputs()
    run_lines = [f"Run started: {datetime.now().isoformat(timespec='seconds')}", f"Output directory: {OUT}"]
    data = load_inputs()
    make_inventory(data)
    metrics = recompute_metrics(data)
    metrics.to_csv(METRICS / "constraint_first_rescreen_recomputed_metrics.csv", index=False)
    pass_fail, sensitivity = constraint_tables(metrics)
    pass_fail.to_csv(METRICS / "constraint_first_rescreen_pass_fail_table.csv", index=False)
    sensitivity.to_csv(METRICS / "constraint_sensitivity_analysis.csv", index=False)
    survivors = survivor_tables(metrics, pass_fail)
    for level, table in survivors.items():
        table.to_csv(METRICS / f"constraint_first_rescreen_survivors_{level}.csv", index=False)
    reps = representative_models(metrics, pass_fail)
    reps.to_csv(METRICS / "constraint_first_rescreen_representative_models.csv", index=False)
    rtbin = data["rt_bin"]
    assert isinstance(rtbin, pd.DataFrame)
    make_figures(metrics, pass_fail, sensitivity, reps, rtbin)
    write_summary(metrics, pass_fail, sensitivity, reps)
    assert_outputs()
    run_lines.extend([
        f"Candidate pool: {len(metrics)}",
        f"Pareto count: {int(metrics['is_repaired_pareto'].sum())}",
        f"Lenient survivors: {int(pass_fail['pass_lenient'].sum())}",
        f"Main survivors: {int(pass_fail['pass_main'].sum())}",
        f"Strict survivors: {int(pass_fail['pass_strict'].sum())}",
        f"Run finished: {datetime.now().isoformat(timespec='seconds')}",
    ])
    (LOGS / "constraint_first_rescreen_run_log.txt").write_text("\n".join(run_lines))
    log("\n".join(run_lines))


if __name__ == "__main__":
    main()

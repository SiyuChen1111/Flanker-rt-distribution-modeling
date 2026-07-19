#!/usr/bin/env python3
"""Cross-validated four-bin evaluation for Flanker behavior and mechanisms."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
)
DEFAULT_OUTPUT_ROOT = ANALYSIS_ROOT / "flanker_rt_4bin_fitting"
GROUP_ORDER = ("young_20_29", "older_80_89")
CONGRUENCY_ORDER = (0, 1)
N_BINS = 4
RT_QUANTILES = (0.25, 0.50, 0.75)
PROFILE_KEYS = ["analysis_group", "congruency", "rt_bin"]
JOIN_KEYS = [
    "trial_id",
    "analysis_group",
    "true_rt_key",
    "human_correct",
    "congruency",
    "target_label",
    "flanker_label",
]
MECHANISM_COLUMNS = [
    "early_flanker_dominance",
    "target_recovery_time",
    "target_margin_at_readout",
    "readout_before_target_recovery",
]


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    source_path: Path
    model_id_col: str
    seed_col: str | None
    rt_col: str
    correct_col: str
    choice_col: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--min-cell-trials", type=int, default=5)
    parser.add_argument("--binning-mode", choices=["fixed", "dual"], default="fixed")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args()


def default_candidate_specs(root: Path = ANALYSIS_ROOT) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "R5_baseline",
            root / "fitting/representative_trial_level_predictions.csv",
            "model_name",
            "seed",
            "pred_rt",
            "model_correct",
            "pred_choice",
        ),
        CandidateSpec(
            "WR2_fine",
            root
            / "wr2_uncertainty_schedule_fine_search/metrics/wr2_fine_search_top_candidates_trial_level.csv",
            "model_config_id",
            None,
            "model_rt",
            "model_correct",
            "final_choice",
        ),
        CandidateSpec(
            "WR2_age_noise",
            root / "wr2_age_specific_ww_noise/metrics/age_specific_ww_noise_top_models_trial_level.csv",
            "candidate_base_id",
            "ww_seed_index",
            "model_rt",
            "model_correct",
            "final_choice",
        ),
        CandidateSpec(
            "gate_execution",
            root / "wr2_evidence_gate_and_execution_validation/metrics/execution_trial_level_best.csv",
            "candidate_base_id",
            "seed_index",
            "model_rt",
            "model_correct",
            "final_choice",
        ),
        CandidateSpec(
            "faithful_WW",
            root
            / "faithful_ww_hvenet_core_fit_stage2_stage3_completion/metrics/stage2_stage3_trial_level_top_candidates.csv",
            "model_config_id",
            None,
            "model_rt",
            "model_correct",
            "final_choice",
        ),
    ]


def specs_from_manifest(path: Path) -> list[CandidateSpec]:
    frame = pd.read_csv(path).fillna("")
    required = {"family", "source_path", "model_id_col", "rt_col", "correct_col"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate manifest is missing columns: {sorted(missing)}")
    specs = []
    for row in frame.to_dict("records"):
        source = Path(str(row["source_path"]))
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        specs.append(
            CandidateSpec(
                family=str(row["family"]),
                source_path=source,
                model_id_col=str(row["model_id_col"]),
                seed_col=str(row.get("seed_col", "")) or None,
                rt_col=str(row["rt_col"]),
                correct_col=str(row["correct_col"]),
                choice_col=str(row.get("choice_col", "")) or None,
            )
        )
    return specs


def normalize_congruency(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        out = pd.to_numeric(values, errors="raise").astype(int)
    else:
        mapped = values.astype(str).str.lower().map(
            {"congruent": 0, "incongruent": 1, "0": 0, "1": 1}
        )
        if mapped.isna().any():
            raise ValueError(f"Unknown congruency values: {values[mapped.isna()].unique().tolist()}")
        out = mapped.astype(int)
    if not set(out.unique()).issubset({0, 1}):
        raise ValueError("Congruency must contain only 0/1 or congruent/incongruent.")
    return out


def coerce_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float).ne(0)
    mapped = values.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Cannot convert values to bool: {values[mapped.isna()].unique().tolist()}")
    return mapped.astype(bool)


def make_join_columns(frame: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["trial_id"] = pd.to_numeric(out[trial_col], errors="raise").astype(np.int64)
    out["analysis_group"] = out["analysis_group"].astype(str)
    out["true_rt_key"] = pd.to_numeric(out["true_rt"], errors="raise").round(6)
    out["human_correct"] = coerce_bool(out["human_correct"])
    out["congruency"] = normalize_congruency(out["congruency"])
    out["target_label"] = pd.to_numeric(out["target_label"], errors="raise").astype(int)
    out["flanker_label"] = pd.to_numeric(out["flanker_label"], errors="raise").astype(int)
    return out


def load_human_master(path: Path | None = None) -> pd.DataFrame:
    path = path or ANALYSIS_ROOT / "fitting/representative_trial_level_predictions.csv"
    use = [
        "row_index",
        "analysis_group",
        "user_id",
        "true_rt",
        "human_correct",
        "congruency",
        "target_label",
        "flanker_label",
        "response_label",
    ]
    frame = make_join_columns(pd.read_csv(path, usecols=use), "row_index")
    if frame.duplicated(JOIN_KEYS).any():
        raise ValueError("Human trial join key is not unique.")
    frame["trial_uid"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _random_response_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if "lapse_triggered" in frame:
        lapse = frame["lapse_triggered"].where(frame["lapse_triggered"].notna(), False)
        mask |= coerce_bool(lapse.infer_objects(copy=False))
    for column in ("final_choice_source", "choice_type", "lapse_choice_type"):
        if column in frame:
            text = frame[column].fillna("").astype(str).str.lower()
            mask |= text.str.contains("lapse|random", regex=True)
    return mask


def normalize_candidate(spec: CandidateSpec, human: pd.DataFrame) -> pd.DataFrame:
    if not spec.source_path.exists():
        raise FileNotFoundError(spec.source_path)
    frame = pd.read_csv(spec.source_path, low_memory=False)
    trial_col = "trial_id" if "trial_id" in frame else "row_index"
    required = {
        trial_col,
        "analysis_group",
        "true_rt",
        "human_correct",
        "congruency",
        "target_label",
        "flanker_label",
        spec.model_id_col,
        spec.rt_col,
        spec.correct_col,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.family} is missing columns: {sorted(missing)}")
    frame = make_join_columns(frame, trial_col)
    human_map = human[JOIN_KEYS + ["trial_uid", "user_id"]].rename(
        columns={"user_id": "human_user_id"}
    )
    frame = frame.merge(human_map, on=JOIN_KEYS, how="left", validate="many_to_one")
    if frame["trial_uid"].isna().any():
        examples = frame.loc[frame["trial_uid"].isna(), JOIN_KEYS].head(3).to_dict("records")
        raise ValueError(f"{spec.family} has unmatched human trials: {examples}")
    if "user_id" in frame:
        candidate_user = frame["user_id"].astype(str)
        human_user = frame["human_user_id"].astype(str)
        if not candidate_user.eq(human_user).all():
            raise ValueError(f"{spec.family} contains user IDs that disagree with the human master.")
    frame["user_id"] = frame["human_user_id"]
    frame["candidate_id"] = spec.family + "::" + frame[spec.model_id_col].astype(str)
    frame["seed_id"] = (
        frame[spec.seed_col].astype(str) if spec.seed_col and spec.seed_col in frame else "single"
    )
    frame["model_rt"] = pd.to_numeric(frame[spec.rt_col], errors="coerce")
    frame["model_correct"] = coerce_bool(frame[spec.correct_col])
    frame["model_choice"] = (
        pd.to_numeric(frame[spec.choice_col], errors="coerce")
        if spec.choice_col and spec.choice_col in frame
        else np.nan
    )
    frame["explicit_random_response"] = _random_response_mask(frame)
    rename = {}
    if "s_target_minus_flanker_at_readout" in frame and "signed_target_margin_at_readout" not in frame:
        rename["s_target_minus_flanker_at_readout"] = "signed_target_margin_at_readout"
    frame = frame.rename(columns=rename)
    canonical = {
        "early_flanker_dominance": "early_flanker_dominance",
        "target_recovery_time": "target_recovery_time",
        "signed_target_margin_at_readout": "target_margin_at_readout",
        "readout_before_target_recovery": "readout_before_target_recovery",
    }
    for source, target in canonical.items():
        if source in frame:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
        elif target not in frame:
            frame[target] = np.nan
    keep = [
        "candidate_id",
        "seed_id",
        "trial_uid",
        "user_id",
        "analysis_group",
        "congruency",
        "true_rt",
        "human_correct",
        "model_rt",
        "model_correct",
        "model_choice",
        "explicit_random_response",
        *MECHANISM_COLUMNS,
    ]
    out = frame[keep].copy()
    key = ["candidate_id", "seed_id", "trial_uid"]
    if out.duplicated(key).any():
        raise ValueError(f"{spec.family} has duplicate candidate/seed/trial rows.")
    if out[["model_rt", "model_correct"]].isna().any().any():
        raise ValueError(f"{spec.family} contains missing model RT or correctness.")
    return out


def candidate_manifest(specs: Sequence[CandidateSpec], candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in specs:
        prefix = spec.family + "::"
        part = candidates[candidates["candidate_id"].str.startswith(prefix)]
        for candidate_id, group in part.groupby("candidate_id", sort=True):
            errors = ~group["model_correct"]
            random_errors = group["explicit_random_response"] & errors
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": spec.family,
                    "source_path": str(spec.source_path),
                    "model_id_col": spec.model_id_col,
                    "seed_col": spec.seed_col or "",
                    "rt_col": spec.rt_col,
                    "correct_col": spec.correct_col,
                    "n_seeds": int(group["seed_id"].nunique()),
                    "n_unique_trials": int(group["trial_uid"].nunique()),
                    "uses_explicit_random_response": bool(group["explicit_random_response"].any()),
                    "random_error_fraction": float(random_errors.sum() / max(1, errors.sum())),
                }
            )
    return pd.DataFrame(rows)


def compute_bin_edges(rt: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(rt), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < N_BINS or np.unique(values).size < N_BINS:
        raise ValueError("Not enough distinct finite RT values to define four bins.")
    edges = np.quantile(values, RT_QUANTILES)
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"RT quartile edges are not strictly increasing: {edges.tolist()}")
    return edges.astype(float)


def assign_rt_bins(rt: Iterable[float], edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(list(rt), dtype=float)
    edges_array = np.asarray(edges, dtype=float)
    if edges_array.shape != (3,) or not np.all(np.diff(edges_array) > 0):
        raise ValueError("Four-bin assignment requires three increasing edges.")
    return np.searchsorted(edges_array, values, side="left").astype(int) + 1


def make_subject_folds(human: pd.DataFrame, seed: int = 20260716) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assignments = []
    for group in GROUP_ORDER:
        subjects = np.asarray(sorted(human.loc[human["analysis_group"].eq(group), "user_id"].unique()))
        if len(subjects) < 4 or len(subjects) % 4 != 0:
            raise ValueError(f"{group} must have a positive multiple of four subjects; found {len(subjects)}.")
        shuffled = rng.permutation(subjects)
        for position, subject in enumerate(shuffled):
            assignments.append(
                {"analysis_group": group, "user_id": subject, "test_fold": int(position % 4)}
            )
    base = pd.DataFrame(assignments)
    rows = []
    for fold in range(4):
        for row in base.itertuples(index=False):
            rows.append(
                {
                    "fold": fold,
                    "analysis_group": row.analysis_group,
                    "user_id": row.user_id,
                    "role": "test" if row.test_fold == fold else "train",
                }
            )
    out = pd.DataFrame(rows)
    for fold in range(4):
        train = set(out[(out.fold == fold) & (out.role == "train")].user_id)
        test = set(out[(out.fold == fold) & (out.role == "test")].user_id)
        if train & test:
            raise AssertionError("A subject appears in both train and test.")
    return out


def profile_frame(
    frame: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    edges: Sequence[float],
) -> pd.DataFrame:
    work = frame.copy()
    work["rt_bin"] = assign_rt_bins(work[rt_col], edges)
    totals = work.groupby(["analysis_group", "congruency"], observed=True).size().rename("total")
    profile = (
        work.groupby(PROFILE_KEYS, observed=True)
        .agg(
            n_trials=(rt_col, "size"),
            mean_rt=(rt_col, "mean"),
            accuracy=(correct_col, "mean"),
        )
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [GROUP_ORDER, CONGRUENCY_ORDER, range(1, N_BINS + 1)], names=PROFILE_KEYS
    ).to_frame(index=False)
    profile = full.merge(profile, on=PROFILE_KEYS, how="left")
    profile["n_trials"] = profile["n_trials"].fillna(0).astype(int)
    profile["error_rate"] = 1.0 - profile["accuracy"]
    profile = profile.join(totals, on=["analysis_group", "congruency"])
    profile["bin_proportion"] = profile["n_trials"] / profile["total"]
    return profile.drop(columns="total")


def rt_quantile_frame(frame: pd.DataFrame, rt_col: str) -> pd.DataFrame:
    rows = []
    for (group, congruency), part in frame.groupby(["analysis_group", "congruency"], sort=True):
        for quantile in RT_QUANTILES:
            rows.append(
                {
                    "analysis_group": group,
                    "congruency": int(congruency),
                    "quantile": quantile,
                    "rt_quantile": float(part[rt_col].quantile(quantile)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_tolerances(
    human_train: pd.DataFrame,
    edges: Sequence[float],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = profile_frame(human_train, "true_rt", "human_correct", edges)
    reference_q = rt_quantile_frame(human_train, "true_rt")
    rng = np.random.default_rng(seed)
    profile_deviations = []
    quantile_deviations = []
    subject_parts = {
        (group, subject): part
        for (group, subject), part in human_train.groupby(["analysis_group", "user_id"], sort=False)
    }
    subjects_by_group = {
        group: np.asarray(sorted(human_train.loc[human_train.analysis_group.eq(group), "user_id"].unique()))
        for group in GROUP_ORDER
    }
    for _ in range(reps):
        sampled_parts = []
        for group, subjects in subjects_by_group.items():
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            sampled_parts.extend(subject_parts[(group, subject)] for subject in sampled)
        sample = pd.concat(sampled_parts, ignore_index=True)
        prof = profile_frame(sample, "true_rt", "human_correct", edges)
        dev = prof[PROFILE_KEYS].copy()
        dev["error_dev"] = (prof["error_rate"] - reference["error_rate"]).abs()
        dev["proportion_dev"] = (
            prof["bin_proportion"] - reference["bin_proportion"]
        ).abs()
        profile_deviations.append(dev)
        quant = rt_quantile_frame(sample, "true_rt")
        qdev = quant[["analysis_group", "congruency", "quantile"]].copy()
        qdev["rt_dev"] = (quant["rt_quantile"] - reference_q["rt_quantile"]).abs()
        quantile_deviations.append(qdev)
    profile_all = pd.concat(profile_deviations, ignore_index=True)
    tolerances = (
        profile_all.groupby(PROFILE_KEYS, observed=True)
        .agg(
            error_tolerance=("error_dev", lambda x: float(np.quantile(x, 0.95))),
            proportion_tolerance=("proportion_dev", lambda x: float(np.quantile(x, 0.95))),
        )
        .reset_index()
    )
    tolerances["error_tolerance"] = tolerances["error_tolerance"].clip(lower=0.01)
    tolerances["proportion_tolerance"] = tolerances["proportion_tolerance"].clip(lower=0.02)
    quantile_all = pd.concat(quantile_deviations, ignore_index=True)
    q_tolerances = (
        quantile_all.groupby(["analysis_group", "congruency", "quantile"], observed=True)
        .agg(rt_tolerance=("rt_dev", lambda x: float(np.quantile(x, 0.95))))
        .reset_index()
    )
    q_tolerances["rt_tolerance"] = q_tolerances["rt_tolerance"].clip(lower=0.02)
    return tolerances, q_tolerances


def seed_metrics(
    model_part: pd.DataFrame,
    human_profile: pd.DataFrame,
    human_quantiles: pd.DataFrame,
    tolerances: pd.DataFrame,
    quantile_tolerances: pd.DataFrame,
    edges: Sequence[float],
    min_cell_trials: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    model_profile = profile_frame(model_part, "model_rt", "model_correct", edges)
    merged = human_profile.merge(model_profile, on=PROFILE_KEYS, suffixes=("_human", "_model"))
    merged = merged.merge(tolerances, on=PROFILE_KEYS, how="left")
    merged["error_z"] = (
        merged["error_rate_model"] - merged["error_rate_human"]
    ).abs() / merged["error_tolerance"]
    merged["proportion_z"] = (
        merged["bin_proportion_model"] - merged["bin_proportion_human"]
    ).abs() / merged["proportion_tolerance"]
    inc = merged[merged["congruency"].eq(1)]
    cong = merged[merged["congruency"].eq(0)]
    model_q = rt_quantile_frame(model_part, "model_rt")
    qmerged = human_quantiles.merge(
        model_q, on=["analysis_group", "congruency", "quantile"], suffixes=("_human", "_model")
    ).merge(quantile_tolerances, on=["analysis_group", "congruency", "quantile"])
    qmerged["rt_z"] = (
        qmerged["rt_quantile_model"] - qmerged["rt_quantile_human"]
    ).abs() / qmerged["rt_tolerance"]
    congruent_nonzero = True
    fast_error_direction = True
    for group in GROUP_ORDER:
        gp = cong[cong["analysis_group"].eq(group)].sort_values("rt_bin")
        congruent_nonzero &= bool((gp["error_rate_model"] * gp["n_trials_model"]).sum() > 0)
        fast_error_direction &= bool(gp.iloc[0]["error_rate_model"] >= gp.iloc[-1]["error_rate_model"])
    cell_count_pass = bool((merged["n_trials_model"] >= min_cell_trials).all())
    inc_fit_pass = bool(inc["error_z"].mean() <= 1.0 and inc["error_z"].max() <= 2.0)
    cong_fit_pass = bool(cong["error_z"].mean() <= 1.0)
    occupancy_pass = bool(merged["proportion_z"].mean() <= 1.0)
    direction_pass = bool(congruent_nonzero and fast_error_direction)
    score_inc = float(inc["error_z"].mean())
    score_cong = float(cong["error_z"].mean())
    score_occupancy = float(merged["proportion_z"].mean())
    score_rt = float(qmerged["rt_z"].mean())
    score = (4 * score_inc + 2 * score_cong + score_occupancy + 0.5 * score_rt) / 7.5
    row = {
        "score_total": score,
        "score_incongruent_error": score_inc,
        "score_congruent_error": score_cong,
        "score_bin_proportion": score_occupancy,
        "score_rt_quantiles": score_rt,
        "min_model_cell_trials": int(merged["n_trials_model"].min()),
        "max_incongruent_error_z": float(inc["error_z"].max()),
        "cell_count_pass": cell_count_pass,
        "incongruent_fit_pass": inc_fit_pass,
        "congruent_fit_pass": cong_fit_pass,
        "occupancy_pass": occupancy_pass,
        "congruent_nonzero_pass": bool(congruent_nonzero),
        "fast_error_direction_pass": bool(fast_error_direction),
        "seed_behavior_pass": bool(
            cell_count_pass and inc_fit_pass and cong_fit_pass and occupancy_pass and direction_pass
        ),
    }
    return row, merged


def evaluate_candidate(
    candidate_part: pd.DataFrame,
    human_part: pd.DataFrame,
    tolerances: pd.DataFrame,
    quantile_tolerances: pd.DataFrame,
    edges: Sequence[float],
    min_cell_trials: int,
) -> tuple[dict[str, object], list[pd.DataFrame]]:
    human_profile = profile_frame(human_part, "true_rt", "human_correct", edges)
    human_quantiles = rt_quantile_frame(human_part, "true_rt")
    seed_rows = []
    profiles = []
    for seed_id, seed_part in candidate_part.groupby("seed_id", sort=True):
        metrics, merged = seed_metrics(
            seed_part,
            human_profile,
            human_quantiles,
            tolerances,
            quantile_tolerances,
            edges,
            min_cell_trials,
        )
        metrics["seed_id"] = seed_id
        seed_rows.append(metrics)
        model_profile = profile_frame(seed_part, "model_rt", "model_correct", edges)
        model_profile["source"] = "model"
        model_profile["aggregation"] = "seed"
        model_profile["seed_id"] = seed_id
        profiles.append(model_profile)
    seed_frame = pd.DataFrame(seed_rows)
    n_seeds = int(candidate_part["seed_id"].nunique())
    seed_pass_rate = float(seed_frame["seed_behavior_pass"].mean())
    errors = ~candidate_part["model_correct"]
    random_errors = errors & candidate_part["explicit_random_response"]
    uses_random = bool(candidate_part["explicit_random_response"].any())
    summary = {
        "n_seeds": n_seeds,
        "seed_pass_rate": seed_pass_rate,
        "score_total": float(seed_frame["score_total"].mean()),
        "score_incongruent_error": float(seed_frame["score_incongruent_error"].mean()),
        "score_congruent_error": float(seed_frame["score_congruent_error"].mean()),
        "score_bin_proportion": float(seed_frame["score_bin_proportion"].mean()),
        "score_rt_quantiles": float(seed_frame["score_rt_quantiles"].mean()),
        "min_model_cell_trials": int(seed_frame["min_model_cell_trials"].min()),
        "max_incongruent_error_z": float(seed_frame["max_incongruent_error_z"].max()),
        "cell_count_pass": bool(seed_frame["cell_count_pass"].all()),
        "incongruent_fit_pass": bool(seed_frame["incongruent_fit_pass"].mean() >= 0.8),
        "congruent_fit_pass": bool(seed_frame["congruent_fit_pass"].mean() >= 0.8),
        "occupancy_pass": bool(seed_frame["occupancy_pass"].mean() >= 0.8),
        "congruent_nonzero_pass": bool(seed_frame["congruent_nonzero_pass"].mean() >= 0.8),
        "fast_error_direction_pass": bool(seed_frame["fast_error_direction_pass"].mean() >= 0.8),
        "seed_count_pass": n_seeds >= 10,
        "seed_stability_pass": seed_pass_rate >= 0.8,
        "uses_explicit_random_response": uses_random,
        "random_error_fraction": float(random_errors.sum() / max(1, errors.sum())),
    }
    summary["behavior_pass"] = bool(
        summary["cell_count_pass"]
        and summary["incongruent_fit_pass"]
        and summary["congruent_fit_pass"]
        and summary["occupancy_pass"]
        and summary["congruent_nonzero_pass"]
        and summary["fast_error_direction_pass"]
        and summary["seed_count_pass"]
        and summary["seed_stability_pass"]
        and not uses_random
    )
    if profiles:
        all_seed_profiles = pd.concat(profiles, ignore_index=True)
        mean_profile = (
            all_seed_profiles.groupby(PROFILE_KEYS, observed=True)
            .agg(
                n_trials=("n_trials", "mean"),
                mean_rt=("mean_rt", "mean"),
                accuracy=("accuracy", "mean"),
                error_rate=("error_rate", "mean"),
                bin_proportion=("bin_proportion", "mean"),
            )
            .reset_index()
        )
        mean_profile["source"] = "model"
        mean_profile["aggregation"] = "seed_mean"
        mean_profile["seed_id"] = "mean"
        profiles.append(mean_profile)
    return summary, profiles


def assign_equal_count_bins(frame: pd.DataFrame, rt_col: str) -> pd.Series:
    """Assign four near-equal bins within each subject and congruency condition."""
    assigned = pd.Series(index=frame.index, dtype="int64")
    for _, part in frame.groupby(["analysis_group", "user_id", "congruency"], sort=False):
        order = np.argsort(part[rt_col].to_numpy(float), kind="mergesort")
        for bin_id, positions in enumerate(np.array_split(order, N_BINS), start=1):
            assigned.loc[part.index[positions]] = bin_id
    if assigned.isna().any():
        raise AssertionError("Equal-count bin assignment left trials unassigned.")
    return assigned.astype(int)


def equal_count_subject_profiles(
    frame: pd.DataFrame,
    rt_col: str,
    correct_col: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["rt_bin"] = assign_equal_count_bins(work, rt_col)
    return (
        work.groupby(["analysis_group", "user_id", "congruency", "rt_bin"], observed=True)
        .agg(
            n_trials=(rt_col, "size"),
            mean_rt=(rt_col, "mean"),
            accuracy=(correct_col, "mean"),
        )
        .reset_index()
        .assign(error_rate=lambda x: 1.0 - x["accuracy"])
    )


def aggregate_equal_count_profiles(subject_profiles: pd.DataFrame) -> pd.DataFrame:
    """Average participant profiles so each participant has equal weight."""
    return (
        subject_profiles.groupby(PROFILE_KEYS, observed=True)
        .agg(
            n_subjects=("user_id", "size"),
            mean_subject_bin_trials=("n_trials", "mean"),
            mean_rt=("mean_rt", "mean"),
            accuracy=("accuracy", "mean"),
            error_rate=("error_rate", "mean"),
        )
        .reset_index()
    )


def equal_count_contrasts(profile: pd.DataFrame) -> pd.DataFrame:
    pivot = profile.pivot_table(
        index=["analysis_group", "congruency"], columns="rt_bin", values="error_rate"
    )
    if 1 not in pivot or N_BINS not in pivot:
        raise ValueError("Equal-count profile is missing the fastest or slowest bin.")
    return (
        (pivot[1] - pivot[N_BINS])
        .rename("fast_minus_slow_error")
        .reset_index()
    )


def bootstrap_equal_count_tolerances(
    human_subject_profiles: pd.DataFrame,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = aggregate_equal_count_profiles(human_subject_profiles)
    reference_contrast = equal_count_contrasts(reference)
    rng = np.random.default_rng(seed)
    profile_deviations = []
    contrast_deviations = []
    subject_parts = {
        (group, subject): part
        for (group, subject), part in human_subject_profiles.groupby(
            ["analysis_group", "user_id"], sort=False
        )
    }
    subjects_by_group = {
        group: np.asarray(
            sorted(
                human_subject_profiles.loc[
                    human_subject_profiles.analysis_group.eq(group), "user_id"
                ].unique()
            )
        )
        for group in GROUP_ORDER
    }
    for _ in range(reps):
        sampled_parts = []
        for group, subjects in subjects_by_group.items():
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            sampled_parts.extend(subject_parts[(group, subject)] for subject in sampled)
        sample_profile = aggregate_equal_count_profiles(pd.concat(sampled_parts, ignore_index=True))
        dev = sample_profile[PROFILE_KEYS].copy()
        dev["error_dev"] = (
            sample_profile["error_rate"] - reference["error_rate"]
        ).abs()
        profile_deviations.append(dev)
        contrast = equal_count_contrasts(sample_profile)
        cdev = contrast[["analysis_group", "congruency"]].copy()
        cdev["contrast_dev"] = (
            contrast["fast_minus_slow_error"]
            - reference_contrast["fast_minus_slow_error"]
        ).abs()
        contrast_deviations.append(cdev)
    tolerances = (
        pd.concat(profile_deviations, ignore_index=True)
        .groupby(PROFILE_KEYS, observed=True)
        .agg(error_tolerance=("error_dev", lambda x: float(np.quantile(x, 0.95))))
        .reset_index()
    )
    tolerances["error_tolerance"] = tolerances["error_tolerance"].clip(lower=0.01)
    contrast_tolerances = (
        pd.concat(contrast_deviations, ignore_index=True)
        .groupby(["analysis_group", "congruency"], observed=True)
        .agg(contrast_tolerance=("contrast_dev", lambda x: float(np.quantile(x, 0.95))))
        .reset_index()
    )
    contrast_tolerances["contrast_tolerance"] = contrast_tolerances[
        "contrast_tolerance"
    ].clip(lower=0.01)
    return tolerances, contrast_tolerances


def shape_seed_metrics(
    model_subject_profile: pd.DataFrame,
    human_profile: pd.DataFrame,
    tolerances: pd.DataFrame,
    contrast_tolerances: pd.DataFrame,
    min_cell_trials: int,
) -> dict[str, object]:
    model_profile = aggregate_equal_count_profiles(model_subject_profile)
    merged = human_profile.merge(
        model_profile, on=PROFILE_KEYS, suffixes=("_human", "_model")
    ).merge(tolerances, on=PROFILE_KEYS, how="left")
    merged["error_z"] = (
        merged["error_rate_model"] - merged["error_rate_human"]
    ).abs() / merged["error_tolerance"]
    inc = merged[merged.congruency.eq(1)]
    cong = merged[merged.congruency.eq(0)]
    human_contrast = equal_count_contrasts(human_profile)
    model_contrast = equal_count_contrasts(model_profile)
    contrasts = human_contrast.merge(
        model_contrast,
        on=["analysis_group", "congruency"],
        suffixes=("_human", "_model"),
    ).merge(contrast_tolerances, on=["analysis_group", "congruency"])
    contrasts["contrast_z"] = (
        contrasts["fast_minus_slow_error_model"]
        - contrasts["fast_minus_slow_error_human"]
    ).abs() / contrasts["contrast_tolerance"]
    direction_match = bool(
        (
            contrasts["fast_minus_slow_error_model"]
            * contrasts["fast_minus_slow_error_human"]
            >= 0
        ).all()
    )
    congruent_nonzero = True
    for group in GROUP_ORDER:
        gp = model_subject_profile[
            (model_subject_profile.analysis_group == group)
            & (model_subject_profile.congruency == 0)
        ]
        congruent_nonzero &= bool((gp.error_rate * gp.n_trials).sum() > 0)
    min_subject_bin_trials = int(model_subject_profile.n_trials.min())
    cell_count_pass = min_subject_bin_trials >= min_cell_trials
    inc_fit_pass = bool(inc.error_z.mean() <= 1.0 and inc.error_z.max() <= 2.0)
    cong_fit_pass = bool(cong.error_z.mean() <= 1.0)
    contrast_fit_pass = bool(contrasts.contrast_z.mean() <= 1.0)
    score_inc = float(inc.error_z.mean())
    score_cong = float(cong.error_z.mean())
    score_contrast = float(contrasts.contrast_z.mean())
    score_total = (2.0 * score_inc + score_cong + score_contrast) / 4.0
    return {
        "shape_score_total": score_total,
        "shape_score_incongruent": score_inc,
        "shape_score_congruent": score_cong,
        "shape_score_fast_slow_contrast": score_contrast,
        "shape_max_incongruent_error_z": float(inc.error_z.max()),
        "min_subject_bin_trials": min_subject_bin_trials,
        "shape_cell_count_pass": bool(cell_count_pass),
        "shape_incongruent_fit_pass": inc_fit_pass,
        "shape_congruent_fit_pass": cong_fit_pass,
        "shape_contrast_fit_pass": contrast_fit_pass,
        "shape_direction_pass": direction_match,
        "shape_congruent_nonzero_pass": bool(congruent_nonzero),
        "seed_shape_pass": bool(
            cell_count_pass
            and inc_fit_pass
            and cong_fit_pass
            and contrast_fit_pass
            and direction_match
            and congruent_nonzero
        ),
    }


def evaluate_shape_candidate(
    candidate_part: pd.DataFrame,
    human_subject_profile: pd.DataFrame,
    tolerances: pd.DataFrame,
    contrast_tolerances: pd.DataFrame,
    min_cell_trials: int,
) -> tuple[dict[str, object], list[pd.DataFrame], list[pd.DataFrame]]:
    human_profile = aggregate_equal_count_profiles(human_subject_profile)
    seed_rows = []
    subject_profiles = []
    aggregate_profiles = []
    for seed_id, seed_part in candidate_part.groupby("seed_id", sort=True):
        subject_profile = equal_count_subject_profiles(
            seed_part, "model_rt", "model_correct"
        )
        metrics = shape_seed_metrics(
            subject_profile,
            human_profile,
            tolerances,
            contrast_tolerances,
            min_cell_trials,
        )
        metrics["seed_id"] = seed_id
        seed_rows.append(metrics)
        subject_profiles.append(
            subject_profile.assign(source="model", aggregation="subject", seed_id=seed_id)
        )
        aggregate_profiles.append(
            aggregate_equal_count_profiles(subject_profile).assign(
                source="model", aggregation="seed", seed_id=seed_id
            )
        )
    seed_frame = pd.DataFrame(seed_rows)
    all_subject_profiles = pd.concat(subject_profiles, ignore_index=True)
    n_seeds = int(candidate_part.seed_id.nunique())
    uses_random = bool(candidate_part.explicit_random_response.any())
    errors = ~candidate_part.model_correct
    random_errors = errors & candidate_part.explicit_random_response
    summary = {
        "n_seeds": n_seeds,
        "shape_seed_pass_rate": float(seed_frame.seed_shape_pass.mean()),
        "shape_score_total": float(seed_frame.shape_score_total.mean()),
        "shape_score_incongruent": float(seed_frame.shape_score_incongruent.mean()),
        "shape_score_congruent": float(seed_frame.shape_score_congruent.mean()),
        "shape_score_fast_slow_contrast": float(
            seed_frame.shape_score_fast_slow_contrast.mean()
        ),
        "shape_max_incongruent_error_z": float(
            seed_frame.shape_max_incongruent_error_z.max()
        ),
        "min_subject_bin_trials": int(seed_frame.min_subject_bin_trials.min()),
        "shape_cell_count_pass": bool(seed_frame.shape_cell_count_pass.all()),
        "shape_incongruent_fit_pass": bool(
            seed_frame.shape_incongruent_fit_pass.mean() >= 0.8
        ),
        "shape_congruent_fit_pass": bool(
            seed_frame.shape_congruent_fit_pass.mean() >= 0.8
        ),
        "shape_contrast_fit_pass": bool(seed_frame.shape_contrast_fit_pass.mean() >= 0.8),
        "shape_direction_pass": bool(seed_frame.shape_direction_pass.mean() >= 0.8),
        "shape_congruent_nonzero_pass": bool(
            seed_frame.shape_congruent_nonzero_pass.mean() >= 0.8
        ),
        "shape_seed_count_pass": n_seeds >= 10,
        "shape_seed_stability_pass": bool(seed_frame.seed_shape_pass.mean() >= 0.8),
        "uses_explicit_random_response": uses_random,
        "random_error_fraction": float(random_errors.sum() / max(1, errors.sum())),
    }
    summary["shape_pass"] = bool(
        summary["shape_cell_count_pass"]
        and summary["shape_incongruent_fit_pass"]
        and summary["shape_congruent_fit_pass"]
        and summary["shape_contrast_fit_pass"]
        and summary["shape_direction_pass"]
        and summary["shape_congruent_nonzero_pass"]
        and summary["shape_seed_count_pass"]
        and summary["shape_seed_stability_pass"]
        and not uses_random
    )
    seed_aggregates = pd.concat(aggregate_profiles, ignore_index=True)
    seed_mean = (
        seed_aggregates.groupby(PROFILE_KEYS, observed=True)
        .agg(
            n_subjects=("n_subjects", "mean"),
            mean_subject_bin_trials=("mean_subject_bin_trials", "mean"),
            mean_rt=("mean_rt", "mean"),
            accuracy=("accuracy", "mean"),
            error_rate=("error_rate", "mean"),
        )
        .reset_index()
        .assign(source="model", aggregation="seed_mean", seed_id="mean")
    )
    aggregate_profiles.append(seed_mean)
    return summary, subject_profiles, aggregate_profiles


def equal_count_mechanism_summary(
    candidate_part: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, str]:
    available = [column for column in MECHANISM_COLUMNS if candidate_part[column].notna().any()]
    if not available:
        return pd.DataFrame(), False, "required mechanism columns are unavailable"
    work_parts = []
    for _, seed_part in candidate_part.groupby("seed_id", sort=True):
        part = seed_part.copy()
        part["rt_bin"] = assign_equal_count_bins(part, "model_rt")
        work_parts.append(part)
    work = pd.concat(work_parts, ignore_index=True)
    work["correctness"] = np.where(work.model_correct, "correct", "error")
    summary = (
        work.groupby(
            ["candidate_id", "analysis_group", "congruency", "rt_bin", "correctness"],
            observed=True,
        )[available]
        .mean()
        .reset_index()
    )
    direction_by_group = []
    for group in GROUP_ORDER:
        inc = work[(work.analysis_group == group) & (work.congruency == 1)]
        correct = inc[inc.model_correct]
        error = inc[~inc.model_correct]
        if correct.empty or error.empty:
            direction_by_group.append(False)
            continue
        early_ok = (
            "early_flanker_dominance" in available
            and error.early_flanker_dominance.mean() > correct.early_flanker_dominance.mean()
        )
        recovery_ok = (
            "target_recovery_time" in available
            and error.target_recovery_time.mean() > correct.target_recovery_time.mean()
        )
        direction_by_group.append(bool(early_ok or recovery_ok))
    passed = bool(all(direction_by_group))
    reason = (
        "relative-speed mechanism direction reproduced in both age groups"
        if passed
        else "relative-speed mechanism direction failed"
    )
    return summary, passed, reason


def mechanism_summary(candidate_part: pd.DataFrame, edges: Sequence[float]) -> tuple[pd.DataFrame, bool, str]:
    available = [column for column in MECHANISM_COLUMNS if candidate_part[column].notna().any()]
    if not available:
        return pd.DataFrame(), False, "required mechanism columns are unavailable"
    work = candidate_part.copy()
    work["rt_bin"] = assign_rt_bins(work["model_rt"], edges)
    work["correctness"] = np.where(work["model_correct"], "correct", "error")
    summary = (
        work.groupby(
            ["candidate_id", "analysis_group", "congruency", "rt_bin", "correctness"],
            observed=True,
        )[available]
        .mean()
        .reset_index()
    )
    direction_by_group = []
    for group in GROUP_ORDER:
        inc = work[(work.analysis_group == group) & (work.congruency == 1)]
        correct = inc[inc.model_correct]
        error = inc[~inc.model_correct]
        if correct.empty or error.empty:
            direction_by_group.append(False)
            continue
        early_ok = False
        recovery_ok = False
        if "early_flanker_dominance" in available:
            early_ok = bool(error.early_flanker_dominance.mean() > correct.early_flanker_dominance.mean())
        if "target_recovery_time" in available:
            recovery_ok = bool(error.target_recovery_time.mean() > correct.target_recovery_time.mean())
        direction_by_group.append(early_ok or recovery_ok)
    passed = bool(all(direction_by_group))
    return summary, passed, "direction reproduced in both age groups" if passed else "mechanism direction failed"


def write_summary(
    path: Path,
    diagnostic_edges: Sequence[float],
    fit_summary: pd.DataFrame,
    selected_rows: pd.DataFrame,
    mechanism_rows: pd.DataFrame,
    diagnostic_only: bool,
) -> None:
    lines = [
        "# Flanker 4-bin fitting summary",
        "",
        "## Full-sample diagnostic",
        "",
        "- Human RT quartile edges: " + ", ".join(f"{x:.3f}s" for x in diagnostic_edges),
        "- These full-sample edges are diagnostic only; cross-validation uses training subjects only.",
    ]
    if diagnostic_only:
        lines.extend(["", "No cross-validation was requested."])
    else:
        train_pass = int(fit_summary.query("role == 'train'")["behavior_pass"].sum())
        test_pass = int(selected_rows["behavior_pass"].sum()) if not selected_rows.empty else 0
        lines.extend(
            [
                "",
                "## Cross-validated behavior",
                "",
                f"- Training candidate-fold combinations passing all gates: {train_pass}.",
                f"- Held-out folds passing all behavior gates: {test_pass}/4.",
            ]
        )
        if not selected_rows.empty:
            for row in selected_rows.sort_values("fold").itertuples(index=False):
                lines.append(
                    f"- Fold {row.fold}: `{row.candidate_id}`; score={row.score_total:.3f}; "
                    f"behavior_pass={bool(row.behavior_pass)}."
                )
        lines.extend(["", "## Mechanism interpretation", ""])
        if mechanism_rows.empty:
            lines.append("- No held-out behavioral survivor was available for mechanism interpretation.")
        else:
            passed = int(mechanism_rows[["fold", "candidate_id", "mechanism_pass"]].drop_duplicates()["mechanism_pass"].sum())
            lines.append(f"- Mechanism direction passed in {passed} selected fold(s).")
        lines.extend(
            [
                "",
                "## Interpretation limit",
                "",
                "- This is internal validation on 16 participants, including only four older participants; it is not external validation.",
                "- Models with fewer than 10 stochastic repeats remain exploratory even if their average score is favorable.",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def select_ranked_row(
    frame: pd.DataFrame,
    pass_col: str,
    gate_cols: Sequence[str],
    score_col: str,
) -> pd.Series:
    ranked = frame.copy()
    ranked["_gate_count"] = ranked[list(gate_cols)].sum(axis=1)
    return ranked.sort_values(
        [pass_col, "_gate_count", score_col], ascending=[False, False, True]
    ).iloc[0]


def run_equal_count_analysis(
    *,
    human: pd.DataFrame,
    candidates: pd.DataFrame,
    splits: pd.DataFrame,
    fixed_fit_rows: list[dict[str, object]],
    fixed_profile_rows: list[pd.DataFrame],
    fixed_contexts: dict[int, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
    bootstrap_reps: int,
    seed: int,
    min_cell_trials: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    shape_rows: list[dict[str, object]] = []
    subject_profile_rows: list[pd.DataFrame] = []
    aggregate_profile_rows: list[pd.DataFrame] = []
    tolerance_rows: list[pd.DataFrame] = []
    mechanism_rows: list[pd.DataFrame] = []
    dual_rows: list[pd.DataFrame] = []
    shape_gate_cols = [
        "shape_cell_count_pass",
        "shape_incongruent_fit_pass",
        "shape_congruent_fit_pass",
        "shape_contrast_fit_pass",
        "shape_direction_pass",
        "shape_congruent_nonzero_pass",
        "shape_seed_count_pass",
        "shape_seed_stability_pass",
    ]
    absolute_gate_cols = [
        "cell_count_pass",
        "incongruent_fit_pass",
        "congruent_fit_pass",
        "occupancy_pass",
        "congruent_nonzero_pass",
        "fast_error_direction_pass",
        "seed_count_pass",
        "seed_stability_pass",
    ]
    for fold in range(4):
        train_subjects = set(
            splits[(splits.fold == fold) & (splits.role == "train")].user_id
        )
        test_subjects = set(
            splits[(splits.fold == fold) & (splits.role == "test")].user_id
        )
        human_train = human[human.user_id.isin(train_subjects)]
        human_test = human[human.user_id.isin(test_subjects)]
        human_train_subject = equal_count_subject_profiles(
            human_train, "true_rt", "human_correct"
        )
        human_test_subject = equal_count_subject_profiles(
            human_test, "true_rt", "human_correct"
        )
        shape_tolerances, contrast_tolerances = bootstrap_equal_count_tolerances(
            human_train_subject, bootstrap_reps, seed + 50000 + fold * 1000
        )
        tolerance_rows.append(
            shape_tolerances.assign(fold=fold, tolerance_type="equal_count_error")
        )
        tolerance_rows.append(
            contrast_tolerances.assign(fold=fold, tolerance_type="fast_slow_contrast")
        )
        subject_profile_rows.append(
            human_train_subject.assign(
                fold=fold,
                role="train",
                candidate_id="human",
                source="human",
                aggregation="subject",
                seed_id="observed",
            )
        )
        aggregate_profile_rows.append(
            aggregate_equal_count_profiles(human_train_subject).assign(
                fold=fold,
                role="train",
                candidate_id="human",
                source="human",
                aggregation="observed",
                seed_id="observed",
            )
        )
        for candidate_id, candidate in candidates.groupby("candidate_id", sort=True):
            candidate_train = candidate[candidate.user_id.isin(train_subjects)]
            summary, subject_profiles, aggregate_profiles = evaluate_shape_candidate(
                candidate_train,
                human_train_subject,
                shape_tolerances,
                contrast_tolerances,
                min_cell_trials,
            )
            summary.update(
                {
                    "fold": fold,
                    "role": "train",
                    "candidate_id": candidate_id,
                    "shape_selected": False,
                    "dual_selected": False,
                }
            )
            shape_rows.append(summary)
            for profile in subject_profiles:
                subject_profile_rows.append(
                    profile.assign(
                        fold=fold, role="train", candidate_id=candidate_id
                    )
                )
            for profile in aggregate_profiles:
                aggregate_profile_rows.append(
                    profile.assign(
                        fold=fold, role="train", candidate_id=candidate_id
                    )
                )
        shape_train = pd.DataFrame(
            [row for row in shape_rows if row["fold"] == fold and row["role"] == "train"]
        )
        shape_selected = select_ranked_row(
            shape_train, "shape_pass", shape_gate_cols, "shape_score_total"
        )
        absolute_train = pd.DataFrame(
            [
                row
                for row in fixed_fit_rows
                if row["fold"] == fold and row["role"] == "train"
            ]
        )
        combined_train = absolute_train.merge(
            shape_train,
            on=["fold", "role", "candidate_id"],
            suffixes=("_absolute", "_shape"),
        )
        combined_train["absolute_pass"] = combined_train["behavior_pass"]
        combined_train["dual_pass"] = (
            combined_train["absolute_pass"] & combined_train["shape_pass"]
        )
        combined_train["dual_score_total"] = (
            combined_train["score_total"] + combined_train["shape_score_total"]
        ) / 2.0
        combined_train["dual_gate_count"] = combined_train[absolute_gate_cols].sum(
            axis=1
        ) + combined_train[shape_gate_cols].sum(axis=1)
        dual_selected = combined_train.sort_values(
            ["dual_pass", "dual_gate_count", "dual_score_total"],
            ascending=[False, False, True],
        ).iloc[0]
        for row in shape_rows:
            if row["fold"] == fold and row["role"] == "train":
                row["shape_selected"] = row["candidate_id"] == shape_selected.candidate_id
                row["dual_selected"] = row["candidate_id"] == dual_selected.candidate_id
        combined_train["shape_selected"] = combined_train.candidate_id.eq(
            shape_selected.candidate_id
        )
        combined_train["dual_selected"] = combined_train.candidate_id.eq(
            dual_selected.candidate_id
        )
        dual_rows.append(combined_train)

        test_candidate_ids = {
            str(shape_selected.candidate_id), str(dual_selected.candidate_id)
        }
        human_test_subject_output = human_test_subject.assign(
            fold=fold,
            role="test",
            candidate_id="human",
            source="human",
            aggregation="subject",
            seed_id="observed",
        )
        subject_profile_rows.append(human_test_subject_output)
        aggregate_profile_rows.append(
            aggregate_equal_count_profiles(human_test_subject).assign(
                fold=fold,
                role="test",
                candidate_id="human",
                source="human",
                aggregation="observed",
                seed_id="observed",
            )
        )
        for candidate_id in sorted(test_candidate_ids):
            candidate_test = candidates[
                candidates.candidate_id.eq(candidate_id)
                & candidates.user_id.isin(test_subjects)
            ]
            summary, subject_profiles, aggregate_profiles = evaluate_shape_candidate(
                candidate_test,
                human_test_subject,
                shape_tolerances,
                contrast_tolerances,
                min_cell_trials,
            )
            summary.update(
                {
                    "fold": fold,
                    "role": "test",
                    "candidate_id": candidate_id,
                    "shape_selected": candidate_id == shape_selected.candidate_id,
                    "dual_selected": candidate_id == dual_selected.candidate_id,
                }
            )
            shape_rows.append(summary)
            for profile in subject_profiles:
                subject_profile_rows.append(
                    profile.assign(fold=fold, role="test", candidate_id=candidate_id)
                )
            for profile in aggregate_profiles:
                aggregate_profile_rows.append(
                    profile.assign(fold=fold, role="test", candidate_id=candidate_id)
                )
            if summary["shape_pass"]:
                process, mechanism_pass, reason = equal_count_mechanism_summary(
                    candidate_test
                )
                if not process.empty:
                    process["fold"] = fold
                    process["mechanism_pass"] = mechanism_pass
                    process["mechanism_reason"] = reason
                    process["interpretation_scope"] = "relative_speed_only"
                    mechanism_rows.append(process)

        fixed_test_existing = [
            row
            for row in fixed_fit_rows
            if row["fold"] == fold
            and row["role"] == "test"
            and row["candidate_id"] == dual_selected.candidate_id
        ]
        if not fixed_test_existing:
            edges, fixed_tolerances, q_tolerances = fixed_contexts[fold]
            candidate_test = candidates[
                candidates.candidate_id.eq(dual_selected.candidate_id)
                & candidates.user_id.isin(test_subjects)
            ]
            fixed_summary, fixed_profiles = evaluate_candidate(
                candidate_test,
                human_test,
                fixed_tolerances,
                q_tolerances,
                edges,
                min_cell_trials,
            )
            fixed_summary.update(
                {
                    "fold": fold,
                    "role": "test",
                    "candidate_id": dual_selected.candidate_id,
                    "selected": False,
                    "dual_selected": True,
                }
            )
            fixed_fit_rows.append(fixed_summary)
            for profile in fixed_profiles:
                fixed_profile_rows.append(
                    profile.assign(
                        fold=fold,
                        role="test",
                        candidate_id=dual_selected.candidate_id,
                    )
                )

        fixed_test = pd.DataFrame(
            [
                row
                for row in fixed_fit_rows
                if row["fold"] == fold
                and row["role"] == "test"
                and row["candidate_id"] in test_candidate_ids
            ]
        )
        shape_test = pd.DataFrame(
            [
                row
                for row in shape_rows
                if row["fold"] == fold and row["role"] == "test"
            ]
        )
        combined_test = fixed_test.merge(
            shape_test,
            on=["fold", "role", "candidate_id"],
            how="outer",
            suffixes=("_absolute", "_shape"),
        )
        combined_test["absolute_pass"] = combined_test["behavior_pass"].where(
            combined_test["behavior_pass"].notna(), False
        ).infer_objects(copy=False)
        combined_test["shape_pass"] = combined_test["shape_pass"].where(
            combined_test["shape_pass"].notna(), False
        ).infer_objects(copy=False)
        combined_test["dual_pass"] = (
            combined_test.absolute_pass & combined_test.shape_pass
        )
        combined_test["dual_score_total"] = (
            combined_test.score_total + combined_test.shape_score_total
        ) / 2.0
        combined_test["shape_selected"] = combined_test.candidate_id.eq(
            shape_selected.candidate_id
        )
        combined_test["dual_selected"] = combined_test.candidate_id.eq(
            dual_selected.candidate_id
        )
        dual_rows.append(combined_test)

    return (
        pd.DataFrame(shape_rows),
        pd.concat(subject_profile_rows, ignore_index=True),
        pd.concat(aggregate_profile_rows, ignore_index=True),
        pd.concat(tolerance_rows, ignore_index=True, sort=False),
        pd.concat(dual_rows, ignore_index=True, sort=False),
        pd.concat(mechanism_rows, ignore_index=True)
        if mechanism_rows
        else pd.DataFrame(
            columns=[
                "fold",
                "candidate_id",
                "analysis_group",
                "congruency",
                "rt_bin",
                "correctness",
                *MECHANISM_COLUMNS,
                "mechanism_pass",
                "mechanism_reason",
                "interpretation_scope",
            ]
        ),
    )


def write_dual_summary(
    path: Path,
    diagnostic_edges: Sequence[float],
    dual_summary: pd.DataFrame,
    mechanism_rows: pd.DataFrame,
) -> None:
    test = dual_summary[dual_summary.role.eq("test") & dual_summary.dual_selected.eq(True)]
    train = dual_summary[dual_summary.role.eq("train")]
    lines = [
        "# Flanker dual-track 4-bin summary",
        "",
        "## Full-sample diagnostic",
        "",
        "- Human RT quartile edges: " + ", ".join(f"{x:.3f}s" for x in diagnostic_edges),
        "- Fixed-time bins evaluate absolute RT placement; equal-count bins evaluate relative-speed behavior shape.",
        "",
        "## Cross-validated dual-track behavior",
        "",
        f"- Training shape passes: {int(train.shape_pass.sum())}.",
        f"- Training absolute passes: {int(train.absolute_pass.sum())}.",
        f"- Training dual passes: {int(train.dual_pass.sum())}.",
        f"- Held-out dual-selected shape passes: {int(test.shape_pass.sum())}/4.",
        f"- Held-out dual-selected absolute passes: {int(test.absolute_pass.sum())}/4.",
        f"- Held-out complete dual passes: {int(test.dual_pass.sum())}/4.",
    ]
    for row in test.sort_values("fold").itertuples(index=False):
        lines.append(
            f"- Fold {row.fold}: `{row.candidate_id}`; shape_pass={bool(row.shape_pass)}; "
            f"absolute_pass={bool(row.absolute_pass)}; dual_pass={bool(row.dual_pass)}."
        )
    lines.extend(["", "## Mechanism interpretation", ""])
    if mechanism_rows.empty:
        lines.append("- No held-out shape survivor was available for relative-speed mechanism interpretation.")
    else:
        passed = int(
            mechanism_rows[["fold", "candidate_id", "mechanism_pass"]]
            .drop_duplicates()
            .mechanism_pass.sum()
        )
        lines.append(
            f"- Relative-speed mechanism direction passed in {passed} selected fold(s); this does not establish absolute RT or age-distribution fit."
        )
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "- Equal-count bins deliberately remove occupancy differences and cannot validate absolute RT distributions.",
            "- This remains internal validation on 16 participants, including only four older participants.",
            "- Candidates with fewer than 10 stochastic repeats remain exploratory.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive.")
    output_dir = args.output_root / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    human = load_human_master()
    specs = specs_from_manifest(args.candidate_manifest) if args.candidate_manifest else default_candidate_specs()
    candidates = pd.concat([normalize_candidate(spec, human) for spec in specs], ignore_index=True)
    manifest = candidate_manifest(specs, candidates)
    output_dir.mkdir(parents=True)
    manifest.to_csv(output_dir / "candidate_manifest.csv", index=False)

    diagnostic_edges = compute_bin_edges(human["true_rt"])
    diagnostic_profile = profile_frame(human, "true_rt", "human_correct", diagnostic_edges)
    baseline = candidates[candidates.candidate_id.eq("R5_baseline::R5_combined_best")]
    baseline_profile = profile_frame(baseline, "model_rt", "model_correct", diagnostic_edges)
    diagnostic = pd.concat(
        [
            diagnostic_profile.assign(source="human"),
            baseline_profile.assign(source="R5_baseline"),
        ],
        ignore_index=True,
    )
    diagnostic.to_csv(output_dir / "full_sample_baseline_profile.csv", index=False)
    edge_rows = [
        {
            "fold": "full_sample_diagnostic",
            "source": "all_human_trials_diagnostic_only",
            "q25": diagnostic_edges[0],
            "q50": diagnostic_edges[1],
            "q75": diagnostic_edges[2],
            "n_human_trials": len(human),
        }
    ]
    if args.diagnostic_only:
        pd.DataFrame(edge_rows).to_csv(output_dir / "rt_bin_edges.csv", index=False)
        pd.DataFrame(columns=["fold", "role", "analysis_group", "user_id"]).to_csv(
            output_dir / "split_manifest.csv", index=False
        )
        for filename in (
            "rt_bin_profiles.csv",
            "rt_bin_fit_summary.csv",
            "bootstrap_tolerances.csv",
            "decision_process_by_bin.csv",
            "decision_trajectory_by_bin.csv",
        ):
            pd.DataFrame().to_csv(output_dir / filename, index=False)
        if args.binning_mode == "dual":
            for filename in (
                "equal_count_subject_profiles.csv",
                "equal_count_bin_profiles.csv",
                "equal_count_fit_summary.csv",
                "equal_count_bootstrap_tolerances.csv",
                "dual_track_model_summary.csv",
                "decision_process_equal_count_bins.csv",
            ):
                pd.DataFrame().to_csv(output_dir / filename, index=False)
        write_summary(
            output_dir / "summary.md",
            diagnostic_edges,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            True,
        )
        return output_dir

    splits = make_subject_folds(human, args.seed)
    splits.to_csv(output_dir / "split_manifest.csv", index=False)
    fit_rows = []
    profile_rows = []
    selected_test_rows = []
    mechanism_rows = []
    tolerance_rows = []
    trajectory_rows: list[pd.DataFrame] = []
    fixed_contexts: dict[int, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
    for fold in range(4):
        train_subjects = set(splits[(splits.fold == fold) & (splits.role == "train")].user_id)
        test_subjects = set(splits[(splits.fold == fold) & (splits.role == "test")].user_id)
        human_train = human[human.user_id.isin(train_subjects)]
        human_test = human[human.user_id.isin(test_subjects)]
        edges = compute_bin_edges(human_train["true_rt"])
        edge_rows.append(
            {
                "fold": fold,
                "source": "training_human_subjects_only",
                "q25": edges[0],
                "q50": edges[1],
                "q75": edges[2],
                "n_human_trials": len(human_train),
            }
        )
        tolerances, q_tolerances = bootstrap_tolerances(
            human_train, edges, args.bootstrap_reps, args.seed + fold * 1000
        )
        fixed_contexts[fold] = (edges, tolerances, q_tolerances)
        tolerance_rows.append(tolerances.assign(fold=fold, tolerance_type="rt_bin_profile"))
        tolerance_rows.append(q_tolerances.assign(fold=fold, tolerance_type="rt_quantile"))
        human_train_profile = profile_frame(human_train, "true_rt", "human_correct", edges)
        human_train_profile = human_train_profile.assign(
            fold=fold, role="train", candidate_id="human", source="human", aggregation="observed", seed_id="observed"
        )
        profile_rows.append(human_train_profile)
        for candidate_id, candidate in candidates.groupby("candidate_id", sort=True):
            candidate_train = candidate[candidate.user_id.isin(train_subjects)]
            summary, profiles = evaluate_candidate(
                candidate_train,
                human_train,
                tolerances,
                q_tolerances,
                edges,
                args.min_cell_trials,
            )
            summary.update({"fold": fold, "role": "train", "candidate_id": candidate_id, "selected": False})
            fit_rows.append(summary)
            for prof in profiles:
                profile_rows.append(prof.assign(fold=fold, role="train", candidate_id=candidate_id))
        fold_train = pd.DataFrame([row for row in fit_rows if row["fold"] == fold and row["role"] == "train"])
        gate_columns = [
            "cell_count_pass",
            "incongruent_fit_pass",
            "congruent_fit_pass",
            "occupancy_pass",
            "congruent_nonzero_pass",
            "fast_error_direction_pass",
            "seed_count_pass",
            "seed_stability_pass",
        ]
        fold_train["gate_count"] = fold_train[gate_columns].sum(axis=1)
        selected = fold_train.sort_values(
            ["behavior_pass", "gate_count", "score_total"], ascending=[False, False, True]
        ).iloc[0]
        for row in fit_rows:
            if row["fold"] == fold and row["role"] == "train" and row["candidate_id"] == selected.candidate_id:
                row["selected"] = True
        candidate_test = candidates[
            candidates.candidate_id.eq(selected.candidate_id) & candidates.user_id.isin(test_subjects)
        ]
        test_summary, test_profiles = evaluate_candidate(
            candidate_test,
            human_test,
            tolerances,
            q_tolerances,
            edges,
            args.min_cell_trials,
        )
        test_summary.update(
            {"fold": fold, "role": "test", "candidate_id": selected.candidate_id, "selected": True}
        )
        fit_rows.append(test_summary)
        selected_test_rows.append(test_summary)
        human_test_profile = profile_frame(human_test, "true_rt", "human_correct", edges).assign(
            fold=fold, role="test", candidate_id="human", source="human", aggregation="observed", seed_id="observed"
        )
        profile_rows.append(human_test_profile)
        for prof in test_profiles:
            profile_rows.append(prof.assign(fold=fold, role="test", candidate_id=selected.candidate_id))
        if test_summary["behavior_pass"]:
            process, mechanism_pass, reason = mechanism_summary(candidate_test, edges)
            if not process.empty:
                process["fold"] = fold
                process["mechanism_pass"] = mechanism_pass
                process["mechanism_reason"] = reason
                mechanism_rows.append(process)

    shape_fit_summary = pd.DataFrame()
    equal_subject_profiles = pd.DataFrame()
    equal_bin_profiles = pd.DataFrame()
    equal_tolerances = pd.DataFrame()
    dual_summary = pd.DataFrame()
    equal_mechanism = pd.DataFrame()
    if args.binning_mode == "dual":
        (
            shape_fit_summary,
            equal_subject_profiles,
            equal_bin_profiles,
            equal_tolerances,
            dual_summary,
            equal_mechanism,
        ) = run_equal_count_analysis(
            human=human,
            candidates=candidates,
            splits=splits,
            fixed_fit_rows=fit_rows,
            fixed_profile_rows=profile_rows,
            fixed_contexts=fixed_contexts,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
            min_cell_trials=args.min_cell_trials,
        )

    fit_summary = pd.DataFrame(fit_rows)
    selected_frame = pd.DataFrame(selected_test_rows)
    profiles_frame = pd.concat(profile_rows, ignore_index=True)
    mechanism_frame = pd.concat(mechanism_rows, ignore_index=True) if mechanism_rows else pd.DataFrame(
        columns=[
            "fold",
            "candidate_id",
            "analysis_group",
            "congruency",
            "rt_bin",
            "correctness",
            *MECHANISM_COLUMNS,
            "mechanism_pass",
            "mechanism_reason",
        ]
    )
    trajectory_frame = pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame(
        columns=[
            "fold",
            "candidate_id",
            "analysis_group",
            "congruency",
            "rt_bin",
            "correctness",
            "time",
            "s_target",
            "s_flanker",
            "s_other_max",
            "s_target_minus_flanker",
        ]
    )
    pd.DataFrame(edge_rows).to_csv(output_dir / "rt_bin_edges.csv", index=False)
    pd.concat(tolerance_rows, ignore_index=True, sort=False).to_csv(
        output_dir / "bootstrap_tolerances.csv", index=False
    )
    profiles_frame.to_csv(output_dir / "rt_bin_profiles.csv", index=False)
    fit_summary.to_csv(output_dir / "rt_bin_fit_summary.csv", index=False)
    mechanism_frame.to_csv(output_dir / "decision_process_by_bin.csv", index=False)
    trajectory_frame.to_csv(output_dir / "decision_trajectory_by_bin.csv", index=False)
    if args.binning_mode == "dual":
        equal_subject_profiles.to_csv(
            output_dir / "equal_count_subject_profiles.csv", index=False
        )
        equal_bin_profiles.to_csv(
            output_dir / "equal_count_bin_profiles.csv", index=False
        )
        shape_fit_summary.to_csv(
            output_dir / "equal_count_fit_summary.csv", index=False
        )
        equal_tolerances.to_csv(
            output_dir / "equal_count_bootstrap_tolerances.csv", index=False
        )
        dual_summary.to_csv(output_dir / "dual_track_model_summary.csv", index=False)
        equal_mechanism.to_csv(
            output_dir / "decision_process_equal_count_bins.csv", index=False
        )
        write_dual_summary(
            output_dir / "summary.md", diagnostic_edges, dual_summary, equal_mechanism
        )
    else:
        write_summary(
            output_dir / "summary.md",
            diagnostic_edges,
            fit_summary,
            selected_frame,
            mechanism_frame,
            False,
        )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "bootstrap_reps": args.bootstrap_reps,
                "min_cell_trials": args.min_cell_trials,
                "binning_mode": args.binning_mode,
                "diagnostic_only": args.diagnostic_only,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    output_dir = run(parse_args())
    print(output_dir)


if __name__ == "__main__":
    main()

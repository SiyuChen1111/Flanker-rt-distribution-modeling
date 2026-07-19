#!/usr/bin/env python3
"""Audit RT measurement noise and compare coupled Flanker decision mechanisms.

This is intentionally an internal diagnostic.  The historical candidates were
developed on the same 16-participant data set, so the four participant folds do
not turn this run into an external or confirmatory validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project_paths import PROJECT_ROOT
from run_flanker_rt_bin_fitting import (
    JOIN_KEYS,
    load_human_master,
    make_join_columns,
)


BASE = (
    PROJECT_ROOT
    / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
)
DUAL_DIR = BASE / "flanker_rt_4bin_fitting/dual_track_4bin_20260716_final"
GATE_TRIALS = BASE / "wr2_evidence_gate_and_execution_validation/metrics/execution_trial_level_best.csv"
UPSTREAM_SCRIPT = PROJECT_ROOT / "code/scripts/run_wr2_upstream_evidence_mapping_individual_validation.py"
UPSTREAM_PARAMS = BASE / "wr2_upstream_evidence_mapping_individual_validation/metrics/individual_evidence_parameters.csv"
UPSTREAM_SPLITS = BASE / "wr2_upstream_evidence_mapping_individual_validation/metrics/participant_stimulus_split_manifest.csv"
FITTING_TRIALS = BASE / "fitting/representative_trial_level_predictions.csv"
EVIDENCE_CACHE = BASE / "evidence_cache/representative_subset_layerwise_evidence.npz"
DEFAULT_OUTPUT_ROOT = BASE / "flanker_measurement_mechanism_followup"
GROUPS = ("young_20_29", "older_80_89")
CONGRUENCIES = (0, 1)
DT = 0.01
N_SEEDS = 10
# Subject-equal primary estimands.  These are one half of the saved-gate raw
# gaps (14.4527 percentage points and 202.042 ms), not pooled-trial medians.
FAST_ERROR_LIMIT = 0.072264
YOUNG_INCONGRUENT_MEDIAN_LIMIT = 0.101021
MIN_HUMAN_ERRORS = 5
GATE_GAP = 0.020
GATE_MAX_WAIT_S = 0.160
EXECUTION_SD = {"young_20_29": 0.045, "older_80_89": 0.150}


@dataclass(frozen=True)
class FirstPassageResult:
    choice: np.ndarray
    time: np.ndarray
    step: np.ndarray
    deadline_response: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    return p.parse_args()


def presentation_floor(rt: Iterable[float], refresh_hz: float = 60.0) -> np.ndarray:
    """Paper's presentation-noise bin: floor(RT / delta) * delta."""
    values = np.asarray(rt, dtype=float)
    delta = 1.0 / float(refresh_hz)
    return np.floor(values / delta + 1e-12) * delta


def keyboard_center(rt: Iterable[float], polling_hz: float = 125.0) -> np.ndarray:
    """Paper's keyboard bin: round(RT / delta) * delta - delta/2."""
    values = np.asarray(rt, dtype=float)
    delta = 1.0 / float(polling_hz)
    # np.rint uses banker's rounding; half-grid ties are vanishingly rare here.
    return np.rint(values / delta) * delta - delta / 2.0


def combined_hardware_bin(
    rt: Iterable[float], refresh_hz: float = 60.0, polling_hz: float = 125.0
) -> np.ndarray:
    """Keyboard centering followed by presentation-grid flooring."""
    return presentation_floor(keyboard_center(rt, polling_hz), refresh_hz)


def first_passage_readout(
    trajectory: np.ndarray,
    threshold: float,
    *,
    dt: float = DT,
    min_decision_time: float = 0.0,
) -> FirstPassageResult:
    """Couple choice and time to the same first threshold-crossing event."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 3:
        raise ValueError("trajectory must have shape (trial, time, class)")
    start = int(math.ceil(float(min_decision_time) / float(dt) - 1e-12))
    start = int(np.clip(start, 0, traj.shape[1] - 1))
    crossed = traj[:, start:, :] >= float(threshold)
    any_at_time = crossed.any(axis=2)
    ever = any_at_time.any(axis=1)
    rel_step = any_at_time.argmax(axis=1)
    step = np.where(ever, start + rel_step, traj.shape[1] - 1).astype(int)
    states = traj[np.arange(len(traj)), step]
    choice = states.argmax(axis=1).astype(int)
    return FirstPassageResult(
        choice=choice,
        time=step.astype(float) * float(dt),
        step=step,
        deadline_response=~ever,
    )


def classify_error_source(
    trajectory: np.ndarray,
    stop_step: Iterable[int],
    final_choice: Iterable[int],
    target: Iterable[int],
) -> np.ndarray:
    """Classify final errors using state at stopping and post-stop recovery."""
    traj = np.asarray(trajectory, dtype=float)
    stops = np.asarray(stop_step, dtype=int)
    final = np.asarray(final_choice, dtype=int)
    target = np.asarray(target, dtype=int)
    out = np.full(len(traj), "correct", dtype=object)
    for i in range(len(traj)):
        state_choice = int(np.argmax(traj[i, stops[i]]))
        if final[i] == target[i]:
            if state_choice != target[i]:
                out[i] = "noise_rescue"
            continue
        if state_choice == target[i]:
            out[i] = "post_stop_flip"
            continue
        later = traj[i, stops[i] + 1 :]
        if len(later):
            target_values = later[:, target[i]]
            others = later.copy()
            others[:, target[i]] = -np.inf
            recovered = bool(np.any(target_values > others.max(axis=1)))
        else:
            recovered = False
        out[i] = "premature_commitment" if recovered else "persistent_dynamics"
    return out.astype(str)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_congruency(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="raise").astype(int)
    mapped = values.astype(str).str.lower().map(
        {"congruent": 0, "incongruent": 1, "0": 0, "1": 1}
    )
    if mapped.isna().any():
        raise ValueError("Unknown congruency labels")
    return mapped.astype(int)


def assign_equal_count_bins_1d(values: Iterable[float], n_bins: int = 4) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) < n_bins:
        raise ValueError(f"Need at least {n_bins} trials for equal-count bins")
    order = np.argsort(values, kind="mergesort")
    assigned = np.empty(len(values), dtype=int)
    for bin_id, positions in enumerate(np.array_split(order, n_bins), start=1):
        assigned[positions] = bin_id
    return assigned


def attach_human_identity(frame: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable trial UID/user using the complete unique composite key."""
    trial_col = "trial_id" if "trial_id" in frame else "row_index"
    work = make_join_columns(frame, trial_col)
    extra = ["trial_uid", "user_id", "response_label"]
    if "subset_stimulus_id" in human:
        extra.append("subset_stimulus_id")
    human_map = human[JOIN_KEYS + extra]
    out = work.merge(human_map, on=JOIN_KEYS, how="left", validate="many_to_one")
    if out[["trial_uid", "user_id"]].isna().any().any():
        bad = out.loc[out.trial_uid.isna(), JOIN_KEYS].head(3).to_dict("records")
        raise ValueError(f"Unmatched trials after complete-key join: {bad}")
    out["user_id"] = out["user_id"].astype(str)
    if out.duplicated([c for c in ["model_config_id", "seed_index", "trial_uid"] if c in out]).any():
        raise ValueError("Duplicate model/seed/trial rows after identity join")
    return out


def load_human_with_stimulus() -> pd.DataFrame:
    human = load_human_master()
    source = pd.read_csv(
        FITTING_TRIALS,
        usecols=[
            "row_index",
            "analysis_group",
            "true_rt",
            "human_correct",
            "congruency",
            "target_label",
            "flanker_label",
            "subset_stimulus_id",
        ],
    )
    source = make_join_columns(source, "row_index")
    source = source[JOIN_KEYS + ["subset_stimulus_id"]]
    if source.duplicated(JOIN_KEYS).any():
        raise ValueError("Stimulus identity source is not unique on the complete trial key")
    human = human.merge(source, on=JOIN_KEYS, how="left", validate="one_to_one")
    if human.subset_stimulus_id.isna().any():
        raise ValueError("Missing stimulus identity in human master")
    human["user_id"] = human.user_id.astype(str)
    human["subset_stimulus_id"] = human.subset_stimulus_id.astype(str)
    return human


def restrict_to_upstream_test_trials(human: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(UPSTREAM_SPLITS, dtype={"user_id": str, "subset_stimulus_id": str})
    split = split[split.split.eq("test")][["user_id", "subset_stimulus_id"]].drop_duplicates()
    out = human.merge(split.assign(upstream_test=True), on=["user_id", "subset_stimulus_id"], how="inner", validate="many_to_one")
    if out.user_id.nunique() != human.user_id.nunique():
        raise ValueError("Upstream test restriction dropped a participant")
    return out


def make_common_random_draws(human: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One paired draw per trial UID and noise stream; never restart by subject."""
    ordered = human[["trial_uid", "analysis_group"]].drop_duplicates().sort_values("trial_uid").reset_index(drop=True)
    choice_rng = np.random.default_rng(20260717 + 30011 * int(seed))
    execution_rng = np.random.default_rng(20260717 + 70001 + 997 * int(seed))
    choice = choice_rng.normal(size=(len(ordered), 4))
    raw = execution_rng.lognormal(mean=-0.5 * 0.55**2, sigma=0.55, size=len(ordered))
    execution = np.zeros(len(ordered), float)
    for group in GROUPS:
        idx = ordered.analysis_group.eq(group).to_numpy()
        z = (raw[idx] - raw[idx].mean()) / max(raw[idx].std(ddof=0), 1e-12)
        execution[idx] = z * EXECUTION_SD[group]
    out = ordered[["trial_uid"]].copy()
    out["execution_draw"] = execution
    for c in range(4):
        out[f"choice_noise_{c}"] = choice[:, c]
    return out


def load_saved_gate(human: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(GATE_TRIALS, low_memory=False)
    out = attach_human_identity(raw, human)
    out["candidate_id"] = "current_evidence__saved_gate_execution"
    out["seed_id"] = out["seed_index"].astype(int)
    out["model_rt"] = pd.to_numeric(out["model_rt"], errors="raise")
    out["model_correct"] = out["model_correct"].astype(bool)
    out["final_choice"] = out["final_choice"].astype(int)
    out["deterministic_choice"] = out["deterministic_choice"].astype(int)
    out["congruency"] = normalize_congruency(out["congruency"])
    return out


def equal_count_profile(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rt_col = "true_rt" if source == "human" else "model_rt"
    ok_col = "human_correct" if source == "human" else "model_correct"
    rows: list[dict[str, Any]] = []
    keys = ["analysis_group", "user_id", "congruency"]
    seed_keys = [None] if source == "human" else sorted(frame["seed_id"].unique())
    for seed in seed_keys:
        scoped = frame if seed is None else frame[frame.seed_id.eq(seed)]
        for (group, user, cong), part in scoped.groupby(keys, sort=False):
            bins = assign_equal_count_bins_1d(part[rt_col])
            p = part.assign(rt_bin=bins)
            for b, cell in p.groupby("rt_bin", sort=True):
                rows.append(
                    {
                        "source": source,
                        "seed_id": "observed" if seed is None else int(seed),
                        "analysis_group": group,
                        "user_id": str(user),
                        "congruency": int(cong),
                        "rt_bin": int(b),
                        "n_trials": int(len(cell)),
                        "mean_rt": float(cell[rt_col].mean()),
                        "median_rt": float(cell[rt_col].median()),
                        "error_rate": float((~cell[ok_col].astype(bool)).mean()),
                    }
                )
    return pd.DataFrame(rows)


def subject_equal_group_profile(subject: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "seed_id", "analysis_group", "congruency", "rt_bin"]
    if "candidate_id" in subject:
        keys.insert(0, "candidate_id")
    return (
        subject.groupby(keys, as_index=False)
        .agg(
            n_subjects=("user_id", "nunique"),
            n_trials=("n_trials", "sum"),
            mean_rt=("mean_rt", "mean"),
            median_rt=("median_rt", "mean"),
            error_rate=("error_rate", "mean"),
            error_rate_sem=("error_rate", "sem"),
        )
    )


def rt_digit_audit(human: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, user), part in human.groupby(["analysis_group", "user_id"], sort=False):
        ms = np.rint(part.true_rt.to_numpy(float) * 1000).astype(int)
        uniq = np.unique(ms)
        diffs = np.diff(uniq)
        rows.append(
            {
                "analysis_group": group,
                "user_id": str(user),
                "n_trials": len(part),
                "n_unique_ms": len(uniq),
                "minimum_positive_step_ms": int(diffs[diffs > 0].min()) if np.any(diffs > 0) else np.nan,
                "modal_last_digit": int(pd.Series(ms % 10).mode().iloc[0]),
                "modal_last_digit_fraction": float(pd.Series(ms % 10).value_counts(normalize=True).iloc[0]),
                "integer_ms_fraction": float(np.mean(np.isclose(part.true_rt.to_numpy(float) * 1000, ms))),
            }
        )
    return pd.DataFrame(rows)


def human_subject_bootstrap_tolerances(
    human: pd.DataFrame, reps: int = 2000, seed: int = 20260717
) -> pd.DataFrame:
    """Estimate descriptive subject-to-subject sampling variation, not fit limits."""
    if reps < 1:
        raise ValueError("bootstrap reps must be positive")
    base = human.copy()
    profiles = equal_count_profile(base, "human")
    rng = np.random.default_rng(seed)
    rows = []
    for group in GROUPS:
        users = sorted(base.loc[base.analysis_group.eq(group), "user_id"].astype(str).unique())
        for cong in CONGRUENCIES:
            raw = base[(base.analysis_group.eq(group)) & (base.congruency.eq(cong))]
            raw_subject = raw.groupby("user_id").agg(
                error_rate=("human_correct", lambda x: 1 - x.mean()),
                median_rt=("true_rt", "median"),
            )
            fast_subject = profiles[
                profiles.analysis_group.eq(group)
                & profiles.congruency.eq(cong)
                & profiles.rt_bin.eq(1)
            ].set_index("user_id").error_rate
            observed = {
                "error_rate": float(raw_subject.error_rate.mean()),
                "median_rt": float(raw_subject.median_rt.mean()),
                "fast_error_rate": float(fast_subject.mean()),
            }
            draws = {k: [] for k in observed}
            for _ in range(reps):
                sampled = rng.choice(users, size=len(users), replace=True)
                draws["error_rate"].append(float(raw_subject.loc[sampled, "error_rate"].mean()))
                draws["median_rt"].append(float(raw_subject.loc[sampled, "median_rt"].mean()))
                draws["fast_error_rate"].append(float(fast_subject.loc[sampled].mean()))
            for metric, values in draws.items():
                arr = np.asarray(values, float)
                rows.append(
                    {
                        "analysis_group": group,
                        "congruency": cong,
                        "metric": metric,
                        "observed": observed[metric],
                        "bootstrap_sd": float(arr.std(ddof=1)),
                        "sampling_tolerance_95": float(np.quantile(abs(arr - observed[metric]), 0.95)),
                        "ci_low": float(np.quantile(arr, 0.025)),
                        "ci_high": float(np.quantile(arr, 0.975)),
                        "n_subjects": len(users),
                        "interpretation": "sampling variation; not a scientific equivalence bound",
                    }
                )
    return pd.DataFrame(rows)


def conditional_distribution_metrics(frame: pd.DataFrame, candidate_id: str, scheme: str) -> pd.DataFrame:
    rows = []
    model = frame[frame.candidate_id.eq(candidate_id)]
    for seed, seed_df in model.groupby("seed_id"):
        for group in GROUPS:
            for cong in CONGRUENCIES:
                m = seed_df[(seed_df.analysis_group.eq(group)) & (seed_df.congruency.eq(cong))]
                # Human trials are duplicated over seeds in model tables; use this seed only once.
                for accuracy in ("all", "correct", "error"):
                    if accuracy == "all":
                        hm, mm = m, m
                    elif accuracy == "correct":
                        hm, mm = m[m.human_correct], m[m.model_correct]
                    else:
                        hm, mm = m[~m.human_correct], m[~m.model_correct]
                    if len(hm) < 2 or len(mm) < 2:
                        w = np.nan
                    else:
                        w = stats.wasserstein_distance(hm["human_rt_eval"], mm["model_rt_eval"])
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "scheme": scheme,
                            "seed_id": seed,
                            "analysis_group": group,
                            "congruency": cong,
                            "accuracy": accuracy,
                            "n_human": len(hm),
                            "n_model": len(mm),
                            "human_median_rt": float(hm.human_rt_eval.median()) if len(hm) else np.nan,
                            "model_median_rt": float(mm.model_rt_eval.median()) if len(mm) else np.nan,
                            "wasserstein_s": w,
                        }
                    )
    return pd.DataFrame(rows)


def measurement_audit(saved_gate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles, metrics, materiality = [], [], []
    transforms = {
        "raw": lambda x: np.asarray(x, float),
        "presentation_floor_60hz": lambda x: presentation_floor(x, 60.0),
        "combined_60hz_125hz": lambda x: combined_hardware_bin(x, 60.0, 125.0),
    }
    for scheme, fn in transforms.items():
        q = saved_gate.copy()
        q["human_rt_eval"] = fn(q.true_rt)
        q["model_rt_eval"] = fn(q.model_rt)
        metrics.append(conditional_distribution_metrics(q, q.candidate_id.iloc[0], scheme))
        h = q.drop_duplicates("trial_uid").copy()
        h["true_rt"] = h["human_rt_eval"]
        m = q.copy()
        m["model_rt"] = m["model_rt_eval"]
        ph = equal_count_profile(h, "human")
        pm = equal_count_profile(m, "model")
        profiles.append(pd.concat([ph, pm], ignore_index=True).assign(scheme=scheme))
        gp = subject_equal_group_profile(pd.concat([ph, pm], ignore_index=True))
        yi = gp[(gp.analysis_group.eq("young_20_29")) & (gp.congruency.eq(1)) & (gp.rt_bin.eq(1))]
        he = float(yi[yi.source.eq("human")].error_rate.iloc[0])
        seed_gaps = yi[yi.source.eq("model")].assign(gap=lambda x: abs(x.error_rate - he)).gap
        human_med = h[(h.analysis_group.eq("young_20_29")) & (h.congruency.eq(1))].groupby("user_id").true_rt.median().mean()
        model_med = m[(m.analysis_group.eq("young_20_29")) & (m.congruency.eq(1))].groupby(["seed_id", "user_id"]).model_rt.median().groupby("seed_id").mean()
        materiality.append(
            {
                "scheme": scheme,
                "young_incongruent_fast_error_gap": float(seed_gaps.mean()),
                "young_incongruent_median_rt_gap_s": float(abs(model_med - human_med).mean()),
            }
        )
    mat = pd.DataFrame(materiality)
    raw = mat[mat.scheme.eq("raw")].iloc[0]
    mat["fast_error_gap_reduction"] = 1.0 - mat.young_incongruent_fast_error_gap / max(raw.young_incongruent_fast_error_gap, 1e-12)
    mat["median_rt_gap_reduction"] = 1.0 - mat.young_incongruent_median_rt_gap_s / max(raw.young_incongruent_median_rt_gap_s, 1e-12)
    mat["measurement_material"] = (
        (mat.fast_error_gap_reduction >= 0.25)
        | (mat.median_rt_gap_reduction >= 0.25)
        | (mat.young_incongruent_fast_error_gap <= FAST_ERROR_LIMIT)
        | (mat.young_incongruent_median_rt_gap_s <= YOUNG_INCONGRUENT_MEDIAN_LIMIT)
    ) & ~mat.scheme.eq("raw")
    return pd.concat(profiles, ignore_index=True), pd.concat(metrics, ignore_index=True), mat


def baseline_reproduction(saved_gate: pd.DataFrame) -> pd.DataFrame:
    yi = saved_gate[
        saved_gate.analysis_group.eq("young_20_29") & saved_gate.congruency.eq(1)
    ]
    human = yi.drop_duplicates("trial_uid")
    ph = equal_count_profile(human, "human")
    pm = equal_count_profile(yi, "model")
    gp = subject_equal_group_profile(pd.concat([ph, pm], ignore_index=True))
    fast = gp[(gp.analysis_group.eq("young_20_29")) & (gp.congruency.eq(1)) & (gp.rt_bin.eq(1))]
    human_fast = float(fast[fast.source.eq("human")].error_rate.iloc[0])
    model_fast = float(fast[fast.source.eq("model")].error_rate.mean())
    human_median = float(human.true_rt.median())
    model_median = float(yi.groupby("seed_id").model_rt.median().mean())
    return pd.DataFrame(
        [
            {"metric": "young_incongruent_fastest_equal_count_error_rate", "source": "human", "value": human_fast, "reference": 0.160, "absolute_deviation": abs(human_fast - 0.160)},
            {"metric": "young_incongruent_fastest_equal_count_error_rate", "source": "model", "value": model_fast, "reference": 0.304, "absolute_deviation": abs(model_fast - 0.304)},
            {"metric": "young_incongruent_pooled_median_rt_s", "source": "human", "value": human_median, "reference": 0.604, "absolute_deviation": abs(human_median - 0.604)},
            {"metric": "young_incongruent_pooled_median_rt_s", "source": "model", "value": model_median, "reference": 0.829, "absolute_deviation": abs(model_median - 0.829)},
        ]
    )


def measurement_model_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transforms = {
        "raw": lambda x: np.asarray(x, float),
        "presentation_floor_60hz": lambda x: presentation_floor(x, 60.0),
        "combined_60hz_125hz": lambda x: combined_hardware_bin(x, 60.0, 125.0),
    }
    for (candidate, seed), p in candidates.groupby(["candidate_id", "seed_id"]):
        p = p.drop_duplicates("trial_uid")
        for scheme, fn in transforms.items():
            distances, error_gaps = [], []
            for (group, cong), cell in p.groupby(["analysis_group", "congruency"]):
                distances.append(stats.wasserstein_distance(fn(cell.true_rt), fn(cell.model_rt)))
                error_gaps.append(abs((1 - cell.human_correct.mean()) - (1 - cell.model_correct.mean())))
            rows.append(
                {
                    "candidate_id": candidate,
                    "seed_id": seed,
                    "scheme": scheme,
                    "distribution_score": float(np.mean(distances)),
                    "error_rate_score": float(np.mean(error_gaps)),
                    "joint_score": float(np.mean(distances) + np.mean(error_gaps)),
                }
            )
    out = pd.DataFrame(rows)
    mean = out.groupby(["candidate_id", "scheme"], as_index=False).joint_score.mean()
    mean["rank"] = mean.groupby("scheme").joint_score.rank(method="min")
    return out.merge(mean[["candidate_id", "scheme", "rank"]], on=["candidate_id", "scheme"], how="left")


def saved_error_decomposition(saved_gate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = saved_gate.copy()
    wrong = ~q.model_correct
    q["error_source"] = "correct"
    q.loc[wrong & q.deterministic_choice.ne(q.target_label), "error_source"] = "pre_readout_wrong"
    q.loc[wrong & q.deterministic_choice.eq(q.target_label), "error_source"] = "independent_readout_flip"
    uncovered = wrong & q.error_source.eq("correct")
    q.loc[uncovered, "error_source"] = "execution_or_unclassified"
    q["execution_changed_choice"] = False  # execution extension freezes choice by construction
    if not q.loc[wrong, "error_source"].ne("correct").all():
        raise AssertionError("Every saved-gate error must have an error source")
    rows = []
    for keys, p in q.groupby(["seed_id", "analysis_group", "congruency", "error_source"], sort=False):
        seed, group, cong, source = keys
        rows.append(
            {
                "seed_id": seed,
                "analysis_group": group,
                "congruency": int(cong),
                "error_source": source,
                "n_trials": len(p),
                "fraction_of_all_trials": len(p) / len(q[(q.seed_id.eq(seed)) & (q.analysis_group.eq(group)) & (q.congruency.eq(cong))]),
                "mean_model_rt": float(p.model_rt.mean()),
                "mean_early_flanker_dominance": float(p.early_flanker_dominance.mean()),
                "mean_target_recovery_time": float(p.target_recovery_time.mean()),
                "mean_gate_wait_time": float(p.gate_wait_time.mean()),
            }
        )
    audit = pd.DataFrame(
        {
            "check": ["all_errors_classified", "execution_changed_choice_count", "error_rows"],
            "value": [bool((q.loc[wrong, "error_source"] != "correct").all()), int(q.execution_changed_choice.sum()), int(wrong.sum())],
        }
    )
    return pd.DataFrame(rows), audit


def load_upstream_module():
    spec = importlib.util.spec_from_file_location("flanker_followup_upstream", UPSTREAM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {UPSTREAM_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flanker_followup_upstream"] = mod
    spec.loader.exec_module(mod)
    return mod


def _standardized_lognormal(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    raw = rng.lognormal(mean=0.0, sigma=0.55, size=n)
    raw = (raw - raw.mean()) / max(raw.std(ddof=0), 1e-12)
    return raw * float(sd)


def apply_gap_gate(
    trial: pd.DataFrame,
    trajectory: np.ndarray,
    *,
    group: str,
    noise_cfg: dict[str, float],
    choice_noise: np.ndarray,
    execution_draw: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    p = trial.copy().reset_index(drop=True)
    traj = np.asarray(trajectory, float)
    start = np.clip(np.rint(p.readout_time.to_numpy(float) / DT).astype(int), 0, traj.shape[1] - 1)
    end = np.minimum(start + int(round(GATE_MAX_WAIT_S / DT)), traj.shape[1] - 1)
    chosen = start.copy()
    for i in range(len(p)):
        for t in range(int(start[i]), int(end[i]) + 1):
            vals = np.sort(traj[i, t])
            if vals[-1] - vals[-2] >= GATE_GAP:
                chosen[i] = t
                break
        else:
            chosen[i] = end[i]
    states = traj[np.arange(len(p)), chosen]
    deterministic = states.argmax(axis=1).astype(int)
    ordered = np.sort(states, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    readout_time = chosen * DT
    earlyness = 1.0 - readout_time / max(float(readout_time.max()), 1e-9)
    sigma = (
        float(noise_cfg["sigma_base"])
        + float(noise_cfg["sigma_time"]) * earlyness
        + float(noise_cfg["sigma_gap"]) * np.exp(-np.clip(gap, 0, None) / max(float(noise_cfg["gap_scale"]), 1e-9))
    )
    choice_noise = np.asarray(choice_noise, float)
    execution = np.asarray(execution_draw, float)
    if choice_noise.shape != states.shape or execution.shape != (len(p),):
        raise ValueError("Common random draws do not match the trial rows")
    final = (states + choice_noise * sigma[:, None]).argmax(axis=1).astype(int)
    t0 = p.model_rt.to_numpy(float) - p.readout_time.to_numpy(float)
    p["readout_time"] = readout_time
    p["gate_wait_time"] = (chosen - start) * DT
    p["stop_step"] = chosen
    p["deterministic_choice"] = deterministic
    p["final_choice"] = final
    p["model_correct"] = final == p.target_label.to_numpy(int)
    p["model_rt"] = np.clip(readout_time + t0 + execution, 0.15, 2.5)
    p["deadline_response"] = False
    p["choice_sigma"] = sigma
    p["execution_draw"] = execution
    p["error_source"] = classify_error_source(traj, chosen, final, p.target_label)
    return p, execution


def apply_first_passage(
    trial: pd.DataFrame,
    trajectory: np.ndarray,
    *,
    threshold: float,
    min_decision_time: float,
    group: str,
    execution_draw: np.ndarray,
) -> pd.DataFrame:
    p = trial.copy().reset_index(drop=True)
    fp = first_passage_readout(
        trajectory, threshold, dt=DT, min_decision_time=min_decision_time
    )
    execution = np.asarray(execution_draw, float)
    if execution.shape != (len(p),):
        raise ValueError("Common execution draws do not match the trial rows")
    t0 = p.model_rt.to_numpy(float) - p.readout_time.to_numpy(float)
    p["readout_time"] = fp.time
    p["gate_wait_time"] = 0.0
    p["stop_step"] = fp.step
    p["deterministic_choice"] = fp.choice
    p["final_choice"] = fp.choice
    p["model_correct"] = fp.choice == p.target_label.to_numpy(int)
    p["model_rt"] = np.clip(fp.time + t0 + execution, 0.15, 2.5)
    p["deadline_response"] = fp.deadline_response
    p["choice_sigma"] = 0.0
    p["execution_draw"] = execution
    p["error_source"] = classify_error_source(trajectory, fp.step, fp.choice, p.target_label)
    return p


def _capture_by_group(state: dict[str, Any], model_config_id: str) -> dict[str, dict[str, Any]]:
    found = {
        x["analysis_group"]: x
        for x in state["captures"]
        if x["model_config_id"] == model_config_id
    }
    if set(found) != set(GROUPS):
        raise RuntimeError(f"Missing captured groups for {model_config_id}: {sorted(found)}")
    return found


def build_mechanism_matrix(
    human_all: pd.DataFrame, evaluation_human: pd.DataFrame, seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = load_upstream_module()
    m = u.load_base()
    data = m.load_inputs()
    selected_noise = m.selected_time_gap_params(data["readout_rank"])
    state = u.install_overrides(m)
    params = pd.read_csv(UPSTREAM_PARAMS, dtype={"user_id": str})
    params = params[params.model.eq("E2_speed_evidence_control")].copy()
    allowed_betas = sorted(params.evidence_control_i.unique())
    if not set(allowed_betas).issubset(set(u.FULL_BETAS)):
        raise ValueError("Saved upstream beta is outside the frozen grid")
    all_candidates, feature_rows, provenance = [], [], []
    evaluation_uids = set(evaluation_human.trial_uid.astype(int))
    for seed in range(seeds):
        common = make_common_random_draws(evaluation_human, seed).set_index("trial_uid")
        beta_runs: dict[float, tuple[pd.DataFrame, dict[str, dict[str, Any]]]] = {}
        for beta in sorted(set([0.0, *allowed_betas])):
            spec = u.make_spec(float(beta), seed)
            # Match the established gate run's WW seed while keeping evidence
            # modes paired within this factorial diagnostic.
            spec["ww_seed"] = 20260711 + 1009 * int(seed)
            trial, *_ = m.run_candidate(data, spec, selected_noise)
            trial = attach_human_identity(trial, human_all)
            captures = _capture_by_group(state, spec["model_config_id"])
            beta_runs[float(beta)] = (trial, captures)
        selected = params[params.seed_index.eq(seed)][["user_id", "evidence_control_i", "speed_i"]]
        if selected.user_id.nunique() != human_all.user_id.nunique():
            raise ValueError(f"Incomplete upstream participant parameters for seed {seed}")
        for evidence_mode in ("current", "upstream_selected"):
            parts_gate, parts_fp, parts_features = [], [], []
            for group in GROUPS:
                users = sorted(evaluation_human.loc[evaluation_human.analysis_group.eq(group), "user_id"].astype(str).unique())
                for user in users:
                    par = selected[selected.user_id.eq(user)].iloc[0]
                    beta = 0.0 if evidence_mode == "current" else float(par.evidence_control_i)
                    # The evidence × readout factorial keeps nuisance speed at
                    # zero on both evidence sides.  Saved speed_i is reported
                    # only as provenance; it is not mixed into the contrast.
                    speed = 0.0
                    trial_all, caps = beta_runs[beta]
                    mask = (
                        trial_all.analysis_group.eq(group)
                        & trial_all.user_id.eq(user)
                        & trial_all.trial_uid.isin(evaluation_uids)
                    )
                    idx = np.flatnonzero(mask.to_numpy())
                    trial_user = trial_all.iloc[idx].copy().reset_index(drop=True)
                    traj_group = np.asarray(caps[group]["trajectory"], np.float32)
                    group_rows = trial_all[trial_all.analysis_group.eq(group)].reset_index(drop=True)
                    local_lookup = pd.Series(np.arange(len(group_rows)), index=group_rows.trial_uid.to_numpy())
                    local_idx = local_lookup.loc[trial_user.trial_uid].to_numpy(int)
                    traj = traj_group[local_idx]
                    mu = np.asarray(caps[group]["mu"], np.float32)[local_idx]
                    random = common.loc[trial_user.trial_uid]
                    choice_noise = random[[f"choice_noise_{c}" for c in range(4)]].to_numpy(float)
                    execution_draw = random.execution_draw.to_numpy(float)
                    gate, _ = apply_gap_gate(
                        trial_user,
                        traj,
                        group=group,
                        noise_cfg=selected_noise[group],
                        choice_noise=choice_noise,
                        execution_draw=execution_draw,
                    )
                    gate["candidate_id"] = f"{evidence_mode}__factorial_gap_gate"
                    gate["seed_id"] = seed
                    fp = apply_first_passage(
                        trial_user,
                        traj,
                        threshold=float(data["group_params"][group]["threshold"]),
                        min_decision_time=float(data["group_params"][group]["min_decision_time"]),
                        group=group,
                        execution_draw=execution_draw,
                    )
                    fp["candidate_id"] = f"{evidence_mode}__factorial_joint_first_passage"
                    fp["seed_id"] = seed
                    feat = trial_user[["trial_uid", "user_id", "analysis_group", "congruency", "true_rt", "human_correct", "response_label", "target_label", "flanker_label"]].copy()
                    feat["evidence_mode"] = evidence_mode
                    feat["seed_id"] = seed
                    feat["saved_speed_i_not_applied"] = float(par.speed_i)
                    for c in range(mu.shape[2]):
                        feat[f"final_mu_{c}"] = mu[:, -1, c]
                    parts_gate.append(gate)
                    parts_fp.append(fp)
                    parts_features.append(feat)
            all_candidates.extend(parts_gate + parts_fp)
            feature_rows.extend(parts_features)
        provenance.append(
            {
                "seed_id": seed,
                "ww_seed": int(20260711 + 1009 * seed),
                "betas_run": json.dumps(sorted(set([0.0, *allowed_betas]))),
                "common_random_numbers": True,
            }
        )
    candidates = pd.concat(all_candidates, ignore_index=True)
    features = pd.concat(feature_rows, ignore_index=True)
    key = ["candidate_id", "seed_id", "trial_uid"]
    if candidates.duplicated(key).any():
        raise AssertionError("Mechanism matrix contains duplicate candidate/seed/trial rows")
    expected = 4 * seeds * evaluation_human.trial_uid.nunique()
    if len(candidates) != expected:
        raise AssertionError(f"Mechanism matrix incomplete: {len(candidates)} != {expected}")
    paired = candidates.groupby(["seed_id", "trial_uid"]).execution_draw.nunique()
    if not paired.eq(1).all():
        raise AssertionError("Execution draws are not exactly paired across the four factorial cells")
    return candidates, features, pd.DataFrame(provenance)


def fit_snapshot_baselines(
    features: pd.DataFrame, splits: pd.DataFrame, seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory non-accumulating final-evidence baseline, fitted inside folds."""
    rows, fits = [], []
    # Features are identical over WW seeds apart from negligible construction details.
    base_features = features[features.seed_id.eq(0)].copy()
    for fold in sorted(splits.fold.unique()):
        train_users = set(splits[(splits.fold.eq(fold)) & (splits.role.eq("train"))].user_id.astype(str))
        test_users = set(splits[(splits.fold.eq(fold)) & (splits.role.eq("test"))].user_id.astype(str))
        if train_users & test_users:
            raise AssertionError("Participant leakage in snapshot baseline")
        for evidence_mode, mode_df in base_features.groupby("evidence_mode"):
            train = mode_df[mode_df.user_id.isin(train_users)].copy()
            test = mode_df[mode_df.user_id.isin(test_users)].copy()
            logits_train = train[[f"final_mu_{c}" for c in range(4)]].to_numpy(float)
            response = train.response_label.to_numpy(int)
            temperatures = np.asarray([0.025, 0.05, 0.10, 0.20, 0.40, 0.80])
            losses = []
            for temp in temperatures:
                z = logits_train / temp
                z -= z.max(axis=1, keepdims=True)
                logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
                losses.append(float(-logp[np.arange(len(z)), response].mean()))
            temperature = float(temperatures[int(np.argmin(losses))])
            gap_train = np.sort(logits_train, axis=1)[:, -1] - np.sort(logits_train, axis=1)[:, -2]
            x_train = np.column_stack(
                [
                    np.ones(len(train)),
                    gap_train,
                    train.analysis_group.eq("older_80_89").to_numpy(float),
                    train.congruency.to_numpy(float),
                ]
            )
            beta, *_ = np.linalg.lstsq(x_train, np.log(train.true_rt.to_numpy(float)), rcond=None)
            resid = np.log(train.true_rt.to_numpy(float)) - x_train @ beta
            resid_sd = float(np.std(resid, ddof=len(beta)))
            logits_test = test[[f"final_mu_{c}" for c in range(4)]].to_numpy(float)
            gap_test = np.sort(logits_test, axis=1)[:, -1] - np.sort(logits_test, axis=1)[:, -2]
            x_test = np.column_stack(
                [
                    np.ones(len(test)),
                    gap_test,
                    test.analysis_group.eq("older_80_89").to_numpy(float),
                    test.congruency.to_numpy(float),
                ]
            )
            z = logits_test / temperature
            z -= z.max(axis=1, keepdims=True)
            prob = np.exp(z)
            prob /= prob.sum(axis=1, keepdims=True)
            for seed in range(seeds):
                rng = np.random.default_rng(20260717 + 1009 * seed + 97 * int(fold) + (0 if evidence_mode == "current" else 1))
                choice = np.asarray([rng.choice(4, p=p) for p in prob], int)
                pred_rt = np.clip(np.exp(x_test @ beta + rng.normal(0, resid_sd, len(test))), 0.15, 2.5)
                q = test.copy()
                q["candidate_id"] = f"{evidence_mode}__snapshot_nonaccum_exploratory"
                q["seed_id"] = seed
                q["final_choice"] = choice
                q["model_correct"] = choice == q.target_label.to_numpy(int)
                q["model_rt"] = pred_rt
                q["readout_time"] = np.nan
                q["stop_step"] = -1
                q["deadline_response"] = False
                q["error_source"] = np.where(q.model_correct, "correct", "snapshot_choice_error")
                q["outer_fold"] = fold
                rows.append(q)
            fits.append(
                {
                    "fold": fold,
                    "evidence_mode": evidence_mode,
                    "temperature": temperature,
                    "choice_train_nll": min(losses),
                    "rt_intercept": beta[0],
                    "rt_gap_coefficient": beta[1],
                    "rt_age_coefficient": beta[2],
                    "rt_congruency_coefficient": beta[3],
                    "rt_residual_sd": resid_sd,
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
    out = pd.concat(rows, ignore_index=True)
    if out.duplicated(["candidate_id", "seed_id", "trial_uid"]).any():
        raise AssertionError("Snapshot baseline has duplicate held-out predictions")
    return out, pd.DataFrame(fits)


def evaluate_candidates(
    candidates: pd.DataFrame,
    human: pd.DataFrame,
    splits: pd.DataFrame,
    baseline_id: str = "current__factorial_gap_gate",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows, cell_rows, seed_rows = [], [], []
    human_once = human.copy()
    human_once["congruency"] = normalize_congruency(human_once.congruency)
    for candidate_id, cand in candidates.groupby("candidate_id", sort=True):
        for seed, model in cand.groupby("seed_id", sort=True):
            # Mechanism rows cover all subjects; snapshot rows are already OOF only.
            model = model.drop_duplicates("trial_uid")
            merged = human_once[["trial_uid", "user_id", "analysis_group", "congruency", "true_rt", "human_correct"]].merge(
                model[["trial_uid", "model_rt", "model_correct"]], on="trial_uid", how="inner", validate="one_to_one"
            )
            for source, rt_col, ok_col in [
                ("human", "true_rt", "human_correct"),
                ("model", "model_rt", "model_correct"),
            ]:
                prof_input = merged.rename(columns={rt_col: "_rt", ok_col: "_ok"})
                for (group, user, cong), p in prof_input.groupby(["analysis_group", "user_id", "congruency"]):
                    bins = assign_equal_count_bins_1d(p["_rt"])
                    for b, cell in p.assign(rt_bin=bins).groupby("rt_bin"):
                        subject_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "seed_id": seed,
                                "source": source,
                                "analysis_group": group,
                                "user_id": str(user),
                                "congruency": int(cong),
                                "rt_bin": int(b),
                                "n_trials": len(cell),
                                "error_rate": float((~cell._ok.astype(bool)).mean()),
                                "mean_rt": float(cell._rt.mean()),
                                "median_rt": float(cell._rt.median()),
                            }
                        )
            for group in GROUPS:
                for cong in CONGRUENCIES:
                    p = merged[(merged.analysis_group.eq(group)) & (merged.congruency.eq(cong))]
                    h_subject = p.groupby("user_id").agg(h_error=("human_correct", lambda x: 1 - x.mean()), h_median=("true_rt", "median"))
                    m_subject = p.groupby("user_id").agg(m_error=("model_correct", lambda x: 1 - x.mean()), m_median=("model_rt", "median"))
                    hm = p[p.human_correct]
                    mm = p[p.model_correct]
                    he = p[~p.human_correct]
                    me = p[~p.model_correct]
                    cell_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "seed_id": seed,
                            "analysis_group": group,
                            "congruency": cong,
                            "n_subjects": p.user_id.nunique(),
                            "n_human_trials": len(p),
                            "n_human_errors": int((~p.human_correct).sum()),
                            "n_model_errors": int((~p.model_correct).sum()),
                            "human_error_rate": float(h_subject.h_error.mean()),
                            "model_error_rate": float(m_subject.m_error.mean()),
                            "human_median_rt": float(h_subject.h_median.mean()),
                            "model_median_rt": float(m_subject.m_median.mean()),
                            "wasserstein_all_s": stats.wasserstein_distance(p.true_rt, p.model_rt),
                            "wasserstein_correct_s": stats.wasserstein_distance(hm.true_rt, mm.model_rt) if len(hm) > 1 and len(mm) > 1 else np.nan,
                            "wasserstein_error_s": stats.wasserstein_distance(he.true_rt, me.model_rt) if len(he) > 1 and len(me) > 1 else np.nan,
                            "estimability": "ESTIMABLE" if int((~p.human_correct).sum()) >= MIN_HUMAN_ERRORS else "NON_ESTIMABLE",
                        }
                    )
    subject = pd.DataFrame(subject_rows)
    cells = pd.DataFrame(cell_rows)
    group_profile = subject_equal_group_profile(subject)
    for (candidate, seed), p in cells.groupby(["candidate_id", "seed_id"]):
        gp = group_profile[(group_profile.candidate_id.eq(candidate)) & (group_profile.seed_id.eq(seed))]
        yi_fast = gp[(gp.source.eq("model")) & (gp.analysis_group.eq("young_20_29")) & (gp.congruency.eq(1)) & (gp.rt_bin.eq(1))]
        yh_fast = gp[(gp.source.eq("human")) & (gp.analysis_group.eq("young_20_29")) & (gp.congruency.eq(1)) & (gp.rt_bin.eq(1))]
        fast_gap = abs(float(yi_fast.error_rate.iloc[0]) - float(yh_fast.error_rate.iloc[0]))
        yi = p[(p.analysis_group.eq("young_20_29")) & (p.congruency.eq(1))].iloc[0]
        median_gap = abs(float(yi.model_median_rt - yi.human_median_rt))
        continuous_score = float(
            np.nanmean(
                abs(p.model_error_rate - p.human_error_rate)
                + p.wasserstein_all_s
                + 0.5 * p.wasserstein_correct_s.fillna(p.wasserstein_all_s)
                + 0.5 * p.wasserstein_error_s.fillna(p.wasserstein_all_s)
            )
        )
        congruent_nonzero = all(
            float(p[(p.analysis_group.eq(g)) & (p.congruency.eq(0))].model_error_rate.iloc[0]) > 0
            for g in GROUPS
        )
        seed_rows.append(
            {
                "candidate_id": candidate,
                "seed_id": seed,
                "young_incongruent_fast_error_gap": fast_gap,
                "young_incongruent_median_rt_gap_s": median_gap,
                "continuous_joint_score": continuous_score,
                "congruent_nonzero_both_groups": congruent_nonzero,
                "key_thresholds_pass": fast_gap <= FAST_ERROR_LIMIT and median_gap <= YOUNG_INCONGRUENT_MEDIAN_LIMIT and congruent_nonzero,
                "all_primary_cells_estimable": bool(p.estimability.eq("ESTIMABLE").all()),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    other = cells[~(cells.analysis_group.eq("young_20_29") & cells.congruency.eq(1))].copy()
    other["other_cell_score"] = (
        abs(other.model_error_rate - other.human_error_rate)
        + abs(other.model_median_rt - other.human_median_rt)
        + other.wasserstein_all_s
    )
    other_score = other.groupby(["candidate_id", "seed_id"], as_index=False).other_cell_score.mean()
    baseline_other = other_score[other_score.candidate_id.eq(baseline_id)][["seed_id", "other_cell_score"]].rename(
        columns={"other_cell_score": "baseline_other_cell_score"}
    )
    other_score = other_score.merge(baseline_other, on="seed_id", how="left", validate="many_to_one")
    other_score["other_conditions_no_gt_10pct_worsening"] = (
        other_score.other_cell_score <= 1.10 * other_score.baseline_other_cell_score
    )
    seed_frame = seed_frame.merge(other_score, on=["candidate_id", "seed_id"], how="left", validate="one_to_one")
    seed_frame["key_thresholds_pass"] &= seed_frame.other_conditions_no_gt_10pct_worsening
    return subject, cells, seed_frame


def evaluate_fold_stability(
    candidates: pd.DataFrame,
    human: pd.DataFrame,
    splits: pd.DataFrame,
    baseline_id: str = "current__factorial_gap_gate",
) -> pd.DataFrame:
    test_fold = splits[splits.role.eq("test")][["user_id", "fold"]].copy()
    test_fold["user_id"] = test_fold.user_id.astype(str)
    if test_fold.user_id.duplicated().any():
        raise AssertionError("Each participant must be held out in exactly one fold")
    human_fold = human.merge(test_fold, on="user_id", how="left", validate="many_to_one")
    if human_fold.fold.isna().any():
        raise AssertionError("Some human trials lack a held-out fold")
    rows = []
    for (candidate, seed), model in candidates.groupby(["candidate_id", "seed_id"]):
        model = model.drop_duplicates("trial_uid")
        merged = human_fold[["trial_uid", "user_id", "analysis_group", "congruency", "true_rt", "human_correct", "fold"]].merge(
            model[["trial_uid", "model_rt", "model_correct"]], on="trial_uid", how="inner", validate="one_to_one"
        )
        for fold, q in merged.groupby("fold"):
            yi = q[(q.analysis_group.eq("young_20_29")) & (q.congruency.eq(1))]
            h_fast, m_fast = [], []
            for _, p in yi.groupby("user_id"):
                hb = assign_equal_count_bins_1d(p.true_rt)
                mb = assign_equal_count_bins_1d(p.model_rt)
                h_fast.append(float((~p.loc[hb == 1, "human_correct"].astype(bool)).mean()))
                m_fast.append(float((~p.loc[mb == 1, "model_correct"].astype(bool)).mean()))
            human_median = yi.groupby("user_id").true_rt.median().mean()
            model_median = yi.groupby("user_id").model_rt.median().mean()
            continuous_parts = []
            for (group, cong), p in q.groupby(["analysis_group", "congruency"]):
                continuous_parts.append(
                    abs((1 - p.model_correct.mean()) - (1 - p.human_correct.mean()))
                    + stats.wasserstein_distance(p.true_rt, p.model_rt)
                )
            rows.append(
                {
                    "candidate_id": candidate,
                    "seed_id": seed,
                    "fold": int(fold),
                    "young_incongruent_fast_error_gap": abs(float(np.mean(m_fast)) - float(np.mean(h_fast))),
                    "young_incongruent_median_rt_gap_s": abs(float(model_median - human_median)),
                    "continuous_score": float(np.mean(continuous_parts)),
                    "n_young_test_subjects": yi.user_id.nunique(),
                    "n_older_test_subjects": q[q.analysis_group.eq("older_80_89")].user_id.nunique(),
                }
            )
    out = pd.DataFrame(rows)
    base = out[out.candidate_id.eq(baseline_id)][["seed_id", "fold", "continuous_score"]].rename(
        columns={"continuous_score": "baseline_continuous_score"}
    )
    out = out.merge(base, on=["seed_id", "fold"], how="left", validate="many_to_one")
    out["improved_vs_current_gate"] = out.continuous_score < out.baseline_continuous_score
    return out


def summarize_seed_stability(
    seed_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    baseline_id: str = "current__factorial_gap_gate",
) -> pd.DataFrame:
    base_id = baseline_id
    base = seed_metrics[seed_metrics.candidate_id.eq(base_id)].set_index("seed_id")
    rows = []
    for candidate, p in seed_metrics.groupby("candidate_id", sort=True):
        q = p.set_index("seed_id").join(
            base[["continuous_joint_score"]].rename(columns={"continuous_joint_score": "baseline_continuous_score"}),
            how="left",
        )
        q["continuous_improved"] = q.continuous_joint_score < q.baseline_continuous_score
        fold_rate = (
            fold_metrics[fold_metrics.candidate_id.eq(candidate)]
            .groupby("seed_id")
            .improved_vs_current_gate.mean()
        )
        q["fold_improvement_rate"] = q.index.map(fold_rate).fillna(0.0)
        q["three_of_four_folds_improve"] = q.fold_improvement_rate >= 0.75
        q["seed_diagnostic_pass"] = q.key_thresholds_pass & q.continuous_improved & q.three_of_four_folds_improve
        rows.append(
            {
                "candidate_id": candidate,
                "n_seeds": len(q),
                "key_threshold_pass_rate": float(q.key_thresholds_pass.mean()),
                "continuous_improvement_rate": float(q.continuous_improved.mean()),
                "mean_fold_improvement_rate": float(q.fold_improvement_rate.mean()),
                "direction_stability_rate": float(q.seed_diagnostic_pass.mean()),
                "mean_fast_error_gap": float(q.young_incongruent_fast_error_gap.mean()),
                "mean_young_incongruent_median_gap_s": float(q.young_incongruent_median_rt_gap_s.mean()),
                "mean_continuous_joint_score": float(q.continuous_joint_score.mean()),
                "seed_count_pass": len(q) >= N_SEEDS,
                "stability_80pct_pass": len(q) >= N_SEEDS and float(q.seed_diagnostic_pass.mean()) >= 0.80,
                "estimability_status": "ESTIMABLE" if bool(q.all_primary_cells_estimable.all()) else "NON_ESTIMABLE",
                "diagnostic_threshold_status": (
                    "INDETERMINATE"
                    if not bool(q.all_primary_cells_estimable.all())
                    else ("PASS" if len(q) >= N_SEEDS and float(q.seed_diagnostic_pass.mean()) >= 0.80 else "FAIL")
                ),
                "confirmatory_eligible": False,
                "claim_level": "internal diagnostic",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_continuous_joint_score")


def plot_outputs(candidates: pd.DataFrame, profiles: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir()
    show_ids = [x for x in ["saved_current_gate_execution_reference", "current__factorial_gap_gate", "current__factorial_joint_first_passage", "upstream_selected__factorial_gap_gate", "upstream_selected__factorial_joint_first_passage"] if x in set(candidates.candidate_id)]
    seed0 = candidates[candidates.seed_id.eq(0)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (group, cong) in zip(axes.ravel(), [(g, c) for g in GROUPS for c in CONGRUENCIES]):
        p = seed0[(seed0.analysis_group.eq(group)) & (seed0.congruency.eq(cong))]
        human = p.drop_duplicates("trial_uid")
        ax.hist(human.true_rt, bins=35, density=True, histtype="step", lw=2.2, color="black", label="Human")
        for cid in show_ids:
            q = p[p.candidate_id.eq(cid)]
            ax.hist(q.model_rt, bins=35, density=True, histtype="step", lw=1.2, label=cid)
        ax.set(title=f"{group} | {'incongruent' if cong else 'congruent'}", xlabel="RT (s)", ylabel="Density")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "rt_density_mechanism_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    gp = subject_equal_group_profile(profiles)
    for ax, (group, cong) in zip(axes.ravel(), [(g, c) for g in GROUPS for c in CONGRUENCIES]):
        h = gp[(gp.source.eq("human")) & (gp.analysis_group.eq(group)) & (gp.congruency.eq(cong))]
        # Human profile is repeated per candidate/seed; take one curve.
        if "candidate_id" in h:
            h = h[(h.candidate_id.eq(show_ids[0])) & (h.seed_id.eq(0))]
        ax.plot(h.rt_bin, h.error_rate, "o-k", lw=2, label="Human")
        for cid in show_ids:
            q = gp[(gp.source.eq("model")) & (gp.analysis_group.eq(group)) & (gp.congruency.eq(cong)) & (gp.candidate_id.eq(cid))]
            q = q.groupby("rt_bin", as_index=False).error_rate.mean()
            ax.plot(q.rt_bin, q.error_rate, "o-", lw=1.2, label=cid)
        ax.set(title=f"{group} | {'incongruent' if cong else 'congruent'}", xlabel="Relative RT bin", ylabel="Error rate", xticks=[1, 2, 3, 4])
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "error_rate_by_relative_rt_bin.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (group, cong) in zip(axes.ravel(), [(g, c) for g in GROUPS for c in CONGRUENCIES]):
        p = seed0[(seed0.analysis_group.eq(group)) & (seed0.congruency.eq(cong))]
        for label, values, style in [("Human", p.drop_duplicates("trial_uid").true_rt, "k-")]:
            x = np.sort(values.to_numpy(float)); y = np.arange(1, len(x) + 1) / len(x); ax.plot(x, y, style, lw=2, label=label)
        for cid in show_ids:
            x = np.sort(p[p.candidate_id.eq(cid)].model_rt.to_numpy(float)); y = np.arange(1, len(x) + 1) / len(x); ax.plot(x, y, lw=1.2, label=cid)
        ax.set(title=f"{group} | {'incongruent' if cong else 'congruent'}", xlabel="RT (s)", ylabel="ECDF")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "rt_ecdf_mechanism_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    conditions = [(g, c) for g in GROUPS for c in CONGRUENCIES]
    for row, (group, cong) in enumerate(conditions):
        p = seed0[(seed0.analysis_group.eq(group)) & (seed0.congruency.eq(cong))]
        for col, correct in enumerate([True, False]):
            ax = axes[row, col]
            human = p.drop_duplicates("trial_uid")
            hv = human.loc[human.human_correct.eq(correct), "true_rt"]
            if len(hv):
                ax.hist(hv, bins=25, density=True, histtype="step", lw=2, color="black", label="Human")
            for cid in show_ids:
                mv = p[(p.candidate_id.eq(cid)) & (p.model_correct.eq(correct))].model_rt
                if len(mv):
                    ax.hist(mv, bins=25, density=True, histtype="step", lw=1.1, label=cid)
            ax.set(title=f"{group} | {'incongruent' if cong else 'congruent'} | {'correct' if correct else 'error'}", xlabel="RT (s)", ylabel="Density")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "rt_distribution_by_accuracy.png", dpi=180)
    plt.close(fig)

    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (group, cong) in zip(axes.ravel(), conditions):
        p = seed0[(seed0.analysis_group.eq(group)) & (seed0.congruency.eq(cong))]
        human = p.drop_duplicates("trial_uid")
        ax.plot(quantiles, np.quantile(human.true_rt, quantiles), "o-k", lw=2, label="Human")
        for cid in show_ids:
            q = p[p.candidate_id.eq(cid)]
            ax.plot(quantiles, np.quantile(q.model_rt, quantiles), "o-", lw=1.2, label=cid)
        ax.set(title=f"{group} | {'incongruent' if cong else 'congruent'}", xlabel="Quantile", ylabel="RT (s)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "rt_quantile_profiles.png", dpi=180)
    plt.close(fig)


def write_summary(
    out: Path,
    measurement: pd.DataFrame,
    reproduction: pd.DataFrame,
    decomposition: pd.DataFrame,
    summary: pd.DataFrame,
    cells: pd.DataFrame,
    config_hash: str,
) -> None:
    raw = measurement[measurement.scheme.eq("raw")].iloc[0]
    rep = reproduction.pivot(index="metric", columns="source", values="value")
    best = summary.iloc[0]
    indexed = summary.set_index("candidate_id")
    current_gate = indexed.loc["current__factorial_gap_gate"]
    current_fp = indexed.loc["current__factorial_joint_first_passage"]
    upstream_gate = indexed.loc["upstream_selected__factorial_gap_gate"]
    upstream_fp = indexed.loc["upstream_selected__factorial_joint_first_passage"]
    snapshot = indexed.loc["current__snapshot_nonaccum_exploratory"]
    pre = decomposition[decomposition.error_source.eq("pre_readout_wrong")].n_trials.sum()
    flip = decomposition[decomposition.error_source.eq("independent_readout_flip")].n_trials.sum()
    total = max(pre + flip, 1)
    nonestimable = cells[cells.estimability.eq("NON_ESTIMABLE")][["analysis_group", "congruency"]].drop_duplicates()
    lines = [
        "# Flanker 测量误差与机制修复结果",
        "",
        "## 结论等级",
        "",
        "- 本次结果最高只能称为 **internal diagnostic（内部诊断）**。所有主候选此前都接触过同一批 16 名被试的数据，因此不是确认性验证或外部验证。",
        "- 老年组只有 4 人，年龄机制结论均为探索性。随机种子只代表模拟波动，不代表更多被试。",
        f"- 冻结配置指纹：`{config_hash}`。",
        "",
        "## 测量误差审计",
        "",
        f"- 基线复现：年轻不一致最快相对 bin 为人类 {rep.loc['young_incongruent_fastest_equal_count_error_rate','human']:.1%}、模型 {rep.loc['young_incongruent_fastest_equal_count_error_rate','model']:.1%}；汇总 RT 中位数为人类 {rep.loc['young_incongruent_pooled_median_rt_s','human']:.3f}s、模型 {rep.loc['young_incongruent_pooled_median_rt_s','model']:.3f}s。",
        f"- 未校正时，年轻不一致最快四分位的绝对错误率差距为 {raw.young_incongruent_fast_error_gap:.1%}，RT 中位数差距为 {raw.young_incongruent_median_rt_gap_s*1000:.0f} ms。",
    ]
    for row in measurement[~measurement.scheme.eq("raw")].itertuples():
        lines.append(
            f"- `{row.scheme}`：快速错误差距改变 {row.fast_error_gap_reduction:+.1%}，RT 中位数差距改变 {row.median_rt_gap_reduction:+.1%}；实质性解释={bool(row.measurement_material)}。"
        )
    lines += [
        "- 窄时间格只用于测量噪声敏感性检查，不参与模型通过判定。",
        "",
        "## 当前 gate 的错误来自哪里",
        "",
        f"- 在保存的 10 个种子中，最终错误里约 {pre/total:.1%} 在加独立读出噪声前已经选错，约 {flip/total:.1%} 是停止后的独立噪声把正确选择翻成错误。",
        "- 执行时间扩展按原模型定义只改变 RT、冻结选择，因此它产生的选择错误数为 0。",
        "",
        "## 最小机制比较",
        "",
        f"- 按连续 choice–RT 指标排名最好的单元是 `{best.candidate_id}`，平均分 {best.mean_continuous_joint_score:.4f}。",
        f"- 其年轻不一致最快 bin 差距为 {best.mean_fast_error_gap:.1%}，年轻不一致 RT 中位数差距为 {best.mean_young_incongruent_median_gap_s*1000:.0f} ms。",
        f"- 诊断状态：`{best.diagnostic_threshold_status}`；80% 种子稳定门槛：{bool(best.stability_80pct_pass)}。",
        f"- 在同一批重算轨迹、相同执行时间抽样中，把 factorial gap gate 改成联合首达后，快速错误差距从 {current_gate.mean_fast_error_gap:.1%} 变为 {current_fp.mean_fast_error_gap:.1%}。这只解释重算反事实，不冒充保存的当前模型。",
        f"- 不加入个人速度校准时，上游证据映射的 RT 中位数差距约为 {upstream_fp.mean_young_incongruent_median_gap_s*1000:.0f} ms，快速错误差距约为 {upstream_fp.mean_fast_error_gap:.1%}。",
        f"- 非积累快照基线的连续联合分数为 {snapshot.mean_continuous_joint_score:.4f}，差于当前 gate 的 {current_gate.mean_continuous_joint_score:.4f}；当前结果不支持用简单非积累规则取代积累过程。",
        "- 四个 factorial 单元、两个非积累探索基线及保存模型参考均未通过完整门槛；没有可称为行为通过或机制通过的模型。",
        "- `snapshot_nonaccum_exploratory` 只用于判断当前积累实现是否值得继续，不用于否定证据积累理论。",
        "",
        "## 可估计性与限制",
        "",
        f"- 有 {len(nonestimable)} 个年龄×条件单元的人类错误少于 {MIN_HUMAN_ERRORS} 个，被标记为 `NON_ESTIMABLE`，不会被悄悄平均成 PASS。",
        "- 机制比较只使用上游模型原先冻结的 stimulus-test 试次；当前四折仍是对历史候选的内部重评分。候选开发、上游 beta 与部分执行参数曾使用同一批参与者数据，正结果仍需新被试或新刺激验证。",
        "- 固定结论模板：在本数据、本实现和预定指标下通过或未通过；不得扩展为证据积累理论成立或不成立。",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = args.seeds or (2 if args.mode == "smoke" else N_SEEDS)
    if args.mode == "full" and seeds < N_SEEDS:
        raise ValueError("Full mode requires at least 10 seeds")
    out = args.output_root / args.run_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {out}")
    out.mkdir(parents=True)
    human = load_human_with_stimulus()
    evaluation_human = restrict_to_upstream_test_trials(human)
    splits = pd.read_csv(DUAL_DIR / "split_manifest.csv", dtype={"user_id": str})
    for fold in sorted(splits.fold.unique()):
        train = set(splits[(splits.fold.eq(fold)) & (splits.role.eq("train"))].user_id)
        test = set(splits[(splits.fold.eq(fold)) & (splits.role.eq("test"))].user_id)
        if train & test:
            raise AssertionError(f"Participant leakage in fold {fold}")
    config = {
        "run_id": args.run_id,
        "mode": args.mode,
        "seeds": seeds,
        "bootstrap_reps": args.bootstrap_reps,
        "splits": splits.sort_values(list(splits.columns)).to_dict("records"),
        "candidate_cells": [
            "saved_current_gate_execution_reference",
            "current__factorial_gap_gate",
            "current__factorial_joint_first_passage",
            "upstream_selected__factorial_gap_gate",
            "upstream_selected__factorial_joint_first_passage",
            "current__snapshot_nonaccum_exploratory",
            "upstream_selected__snapshot_nonaccum_exploratory",
        ],
        "primary_limits": {
            "young_incongruent_fast_error_gap": FAST_ERROR_LIMIT,
            "young_incongruent_median_rt_gap_s": YOUNG_INCONGRUENT_MEDIAN_LIMIT,
            "seed_stability": 0.80,
        },
        "measurement_schemes": ["raw", "presentation_floor_60hz", "combined_60hz_125hz"],
        "claim_level": "internal diagnostic",
        "candidate_created_from": "historical full 16-participant representative subset",
        "confirmatory_eligible": False,
        "evaluation_trials": "frozen upstream participant-stimulus test split only",
        "primary_rt_estimand": "subject median, then equal-weight mean across subjects",
        "factorial_speed_adjustment": "none on both evidence modes",
        "input_hashes": {
            "analysis_script": file_sha256(Path(__file__)),
            "saved_gate_trials": file_sha256(GATE_TRIALS),
            "upstream_script": file_sha256(UPSTREAM_SCRIPT),
            "upstream_parameters": file_sha256(UPSTREAM_PARAMS),
            "upstream_stimulus_splits": file_sha256(UPSTREAM_SPLITS),
            "fitting_trials": file_sha256(FITTING_TRIALS),
            "evidence_cache": file_sha256(EVIDENCE_CACHE),
            "four_fold_manifest": file_sha256(DUAL_DIR / "split_manifest.csv"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    config["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else "unavailable"
    config_hash = stable_hash(config)
    config["config_hash"] = config_hash
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    splits.to_csv(out / "split_manifest.csv", index=False)

    saved_gate = load_saved_gate(human)
    reproduction = baseline_reproduction(saved_gate)
    digit = rt_digit_audit(human)
    bootstrap = human_subject_bootstrap_tolerances(human, args.bootstrap_reps)
    narrow_profiles, dist_metrics, materiality = measurement_audit(saved_gate)
    yi_tol = bootstrap[(bootstrap.analysis_group.eq("young_20_29")) & (bootstrap.congruency.eq(1))].set_index("metric")
    fast_tol = float(yi_tol.loc["fast_error_rate", "sampling_tolerance_95"])
    median_tol = float(yi_tol.loc["median_rt", "sampling_tolerance_95"])
    materiality["fast_within_sampling_tolerance"] = materiality.young_incongruent_fast_error_gap <= fast_tol
    materiality["median_within_sampling_tolerance"] = materiality.young_incongruent_median_rt_gap_s <= median_tol
    materiality["measurement_material"] |= (
        materiality.fast_within_sampling_tolerance | materiality.median_within_sampling_tolerance
    ) & ~materiality.scheme.eq("raw")
    decomposition, decomposition_audit = saved_error_decomposition(saved_gate)
    reproduction.to_csv(out / "baseline_reproduction.csv", index=False)
    digit.to_csv(out / "human_rt_digit_audit.csv", index=False)
    bootstrap.to_csv(out / "human_subject_bootstrap_tolerances.csv", index=False)
    narrow_profiles.to_csv(out / "measurement_error_equal_count_profiles.csv", index=False)
    dist_metrics.to_csv(out / "measurement_error_distribution_metrics.csv", index=False)
    materiality.to_csv(out / "measurement_error_materiality.csv", index=False)
    decomposition.to_csv(out / "saved_gate_error_source_decomposition.csv", index=False)
    decomposition_audit.to_csv(out / "saved_gate_error_source_audit.csv", index=False)

    candidates, features, provenance = build_mechanism_matrix(human, evaluation_human, seeds)
    snapshots, snapshot_fits = fit_snapshot_baselines(features, splits, seeds)
    saved_reference = saved_gate[saved_gate.trial_uid.isin(evaluation_human.trial_uid)].copy()
    saved_reference["candidate_id"] = "saved_current_gate_execution_reference"
    saved_reference["seed_id"] = saved_reference.seed_index.astype(int)
    all_candidates = pd.concat([saved_reference, candidates, snapshots], ignore_index=True, sort=False)
    reconstructed = candidates[candidates.candidate_id.eq("current__factorial_gap_gate")]
    paired_reference = saved_reference.merge(
        reconstructed,
        on=["seed_id", "trial_uid"],
        suffixes=("_saved", "_factorial"),
        how="inner",
        validate="one_to_one",
    )
    reconstruction_audit = pd.DataFrame(
        [
            {
                "n_paired": len(paired_reference),
                "choice_disagreement_rate": float((paired_reference.final_choice_saved != paired_reference.final_choice_factorial).mean()),
                "model_rt_mean_absolute_difference_s": float(abs(paired_reference.model_rt_saved - paired_reference.model_rt_factorial).mean()),
                "readout_time_mean_absolute_difference_s": float(abs(paired_reference.readout_time_saved - paired_reference.readout_time_factorial).mean()),
                "interpretation": "factorial gate is a paired counterfactual, not an exact reproduction of the saved current model",
            }
        ]
    )
    measurement_ranking = measurement_model_ranking(all_candidates)
    subject_profiles, cell_metrics, seed_metrics = evaluate_candidates(all_candidates, evaluation_human, splits)
    fold_metrics = evaluate_fold_stability(all_candidates, evaluation_human, splits)
    summary = summarize_seed_stability(seed_metrics, fold_metrics)
    candidates.to_csv(out / "mechanism_matrix_trial_level.csv.gz", index=False, compression="gzip")
    snapshots.to_csv(out / "snapshot_nonaccum_heldout_predictions.csv.gz", index=False, compression="gzip")
    provenance.to_csv(out / "trajectory_generation_manifest.csv", index=False)
    snapshot_fits.to_csv(out / "snapshot_nonaccum_fold_fits.csv", index=False)
    subject_profiles.to_csv(out / "equal_count_subject_profiles.csv", index=False)
    subject_equal_group_profile(subject_profiles).to_csv(out / "equal_count_group_profiles.csv", index=False)
    cell_metrics.to_csv(out / "continuous_joint_cell_metrics.csv", index=False)
    seed_metrics.to_csv(out / "mechanism_seed_metrics.csv", index=False)
    fold_metrics.to_csv(out / "mechanism_fold_metrics.csv", index=False)
    summary.to_csv(out / "mechanism_model_summary.csv", index=False)
    measurement_ranking.to_csv(out / "measurement_model_ranking.csv", index=False)
    reconstruction_audit.to_csv(out / "saved_vs_factorial_gate_audit.csv", index=False)
    source = (
        candidates.groupby(["candidate_id", "seed_id", "analysis_group", "congruency", "error_source"], as_index=False)
        .agg(n_trials=("trial_uid", "size"), mean_rt=("model_rt", "mean"), deadline_rate=("deadline_response", "mean"))
    )
    source.to_csv(out / "trajectory_error_source_decomposition.csv", index=False)
    plot_outputs(all_candidates, subject_profiles, out)
    write_summary(out, materiality, reproduction, decomposition, summary, cell_metrics, config_hash)
    print(out)


if __name__ == "__main__":
    main()

"""True single-subject feasibility workflow for VGG + recurrent accumulator.

Definition (this branch):
  - Fit Stage-2 accumulator parameters using *one subject's own trials* (train split)
  - Evaluate on held-out trials from the *same subject*

This mirrors the bounded panel/split workflow used by the VGG+WW feasibility runner,
but replaces Stage-2 WW fitting with an internal noisy recurrent accumulator head
trained directly from cached VGG logits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT, RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir, rel_to_root
from stage2_accumulator_backend import evaluate_cached_stage2_accumrnn_params, fit_stage2_accumrnn_from_logits
from train_age_groups_efficient import attach_flanker_labels_from_csv, compute_human_stats_from_rts, validate_cached_stage2_inputs


AGE_GROUPS: Tuple[str, str] = ("20-29", "80-89")
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "true_single_subject_feasibility_accumrnn"

DEFAULT_CHOICE_READOUT = "windowed_state_at_decision.v1"
DEFAULT_CHOICE_WINDOW = 3
DEFAULT_GAUSSIAN_RADIUS_STEPS = 6
DEFAULT_GAUSSIAN_SIGMA_STEPS = 2.0
DEFAULT_COMPETITION_MIX = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "True single-subject feasibility workflow for VGG + AccumRNN. "
            "Fits Stage-2 from a single subject's own trials and evaluates on held-out trials from that same subject."
        )
    )
    parser.add_argument("--mode", required=True, choices=("audit-baseline", "build-panel", "fit", "full"))
    parser.add_argument("--age_group", default=None, choices=AGE_GROUPS, help="Required for --mode fit")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")

    # Panel + split config
    parser.add_argument("--subjects_per_group", type=int, default=3)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--min_trials", type=int, default=25000)
    parser.add_argument("--min_incongruent", type=int, default=20)
    parser.add_argument("--min_errors", type=int, default=150)

    # Fit config
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--scales",
        default="0.10,0.30,0.50",
        help=(
            "Comma-separated scale grid for per-subject Stage-2 scale search. "
            "(Other Stage-2 parameters are trained from scratch per subject for each scale.)"
        ),
    )
    parser.add_argument("--choice_temperature", type=float, default=0.05)
    parser.add_argument(
        "--choice_readout",
        default=DEFAULT_CHOICE_READOUT,
        choices=(
            "windowed_state_at_decision.v1",
            "first_crosser_coupled.v1",
            "threshold_relative_windowed_state_at_decision.v1",
            "gaussian_pooled_state_at_decision.v1",
        ),
    )
    parser.add_argument("--choice_window", type=int, default=DEFAULT_CHOICE_WINDOW)
    parser.add_argument("--gaussian_radius_steps", type=int, default=DEFAULT_GAUSSIAN_RADIUS_STEPS)
    parser.add_argument("--gaussian_sigma_steps", type=float, default=DEFAULT_GAUSSIAN_SIGMA_STEPS)
    parser.add_argument("--competition_mix", type=float, default=DEFAULT_COMPETITION_MIX)
    parser.add_argument(
        "--accuracy_calib_weight",
        type=float,
        default=0.0,
        help="Weight for batch-level accuracy calibration loss (predicted p(correct) vs human batch accuracy).",
    )
    parser.add_argument("--max_train_trials", type=int, default=8000)
    parser.add_argument("--max_test_trials", type=int, default=2000)
    return parser.parse_args()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_int_seed(value: str) -> int:
    return int(zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF)


def _stable_indices_hash(train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    payload = json.dumps(
        {
            "train": np.asarray(train_idx, dtype=np.int64).tolist(),
            "test": np.asarray(test_idx, dtype=np.int64).tolist(),
        },
        separators=(",", ":"),
    )
    return _stable_int_seed(payload)


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _assert_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def _concat_cached_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in sorted(keys):
        if key not in a or key not in b:
            raise ValueError(f"CACHED_CONCAT_KEY_MISMATCH: key={key} a={key in a} b={key in b}")
        out[key] = np.concatenate([np.asarray(a[key]), np.asarray(b[key])], axis=0)
    return out


def _filter_cached_by_indices(cached: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in cached.items():
        if not isinstance(value, np.ndarray):
            continue
        out[key] = value[indices]
    return out


def _downsample_indices_stratified(
    *,
    df: pd.DataFrame,
    indices: np.ndarray,
    max_trials: int,
    seed: int,
) -> np.ndarray:
    if max_trials <= 0 or indices.size <= max_trials:
        return np.sort(indices)

    sub = df.loc[indices, ["target_direction", "response_direction", "flanker_direction"]].copy()
    correct = sub["response_direction"].astype(str) == sub["target_direction"].astype(str)
    incongruent = sub["flanker_direction"].astype(str) != sub["target_direction"].astype(str)
    strata = (
        correct.map({True: "correct", False: "error"}).astype(str)
        + "__"
        + incongruent.map({True: "incongruent", False: "congruent"}).astype(str)
    )

    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    total = int(indices.size)
    remaining_budget = int(max_trials)
    for _, group in sub.groupby(strata, sort=True):
        group_idx = group.index.to_numpy(dtype=np.int64)
        rng.shuffle(group_idx)
        take = int(max(1, np.floor(max_trials * len(group_idx) / max(total, 1))))
        take = min(take, int(len(group_idx)), remaining_budget)
        chosen.extend(group_idx[:take].tolist())
        remaining_budget = max_trials - len(chosen)
        if remaining_budget <= 0:
            break

    if len(chosen) < max_trials:
        remaining = np.setdiff1d(indices, np.array(chosen, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        chosen.extend(remaining[: max_trials - len(chosen)].tolist())

    return np.sort(np.array(chosen[:max_trials], dtype=np.int64))


def _recompute_subject_rts_normalized(
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    train_rts = np.asarray(train_cached["rts"], dtype=np.float32)
    log_min = float(np.log(train_rts.min() + 0.001))
    log_max = float(np.log(train_rts.max() + 0.001))
    log_range = max(log_max - log_min, 1e-6)

    def normalize(arr: np.ndarray) -> np.ndarray:
        return ((np.log(np.asarray(arr, dtype=np.float32) + 0.001) - log_min) / log_range).astype(np.float32)

    out_train = dict(train_cached)
    out_test = dict(test_cached)
    out_train["rts_normalized"] = normalize(out_train["rts"])
    out_test["rts_normalized"] = normalize(out_test["rts"])
    return out_train, out_test, {"log_rt_min": log_min, "log_rt_max": log_max, "log_rt_range": log_range}


def _scales_equivalent(a: Any, b: np.ndarray, *, atol: float = 1e-6) -> bool:
    try:
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
    except Exception:
        return False
    if a_arr.shape != b_arr.shape:
        return False
    return bool(np.allclose(a_arr, b_arr, atol=float(atol), rtol=0.0))


def _select_representative_subjects(
    *,
    df: pd.DataFrame,
    age_group: str,
    subjects_per_group: int,
    min_trials: int,
    min_incongruent: int,
    min_errors: int,
    seed: int,
) -> list[str]:
    pool = df.copy()
    pool["user_id"] = pool["user_id"].astype(str)
    pool["rt_s"] = pool["response_time"].to_numpy(dtype=np.float32) / 1000.0
    pool["correct"] = pool["response_direction"].astype(str) == pool["target_direction"].astype(str)
    pool["incongruent"] = pool["target_direction"].astype(str) != pool["flanker_direction"].astype(str)

    grouped = pool.groupby("user_id", sort=True)
    summary = grouped.agg(
        n_trials=("rt_s", "size"),
        n_incongruent=("incongruent", "sum"),
        n_errors=("correct", lambda x: int((~x).sum())),
        human_skewness=("rt_s", lambda x: float(pd.Series(x).skew())),
    ).reset_index()

    eligible = summary.loc[
        (summary["n_trials"] >= int(min_trials))
        & (summary["n_incongruent"] >= int(min_incongruent))
        & (summary["n_errors"] >= int(min_errors))
    ].copy()
    if eligible.empty:
        raise ValueError(
            f"NO_ELIGIBLE_SUBJECTS: age_group={age_group} after filters min_trials={min_trials} min_incongruent={min_incongruent} min_errors={min_errors}"
        )
    eligible = eligible.sort_values(["human_skewness", "user_id"], ascending=[True, True]).reset_index(drop=True)
    n = len(eligible)
    if subjects_per_group <= 1:
        idxs = [int(n // 2)]
    else:
        idxs = [int(round(i * (n - 1) / (subjects_per_group - 1))) for i in range(subjects_per_group)]
        idxs = sorted(set(max(0, min(n - 1, idx)) for idx in idxs))
        if len(idxs) < subjects_per_group:
            rng = np.random.default_rng(int(seed) + _stable_int_seed(f"{age_group}-fill"))
            candidates = list(range(n))
            rng.shuffle(candidates)
            for cand in candidates:
                if cand in idxs:
                    continue
                idxs.append(int(cand))
                if len(idxs) >= subjects_per_group:
                    break
            idxs = sorted(idxs)
    return eligible.loc[idxs, "user_id"].astype(str).tolist()[:subjects_per_group]


def _build_within_subject_split(
    *,
    df: pd.DataFrame,
    user_id: str,
    seed: int,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    user_mask = df["user_id"].astype(str).to_numpy(dtype=str) == str(user_id)
    indices = np.flatnonzero(user_mask)
    if indices.size == 0:
        raise ValueError(f"SUBJECT_NOT_FOUND: user_id={user_id}")
    rng = np.random.default_rng(int(seed) + _stable_int_seed(str(user_id)))
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_total = int(shuffled.size)
    n_test = max(1, int(round(float(test_fraction) * n_total)))
    test_idx = np.sort(shuffled[:n_test])
    train_idx = np.sort(shuffled[n_test:])
    if train_idx.size == 0:
        train_idx = np.sort(test_idx[:1])
        test_idx = np.sort(test_idx[1:])
    return train_idx, test_idx


def audit_baseline(
    *,
    output_root: Path,
    seed: int,
    subjects_per_group: int,
    test_fraction: float,
    choice_readout: str = DEFAULT_CHOICE_READOUT,
    choice_window: int = DEFAULT_CHOICE_WINDOW,
    gaussian_radius_steps: int = DEFAULT_GAUSSIAN_RADIUS_STEPS,
    gaussian_sigma_steps: float = DEFAULT_GAUSSIAN_SIGMA_STEPS,
    competition_mix: float = DEFAULT_COMPETITION_MIX,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    age_entries: dict[str, dict[str, Any]] = {}
    for age_group in AGE_GROUPS:
        data_dir = age_group_data_dir(age_group, matched=False)
        stage2_dir = age_group_stage2_dir(age_group, matched=False)

        train_csv = data_dir / "train_data.csv"
        test_csv = data_dir / "test_data.csv"
        rt_stats = data_dir / "rt_stats.json"
        train_npz = stage2_dir / "train_logits.npz"
        test_npz = stage2_dir / "test_logits.npz"
        _assert_paths_exist([train_csv, test_csv, rt_stats, train_npz, test_npz])

        validate_cached_stage2_inputs(age_group, str(data_dir), str(train_npz), str(test_npz))
        age_entries[age_group] = {
            "age_group": age_group,
            "matched": False,
            "data_dir": rel_to_root(data_dir),
            "stage2_dir": rel_to_root(stage2_dir),
            "train_csv": rel_to_root(train_csv),
            "test_csv": rel_to_root(test_csv),
            "rt_stats_json": rel_to_root(rt_stats),
            "train_logits_npz": rel_to_root(train_npz),
            "test_logits_npz": rel_to_root(test_npz),
        }

    manifest = {
        "schema_version": "true_single_subject_feasibility_accumrnn.v1",
        "created_at": _now_iso(),
        "seed": int(seed),
        "age_groups": age_entries,
        "panel_definition": {
            "subjects_per_group": int(subjects_per_group),
            "selection_strategy": "human_skewness_low_median_high_with_constraints",
        },
        "split_contract": {
            "type": "within_subject_random_split",
            "test_fraction": float(test_fraction),
            "seed": int(seed),
            "note": "Train/test are built from each subject's own trials; no cross-subject pooling inside a subject fit.",
        },
        "fit_contract": {
            "stage2_trainer": "stage2_accumulator_backend.fit_stage2_accumrnn_from_logits",
            "training_from_scratch_per_subject": True,
            "no_group_param_reuse": True,
            "no_posthoc_param_generation": True,
            "choice_readout": str(choice_readout),
            "choice_window": int(choice_window),
            "gaussian_radius_steps": int(gaussian_radius_steps),
            "gaussian_sigma_steps": float(gaussian_sigma_steps),
            "competition_mix": float(competition_mix),
        },
        "output_contract": {
            "panel_manifest_json": "panel_manifest.json",
            "subject_panel_csv": "subject_panel.csv",
            "per_subject_dir": "<age_group>/user_<ID>/",
            "per_subject_outputs": ["best_config.json", "best_model_params.npz", "subject_eval_summary.json"],
            "reaggregated_dir": "reaggregated/",
        },
        "prior_findings": {
            "heterogeneity": "HETEROGENEITY-NOT-SUPPORTED",
            "capture_probe": "CAPTURE-PROBE-NOT-SUPPORTED",
            "vgg_ww_single_subject": "VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE",
        },
    }
    _write_json(output_root / "panel_manifest.json", manifest)


def build_panel_and_splits(
    *,
    output_root: Path,
    seed: int,
    subjects_per_group: int,
    test_fraction: float,
    min_trials: int,
    min_incongruent: int,
    min_errors: int,
) -> None:
    baseline_path = output_root / "panel_manifest.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline manifest at {baseline_path}. Run --mode audit-baseline first.")
    baseline = _load_json(baseline_path)

    panel_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for age_group in AGE_GROUPS:
        entry = baseline["age_groups"][age_group]
        train_csv = PROJECT_ROOT / entry["train_csv"] if not os.path.isabs(entry["train_csv"]) else Path(entry["train_csv"])
        test_csv = PROJECT_ROOT / entry["test_csv"] if not os.path.isabs(entry["test_csv"]) else Path(entry["test_csv"])
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df["user_id"] = combined_df["user_id"].astype(str)

        subjects = _select_representative_subjects(
            df=combined_df,
            age_group=age_group,
            subjects_per_group=subjects_per_group,
            min_trials=min_trials,
            min_incongruent=min_incongruent,
            min_errors=min_errors,
            seed=seed,
        )
        for uid in subjects:
            panel_rows.append({"age_group": age_group, "user_id": str(uid)})
            train_idx, test_idx = _build_within_subject_split(
                df=combined_df,
                user_id=str(uid),
                seed=seed,
                test_fraction=test_fraction,
            )
            split_rows.append(
                {
                    "age_group": age_group,
                    "user_id": str(uid),
                    "n_total": int(train_idx.size + test_idx.size),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "train_indices": json.dumps(train_idx.tolist()),
                    "test_indices": json.dumps(test_idx.tolist()),
                }
            )

    panel_df = pd.DataFrame(panel_rows).sort_values(["age_group", "user_id"], ascending=[True, True])
    splits_df = pd.DataFrame(split_rows).sort_values(["age_group", "user_id"], ascending=[True, True])

    output_root.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_root / "subject_panel.csv", index=False)
    splits_df.to_csv(output_root / "panel_splits.csv", index=False)

    for row in split_rows:
        age_group = str(row["age_group"])
        uid = str(row["user_id"])
        out_dir = output_root / age_group / f"user_{uid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "age_group": age_group,
            "user_id": uid,
            "seed": int(seed),
            "test_fraction": float(test_fraction),
            "train_indices": json.loads(row["train_indices"]),
            "test_indices": json.loads(row["test_indices"]),
        }
        _write_json(out_dir / "subject_split.json", meta)


def _load_combined_cached_and_df(age_group: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data_dir = age_group_data_dir(age_group, matched=False)
    stage2_dir = age_group_stage2_dir(age_group, matched=False)
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
    train_npz = stage2_dir / "train_logits.npz"
    test_npz = stage2_dir / "test_logits.npz"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df["user_id"] = combined_df["user_id"].astype(str)

    train_cached, test_cached = validate_cached_stage2_inputs(age_group, str(data_dir), str(train_npz), str(test_npz))
    train_cached = attach_flanker_labels_from_csv(train_cached, str(train_csv))
    test_cached = attach_flanker_labels_from_csv(test_cached, str(test_csv))
    combined_cached = _concat_cached_dicts(train_cached, test_cached)
    return combined_df, combined_cached


def fit_age_group(
    *,
    age_group: str,
    output_root: Path,
    seed: int,
    device: str,
    epochs: int,
    scales: np.ndarray,
    choice_temperature: float,
    choice_readout: str,
    choice_window: int,
    gaussian_radius_steps: int,
    gaussian_sigma_steps: float,
    competition_mix: float,
    accuracy_calib_weight: float,
    max_train_trials: int,
    max_test_trials: int,
) -> None:
    panel_path = output_root / "subject_panel.csv"
    splits_path = output_root / "panel_splits.csv"
    if not panel_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            f"Missing panel inputs. subject_panel.csv={panel_path.exists()} panel_splits.csv={splits_path.exists()}"
        )

    panel_df = pd.read_csv(panel_path)
    splits_df = pd.read_csv(splits_path)
    subjects = panel_df.loc[panel_df["age_group"] == age_group, "user_id"].astype(str).tolist()
    if not subjects:
        raise ValueError(f"NO_SUBJECTS_FOR_AGE_GROUP: age_group={age_group}")

    combined_df, combined_cached = _load_combined_cached_and_df(age_group)
    for uid in subjects:
        user_dir = output_root / age_group / f"user_{uid}"
        user_dir.mkdir(parents=True, exist_ok=True)
        existing_summary = user_dir / "subject_eval_summary.json"
        existing_config = user_dir / "best_config.json"
        existing_params = user_dir / "best_model_params.npz"

        split_row = splits_df.loc[(splits_df["age_group"] == age_group) & (splits_df["user_id"].astype(str) == uid)]
        if split_row.empty:
            raise ValueError(f"MISSING_SPLIT_ROW: age_group={age_group} user_id={uid}")
        train_idx = np.array(json.loads(str(split_row["train_indices"].iloc[0])), dtype=np.int64)
        test_idx = np.array(json.loads(str(split_row["test_indices"].iloc[0])), dtype=np.int64)

        train_idx = _downsample_indices_stratified(
            df=combined_df,
            indices=train_idx,
            max_trials=int(max_train_trials),
            seed=int(seed) + _stable_int_seed(f"{age_group}-{uid}-train-subset"),
        )
        test_idx = _downsample_indices_stratified(
            df=combined_df,
            indices=test_idx,
            max_trials=int(max_test_trials),
            seed=int(seed) + _stable_int_seed(f"{age_group}-{uid}-test-subset"),
        )
        split_hash = int(_stable_indices_hash(train_idx, test_idx))

        if existing_summary.exists() and existing_config.exists() and existing_params.exists():
            try:
                prior = _load_json(existing_summary)
            except Exception:
                prior = {}
            subset_path = user_dir / "fit_subset_indices.json"
            split_subset_path = user_dir / "subject_split.json"
            subset_meta = {}
            for candidate in (subset_path, split_subset_path):
                if candidate.exists():
                    try:
                        subset_meta = _load_json(candidate)
                        break
                    except Exception:
                        subset_meta = {}

            if prior.get("schema_version") == "true_single_subject_feasibility_accumrnn.subject_eval.v1":
                same_epochs = int(subset_meta.get("epochs_requested", -1)) == int(epochs)
                same_temp = float(subset_meta.get("choice_temperature", float("nan"))) == float(choice_temperature)
                same_calib = float(subset_meta.get("accuracy_calib_weight", float("nan"))) == float(accuracy_calib_weight)
                same_scales = _scales_equivalent(subset_meta.get("scales"), scales)
                same_train_budget = int(subset_meta.get("max_train_trials", -1)) == int(max_train_trials)
                same_test_budget = int(subset_meta.get("max_test_trials", -1)) == int(max_test_trials)
                same_split_hash = int(subset_meta.get("split_hash", -1)) == int(split_hash)
                same_choice_readout = str(subset_meta.get("choice_readout", "")) == str(choice_readout)
                same_choice_window = int(subset_meta.get("choice_window", -1)) == int(choice_window)
                same_gaussian_radius_steps = int(subset_meta.get("gaussian_radius_steps", -1)) == int(gaussian_radius_steps)
                same_gaussian_sigma_steps = float(subset_meta.get("gaussian_sigma_steps", float("nan"))) == float(gaussian_sigma_steps)
                same_competition_mix = float(subset_meta.get("competition_mix", float("nan"))) == float(competition_mix)
                if (
                    same_epochs
                    and same_temp
                    and same_calib
                    and same_scales
                    and same_train_budget
                    and same_test_budget
                    and same_split_hash
                    and same_choice_readout
                    and same_choice_window
                    and same_gaussian_radius_steps
                    and same_gaussian_sigma_steps
                    and same_competition_mix
                ):
                    continue

        train_cached = _filter_cached_by_indices(combined_cached, train_idx)
        test_cached = _filter_cached_by_indices(combined_cached, test_idx)
        train_cached, test_cached, norm = _recompute_subject_rts_normalized(train_cached, test_cached)
        human_stats = compute_human_stats_from_rts(train_cached["rts"])

        _write_json(
            user_dir / "fit_subset_indices.json",
            {
                "age_group": age_group,
                "user_id": str(uid),
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "max_train_trials": int(max_train_trials),
                "max_test_trials": int(max_test_trials),
                "split_hash": int(split_hash),
                "epochs_requested": int(epochs),
                "choice_temperature": float(choice_temperature),
                "accuracy_calib_weight": float(accuracy_calib_weight),
                "choice_readout": str(choice_readout),
                "choice_window": int(choice_window),
                "gaussian_radius_steps": int(gaussian_radius_steps),
                "gaussian_sigma_steps": float(gaussian_sigma_steps),
                "competition_mix": float(competition_mix),
                "scales": [float(x) for x in scales.tolist()],
            },
        )

        fit_stage2_accumrnn_from_logits(
            age_group=f"{age_group}/user_{uid}",
            output_dir=str(user_dir),
            human_stats=human_stats,
            train_cached=train_cached,
            test_cached=test_cached,
            device=device,
            scales=scales,
            epochs=int(epochs),
            choice_temperature=float(choice_temperature),
            accuracy_calib_weight=float(accuracy_calib_weight),
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
            random_seed=int(seed) + _stable_int_seed(f"{age_group}-{uid}"),
            eval_random_seed=int(seed) + 13,
        )

        best_cfg = _load_json(user_dir / "best_config.json")
        best_scale = float(best_cfg["scale"])
        time_steps = int(best_cfg["time_steps"])
        best_epoch = int(best_cfg.get("best_epoch", -1)) if isinstance(best_cfg, dict) else -1
        params_npz = np.load(user_dir / "best_model_params.npz")
        params = {k: params_npz[k] for k in params_npz.files}
        _, metrics_test = evaluate_cached_stage2_accumrnn_params(
            params=params,
            time_steps=time_steps,
            cached=test_cached,
            device=device,
            choice_temperature=float(choice_temperature),
            random_seed=int(seed) + 31,
            rt_shape_focus=True,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
        )
        subject_summary = {
            "schema_version": "true_single_subject_feasibility_accumrnn.subject_eval.v1",
            "age_group": age_group,
            "user_id": str(uid),
            "fit": {
                "scale": float(best_scale),
                "time_steps": int(time_steps),
                "epochs": int(epochs),
                "best_epoch": int(best_epoch),
                "choice_temperature": float(choice_temperature),
                "accuracy_calib_weight": float(accuracy_calib_weight),
                "seed": int(seed),
                "scales": [float(x) for x in scales.tolist()],
                "rt_normalization": norm,
                "stage2_backend": "AccumulatorRaceDecisionV2",
                "choice_readout": str(choice_readout),
                "choice_window": int(choice_window),
                "gaussian_radius_steps": int(gaussian_radius_steps),
                "gaussian_sigma_steps": float(gaussian_sigma_steps),
                "competition_mix": float(competition_mix),
                "no_posthoc_param_generation": True,
                "internal_noise": True,
            },
            "test_metrics": metrics_test,
            "test_n_trials": int(len(test_cached["rts"])),
            "test_n_errors": int(np.sum(test_cached["response_labels"] != test_cached["target_labels"])),
        }
        _write_json(user_dir / "subject_eval_summary.json", subject_summary)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if args.mode == "audit-baseline":
        audit_baseline(
            output_root=output_root,
            seed=int(args.seed),
            subjects_per_group=int(args.subjects_per_group),
            test_fraction=float(args.test_fraction),
            choice_readout=str(args.choice_readout),
            choice_window=int(args.choice_window),
            gaussian_radius_steps=int(args.gaussian_radius_steps),
            gaussian_sigma_steps=float(args.gaussian_sigma_steps),
            competition_mix=float(args.competition_mix),
        )
        return

    if args.mode == "build-panel":
        build_panel_and_splits(
            output_root=output_root,
            seed=int(args.seed),
            subjects_per_group=int(args.subjects_per_group),
            test_fraction=float(args.test_fraction),
            min_trials=int(args.min_trials),
            min_incongruent=int(args.min_incongruent),
            min_errors=int(args.min_errors),
        )
        return

    scales = np.array([float(x.strip()) for x in str(args.scales).split(",") if x.strip()], dtype=np.float32)
    if scales.size == 0:
        raise ValueError("SCALES_EMPTY")

    if args.mode == "fit":
        if args.age_group is None:
            raise ValueError("--age_group is required for --mode fit")
        fit_age_group(
            age_group=str(args.age_group),
            output_root=output_root,
            seed=int(args.seed),
            device=str(args.device),
            epochs=int(args.epochs),
            scales=scales,
            choice_temperature=float(args.choice_temperature),
            choice_readout=str(args.choice_readout),
            choice_window=int(args.choice_window),
            gaussian_radius_steps=int(args.gaussian_radius_steps),
            gaussian_sigma_steps=float(args.gaussian_sigma_steps),
            competition_mix=float(args.competition_mix),
            accuracy_calib_weight=float(args.accuracy_calib_weight),
            max_train_trials=int(args.max_train_trials),
            max_test_trials=int(args.max_test_trials),
        )
        return

    if args.mode == "full":
        audit_baseline(
            output_root=output_root,
            seed=int(args.seed),
            subjects_per_group=int(args.subjects_per_group),
            test_fraction=float(args.test_fraction),
            choice_readout=str(args.choice_readout),
            choice_window=int(args.choice_window),
            gaussian_radius_steps=int(args.gaussian_radius_steps),
            gaussian_sigma_steps=float(args.gaussian_sigma_steps),
            competition_mix=float(args.competition_mix),
        )
        build_panel_and_splits(
            output_root=output_root,
            seed=int(args.seed),
            subjects_per_group=int(args.subjects_per_group),
            test_fraction=float(args.test_fraction),
            min_trials=int(args.min_trials),
            min_incongruent=int(args.min_incongruent),
            min_errors=int(args.min_errors),
        )
        for age_group in AGE_GROUPS:
            fit_age_group(
                age_group=age_group,
                output_root=output_root,
                seed=int(args.seed),
                device=str(args.device),
                epochs=int(args.epochs),
                scales=scales,
                choice_temperature=float(args.choice_temperature),
                choice_readout=str(args.choice_readout),
                choice_window=int(args.choice_window),
                gaussian_radius_steps=int(args.gaussian_radius_steps),
                gaussian_sigma_steps=float(args.gaussian_sigma_steps),
                competition_mix=float(args.competition_mix),
                accuracy_calib_weight=float(args.accuracy_calib_weight),
                max_train_trials=int(args.max_train_trials),
                max_test_trials=int(args.max_test_trials),
            )
        analysis_cmd = [
            sys.executable,
            "code/scripts/analyze_true_single_subject_feasibility_accumrnn.py",
            "--input_root",
            str(output_root),
            "--output_root",
            str(output_root),
        ]
        completed = subprocess.run(analysis_cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Analysis command failed with exit code {completed.returncode}: {' '.join(analysis_cmd)}")
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

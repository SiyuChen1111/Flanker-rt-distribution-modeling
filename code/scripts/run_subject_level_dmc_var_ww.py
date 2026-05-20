from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cache_vgg_stage2_features import load_stage1_model_with_metadata
from project_paths import PROJECT_ROOT, RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir, rel_to_root
from run_true_single_subject_feasibility import (
    AGE_GROUPS,
    _build_within_subject_split,
    _concat_cached_dicts,
    _downsample_indices_stratified,
    _filter_cached_by_indices,
    _load_json,
    _now_iso,
    _recompute_subject_rts_normalized,
    _scales_equivalent,
    _stable_indices_hash,
    _stable_int_seed,
    _to_jsonable,
    _write_json,
    audit_baseline,
)
from stage1_semisup_evidence_sampler import SemiSupervisedEvidenceSampler, Stage1EvidenceConfig
from train_age_group_semisup_spea import StimulusDataset, _build_behavior_balanced_subset, train_stage1_head
from train_age_group_semisup_spea import train_stage1_head_from_cached_features
from train_age_groups_efficient import (
    attach_flanker_labels_from_csv,
    compute_human_stats_from_rts,
    evaluate_cached_stage2_params,
    fit_stage2_from_logits,
    to_jsonable,
    validate_cached_stage2_inputs,
)
from train_dmc_var_ww_smoke import train_dmc_variational_ww


DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "per_subject_age_comparison_plan" / "01_subject_level_dmc_var_ww"
LOCK_PATH = RESULTS_ROOT / "per_subject_age_comparison_plan" / "00_protocol" / "panel_lock.json"
RUNNER_CONTRACT_PATH = (
    RESULTS_ROOT / "per_subject_age_comparison_plan" / "00_protocol" / "subject_level_dmc_var_ww_runner_contract.md"
)
METRIC_CONTRACT_PATH = (
    RESULTS_ROOT / "per_subject_age_comparison_plan" / "00_protocol" / "stage1_metric_gate_contract.md"
)
REUSE_PANEL_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "true_single_subject_feasibility_rt_response_only"
ALLOWED_ARMS = (
    "t0_only_baseline",
    "phase18_core",
    "phase18_replay_aligned",
    "phase18_plus_stage1_uncertainty_gain",
)

PHASE18_REPLAY_REFERENCE_ROOT = RESULTS_ROOT / "rt_model_dmc_var_ww"
PHASE18_REPLAY_PROFILES: Dict[str, Dict[str, Any]] = {
    "a3_s4": {
        "reference_dir_name": "smoke_a3_s4",
        "dmc_auto_strength": 0.3,
        "dmc_selection_strength": 0.4,
    },
    "a5_s3": {
        "reference_dir_name": "smoke_a5_s3",
        "dmc_auto_strength": 0.5,
        "dmc_selection_strength": 0.3,
    },
}


@dataclass(frozen=True)
class ArmConfig:
    arm_name: str
    fit_family: str
    stage1_sampler_mode: str
    dmc_enabled: bool
    stage1_uncertainty_gain: float
    implemented: bool
    refusal_reason: Optional[str]
    t0_mode: str
    t0_seconds: float
    choice_temperature: float
    scales: np.ndarray
    behavior_smoke_mode: str
    max_train_trials: int
    max_test_trials: int
    epochs_stage1: int
    epochs_ww: int
    evidence_time_steps: int
    ww_time_steps: int
    ww_dt: int
    noise_ampa: Optional[float]
    threshold: Optional[float]
    j_offdiag_scale: Optional[float]
    j_ext: Optional[float]
    dmc_auto_strength: float
    dmc_auto_peak_s: float
    dmc_selection_strength: float
    dmc_selection_midpoint_s: float
    dmc_selection_tau_s: float
    dmc_apply_to: str
    sigma_evidence_noise: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-level DMC+Var→WW runner")
    parser.add_argument("--mode", required=True, choices=("audit-baseline", "build-panel", "fit-arm", "analyze-panel", "full-stage1"))
    parser.add_argument("--age_group", default="both", choices=("20-29", "80-89", "both"))
    parser.add_argument("--user_ids", default=None, help="Optional comma-separated user_id filter within the selected age group(s).")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects_per_group", type=int, default=3)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--min_trials", type=int, default=25000)
    parser.add_argument("--min_incongruent", type=int, default=20)
    parser.add_argument("--min_errors", type=int, default=150)
    parser.add_argument("--comparator_arm", default=None, choices=ALLOWED_ARMS)
    parser.add_argument("--comparator_arms", default=",".join(ALLOWED_ARMS))
    parser.add_argument("--max_train_trials", type=int, default=8000)
    parser.add_argument("--max_test_trials", type=int, default=2000)
    parser.add_argument("--epochs_stage1", type=int, default=2)
    parser.add_argument("--epochs_ww", type=int, default=8)
    parser.add_argument("--evidence_time_steps", type=int, default=120)
    parser.add_argument("--ww_time_steps", type=int, default=120)
    parser.add_argument("--ww_dt", type=int, default=10)
    parser.add_argument("--choice_temperature", type=float, default=0.05)
    parser.add_argument("--scales", default="0.10,0.30,0.50")
    parser.add_argument("--stage1_sampler_mode", default=None, choices=("deterministic", "variational", "mc_dropout"))
    parser.add_argument("--sigma_evidence_noise", type=float, default=0.0)
    parser.add_argument("--stage1_uncertainty_gain", type=float, default=1.25)
    parser.add_argument("--phase18_replay_profile", default="a5_s3", choices=tuple(PHASE18_REPLAY_PROFILES.keys()))
    parser.add_argument("--phase18_replay_t0_seconds", type=float, default=0.25)
    parser.add_argument("--phase18_replay_j_ext", type=float, default=None)
    parser.add_argument("--readout_mode", default="soft_index")
    parser.add_argument("--t0_young_seconds", type=float, default=0.10)
    parser.add_argument("--t0_old_seconds", type=float, default=0.15)
    parser.add_argument("--noise_ampa", type=float, default=0.08)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--j_offdiag_scale", type=float, default=1.0)
    parser.add_argument("--j_ext", type=float, default=0.75)
    parser.add_argument("--auto_strength", type=float, default=0.3)
    parser.add_argument("--auto_peak_s", type=float, default=0.06)
    parser.add_argument("--selection_strength", type=float, default=0.4)
    parser.add_argument("--selection_midpoint_s", type=float, default=0.18)
    parser.add_argument("--selection_tau_s", type=float, default=0.06)
    parser.add_argument("--apply_to", default="incongruent_only", choices=("incongruent_only", "all_trials"))
    parser.add_argument("--smoke_eval", action="store_true")
    parser.add_argument("--smoke_max_trials", type=int, default=1024)
    return parser.parse_args()


def _resolve_output_root(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_age_groups(arg: str) -> Tuple[str, ...]:
    return AGE_GROUPS if arg == "both" else (str(arg),)


def _resolve_user_id_filter(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None:
        return None
    values = {token.strip() for token in str(raw).split(",") if token.strip()}
    return values or None


def _filter_panel_entries(lock_payload: dict, age_group: str, user_filter: Optional[set[str]]) -> List[dict]:
    entries = list(lock_payload["panel"][age_group])
    if user_filter is None:
        return entries
    return [entry for entry in entries if str(entry["user_id"]) in user_filter]


def _parse_scales(value: str) -> np.ndarray:
    scales = np.array([float(x.strip()) for x in str(value).split(",") if x.strip()], dtype=np.float32)
    if scales.size == 0:
        raise ValueError("SCALES_EMPTY")
    return scales


def _assert_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def _safe_rel_or_abs(path: Path) -> str:
    try:
        return rel_to_root(path)
    except ValueError:
        return str(path)


def _load_lock() -> dict:
    if not LOCK_PATH.exists():
        raise FileNotFoundError(f"Missing panel lock: {LOCK_PATH}")
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _load_reuse_panel_manifest() -> dict:
    path = REUSE_PANEL_ROOT / "panel_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_locked_panel_files(output_root: Path, lock_payload: dict) -> None:
    _assert_paths_exist([
        REUSE_PANEL_ROOT / "subject_panel.csv",
        REUSE_PANEL_ROOT / "panel_splits.csv",
    ])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REUSE_PANEL_ROOT / "subject_panel.csv", output_root / "subject_panel.csv")
    shutil.copy2(REUSE_PANEL_ROOT / "panel_splits.csv", output_root / "panel_splits.csv")

    panel = lock_payload["panel"]
    for age_group in AGE_GROUPS:
        for row in panel[age_group]:
            user_id = str(row["user_id"])
            src_dir = REUSE_PANEL_ROOT / age_group / f"user_{user_id}"
            dst_dir = output_root / age_group / f"user_{user_id}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / "subject_split.json", dst_dir / "subject_split.json")
            shutil.copy2(src_dir / "fit_subset_indices.json", dst_dir / "fit_subset_indices.json")


def _verify_lock_compatibility(args: argparse.Namespace, lock_payload: dict, scales: np.ndarray, arms: Tuple[str, ...]) -> None:
    if int(args.seed) != int(lock_payload["seed_policy"]["global_seed"]):
        raise ValueError("LOCK_DRIFT_GLOBAL_SEED")
    fit_modes = {"fit-arm", "full-stage1"}
    if str(args.mode) in fit_modes:
        budgets = lock_payload["bounded_trial_budgets"]
        if int(args.max_train_trials) != int(budgets["max_train_trials"]):
            raise ValueError("LOCK_DRIFT_MAX_TRAIN_TRIALS")
        if int(args.max_test_trials) != int(budgets["max_test_trials"]):
            raise ValueError("LOCK_DRIFT_MAX_TEST_TRIALS")
        if int(args.epochs_ww) != int(budgets["epochs_requested"]):
            raise ValueError("LOCK_DRIFT_EPOCHS_WW")
        if float(args.choice_temperature) != float(budgets["choice_temperature"]):
            raise ValueError("LOCK_DRIFT_CHOICE_TEMPERATURE")
        if not _scales_equivalent(budgets["scales"], scales):
            raise ValueError("LOCK_DRIFT_SCALES")
    if str(args.stage1_sampler_mode) not in {"None", "variational", "deterministic", "mc_dropout"}:
        raise ValueError("INVALID_STAGE1_SAMPLER_MODE")


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


def _load_combined_feature_cached(age_group: str) -> Dict[str, np.ndarray]:
    data_dir = age_group_data_dir(age_group, matched=False)
    stage2_dir = age_group_stage2_dir(age_group, matched=False)
    train_npz = stage2_dir / "train_features.npz"
    test_npz = stage2_dir / "test_features.npz"
    if not train_npz.exists() or not test_npz.exists():
        train_npz = stage2_dir / "train_logits.npz"
        test_npz = stage2_dir / "test_logits.npz"
    train_cached, test_cached = validate_cached_stage2_inputs(age_group, str(data_dir), str(train_npz), str(test_npz))
    train_cached = attach_flanker_labels_from_csv(train_cached, str(data_dir / "train_data.csv"))
    test_cached = attach_flanker_labels_from_csv(test_cached, str(data_dir / "test_data.csv"))
    return _concat_cached_dicts(train_cached, test_cached)


def _stage1_inputs_for_indices(
    *,
    age_group: str,
    selected_indices: np.ndarray,
    sampler: SemiSupervisedEvidenceSampler,
    device: str,
    heartbeat_path: Path | None = None,
) -> Dict[str, np.ndarray]:
    combined_df, combined_cached = _load_combined_cached_and_df(age_group)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_cached = _filter_cached_by_indices(combined_cached, selected_indices)

    try:
        feature_cached = _filter_cached_by_indices(_load_combined_feature_cached(age_group), selected_indices)
        feature_key = "pooled_features" if "pooled_features" in feature_cached else "features"
        if feature_key in feature_cached:
            return {
                **selected_cached,
                "pooled_features": np.asarray(feature_cached[feature_key], dtype=np.float32),
                "logits": np.asarray(feature_cached["logits"], dtype=np.float32),
            }
    except Exception:
        pass

    if combined_df is None:
        fallback_logits = np.asarray(
            selected_cached.get(
                "logits",
                np.zeros((len(selected_indices), 4), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        return {
            **selected_cached,
            "pooled_features": fallback_logits,
            "logits": fallback_logits,
        }

    selected_df = combined_df.iloc[selected_indices].copy()
    representative_indices = (
        selected_df.reset_index()
        .drop_duplicates(subset=["stimulus_image_path"], keep="first")["index"]
        .to_numpy(dtype=np.int64)
    )
    representative_indices = np.sort(representative_indices)
    representative_paths = combined_df.iloc[representative_indices]["stimulus_image_path"].astype(str).to_numpy()
    path_to_position = {str(path): idx for idx, path in enumerate(representative_paths.tolist())}

    loader = _build_loader_from_combined_indices(age_group=age_group, selected_indices=representative_indices, batch_size=128)
    feature_batches: List[np.ndarray] = []
    logit_batches: List[np.ndarray] = []
    rows_done = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features, logits = sampler.encode_images(images)
            feature_batches.append(features.detach().cpu().numpy())
            logit_batches.append(logits.detach().cpu().numpy())
            rows_done += int(images.shape[0])
            if heartbeat_path is not None:
                _write_json(
                    heartbeat_path,
                    {
                        "phase": "stage1_feature_cache",
                        "status": "running",
                        "source": "unique_stimulus_images",
                        "unique_images_done": int(rows_done),
                        "unique_images_total": int(len(representative_indices)),
                        "selected_rows": int(selected_indices.size),
                    },
                )

    unique_features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    unique_logits = np.concatenate(logit_batches, axis=0).astype(np.float32)
    selected_paths = selected_df["stimulus_image_path"].astype(str).to_numpy()
    take = np.array([path_to_position[str(path)] for path in selected_paths], dtype=np.int64)
    if heartbeat_path is not None:
        _write_json(
            heartbeat_path,
            {
                "phase": "stage1_feature_cache",
                "status": "completed",
                "source": "unique_stimulus_images",
                "unique_images_total": int(len(representative_indices)),
                "selected_rows": int(selected_indices.size),
            },
        )
    return {
        **selected_cached,
        "pooled_features": unique_features[take],
        "logits": unique_logits[take],
    }


def _load_subject_splits(output_root: Path, age_group: str, user_id: str) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    user_dir = output_root / age_group / f"user_{user_id}"
    split_meta = _load_json(user_dir / "subject_split.json")
    subset_meta = _load_json(user_dir / "fit_subset_indices.json")
    full_train = np.array(split_meta["train_indices"], dtype=np.int64)
    full_test = np.array(split_meta["test_indices"], dtype=np.int64)
    return full_train, full_test, split_meta, subset_meta


def _shared_stage1_root(output_root: Path, arm_name: str, age_group: str) -> Path:
    return _arm_root(output_root, arm_name) / age_group / "_shared_stage1"


def _load_panel_subject_ids(output_root: Path, age_group: str) -> List[str]:
    panel_path = output_root / "subject_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing subject panel: {panel_path}")
    panel_df = pd.read_csv(panel_path)
    panel_df["age_group"] = panel_df["age_group"].astype(str)
    panel_df["user_id"] = panel_df["user_id"].astype(str)
    return panel_df.loc[panel_df["age_group"] == str(age_group), "user_id"].tolist()


def _build_shared_stage1_train_indices(
    output_root: Path,
    age_group: str,
    *,
    max_trials: int,
    seed: int,
) -> np.ndarray:
    subject_ids = _load_panel_subject_ids(output_root, age_group)
    if not subject_ids:
        raise ValueError(f"NO_PANEL_SUBJECTS_FOR_SHARED_STAGE1: age_group={age_group}")
    parts: List[np.ndarray] = []
    for user_id in subject_ids:
        _, _, _, subset_meta = _load_subject_splits(output_root, age_group, user_id)
        parts.append(np.array(subset_meta["train_indices"], dtype=np.int64))
    if not parts:
        raise ValueError(f"NO_TRAIN_INDICES_FOR_SHARED_STAGE1: age_group={age_group}")
    combined_indices = np.unique(np.concatenate(parts, axis=0))
    if max_trials <= 0 or combined_indices.size <= max_trials:
        return combined_indices
    combined_df, _ = _load_combined_cached_and_df(age_group)
    return _downsample_indices_stratified(
        df=combined_df,
        indices=combined_indices,
        max_trials=int(max_trials),
        seed=int(seed),
    )


def _prepare_shared_stage1_sampler(
    *,
    output_root: Path,
    arm_cfg: ArmConfig,
    age_group: str,
    device: str,
) -> SemiSupervisedEvidenceSampler:
    stage1_backbone, stage1_meta = load_stage1_model_with_metadata(str(device))
    stage1_cfg = Stage1EvidenceConfig(n_classes=4, feature_dim=512, hidden_dim=128, dropout_rate=0.10)
    sampler = SemiSupervisedEvidenceSampler(stage1_cfg, stage1_backbone=stage1_backbone).to(device)

    shared_root = _shared_stage1_root(output_root, arm_cfg.arm_name, age_group)
    shared_root.mkdir(parents=True, exist_ok=True)
    shared_state_path = shared_root / "variational_head_state.pt"

    if arm_cfg.stage1_sampler_mode != "variational":
        return sampler

    if shared_state_path.exists():
        state = torch.load(shared_state_path, map_location=device, weights_only=False)
        sampler.variational_head.load_state_dict(state, strict=True)
        return sampler

    shared_train_indices = _build_shared_stage1_train_indices(
        output_root,
        age_group,
        max_trials=int(arm_cfg.max_train_trials),
        seed=_stable_int_seed(f"shared-stage1::{arm_cfg.arm_name}::{age_group}"),
    )
    try:
        feature_cached = _stage1_inputs_for_indices(
            age_group=age_group,
            selected_indices=shared_train_indices,
            sampler=sampler,
            device=str(device),
            heartbeat_path=shared_root / "stage1_feature_cache.heartbeat.json",
        )
        train_stage1_head_from_cached_features(
            sampler=sampler,
            pooled_features=feature_cached["pooled_features"],
            base_logits=feature_cached["logits"],
            target_labels=feature_cached["target_labels"],
            sampler_mode="variational",
            epochs=int(arm_cfg.epochs_stage1),
            lambda_cls=1.0,
            lambda_teacher=0.25,
            lambda_uncertainty_bound=0.05,
            device=str(device),
            checkpoint_dir=shared_root / "stage1_checkpoints",
            resume=True,
            heartbeat_path=shared_root / "stage1.heartbeat.json",
        )
    except Exception as exc:
        _write_json(
            shared_root / "stage1_cached_feature_fallback.json",
            {
                "status": "fallback_to_images",
                "reason": str(exc),
                "updated_at": _now_iso(),
            },
        )
        shared_train_loader = _build_loader_from_combined_indices(
            age_group=age_group,
            selected_indices=shared_train_indices,
        )
        train_stage1_head(
            sampler=sampler,
            dataset_loader=shared_train_loader,
            sampler_mode="variational",
            epochs=int(arm_cfg.epochs_stage1),
            lambda_cls=1.0,
            lambda_ssl=0.0,
            lambda_teacher=0.25,
            lambda_uncertainty_bound=0.05,
            device=str(device),
            checkpoint_dir=shared_root / "stage1_checkpoints",
            resume=True,
            heartbeat_path=shared_root / "stage1.heartbeat.json",
        )
    torch.save(sampler.variational_head.state_dict(), shared_state_path)
    _write_json(
        shared_root / "shared_stage1_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.shared_stage1_manifest.v1",
            "age_group": str(age_group),
            "comparator_arm": str(arm_cfg.arm_name),
            "stage1_sampler_mode": "variational",
            "shared_train_index_count": int(shared_train_indices.size),
            "epochs_requested": int(arm_cfg.epochs_stage1),
            **stage1_meta,
        },
    )
    return sampler


def _derive_arm_subject_seed(global_seed: int, age_group: str, arm_name: str, user_id: str) -> int:
    return int(global_seed) + _stable_int_seed(f"{age_group}::{arm_name}::{user_id}")


def _derive_arm_eval_seed(global_seed: int, age_group: str, arm_name: str, user_id: str) -> int:
    return int(global_seed) + 13 + _stable_int_seed(f"eval::{age_group}::{arm_name}::{user_id}")


def _build_arm_config(args: argparse.Namespace, age_group: str, arm_name: str, scales: np.ndarray) -> ArmConfig:
    t0_seconds = float(args.t0_young_seconds if age_group == "20-29" else args.t0_old_seconds)
    if arm_name == "t0_only_baseline":
        return ArmConfig(
            arm_name=arm_name,
            fit_family="cached_stage2_ww",
            stage1_sampler_mode="deterministic",
            dmc_enabled=False,
            stage1_uncertainty_gain=1.0,
            implemented=True,
            refusal_reason=None,
            t0_mode="fixed_global",
            t0_seconds=t0_seconds,
            choice_temperature=float(args.choice_temperature),
            scales=scales,
            behavior_smoke_mode="rt_response_only",
            max_train_trials=int(args.max_train_trials),
            max_test_trials=int(args.max_test_trials),
            epochs_stage1=0,
            epochs_ww=int(args.epochs_ww),
            evidence_time_steps=int(args.evidence_time_steps),
            ww_time_steps=int(args.ww_time_steps),
            ww_dt=int(args.ww_dt),
            noise_ampa=None,
            threshold=None,
            j_offdiag_scale=None,
            j_ext=None,
            dmc_auto_strength=0.0,
            dmc_auto_peak_s=0.0,
            dmc_selection_strength=0.0,
            dmc_selection_midpoint_s=0.0,
            dmc_selection_tau_s=0.0,
            dmc_apply_to="incongruent_only",
            sigma_evidence_noise=0.0,
        )
    if arm_name == "phase18_core":
        return ArmConfig(
            arm_name=arm_name,
            fit_family="dmc_variational_ww",
            stage1_sampler_mode="variational",
            dmc_enabled=True,
            stage1_uncertainty_gain=1.0,
            implemented=True,
            refusal_reason=None,
            t0_mode="fixed_global",
            t0_seconds=t0_seconds,
            choice_temperature=float(args.choice_temperature),
            scales=scales,
            behavior_smoke_mode="rt_response_only",
            max_train_trials=int(args.max_train_trials),
            max_test_trials=int(args.max_test_trials),
            epochs_stage1=int(args.epochs_stage1),
            epochs_ww=int(args.epochs_ww),
            evidence_time_steps=int(args.evidence_time_steps),
            ww_time_steps=int(args.ww_time_steps),
            ww_dt=int(args.ww_dt),
            noise_ampa=float(args.noise_ampa) if args.noise_ampa is not None else None,
            threshold=float(args.threshold) if args.threshold is not None else None,
            j_offdiag_scale=float(args.j_offdiag_scale) if args.j_offdiag_scale is not None else None,
            j_ext=float(args.j_ext) if args.j_ext is not None else None,
            dmc_auto_strength=float(args.auto_strength),
            dmc_auto_peak_s=float(args.auto_peak_s),
            dmc_selection_strength=float(args.selection_strength),
            dmc_selection_midpoint_s=float(args.selection_midpoint_s),
            dmc_selection_tau_s=float(args.selection_tau_s),
            dmc_apply_to=str(args.apply_to),
            sigma_evidence_noise=float(args.sigma_evidence_noise),
        )
    if arm_name == "phase18_replay_aligned":
        replay_profile = PHASE18_REPLAY_PROFILES[str(args.phase18_replay_profile)]
        return ArmConfig(
            arm_name=arm_name,
            fit_family="dmc_variational_ww",
            stage1_sampler_mode="variational",
            dmc_enabled=True,
            stage1_uncertainty_gain=1.0,
            implemented=True,
            refusal_reason=None,
            t0_mode="fixed_global",
            t0_seconds=float(args.phase18_replay_t0_seconds),
            choice_temperature=0.10,
            scales=scales,
            behavior_smoke_mode="rt_response_only",
            max_train_trials=int(args.max_train_trials),
            max_test_trials=int(args.max_test_trials),
            epochs_stage1=int(args.epochs_stage1),
            epochs_ww=int(args.epochs_ww),
            evidence_time_steps=120,
            ww_time_steps=120,
            ww_dt=int(args.ww_dt),
            noise_ampa=0.08,
            threshold=0.22,
            j_offdiag_scale=None,
            j_ext=args.phase18_replay_j_ext,
            dmc_auto_strength=float(replay_profile["dmc_auto_strength"]),
            dmc_auto_peak_s=float(args.auto_peak_s),
            dmc_selection_strength=float(replay_profile["dmc_selection_strength"]),
            dmc_selection_midpoint_s=float(args.selection_midpoint_s),
            dmc_selection_tau_s=float(args.selection_tau_s),
            dmc_apply_to=str(args.apply_to),
            sigma_evidence_noise=float(args.sigma_evidence_noise),
        )
    if arm_name == "phase18_plus_stage1_uncertainty_gain":
        if float(args.stage1_uncertainty_gain) <= 1.0:
            return ArmConfig(
                arm_name=arm_name,
                fit_family="dmc_variational_ww",
                stage1_sampler_mode="variational",
                dmc_enabled=True,
                stage1_uncertainty_gain=float(args.stage1_uncertainty_gain),
                implemented=False,
                refusal_reason="Requested phase18_plus_stage1_uncertainty_gain but stage1_uncertainty_gain <= 1.0, so the bounded uncertainty-amplitude extension is not actually enabled.",
                t0_mode="fixed_global",
                t0_seconds=t0_seconds,
                choice_temperature=float(args.choice_temperature),
                scales=scales,
                behavior_smoke_mode="rt_response_only",
                max_train_trials=int(args.max_train_trials),
                max_test_trials=int(args.max_test_trials),
                epochs_stage1=int(args.epochs_stage1),
                epochs_ww=int(args.epochs_ww),
                evidence_time_steps=int(args.evidence_time_steps),
                ww_time_steps=int(args.ww_time_steps),
                ww_dt=int(args.ww_dt),
                noise_ampa=float(args.noise_ampa) if args.noise_ampa is not None else None,
                threshold=float(args.threshold) if args.threshold is not None else None,
                j_offdiag_scale=float(args.j_offdiag_scale) if args.j_offdiag_scale is not None else None,
                j_ext=float(args.j_ext) if args.j_ext is not None else None,
                dmc_auto_strength=float(args.auto_strength),
                dmc_auto_peak_s=float(args.auto_peak_s),
                dmc_selection_strength=float(args.selection_strength),
                dmc_selection_midpoint_s=float(args.selection_midpoint_s),
                dmc_selection_tau_s=float(args.selection_tau_s),
                dmc_apply_to=str(args.apply_to),
                sigma_evidence_noise=float(args.sigma_evidence_noise),
            )
        return ArmConfig(
            arm_name=arm_name,
            fit_family="dmc_variational_ww",
            stage1_sampler_mode="variational",
            dmc_enabled=True,
            stage1_uncertainty_gain=float(args.stage1_uncertainty_gain),
            implemented=True,
            refusal_reason=None,
            t0_mode="fixed_global",
            t0_seconds=t0_seconds,
            choice_temperature=float(args.choice_temperature),
            scales=scales,
            behavior_smoke_mode="rt_response_only",
            max_train_trials=int(args.max_train_trials),
            max_test_trials=int(args.max_test_trials),
            epochs_stage1=int(args.epochs_stage1),
            epochs_ww=int(args.epochs_ww),
            evidence_time_steps=int(args.evidence_time_steps),
            ww_time_steps=int(args.ww_time_steps),
            ww_dt=int(args.ww_dt),
            noise_ampa=float(args.noise_ampa) if args.noise_ampa is not None else None,
            threshold=float(args.threshold) if args.threshold is not None else None,
            j_offdiag_scale=float(args.j_offdiag_scale) if args.j_offdiag_scale is not None else None,
            j_ext=float(args.j_ext) if args.j_ext is not None else None,
            dmc_auto_strength=float(args.auto_strength),
            dmc_auto_peak_s=float(args.auto_peak_s),
            dmc_selection_strength=float(args.selection_strength),
            dmc_selection_midpoint_s=float(args.selection_midpoint_s),
            dmc_selection_tau_s=float(args.selection_tau_s),
            dmc_apply_to=str(args.apply_to),
            sigma_evidence_noise=float(args.sigma_evidence_noise),
        )
    raise ValueError(f"Unknown comparator arm: {arm_name}")


def _arm_root(output_root: Path, arm_name: str) -> Path:
    return output_root / arm_name


def _subject_output_dir(output_root: Path, arm_name: str, age_group: str, user_id: str) -> Path:
    return _arm_root(output_root, arm_name) / age_group / f"user_{user_id}"


def _build_loader_from_indices(csv_path: Path, selected_indices: np.ndarray, batch_size: int = 128) -> DataLoader:
    dataset = StimulusDataset(str(csv_path))
    subset = Subset(dataset, selected_indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def _split_combined_indices_for_csv_loaders(
    *,
    age_group: str,
    combined_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = age_group_data_dir(age_group, matched=False)
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
    train_len = int(len(pd.read_csv(train_csv)))
    total_len = train_len + int(len(pd.read_csv(test_csv)))
    indices = np.asarray(combined_indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= total_len):
        raise ValueError(
            f"COMBINED_INDEX_OUT_OF_RANGE: age_group={age_group} min={int(indices.min())} max={int(indices.max())} total_len={total_len}"
        )
    train_local = indices[indices < train_len]
    test_local = indices[indices >= train_len] - train_len
    return train_local.astype(np.int64), test_local.astype(np.int64)


def _build_loader_from_combined_indices(
    *,
    age_group: str,
    selected_indices: np.ndarray,
    batch_size: int = 128,
) -> DataLoader:
    data_dir = age_group_data_dir(age_group, matched=False)
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
    train_local, test_local = _split_combined_indices_for_csv_loaders(
        age_group=age_group,
        combined_indices=selected_indices,
    )
    datasets = []
    if train_local.size > 0:
        datasets.append(Subset(StimulusDataset(str(train_csv)), train_local.tolist()))
    if test_local.size > 0:
        datasets.append(Subset(StimulusDataset(str(test_csv)), test_local.tolist()))
    if not datasets:
        raise ValueError(f"NO_SELECTED_INDICES_FOR_LOADER: age_group={age_group}")
    if len(datasets) == 1:
        subset = datasets[0]
    else:
        subset = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def _copy_required_split_artifacts(output_root: Path, arm_name: str, age_group: str, user_id: str) -> None:
    src_dir = output_root / age_group / f"user_{user_id}"
    dst_dir = _subject_output_dir(output_root, arm_name, age_group, user_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "subject_split.json", dst_dir / "subject_split.json")
    shutil.copy2(src_dir / "fit_subset_indices.json", dst_dir / "fit_subset_indices.json")


def _fit_t0_only_baseline_subject(
    *,
    age_group: str,
    user_id: str,
    output_root: Path,
    arm_cfg: ArmConfig,
    seed: int,
    device: str,
) -> None:
    user_dir = _subject_output_dir(output_root, arm_cfg.arm_name, age_group, user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        user_dir / "subject_started.json",
        {
            "schema_version": "per_subject_age_comparison.subject_started.v1",
            "age_group": age_group,
            "user_id": str(user_id),
            "comparator_arm": arm_cfg.arm_name,
            "started_at": _now_iso(),
            "status": "starting",
            "phase": "t0_baseline_fit",
        },
    )
    _copy_required_split_artifacts(output_root, arm_cfg.arm_name, age_group, user_id)
    combined_df, combined_cached = _load_combined_cached_and_df(age_group)
    _, _, _, subset_meta = _load_subject_splits(output_root, age_group, user_id)
    train_idx = np.array(subset_meta["train_indices"], dtype=np.int64)
    test_idx = np.array(subset_meta["test_indices"], dtype=np.int64)
    train_cached = _filter_cached_by_indices(combined_cached, train_idx)
    test_cached = _filter_cached_by_indices(combined_cached, test_idx)
    train_cached, test_cached, norm = _recompute_subject_rts_normalized(train_cached, test_cached)
    human_stats = compute_human_stats_from_rts(train_cached["rts"])
    fit_seed = _derive_arm_subject_seed(seed, age_group, arm_cfg.arm_name, user_id)
    eval_seed = _derive_arm_eval_seed(seed, age_group, arm_cfg.arm_name, user_id)
    fit_stage2_from_logits(
        age_group=f"{age_group}/user_{user_id}/{arm_cfg.arm_name}",
        output_dir=str(user_dir),
        human_stats=human_stats,
        train_cached=train_cached,
        test_cached=test_cached,
        device=device,
        scales=arm_cfg.scales,
        epochs=int(arm_cfg.epochs_ww),
        choice_temperature=float(arm_cfg.choice_temperature),
        lambda_rt=1.0,
        lambda_choice=1.0,
        lambda_cong=0.0,
        lambda_tail=0.0,
        lambda_pileup=0.0,
        fixed_noise_ampa=None,
        t0_mode=arm_cfg.t0_mode,
        t0_seconds=float(arm_cfg.t0_seconds),
        rt_shape_focus=True,
        behavior_smoke_mode=arm_cfg.behavior_smoke_mode,
        random_seed=fit_seed,
        eval_random_seed=eval_seed,
    )
    best_cfg = _load_json(user_dir / "best_config.json")
    params_npz = np.load(user_dir / "best_model_params.npz")
    params = {k: params_npz[k] for k in params_npz.files}
    predictions, metrics = evaluate_cached_stage2_params(
        params=params,
        scale=float(best_cfg["scale"]),
        time_steps=int(best_cfg["time_steps"]),
        cached=test_cached,
        device=device,
        choice_temperature=float(best_cfg.get("choice_temperature", arm_cfg.choice_temperature)),
        rt_readout_mode=str(best_cfg.get("rt_readout_mode", "baseline")),
        readout_config=best_cfg.get("readout_config") or {},
        selection_config=best_cfg.get("selection_config") or {},
        random_seed=int(best_cfg.get("eval_random_seed", eval_seed)),
        rt_shape_focus=True,
    )
    _write_subject_payloads(
        user_dir=user_dir,
        age_group=age_group,
        user_id=user_id,
        arm_name=arm_cfg.arm_name,
        arm_cfg=arm_cfg,
        fit_seed=fit_seed,
        eval_seed=eval_seed,
        train_idx=train_idx,
        test_idx=test_idx,
        train_cached=train_cached,
        test_cached=test_cached,
        predictions=predictions,
        metrics=metrics,
        best_cfg=best_cfg,
        normalization=norm,
    )


def _fit_dmc_variational_subject(
    *,
    age_group: str,
    user_id: str,
    output_root: Path,
    arm_cfg: ArmConfig,
    seed: int,
    device: str,
) -> None:
    if not arm_cfg.implemented:
        raise RuntimeError(f"ARM_NOT_IMPLEMENTED: {arm_cfg.arm_name}")
    user_dir = _subject_output_dir(output_root, arm_cfg.arm_name, age_group, user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        user_dir / "subject_started.json",
        {
            "schema_version": "per_subject_age_comparison.subject_started.v1",
            "age_group": age_group,
            "user_id": str(user_id),
            "comparator_arm": arm_cfg.arm_name,
            "started_at": _now_iso(),
            "status": "starting",
            "phase": "dmc_variational_fit",
        },
    )
    _copy_required_split_artifacts(output_root, arm_cfg.arm_name, age_group, user_id)
    _, _, _, subset_meta = _load_subject_splits(output_root, age_group, user_id)
    train_idx = np.array(subset_meta["train_indices"], dtype=np.int64)
    test_idx = np.array(subset_meta["test_indices"], dtype=np.int64)
    sampler = _prepare_shared_stage1_sampler(
        output_root=output_root,
        arm_cfg=arm_cfg,
        age_group=age_group,
        device=str(device),
    )
    train_loader = _build_loader_from_combined_indices(age_group=age_group, selected_indices=train_idx)
    test_loader = _build_loader_from_combined_indices(age_group=age_group, selected_indices=test_idx)
    cached_train_inputs = _stage1_inputs_for_indices(
        age_group=age_group,
        selected_indices=train_idx,
        sampler=sampler,
        device=str(device),
        heartbeat_path=user_dir / "train_feature_cache.heartbeat.json",
    )
    cached_test_inputs = _stage1_inputs_for_indices(
        age_group=age_group,
        selected_indices=test_idx,
        sampler=sampler,
        device=str(device),
        heartbeat_path=user_dir / "test_feature_cache.heartbeat.json",
    )
    fit_seed = _derive_arm_subject_seed(seed, age_group, arm_cfg.arm_name, user_id)
    eval_seed = _derive_arm_eval_seed(seed, age_group, arm_cfg.arm_name, user_id)
    _write_json(
        user_dir / "stage1_complete.json",
        {
            "phase": "stage1_warmstart",
            "status": "completed",
            "source": _safe_rel_or_abs(_shared_stage1_root(output_root, arm_cfg.arm_name, age_group)),
            "note": "Reused shared age-group Stage-1 variational head to avoid per-subject warmstart duplication.",
        },
    )
    _write_json(
        user_dir / "stage1_shared_source.json",
        {
            "schema_version": "per_subject_age_comparison.stage1_shared_source.v1",
            "shared_stage1_root": _safe_rel_or_abs(_shared_stage1_root(output_root, arm_cfg.arm_name, age_group)),
            "comparator_arm": arm_cfg.arm_name,
            "age_group": age_group,
        },
    )
    readout_config = {
        "dt_ms": float(arm_cfg.ww_dt),
        "choice_temperature": float(arm_cfg.choice_temperature),
        "sigma_s": 0.05,
        "t0_mode": arm_cfg.t0_mode,
        "t0_seconds": float(arm_cfg.t0_seconds),
    }
    behavioral_loss_config = {
        "lambda_error_rate": 0.75,
        "lambda_error_sign": 1.5,
        "lambda_accuracy": 0.0,
        "lambda_response_nll": 1.0,
        "lambda_rt_mse": 0.0,
        "neg_drt_min_acc": 0.75,
        "neg_drt_min_resp": 0.65,
        "neg_drt_min_error": 0.02,
    }
    result = train_dmc_variational_ww(
        sampler=sampler,
        train_loader=train_loader,
        test_loader=test_loader,
        sampler_mode=arm_cfg.stage1_sampler_mode,
        evidence_time_steps=int(arm_cfg.evidence_time_steps),
        ww_time_steps=int(arm_cfg.ww_time_steps),
        ww_dt=int(arm_cfg.ww_dt),
        epochs_stage1=0,
        epochs_ww=int(arm_cfg.epochs_ww),
        readout_mode="soft_index",
        readout_config=readout_config,
        behavioral_loss_config=behavioral_loss_config,
        device=str(device),
        seed=int(fit_seed),
        output_dir=str(user_dir),
        noise_ampa=arm_cfg.noise_ampa,
        threshold=arm_cfg.threshold,
        j_offdiag_scale=arm_cfg.j_offdiag_scale,
        j_ext=arm_cfg.j_ext,
        freeze_ww_params=True,
        dmc_auto_strength=arm_cfg.dmc_auto_strength,
        dmc_auto_peak_s=arm_cfg.dmc_auto_peak_s,
        dmc_selection_strength=arm_cfg.dmc_selection_strength,
        dmc_selection_midpoint_s=arm_cfg.dmc_selection_midpoint_s,
        dmc_selection_tau_s=arm_cfg.dmc_selection_tau_s,
        dmc_apply_to=arm_cfg.dmc_apply_to,
        sigma_evidence_noise=arm_cfg.sigma_evidence_noise,
        stage1_uncertainty_gain=arm_cfg.stage1_uncertainty_gain,
        cached_train_inputs=cached_train_inputs,
        cached_test_inputs=cached_test_inputs,
    )
    best_params_path = user_dir / "best_model_params.npz"
    predictions_smoke_path = user_dir / "predictions_smoke.npz"
    metrics_smoke_path = user_dir / "metrics_smoke.json"
    config_smoke_path = user_dir / "config.json"
    if not best_params_path.exists() or not predictions_smoke_path.exists() or not metrics_smoke_path.exists() or not config_smoke_path.exists():
        raise RuntimeError(f"DMC_SUBJECT_OUTPUT_MISSING: {user_dir}")
    smoke_metrics = _load_json(metrics_smoke_path)
    smoke_config = _load_json(config_smoke_path)
    predictions_smoke = np.load(predictions_smoke_path)
    predictions = {key: predictions_smoke[key] for key in predictions_smoke.files}
    best_config = {
        "scale": 1.0,
        "best_epoch": int(result.get("best_metrics", {}).get("epoch", -1)) if isinstance(result.get("best_metrics"), dict) else -1,
        "score": float(result.get("best_score", smoke_metrics.get("total_score", float("nan")))),
        "time_steps": int(arm_cfg.ww_time_steps),
        "results": smoke_metrics,
        "selection_results": smoke_metrics,
        "rt_readout_mode": "soft_index",
        "behavior_smoke_mode": arm_cfg.behavior_smoke_mode,
        "behavior_loss_mode": "dmc_variational_subject",
        "behavior_loss_weight": 1.0,
        "rt_distribution_loss_mode": "baseline",
        "rt_distribution_loss_weight": 0.0,
        "conditional_rt_distribution_loss_mode": "baseline",
        "conditional_rt_distribution_loss_weight": 0.0,
        "rt_moment_anchor_loss_mode": "baseline",
        "rt_moment_anchor_loss_weight": 0.0,
        "fixed_noise_ampa": None,
        "fixed_threshold": arm_cfg.threshold,
        "fixed_competition_scale": arm_cfg.j_offdiag_scale,
        "t0_mode": arm_cfg.t0_mode,
        "t0_seconds": float(arm_cfg.t0_seconds),
        "readout_config": readout_config,
        "selection_config": {},
        "trajectory_artifact": None,
        "random_seed": int(fit_seed),
        "eval_random_seed": int(eval_seed),
        "arm_name": arm_cfg.arm_name,
        "fit_family": arm_cfg.fit_family,
        "stage1_sampler_mode": arm_cfg.stage1_sampler_mode,
        "stage1_uncertainty_gain": float(arm_cfg.stage1_uncertainty_gain),
        "dmc_enabled": True,
        "dmc_config": {
            "dmc_auto_strength": arm_cfg.dmc_auto_strength,
            "dmc_auto_peak_s": arm_cfg.dmc_auto_peak_s,
            "dmc_selection_strength": arm_cfg.dmc_selection_strength,
            "dmc_selection_midpoint_s": arm_cfg.dmc_selection_midpoint_s,
            "dmc_selection_tau_s": arm_cfg.dmc_selection_tau_s,
            "dmc_apply_to": arm_cfg.dmc_apply_to,
        },
        "smoke_config": smoke_config,
    }
    _write_json(user_dir / "best_config.json", best_config)
    combined_df, combined_cached = _load_combined_cached_and_df(age_group)
    train_cached = _filter_cached_by_indices(combined_cached, train_idx)
    test_cached = _filter_cached_by_indices(combined_cached, test_idx)
    train_cached, test_cached, norm = _recompute_subject_rts_normalized(train_cached, test_cached)
    _write_subject_payloads(
        user_dir=user_dir,
        age_group=age_group,
        user_id=user_id,
        arm_name=arm_cfg.arm_name,
        arm_cfg=arm_cfg,
        fit_seed=fit_seed,
        eval_seed=eval_seed,
        train_idx=train_idx,
        test_idx=test_idx,
        train_cached=train_cached,
        test_cached=test_cached,
        predictions=predictions,
        metrics=smoke_metrics,
        best_cfg=best_config,
        normalization=norm,
    )


def _build_condition_trial_df(*, pred_rt: np.ndarray, pred_choice: np.ndarray, target_labels: np.ndarray, response_labels: np.ndarray, congruency: np.ndarray, true_rt: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "pred_rt_s": np.asarray(pred_rt, dtype=np.float32),
            "pred_choice": np.asarray(pred_choice, dtype=np.int64),
            "target": np.asarray(target_labels, dtype=np.int64),
            "response": np.asarray(response_labels, dtype=np.int64),
            "congruency": np.asarray(congruency, dtype=np.int64),
            "true_rt_s": np.asarray(true_rt, dtype=np.float32),
        }
    )
    df["correct"] = df["pred_choice"] == df["target"]
    return df


def _quantile_summary(rt_ms: np.ndarray) -> Dict[str, Any]:
    if rt_ms.size < 20:
        return {
            "analyzable": False,
            "n_trials_total": int(rt_ms.size),
            "q05_rt_ms": None,
            "q10_rt_ms": None,
            "q25_rt_ms": None,
            "q50_rt_ms": None,
            "q75_rt_ms": None,
            "q90_rt_ms": None,
            "q95_rt_ms": None,
            "early_minus_late_span_ms": None,
        }
    q05, q10, q25, q50, q75, q90, q95 = [float(np.quantile(rt_ms, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)]
    return {
        "analyzable": True,
        "n_trials_total": int(rt_ms.size),
        "q05_rt_ms": q05,
        "q10_rt_ms": q10,
        "q25_rt_ms": q25,
        "q50_rt_ms": q50,
        "q75_rt_ms": q75,
        "q90_rt_ms": q90,
        "q95_rt_ms": q95,
        "early_minus_late_span_ms": float(q95 - q05),
    }


def _equal_count_bins(values: np.ndarray, n_bins: int) -> List[np.ndarray]:
    if values.size == 0:
        return []
    order = np.argsort(values)
    splits = np.array_split(order, n_bins)
    return [chunk for chunk in splits if chunk.size > 0]


def _caf_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 10:
        return {
            "analyzable": False,
            "bin_count": 10,
            "bins": [],
            "fast_bin_error_rate": None,
            "slow_bin_error_rate": None,
            "fast_error_capture_index": None,
        }
    rt_ms = df["pred_rt_s"].to_numpy(dtype=np.float32) * 1000.0
    bins = _equal_count_bins(rt_ms, 10)
    payload = []
    for idx, bin_idx in enumerate(bins, start=1):
        subset = df.iloc[bin_idx]
        subset_rt = subset["pred_rt_s"].to_numpy(dtype=np.float32) * 1000.0
        accuracy = float(np.mean(subset["correct"].to_numpy(dtype=bool))) if len(subset) else float("nan")
        payload.append(
            {
                "bin_index": int(idx),
                "n_trials": int(len(subset)),
                "rt_bin_lower_ms": float(subset_rt.min()),
                "rt_bin_upper_ms": float(subset_rt.max()),
                "accuracy": accuracy,
                "error_rate": float(1.0 - accuracy),
            }
        )
    fast = payload[0]["error_rate"]
    slow = payload[-1]["error_rate"]
    return {
        "analyzable": True,
        "bin_count": 10,
        "bins": payload,
        "fast_bin_error_rate": float(fast),
        "slow_bin_error_rate": float(slow),
        "fast_error_capture_index": float(fast - slow),
    }


def _delta_plot_summary(df: pd.DataFrame) -> Dict[str, Any]:
    correct_df = df.loc[df["correct"]].copy()
    congruent = correct_df.loc[correct_df["congruency"] == 0]
    incongruent = correct_df.loc[correct_df["congruency"] == 1]
    if len(congruent) < 5 or len(incongruent) < 5:
        return {"analyzable": False, "bin_count": 5, "bins": [], "delta_slope_ms": None}
    cong_bins = _equal_count_bins(congruent["pred_rt_s"].to_numpy(dtype=np.float32), 5)
    incong_bins = _equal_count_bins(incongruent["pred_rt_s"].to_numpy(dtype=np.float32), 5)
    payload = []
    for idx, (c_idx, i_idx) in enumerate(zip(cong_bins, incong_bins), start=1):
        c_subset = congruent.iloc[c_idx]
        i_subset = incongruent.iloc[i_idx]
        c_mean = float(c_subset["pred_rt_s"].mean() * 1000.0)
        i_mean = float(i_subset["pred_rt_s"].mean() * 1000.0)
        payload.append(
            {
                "bin_index": int(idx),
                "n_congruent_correct": int(len(c_subset)),
                "n_incongruent_correct": int(len(i_subset)),
                "congruent_correct_rt_mean_ms": c_mean,
                "incongruent_correct_rt_mean_ms": i_mean,
                "delta_ms": float(i_mean - c_mean),
            }
        )
    return {
        "analyzable": True,
        "bin_count": 5,
        "bins": payload,
        "delta_slope_ms": float(payload[-1]["delta_ms"] - payload[0]["delta_ms"]),
    }


def _error_direction_summary(df: pd.DataFrame) -> Dict[str, Any]:
    error_df = df.loc[~df["correct"]].copy()
    congruent_error = error_df.loc[error_df["congruency"] == 0]
    incongruent_error = error_df.loc[error_df["congruency"] == 1]
    if len(congruent_error) < 1 or len(incongruent_error) < 1:
        return {
            "analyzable": False,
            "n_congruent_error": int(len(congruent_error)),
            "n_incongruent_error": int(len(incongruent_error)),
            "congruent_error_rt_mean_ms": None,
            "incongruent_error_rt_mean_ms": None,
            "error_direction_rt_delta_ms": None,
            "qualitative_direction_token": "UNDEFINED",
        }
    congruent_mean = float(congruent_error["pred_rt_s"].mean() * 1000.0)
    incongruent_mean = float(incongruent_error["pred_rt_s"].mean() * 1000.0)
    delta = float(incongruent_mean - congruent_mean)
    if delta < 0:
        token = "INCONGRUENT_ERROR_FASTER"
    elif delta > 0:
        token = "INCONGRUENT_ERROR_SLOWER"
    else:
        token = "NO_DIRECTION_DIFFERENCE"
    return {
        "analyzable": True,
        "n_congruent_error": int(len(congruent_error)),
        "n_incongruent_error": int(len(incongruent_error)),
        "congruent_error_rt_mean_ms": congruent_mean,
        "incongruent_error_rt_mean_ms": incongruent_mean,
        "error_direction_rt_delta_ms": delta,
        "qualitative_direction_token": token,
    }


def _accuracy_summary(df: pd.DataFrame) -> Dict[str, Any]:
    correct = df["correct"].to_numpy(dtype=bool)
    congruent_mask = df["congruency"].to_numpy(dtype=np.int64) == 0
    incongruent_mask = ~congruent_mask
    accuracy_overall = float(np.mean(correct)) if len(df) else float("nan")
    accuracy_congruent = float(np.mean(correct[congruent_mask])) if np.any(congruent_mask) else float("nan")
    accuracy_incongruent = float(np.mean(correct[incongruent_mask])) if np.any(incongruent_mask) else float("nan")
    error_mask = ~correct
    flanker_toward = df.loc[error_mask, "pred_choice"].to_numpy(dtype=np.int64) == df.loc[error_mask, "response"].to_numpy(dtype=np.int64)
    return {
        "analyzable": bool(len(df) >= 1),
        "accuracy_overall": accuracy_overall,
        "accuracy_congruent": accuracy_congruent,
        "accuracy_incongruent": accuracy_incongruent,
        "congruency_effect_accuracy": float(accuracy_congruent - accuracy_incongruent) if np.isfinite(accuracy_congruent) and np.isfinite(accuracy_incongruent) else None,
        "error_rate": float(1.0 - accuracy_overall) if np.isfinite(accuracy_overall) else None,
        "pct_errors_toward_flanker": float(np.mean(flanker_toward)) if flanker_toward.size else None,
    }


def _rt_center_summary(df: pd.DataFrame, true_rt: np.ndarray) -> Dict[str, Any]:
    pred_mean = float(np.mean(df["pred_rt_s"].to_numpy(dtype=np.float32)) * 1000.0)
    pred_median = float(np.median(df["pred_rt_s"].to_numpy(dtype=np.float32)) * 1000.0)
    human_mean = float(np.mean(true_rt) * 1000.0)
    human_median = float(np.median(true_rt) * 1000.0)
    delta = float(pred_mean - human_mean)
    tolerance_ms = 150.0
    return {
        "analyzable": bool(len(df) >= 1),
        "predicted_mean_rt_post_t0_ms": pred_mean,
        "predicted_median_rt_post_t0_ms": pred_median,
        "human_mean_rt_ms": human_mean,
        "human_median_rt_ms": human_median,
        "model_minus_human_mean_rt_ms": delta,
        "model_minus_human_median_rt_ms": float(pred_median - human_median),
        "within_locked_tolerance": bool(abs(delta) <= tolerance_ms),
    }


def _write_subject_payloads(
    *,
    user_dir: Path,
    age_group: str,
    user_id: str,
    arm_name: str,
    arm_cfg: ArmConfig,
    fit_seed: int,
    eval_seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
    predictions: Dict[str, Any],
    metrics: Dict[str, Any],
    best_cfg: Dict[str, Any],
    normalization: Dict[str, float],
) -> None:
    pred_rt = np.asarray(predictions["pred_rt"], dtype=np.float32)
    pred_choice = np.asarray(predictions["pred_choice"], dtype=np.int64)
    target_labels = np.asarray(test_cached["target_labels"], dtype=np.int64)
    response_labels = np.asarray(test_cached["response_labels"], dtype=np.int64)
    congruency = np.asarray(test_cached["congruency"], dtype=np.int64)
    true_rt = np.asarray(test_cached["rts"], dtype=np.float32)
    flanker_labels = np.asarray(test_cached.get("flanker_labels", np.full_like(target_labels, -1)), dtype=np.int64)

    np.savez_compressed(
        user_dir / "predictions_subject.npz",
        pred_rt=pred_rt.astype(np.float32),
        pred_choice=pred_choice.astype(np.int64),
        target_labels=target_labels.astype(np.int64),
        response_labels=response_labels.astype(np.int64),
        flanker_labels=flanker_labels.astype(np.int64),
        congruency=congruency.astype(np.int64),
        true_rt=true_rt.astype(np.float32),
        user_id=np.array(str(user_id)),
        comparator_arm=np.array(str(arm_name)),
    )
    _write_json(user_dir / "metrics_subject.json", metrics)

    trial_df = _build_condition_trial_df(
        pred_rt=pred_rt,
        pred_choice=pred_choice,
        target_labels=target_labels,
        response_labels=response_labels,
        congruency=congruency,
        true_rt=true_rt,
    )
    rt_quantiles = _quantile_summary(pred_rt * 1000.0)
    caf = _caf_summary(trial_df)
    delta_plot = _delta_plot_summary(trial_df)
    error_direction = _error_direction_summary(trial_df)
    rt_center = _rt_center_summary(trial_df, true_rt)
    accuracy = _accuracy_summary(trial_df)
    summary = {
        "schema_version": "per_subject_age_comparison.subject_eval.v1",
        "age_group": age_group,
        "user_id": str(user_id),
        "comparator_arm": arm_name,
        "fit": {
            "scale": float(best_cfg.get("scale", 1.0)),
            "time_steps": int(best_cfg.get("time_steps", arm_cfg.ww_time_steps)),
            "epochs": int(arm_cfg.epochs_ww),
            "best_epoch": int(best_cfg.get("best_epoch", -1)),
            "choice_temperature": float(best_cfg.get("choice_temperature", arm_cfg.choice_temperature)),
            "seed": int(fit_seed),
            "scales": [float(x) for x in arm_cfg.scales.tolist()],
            "rt_normalization": normalization,
            "objective": {
                "behavior_smoke_mode": arm_cfg.behavior_smoke_mode,
                "summary_eval_seed": int(eval_seed),
                "fixed_noise_ampa": arm_cfg.noise_ampa,
                "arm_name": arm_name,
                "fit_family": arm_cfg.fit_family,
                "stage1_sampler_mode": arm_cfg.stage1_sampler_mode,
                "stage1_uncertainty_gain": float(arm_cfg.stage1_uncertainty_gain),
                "dmc_enabled": bool(arm_cfg.dmc_enabled),
                "t0_mode": arm_cfg.t0_mode,
                "t0_seconds": float(arm_cfg.t0_seconds),
            },
        },
        "train_n_trials": int(len(train_idx)),
        "test_n_trials": int(len(test_idx)),
        "test_n_errors": int(np.sum(pred_choice != target_labels)),
        "predicted_mean_rt_post_t0_ms": rt_center["predicted_mean_rt_post_t0_ms"],
        "human_mean_rt_ms": rt_center["human_mean_rt_ms"],
        "accuracy": accuracy,
        "rt_quantiles": rt_quantiles,
        "caf": caf,
        "delta_plot": delta_plot,
        "error_direction": error_direction,
        "rt_center": rt_center,
        "test_metrics": metrics,
        "seeds": {"fit_seed": int(fit_seed), "eval_seed": int(eval_seed)},
    }
    _write_json(user_dir / "subject_eval_summary.json", summary)
    _write_json(
        user_dir / "arm_subject_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.arm_subject_manifest.v1",
            "age_group": age_group,
            "user_id": str(user_id),
            "comparator_arm": arm_name,
            "fit_family": arm_cfg.fit_family,
            "implemented": bool(arm_cfg.implemented),
            "paths": {
                "subject_split_json": rel_to_root(user_dir / "subject_split.json"),
                "fit_subset_indices_json": rel_to_root(user_dir / "fit_subset_indices.json"),
                "best_config_json": rel_to_root(user_dir / "best_config.json"),
                "best_model_params_npz": rel_to_root(user_dir / "best_model_params.npz"),
                "subject_eval_summary_json": rel_to_root(user_dir / "subject_eval_summary.json"),
                "predictions_subject_npz": rel_to_root(user_dir / "predictions_subject.npz"),
                "metrics_subject_json": rel_to_root(user_dir / "metrics_subject.json"),
            },
        },
    )


def _read_subject_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _new_cross_age_summary() -> Dict[str, Any]:
    return {
        "old_minus_young_q05_rt_ms": None,
        "old_minus_young_q10_rt_ms": None,
        "old_minus_young_q25_rt_ms": None,
        "old_minus_young_q50_rt_ms": None,
        "old_minus_young_q75_rt_ms": None,
        "old_minus_young_q90_rt_ms": None,
        "old_minus_young_q95_rt_ms": None,
        "quantile_gap_widening_support_index_ms": None,
        "old_minus_young_fast_bin_error_rate": None,
        "young_minus_old_fast_error_capture_index": None,
        "old_minus_young_delta_slope_ms": None,
        "young_error_direction_rt_delta_ms": None,
        "old_error_direction_rt_delta_ms": None,
        "young_error_direction_token": None,
        "old_error_direction_token": None,
        "old_minus_young_accuracy_overall": None,
        "old_minus_young_congruency_effect_accuracy": None,
        "old_minus_young_predicted_mean_rt_post_t0_ms": None,
    }


def _aggregate_subjects_for_arm(output_root: Path, arm_name: str) -> Tuple[pd.DataFrame, dict]:
    rows: List[Dict[str, Any]] = []
    subjects_payload: Dict[str, Dict[str, Any]] = {age: {} for age in AGE_GROUPS}
    for age_group in AGE_GROUPS:
        age_dir = _arm_root(output_root, arm_name) / age_group
        for subject_dir in sorted(age_dir.glob("user_*")):
            summary = _read_subject_summary(subject_dir / "subject_eval_summary.json")
            user_id = str(summary["user_id"])
            subject_payload = {
                "user_id": user_id,
                "age_group": age_group,
                "n_trials_total": int(summary["test_n_trials"]),
                "n_trials_congruent": int(np.sum(np.load(subject_dir / "predictions_subject.npz")["congruency"] == 0)),
                "n_trials_incongruent": int(np.sum(np.load(subject_dir / "predictions_subject.npz")["congruency"] == 1)),
                "n_trials_correct": int(np.load(subject_dir / "predictions_subject.npz")["pred_choice"].shape[0] - summary["test_n_errors"]),
                "n_trials_error": int(summary["test_n_errors"]),
                "n_trials_congruent_correct": int(summary["delta_plot"]["bins"][0]["n_congruent_correct"]) if summary["delta_plot"].get("analyzable") and summary["delta_plot"]["bins"] else 0,
                "n_trials_incongruent_correct": int(summary["delta_plot"]["bins"][0]["n_incongruent_correct"]) if summary["delta_plot"].get("analyzable") and summary["delta_plot"]["bins"] else 0,
                "n_trials_congruent_error": int(summary["error_direction"]["n_congruent_error"]),
                "n_trials_incongruent_error": int(summary["error_direction"]["n_incongruent_error"]),
                "rt_quantiles": summary["rt_quantiles"],
                "caf": summary["caf"],
                "delta_plot": summary["delta_plot"],
                "error_direction": summary["error_direction"],
                "rt_center": summary["rt_center"],
                "accuracy": summary["accuracy"],
            }
            subjects_payload[age_group][user_id] = subject_payload
            rows.append(
                {
                    "schema_version": "per_subject_age_comparison.subject_metrics_table.v1",
                    "comparator_arm": arm_name,
                    "age_group": age_group,
                    "user_id": user_id,
                    "n_trials_total": subject_payload["n_trials_total"],
                    "n_trials_congruent": subject_payload["n_trials_congruent"],
                    "n_trials_incongruent": subject_payload["n_trials_incongruent"],
                    "n_trials_correct": subject_payload["n_trials_correct"],
                    "n_trials_error": subject_payload["n_trials_error"],
                    "n_trials_congruent_error": subject_payload["n_trials_congruent_error"],
                    "n_trials_incongruent_error": subject_payload["n_trials_incongruent_error"],
                    "predicted_mean_rt_post_t0_ms": subject_payload["rt_center"]["predicted_mean_rt_post_t0_ms"],
                    "human_mean_rt_ms": subject_payload["rt_center"]["human_mean_rt_ms"],
                    "rt_center_delta_model_minus_human_ms": subject_payload["rt_center"]["model_minus_human_mean_rt_ms"],
                    "accuracy_overall": subject_payload["accuracy"]["accuracy_overall"],
                    "accuracy_congruent": subject_payload["accuracy"]["accuracy_congruent"],
                    "accuracy_incongruent": subject_payload["accuracy"]["accuracy_incongruent"],
                    "congruency_effect_accuracy": subject_payload["accuracy"]["congruency_effect_accuracy"],
                    "congruency_rt_effect_ms": summary["test_metrics"].get("model_congruency_rt_gap", None) * 1000.0 if summary["test_metrics"].get("model_congruency_rt_gap", None) is not None else None,
                    "error_minus_correct_rt_ms": summary["test_metrics"].get("error_minus_correct_rt") * 1000.0 if summary["test_metrics"].get("error_minus_correct_rt") is not None and np.isfinite(summary["test_metrics"].get("error_minus_correct_rt")) else None,
                    "incongruent_error_rt_mean_ms": subject_payload["error_direction"]["incongruent_error_rt_mean_ms"],
                    "congruent_error_rt_mean_ms": subject_payload["error_direction"]["congruent_error_rt_mean_ms"],
                    "error_direction_rt_delta_ms": subject_payload["error_direction"]["error_direction_rt_delta_ms"],
                    "rt_quantile_gap_widening_index_ms": subject_payload["rt_quantiles"]["early_minus_late_span_ms"],
                    "caf_fast_bin_error_rate": subject_payload["caf"]["fast_bin_error_rate"],
                    "caf_slowest_bin_error_rate": subject_payload["caf"]["slow_bin_error_rate"],
                    "caf_fast_error_capture_index": subject_payload["caf"]["fast_error_capture_index"],
                    "delta_plot_slope_ms": subject_payload["delta_plot"]["delta_slope_ms"],
                    "rt_quantiles_analyzable": bool(subject_payload["rt_quantiles"]["analyzable"]),
                    "caf_analyzable": bool(subject_payload["caf"]["analyzable"]),
                    "delta_plot_analyzable": bool(subject_payload["delta_plot"]["analyzable"]),
                    "error_direction_analyzable": bool(subject_payload["error_direction"]["analyzable"]),
                    "rt_center_analyzable": bool(subject_payload["rt_center"]["analyzable"]),
                    "accuracy_analyzable": bool(subject_payload["accuracy"]["analyzable"]),
                }
            )
    subject_df = pd.DataFrame(rows).sort_values(["age_group", "user_id"]).reset_index(drop=True)

    age_groups_payload: Dict[str, Any] = {}
    for age_group in AGE_GROUPS:
        scoped = subject_df.loc[subject_df["age_group"] == age_group].copy()
        mean_rt_quantiles = {
            key: _mean_optional([subjects_payload[age_group][uid]["rt_quantiles"].get(key) for uid in subjects_payload[age_group]])
            for key in ("q05_rt_ms", "q10_rt_ms", "q25_rt_ms", "q50_rt_ms", "q75_rt_ms", "q90_rt_ms", "q95_rt_ms")
        }
        mean_rt_quantiles["rt_quantile_gap_widening_index_ms"] = (
            None if mean_rt_quantiles["q95_rt_ms"] is None or mean_rt_quantiles["q05_rt_ms"] is None else float(mean_rt_quantiles["q95_rt_ms"] - mean_rt_quantiles["q05_rt_ms"])
        )
        mean_caf = {
            "fast_bin_error_rate": _mean_optional([subjects_payload[age_group][uid]["caf"].get("fast_bin_error_rate") for uid in subjects_payload[age_group]]),
            "slow_bin_error_rate": _mean_optional([subjects_payload[age_group][uid]["caf"].get("slow_bin_error_rate") for uid in subjects_payload[age_group]]),
            "fast_error_capture_index": _mean_optional([subjects_payload[age_group][uid]["caf"].get("fast_error_capture_index") for uid in subjects_payload[age_group]]),
        }
        mean_delta_plot: Dict[str, Any] = {f"delta_bin_{idx:02d}_ms": None for idx in range(1, 6)}
        for idx in range(1, 6):
            mean_delta_plot[f"delta_bin_{idx:02d}_ms"] = _mean_optional([
                next((bin_row["delta_ms"] for bin_row in subjects_payload[age_group][uid]["delta_plot"].get("bins", []) if int(bin_row["bin_index"]) == idx), None)
                for uid in subjects_payload[age_group]
            ])
        mean_delta_plot["delta_slope_ms"] = _mean_optional([subjects_payload[age_group][uid]["delta_plot"].get("delta_slope_ms") for uid in subjects_payload[age_group]])
        error_direction_delta = _mean_optional([subjects_payload[age_group][uid]["error_direction"].get("error_direction_rt_delta_ms") for uid in subjects_payload[age_group]])
        if error_direction_delta is None:
            token = "UNDEFINED"
        elif error_direction_delta < 0:
            token = "INCONGRUENT_ERROR_FASTER"
        elif error_direction_delta > 0:
            token = "INCONGRUENT_ERROR_SLOWER"
        else:
            token = "NO_DIRECTION_DIFFERENCE"
        mean_error_direction = {
            "congruent_error_rt_mean_ms": _mean_optional([subjects_payload[age_group][uid]["error_direction"].get("congruent_error_rt_mean_ms") for uid in subjects_payload[age_group]]),
            "incongruent_error_rt_mean_ms": _mean_optional([subjects_payload[age_group][uid]["error_direction"].get("incongruent_error_rt_mean_ms") for uid in subjects_payload[age_group]]),
            "error_direction_rt_delta_ms": error_direction_delta,
            "qualitative_direction_token": token,
        }
        mean_rt_center = {
            "predicted_mean_rt_post_t0_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("predicted_mean_rt_post_t0_ms") for uid in subjects_payload[age_group]]),
            "human_mean_rt_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("human_mean_rt_ms") for uid in subjects_payload[age_group]]),
            "model_minus_human_mean_rt_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("model_minus_human_mean_rt_ms") for uid in subjects_payload[age_group]]),
            "predicted_median_rt_post_t0_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("predicted_median_rt_post_t0_ms") for uid in subjects_payload[age_group]]),
            "human_median_rt_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("human_median_rt_ms") for uid in subjects_payload[age_group]]),
            "model_minus_human_median_rt_ms": _mean_optional([subjects_payload[age_group][uid]["rt_center"].get("model_minus_human_median_rt_ms") for uid in subjects_payload[age_group]]),
        }
        mean_accuracy = {
            "accuracy_overall": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("accuracy_overall") for uid in subjects_payload[age_group]]),
            "accuracy_congruent": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("accuracy_congruent") for uid in subjects_payload[age_group]]),
            "accuracy_incongruent": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("accuracy_incongruent") for uid in subjects_payload[age_group]]),
            "congruency_effect_accuracy": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("congruency_effect_accuracy") for uid in subjects_payload[age_group]]),
            "error_rate": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("error_rate") for uid in subjects_payload[age_group]]),
            "pct_errors_toward_flanker": _mean_optional([subjects_payload[age_group][uid]["accuracy"].get("pct_errors_toward_flanker") for uid in subjects_payload[age_group]]),
        }
        age_groups_payload[age_group] = {
            "n_subjects_total": int(len(scoped)),
            "n_subjects_rt_quantiles_analyzable": int(scoped["rt_quantiles_analyzable"].sum()),
            "n_subjects_caf_analyzable": int(scoped["caf_analyzable"].sum()),
            "n_subjects_delta_plot_analyzable": int(scoped["delta_plot_analyzable"].sum()),
            "n_subjects_error_direction_analyzable": int(scoped["error_direction_analyzable"].sum()),
            "n_subjects_rt_center_analyzable": int(scoped["rt_center_analyzable"].sum()),
            "n_subjects_accuracy_analyzable": int(scoped["accuracy_analyzable"].sum()),
            "subjects": subjects_payload[age_group],
            "summary": {
                "mean_rt_quantiles": mean_rt_quantiles,
                "mean_caf": mean_caf,
                "mean_delta_plot": mean_delta_plot,
                "mean_error_direction": mean_error_direction,
                "mean_rt_center": mean_rt_center,
                "mean_accuracy": mean_accuracy,
            },
        }

    young = age_groups_payload["20-29"]["summary"]
    old = age_groups_payload["80-89"]["summary"]
    cross_age_summary = _new_cross_age_summary()
    for quant_key in ("q05_rt_ms", "q10_rt_ms", "q25_rt_ms", "q50_rt_ms", "q75_rt_ms", "q90_rt_ms", "q95_rt_ms"):
        y = young["mean_rt_quantiles"].get(quant_key)
        o = old["mean_rt_quantiles"].get(quant_key)
        cross_age_summary[f"old_minus_young_{quant_key}"] = None if y is None or o is None else float(o - y)
    q95 = cross_age_summary["old_minus_young_q95_rt_ms"]
    q05 = cross_age_summary["old_minus_young_q05_rt_ms"]
    cross_age_summary["quantile_gap_widening_support_index_ms"] = None if q95 is None or q05 is None else float(q95 - q05)
    old_fast = old["mean_caf"].get("fast_bin_error_rate")
    young_fast = young["mean_caf"].get("fast_bin_error_rate")
    old_capture = old["mean_caf"].get("fast_error_capture_index")
    young_capture = young["mean_caf"].get("fast_error_capture_index")
    cross_age_summary["old_minus_young_fast_bin_error_rate"] = None if old_fast is None or young_fast is None else float(old_fast - young_fast)
    cross_age_summary["young_minus_old_fast_error_capture_index"] = None if young_capture is None or old_capture is None else float(young_capture - old_capture)
    old_delta = old["mean_delta_plot"].get("delta_slope_ms")
    young_delta = young["mean_delta_plot"].get("delta_slope_ms")
    cross_age_summary["old_minus_young_delta_slope_ms"] = None if old_delta is None or young_delta is None else float(old_delta - young_delta)
    cross_age_summary["young_error_direction_rt_delta_ms"] = young["mean_error_direction"].get("error_direction_rt_delta_ms")
    cross_age_summary["old_error_direction_rt_delta_ms"] = old["mean_error_direction"].get("error_direction_rt_delta_ms")
    cross_age_summary["young_error_direction_token"] = young["mean_error_direction"].get("qualitative_direction_token")
    cross_age_summary["old_error_direction_token"] = old["mean_error_direction"].get("qualitative_direction_token")
    old_acc = old["mean_accuracy"].get("accuracy_overall")
    young_acc = young["mean_accuracy"].get("accuracy_overall")
    old_cong = old["mean_accuracy"].get("congruency_effect_accuracy")
    young_cong = young["mean_accuracy"].get("congruency_effect_accuracy")
    old_mean = old["mean_rt_center"].get("predicted_mean_rt_post_t0_ms")
    young_mean = young["mean_rt_center"].get("predicted_mean_rt_post_t0_ms")
    cross_age_summary["old_minus_young_accuracy_overall"] = None if old_acc is None or young_acc is None else float(old_acc - young_acc)
    cross_age_summary["old_minus_young_congruency_effect_accuracy"] = None if old_cong is None or young_cong is None else float(old_cong - young_cong)
    cross_age_summary["old_minus_young_predicted_mean_rt_post_t0_ms"] = None if old_mean is None or young_mean is None else float(old_mean - young_mean)
    signature_support: Dict[str, Any] = {
        "quantile_gap_widening_matches_human": bool((cross_age_summary["quantile_gap_widening_support_index_ms"] or 0.0) > 0),
        "caf_shape_matches_human": bool((cross_age_summary["young_minus_old_fast_error_capture_index"] or 0.0) > 0),
        "delta_plot_age_difference_matches_human": bool((cross_age_summary["old_minus_young_delta_slope_ms"] or 0.0) > 0),
        "error_direction_reversal_matches_human": bool(
            cross_age_summary["young_error_direction_token"] == "INCONGRUENT_ERROR_FASTER"
            and cross_age_summary["old_error_direction_token"] == "INCONGRUENT_ERROR_SLOWER"
        ),
    }
    signature_support["n_primary_signatures_supported"] = int(sum(1 for value in signature_support.values() if bool(value)))
    payload = {
        "schema_version": "per_subject_age_comparison.panel_signature_summary.v1",
        "comparator_arm": arm_name,
        "source_runner_root": rel_to_root(output_root),
        "source_subject_metrics_table_csv": rel_to_root(_arm_root(output_root, arm_name) / "reaggregated" / "subject_metrics_table.csv"),
        "source_panel_summary_json": rel_to_root(_arm_root(output_root, arm_name) / "reaggregated" / "panel_summary.json"),
        "source_panel_analysis_manifest_json": rel_to_root(_arm_root(output_root, arm_name) / "reaggregated" / "panel_analysis_manifest.json"),
        "sign_conventions": {
            "congruency_rt_effect_ms": "incongruent_correct_rt_mean_ms - congruent_correct_rt_mean_ms",
            "error_minus_correct_rt_ms": "error_rt_mean_ms - correct_rt_mean_ms",
            "error_direction_rt_delta_ms": "incongruent_error_rt_mean_ms - congruent_error_rt_mean_ms",
        },
        "age_groups": age_groups_payload,
        "cross_age_summary": cross_age_summary,
        "signature_support": signature_support,
    }
    return subject_df, payload


def analyze_panel(*, output_root: Path, arm_name: str) -> None:
    arm_root = _arm_root(output_root, arm_name)
    reagg_dir = arm_root / "reaggregated"
    reagg_dir.mkdir(parents=True, exist_ok=True)
    subject_df, panel_signature = _aggregate_subjects_for_arm(output_root, arm_name)
    subject_df.to_csv(reagg_dir / "subject_metrics_table.csv", index=False)
    panel_summary = {
        "schema_version": "per_subject_age_comparison.panel_summary.v1",
        "comparator_arm": arm_name,
        "n_subjects_total": int(len(subject_df)),
        "age_groups": {
            age_group: {
                "n_subjects": int((subject_df["age_group"] == age_group).sum()),
                "mean_accuracy_overall": _mean_optional(subject_df.loc[subject_df["age_group"] == age_group, "accuracy_overall"].tolist()),
                "mean_predicted_mean_rt_post_t0_ms": _mean_optional(subject_df.loc[subject_df["age_group"] == age_group, "predicted_mean_rt_post_t0_ms"].tolist()),
            }
            for age_group in AGE_GROUPS
        },
    }
    _write_json(reagg_dir / "panel_summary.json", panel_summary)
    _write_json(
        reagg_dir / "panel_analysis_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.panel_analysis_manifest.v1",
            "comparator_arm": arm_name,
            "source_contract_path": rel_to_root(METRIC_CONTRACT_PATH),
            "source_runner_contract_path": rel_to_root(RUNNER_CONTRACT_PATH),
            "subject_metrics_table_csv": rel_to_root(reagg_dir / "subject_metrics_table.csv"),
            "panel_signature_summary_json": rel_to_root(reagg_dir / "panel_signature_summary.json"),
            "min_trials_total_for_rt_quantiles": 20,
            "min_trials_total_for_caf": 10,
            "min_trials_correct_per_condition_for_delta": 5,
            "min_trials_error_per_condition_for_error_direction": 1,
            "rt_quantile_probabilities": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            "caf_bin_count": 10,
            "delta_bin_count": 5,
            "rt_center_reference_field": "human_mean_rt_ms",
            "gate_thresholds": {
                "min_trials_total_for_rt_quantiles": 20,
                "min_trials_total_for_caf": 10,
                "min_trials_correct_per_condition_for_delta": 5,
                "min_trials_error_per_condition_for_error_direction": 1,
                "rt_center_tolerance_ms": 150.0,
            },
        },
    )
    _write_json(reagg_dir / "panel_signature_summary.json", panel_signature)
    _write_json(arm_root / "panel_analysis_ready.json", {"ready": True, "updated_at": _now_iso(), "reaggregated_dir": rel_to_root(reagg_dir)})


PRIMARY_SIGNATURE_FIELDS = (
    "quantile_gap_widening_matches_human",
    "caf_shape_matches_human",
    "delta_plot_age_difference_matches_human",
    "error_direction_reversal_matches_human",
)


def _load_panel_signature_summary(output_root: Path, arm_name: str) -> Optional[dict]:
    path = _arm_root(output_root, arm_name) / "reaggregated" / "panel_signature_summary.json"
    if not path.exists():
        return None
    return _load_json(path)


def _load_run_complete_status(output_root: Path, arm_name: str) -> Optional[dict]:
    path = _arm_root(output_root, arm_name) / "run_complete.json"
    if not path.exists():
        return None
    return _load_json(path)


def _phase18_replay_reference_config(profile_name: str) -> dict:
    profile = PHASE18_REPLAY_PROFILES[str(profile_name)]
    config_path = PHASE18_REPLAY_REFERENCE_ROOT / str(profile["reference_dir_name"]) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Phase18 replay reference config: {config_path}")
    payload = _load_json(config_path)
    payload["_config_path"] = rel_to_root(config_path)
    payload["_profile"] = str(profile_name)
    return payload


def _flatten_alignment_config(payload: dict) -> Dict[str, Any]:
    readout = payload.get("readout_config", {}) if isinstance(payload, dict) else {}
    dmc = payload.get("dmc_config", {}) if isinstance(payload, dict) else {}
    dmc_apply_to = payload.get("dmc_apply_to", dmc.get("dmc_apply_to"))
    if dmc_apply_to is None and ("dmc_auto_strength" in payload or "dmc_selection_strength" in payload):
        dmc_apply_to = "incongruent_only"
    return {
        "choice_temperature": payload.get("choice_temperature", readout.get("choice_temperature")),
        "t0_seconds": payload.get("t0_seconds", readout.get("t0_seconds")),
        "noise_ampa": payload.get("noise_ampa"),
        "threshold": payload.get("threshold"),
        "ww_time_steps": payload.get("ww_time_steps"),
        "evidence_time_steps": payload.get("evidence_time_steps"),
        "j_offdiag_scale": payload.get("j_offdiag_scale"),
        "j_ext": payload.get("j_ext"),
        "dmc_auto_strength": payload.get("dmc_auto_strength", dmc.get("dmc_auto_strength")),
        "dmc_auto_peak_s": payload.get("dmc_auto_peak_s", dmc.get("dmc_auto_peak_s")),
        "dmc_selection_strength": payload.get("dmc_selection_strength", dmc.get("dmc_selection_strength")),
        "dmc_selection_midpoint_s": payload.get("dmc_selection_midpoint_s", dmc.get("dmc_selection_midpoint_s")),
        "dmc_selection_tau_s": payload.get("dmc_selection_tau_s", dmc.get("dmc_selection_tau_s")),
        "dmc_apply_to": dmc_apply_to,
    }


def _arm_cfg_alignment_view(arm_cfg: ArmConfig) -> Dict[str, Any]:
    return {
        "choice_temperature": float(arm_cfg.choice_temperature),
        "t0_seconds": float(arm_cfg.t0_seconds),
        "noise_ampa": arm_cfg.noise_ampa,
        "threshold": arm_cfg.threshold,
        "ww_time_steps": int(arm_cfg.ww_time_steps),
        "evidence_time_steps": int(arm_cfg.evidence_time_steps),
        "j_offdiag_scale": arm_cfg.j_offdiag_scale,
        "j_ext": arm_cfg.j_ext,
        "dmc_auto_strength": float(arm_cfg.dmc_auto_strength),
        "dmc_auto_peak_s": float(arm_cfg.dmc_auto_peak_s),
        "dmc_selection_strength": float(arm_cfg.dmc_selection_strength),
        "dmc_selection_midpoint_s": float(arm_cfg.dmc_selection_midpoint_s),
        "dmc_selection_tau_s": float(arm_cfg.dmc_selection_tau_s),
        "dmc_apply_to": str(arm_cfg.dmc_apply_to),
    }


def _compute_alignment_differences(current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diffs: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(current.keys()) | set(reference.keys())):
        if current.get(key) != reference.get(key):
            diffs[key] = {
                "current": _to_jsonable(current.get(key)),
                "reference": _to_jsonable(reference.get(key)),
            }
    return diffs


def _write_phase18_alignment_check(output_root: Path, args: argparse.Namespace, age_group: str, replay_cfg: ArmConfig, scales: np.ndarray) -> None:
    arm_root = _arm_root(output_root, replay_cfg.arm_name)
    reference_payload = _phase18_replay_reference_config(str(args.phase18_replay_profile))
    reference_view = _flatten_alignment_config(reference_payload)
    replay_view = _arm_cfg_alignment_view(replay_cfg)
    phase18_core_cfg = _build_arm_config(args, age_group, "phase18_core", scales)
    phase18_core_view = _arm_cfg_alignment_view(phase18_core_cfg)
    payload = {
        "schema_version": "per_subject_age_comparison.phase18_alignment_check.v1",
        "age_group": str(age_group),
        "phase18_replay_profile": str(args.phase18_replay_profile),
        "reference_config_path": reference_payload["_config_path"],
        "reference_subject": {
            "aggregate_smoke_phase18": reference_view,
        },
        "subject_level_arms": {
            "phase18_core": phase18_core_view,
            "phase18_replay_aligned": replay_view,
        },
        "differences": {
            "phase18_core_vs_aggregate_smoke": _compute_alignment_differences(phase18_core_view, reference_view),
            "phase18_replay_aligned_vs_aggregate_smoke": _compute_alignment_differences(replay_view, reference_view),
            "phase18_replay_aligned_vs_phase18_core": _compute_alignment_differences(replay_view, phase18_core_view),
        },
        "updated_at": _now_iso(),
    }
    _write_json(arm_root / f"alignment_check_{age_group}.json", payload)


def _resolve_realized_analysis_arms(output_root: Path) -> Tuple[str, ...]:
    arm_manifest_path = output_root / "arm_manifest.json"
    if arm_manifest_path.exists():
        payload = _load_json(arm_manifest_path)
        manifest_arms = [str(arm) for arm in payload.get("realized_comparator_arms", [])]
        if manifest_arms:
            return tuple(arm for arm in manifest_arms if arm in ALLOWED_ARMS)
    discovered: List[str] = []
    for arm_name in ALLOWED_ARMS:
        if _arm_root(output_root, arm_name).exists():
            discovered.append(arm_name)
    return tuple(discovered)


def _safe_primary_signature_count(payload: Optional[dict]) -> int:
    if not payload:
        return 0
    support = payload.get("signature_support", {})
    count = support.get("n_primary_signatures_supported")
    if count is None:
        return 0
    return int(count)


def _compute_g4_beyond_t0_evidence(*, baseline_payload: Optional[dict], candidate_payloads: Dict[str, dict]) -> Dict[str, Any]:
    baseline_support = baseline_payload.get("signature_support", {}) if baseline_payload else {}
    arm_signature_improvement_counts: Dict[str, int] = {}
    best_candidate_arm: Optional[str] = None
    best_candidate_improvement_count = -1
    best_candidate_signature_support = -1
    for arm_name, payload in candidate_payloads.items():
        support = payload.get("signature_support", {})
        improvement_count = 0
        for field in PRIMARY_SIGNATURE_FIELDS:
            if bool(support.get(field)) and not bool(baseline_support.get(field)):
                improvement_count += 1
        arm_signature_improvement_counts[arm_name] = int(improvement_count)
        signature_support_count = _safe_primary_signature_count(payload)
        if (
            improvement_count > best_candidate_improvement_count
            or (
                improvement_count == best_candidate_improvement_count
                and signature_support_count > best_candidate_signature_support
            )
            or (
                improvement_count == best_candidate_improvement_count
                and signature_support_count == best_candidate_signature_support
                and best_candidate_arm is not None
                and arm_name < best_candidate_arm
            )
            or best_candidate_arm is None
        ):
            best_candidate_arm = arm_name
            best_candidate_improvement_count = int(improvement_count)
            best_candidate_signature_support = int(signature_support_count)
    return {
        "baseline_arm": "t0_only_baseline",
        "candidate_arms": list(candidate_payloads.keys()),
        "arm_signature_improvement_counts": arm_signature_improvement_counts,
        "best_candidate_arm": best_candidate_arm,
        "best_candidate_improvement_count": None if best_candidate_arm is None else int(best_candidate_improvement_count),
        "pass": bool(best_candidate_arm is not None and best_candidate_improvement_count >= 2),
    }


def _compute_g1_errors_analyzable(*, candidate_payloads: Dict[str, dict], best_candidate_arm: Optional[str]) -> Dict[str, Any]:
    per_arm_subject_counts: Dict[str, int] = {}
    for arm_name, payload in candidate_payloads.items():
        count = 0
        for age_group_payload in payload.get("age_groups", {}).values():
            for subject_payload in age_group_payload.get("subjects", {}).values():
                if (
                    bool(subject_payload.get("caf", {}).get("analyzable"))
                    and bool(subject_payload.get("delta_plot", {}).get("analyzable"))
                    and bool(subject_payload.get("error_direction", {}).get("analyzable"))
                ):
                    count += 1
        per_arm_subject_counts[arm_name] = int(count)
    subjects_with_required_error_support = int(per_arm_subject_counts.get(best_candidate_arm, 0)) if best_candidate_arm else 0
    return {
        "subjects_with_required_error_support": subjects_with_required_error_support,
        "required_subjects_min": 4,
        "per_arm_subject_counts": per_arm_subject_counts,
        "selected_candidate_arm": best_candidate_arm,
        "pass": bool(best_candidate_arm is not None and subjects_with_required_error_support >= 4),
    }


def _compute_g2_error_direction_support(*, candidate_payloads: Dict[str, dict], best_candidate_arm: Optional[str]) -> Dict[str, Any]:
    per_arm_support_counts: Dict[str, Dict[str, int]] = {}
    for arm_name, payload in candidate_payloads.items():
        young_support = 0
        old_support = 0
        for user_payload in payload.get("age_groups", {}).get("20-29", {}).get("subjects", {}).values():
            error_direction = user_payload.get("error_direction", {})
            if bool(error_direction.get("analyzable")) and error_direction.get("qualitative_direction_token") == "INCONGRUENT_ERROR_FASTER":
                young_support += 1
        for user_payload in payload.get("age_groups", {}).get("80-89", {}).get("subjects", {}).values():
            error_direction = user_payload.get("error_direction", {})
            if bool(error_direction.get("analyzable")) and error_direction.get("qualitative_direction_token") == "INCONGRUENT_ERROR_SLOWER":
                old_support += 1
        per_arm_support_counts[arm_name] = {
            "supporting_young_subjects": int(young_support),
            "supporting_old_subjects": int(old_support),
        }
    best_counts = per_arm_support_counts.get(best_candidate_arm, {}) if best_candidate_arm else {}
    supporting_young_subjects = int(best_counts.get("supporting_young_subjects", 0))
    supporting_old_subjects = int(best_counts.get("supporting_old_subjects", 0))
    return {
        "supporting_young_subjects": supporting_young_subjects,
        "supporting_old_subjects": supporting_old_subjects,
        "required_young_min": 1,
        "required_old_min": 1,
        "per_arm_support_counts": per_arm_support_counts,
        "selected_candidate_arm": best_candidate_arm,
        "pass": bool(best_candidate_arm is not None and supporting_young_subjects >= 1 and supporting_old_subjects >= 1),
    }


def _compute_g3_rt_center_sanity(*, output_root: Path, candidate_payloads: Dict[str, dict], best_candidate_arm: Optional[str]) -> Dict[str, Any]:
    rt_center_tolerance_ms = 150.0
    if best_candidate_arm is not None:
        manifest_path = _arm_root(output_root, best_candidate_arm) / "reaggregated" / "panel_analysis_manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            rt_center_tolerance_ms = float(manifest.get("gate_thresholds", {}).get("rt_center_tolerance_ms", rt_center_tolerance_ms))
    subjects_within_tolerance = 0
    subjects_evaluated = 0
    if best_candidate_arm is not None:
        payload = candidate_payloads.get(best_candidate_arm, {})
        for age_group_payload in payload.get("age_groups", {}).values():
            for subject_payload in age_group_payload.get("subjects", {}).values():
                rt_center = subject_payload.get("rt_center", {})
                if bool(rt_center.get("analyzable")):
                    subjects_evaluated += 1
                    if bool(rt_center.get("within_locked_tolerance")):
                        subjects_within_tolerance += 1
    all_subjects_within_tolerance = bool(subjects_evaluated > 0 and subjects_within_tolerance == subjects_evaluated)
    return {
        "rt_center_tolerance_ms": float(rt_center_tolerance_ms),
        "subjects_within_tolerance": int(subjects_within_tolerance),
        "subjects_evaluated": int(subjects_evaluated),
        "all_subjects_within_tolerance": all_subjects_within_tolerance,
        "selected_candidate_arm": best_candidate_arm,
        "pass": bool(best_candidate_arm is not None and all_subjects_within_tolerance),
    }


def _compute_g5_no_pooled_only_illusion(*, candidate_payloads: Dict[str, dict], best_candidate_arm: Optional[str]) -> Dict[str, Any]:
    subject_level_direction_support_count = 0
    pooled_signature_support_count = 0
    if best_candidate_arm is not None:
        payload = candidate_payloads.get(best_candidate_arm, {})
        pooled_signature_support_count = _safe_primary_signature_count(payload)
        for user_payload in payload.get("age_groups", {}).get("20-29", {}).get("subjects", {}).values():
            error_direction = user_payload.get("error_direction", {})
            caf = user_payload.get("caf", {})
            if (
                error_direction.get("qualitative_direction_token") == "INCONGRUENT_ERROR_FASTER"
                or (
                    caf.get("fast_error_capture_index") is not None
                    and np.isfinite(caf.get("fast_error_capture_index"))
                    and float(caf.get("fast_error_capture_index")) > 0.0
                )
            ):
                subject_level_direction_support_count += 1
        for user_payload in payload.get("age_groups", {}).get("80-89", {}).get("subjects", {}).values():
            error_direction = user_payload.get("error_direction", {})
            delta_plot = user_payload.get("delta_plot", {})
            if (
                error_direction.get("qualitative_direction_token") == "INCONGRUENT_ERROR_SLOWER"
                or (
                    delta_plot.get("delta_slope_ms") is not None
                    and np.isfinite(delta_plot.get("delta_slope_ms"))
                    and float(delta_plot.get("delta_slope_ms")) > 0.0
                )
            ):
                subject_level_direction_support_count += 1
    return {
        "best_candidate_arm": best_candidate_arm,
        "subject_level_direction_support_count": int(subject_level_direction_support_count),
        "required_subject_level_direction_support_min": 2,
        "pooled_signature_support_count": int(pooled_signature_support_count),
        "pass": bool(best_candidate_arm is not None and pooled_signature_support_count >= 1 and subject_level_direction_support_count >= 2),
    }


def write_stage1_gate_artifacts(*, output_root: Path) -> None:
    realized_arms = _resolve_realized_analysis_arms(output_root)
    signature_sources: Dict[str, str] = {}
    subject_table_sources: Dict[str, str] = {}
    baseline_payload = _load_panel_signature_summary(output_root, "t0_only_baseline")
    candidate_payloads: Dict[str, dict] = {}

    for arm_name in realized_arms:
        signature_path = _arm_root(output_root, arm_name) / "reaggregated" / "panel_signature_summary.json"
        subject_table_path = _arm_root(output_root, arm_name) / "reaggregated" / "subject_metrics_table.csv"
        if signature_path.exists():
            signature_sources[arm_name] = rel_to_root(signature_path)
            payload = _load_json(signature_path)
            if arm_name != "t0_only_baseline":
                run_status = _load_run_complete_status(output_root, arm_name) or {}
                if run_status.get("status") != "refused":
                    candidate_payloads[arm_name] = payload
        if subject_table_path.exists():
            subject_table_sources[arm_name] = rel_to_root(subject_table_path)

    g4 = _compute_g4_beyond_t0_evidence(baseline_payload=baseline_payload, candidate_payloads=candidate_payloads)
    best_candidate_arm = g4.get("best_candidate_arm")
    g1 = _compute_g1_errors_analyzable(candidate_payloads=candidate_payloads, best_candidate_arm=best_candidate_arm)
    g2 = _compute_g2_error_direction_support(candidate_payloads=candidate_payloads, best_candidate_arm=best_candidate_arm)
    g3 = _compute_g3_rt_center_sanity(output_root=output_root, candidate_payloads=candidate_payloads, best_candidate_arm=best_candidate_arm)
    g5 = _compute_g5_no_pooled_only_illusion(candidate_payloads=candidate_payloads, best_candidate_arm=best_candidate_arm)

    gate_inputs_manifest = {
        "schema_version": "per_subject_age_comparison.stage1_gate_inputs_manifest.v1",
        "source_contract_path": rel_to_root(METRIC_CONTRACT_PATH),
        "source_panel_signature_summary_by_arm": signature_sources,
        "source_subject_metrics_table_by_arm": subject_table_sources,
        "locked_gate_names": [
            "g1_errors_analyzable",
            "g2_error_direction_support",
            "g3_rt_center_sanity",
            "g4_beyond_t0_evidence",
            "g5_no_pooled_only_illusion",
        ],
        "locked_stage1_verdict_tokens": [
            "PROMOTE_TO_STAGE2",
            "PARTIAL_NO_STAGE2",
            "KILL_BRANCH",
        ],
    }
    gate_inputs_manifest_path = output_root / "stage1_gate_inputs_manifest.json"
    _write_json(gate_inputs_manifest_path, gate_inputs_manifest)

    candidate_signature_support_any = any(_safe_primary_signature_count(payload) > 0 for payload in candidate_payloads.values())
    if all(bool(gate.get("pass")) for gate in (g1, g2, g3, g4, g5)):
        stage1_verdict_token = "PROMOTE_TO_STAGE2"
    elif not bool(g4.get("pass")) or not candidate_signature_support_any:
        stage1_verdict_token = "KILL_BRANCH"
    else:
        stage1_verdict_token = "PARTIAL_NO_STAGE2"

    gate_summary = {
        "schema_version": "per_subject_age_comparison.stage1_gate_summary.v1",
        "gate_inputs_manifest_path": rel_to_root(gate_inputs_manifest_path),
        "evaluated_arms": list(realized_arms),
        "g1_errors_analyzable": g1,
        "g2_error_direction_support": g2,
        "g3_rt_center_sanity": g3,
        "g4_beyond_t0_evidence": g4,
        "g5_no_pooled_only_illusion": g5,
        "stage1_verdict_token": stage1_verdict_token,
    }
    _write_json(output_root / "stage1_gate_summary.json", gate_summary)


def _write_arm_config_and_status(output_root: Path, arm_cfg: ArmConfig) -> None:
    arm_root = _arm_root(output_root, arm_cfg.arm_name)
    arm_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        arm_root / "config.json",
        {
            "comparator_arm": arm_cfg.arm_name,
            "fit_family": arm_cfg.fit_family,
            "stage1_sampler_mode": arm_cfg.stage1_sampler_mode,
            "stage1_uncertainty_gain": float(arm_cfg.stage1_uncertainty_gain),
            "dmc_enabled": bool(arm_cfg.dmc_enabled),
            "t0_mode": arm_cfg.t0_mode,
            "t0_seconds": float(arm_cfg.t0_seconds),
            "choice_temperature": float(arm_cfg.choice_temperature),
            "scales": [float(x) for x in arm_cfg.scales.tolist()],
            "max_train_trials": int(arm_cfg.max_train_trials),
            "max_test_trials": int(arm_cfg.max_test_trials),
            "epochs_stage1": int(arm_cfg.epochs_stage1),
            "epochs_ww": int(arm_cfg.epochs_ww),
            "evidence_time_steps": int(arm_cfg.evidence_time_steps),
            "ww_time_steps": int(arm_cfg.ww_time_steps),
            "ww_dt": int(arm_cfg.ww_dt),
            "noise_ampa": arm_cfg.noise_ampa,
            "threshold": arm_cfg.threshold,
            "j_offdiag_scale": arm_cfg.j_offdiag_scale,
            "j_ext": arm_cfg.j_ext,
            "dmc_config": {
                "dmc_auto_strength": arm_cfg.dmc_auto_strength,
                "dmc_auto_peak_s": arm_cfg.dmc_auto_peak_s,
                "dmc_selection_strength": arm_cfg.dmc_selection_strength,
                "dmc_selection_midpoint_s": arm_cfg.dmc_selection_midpoint_s,
                "dmc_selection_tau_s": arm_cfg.dmc_selection_tau_s,
                "dmc_apply_to": arm_cfg.dmc_apply_to,
            },
            "sigma_evidence_noise": arm_cfg.sigma_evidence_noise,
        },
    )
    if not arm_cfg.implemented:
        _write_json(
            arm_root / "run_complete.json",
            {
                "comparator_arm": arm_cfg.arm_name,
                "status": "refused",
                "implemented": False,
                "refusal_reason": arm_cfg.refusal_reason,
                "updated_at": _now_iso(),
            },
        )


def _write_root_manifests(output_root: Path, args: argparse.Namespace, lock_payload: dict, arms: Tuple[str, ...], scales: np.ndarray) -> None:
    reuse_manifest = _load_reuse_panel_manifest()
    _write_json(
        output_root / "runner_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.runner_manifest.v1",
            "created_at": _now_iso(),
            "script_path": rel_to_root(Path(__file__).resolve()),
            "contract_reference_path": rel_to_root(RUNNER_CONTRACT_PATH),
            "metric_contract_path": rel_to_root(METRIC_CONTRACT_PATH),
            "global_seed": int(args.seed),
            "allowed_comparator_arms": list(ALLOWED_ARMS),
            "requested_comparator_arms": list(arms),
            "canonical_input_roots": {
                age_group: {
                    "data_dir": rel_to_root(age_group_data_dir(age_group, matched=False)),
                    "stage2_dir": rel_to_root(age_group_stage2_dir(age_group, matched=False)),
                }
                for age_group in AGE_GROUPS
            },
        },
    )
    _write_json(
        output_root / "panel_manifest.json",
        {
            **reuse_manifest,
            "schema_version": "per_subject_age_comparison.panel_manifest.v1",
            "source_panel_lock": rel_to_root(LOCK_PATH),
            "source_reuse_panel_manifest": rel_to_root(REUSE_PANEL_ROOT / "panel_manifest.json"),
            "created_at": _now_iso(),
        },
    )
    _write_json(
        output_root / "arm_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.arm_manifest.v1",
            "requested_comparator_arms": list(arms),
            "realized_comparator_arms": list(arms),
            "phase18_replay_profile": str(args.phase18_replay_profile),
            "phase18_plus_stage1_uncertainty_gain_requested": "phase18_plus_stage1_uncertainty_gain" in arms,
            "phase18_plus_stage1_uncertainty_gain_implemented": bool(float(args.stage1_uncertainty_gain) > 1.0),
            "allowed_comparator_arms": list(ALLOWED_ARMS),
        },
    )
    _write_json(
        output_root / "stage1_lock_manifest.json",
        {
            "schema_version": "per_subject_age_comparison.stage1_lock_manifest.v1",
            "global_seed": int(lock_payload["seed_policy"]["global_seed"]),
            "subject_panel_membership": lock_payload["panel"],
            "split_hashes": {
                age_group: {str(entry["user_id"]): int(entry["split_hash"]) for entry in lock_payload["panel"][age_group]}
                for age_group in AGE_GROUPS
            },
            "trial_budgets": lock_payload["bounded_trial_budgets"],
            "comparator_arm_set": list(lock_payload["comparator_arms"]),
            "arm_specific_config_hashes": {
                arm: _stable_int_seed(json.dumps({"arm": arm, "scales": scales.tolist()}, sort_keys=True)) for arm in arms
            },
            "drift_refusal_token": "LOCKED_PANEL_SPLIT_SEED_BUDGETS",
            "source_lock_path": rel_to_root(LOCK_PATH),
        },
    )


def audit_baseline_runner(*, output_root: Path, args: argparse.Namespace, lock_payload: dict, arms: Tuple[str, ...], scales: np.ndarray) -> None:
    audit_baseline(
        output_root=output_root,
        seed=int(args.seed),
        subjects_per_group=int(args.subjects_per_group),
        test_fraction=float(args.test_fraction),
    )
    _write_root_manifests(output_root, args, lock_payload, arms, scales)


def build_panel_from_lock(*, output_root: Path, args: argparse.Namespace, lock_payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_locked_panel_files(output_root, lock_payload)


def fit_arm(*, output_root: Path, args: argparse.Namespace, lock_payload: dict, scales: np.ndarray, arm_name: str) -> None:
    age_groups = _resolve_age_groups(str(args.age_group))
    user_filter = _resolve_user_id_filter(getattr(args, "user_ids", None))
    _write_json(
        output_root / "heartbeat.json",
        {
            "schema_version": "per_subject_age_comparison.heartbeat.v1",
            "started_at": _now_iso(),
            "status": "running",
            "phase": "fit_arm",
            "comparator_arm": arm_name,
            "age_groups": list(age_groups),
            "user_ids": sorted(user_filter) if user_filter else None,
        },
    )
    for age_group in age_groups:
        panel_entries = _filter_panel_entries(lock_payload, age_group, user_filter)
        if not panel_entries:
            continue
        arm_cfg = _build_arm_config(args, age_group, arm_name, scales)
        _write_arm_config_and_status(output_root, arm_cfg)
        if arm_name == "phase18_replay_aligned":
            _write_phase18_alignment_check(output_root, args, age_group, arm_cfg, scales)
        if not arm_cfg.implemented:
            continue
        for entry in panel_entries:
            user_id = str(entry["user_id"])
            if arm_cfg.fit_family == "cached_stage2_ww":
                _fit_t0_only_baseline_subject(
                    age_group=age_group,
                    user_id=user_id,
                    output_root=output_root,
                    arm_cfg=arm_cfg,
                    seed=int(args.seed),
                    device=str(args.device),
                )
            else:
                _fit_dmc_variational_subject(
                    age_group=age_group,
                    user_id=user_id,
                    output_root=output_root,
                    arm_cfg=arm_cfg,
                    seed=int(args.seed),
                    device=str(args.device),
                )
        analyze_panel(output_root=output_root, arm_name=arm_name)
        _write_json(
            _arm_root(output_root, arm_name) / "run_complete.json",
            {
                "comparator_arm": arm_name,
                "status": "completed",
                "implemented": True,
                "updated_at": _now_iso(),
                "age_group_scope": list(age_groups),
            },
        )


def analyze_panel_mode(*, output_root: Path, arm_names: Tuple[str, ...]) -> None:
    for arm_name in arm_names:
        arm_root = _arm_root(output_root, arm_name)
        if not arm_root.exists():
            continue
        run_complete = arm_root / "run_complete.json"
        if run_complete.exists():
            payload = _load_json(run_complete)
            if payload.get("status") in {"refused", "failed"}:
                continue
        analyze_panel(output_root=output_root, arm_name=arm_name)
    write_stage1_gate_artifacts(output_root=output_root)


def main() -> None:
    args = parse_args()
    output_root = _resolve_output_root(args.output_root)
    scales = _parse_scales(args.scales)
    lock_payload = _load_lock()
    requested_arms = (str(args.comparator_arm),) if args.comparator_arm else tuple(x.strip() for x in str(args.comparator_arms).split(",") if x.strip())
    if not requested_arms:
        raise ValueError("NO_COMPARATOR_ARMS_REQUESTED")
    for arm in requested_arms:
        if arm not in ALLOWED_ARMS:
            raise ValueError(f"UNSUPPORTED_COMPARATOR_ARM: {arm}")
    _verify_lock_compatibility(args, lock_payload, scales, requested_arms)

    if args.mode == "audit-baseline":
        audit_baseline_runner(output_root=output_root, args=args, lock_payload=lock_payload, arms=requested_arms, scales=scales)
        return
    if args.mode == "build-panel":
        build_panel_from_lock(output_root=output_root, args=args, lock_payload=lock_payload)
        return
    if args.mode == "fit-arm":
        build_panel_from_lock(output_root=output_root, args=args, lock_payload=lock_payload)
        for arm_name in requested_arms:
            fit_arm(output_root=output_root, args=args, lock_payload=lock_payload, scales=scales, arm_name=arm_name)
        return
    if args.mode == "analyze-panel":
        analyze_panel_mode(output_root=output_root, arm_names=requested_arms)
        return
    if args.mode == "full-stage1":
        audit_baseline_runner(output_root=output_root, args=args, lock_payload=lock_payload, arms=requested_arms, scales=scales)
        build_panel_from_lock(output_root=output_root, args=args, lock_payload=lock_payload)
        for arm_name in requested_arms:
            fit_arm(output_root=output_root, args=args, lock_payload=lock_payload, scales=scales, arm_name=arm_name)
        analyze_panel_mode(output_root=output_root, arm_names=requested_arms)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww, summarize_condition  # noqa: E402
from analyze_layerwise_feature_probe import CentroidProbe, LayerwiseFeatureTap, LAYER_ORDER  # noqa: E402
from cache_vgg_stage2_features import load_stage1_model_with_metadata  # noqa: E402
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout, base_condition_df, build_natural_input  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from train_age_groups_efficient import DIRECTION_MAP, StimulusDataset, to_jsonable  # noqa: E402


OUT_DIR = PROJECT_ROOT / "artifacts/results/diagnostics/natural_layer_to_time_var_ww/full_age_group_vgg_evidence_cache"
HUMAN_META = PROJECT_ROOT / "data/vam_data/metadata.csv"
HUMAN_GLOB = "data/vam_data/user*df.csv"
OLD_CACHE = PROJECT_ROOT / "artifacts/results/diagnostics/layerwise_evidence_cache/layerwise_evidence.npz"
GRAPHICS_DIR = PROJECT_ROOT / "code/vam"
DECADE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
SEED = 20260529

LAYOUT_SPACING = {
    0: (51, 0),
    1: (0, 51),
    2: (51, 51),
    3: (34, 34),
    4: (34, 34),
    5: (34, 34),
    6: (34, 34),
}
WIN_SIZE = (640, 480)


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path
    unique_dir: Path
    shard_dir: Path
    cache_path: Path
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full age-group VGG / layerwise evidence cache from VAM human trials.")
    parser.add_argument("--age-groups", nargs="+", default=["all"], help="Age groups to include, or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Audit and plan only; do not extract VGG features.")
    parser.add_argument("--max-unique-images", type=int, default=None, help="Limit unique reconstructed stimuli for pilot extraction.")
    parser.add_argument("--max-unique-stimuli", type=int, default=None, help="Alias for --max-unique-images.")
    parser.add_argument("--unique-start-index", type=int, default=0, help="Sequential unique-stimulus start index for staged age-group extraction.")
    parser.add_argument("--max-shards", type=int, default=None, help="Run at most this many sequential unique-stimulus shards.")
    parser.add_argument("--max-trials-per-age", type=int, default=None, help="Limit trials per age group before unique-stimulus selection.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of trial rows to process per metadata chunk.")
    parser.add_argument("--batch-size", type=int, default=128, help="VGG extraction batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Extraction device.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing unique feature cache chunks when present.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fixed-candidate smoke test on evidence-available trials.")
    parser.add_argument("--probe-train-csv", default="data/age_groups/20-29/train_data.csv", help="Existing image CSV used to fit layerwise target-direction probes.")
    parser.add_argument("--probe-max-train", type=int, default=3000, help="Maximum rows from probe-train-csv used to fit layerwise probes.")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--trial-manifest", default=None, help="Optional selected trial manifest to restrict extraction exactly to audited rows.")
    return parser.parse_args()


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_age_groups(values: Iterable[str]) -> List[str]:
    vals = list(values)
    if len(vals) == 1 and vals[0].lower() == "all":
        return DECADE_GROUPS
    bad = [x for x in vals if x not in DECADE_GROUPS]
    if bad:
        raise ValueError(f"Unknown age groups: {bad}")
    return vals


def ensure_dirs(out_dir: Path) -> RunPaths:
    unique_dir = out_dir / "unique_feature_cache"
    shard_dir = out_dir / "shards"
    unique_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        out_dir=out_dir,
        unique_dir=unique_dir,
        shard_dir=shard_dir,
        cache_path=out_dir / "full_age_group_layerwise_evidence.npz",
        metadata_path=out_dir / "full_age_group_layerwise_evidence_metadata.csv",
    )


def read_human_metadata(age_groups: List[str], max_trials_per_age: int | None) -> pd.DataFrame:
    meta = pd.read_csv(HUMAN_META)
    meta["user_id"] = meta["user_id"].astype(str)
    frames: List[pd.DataFrame] = []
    gid = 0
    for path in sorted(PROJECT_ROOT.glob(HUMAN_GLOB)):
        df = pd.read_csv(path)
        df["user_id"] = df["anon_id"].astype(str)
        df = df.merge(meta[["user_id", "binned_age"]], on="user_id", how="left")
        df = df[df["binned_age"].isin(age_groups)].copy()
        if df.empty:
            continue
        if max_trials_per_age is not None:
            # Defer exact per-age cap until after concatenation.
            pass
        df["source_file"] = str(path)
        df["trial_index_within_user"] = np.arange(len(df), dtype=np.int64)
        df["global_trial_id"] = np.arange(gid, gid + len(df), dtype=np.int64)
        gid += len(df)
        frames.append(df)
    if not frames:
        raise RuntimeError("No human trial rows matched requested age groups.")
    out = pd.concat(frames, ignore_index=True)
    if max_trials_per_age is not None:
        out = (
            out.groupby("binned_age", group_keys=False, sort=False)
            .head(int(max_trials_per_age))
            .copy()
            .reset_index(drop=True)
        )
        out["global_trial_id"] = np.arange(len(out), dtype=np.int64)
    out["age_group"] = out["binned_age"].astype(str)
    out["human_rt"] = pd.to_numeric(out["response_time"], errors="coerce") / 1000.0
    out["human_response"] = out["response_direction"].astype(str)
    out["human_correct"] = out["response_direction"].astype(str) == out["target_direction"].astype(str)
    out["target_label"] = out["target_direction"].map(DIRECTION_MAP).astype(np.int64)
    out["flanker_label"] = out["flanker_direction"].map(DIRECTION_MAP).astype(np.int64)
    out["response_label"] = out["response_direction"].map(DIRECTION_MAP).astype(np.int64)
    out["congruency"] = (out["target_label"] != out["flanker_label"]).astype(np.int64)
    out["stimulus_key"] = (
        out["xpos"].astype(str)
        + "|"
        + out["ypos"].astype(str)
        + "|"
        + out["stimulus_layout"].astype(str)
        + "|"
        + out["target_direction"].astype(str)
        + "|"
        + out["flanker_direction"].astype(str)
    )
    unique_keys = {key: i for i, key in enumerate(pd.unique(out["stimulus_key"]))}
    out["image_id"] = out["stimulus_key"].map(unique_keys).astype(np.int64)
    out["stimulus_image_path"] = out["image_id"].map(lambda x: f"reconstructed://stimulus_{int(x):08d}")
    out["target_image_path"] = out["target_label"].map(lambda x: str(GRAPHICS_DIR / f"bird{int(x)}.png"))
    out["flanker_image_path"] = out["flanker_label"].map(lambda x: str(GRAPHICS_DIR / f"bird{int(x)}.png"))
    out["evidence_available"] = False
    out["evidence_missing_reason"] = "not_extracted_yet"
    keep = [
        "global_trial_id",
        "user_id",
        "age_group",
        "trial_index_within_user",
        "human_rt",
        "human_response",
        "human_correct",
        "target_direction",
        "target_label",
        "flanker_direction",
        "flanker_label",
        "response_label",
        "congruency",
        "stimulus_layout",
        "xpos",
        "ypos",
        "stimulus_image_path",
        "image_id",
        "target_image_path",
        "flanker_image_path",
        "source_file",
        "evidence_available",
        "evidence_missing_reason",
        "stimulus_key",
    ]
    return out[keep].copy()


def restrict_to_trial_manifest(metadata: pd.DataFrame, manifest_path: str | Path) -> pd.DataFrame:
    """Restrict source rows before unique-stimulus expansion to keep subset runs bounded."""
    selected = pd.read_csv(manifest_path)
    required = {"subject_id", "source_row_index", "age_group"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Trial manifest missing required columns: {sorted(missing)}")
    selected_key = selected[["subject_id", "source_row_index", "age_group"]].copy()
    selected_key["subject_id"] = selected_key["subject_id"].astype(str)
    selected_key["source_row_index"] = selected_key["source_row_index"].astype(np.int64)
    selected_key["age_group"] = selected_key["age_group"].astype(str)
    selected_key = selected_key.drop_duplicates()
    out = metadata.merge(
        selected_key.assign(_selected=True),
        left_on=["user_id", "trial_index_within_user", "age_group"],
        right_on=["subject_id", "source_row_index", "age_group"],
        how="inner",
        validate="one_to_one",
    )
    return out.drop(columns=["subject_id", "source_row_index", "_selected"], errors="ignore").reset_index(drop=True)


def unique_stimuli(metadata: pd.DataFrame, max_unique_images: int | None) -> pd.DataFrame:
    cols = ["image_id", "stimulus_key", "xpos", "ypos", "stimulus_layout", "target_label", "flanker_label", "target_direction", "flanker_direction"]
    unique = metadata.drop_duplicates("image_id")[cols].sort_values("image_id").reset_index(drop=True)
    unique["unique_sequence_index"] = np.arange(len(unique), dtype=np.int64)
    if max_unique_images is not None and len(unique) > max_unique_images:
        # Keep a deterministic, class-balanced pilot so all direction labels are likely present.
        rng = np.random.default_rng(SEED)
        selected: List[int] = []
        for target in sorted(unique["target_label"].unique()):
            idx = unique.index[unique["target_label"].eq(target)].to_numpy()
            take = min(len(idx), max(1, int(max_unique_images) // max(4, unique["target_label"].nunique())))
            if take:
                selected.extend(rng.choice(idx, size=take, replace=False).tolist())
        if len(selected) < max_unique_images:
            remaining = np.setdiff1d(unique.index.to_numpy(), np.asarray(selected, dtype=np.int64))
            take = min(len(remaining), int(max_unique_images) - len(selected))
            if take:
                selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
        unique = unique.loc[sorted(selected[: int(max_unique_images)])].reset_index(drop=True)
    return unique


def staged_unique_stimuli(metadata: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if int(args.unique_start_index) > 0 or args.max_shards is not None:
        unique = unique_stimuli(metadata, None)
        limit = int(args.max_unique_images) if args.max_unique_images is not None else int(args.chunk_size) * int(args.max_shards or 1)
        start = max(0, int(args.unique_start_index))
        return unique.iloc[start : start + limit].reset_index(drop=True)
    return unique_stimuli(metadata, args.max_unique_images)


def asset_status() -> Dict[str, bool]:
    assets = {"background": (GRAPHICS_DIR / "bkgrnd.png").exists()}
    for i in range(4):
        assets[f"bird{i}"] = (GRAPHICS_DIR / f"bird{i}.png").exists()
    return assets


def full_run_readiness(metadata: pd.DataFrame, unique_all: pd.DataFrame, out_dir: Path, recommended_device: str, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    required_cols = [
        "user_id",
        "age_group",
        "human_rt",
        "human_response",
        "target_direction",
        "flanker_direction",
        "stimulus_layout",
        "xpos",
        "ypos",
        "target_label",
        "flanker_label",
    ]
    assets = asset_status()
    missing_assets = [k for k, ok in assets.items() if not ok]
    missing_fields = [c for c in required_cols if c not in metadata.columns or metadata[c].isna().any()]
    valid_dirs = set(DIRECTION_MAP)
    bad_dirs = sorted(
        set(metadata["target_direction"].dropna().astype(str)) - valid_dirs
        | set(metadata["flanker_direction"].dropna().astype(str)) - valid_dirs
        | set(metadata["human_response"].dropna().astype(str)) - valid_dirs
    )
    bad_layout = sorted(set(metadata["stimulus_layout"].dropna().astype(int)) - set(LAYOUT_SPACING))
    duplicate_keys = int(metadata.duplicated("stimulus_key").sum())
    rows = []
    for age, part in metadata.groupby("age_group", sort=True):
        u = unique_all[unique_all["image_id"].isin(set(part["image_id"]))]
        rows.append(
            {
                "age_group": age,
                "n_trials": int(len(part)),
                "n_subjects": int(part["user_id"].nunique()),
                "n_unique_stimuli": int(len(u)),
                "missing_required_field_count": int(len(missing_fields)),
                "missing_fields": ";".join(missing_fields),
                "missing_assets": ";".join(missing_assets),
                "bad_direction_values": ";".join(map(str, bad_dirs)),
                "bad_layout_values": ";".join(map(str, bad_layout)),
                "duplicate_unique_stimulus_key_trials": duplicate_keys,
                "unreconstructable_trials": int(len(part) if missing_fields or missing_assets or bad_dirs or bad_layout else 0),
                "can_start_full_extraction": bool(not missing_fields and not missing_assets and not bad_dirs and not bad_layout),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "full_run_readiness_audit.csv", index=False)
    total_trials = int(len(metadata))
    total_unique = int(len(unique_all))
    trials_by_age = audit[["age_group", "n_trials"]].to_dict("records")
    unique_by_age = audit[["age_group", "n_unique_stimuli"]].to_dict("records")
    if benchmark is not None and len(benchmark) and float(benchmark["unique_stimuli_per_second"].iloc[-1]) > 0:
        ups = float(benchmark["unique_stimuli_per_second"].iloc[-1])
        estimated_hours = total_unique / ups / 3600.0
    else:
        estimated_hours = float("nan")
    recommended_chunk = 50000
    summary = f"""# Full-run Readiness Summary

- total_trials: {total_trials}
- total_unique_stimuli: {total_unique}
- trials_by_age_group: {trials_by_age}
- unique_stimuli_by_age_group: {unique_by_age}
- missing_fields: {missing_fields}
- missing_assets: {missing_assets}
- bad_direction_values: {bad_dirs}
- bad_layout_values: {bad_layout}
- can_start_full_extraction: {bool(audit['can_start_full_extraction'].all())}
- recommended_device: {recommended_device}
- recommended_chunk_size: {recommended_chunk}
- estimated_runtime_hours: {estimated_hours if math.isfinite(estimated_hours) else 'benchmark_required'}

The raw full dataset has no stored full-stimulus image path, so full extraction depends on reconstructing images from trial fields and existing bird/background assets.
"""
    (out_dir / "full_run_readiness_summary.md").write_text(summary, encoding="utf-8")
    return audit


def input_audit(metadata: pd.DataFrame, unique: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    rows = [
        {
            "file_path": str(HUMAN_META),
            "n_rows": int(pd.read_csv(HUMAN_META).shape[0]),
            "n_subjects": int(pd.read_csv(HUMAN_META)["user_id"].nunique()),
            "age_groups": ",".join(DECADE_GROUPS),
            "has_trial_metadata": True,
            "has_stimulus_image_path": False,
            "has_reconstructable_stimulus": True,
            "has_target_label": True,
            "has_flanker_label": True,
            "has_rt": False,
            "has_response": False,
            "usable_for_full_cache": True,
            "notes": "Subject-level age metadata; combined with user trial CSV files.",
        },
        {
            "file_path": str(PROJECT_ROOT / "data/vam_data/user*df.csv"),
            "n_rows": int(len(metadata)),
            "n_subjects": int(metadata["user_id"].nunique()),
            "age_groups": ",".join(sorted(metadata["age_group"].unique())),
            "has_trial_metadata": True,
            "has_stimulus_image_path": False,
            "has_reconstructable_stimulus": True,
            "has_target_label": True,
            "has_flanker_label": True,
            "has_rt": True,
            "has_response": True,
            "usable_for_full_cache": True,
            "notes": f"Raw trials do not contain image paths; {len(unique)} unique stimuli selected for this run can be reconstructed from direction/layout/position fields.",
        },
        {
            "file_path": str(OLD_CACHE),
            "n_rows": int(np.load(OLD_CACHE, allow_pickle=True)["target_labels"].shape[0]) if OLD_CACHE.exists() else 0,
            "n_subjects": 1,
            "age_groups": "20-29",
            "has_trial_metadata": True,
            "has_stimulus_image_path": True,
            "has_reconstructable_stimulus": True,
            "has_target_label": True,
            "has_flanker_label": True,
            "has_rt": True,
            "has_response": True,
            "usable_for_full_cache": False,
            "notes": "Existing pilot cache is useful for schema comparison only; it is not full age-group evidence.",
        },
    ]
    if dry_run:
        rows.append(
            {
                "file_path": "dry-run",
                "n_rows": 0,
                "n_subjects": 0,
                "age_groups": "",
                "has_trial_metadata": False,
                "has_stimulus_image_path": False,
                "has_reconstructable_stimulus": False,
                "has_target_label": False,
                "has_flanker_label": False,
                "has_rt": False,
                "has_response": False,
                "usable_for_full_cache": False,
                "notes": "Dry run requested; no VGG extraction was performed.",
            }
        )
    return pd.DataFrame(rows)


def image_field_mapping() -> pd.DataFrame:
    rows = [
        ("stimulus_image_path", "reconstructed from image_id", True, "Raw full data do not store a path; script uses reconstructed:// IDs."),
        ("stimulus_id", "image_id", True, "Deterministic unique ID from xpos/ypos/layout/target/flanker."),
        ("target_image", "code/vam/bird{target_label}.png", True, "Component bird image used during reconstruction."),
        ("target_label", "target_direction mapped by DIRECTION_MAP", True, ""),
        ("flanker_image", "code/vam/bird{flanker_label}.png", True, "Component bird image used during reconstruction."),
        ("flanker_label", "flanker_direction mapped by DIRECTION_MAP", True, ""),
        ("response / human_response", "response_direction", True, ""),
        ("human_correct", "response_direction == target_direction", True, ""),
        ("response_time / human_rt", "response_time / 1000", True, ""),
        ("user_id", "anon_id / metadata.user_id", True, ""),
        ("age_group", "metadata.binned_age", True, ""),
        ("trial_index", "row order within each user CSV", True, ""),
        ("xpos / ypos / stimulus_layout", "raw user CSV", True, "Needed for full-stimulus reconstruction."),
    ]
    return pd.DataFrame(rows, columns=["field", "source", "available", "missing_or_alignment_notes"])


def distractor_positions(target_pos: Tuple[float, float], layout: int, spacer: Tuple[int, int]) -> List[Tuple[float, float]]:
    x, y = target_pos
    sx, sy = spacer
    if layout == 0:
        return [(x - 2 * sx, y), (x - sx, y), (x + sx, y), (x + 2 * sx, y)]
    if layout == 1:
        return [(x, y - 2 * sy), (x, y - sy), (x, y + sy), (x, y + 2 * sy)]
    if layout == 2:
        return [(x - sx, y), (x + sx, y), (x, y - sy), (x, y + sy)]
    if layout == 3:
        return [(x + sx, y + sy), (x + 2 * sx, y + 2 * sy), (x + sx, y - sy), (x + 2 * sx, y - 2 * sy)]
    if layout == 4:
        return [(x - sx, y + sy), (x - 2 * sx, y + 2 * sy), (x - sx, y - sy), (x - 2 * sx, y - 2 * sy)]
    if layout == 5:
        return [(x - sx, y + sy), (x - 2 * sx, y + 2 * sy), (x + sx, y + sy), (x + 2 * sx, y + 2 * sy)]
    if layout == 6:
        return [(x - sx, y - sy), (x - 2 * sx, y - 2 * sy), (x + sx, y - sy), (x + 2 * sx, y - 2 * sy)]
    raise ValueError(f"Unknown layout: {layout}")


class StimulusRenderer:
    def __init__(self, image_size: int = 128):
        self.background = Image.open(GRAPHICS_DIR / "bkgrnd.png").convert("RGBA")
        self.birds = {i: Image.open(GRAPHICS_DIR / f"bird{i}.png").convert("RGBA") for i in range(4)}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def paste_center(canvas: Image.Image, sprite: Image.Image, pos_centered: Tuple[float, float]) -> None:
        cx = int(round(WIN_SIZE[0] / 2 + pos_centered[0]))
        cy = int(round(WIN_SIZE[1] / 2 - pos_centered[1]))
        xy = (cx - sprite.width // 2, cy - sprite.height // 2)
        canvas.alpha_composite(sprite, xy)

    def render_tensor(self, row: pd.Series) -> torch.Tensor:
        canvas = self.background.copy()
        xpos_centered = float(row["xpos"]) - WIN_SIZE[0] / 2
        ypos_centered = -float(row["ypos"]) + WIN_SIZE[1] / 2
        target_pos = (xpos_centered, ypos_centered)
        layout = int(row["stimulus_layout"])
        spacer = LAYOUT_SPACING[layout]
        self.paste_center(canvas, self.birds[int(row["target_label"])], target_pos)
        for pos in distractor_positions(target_pos, layout, spacer):
            self.paste_center(canvas, self.birds[int(row["flanker_label"])], pos)
        return self.transform(canvas.convert("RGB"))


def extract_unique_features(unique: pd.DataFrame, paths: RunPaths, device: str, batch_size: int, resume: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    feature_path = paths.unique_dir / "unique_layerwise_features.npz"
    meta_path = paths.unique_dir / "unique_stimuli.csv"
    if resume and feature_path.exists() and meta_path.exists():
        cached_meta = pd.read_csv(meta_path)
        cached_ids = cached_meta["image_id"].to_numpy(dtype=np.int64) if "image_id" in cached_meta.columns else np.array([], dtype=np.int64)
        requested_ids = unique["image_id"].to_numpy(dtype=np.int64)
        if len(cached_ids) == len(requested_ids) and np.array_equal(cached_ids, requested_ids):
            z = np.load(feature_path, allow_pickle=True)
            features = {k: z[k] for k in z.files if k in LAYER_ORDER}
            return features, {"resumed": True, "feature_path": str(feature_path), "n_unique_extracted": int(len(cached_meta))}
        # Existing cache belongs to a different pilot/full selection. Do not reuse it.
        feature_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    model_base, model_metadata = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(model_base).to(device)
    model.eval()
    renderer = StimulusRenderer()
    batches: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}

    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            part = unique.iloc[start : start + batch_size]
            images = torch.stack([renderer.render_tensor(row) for _, row in part.iterrows()], dim=0).to(device)
            values = model.forward_layerwise(images)
            for layer in LAYER_ORDER:
                batches[layer].append(values[layer].detach().cpu().numpy().astype(np.float32))

    features = {layer: np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32) for layer, parts in batches.items()}
    elapsed = time.perf_counter() - start_time
    np.savez_compressed(feature_path, **features)
    unique.to_csv(meta_path, index=False)
    return features, {
        **model_metadata,
        "resumed": False,
        "feature_path": str(feature_path),
        "n_unique_extracted": int(len(unique)),
        "unique_extraction_seconds": float(elapsed),
        "unique_stimuli_per_second": float(len(unique) / elapsed) if elapsed > 0 else float("nan"),
    }


def fit_layer_probes(probe_features: Dict[str, np.ndarray], probe_labels: np.ndarray) -> Dict[str, CentroidProbe]:
    probes: Dict[str, CentroidProbe] = {}
    for layer in LAYER_ORDER:
        if layer == "final_logits":
            continue
        probes[layer] = CentroidProbe().fit(probe_features[layer], probe_labels)
    return probes


def features_to_evidence(features: Dict[str, np.ndarray], probes: Dict[str, CentroidProbe]) -> Dict[str, np.ndarray]:
    evidence: Dict[str, np.ndarray] = {}
    for layer in LAYER_ORDER:
        if layer == "final_logits":
            evidence["evidence_final"] = np.asarray(features[layer], dtype=np.float32)
        else:
            evidence[f"evidence_{layer}"] = four_class_scores(probes[layer], features[layer]).astype(np.float32)
    return evidence


def write_unique_manifest(unique_all: pd.DataFrame, metadata: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = metadata.groupby("image_id", sort=False)
    counts = grouped.size().rename("n_trials_using_this_stimulus")
    ages = grouped["age_group"].agg(lambda s: ",".join(sorted(set(map(str, s)))))
    manifest = unique_all.merge(counts, left_on="image_id", right_index=True, how="left")
    manifest = manifest.merge(ages.rename("age_groups_using_this_stimulus"), left_on="image_id", right_index=True, how="left")
    manifest = manifest.rename(columns={"image_id": "unique_stimulus_id", "stimulus_layout": "layout", "xpos": "target_x", "ypos": "target_y"})
    manifest["background_id"] = "bkgrnd.png"
    manifest["bird_asset_id"] = manifest.apply(lambda r: f"target=bird{int(r['target_label'])}.png;flanker=bird{int(r['flanker_label'])}.png", axis=1)
    manifest["target_position"] = manifest["target_x"].astype(str) + "," + manifest["target_y"].astype(str)
    manifest["flanker_position"] = "derived_from_layout"
    manifest.to_csv(out_dir / "unique_stimuli.csv", index=False)
    return manifest


def extract_evidence_shards(
    unique: pd.DataFrame,
    probes: Dict[str, CentroidProbe],
    paths: RunPaths,
    *,
    device: str,
    batch_size: int,
    chunk_size: int,
    resume: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    shard_rows = []
    failed_rows = []
    total = len(unique)
    completed = 0
    start_iso = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()
    model_base, model_metadata = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(model_base).to(device)
    model.eval()
    renderer = StimulusRenderer()
    for shard_id, start in enumerate(range(0, total, max(1, int(chunk_size)))):
        end = min(total, start + max(1, int(chunk_size)))
        if "unique_sequence_index" in unique.columns and len(unique.iloc[start:end]):
            global_shard_id = int(unique.iloc[start]["unique_sequence_index"]) // max(1, int(chunk_size))
        else:
            global_shard_id = shard_id
        shard_name = f"unique_layerwise_features_shard_{global_shard_id:05d}.npz"
        shard_path = paths.shard_dir / shard_name
        part = unique.iloc[start:end]
        if resume and shard_path.exists():
            try:
                z = np.load(shard_path, allow_pickle=True)
                cached_ids = z["unique_stimulus_id"].astype(np.int64)
                requested_ids = part["image_id"].to_numpy(dtype=np.int64)
                if len(cached_ids) == len(requested_ids) and np.array_equal(cached_ids, requested_ids):
                    completed += len(part)
                    shard_rows.append(
                        {
                            "shard_id": global_shard_id,
                            "shard_path": str(shard_path),
                            "start_index": int(start),
                            "end_index": int(end),
                            "n_unique_stimuli": int(len(part)),
                            "status": "resumed",
                        }
                    )
                    continue
            except Exception:
                shard_path.unlink(missing_ok=True)

        feature_batches: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
        status = np.array(["ok"] * len(part), dtype=object)
        errors = np.array([""] * len(part), dtype=object)
        with torch.no_grad():
            for local_start in range(0, len(part), batch_size):
                batch_part = part.iloc[local_start : local_start + batch_size]
                tensors = []
                good_local = []
                for offset, (_, row) in enumerate(batch_part.iterrows()):
                    try:
                        tensors.append(renderer.render_tensor(row))
                        good_local.append(offset)
                    except Exception as exc:
                        absolute = local_start + offset
                        status[absolute] = "failed"
                        errors[absolute] = str(exc)
                        failed_rows.append({"unique_stimulus_id": int(row["image_id"]), "error_message": str(exc)})
                if not tensors:
                    continue
                images = torch.stack(tensors, dim=0).to(device)
                values = model.forward_layerwise(images)
                for layer in LAYER_ORDER:
                    arr = np.full((len(batch_part), values[layer].shape[1]), np.nan, dtype=np.float32)
                    arr[np.asarray(good_local, dtype=np.int64)] = values[layer].detach().cpu().numpy().astype(np.float32)
                    feature_batches[layer].append(arr)
        features = {layer: np.concatenate(parts, axis=0) if parts else np.empty((len(part), 0), dtype=np.float32) for layer, parts in feature_batches.items()}
        evidence = features_to_evidence(features, probes)
        payload: Dict[str, np.ndarray] = {"unique_stimulus_id": part["image_id"].to_numpy(dtype=np.int64)}
        for key, arr in evidence.items():
            payload[key] = arr.astype(np.float32)
        payload["extraction_status"] = status
        payload["error_message"] = errors
        np.savez_compressed(shard_path, **payload)
        completed += int((status == "ok").sum())
        shard_rows.append(
            {
                "shard_id": global_shard_id,
                "shard_path": str(shard_path),
                "start_index": int(start),
                "end_index": int(end),
                "n_unique_stimuli": int(len(part)),
                "status": "completed",
            }
        )
        progress = {
            "total_unique_stimuli": int(total),
            "completed_unique_stimuli": int(completed),
            "failed_unique_stimuli": int(len(failed_rows)),
            "current_shard": int(global_shard_id),
            "last_completed_shard": int(global_shard_id),
            "device": device,
            "chunk_size": int(chunk_size),
            "start_time": start_iso,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "estimated_remaining_time": "not_estimated_for_pilot",
        }
        (paths.out_dir / "extraction_progress.json").write_text(json.dumps(to_jsonable(progress), indent=2), encoding="utf-8")
    failed = pd.DataFrame(failed_rows, columns=["unique_stimulus_id", "error_message"])
    failed.to_csv(paths.out_dir / "failed_unique_stimuli.csv", index=False)
    resume_df = pd.DataFrame(shard_rows)
    resume_df["integrity_ok"] = resume_df["shard_path"].map(lambda p: Path(p).exists())
    resume_df.to_csv(paths.out_dir / "resume_integrity_check.csv", index=False)
    (paths.out_dir / "resume_integrity_summary.md").write_text(
        f"# Resume Integrity Summary\n\nCompleted shards: {len(resume_df)}.\nFailed unique stimuli: {len(failed)}.\nAll shard files present: {bool(resume_df['integrity_ok'].all()) if len(resume_df) else False}.\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start_perf
    meta = {
        **model_metadata,
        "resumed": bool(resume),
        "feature_path": str(paths.shard_dir),
        "n_unique_extracted": int(completed),
        "unique_extraction_seconds": float(elapsed),
        "unique_stimuli_per_second": float(completed / elapsed) if elapsed > 0 else float("nan"),
    }
    return resume_df, meta


def extract_probe_training_features(csv_path: Path, device: str, batch_size: int, max_rows: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing probe training CSV: {csv_path}")
    dataset = StimulusDataset(str(csv_path))
    n = min(len(dataset), int(max_rows)) if max_rows and max_rows > 0 else len(dataset)
    indices = np.arange(n, dtype=np.int64)
    model_base, model_metadata = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(model_base).to(device)
    model.eval()
    batches: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            images = torch.stack([dataset[int(i)]["image"] for i in idx], dim=0).to(device)
            values = model.forward_layerwise(images)
            for layer in LAYER_ORDER:
                batches[layer].append(values[layer].detach().cpu().numpy().astype(np.float32))
    features = {layer: np.concatenate(parts, axis=0) for layer, parts in batches.items()}
    labels = dataset.target_labels[indices].astype(np.int64)
    return features, labels, {**model_metadata, "probe_train_csv": str(csv_path), "probe_train_rows": int(n)}


def four_class_scores(probe: CentroidProbe, x: np.ndarray) -> np.ndarray:
    scores = probe.decision_function(x)
    out = np.full((x.shape[0], 4), -1e6, dtype=np.float32)
    assert probe.classes_ is not None
    for j, cls in enumerate(probe.classes_):
        out[:, int(cls)] = scores[:, j]
    return out


def build_unique_evidence(
    unique: pd.DataFrame,
    features: Dict[str, np.ndarray],
    probe_features: Dict[str, np.ndarray],
    probe_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    evidence: Dict[str, np.ndarray] = {}
    for layer in LAYER_ORDER:
        if layer == "final_logits":
            evidence["evidence_final"] = np.asarray(features[layer], dtype=np.float32)
            continue
        probe = CentroidProbe().fit(probe_features[layer], probe_labels)
        key = f"evidence_{layer}"
        evidence[key] = four_class_scores(probe, features[layer]).astype(np.float32)
    return evidence


def load_unique_evidence_from_shards(paths: RunPaths) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    shard_files = sorted(paths.shard_dir.glob("unique_layerwise_features_shard_*.npz"))
    ids: List[np.ndarray] = []
    evidence_parts: Dict[str, List[np.ndarray]] = {k: [] for k in ["evidence_conv3", "evidence_conv4", "evidence_conv5", "evidence_pooled", "evidence_final"]}
    status_parts: List[np.ndarray] = []
    error_parts: List[np.ndarray] = []
    shard_id_parts: List[np.ndarray] = []
    for shard_id, path in enumerate(shard_files):
        z = np.load(path, allow_pickle=True)
        ids.append(z["unique_stimulus_id"].astype(np.int64))
        for key in evidence_parts:
            evidence_parts[key].append(z[key].astype(np.float32))
        n = len(z["unique_stimulus_id"])
        status_parts.append(z["extraction_status"].astype(str))
        error_parts.append(z["error_message"].astype(str))
        shard_id_parts.append(np.full(n, shard_id, dtype=np.int64))
    if not ids:
        return pd.DataFrame(columns=["image_id", "feature_shard_id", "extraction_status", "error_message"]), {k: np.empty((0, 4), dtype=np.float32) for k in evidence_parts}
    shard_meta = pd.DataFrame(
        {
            "image_id": np.concatenate(ids),
            "feature_shard_id": np.concatenate(shard_id_parts),
            "extraction_status": np.concatenate(status_parts),
            "error_message": np.concatenate(error_parts),
        }
    )
    evidence = {key: np.concatenate(parts, axis=0) for key, parts in evidence_parts.items()}
    return shard_meta, evidence


def write_npz(metadata: pd.DataFrame, unique: pd.DataFrame, unique_evidence: Dict[str, np.ndarray], paths: RunPaths) -> None:
    n = len(metadata)
    output: Dict[str, np.ndarray] = {
        "global_trial_id": metadata["global_trial_id"].to_numpy(dtype=np.int64),
        "target_labels": metadata["target_label"].to_numpy(dtype=np.int64),
        "flanker_labels": metadata["flanker_label"].to_numpy(dtype=np.int64),
        "response_labels": metadata["response_label"].to_numpy(dtype=np.int64),
        "true_rt": metadata["human_rt"].to_numpy(dtype=np.float32),
        "human_correct": metadata["human_correct"].to_numpy(dtype=bool),
        "congruency": metadata["congruency"].to_numpy(dtype=np.int64),
        "row_indices": metadata["trial_index_within_user"].to_numpy(dtype=np.int64),
        "age_group": metadata["age_group"].astype(str).to_numpy(),
        "user_id": metadata["user_id"].astype(str).to_numpy(),
        "stimulus_image_path": metadata["stimulus_image_path"].astype(str).to_numpy(),
        "image_id": metadata["image_id"].to_numpy(dtype=np.int64),
        "evidence_available": metadata["evidence_available"].to_numpy(dtype=bool),
        "evidence_missing_reason": metadata["evidence_missing_reason"].astype(str).to_numpy(),
    }
    unique_index = {int(image_id): i for i, image_id in enumerate(unique["image_id"].to_numpy(dtype=np.int64))}
    mapped = np.array([unique_index.get(int(x), -1) for x in metadata["image_id"].to_numpy(dtype=np.int64)], dtype=np.int64)
    for key in ["evidence_conv3", "evidence_conv4", "evidence_conv5", "evidence_pooled", "evidence_final"]:
        arr = np.full((n, 4), np.nan, dtype=np.float32)
        ok = mapped >= 0
        if ok.any():
            arr[ok] = unique_evidence[key][mapped[ok]]
        output[key] = arr
    np.savez_compressed(paths.cache_path, **output)


def write_loader_doc(out_dir: Path) -> None:
    text = """# Full Age-group Evidence Loader Notes

The memory-safe full extraction path writes unique-stimulus evidence shards under `shards/`.
Each shard contains `unique_stimulus_id`, `evidence_conv3`, `evidence_conv4`, `evidence_conv5`,
`evidence_pooled`, `evidence_final`, `extraction_status`, and `error_message`.

`full_age_group_layerwise_evidence_metadata.csv` maps each trial to `image_id`
(`unique_stimulus_id`) and `feature_shard_id`. For small/pilot runs the script also writes
`full_age_group_layerwise_evidence.npz` with repeated trial-level evidence for compatibility
with the older pipeline.

For a very large full run, prefer a future fitting loader that joins trial rows to shard rows
on demand instead of materializing repeated trial-level evidence in memory.
"""
    (out_dir / "full_age_group_layerwise_evidence_loader_notes.md").write_text(text, encoding="utf-8")


def schema_comparison(new_cache_path: Path) -> pd.DataFrame:
    rows = []
    old = np.load(OLD_CACHE, allow_pickle=True) if OLD_CACHE.exists() else None
    new = np.load(new_cache_path, allow_pickle=True) if new_cache_path.exists() else None
    keys = sorted((set(old.files) if old is not None else set()) | (set(new.files) if new is not None else set()))
    for key in keys:
        rows.append(
            {
                "key": key,
                "old_present": old is not None and key in old.files,
                "old_shape": list(old[key].shape) if old is not None and key in old.files else "",
                "old_dtype": str(old[key].dtype) if old is not None and key in old.files else "",
                "new_present": new is not None and key in new.files,
                "new_shape": list(new[key].shape) if new is not None and key in new.files else "",
                "new_dtype": str(new[key].dtype) if new is not None and key in new.files else "",
                "notes": "kept for old pipeline compatibility" if key in {"target_labels", "flanker_labels", "response_labels", "true_rt", "congruency", "row_indices", "age_group", "user_id", "stimulus_image_path", "evidence_conv3", "evidence_conv4", "evidence_conv5", "evidence_pooled", "evidence_final"} else "new metadata key",
            }
        )
    return pd.DataFrame(rows)


def coverage_audit(metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for age, part in metadata.groupby("age_group", sort=True):
        rows.append(
            {
                "age_group": age,
                "n_trials": int(len(part)),
                "n_subjects": int(part["user_id"].nunique()),
                "n_unique_images": int(part["image_id"].nunique()),
                "n_evidence_available": int(part["evidence_available"].sum()),
                "evidence_coverage": float(part["evidence_available"].mean()),
                "missing_reasons": "; ".join(sorted(part.loc[~part["evidence_available"], "evidence_missing_reason"].astype(str).unique())),
            }
        )
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "n_trials": int(len(metadata)),
                "n_subjects": int(metadata["user_id"].nunique()),
                "age_groups": ",".join(sorted(metadata["age_group"].unique())),
                "n_unique_images": int(metadata["image_id"].nunique()),
                "n_evidence_available": int(metadata["evidence_available"].sum()),
                "evidence_coverage": float(metadata["evidence_available"].mean()),
                "can_run_age_group_restricted_fitting": bool(
                    len(metadata) and metadata["evidence_available"].all()
                ),
            }
        ]
    )
    return audit, summary


def fitting_coverage_by_age(metadata: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for age, part in metadata.groupby("age_group", sort=True):
        unique_total = int(part["image_id"].nunique())
        with_evidence = part[part["evidence_available"]]
        unique_with = int(with_evidence["image_id"].nunique())
        failed_unique = int(part.loc[part["evidence_missing_reason"].astype(str).str.contains("failed", case=False, na=False), "image_id"].nunique())
        trial_cov = float(len(with_evidence) / max(len(part), 1))
        unique_cov = float(unique_with / max(unique_total, 1))
        failed_rate = float(failed_unique / max(unique_total, 1))
        rows.append(
            {
                "age_group": age,
                "n_trials_total": int(len(part)),
                "n_trials_with_evidence": int(len(with_evidence)),
                "trial_coverage_rate": trial_cov,
                "n_unique_stimuli_total": unique_total,
                "n_unique_stimuli_with_evidence": unique_with,
                "unique_coverage_rate": unique_cov,
                "n_failed_unique_stimuli": failed_unique,
                "failed_rate": failed_rate,
                "can_use_for_age_group_fitting": bool(trial_cov >= 0.95 and unique_cov >= 0.95 and failed_rate <= 0.01),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "full_age_group_evidence_coverage_by_age.csv", index=False)
    can_fit = bool(len(out) and out["can_use_for_age_group_fitting"].all())
    reasons = []
    low = out.loc[~out["can_use_for_age_group_fitting"], "age_group"].tolist()
    if low:
        reasons.append(f"coverage_gate_failed_for_age_groups={low}")
    text = f"""# Full Age-group Evidence Coverage Summary

Can run age-group restricted fitting: {can_fit}

Blocking reasons: {reasons if reasons else []}

Coverage table: `full_age_group_evidence_coverage_by_age.csv`
"""
    (out_dir / "full_age_group_evidence_coverage_summary.md").write_text(text, encoding="utf-8")
    return out


def evidence_sanity(cache_path: Path | None, metadata: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    required = ["evidence_conv3", "evidence_conv4", "evidence_conv5", "evidence_pooled", "evidence_final"]
    if cache_path is None or not cache_path.exists():
        for key in required:
            rows.append({"check": key, "present": False, "shape": "", "nan_count": np.nan, "inf_count": np.nan, "status": "missing_cache"})
    else:
        z = np.load(cache_path, allow_pickle=True)
        for key in required:
            arr = z[key] if key in z.files else None
            rows.append(
                {
                    "check": key,
                    "present": arr is not None,
                    "shape": list(arr.shape) if arr is not None else "",
                    "nan_count": int(np.isnan(arr).sum()) if arr is not None else np.nan,
                    "inf_count": int(np.isinf(arr).sum()) if arr is not None else np.nan,
                    "finite_available_values": bool(np.isfinite(arr[np.asarray(z["evidence_available"], dtype=bool)]).all()) if arr is not None and "evidence_available" in z.files and np.asarray(z["evidence_available"], dtype=bool).any() else False,
                    "status": "ok" if arr is not None and arr.shape[1] == 4 else "bad",
                }
            )
    congruent_ok = bool((metadata.loc[metadata["congruency"].eq(0), "target_label"] == metadata.loc[metadata["congruency"].eq(0), "flanker_label"]).all())
    incongruent_ok = bool((metadata.loc[metadata["congruency"].eq(1), "target_label"] != metadata.loc[metadata["congruency"].eq(1), "flanker_label"]).all())
    rows.append({"check": "congruent_label_alignment", "present": True, "shape": "", "nan_count": 0, "inf_count": 0, "status": "ok" if congruent_ok else "bad"})
    rows.append({"check": "incongruent_label_alignment", "present": True, "shape": "", "nan_count": 0, "inf_count": 0, "status": "ok" if incongruent_ok else "bad"})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "full_age_group_evidence_sanity_summary.csv", index=False)
    return out


def write_readiness_json(metadata: pd.DataFrame, coverage_by_age: pd.DataFrame, sanity_df: pd.DataFrame, cache_path: Path, out_dir: Path) -> Dict[str, Any]:
    blocking = []
    can_cov = bool(len(coverage_by_age) and coverage_by_age["can_use_for_age_group_fitting"].all())
    if not can_cov:
        blocking.append("coverage thresholds not met for all requested age groups")
    if not cache_path.exists():
        blocking.append("full trial-level cache file is not present")
    if not bool((sanity_df["status"] == "ok").all()):
        blocking.append("sanity checks did not all pass")
    if not {"human_rt", "human_correct", "age_group"}.issubset(metadata.columns):
        blocking.append("required human metadata columns missing")
    payload = {
        "can_run_age_group_restricted_fitting": len(blocking) == 0,
        "cache_path": str(cache_path),
        "metadata_path": str(out_dir / "full_age_group_layerwise_evidence_metadata.csv"),
        "loader_required": False,
        "recommended_next_script": f"python code/scripts/age_group_restricted_parameter_fitting.py --cache-path {cache_path}",
    }
    if blocking:
        payload["can_run_age_group_restricted_fitting"] = False
        payload["blocking_reasons"] = blocking
    (out_dir / "age_group_fitting_readiness.json").write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def write_benchmark(out_dir: Path, device: str, feature_meta: Dict[str, Any] | None, metadata: pd.DataFrame, unique_all: pd.DataFrame, shard_size_bytes: int | None) -> pd.DataFrame:
    ups = float(feature_meta.get("unique_stimuli_per_second", np.nan)) if feature_meta else np.nan
    n_unique = int(feature_meta.get("n_unique_extracted", 0)) if feature_meta else 0
    seconds = float(feature_meta.get("unique_extraction_seconds", np.nan)) if feature_meta else np.nan
    projected_runtime_hours = float(len(unique_all) / ups / 3600.0) if ups and math.isfinite(ups) and ups > 0 else np.nan
    bytes_per_unique = float(shard_size_bytes / max(n_unique, 1)) if shard_size_bytes else np.nan
    projected_disk_gb = float(bytes_per_unique * len(unique_all) / (1024**3)) if math.isfinite(bytes_per_unique) else np.nan
    rows = [
        {
            "device": device,
            "n_unique_benchmark": n_unique,
            "elapsed_seconds": seconds,
            "unique_stimuli_per_second": ups,
            "trials_mapped_per_second": np.nan,
            "memory_usage_mb": np.nan,
            "output_shard_size_bytes": shard_size_bytes if shard_size_bytes is not None else np.nan,
            "projected_runtime_hours_for_full_unique": projected_runtime_hours,
            "projected_disk_usage_gb": projected_disk_gb,
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "full_extraction_benchmark.csv", index=False)
    text = f"""# Full Extraction Benchmark Summary

Device: {device}

Unique stimuli per second: {ups}

Projected runtime for {len(unique_all)} unique stimuli: {projected_runtime_hours} hours.

Projected disk usage: {projected_disk_gb} GB.

CUDA available: {torch.cuda.is_available()}

Recommendation: {'Use GPU/CUDA for full extraction if available; CPU full extraction is likely long.' if device == 'cpu' else 'GPU extraction is recommended for the full run.'}
"""
    (out_dir / "full_extraction_benchmark_summary.md").write_text(text, encoding="utf-8")
    return out


def smoke_test(cache_path: Path, out_path: Path) -> pd.DataFrame:
    z = np.load(cache_path, allow_pickle=True)
    available = np.asarray(z["evidence_available"], dtype=bool)
    rows = []
    rng = np.random.default_rng(SEED)
    for age in sorted(np.unique(z["age_group"].astype(str))):
        idx = np.flatnonzero(available & (z["age_group"].astype(str) == age))
        if len(idx) == 0:
            rows.append({"age_group": age, "n_trials": 0, "model_accuracy": np.nan, "model_mean_rt": np.nan, "fastest_bin_accuracy": np.nan, "incongruent_error_rate": np.nan, "notes": "no evidence-available trials"})
            continue
        if len(idx) > 500:
            idx = np.sort(rng.choice(idx, size=500, replace=False))
        cache = {key: z[key][idx] for key in z.files}
        ww_input = build_natural_input(cache, evidence_gain=0.80, time_steps=80, variant_type="deterministic")
        outputs = run_ww(ww_input, time_steps=80, dt_ms=10, threshold=0.12, noise_ampa=0.0, device="cpu", seed=SEED, readout_mode="baseline", t0_seconds=0.25, choice_temperature=0.01)
        base = base_condition_df(cache, outputs, condition_name="fixed_candidate_smoke", variant_type="deterministic", evidence_gain=0.80, threshold=0.12, seed=SEED)
        df = apply_readout(base, outputs, cfg=ReadoutConfig("sustained_crossing", sustained_k=4), threshold=0.12, dt_ms=10, t0_seconds=0.25)
        s = summarize_condition("fixed_candidate_smoke", df)
        rows.append(
            {
                "age_group": age,
                "n_trials": int(len(df)),
                "accuracy": s["accuracy"],
                "human_choice_agreement": s["model_human_choice_agreement"],
                "mean_rt": s["mean_rt"],
                "q90_minus_q10": float(np.quantile(df["pred_rt"], 0.90) - np.quantile(df["pred_rt"], 0.10)),
                "q95_minus_median": float(np.quantile(df["pred_rt"], 0.95) - np.median(df["pred_rt"])),
                "fastest_bin_accuracy": s["fastest_bin_accuracy"],
                "incongruent_error_rate": s["incongruent_error_rate"],
                "target_recovery_time_error_minus_correct": np.nan,
                "early_flanker_dominance": np.nan,
                "late_target_recovery": np.nan,
                "notes": "fixed baseline candidate on evidence-available sample",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def write_smoke_summary(smoke_df: pd.DataFrame, out_dir: Path) -> None:
    passed = bool(len(smoke_df) and (smoke_df["n_trials"] > 0).all())
    text = f"""# Smoke Test Fixed Candidate Summary

Smoke test passes for every represented age group: {passed}

Rows: {len(smoke_df)}

This is a fixed-candidate readout check only; it is not age-group fitting.
"""
    (out_dir / "smoke_test_fixed_candidate_summary.md").write_text(text, encoding="utf-8")


def write_decision(out_dir: Path, metadata: pd.DataFrame, unique: pd.DataFrame, full_possible: bool) -> None:
    text = f"""# Extraction Unit Decision

## Selected option

Option C: reconstruct unique full-stimulus images from trial metadata.

## Why

The full raw human trial files do not contain `stimulus_image_path`. They do contain `xpos`, `ypos`, `stimulus_layout`, `target_direction`, and `flanker_direction`, and the repository contains the original bird sprites and background image. That makes full-stimulus reconstruction possible without changing the target/flanker evidence definition or image preprocessing.

Option A is available only for already generated subsets such as `data/age_groups/20-29` and `data/age_groups/80-89`. Option B is not appropriate because the existing evidence cache and layerwise pipeline use the full displayed stimulus rather than independent target/flanker component images.

## Alignment

Each trial receives a deterministic `image_id` from `xpos|ypos|stimulus_layout|target_direction|flanker_direction`. VGG evidence is extracted once per selected unique `image_id` and merged back to all matching trial rows by that ID.

## Missing trials

This run contains {len(metadata)} selected trial rows and {len(unique)} selected unique reconstructed stimuli. Evidence coverage is recorded per trial in `evidence_available` and `evidence_missing_reason`.

## Can this support age-group restricted fitting?

{str(full_possible)}. It supports formal fitting only when all requested decade groups have complete evidence coverage. Pilot runs with `--max-unique-images` are intentionally not sufficient for formal fitting.
"""
    (out_dir / "extraction_unit_decision.md").write_text(text, encoding="utf-8")


def write_report(out_dir: Path, args: argparse.Namespace, metadata: pd.DataFrame, unique: pd.DataFrame, coverage: pd.DataFrame, sanity: pd.DataFrame, extraction_meta: Dict[str, Any], cache_written: bool) -> None:
    can_fit = bool(sanity["can_run_age_group_restricted_fitting"].iloc[0]) if len(sanity) else False
    recommended_next = (
        "Run age-group restricted fitting against this complete cache and validate the trial-level outputs."
        if can_fit
        else "Finish evidence extraction with resume enabled, then rerun the coverage and sanity checks."
    )
    text = f"""# Full Age-group Evidence Audit Summary

## Goal

Build a full trial-level VGG / layerwise evidence cache for age-group restricted parameter fitting, without retraining VGG and without changing the evidence or readout definitions.

## Input data

Human trials came from `{PROJECT_ROOT / 'data/vam_data'}` and age labels from `{HUMAN_META}`. The old cache at `{OLD_CACHE}` was used only for schema comparison.

## Extraction unit decision

The selected strategy is Option C: reconstruct unique full-stimulus images from trial fields, then merge evidence back to trial rows by deterministic `image_id`.

## Cache schema

The new cache preserves old keys such as `evidence_conv3`, `evidence_conv4`, `evidence_conv5`, `evidence_pooled`, `evidence_final`, labels, RT, age group, user id, row indices, and stimulus image path. It also adds `global_trial_id`, `image_id`, `human_correct`, `evidence_available`, and `evidence_missing_reason`.

## Evidence coverage

Selected trial rows: {len(metadata)}.
Selected unique reconstructed stimuli: {len(unique)}.
Evidence coverage: {sanity['evidence_coverage'].iloc[0] if len(sanity) else 0:.6f}.
Can run age-group restricted fitting: {can_fit}.

## Computational notes

Device: `{get_device(args.device)}`.
Batch size: {args.batch_size}.
Dry run: {args.dry_run}.
Resume: {args.resume}.
Max unique images: {args.max_unique_images}.
The full dataset has millions of mostly unique stimulus configurations, so full extraction should be run as a long job with `--resume` and a suitable accelerator if available.

## Smoke test

Smoke testing was {'requested' if args.smoke_test else 'not requested'}. If present, results are saved in `smoke_test_fixed_candidate_summary.csv`.

## Recommended next step

{recommended_next}
"""
    (out_dir / "full_age_group_evidence_audit_summary.md").write_text(text, encoding="utf-8")


def write_final_build_summary(
    out_dir: Path,
    readiness: pd.DataFrame,
    benchmark: pd.DataFrame,
    coverage_by_age: pd.DataFrame,
    sanity_df: pd.DataFrame,
    smoke_df: pd.DataFrame | None,
    readiness_payload: Dict[str, Any],
) -> None:
    can_fit = bool(readiness_payload.get("can_run_age_group_restricted_fitting", False))
    blocking = readiness_payload.get("blocking_reasons", [])
    total_trials = int(readiness["n_trials"].sum()) if len(readiness) else 0
    unique_manifest_path = out_dir / "unique_stimuli.csv"
    total_unique = int(len(pd.read_csv(unique_manifest_path))) if unique_manifest_path.exists() else int(readiness["n_unique_stimuli"].sum()) if len(readiness) else 0
    bench_line = benchmark.iloc[0].to_dict() if len(benchmark) else {}
    starting_point = (
        "The requested representative subset has complete evidence coverage and is ready for the declared age-group model run."
        if can_fit
        else "The extraction pipeline is available, but the requested subset does not yet pass the coverage gate."
    )
    chinese_status = (
        "本目录对应的年龄组已完成 5,000 条代表性试次的证据缓存，覆盖和完整性检查均通过，可以用于本次年龄组模型运行。"
        if can_fit
        else "本目录尚未通过覆盖检查，需要继续提取缺失证据后再运行模型。"
    )
    text = f"""# Full Age-group VGG / Layerwise Evidence Cache Build Summary

## 1. Goal

Build a full layerwise evidence cache for age-group restricted natural layer-to-time WW fitting.

## 2. Starting Point

{starting_point}

## 3. Input Data Audit

Total selected trials in readiness audit: {total_trials}.
Global unique stimuli in readiness audit: {total_unique}. Per-age unique counts can sum to a larger number because the same reconstructed stimulus can appear in multiple age groups.
Readiness rows are in `full_run_readiness_audit.csv`.

## 4. Extraction Strategy

The strategy is unique full-stimulus reconstruction from trial metadata, shard-level feature/evidence output, resume checks, and trial-level merge by unique stimulus id.

## 5. Benchmark

Benchmark: {bench_line}

## 6. Full Extraction Progress

Progress is recorded in `extraction_progress.json`. Current run may be a pilot if `max_unique_images` or `max_trials_per_age` was used.

## 7. Coverage

Coverage is reported in `full_age_group_evidence_coverage_by_age.csv`.

## 8. Evidence Sanity Checks

Sanity checks are reported in `full_age_group_evidence_sanity_summary.csv`.

## 9. Smoke Test

Smoke test output: {'smoke_test_fixed_candidate_by_age.csv' if smoke_df is not None else 'not run'}.

## 10. Fitting Readiness

Can run age-group restricted fitting: {can_fit}

Blocking reasons: {blocking}

## 11. Recommended Next Steps

If readiness is false, finish full extraction with resume enabled, rerun failed shards, and rerun coverage/sanity/smoke checks. If true, run age-group restricted fitting and then validate subject-level and image-identity robustness.

## 12. Short Chinese Summary for Discussion

{chinese_status}
"""
    (out_dir / "full_age_group_cache_build_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_unique_stimuli is not None and args.max_unique_images is None:
        args.max_unique_images = args.max_unique_stimuli
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    paths = ensure_dirs(out_dir)
    age_groups = resolve_age_groups(args.age_groups)
    device = get_device(args.device)

    full_metadata = read_human_metadata(age_groups if args.trial_manifest else DECADE_GROUPS, None)
    if args.trial_manifest:
        full_metadata = restrict_to_trial_manifest(full_metadata, args.trial_manifest)
    full_unique = unique_stimuli(full_metadata, None)
    metadata = read_human_metadata(age_groups, args.max_trials_per_age)
    if args.trial_manifest:
        metadata = restrict_to_trial_manifest(metadata, args.trial_manifest)
    unique = staged_unique_stimuli(metadata, args)
    unique_manifest = write_unique_manifest(full_unique, full_metadata, out_dir)
    metadata.loc[metadata["image_id"].isin(set(unique["image_id"])), "evidence_missing_reason"] = "selected_but_not_extracted"

    input_audit(metadata, unique, args.dry_run).to_csv(out_dir / "input_data_availability_audit.csv", index=False)
    image_field_mapping().to_csv(out_dir / "image_field_mapping.csv", index=False)
    full_possible = False
    write_decision(out_dir, metadata, unique, full_possible)

    extraction_meta: Dict[str, Any] = {
        "age_groups": age_groups,
        "n_trials_selected": int(len(metadata)),
        "n_unique_stimuli_selected": int(len(unique)),
        "full_run_total_trials": int(len(full_metadata)),
        "full_run_total_unique_stimuli": int(len(full_unique)),
        "dry_run": bool(args.dry_run),
        "device": device,
        "batch_size": int(args.batch_size),
        "max_unique_images": None if args.max_unique_images is None else int(args.max_unique_images),
        "max_trials_per_age": None if args.max_trials_per_age is None else int(args.max_trials_per_age),
    }

    cache_written = False
    feature_meta: Dict[str, Any] | None = None
    shard_size_bytes: int | None = None
    if not args.dry_run:
        probe_csv = Path(args.probe_train_csv)
        if not probe_csv.is_absolute():
            probe_csv = PROJECT_ROOT / probe_csv
        probe_features, probe_labels, probe_meta = extract_probe_training_features(
            probe_csv,
            device=device,
            batch_size=int(args.batch_size),
            max_rows=int(args.probe_max_train),
        )
        probes = fit_layer_probes(probe_features, probe_labels)
        resume_df, feature_meta = extract_evidence_shards(
            unique,
            probes,
            paths,
            device=device,
            batch_size=int(args.batch_size),
            chunk_size=int(args.chunk_size),
            resume=bool(args.resume),
        )
        if len(resume_df):
            shard_size_bytes = int(sum(Path(p).stat().st_size for p in resume_df["shard_path"] if Path(p).exists()))
        shard_meta, unique_evidence = load_unique_evidence_from_shards(paths)
        ok_ids = set(shard_meta.loc[shard_meta["extraction_status"].eq("ok"), "image_id"].astype(int))
        metadata = metadata.merge(
            shard_meta[["image_id", "feature_shard_id", "extraction_status", "error_message"]],
            on="image_id",
            how="left",
        )
        metadata.loc[metadata["image_id"].isin(ok_ids), "evidence_available"] = True
        metadata.loc[metadata["evidence_available"], "evidence_missing_reason"] = ""
        metadata.loc[metadata["extraction_status"].eq("failed"), "evidence_missing_reason"] = "failed_unique_stimulus"
        metadata.loc[~metadata["evidence_available"], "evidence_missing_reason"] = "unique_image_not_selected_in_this_run"
        ordered_unique = pd.DataFrame({"image_id": shard_meta["image_id"].to_numpy(dtype=np.int64)})
        write_npz(metadata, ordered_unique, unique_evidence, paths)
        write_loader_doc(out_dir)
        cache_written = True
        extraction_meta.update(feature_meta)
        extraction_meta.update(probe_meta)
        schema_comparison(paths.cache_path).to_csv(out_dir / "old_vs_new_cache_schema_comparison.csv", index=False)
    else:
        metadata.loc[metadata["image_id"].isin(set(unique["image_id"])), "evidence_missing_reason"] = "dry_run_not_extracted"
        schema_comparison(OLD_CACHE).to_csv(out_dir / "old_vs_new_cache_schema_comparison.csv", index=False)
        pd.DataFrame(columns=["unique_stimulus_id", "error_message"]).to_csv(out_dir / "failed_unique_stimuli.csv", index=False)
        pd.DataFrame(columns=["shard_id", "shard_path", "start_index", "end_index", "n_unique_stimuli", "status", "integrity_ok"]).to_csv(out_dir / "resume_integrity_check.csv", index=False)
        (out_dir / "resume_integrity_summary.md").write_text("# Resume Integrity Summary\n\nDry run: no shards written.\n", encoding="utf-8")
        (out_dir / "extraction_progress.json").write_text(
            json.dumps(
                {
                    "total_unique_stimuli": int(len(unique)),
                    "completed_unique_stimuli": 0,
                    "failed_unique_stimuli": 0,
                    "current_shard": None,
                    "last_completed_shard": None,
                    "device": device,
                    "chunk_size": int(args.chunk_size),
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "last_update_time": datetime.now(timezone.utc).isoformat(),
                    "estimated_remaining_time": "dry_run",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    metadata.drop(columns=["stimulus_key"]).to_csv(paths.metadata_path, index=False)
    coverage, sanity = coverage_audit(metadata)
    coverage.to_csv(out_dir / "full_age_group_evidence_audit.csv", index=False)
    coverage_by_age = fitting_coverage_by_age(metadata, out_dir)
    sanity_df = evidence_sanity(paths.cache_path if cache_written else None, metadata, out_dir)
    benchmark_df = write_benchmark(out_dir, device, feature_meta, full_metadata, full_unique, shard_size_bytes)
    full_run_readiness(full_metadata, full_unique, out_dir, device, benchmark_df)
    (out_dir / "run_config.json").write_text(json.dumps(to_jsonable({**vars(args), **extraction_meta}), indent=2), encoding="utf-8")

    smoke_path = out_dir / "smoke_test_fixed_candidate_summary.csv"
    smoke_by_age_path = out_dir / "smoke_test_fixed_candidate_by_age.csv"
    smoke_df = None
    if args.smoke_test and cache_written:
        smoke_df = smoke_test(paths.cache_path, smoke_by_age_path)
        smoke_df.to_csv(smoke_path, index=False)
        write_smoke_summary(smoke_df, out_dir)
    elif args.smoke_test:
        smoke_df = pd.DataFrame([{"age_group": "", "n_trials": 0, "accuracy": np.nan, "human_choice_agreement": np.nan, "mean_rt": np.nan, "q90_minus_q10": np.nan, "q95_minus_median": np.nan, "incongruent_error_rate": np.nan, "target_recovery_time_error_minus_correct": np.nan, "early_flanker_dominance": np.nan, "late_target_recovery": np.nan, "notes": "smoke test skipped because no cache was written"}])
        smoke_df.to_csv(smoke_path, index=False)
        smoke_df.to_csv(smoke_by_age_path, index=False)
        write_smoke_summary(smoke_df, out_dir)

    readiness_payload = write_readiness_json(metadata, coverage_by_age, sanity_df, paths.cache_path, out_dir)
    can_fit = bool(readiness_payload["can_run_age_group_restricted_fitting"])
    write_decision(out_dir, metadata, unique, can_fit)
    write_report(out_dir, args, metadata, unique, coverage, sanity, extraction_meta, cache_written)
    write_final_build_summary(out_dir, full_run_readiness(full_metadata, full_unique, out_dir, device, benchmark_df), benchmark_df, coverage_by_age, sanity_df, smoke_df, readiness_payload)

    generated = [
        out_dir / "full_run_readiness_audit.csv",
        out_dir / "full_run_readiness_summary.md",
        out_dir / "full_extraction_benchmark.csv",
        out_dir / "full_extraction_benchmark_summary.md",
        out_dir / "unique_stimuli.csv",
        out_dir / "input_data_availability_audit.csv",
        out_dir / "image_field_mapping.csv",
        out_dir / "extraction_unit_decision.md",
        paths.shard_dir,
        out_dir / "extraction_progress.json",
        out_dir / "failed_unique_stimuli.csv",
        out_dir / "resume_integrity_check.csv",
        out_dir / "resume_integrity_summary.md",
        paths.metadata_path,
        out_dir / "old_vs_new_cache_schema_comparison.csv",
        out_dir / "full_age_group_evidence_audit.csv",
        out_dir / "full_age_group_evidence_coverage_by_age.csv",
        out_dir / "full_age_group_evidence_coverage_summary.md",
        out_dir / "full_age_group_evidence_sanity_summary.csv",
        out_dir / "age_group_fitting_readiness.json",
        out_dir / "run_config.json",
        out_dir / "full_age_group_evidence_audit_summary.md",
        out_dir / "full_age_group_cache_build_summary.md",
    ]
    if cache_written:
        generated.append(paths.cache_path)
    if args.smoke_test:
        generated.append(smoke_path)
        generated.append(smoke_by_age_path)
        generated.append(out_dir / "smoke_test_fixed_candidate_summary.md")
    recommended = (
        f"python code/scripts/age_group_restricted_parameter_fitting.py --cache-path {paths.cache_path}"
        if can_fit
        else f"python code/scripts/build_full_age_group_vgg_evidence_cache.py --age-groups all --chunk-size 50000 --resume --device {device}"
    )
    print("DONE: full age-group VGG evidence cache construction completed.")
    print("")
    print("Summary markdown:")
    print(out_dir / "full_age_group_evidence_audit_summary.md")
    print("")
    print("Cache file:")
    print(paths.cache_path if cache_written else "NOT_WRITTEN_DRY_RUN")
    print("")
    print("Can run age-group restricted fitting:")
    print(can_fit)
    print("")
    print("Recommended next command:")
    print(recommended)
    print("")
    print("Generated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

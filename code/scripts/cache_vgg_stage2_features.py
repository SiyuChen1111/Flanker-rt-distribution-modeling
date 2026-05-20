import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import CHECKPOINTS_TEST_ROOT, PROJECT_ROOT
from run_dynamic_selection_single_subject import build_alignment_report
from train_age_groups_efficient import DIRECTION_MAP, StimulusDataset, load_cached_logits_npz, to_jsonable
from vgg_wongwang_lim import VGGFeatureExtractor


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class VGGFeatureTap(VGGFeatureExtractor):
    def forward_features_and_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        original_device = x.device
        if original_device.type == "mps":
            x = self.avgpool(x.cpu())
            x = x.to(original_device)
        else:
            x = self.avgpool(x)
        pooled = torch.flatten(x, 1)
        logits = self.classifier(pooled)
        return pooled, logits


def build_loader(csv_path: Path, batch_size: int) -> Tuple[StimulusDataset, DataLoader]:
    dataset = StimulusDataset(str(csv_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return dataset, loader


def resolve_stage1_checkpoint() -> Path:
    return CHECKPOINTS_TEST_ROOT / "stage1" / "best_model.pth"


def load_stage1_model_with_metadata(device: str) -> Tuple[VGGFeatureTap, Dict[str, Any]]:
    stage1_path = resolve_stage1_checkpoint()
    if not stage1_path.exists():
        raise FileNotFoundError(f"Missing Stage-1 checkpoint at {stage1_path}")
    model = VGGFeatureTap(pretrained=False, n_classes=4)
    checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    metadata = {
        "stage1_checkpoint_path": str(stage1_path),
        "strict_load": False,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "loaded_key_count": int(len(checkpoint["model_state_dict"])),
    }
    return model, metadata


def load_stage1_model(device: str) -> VGGFeatureTap:
    model, _ = load_stage1_model_with_metadata(device)
    return model


def extract_features_and_logits(model: VGGFeatureTap, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    feature_batches = []
    logit_batches = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features, logits = model.forward_features_and_logits(images)
            feature_batches.append(features.cpu().numpy())
            logit_batches.append(logits.cpu().numpy())
    return np.concatenate(feature_batches, axis=0), np.concatenate(logit_batches, axis=0)


def compute_alignment_hash(cached_logits: Dict[str, np.ndarray], features: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cached_logits["target_labels"], dtype=np.int64).tobytes())
    digest.update(np.asarray(cached_logits["response_labels"], dtype=np.int64).tobytes())
    digest.update(np.asarray(cached_logits["rts"], dtype=np.float32).tobytes())
    digest.update(np.asarray(cached_logits["congruency"], dtype=np.int64).tobytes())
    digest.update(np.asarray(features.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def save_feature_npz(
    path: Path,
    dataset: StimulusDataset,
    cached_logits: Dict[str, np.ndarray],
    features: np.ndarray,
    logits_from_model: np.ndarray,
) -> None:
    output = {
        "features": features.astype(np.float32),
        "pooled_features": features.astype(np.float32),
        "logits": logits_from_model.astype(np.float32),
        "cached_logits": np.asarray(cached_logits["logits"], dtype=np.float32),
        "row_indices": np.arange(len(dataset), dtype=np.int64),
        "target_labels": np.asarray(cached_logits["target_labels"], dtype=np.int64),
        "response_labels": np.asarray(cached_logits["response_labels"], dtype=np.int64),
        "flanker_labels": np.asarray(cached_logits.get("flanker_labels", dataset.flanker_labels), dtype=np.int64),
        "congruency": np.asarray(cached_logits["congruency"], dtype=np.int64),
        "rts": np.asarray(cached_logits["rts"], dtype=np.float32),
        "rts_normalized": np.asarray(cached_logits["rts_normalized"], dtype=np.float32),
        "user_id": dataset.data["user_id"].astype(str).to_numpy(),
        "stimulus_image_path": dataset.image_paths,
    }
    np.savez_compressed(path, **output)


def verify_split_alignment(csv_path: Path, cached_logits: Dict[str, np.ndarray]) -> None:
    df = pd.read_csv(csv_path)
    build_alignment_report(df, cached_logits)


def sync_cached_labels_from_csv(csv_path: Path, cached_logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    synced = dict(cached_logits)
    synced["flanker_labels"] = df["flanker_direction"].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
    return synced


def process_split(
    *,
    split: str,
    age_group: str,
    data_dir: Path,
    stage2_dir: Path,
    output_dir: Path,
    model: VGGFeatureTap,
    device: str,
    batch_size: int,
) -> Dict[str, object]:
    csv_path = data_dir / f"{split}_data.csv"
    logits_path = stage2_dir / f"{split}_logits.npz"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for split={split}: {csv_path}")
    if not logits_path.exists():
        raise FileNotFoundError(f"Missing cached logits for split={split}: {logits_path}")

    dataset, loader = build_loader(csv_path, batch_size=batch_size)
    cached_logits = load_cached_logits_npz(str(logits_path))
    cached_logits = sync_cached_labels_from_csv(csv_path, cached_logits)
    verify_split_alignment(csv_path, cached_logits)
    if len(dataset) != len(cached_logits["logits"]):
        raise ValueError(
            f"FEATURE_CACHE_LENGTH_MISMATCH: age_group={age_group} split={split} csv={len(dataset)} cached={len(cached_logits['logits'])}"
        )

    features, logits_from_model = extract_features_and_logits(model, loader, device)
    if features.shape[0] != logits_from_model.shape[0] or features.shape[0] != len(dataset):
        raise ValueError(
            f"FEATURE_CACHE_SHAPE_MISMATCH: split={split} features={features.shape} logits={logits_from_model.shape} n={len(dataset)}"
        )

    output_path = output_dir / f"{split}_features.npz"
    save_feature_npz(output_path, dataset, cached_logits, features, logits_from_model)
    return {
        "split": split,
        "n_rows": int(len(dataset)),
        "feature_dim": int(features.shape[1]),
        "alignment_hash": compute_alignment_hash(cached_logits, features),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen VGG Stage-2 features with row-aligned metadata.")
    parser.add_argument("--age_group", required=True, choices=["20-29", "80-89"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--feature_layer", choices=["penultimate"], default="penultimate")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = data_root / args.age_group
    stage2_dir = checkpoint_root / args.age_group / "stage2"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing age-group data directory: {data_dir}")
    if not stage2_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 checkpoint directory: {stage2_dir}")

    model = load_stage1_model(device)
    splits = [args.split] if args.split != "both" else ["train", "test"]
    records = []
    for split in splits:
        records.append(
            process_split(
                split=split,
                age_group=args.age_group,
                data_dir=data_dir,
                stage2_dir=stage2_dir,
                output_dir=output_dir,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )
        )

    feature_dim = int(records[0]["feature_dim"] if records else 0)
    alignment_hash = hashlib.sha256("".join(record["alignment_hash"] for record in records).encode("utf-8")).hexdigest()
    manifest = {
        "age_group": args.age_group,
        "data_root": str(data_root),
        "checkpoint_path": str(resolve_stage1_checkpoint()),
        "stage2_checkpoint_root": str(stage2_dir),
        "feature_layer": args.feature_layer,
        "feature_dim": feature_dim,
        "n_train": int(next((r["n_rows"] for r in records if r["split"] == "train"), 0)),
        "n_test": int(next((r["n_rows"] for r in records if r["split"] == "test"), 0)),
        "alignment_hash": alignment_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "records": records,
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

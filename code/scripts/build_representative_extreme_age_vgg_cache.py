#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_global_vgg_evidence_cache import (  # noqa: E402
    REQUIRED_EVIDENCE_KEYS as EVIDENCE_KEYS,
    four_class_scores,
    load_extraction_deps,
    simple_jsonable,
)
from project_paths import PROJECT_ROOT  # noqa: E402

ANALYSIS_NAME = "representative_extreme_age_subset_5000"
OUT_DIR = PROJECT_ROOT / "artifacts/results/diagnostics/natural_layer_to_time_var_ww" / ANALYSIS_NAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build representative subset stimulus-level VGG/layerwise evidence cache.")
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--probe-train-csv", default="data/age_groups/20-29/train_data.csv")
    p.add_argument("--probe-max-train", type=int, default=2000)
    return p.parse_args()


def require_gate(root: Path) -> None:
    gate_path = root / "audits/representativeness_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Representativeness gate is missing; run build_representative_extreme_age_subset.py first.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_passed", False):
        raise RuntimeError("Representativeness gate did not pass; evidence extraction is blocked.")


def extract_evidence(root: Path, unique: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    npz_path = root / "evidence_cache/representative_subset_layerwise_evidence.npz"
    meta_path = root / "evidence_cache/representative_subset_layerwise_evidence_metadata.csv"
    shard_dir = root / "evidence_cache/shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and npz_path.exists() and meta_path.exists():
        return pd.read_csv(meta_path)

    deps = load_extraction_deps()
    torch = deps["torch"]
    device = deps["get_device"](args.device)
    probe_csv = Path(args.probe_train_csv)
    if not probe_csv.is_absolute():
        probe_csv = PROJECT_ROOT / probe_csv

    t0 = time.perf_counter()
    probe_features, probe_labels, model_meta = deps["extract_probe_training_features"](
        probe_csv, device=device, batch_size=args.batch_size, max_rows=args.probe_max_train
    )
    probes = deps["fit_layer_probes"](probe_features, probe_labels)
    base_model, stage_meta = deps["load_stage1_model_with_metadata"](device)
    model = deps["LayerwiseFeatureTap"](base_model).to(device)
    model.eval()
    renderer = deps["StimulusRenderer"]()
    layer_order = ["conv3", "conv4", "conv5", "pooled", "final_logits"]
    evidence_parts: Dict[str, List[np.ndarray]] = {k: [] for k in EVIDENCE_KEYS}
    status: List[str] = []
    errors: List[str] = []

    for start in range(0, len(unique), args.chunk_size):
        part = unique.iloc[start : start + args.chunk_size].copy()
        shard_path = shard_dir / f"representative_subset_evidence_shard_{start // args.chunk_size:06d}.npz"
        if args.resume and shard_path.exists():
            z = np.load(shard_path, allow_pickle=True)
            for key in EVIDENCE_KEYS:
                evidence_parts[key].append(z[key].astype(np.float32))
            status.extend(z["extraction_status"].astype(str).tolist())
            errors.extend(z["error_message"].astype(str).tolist())
            continue
        shard_features: Dict[str, List[np.ndarray]] = {k: [] for k in layer_order}
        shard_status = np.array(["ok"] * len(part), dtype=object)
        shard_errors = np.array([""] * len(part), dtype=object)
        with torch.no_grad():
            for bstart in range(0, len(part), args.batch_size):
                batch = part.iloc[bstart : bstart + args.batch_size]
                tensors = []
                good = []
                for off, (_, row) in enumerate(batch.iterrows()):
                    try:
                        render_row = pd.Series(
                            {
                                "xpos": row["xpos"],
                                "ypos": row["ypos"],
                                "stimulus_layout": row["layout"],
                                "target_label": row["target_label"],
                                "flanker_label": row["flanker_label"],
                            }
                        )
                        tensors.append(renderer.render_tensor(render_row))
                        good.append(off)
                    except Exception as exc:
                        shard_status[bstart + off] = "failed"
                        shard_errors[bstart + off] = str(exc)
                if not tensors:
                    for layer in layer_order:
                        shard_features[layer].append(np.full((len(batch), 4), np.nan, dtype=np.float32))
                    continue
                images = torch.stack(tensors, dim=0).to(device)
                vals = model.forward_layerwise(images)
                for layer in layer_order:
                    arr = np.full((len(batch), vals[layer].shape[1]), np.nan, dtype=np.float32)
                    arr[np.asarray(good, dtype=np.int64)] = vals[layer].detach().cpu().numpy().astype(np.float32)
                    shard_features[layer].append(arr)
        features = {layer: np.concatenate(parts, axis=0) for layer, parts in shard_features.items()}
        shard_evidence = {
            "evidence_conv3": four_class_scores(probes["conv3"], features["conv3"]),
            "evidence_conv4": four_class_scores(probes["conv4"], features["conv4"]),
            "evidence_conv5": four_class_scores(probes["conv5"], features["conv5"]),
            "evidence_pooled": four_class_scores(probes["pooled"], features["pooled"]),
            "evidence_final": features["final_logits"].astype(np.float32),
        }
        payload = {
            "subset_stimulus_id": part["subset_stimulus_id"].to_numpy(dtype=np.int64),
            "global_stimulus_key": part["global_stimulus_key"].astype(str).to_numpy(),
            "target_label": part["target_label"].to_numpy(dtype=np.int64),
            "flanker_label": part["flanker_label"].to_numpy(dtype=np.int64),
            "congruency": part["congruency"].to_numpy(dtype=np.int64),
            "extraction_status": shard_status,
            "error_message": shard_errors,
            **{k: v.astype(np.float32) for k, v in shard_evidence.items()},
        }
        np.savez_compressed(shard_path, **payload)
        for key in EVIDENCE_KEYS:
            evidence_parts[key].append(shard_evidence[key].astype(np.float32))
        status.extend(shard_status.astype(str).tolist())
        errors.extend(shard_errors.astype(str).tolist())

    payload = {
        "subset_stimulus_id": unique["subset_stimulus_id"].to_numpy(dtype=np.int64),
        "global_stimulus_key": unique["global_stimulus_key"].astype(str).to_numpy(),
        "target_labels": unique["target_label"].to_numpy(dtype=np.int64),
        "flanker_labels": unique["flanker_label"].to_numpy(dtype=np.int64),
        "congruency": unique["congruency"].to_numpy(dtype=np.int64),
        "extraction_status": np.asarray(status, dtype=object),
        "error_message": np.asarray(errors, dtype=object),
    }
    for key in EVIDENCE_KEYS:
        payload[key] = np.concatenate(evidence_parts[key], axis=0).astype(np.float32)
    np.savez_compressed(npz_path, **payload)
    meta = unique.copy()
    meta["extraction_status"] = status
    meta["error_message"] = errors
    meta.to_csv(meta_path, index=False)
    run_meta = {
        "device": device,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "n_unique_stimuli": int(len(unique)),
        "n_failed": int((meta["extraction_status"] != "ok").sum()),
        "seconds": float(time.perf_counter() - t0),
        "probe_metadata": simple_jsonable(model_meta),
        "stage1_metadata": simple_jsonable(stage_meta),
    }
    (root / "evidence_cache/extraction_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return meta


def write_audit(root: Path, meta: pd.DataFrame) -> None:
    failed = int((meta["extraction_status"] != "ok").sum())
    coverage = float((meta["extraction_status"] == "ok").mean()) if len(meta) else 0.0
    pd.DataFrame(
        [
            {
                "analysis_name": ANALYSIS_NAME,
                "n_unique_stimuli": int(len(meta)),
                "ok_stimuli": int((meta["extraction_status"] == "ok").sum()),
                "failed_stimuli": failed,
                "evidence_coverage": coverage,
                "coverage_gate_passed": bool(coverage == 1.0 and failed == 0),
            }
        ]
    ).to_csv(root / "audits/representative_subset_evidence_coverage_audit.csv", index=False)
    text = f"""# Representative Subset Evidence Summary

Stimulus-level layerwise evidence cache for `{ANALYSIS_NAME}`.

- unique stimuli: {len(meta)}
- evidence coverage: {coverage:.6f}
- failed stimuli: {failed}
- coverage gate passed: {coverage == 1.0 and failed == 0}

Evidence was extracted once per `global_stimulus_key`. It is not age-level or subject-level evidence.
"""
    (root / "summaries/representative_subset_evidence_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    require_gate(root)
    unique_path = root / "manifests/representative_subset_unique_stimuli.csv"
    if not unique_path.exists():
        raise RuntimeError("Unique stimulus manifest is missing.")
    unique = pd.read_csv(unique_path)
    meta = extract_evidence(root, unique, args)
    write_audit(root, meta)
    failed = int((meta["extraction_status"] != "ok").sum())
    print(json.dumps({"n_unique_stimuli": int(len(meta)), "failed_stimuli": failed, "evidence_coverage": float((meta["extraction_status"] == "ok").mean())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

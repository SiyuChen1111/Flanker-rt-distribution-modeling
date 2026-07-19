#!/usr/bin/env python3
"""Audit spatially separated target/flanker evidence before dual-route WW fitting.

This entry point deliberately stops after the representation audit unless the
source evidence is usable.  It keeps the trained VGG and direction probes fixed
and changes only which stimulus elements are rendered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_layerwise_feature_probe import LayerwiseFeatureTap  # noqa: E402
from build_full_age_group_vgg_evidence_cache import (  # noqa: E402
    LAYER_ORDER,
    LAYOUT_SPACING,
    WIN_SIZE,
    StimulusRenderer,
    distractor_positions,
    extract_probe_training_features,
    fit_layer_probes,
    four_class_scores,
    get_device,
    load_stage1_model_with_metadata,
)
from project_paths import PROJECT_ROOT  # noqa: E402


BASE_DIR = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "dual_source_conflict_test"
MANIFEST_PATH = BASE_DIR / "manifests/representative_subset_unique_stimuli.csv"
EXISTING_CACHE_PATH = BASE_DIR / "evidence_cache/representative_subset_layerwise_evidence.npz"
EVIDENCE_LAYERS = ["conv3", "conv4", "conv5", "pooled", "final"]
VARIANTS = ("full", "target_only", "flanker_only")
SEED = 20260719


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target/flanker source-separated VGG evidence audit.")
    p.add_argument("--run-id", required=True, help="New output directory name.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--max-stimuli", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--probe-train-csv", default="data/age_groups/20-29/train_data.csv")
    p.add_argument("--probe-max-train", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class SourceSeparatedRenderer(StimulusRenderer):
    """Render the same stimulus with full, target-only, or flanker-only content."""

    def render_image(self, row: pd.Series, variant: str) -> Image.Image:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown stimulus variant: {variant}")
        canvas = self.background.copy()
        xpos_centered = float(row["xpos"]) - WIN_SIZE[0] / 2
        ypos_centered = -float(row["ypos"]) + WIN_SIZE[1] / 2
        target_pos = (xpos_centered, ypos_centered)
        layout = int(row.get("stimulus_layout", row.get("layout")))
        spacer = LAYOUT_SPACING[layout]
        if variant in {"full", "target_only"}:
            self.paste_center(canvas, self.birds[int(row["target_label"])], target_pos)
        if variant in {"full", "flanker_only"}:
            for pos in distractor_positions(target_pos, layout, spacer):
                self.paste_center(canvas, self.birds[int(row["flanker_label"])], pos)
        return canvas.convert("RGB")

    def render_tensor(self, row: pd.Series, variant: str = "full") -> torch.Tensor:
        return self.transform(self.render_image(row, variant))


def balanced_manifest_sample(manifest: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Deterministically sample across congruency and both direction labels."""
    if n >= len(manifest):
        return manifest.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    strata = manifest.groupby(["congruency", "target_label", "flanker_label"], sort=True).groups
    selected: list[int] = []
    per = max(1, n // max(1, len(strata)))
    for idx in strata.values():
        arr = np.asarray(list(idx), dtype=np.int64)
        selected.extend(rng.choice(arr, size=min(per, len(arr)), replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < n:
        remaining = np.setdiff1d(manifest.index.to_numpy(), np.asarray(selected, dtype=np.int64))
        selected.extend(rng.choice(remaining, size=n - len(selected), replace=False).tolist())
    return manifest.loc[sorted(selected[:n])].reset_index(drop=True)


def ensure_output(args: argparse.Namespace) -> Path:
    out = Path(args.output_root) / args.run_id
    if out.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {out}. Use a new --run-id or --resume.")
    for name in ["cache", "metrics", "figures", "summaries", "audits"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    required = {"subset_stimulus_id", "xpos", "ypos", "layout", "target_label", "flanker_label", "congruency"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    out = df.copy()
    out["stimulus_layout"] = out["layout"].astype(int)
    for col in ["subset_stimulus_id", "target_label", "flanker_label", "congruency"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    if out["subset_stimulus_id"].duplicated().any():
        raise ValueError("subset_stimulus_id is not one-to-one in the manifest")
    return out


def _feature_to_evidence(layer: str, value: np.ndarray, probes: dict[str, Any]) -> np.ndarray:
    if layer == "final_logits":
        return value.astype(np.float32)
    return four_class_scores(probes[layer], value).astype(np.float32)


def extract_source_evidence(manifest: pd.DataFrame, args: argparse.Namespace, out: Path) -> dict[str, np.ndarray]:
    cache_path = out / "cache/source_separated_evidence.npz"
    if args.resume and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    device = get_device(args.device)
    probe_csv = Path(args.probe_train_csv)
    if not probe_csv.is_absolute():
        probe_csv = PROJECT_ROOT / probe_csv
    probe_features, probe_labels, probe_meta = extract_probe_training_features(
        probe_csv, device=device, batch_size=args.batch_size, max_rows=args.probe_max_train
    )
    probes = fit_layer_probes(probe_features, probe_labels)
    base_model, model_meta = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(base_model).to(device)
    model.eval()
    renderer = SourceSeparatedRenderer()
    accum: dict[str, list[np.ndarray]] = {
        f"evidence_{variant}_{layer}": [] for variant in VARIANTS for layer in EVIDENCE_LAYERS
    }
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(manifest), args.batch_size):
            part = manifest.iloc[start : start + args.batch_size]
            for variant in VARIANTS:
                images = torch.stack(
                    [renderer.render_tensor(row, variant) for _, row in part.iterrows()], dim=0
                ).to(device)
                values = model.forward_layerwise(images)
                for raw_layer, short_layer in zip(LAYER_ORDER, EVIDENCE_LAYERS):
                    arr = values[raw_layer].detach().cpu().numpy().astype(np.float32)
                    accum[f"evidence_{variant}_{short_layer}"].append(
                        _feature_to_evidence(raw_layer, arr, probes)
                    )
            done = min(start + args.batch_size, len(manifest))
            print(json.dumps({"extracted": done, "total": len(manifest)}, ensure_ascii=False), flush=True)

    payload: dict[str, np.ndarray] = {
        "subset_stimulus_id": manifest["subset_stimulus_id"].to_numpy(np.int64),
        "target_labels": manifest["target_label"].to_numpy(np.int64),
        "flanker_labels": manifest["flanker_label"].to_numpy(np.int64),
        "congruency": manifest["congruency"].to_numpy(np.int64),
    }
    payload.update({k: np.concatenate(v, axis=0).astype(np.float32) for k, v in accum.items()})
    np.savez_compressed(cache_path, **payload)
    manifest.to_csv(out / "cache/source_separated_manifest.csv", index=False)
    metadata = {
        "device": device,
        "n_stimuli": int(len(manifest)),
        "batch_size": int(args.batch_size),
        "probe_max_train": int(args.probe_max_train),
        "elapsed_seconds": float(time.perf_counter() - started),
        "probe_metadata": str(probe_meta),
        "model_metadata": str(model_meta),
        "variants": list(VARIANTS),
    }
    (out / "cache/extraction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return payload


def directional_margin(evidence: np.ndarray, label: np.ndarray) -> np.ndarray:
    rows = np.arange(len(evidence))
    chosen = evidence[rows, label]
    other = evidence.copy()
    other[rows, label] = -np.inf
    return chosen - other.max(axis=1)


def source_separation_pass(audit: pd.DataFrame) -> bool:
    """Pre-full-run gate: each source needs one reliable non-final evidence layer."""
    all_rows = audit[audit["condition"].eq("all") & audit["layer"].ne("final")]
    checks = []
    for variant in ["target_only", "flanker_only"]:
        part = all_rows[all_rows["variant"].eq(variant)]
        checks.append(bool((part["direction_accuracy"] >= 0.90).any() and (part["mean_expected_margin"] > 0).any()))
    return bool(all(checks))


def representation_audit(payload: dict[str, np.ndarray], out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = payload["target_labels"].astype(int)
    flanker = payload["flanker_labels"].astype(int)
    congruency = payload["congruency"].astype(int)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        expected = flanker if variant == "flanker_only" else target
        for layer in EVIDENCE_LAYERS:
            ev = np.asarray(payload[f"evidence_{variant}_{layer}"], dtype=float)
            pred = ev.argmax(axis=1)
            margin = directional_margin(ev, expected)
            for condition, mask in [("all", np.ones(len(ev), dtype=bool)), ("congruent", congruency == 0), ("incongruent", congruency == 1)]:
                rows.append(
                    {
                        "variant": variant,
                        "layer": layer,
                        "condition": condition,
                        "expected_source": "flanker" if variant == "flanker_only" else "target",
                        "n": int(mask.sum()),
                        "direction_accuracy": float((pred[mask] == expected[mask]).mean()),
                        "mean_expected_margin": float(margin[mask].mean()),
                    }
                )
    audit = pd.DataFrame(rows)

    full_cache_check = {"available": False, "n_matched": 0, "mean_layer_correlation": np.nan, "max_abs_error": np.nan}
    if EXISTING_CACHE_PATH.exists():
        with np.load(EXISTING_CACHE_PATH, allow_pickle=True) as old:
            old_ids = old["subset_stimulus_id"].astype(int)
            lookup = {int(v): i for i, v in enumerate(old_ids)}
            ids = payload["subset_stimulus_id"].astype(int)
            old_idx = np.asarray([lookup[int(v)] for v in ids], dtype=int)
            cors, max_err, agreements = [], [], []
            for layer in EVIDENCE_LAYERS:
                new = np.asarray(payload[f"evidence_full_{layer}"], dtype=float)
                previous = np.asarray(old[f"evidence_{layer}"][old_idx], dtype=float)
                cors.append(float(np.corrcoef(new.ravel(), previous.ravel())[0, 1]))
                max_err.append(float(np.max(np.abs(new - previous))))
                agreements.append(float((new.argmax(axis=1) == previous.argmax(axis=1)).mean()))
            full_cache_check = {
                "available": True,
                "n_matched": int(len(ids)),
                "mean_layer_correlation": float(np.mean(cors)),
                "min_layer_correlation": float(np.min(cors)),
                "max_abs_error": float(np.max(max_err)),
                "min_layer_argmax_agreement": float(np.min(agreements)),
            }

    source_gate = source_separation_pass(audit)
    reconstruction_gate = bool(
        full_cache_check["available"]
        and full_cache_check["mean_layer_correlation"] >= 0.99
        and full_cache_check.get("min_layer_argmax_agreement", 0.0) >= 0.99
    )
    verdict = {
        "source_separation_gate_passed": source_gate,
        "reconstruction_gate_passed": reconstruction_gate,
        "representation_audit_passed": bool(source_gate and reconstruction_gate),
        "gate_definition": {
            "source": "at least one non-final layer >= .90 with positive expected-direction margin for both target-only and flanker-only",
            "reconstruction": "mean layer correlation >= .99 and minimum layer argmax agreement >= .99",
        },
        "gate_revision_status": "fixed after smoke calibration and before the full-sample audit",
        "full_cache_check": full_cache_check,
    }
    audit.to_csv(out / "metrics/source_direction_decoding_audit.csv", index=False)
    (out / "audits/representation_gate.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return audit, verdict


def plot_audit(audit: pd.DataFrame, out: Path) -> None:
    data = audit[audit["condition"].eq("all") & audit["variant"].isin(["target_only", "flanker_only"])].copy()
    order = EVIDENCE_LAYERS
    colors = {"target_only": "#0072B2", "flanker_only": "#E69F00"}
    labels = {"target_only": "Target-only: target direction", "flanker_only": "Flanker-only: flanker direction"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    x = np.arange(len(order))
    for variant, marker in [("target_only", "o"), ("flanker_only", "s")]:
        part = data[data["variant"].eq(variant)].set_index("layer").loc[order]
        ax.plot(x, part["direction_accuracy"], marker=marker, linewidth=2, color=colors[variant], label=labels[variant])
    ax.axhline(0.25, color="#666666", linestyle="--", linewidth=1, label="Chance")
    ax.axhline(0.75, color="#999999", linestyle=":", linewidth=1, label="Audit threshold")
    ax.set_xticks(x, ["Conv3", "Conv4", "Conv5", "Pooled", "Final"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Direction decoding accuracy")
    ax.set_xlabel("Frozen VGG evidence layer")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(direction="in")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"figures/source_direction_decoding.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(args: argparse.Namespace, out: Path, verdict: dict[str, Any], audit: pd.DataFrame) -> None:
    all_rows = audit[audit["condition"].eq("all")]
    lines = [
        "# Flanker 双通道表征审计",
        "",
        f"- 模式：{args.mode}",
        f"- 刺激数量：{int(all_rows['n'].max())}",
        f"- 完整刺激重建通过：{verdict['reconstruction_gate_passed']}",
        f"- target/flanker 来源分离通过：{verdict['source_separation_gate_passed']}",
        f"- 是否允许进入双通道 Wong–Wang 比较：{verdict['representation_audit_passed']}",
        "",
        "## 各层方向解码",
        "",
    ]
    for variant in ["target_only", "flanker_only"]:
        part = all_rows[all_rows["variant"].eq(variant)]
        values = ", ".join(f"{r.layer}={r.direction_accuracy:.3f}" for r in part.itertuples())
        lines.append(f"- {variant}: {values}")
    lines += [
        "",
        "## 下一步",
        "",
        "- 只有正式 full 审计通过后，才运行同步、双通道和反向时序对照的 Wong–Wang 比较。",
        "- 若正式审计失败，停止后端拟合，并把失败定位为当前 VGG 方向证据无法可靠区分空间来源。",
        "- 来源门槛在冒烟校准后、正式全样本运行前固定；正式结果产生后不再调整。",
    ]
    (out / "summaries/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output(args)
    manifest = normalize_manifest(pd.read_csv(MANIFEST_PATH))
    max_stimuli = args.max_stimuli
    if args.mode == "smoke" and max_stimuli is None:
        max_stimuli = 96
    if max_stimuli is not None:
        manifest = balanced_manifest_sample(manifest, min(int(max_stimuli), len(manifest)))
    payload = extract_source_evidence(manifest, args, out)
    audit, verdict = representation_audit(payload, out)
    plot_audit(audit, out)
    write_summary(args, out, verdict, audit)
    print(json.dumps({"output_dir": str(out), **verdict}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

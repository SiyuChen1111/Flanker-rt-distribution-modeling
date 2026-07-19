#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cache_vgg_stage2_features import load_stage1_model_with_metadata
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import DIRECTION_MAP, StimulusDataset, to_jsonable


TASKS = {
    "target": "target_label",
    "flanker": "flanker_label",
    "congruency": "congruency",
}

LAYER_TAPS = {
    "conv3": 15,
    "conv4": 22,
    "conv5": 29,
}

LAYER_ORDER = ["conv3", "conv4", "conv5", "pooled", "final_logits"]
CHANCE_LEVEL = {"target": 0.25, "flanker": 0.25, "congruency": 0.50}


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def balanced_indices(dataset: StimulusDataset, max_rows: int, seed: int) -> np.ndarray:
    n_rows = len(dataset)
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    keys = np.asarray(
        [
            f"{dataset.target_labels[i]}_{dataset.flanker_labels[i]}_{dataset.response_labels[i] != dataset.target_labels[i]}"
            for i in range(n_rows)
        ]
    )
    selected: List[int] = []
    unique_keys = np.unique(keys)
    per_key = max(1, max_rows // max(len(unique_keys), 1))
    for key in unique_keys:
        candidates = np.flatnonzero(keys == key)
        take = min(len(candidates), per_key)
        if take:
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < max_rows:
        remaining = np.setdiff1d(np.arange(n_rows, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        take = min(len(remaining), max_rows - len(selected))
        if take:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return np.asarray(sorted(selected[:max_rows]), dtype=np.int64)


class LayerwiseFeatureTap(nn.Module):
    def __init__(self, stage1_model: nn.Module):
        super().__init__()
        self.features = stage1_model.features
        self.avgpool = stage1_model.avgpool
        self.classifier = stage1_model.classifier

    def forward_layerwise(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        z = x
        for idx, layer in enumerate(self.features):
            z = layer(z)
            for name, tap_idx in LAYER_TAPS.items():
                if idx == tap_idx:
                    outputs[name] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(z, (1, 1)), 1)
        original_device = z.device
        if original_device.type == "mps":
            pooled_map = self.avgpool(z.cpu()).to(original_device)
        else:
            pooled_map = self.avgpool(z)
        pooled = torch.flatten(pooled_map, 1)
        outputs["pooled"] = pooled
        outputs["final_logits"] = self.classifier(pooled)
        return outputs


def extract_layerwise_features(
    csv_path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    dataset = StimulusDataset(str(csv_path))
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    stage1_model, model_metadata = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(stage1_model).to(device)
    model.eval()

    batches: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
    with torch.no_grad():
        for batch in loader:
            values = model.forward_layerwise(batch["image"].to(device))
            for layer in LAYER_ORDER:
                batches[layer].append(values[layer].detach().cpu().numpy().astype(np.float32))

    features = {layer: np.concatenate(parts, axis=0) for layer, parts in batches.items()}
    meta = pd.DataFrame(
        {
            "row_index": indices,
            "target_label": dataset.target_labels[indices],
            "flanker_label": dataset.flanker_labels[indices],
            "response_label": dataset.response_labels[indices],
            "congruency": dataset.congruency[indices],
            "human_correct": dataset.response_labels[indices] == dataset.target_labels[indices],
            "true_rt": dataset.rts[indices],
            "user_id": dataset.data["user_id"].astype(str).to_numpy()[indices],
            "stimulus_image_path": dataset.image_paths[indices],
        }
    )
    split_metadata = {
        "csv_path": str(csv_path),
        "selected_rows": int(len(indices)),
        "source_rows": int(len(dataset)),
        "missing_image_count": int(dataset.missing_image_count),
        "missing_image_rate": float(dataset.missing_image_rate),
        "feature_shapes": {layer: list(value.shape) for layer, value in features.items()},
        **model_metadata,
    }
    return meta, features, split_metadata


class CentroidProbe:
    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CentroidProbe":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.classes_ = np.unique(y)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        xs = np.clip((x - self.mean_) / self.scale_, -10.0, 10.0)
        centroids = []
        for class_value in self.classes_:
            class_rows = xs[y == class_value]
            if len(class_rows) == 0:
                centroids.append(np.zeros(xs.shape[1], dtype=np.float32))
            else:
                centroids.append(class_rows.mean(axis=0).astype(np.float32))
        self.coef_ = np.stack(centroids, axis=0)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Probe is not fitted")
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs = np.clip((x - self.mean_) / self.scale_, -10.0, 10.0)
        scores = []
        for centroid in self.coef_:
            diff = xs - centroid
            scores.append(-np.sum(diff * diff, axis=1))
        return np.stack(scores, axis=1).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Probe is not fitted")
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]


def make_probe(seed: int) -> CentroidProbe:
    del seed
    return CentroidProbe()


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> Tuple[CentroidProbe, np.ndarray, np.ndarray, Dict[str, float]]:
    model = make_probe(seed)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    evidence = np.asarray(model.decision_function(test_x), dtype=np.float32)
    metrics = {
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
    }
    return model, pred.astype(np.int64), evidence.astype(np.float32), metrics


def trial_metrics(test_meta: pd.DataFrame, layer: str, task: str, pred: np.ndarray) -> pd.DataFrame:
    out = test_meta[
        [
            "age_group",
            "row_index",
            "target_label",
            "flanker_label",
            "response_label",
            "congruency",
            "human_correct",
            "true_rt",
            "user_id",
            "stimulus_image_path",
        ]
    ].copy()
    out["feature_layer"] = layer
    out["probe_task"] = task
    out["probe_true"] = out[TASKS[task]].to_numpy(dtype=np.int64)
    out["probe_pred"] = pred
    out["probe_correct"] = out["probe_true"] == out["probe_pred"]
    out["rt_bin"] = np.where(out["true_rt"] <= out["true_rt"].median(), "fast", "slow")
    return out


def safe_balanced_accuracy(group: pd.DataFrame) -> float:
    if group["probe_true"].nunique() <= 1:
        return float("nan")
    return float(balanced_accuracy_score(group["probe_true"], group["probe_pred"]))


def summarize_trials(trial_level: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groupings = [
        ["feature_layer", "probe_task"],
        ["feature_layer", "probe_task", "age_group"],
        ["feature_layer", "probe_task", "human_correct"],
        ["feature_layer", "probe_task", "rt_bin"],
        ["feature_layer", "probe_task", "congruency"],
        ["feature_layer", "probe_task", "age_group", "human_correct"],
    ]
    for keys in groupings:
        for name, group in trial_level.groupby(keys, dropna=False):
            if not isinstance(name, tuple):
                name = (name,)
            row = {key: value for key, value in zip(keys, name)}
            row.update(
                {
                    "group": " | ".join(f"{key}={value}" for key, value in zip(keys, name)),
                    "n_trials": int(len(group)),
                    "accuracy": float(group["probe_correct"].mean()),
                    "balanced_accuracy": safe_balanced_accuracy(group),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_gap_rows(summary: pd.DataFrame) -> pd.DataFrame:
    overall = summary[
        summary["group"].eq("overall")
        & summary["feature_layer"].notna()
        & summary["probe_task"].isin(["target", "flanker"])
    ]
    rows = []
    for layer in LAYER_ORDER:
        target = overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "target")]
        flanker = overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "flanker")]
        if target.empty or flanker.empty:
            continue
        rows.append(
            {
                "feature_layer": layer,
                "probe_task": "target_minus_flanker_gap",
                "group": "overall",
                "n_trials": int(target["n_trials"].iloc[0]),
                "accuracy": float(target["accuracy"].iloc[0] - flanker["accuracy"].iloc[0]),
                "balanced_accuracy": float(target["balanced_accuracy"].iloc[0] - flanker["balanced_accuracy"].iloc[0]),
            }
        )
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def plot_metric(summary: pd.DataFrame, task: str, output_path: Path, ylabel: str = "Probe accuracy") -> None:
    rows = summary[(summary["group"] == "overall") & (summary["probe_task"] == task)].copy()
    rows["feature_layer"] = pd.Categorical(rows["feature_layer"], categories=LAYER_ORDER, ordered=True)
    rows = rows.sort_values("feature_layer")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(rows["feature_layer"].astype(str), rows["accuracy"], color="#4C78A8")
    if task in CHANCE_LEVEL:
        ax.axhline(CHANCE_LEVEL[task], color="#333333", linewidth=1, linestyle="--")
        ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Representation")
    ax.set_title(task.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def probe_coefficients(model: CentroidProbe) -> np.ndarray:
    if model.coef_ is None:
        raise RuntimeError("Probe is not fitted")
    coef = np.asarray(model.coef_, dtype=np.float64)
    return coef


def geometry_summary(
    probes: Dict[Tuple[str, str], CentroidProbe],
    probe_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overall = probe_summary[probe_summary["group"] == "overall"]
    for layer in LAYER_ORDER:
        target_probe = probes[(layer, "target")]
        flanker_probe = probes[(layer, "flanker")]
        target_coef = probe_coefficients(target_probe)
        flanker_coef = probe_coefficients(flanker_probe)
        per_class = [
            cosine(target_coef[class_idx], flanker_coef[class_idx])
            for class_idx in range(min(target_coef.shape[0], flanker_coef.shape[0]))
        ]
        target_acc = float(
            overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "target")]["accuracy"].iloc[0]
        )
        flanker_acc = float(
            overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "flanker")]["accuracy"].iloc[0]
        )
        congruency_acc = float(
            overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "congruency")]["accuracy"].iloc[0]
        )
        rows.append(
            {
                "feature_layer": layer,
                "target_flanker_weight_cosine_flat": cosine(target_coef.ravel(), flanker_coef.ravel()),
                "target_flanker_weight_cosine_mean_class": float(np.nanmean(per_class)),
                "target_flanker_weight_cosine_min_class": float(np.nanmin(per_class)),
                "target_flanker_weight_cosine_max_class": float(np.nanmax(per_class)),
                "target_accuracy": target_acc,
                "flanker_accuracy": flanker_acc,
                "congruency_accuracy": congruency_acc,
                "target_flanker_accuracy_gap": target_acc - flanker_acc,
                "approximation": "cosine between standardized logistic-regression class weights",
            }
        )
    return pd.DataFrame(rows)


def plot_geometry(geometry: pd.DataFrame, figure_dir: Path) -> None:
    rows = geometry.copy()
    rows["feature_layer"] = pd.Categorical(rows["feature_layer"], categories=LAYER_ORDER, ordered=True)
    rows = rows.sort_values("feature_layer")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(rows["feature_layer"].astype(str), rows["target_flanker_weight_cosine_flat"], marker="o", color="#4C78A8")
    ax.axhline(0.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylabel("Target/flanker weight cosine")
    ax.set_xlabel("Representation")
    ax.set_title("Approximate Target-Flanker Alignment")
    fig.tight_layout()
    fig.savefig(figure_dir / "subspace_alignment_by_layer.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(rows["target_flanker_weight_cosine_flat"], rows["flanker_accuracy"], s=60, color="#F58518")
    for _, row in rows.iterrows():
        ax.annotate(str(row["feature_layer"]), (row["target_flanker_weight_cosine_flat"], row["flanker_accuracy"]))
    ax.set_xlabel("Target/flanker weight cosine")
    ax.set_ylabel("Flanker decoding accuracy")
    ax.set_title("Decodability vs Alignment")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_flanker_decodability_vs_alignment.png", dpi=220)
    plt.close(fig)


def plot_gap(summary: pd.DataFrame, output_path: Path) -> None:
    rows = summary[(summary["group"] == "overall") & (summary["probe_task"] == "target_minus_flanker_gap")].copy()
    rows["feature_layer"] = pd.Categorical(rows["feature_layer"], categories=LAYER_ORDER, ordered=True)
    rows = rows.sort_values("feature_layer")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(rows["feature_layer"].astype(str), rows["accuracy"], color="#54A24B")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_ylabel("Target accuracy - flanker accuracy")
    ax.set_xlabel("Representation")
    ax.set_title("Target-Flanker Decoding Gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_probe_memo(
    output_path: Path,
    summary: pd.DataFrame,
    metadata: Dict[str, Any],
    leakage: Dict[str, Any],
) -> None:
    overall = summary[summary["group"] == "overall"]

    def acc(layer: str, task: str) -> float:
        rows = overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == task)]
        return float(rows["accuracy"].iloc[0]) if not rows.empty else float("nan")

    flanker_best = max(LAYER_ORDER, key=lambda layer: acc(layer, "flanker"))
    target_best = max(LAYER_ORDER, key=lambda layer: acc(layer, "target"))
    memo = f"""# Layer-wise Feature Probe

## Files read

- `/Users/siyu/Downloads/layerwise_dmc_replacement_agent_prompt.md`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/cache_vgg_stage2_features.py`
- `code/scripts/stage1_evidence_sampler.py`
- `code/scripts/train_variational_ww_smoke.py`
- `code/scripts/train_dmc_var_ww_smoke.py`
- `code/scripts/run_subject_level_dmc_var_ww.py`
- `code/scripts/analyze_dynamic_selection_single_subject.py`
- `code/scripts/analyze_hidden_feature_flanker_probe.py`
- `code/scripts/analyze_no_dmc_logit_evidence_audit.py`
- `artifacts/results/diagnostics/hidden_feature_flanker_probe_full/hidden_feature_flanker_probe_full.md`
- `artifacts/results/diagnostics/hidden_feature_flanker_probe_full/hidden_feature_probe_summary_full.csv`
- `artifacts/results/diagnostics/hidden_feature_flanker_probe_full/hidden_feature_probe_trial_level_full.csv`
- `artifacts/results/diagnostics/hidden_feature_flanker_probe_full/metadata.json`
- `artifacts/results/diagnostics/dmc_replacement_and_noise_next_steps_report.md`
- `artifacts/results/diagnostics/stage1_auxiliary_loss_implementation_plan.md`
- `artifacts/results/diagnostics/ar1_evidence_noise_ablation_proposal.md`
- Jaffe et al. PDF paths were located under `docs/papers/` and `code/vam/`; this smoke run used the local project summaries rather than parsing the PDF.

## Commands run

```bash
python3 code/scripts/analyze_layerwise_feature_probe.py --data_root {metadata['data_root']} --age_groups {','.join(metadata['age_groups'])} --max_train_per_age {metadata['max_train_per_age']} --max_test_per_age {metadata['max_test_per_age']} --batch_size {metadata['batch_size']} --device {metadata['device']}
```

## Outputs

- `layerwise_probe_summary.csv`
- `layerwise_probe_trial_level.csv`
- `metadata.json`
- figures in `figures/`

## Layer mapping

- `conv3`: VGG16 features index 15, after block-3 final ReLU (`conv3_3`).
- `conv4`: VGG16 features index 22, after block-4 final ReLU (`conv4_3`).
- `conv5`: VGG16 features index 29, after block-5 final ReLU (`conv5_3`).
- `pooled`: adaptive average pooled final VGG feature.
- `final_logits`: current target-classifier output.

## Main numbers

| layer | target acc | flanker acc | congruency acc | target-flanker gap |
|---|---:|---:|---:|---:|
"""
    for layer in LAYER_ORDER:
        memo += f"| {layer} | {acc(layer, 'target'):.3f} | {acc(layer, 'flanker'):.3f} | {acc(layer, 'congruency'):.3f} | {acc(layer, 'target_minus_flanker_gap'):.3f} |\n"

    memo += f"""
## Leakage checks

- Split type: existing train/test split, not a random trial-level split.
- Train rows used: {metadata['n_train']}; test rows used: {metadata['n_test']}.
- Train/test image-path overlap count: {leakage['image_path_overlap_count']}.
- Train/test stimulus-id overlap count: {leakage['stimulus_id_overlap_count']}.
- Train/test user overlap count: {leakage['user_overlap_count']}.
- Subject-level split check: not completed in this smoke run.
- Image-identity split check: not completed; overlap above means image identity leakage remains possible.
- Full previous pooled/final result was read from the prior diagnostic report; this run is a smaller layer-localization smoke pass.

## Interpretation

- Strongest flanker retention in this run: `{flanker_best}`.
- Most target-oriented by target accuracy: `{target_best}`.
- Final logits suppress flanker relative to hidden features if `final_logits` flanker accuracy is lower than conv/pooled flanker accuracy.
- A middle-layer conflict-rich representation is supported when conv3/conv4/conv5 jointly decode both target and flanker well and have a small target-flanker gap.
- This is enough to proceed to a layer-wise evidence-to-Wong-Wang smoke test only as a diagnostic. It does not yet prove human-like fast errors or replace hand-crafted DMC.

## What this supports

The current model contains layer-wise visual information that can be converted into 4-choice evidence with lightweight readouts.

## What this does not support

It does not establish a clean subject-level or image-identity generalization result, and it does not test Wong-Wang behavior yet.

## Next step

Use `layerwise_evidence_cache/layerwise_evidence.npz` for a no-DMC single-layer Wong-Wang smoke comparison, then test a fixed layer-time gate only if middle-layer evidence is more conflict-rich than final logits.
"""
    output_path.write_text(memo, encoding="utf-8")


def write_geometry_memo(output_path: Path, geometry: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    best_align = geometry.iloc[geometry["target_flanker_weight_cosine_flat"].abs().argmax()]
    conflict_like = geometry.sort_values(["flanker_accuracy", "congruency_accuracy"], ascending=False).iloc[0]
    target_like = geometry.sort_values(["target_flanker_accuracy_gap", "target_accuracy"], ascending=False).iloc[0]
    memo = f"""# Layer-wise Representation Geometry

## Commands run

```bash
python3 code/scripts/analyze_layerwise_feature_probe.py --data_root {metadata['data_root']} --age_groups {','.join(metadata['age_groups'])} --max_train_per_age {metadata['max_train_per_age']} --max_test_per_age {metadata['max_test_per_age']} --batch_size {metadata['batch_size']} --device {metadata['device']}
```

## Outputs

- `layerwise_geometry_summary.csv`
- figures in `figures/`

## Method

This is a first-pass approximation: for each layer, separate logistic-regression probes were trained for target and flanker direction, then the cosine similarity between their standardized class-weight vectors was measured.

## Main result

- Largest absolute target/flanker alignment: `{best_align['feature_layer']}` ({best_align['target_flanker_weight_cosine_flat']:.3f}).
- Most conflict-like by flanker and congruency decodability: `{conflict_like['feature_layer']}`.
- Most late target-selection-like by target-flanker gap: `{target_like['feature_layer']}`.

## Interpretation

If middle layers show both high flanker decodability and stronger target/flanker alignment than final logits, they are plausible early conflict evidence sources. If final logits show a larger target-flanker gap, they are a plausible late target-oriented source.

## What this supports

It supports treating layer-wise evidence as a candidate natural DMC replacement only if the evidence cache confirms that middle layers expose stronger flanker/conflict evidence than the final head.

## What this does not support

This geometry analysis is not a full subspace or image-identity generalization analysis. It should not be interpreted as proof of orthogonalization without a stronger split and a fuller subspace method.
"""
    output_path.write_text(memo, encoding="utf-8")


def write_evidence_memo(output_path: Path, summary: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    overall = summary[summary["group"] == "overall"]
    rows = []
    for layer in LAYER_ORDER:
        target = overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "target")]["accuracy"].iloc[0]
        flanker = overall[(overall["feature_layer"] == layer) & (overall["probe_task"] == "flanker")]["accuracy"].iloc[0]
        rows.append((layer, float(target), float(flanker), float(target - flanker)))
    target_dom = max(rows, key=lambda x: x[3])
    conflict_rich = min(rows, key=lambda x: abs(x[3]))
    memo = f"""# Layer-wise Evidence Cache

## Output

- `layerwise_evidence.npz`
- `metadata.json`

## Minimal implementation

Each conv/pooled representation was converted into 4-class direction evidence using the target-direction logistic probe trained in the feature-probe step. `final_logits` uses the model's own 4-class target logits. This avoids long training and creates a small cache for the next Wong-Wang smoke test.

## Evidence arrays

The cache contains `evidence_conv3`, `evidence_conv4`, `evidence_conv5`, `evidence_pooled`, and `evidence_final`, plus labels, RT, row indices, age group, user id, and stimulus image path.

## Interpretation

- Most target-dominant source by target-minus-flanker probe gap: `{target_dom[0]}`.
- Most conflict-rich source by smallest target/flanker gap: `{conflict_rich[0]}`.
- This cache is suitable for a single-layer Wong-Wang diagnostic, but not yet for a final claim because the probe split still permits image-identity overlap.

## Commands run

```bash
python3 code/scripts/analyze_layerwise_feature_probe.py --data_root {metadata['data_root']} --age_groups {','.join(metadata['age_groups'])} --max_train_per_age {metadata['max_train_per_age']} --max_test_per_age {metadata['max_test_per_age']} --batch_size {metadata['batch_size']} --device {metadata['device']}
```
"""
    output_path.write_text(memo, encoding="utf-8")


def save_evidence_cache(
    output_dir: Path,
    test_meta: pd.DataFrame,
    evidence_by_layer: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    output = {
        "target_labels": test_meta["target_label"].to_numpy(dtype=np.int64),
        "flanker_labels": test_meta["flanker_label"].to_numpy(dtype=np.int64),
        "response_labels": test_meta["response_label"].to_numpy(dtype=np.int64),
        "true_rt": test_meta["true_rt"].to_numpy(dtype=np.float32),
        "congruency": test_meta["congruency"].to_numpy(dtype=np.int64),
        "row_indices": test_meta["row_index"].to_numpy(dtype=np.int64),
        "age_group": test_meta["age_group"].astype(str).to_numpy(),
        "user_id": test_meta["user_id"].astype(str).to_numpy(),
        "stimulus_image_path": test_meta["stimulus_image_path"].astype(str).to_numpy(),
    }
    for layer in LAYER_ORDER:
        key = "evidence_final" if layer == "final_logits" else f"evidence_{layer}"
        output[key] = evidence_by_layer[layer].astype(np.float32)
    np.savez_compressed(output_dir / "layerwise_evidence.npz", **output)
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/age_groups_matched")
    parser.add_argument("--output_root", default="artifacts/results/diagnostics")
    parser.add_argument("--age_groups", default="20-29")
    parser.add_argument("--max_train_per_age", type=int, default=3000)
    parser.add_argument("--max_test_per_age", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260523)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    probe_dir = output_root / "layerwise_feature_probe"
    geometry_dir = output_root / "layerwise_representation_geometry"
    evidence_dir = output_root / "layerwise_evidence_cache"
    for directory in [probe_dir / "figures", geometry_dir / "figures", evidence_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    age_groups = [part.strip() for part in str(args.age_groups).split(",") if part.strip()]
    train_meta_parts: List[pd.DataFrame] = []
    test_meta_parts: List[pd.DataFrame] = []
    train_features: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
    test_features: Dict[str, List[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
    split_metadata: Dict[str, Any] = {}

    for age_idx, age_group in enumerate(age_groups):
        for split, max_rows, target_meta, target_features in [
            ("train", int(args.max_train_per_age), train_meta_parts, train_features),
            ("test", int(args.max_test_per_age), test_meta_parts, test_features),
        ]:
            csv_path = data_root / age_group / f"{split}_data.csv"
            dataset = StimulusDataset(str(csv_path))
            indices = balanced_indices(dataset, max_rows=max_rows, seed=int(args.seed) + age_idx + (100 if split == "test" else 0))
            meta, features, split_info = extract_layerwise_features(
                csv_path=csv_path,
                indices=indices,
                batch_size=int(args.batch_size),
                device=str(args.device),
            )
            meta["age_group"] = age_group
            target_meta.append(meta)
            for layer in LAYER_ORDER:
                target_features[layer].append(features[layer])
            split_metadata[f"{age_group}_{split}"] = split_info

    train_meta = pd.concat(train_meta_parts, ignore_index=True)
    test_meta = pd.concat(test_meta_parts, ignore_index=True)
    train_x = {layer: np.concatenate(parts, axis=0) for layer, parts in train_features.items()}
    test_x = {layer: np.concatenate(parts, axis=0) for layer, parts in test_features.items()}

    probes: Dict[Tuple[str, str], CentroidProbe] = {}
    summary_rows: List[Dict[str, Any]] = []
    trial_frames: List[pd.DataFrame] = []
    evidence_by_layer: Dict[str, np.ndarray] = {}

    for layer in LAYER_ORDER:
        for task, label_col in TASKS.items():
            model, pred, evidence, metrics = fit_probe(
                train_x=train_x[layer],
                train_y=train_meta[label_col].to_numpy(dtype=np.int64),
                test_x=test_x[layer],
                test_y=test_meta[label_col].to_numpy(dtype=np.int64),
                seed=int(args.seed),
            )
            probes[(layer, task)] = model
            if task == "target":
                evidence_by_layer[layer] = test_x[layer] if layer == "final_logits" else evidence
            summary_rows.append(
                {
                    "feature_layer": layer,
                    "probe_task": task,
                    "group": "overall",
                    "n_train": int(len(train_meta)),
                    "n_trials": int(len(test_meta)),
                    **metrics,
                }
            )
            trial_frames.append(trial_metrics(test_meta, layer, task, pred))

    trial_level = pd.concat(trial_frames, ignore_index=True)
    summary = pd.concat([pd.DataFrame(summary_rows), summarize_trials(trial_level)], ignore_index=True, sort=False)
    summary = add_gap_rows(summary)

    summary.to_csv(probe_dir / "layerwise_probe_summary.csv", index=False)
    trial_level.to_csv(probe_dir / "layerwise_probe_trial_level.csv", index=False)
    plot_metric(summary, "target", probe_dir / "figures" / "target_decoding_by_layer.png")
    plot_metric(summary, "flanker", probe_dir / "figures" / "flanker_decoding_by_layer.png")
    plot_metric(summary, "congruency", probe_dir / "figures" / "congruency_decoding_by_layer.png")
    plot_gap(summary, probe_dir / "figures" / "target_flanker_gap_by_layer.png")

    geometry = geometry_summary(probes, summary)
    geometry.to_csv(geometry_dir / "layerwise_geometry_summary.csv", index=False)
    plot_geometry(geometry, geometry_dir / "figures")

    leakage = {
        "image_path_overlap_count": int(
            len(set(train_meta["stimulus_image_path"].astype(str)) & set(test_meta["stimulus_image_path"].astype(str)))
        ),
        "stimulus_id_overlap_count": int(
            len(
                set(Path(path).stem for path in train_meta["stimulus_image_path"].astype(str))
                & set(Path(path).stem for path in test_meta["stimulus_image_path"].astype(str))
            )
        ),
        "user_overlap_count": int(len(set(train_meta["user_id"].astype(str)) & set(test_meta["user_id"].astype(str)))),
    }
    metadata = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "age_groups": age_groups,
        "max_train_per_age": int(args.max_train_per_age),
        "max_test_per_age": int(args.max_test_per_age),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "seed": int(args.seed),
        "n_train": int(len(train_meta)),
        "n_test": int(len(test_meta)),
        "layer_order": LAYER_ORDER,
        "layer_taps": LAYER_TAPS,
        "direction_map": DIRECTION_MAP,
        "split_metadata": split_metadata,
        "leakage_checks": leakage,
    }
    (probe_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    write_probe_memo(probe_dir / "layerwise_probe_memo.md", summary, metadata, leakage)
    write_geometry_memo(geometry_dir / "layerwise_geometry_memo.md", geometry, metadata)
    save_evidence_cache(evidence_dir, test_meta, evidence_by_layer, metadata)
    write_evidence_memo(evidence_dir / "layerwise_evidence_cache_memo.md", summary, metadata)


if __name__ == "__main__":
    main()

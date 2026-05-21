from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cache_vgg_stage2_features import load_stage1_model
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import StimulusDataset, to_jsonable

_STAGE1_MODULE = importlib.import_module("stage1_evidence_sampler")
Stage1EvidenceConfig = _STAGE1_MODULE.Stage1EvidenceConfig
SemiSupervisedEvidenceSampler = _STAGE1_MODULE.SemiSupervisedEvidenceSampler

_SPEA_BACKEND_MODULE = importlib.import_module("stage2_spea_backend")
SemiSupervisedSPEAConfig = _SPEA_BACKEND_MODULE.SemiSupervisedSPEAConfig
fit_spea_from_stage1_inputs = _SPEA_BACKEND_MODULE.fit_spea_from_stage1_inputs


VARIANTS = {
    "v0_deterministic",
    "v1_stage2_noise_only",
    "v2_supervised_variational_stage1",
    "v3_semisup_variational_stage1",
    "v4_semisup_stage1_plus_stage2_noise",
    "v5_mc_dropout_stage1",
    "stage1_semisup_variational_warmstart",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SemiSupervisedSPEA-v1 variants from raw images.")
    parser.add_argument("--age_group", required=True, choices=["20-29", "80-89"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_accumulator", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=120)
    parser.add_argument("--dt_ms", type=int, default=10)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_ssl", type=float, default=0.5)
    parser.add_argument("--lambda_uncertainty_bound", type=float, default=0.05)
    parser.add_argument("--lambda_behavior", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.10)
    parser.add_argument("--stage2_noise", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--stage1_train_mode", choices=["frozen_backbone"], default="frozen_backbone")
    parser.add_argument("--smoke_eval", action="store_true")
    parser.add_argument("--smoke_eval_mode", choices=["behavior_balanced"], default="behavior_balanced")
    parser.add_argument("--smoke_max_trials", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {".", ".."}:
        return (Path.cwd() / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(csv_path: Path, *, batch_size: int) -> Tuple[StimulusDataset, DataLoader]:
    dataset = StimulusDataset(str(csv_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader


def _build_behavior_balanced_subset(dataset: StimulusDataset, *, max_trials: int, seed: int) -> List[int]:
    total_rows = len(dataset)
    if total_rows <= max_trials:
        return list(range(total_rows))
    rng = np.random.default_rng(seed)
    response_labels = np.asarray(dataset.response_labels)
    target_labels = np.asarray(dataset.target_labels)
    congruency = np.asarray(dataset.congruency)
    error_indices = np.flatnonzero(response_labels != target_labels)
    congruent_indices = np.flatnonzero(congruency == 0)
    incongruent_indices = np.flatnonzero(congruency == 1)
    selected: List[int] = []
    selected_set = set()

    def add_candidates(candidates: np.ndarray, limit: int) -> None:
        if limit <= 0 or len(candidates) == 0 or len(selected) >= max_trials:
            return
        shuffled = np.array(candidates, copy=True)
        rng.shuffle(shuffled)
        added = 0
        for idx in shuffled:
            idx_i = int(idx)
            if idx_i in selected_set:
                continue
            selected.append(idx_i)
            selected_set.add(idx_i)
            added += 1
            if len(selected) >= max_trials or added >= limit:
                break

    add_candidates(error_indices, min(len(error_indices), max(64, max_trials // 10)))
    add_candidates(congruent_indices, max_trials // 2)
    add_candidates(incongruent_indices, max_trials // 2)
    if len(selected) < max_trials:
        remaining = np.setdiff1d(np.arange(total_rows, dtype=np.int64), np.asarray(selected, dtype=np.int64), assume_unique=False)
        add_candidates(remaining, max_trials - len(selected))
    return sorted(selected[:max_trials])


def _maybe_smoke_subset(dataset: StimulusDataset, *, seed: int, max_trials: int | None) -> Subset | StimulusDataset:
    if max_trials is None:
        return dataset
    indices = _build_behavior_balanced_subset(dataset, max_trials=max_trials, seed=seed)
    return Subset(dataset, indices)


def _stage1_mode(variant: str) -> str:
    if variant in {"v0_deterministic", "v1_stage2_noise_only"}:
        return "deterministic"
    if variant in {"v2_supervised_variational_stage1", "v3_semisup_variational_stage1", "v4_semisup_stage1_plus_stage2_noise", "stage1_semisup_variational_warmstart"}:
        return "variational"
    if variant == "v5_mc_dropout_stage1":
        return "mc_dropout"
    raise ValueError(f"Unknown variant: {variant}")


def _stage2_noise_scale(variant: str, cli_setting: str) -> float:
    enabled = variant in {"v1_stage2_noise_only", "v4_semisup_stage1_plus_stage2_noise"} or cli_setting == "enabled"
    return 0.10 if enabled else 0.0


def _uncertainty_bound_loss(sigma: torch.Tensor) -> torch.Tensor:
    sigma_mean = sigma.mean()
    lower = F.relu(0.05 - sigma_mean)
    upper = F.relu(sigma_mean - 1.50)
    return lower + upper


def _consistency_loss(left_mu: torch.Tensor, right_mu: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(left_mu, right_mu)


def _teacher_alignment_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_probs,
        reduction="batchmean",
    )


def _augment_batch(images: torch.Tensor, *, noise_scale: float) -> torch.Tensor:
    noise = torch.randn_like(images) * noise_scale
    return (images + noise).clamp(min=-3.0, max=3.0)


def train_stage1_head(
    *,
    sampler: SemiSupervisedEvidenceSampler,
    dataset_loader: DataLoader,
    sampler_mode: str,
    epochs: int,
    lambda_cls: float,
    lambda_ssl: float,
    lambda_teacher: float,
    lambda_uncertainty_bound: float,
    device: str,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    heartbeat_path: Path | None = None,
) -> Dict[str, float]:
    if sampler_mode == "deterministic" or epochs <= 0:
        return {
            "stage1_train_ce_loss": float("nan"),
            "stage1_train_accuracy": float("nan"),
            "stage1_train_teacher_loss": float("nan"),
            "stage1_train_uncertainty_bound_loss": float("nan"),
        }

    sampler.stage1_backbone.eval()
    for param in sampler.stage1_backbone.parameters():
        param.requires_grad = False

    trainable_params: List[torch.nn.Parameter]
    if sampler_mode == "variational":
        trainable_module = sampler.variational_head
        trainable_params = list(sampler.variational_head.parameters())
    elif sampler_mode == "mc_dropout":
        trainable_module = sampler.mc_dropout_head
        trainable_params = list(sampler.mc_dropout_head.parameters())
    else:
        raise ValueError(f"Unknown sampler_mode: {sampler_mode}")

    optimizer = Adam(trainable_params, lr=1e-3)
    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "stage1_checkpoint.pt"

    start_epoch = 0
    last_checkpoint_metrics: Dict[str, float] | None = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        module_state = checkpoint.get("module_state")
        optimizer_state = checkpoint.get("optimizer_state")
        if module_state is not None:
            trainable_module.load_state_dict(module_state)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        saved_metrics = checkpoint.get("metrics")
        if isinstance(saved_metrics, dict):
            last_checkpoint_metrics = {str(k): float(v) for k, v in saved_metrics.items() if isinstance(v, (float, int, np.floating))}

    if heartbeat_path is not None:
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(
            json.dumps(
                to_jsonable(
                    {
                        "phase": "stage1_training",
                        "status": "running",
                        "sampler_mode": sampler_mode,
                        "epochs_requested": int(epochs),
                        "start_epoch": int(start_epoch),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    if start_epoch >= epochs:
        return last_checkpoint_metrics or {
            "stage1_train_ce_loss": float("nan"),
            "stage1_train_accuracy": float("nan"),
            "stage1_train_teacher_loss": float("nan"),
            "stage1_train_uncertainty_bound_loss": float("nan"),
        }

    last_epoch_cls_losses: List[float] = []
    last_epoch_teacher_losses: List[float] = []
    last_epoch_bound_losses: List[float] = []
    last_epoch_accuracies: List[float] = []
    for epoch_idx in range(start_epoch, epochs):
        last_epoch_cls_losses = []
        last_epoch_teacher_losses = []
        last_epoch_bound_losses = []
        last_epoch_accuracies = []
        for batch in dataset_loader:
            images = batch["image"].to(device)
            target_labels = batch["target_label"].to(device)
            weak_images = _augment_batch(images, noise_scale=0.01)
            strong_images = _augment_batch(images, noise_scale=0.05)
            pred_logits: torch.Tensor
            with torch.no_grad():
                weak_features, weak_logits = sampler.encode_images(weak_images)
                strong_features, _ = sampler.encode_images(strong_images)
            if sampler_mode == "variational":
                weak_mu, weak_sigma = sampler.variational_head(weak_features)
                strong_mu, _ = sampler.variational_head(strong_features)
                cls_loss = F.cross_entropy(weak_mu, target_labels)
                ssl_loss = _consistency_loss(weak_mu, strong_mu)
                teacher_loss = _teacher_alignment_loss(weak_mu, weak_logits)
                bound_loss = _uncertainty_bound_loss(weak_sigma)
                loss = (
                    lambda_cls * cls_loss
                    + lambda_ssl * ssl_loss
                    + lambda_teacher * teacher_loss
                    + lambda_uncertainty_bound * bound_loss
                )
                teacher_loss_value = float(teacher_loss.detach().cpu().item())
                pred_logits = weak_mu
            else:
                weak_pred = sampler.mc_dropout_head(weak_features)
                strong_pred = sampler.mc_dropout_head(strong_features)
                cls_loss = F.cross_entropy(weak_pred, target_labels)
                ssl_loss = _consistency_loss(weak_pred, strong_pred)
                bound_loss = torch.zeros((), device=device)
                loss = lambda_cls * cls_loss + lambda_ssl * ssl_loss
                teacher_loss_value = 0.0
                pred_logits = weak_pred
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_accuracy = float((pred_logits.argmax(dim=1) == target_labels).float().mean().detach().cpu().item())
            last_epoch_cls_losses.append(float(cls_loss.detach().cpu().item()))
            last_epoch_teacher_losses.append(teacher_loss_value)
            last_epoch_bound_losses.append(float(bound_loss.detach().cpu().item()))
            last_epoch_accuracies.append(batch_accuracy)

        epoch_metrics = {
            "stage1_train_ce_loss": float(np.mean(last_epoch_cls_losses)) if last_epoch_cls_losses else float("nan"),
            "stage1_train_accuracy": float(np.mean(last_epoch_accuracies)) if last_epoch_accuracies else float("nan"),
            "stage1_train_teacher_loss": float(np.mean(last_epoch_teacher_losses)) if last_epoch_teacher_losses else float("nan"),
            "stage1_train_uncertainty_bound_loss": float(np.mean(last_epoch_bound_losses)) if last_epoch_bound_losses else float("nan"),
        }
        if checkpoint_path is not None:
            metrics_payload = {k: float(v) for k, v in epoch_metrics.items()}
            torch.save(
                {
                    "epoch": int(epoch_idx),
                    "module_state": trainable_module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics_payload,
                },
                checkpoint_path,
            )
        if heartbeat_path is not None:
            heartbeat_path.write_text(
                json.dumps(
                    to_jsonable(
                        {
                            "phase": "stage1_training",
                            "status": "running",
                            "sampler_mode": sampler_mode,
                            "epoch": int(epoch_idx + 1),
                            "epochs_requested": int(epochs),
                            "metrics": epoch_metrics,
                        }
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

    return {
        "stage1_train_ce_loss": float(np.mean(last_epoch_cls_losses)) if last_epoch_cls_losses else float("nan"),
        "stage1_train_accuracy": float(np.mean(last_epoch_accuracies)) if last_epoch_accuracies else float("nan"),
        "stage1_train_teacher_loss": float(np.mean(last_epoch_teacher_losses)) if last_epoch_teacher_losses else float("nan"),
        "stage1_train_uncertainty_bound_loss": float(np.mean(last_epoch_bound_losses)) if last_epoch_bound_losses else float("nan"),
    }


def train_stage1_head_from_cached_features(
    *,
    sampler: SemiSupervisedEvidenceSampler,
    pooled_features: np.ndarray,
    base_logits: np.ndarray,
    target_labels: np.ndarray,
    sampler_mode: str,
    epochs: int,
    lambda_cls: float,
    lambda_teacher: float,
    lambda_uncertainty_bound: float,
    device: str,
    batch_size: int = 128,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    heartbeat_path: Path | None = None,
) -> Dict[str, float]:
    if sampler_mode == "deterministic" or epochs <= 0:
        return {
            "stage1_train_ce_loss": float("nan"),
            "stage1_train_accuracy": float("nan"),
            "stage1_train_teacher_loss": float("nan"),
            "stage1_train_uncertainty_bound_loss": float("nan"),
        }

    sampler.stage1_backbone.eval()
    for param in sampler.stage1_backbone.parameters():
        param.requires_grad = False

    if sampler_mode == "variational":
        trainable_module = sampler.variational_head
        trainable_params = list(sampler.variational_head.parameters())
    elif sampler_mode == "mc_dropout":
        trainable_module = sampler.mc_dropout_head
        trainable_params = list(sampler.mc_dropout_head.parameters())
    else:
        raise ValueError(f"Unknown sampler_mode: {sampler_mode}")

    optimizer = Adam(trainable_params, lr=1e-3)
    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "stage1_checkpoint.pt"

    start_epoch = 0
    last_checkpoint_metrics: Dict[str, float] | None = None
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        module_state = checkpoint.get("module_state")
        optimizer_state = checkpoint.get("optimizer_state")
        if module_state is not None:
            trainable_module.load_state_dict(module_state)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        saved_metrics = checkpoint.get("metrics")
        if isinstance(saved_metrics, dict):
            last_checkpoint_metrics = {str(k): float(v) for k, v in saved_metrics.items() if isinstance(v, (float, int, np.floating))}

    if heartbeat_path is not None:
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(
            json.dumps(
                to_jsonable(
                    {
                        "phase": "stage1_training",
                        "status": "running",
                        "source": "cached_features",
                        "sampler_mode": sampler_mode,
                        "epochs_requested": int(epochs),
                        "start_epoch": int(start_epoch),
                        "n_rows": int(len(target_labels)),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    if start_epoch >= epochs:
        return last_checkpoint_metrics or {
            "stage1_train_ce_loss": float("nan"),
            "stage1_train_accuracy": float("nan"),
            "stage1_train_teacher_loss": float("nan"),
            "stage1_train_uncertainty_bound_loss": float("nan"),
        }

    features_t = torch.tensor(np.asarray(pooled_features, dtype=np.float32), dtype=torch.float32, device=device)
    logits_t = torch.tensor(np.asarray(base_logits, dtype=np.float32), dtype=torch.float32, device=device)
    target_t = torch.tensor(np.asarray(target_labels, dtype=np.int64), dtype=torch.long, device=device)
    n_rows = int(target_t.shape[0])

    last_epoch_cls_losses: List[float] = []
    last_epoch_teacher_losses: List[float] = []
    last_epoch_bound_losses: List[float] = []
    last_epoch_accuracies: List[float] = []
    for epoch_idx in range(start_epoch, epochs):
        last_epoch_cls_losses = []
        last_epoch_teacher_losses = []
        last_epoch_bound_losses = []
        last_epoch_accuracies = []
        order = torch.randperm(n_rows, device=device)
        for start in range(0, n_rows, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            weak_features = features_t[idx]
            weak_logits = logits_t[idx]
            target_batch = target_t[idx]
            if sampler_mode == "variational":
                weak_mu, weak_sigma = sampler.variational_head(weak_features)
                cls_loss = F.cross_entropy(weak_mu, target_batch)
                teacher_loss = _teacher_alignment_loss(weak_mu, weak_logits)
                bound_loss = _uncertainty_bound_loss(weak_sigma)
                loss = lambda_cls * cls_loss + lambda_teacher * teacher_loss + lambda_uncertainty_bound * bound_loss
                teacher_loss_value = float(teacher_loss.detach().cpu().item())
                pred_logits = weak_mu
            else:
                weak_pred = sampler.mc_dropout_head(weak_features)
                cls_loss = F.cross_entropy(weak_pred, target_batch)
                bound_loss = torch.zeros((), device=device)
                loss = lambda_cls * cls_loss
                teacher_loss_value = 0.0
                pred_logits = weak_pred
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_accuracy = float((pred_logits.argmax(dim=1) == target_batch).float().mean().detach().cpu().item())
            last_epoch_cls_losses.append(float(cls_loss.detach().cpu().item()))
            last_epoch_teacher_losses.append(teacher_loss_value)
            last_epoch_bound_losses.append(float(bound_loss.detach().cpu().item()))
            last_epoch_accuracies.append(batch_accuracy)

        epoch_metrics = {
            "stage1_train_ce_loss": float(np.mean(last_epoch_cls_losses)) if last_epoch_cls_losses else float("nan"),
            "stage1_train_accuracy": float(np.mean(last_epoch_accuracies)) if last_epoch_accuracies else float("nan"),
            "stage1_train_teacher_loss": float(np.mean(last_epoch_teacher_losses)) if last_epoch_teacher_losses else float("nan"),
            "stage1_train_uncertainty_bound_loss": float(np.mean(last_epoch_bound_losses)) if last_epoch_bound_losses else float("nan"),
        }
        if checkpoint_path is not None:
            torch.save(
                {
                    "epoch": int(epoch_idx),
                    "module_state": trainable_module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": {k: float(v) for k, v in epoch_metrics.items()},
                },
                checkpoint_path,
            )
        if heartbeat_path is not None:
            heartbeat_path.write_text(
                json.dumps(
                    to_jsonable(
                        {
                            "phase": "stage1_training",
                            "status": "running",
                            "source": "cached_features",
                            "sampler_mode": sampler_mode,
                            "epoch": int(epoch_idx + 1),
                            "epochs_requested": int(epochs),
                            "n_rows": int(n_rows),
                            "metrics": epoch_metrics,
                        }
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

    return {
        "stage1_train_ce_loss": float(np.mean(last_epoch_cls_losses)) if last_epoch_cls_losses else float("nan"),
        "stage1_train_accuracy": float(np.mean(last_epoch_accuracies)) if last_epoch_accuracies else float("nan"),
        "stage1_train_teacher_loss": float(np.mean(last_epoch_teacher_losses)) if last_epoch_teacher_losses else float("nan"),
        "stage1_train_uncertainty_bound_loss": float(np.mean(last_epoch_bound_losses)) if last_epoch_bound_losses else float("nan"),
    }


def evaluate_stage1_head(
    *,
    sampler: SemiSupervisedEvidenceSampler,
    dataset_loader: DataLoader,
    sampler_mode: str,
    device: str,
) -> Dict[str, float]:
    if sampler_mode == "deterministic":
        return {
            "stage1_eval_ce_loss": float("nan"),
            "stage1_eval_accuracy": float("nan"),
        }

    cls_losses: List[float] = []
    accuracies: List[float] = []
    sampler.eval()
    sampler.stage1_backbone.eval()
    with torch.no_grad():
        for batch in dataset_loader:
            images = batch["image"].to(device)
            target_labels = batch["target_label"].to(device)
            pooled_features, _ = sampler.encode_images(images)
            logits = sampler.variational_head(pooled_features)[0] if sampler_mode == "variational" else sampler.mc_dropout_head(pooled_features)
            cls_loss = F.cross_entropy(logits, target_labels)
            cls_losses.append(float(cls_loss.detach().cpu().item()))
            accuracies.append(float((logits.argmax(dim=1) == target_labels).float().mean().detach().cpu().item()))
    return {
        "stage1_eval_ce_loss": float(np.mean(cls_losses)) if cls_losses else float("nan"),
        "stage1_eval_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
    }

def build_stage1_bundle(
    *,
    sampler: SemiSupervisedEvidenceSampler,
    dataset_loader: DataLoader,
    sampler_mode: str,
    time_steps: int,
    device: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    rows: Dict[str, List[np.ndarray]] = {
        "evidence_samples": [],
        "evidence_sample_mu": [],
        "evidence_sample_sigma": [],
        "stage1_uncertainty_scalar": [],
        "entropy_mean": [],
        "entropy_var": [],
        "margin_mean": [],
        "margin_var": [],
        "target_labels": [],
        "response_labels": [],
        "rts": [],
        "congruency": [],
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampler.eval()
    with torch.no_grad():
        for batch in dataset_loader:
            images = batch["image"].to(device)
            payload = sampler.sample_from_images(
                images=images,
                time_steps=time_steps,
                sampler_mode=sampler_mode,
                generator=generator,
            )
            for key in (
                "evidence_samples",
                "evidence_sample_mu",
                "evidence_sample_sigma",
                "stage1_uncertainty_scalar",
                "entropy_mean",
                "entropy_var",
                "margin_mean",
                "margin_var",
            ):
                rows[key].append(payload[key].detach().cpu().numpy())
            rows["target_labels"].append(batch["target_label"].cpu().numpy())
            rows["response_labels"].append(batch["response_label"].cpu().numpy())
            rows["rts"].append(batch["rt"].cpu().numpy())
            rows["congruency"].append(batch["congruency"].cpu().numpy())

    bundled: Dict[str, np.ndarray] = {}
    for key, value in rows.items():
        normalized = []
        for item in value:
            arr = np.asarray(item)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            normalized.append(arr)
        bundled[key] = np.concatenate(normalized, axis=0)
    return bundled


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    data_root = _resolve_path(args.data_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = data_root / args.age_group
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"

    train_dataset_full, _ = _loader(train_csv, batch_size=int(args.batch_size))
    test_dataset_full, _ = _loader(test_csv, batch_size=int(args.batch_size))
    if args.smoke_eval:
        train_dataset = _maybe_smoke_subset(train_dataset_full, seed=int(args.seed), max_trials=int(args.smoke_max_trials))
        test_dataset = _maybe_smoke_subset(test_dataset_full, seed=int(args.seed) + 1, max_trials=int(args.smoke_max_trials))
    else:
        train_dataset = train_dataset_full
        test_dataset = test_dataset_full
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    stage1_backbone = load_stage1_model(str(args.device))
    stage1_cfg = Stage1EvidenceConfig(n_classes=4, feature_dim=512, hidden_dim=128, dropout_rate=float(args.dropout_rate))
    sampler = SemiSupervisedEvidenceSampler(stage1_cfg, stage1_backbone=stage1_backbone).to(args.device)

    sampler_mode = _stage1_mode(args.variant)
    train_stage1_head(
        sampler=sampler,
        dataset_loader=train_loader,
        sampler_mode=sampler_mode,
        epochs=int(args.epochs_stage1),
        lambda_cls=float(args.lambda_cls),
        lambda_ssl=float(args.lambda_ssl if args.variant in {"v3_semisup_variational_stage1", "v4_semisup_stage1_plus_stage2_noise", "v5_mc_dropout_stage1", "stage1_semisup_variational_warmstart"} else 0.0),
        lambda_teacher=0.0,
        lambda_uncertainty_bound=float(args.lambda_uncertainty_bound),
        device=str(args.device),
    )

    train_bundle = build_stage1_bundle(
        sampler=sampler,
        dataset_loader=train_loader,
        sampler_mode=sampler_mode,
        time_steps=int(args.time_steps),
        device=str(args.device),
        seed=int(args.seed),
    )
    test_bundle = build_stage1_bundle(
        sampler=sampler,
        dataset_loader=test_loader,
        sampler_mode=sampler_mode,
        time_steps=int(args.time_steps),
        device=str(args.device),
        seed=int(args.seed) + 1,
    )

    stage1_manifest = {
        "variant": args.variant,
        "sampler_mode": sampler_mode,
        "stage1_train_mode": args.stage1_train_mode,
        "epochs_stage1": int(args.epochs_stage1),
        "n_train": int(len(train_dataset)),
        "n_test": int(len(test_dataset)),
        "uses_raw_image_forward": True,
        "smoke_eval": bool(args.smoke_eval),
    }
    train_manifest = {
        "age_group": args.age_group,
        "csv_path": str(train_csv),
        "n_rows": int(len(train_dataset)),
        "label_keys_used": ["target_label", "response_label", "rt", "congruency"],
        "response_labels_used": True,
        "rts_used": True,
        "seed": int(args.seed),
    }
    test_manifest = {
        "age_group": args.age_group,
        "csv_path": str(test_csv),
        "n_rows": int(len(test_dataset)),
        "label_keys_used": ["target_label", "response_label", "rt", "congruency"],
        "response_labels_used": True,
        "rts_used": True,
        "seed": int(args.seed) + 1,
    }
    _write_json(output_root / "stage1_data_manifest.json", stage1_manifest)
    _write_json(output_root / "train_inputs_manifest.json", train_manifest)
    _write_json(output_root / "test_inputs_manifest.json", test_manifest)

    if args.variant == "stage1_semisup_variational_warmstart":
        metrics = {
            "variant": args.variant,
            "stage1_uncertainty_mean": float(train_bundle["stage1_uncertainty_scalar"].mean()),
            "stage1_uncertainty_std": float(train_bundle["stage1_uncertainty_scalar"].std()),
        }
        _write_json(output_root / "metrics_smoke.json", metrics)
        return

    spea_cfg = SemiSupervisedSPEAConfig(
        n_classes=4,
        dt_ms=int(args.dt_ms),
        time_steps=int(args.time_steps),
        hidden_dim=64,
        evidence_dim=4,
        choice_temperature=float(args.choice_temperature),
        lambda_behavior_choice=float(args.lambda_behavior),
        lambda_behavior_rt=2.0,
        lambda_quantile=0.25,
        lambda_center=1.0,
        lambda_error_rate=0.75,
        lambda_error_sign=0.75,
        lambda_accuracy=0.75,
        lambda_rate_reg=0.05,
        stage2_noise_scale=_stage2_noise_scale(args.variant, args.stage2_noise),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs_accumulator),
        eval_seed=int(args.seed) + 11,
    )
    fit_spea_from_stage1_inputs(
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        config=spea_cfg,
        output_dir=str(output_root),
        device=str(args.device),
        random_seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

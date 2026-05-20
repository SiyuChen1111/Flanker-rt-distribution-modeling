from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_age_groups_efficient import compute_human_stats_from_rts, evaluate_joint_behavior, to_jsonable


@dataclass
class SemiSupervisedSPEAConfig:
    n_classes: int = 4
    dt_ms: int = 10
    time_steps: int = 120
    hidden_dim: int = 64
    evidence_dim: int = 4
    choice_temperature: float = 0.10
    choice_readout_mode: str = "weighted_evidence"
    temperature_calibration: bool = False
    min_steps: int = 10
    hard_min_steps: bool = True
    beta: float = 12.0
    threshold_min: float = 0.05
    leak_init: float = 0.5
    competition_init: float = 0.1
    stage2_noise_scale: float = 0.0
    lambda_behavior_choice: float = 1.0
    lambda_response_choice: float = 1.0
    lambda_behavior_rt: float = 2.0
    lambda_quantile: float = 0.25
    lambda_center: float = 1.0
    lambda_error_rate: float = 0.75
    lambda_error_sign: float = 0.75
    lambda_accuracy: float = 0.75
    lambda_accuracy_calibration: float = 0.75
    lambda_rate_reg: float = 0.05
    evidence_sequences_train: int = 1
    evidence_sequences_eval: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 5
    eval_seed: Optional[int] = None
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


READOUT_MODE_ALIASES = {
    "weighted_evidence": "weighted_evidence",
    "hard_stop_time_evidence": "hard_stop_time_evidence",
    "sampled_stop_time_evidence": "sampled_stop_time_evidence",
    "first_crosser": "first_crosser",
    "stop_time_evidence": "hard_stop_time_evidence",
}


def normalize_choice_readout_mode(mode: str) -> str:
    normalized = READOUT_MODE_ALIASES.get(mode, mode)
    if normalized not in {
        "weighted_evidence",
        "hard_stop_time_evidence",
        "sampled_stop_time_evidence",
        "first_crosser",
    }:
        raise ValueError(f"Unknown choice_readout_mode: {mode}")
    return normalized


def _gather_step_evidence(evidence_trajectory: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
    gather_index = step_index.long().view(-1, 1, 1).expand(-1, 1, evidence_trajectory.shape[-1])
    return evidence_trajectory.gather(1, gather_index).squeeze(1)


def _compute_first_crosser_step(evidence_trajectory: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    crossed = evidence_trajectory.amax(dim=2) >= threshold.unsqueeze(1)
    has_cross = crossed.any(dim=1)
    first_hit = crossed.float().argmax(dim=1)
    fallback = torch.full_like(first_hit, evidence_trajectory.shape[1] - 1)
    return torch.where(has_cross, first_hit, fallback)


def _softmax_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    if temperature.ndim == 0:
        scaled = logits / temperature.clamp_min(1e-6)
    else:
        scaled = logits / temperature.clamp_min(1e-6).unsqueeze(1)
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=50.0, neginf=-50.0)
    return torch.softmax(scaled, dim=1)


def _readout_choice_probs(
    *,
    evidence_trajectory: torch.Tensor,
    stop_probability: torch.Tensor,
    hard_stop_step: torch.Tensor,
    first_crosser_step: torch.Tensor,
    sampled_stop_step: Optional[torch.Tensor],
    choice_temperature: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized_mode = normalize_choice_readout_mode(mode)
    if normalized_mode == "weighted_evidence":
        readout_evidence = (stop_probability.unsqueeze(-1) * evidence_trajectory).sum(dim=1)
    elif normalized_mode == "hard_stop_time_evidence":
        readout_evidence = _gather_step_evidence(evidence_trajectory, hard_stop_step)
    elif normalized_mode == "sampled_stop_time_evidence":
        if sampled_stop_step is None:
            raise ValueError("sampled_stop_time_evidence requires sampled_stop_step")
        readout_evidence = _gather_step_evidence(evidence_trajectory, sampled_stop_step)
    elif normalized_mode == "first_crosser":
        readout_evidence = _gather_step_evidence(evidence_trajectory, first_crosser_step)
    else:
        raise ValueError(f"Unhandled readout mode: {normalized_mode}")
    choice_probs = _softmax_temperature(readout_evidence, choice_temperature)
    return choice_probs, readout_evidence


def _numpy_choice_probs_from_mode(
    *,
    evidence_trajectory: np.ndarray,
    stop_probability: np.ndarray,
    hard_stop_step: np.ndarray,
    first_crosser_step: np.ndarray,
    sampled_stop_step: Optional[np.ndarray],
    choice_temperature: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized_mode = normalize_choice_readout_mode(mode)
    if normalized_mode == "weighted_evidence":
        readout_evidence = (stop_probability[..., None] * evidence_trajectory).sum(axis=1)
    elif normalized_mode == "hard_stop_time_evidence":
        readout_evidence = evidence_trajectory[np.arange(evidence_trajectory.shape[0]), hard_stop_step]
    elif normalized_mode == "sampled_stop_time_evidence":
        if sampled_stop_step is None:
            raise ValueError("sampled_stop_time_evidence requires sampled_stop_step")
        readout_evidence = evidence_trajectory[np.arange(evidence_trajectory.shape[0]), sampled_stop_step]
    elif normalized_mode == "first_crosser":
        readout_evidence = evidence_trajectory[np.arange(evidence_trajectory.shape[0]), first_crosser_step]
    else:
        raise ValueError(f"Unhandled readout mode: {normalized_mode}")
    scaled = readout_evidence / max(float(choice_temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=50.0, neginf=-50.0)
    probs = np.exp(scaled)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs.astype(np.float32), readout_evidence.astype(np.float32)


class SemiSupervisedSPEA(nn.Module):
    def __init__(self, config: SemiSupervisedSPEAConfig):
        super().__init__()
        self.config = config
        self.evidence_projection = nn.Linear(config.evidence_dim, config.hidden_dim)
        self.acc_projection = nn.Linear(config.n_classes, config.hidden_dim)
        self.time_embedding = nn.Embedding(config.time_steps, config.hidden_dim)
        self.gru = nn.GRUCell(config.hidden_dim * 3, config.hidden_dim)
        self.delta_head = nn.Linear(config.hidden_dim + config.evidence_dim, config.n_classes)
        self.global_threshold_raw = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.global_t0_raw = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        self.leak_raw = nn.Parameter(torch.tensor(float(config.leak_init), dtype=torch.float32))
        self.competition_raw = nn.Parameter(torch.tensor(float(config.competition_init), dtype=torch.float32))
        self.class_bias = nn.Parameter(torch.zeros(config.n_classes, dtype=torch.float32))
        self.choice_temperature_raw = nn.Parameter(torch.tensor(np.log(np.expm1(max(float(config.choice_temperature), 1e-4))), dtype=torch.float32))

    def compute_soft_stopping(self, evidence_trajectory: torch.Tensor, threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_evidence = evidence_trajectory.amax(dim=2)
        margin = max_evidence - threshold.unsqueeze(1)
        hazard = torch.sigmoid(self.config.beta * margin)
        if self.config.hard_min_steps and self.config.min_steps > 0:
            hazard = hazard.clone()
            hazard[:, : self.config.min_steps] = 0.0
        one_minus = (1.0 - hazard).clamp(min=1e-6, max=1.0)
        survival_prev = torch.cumprod(
            torch.cat([torch.ones_like(one_minus[:, :1]), one_minus[:, :-1]], dim=1),
            dim=1,
        )
        stop_probability = hazard * survival_prev
        leftover = (1.0 - stop_probability.sum(dim=1, keepdim=True)).clamp(min=0.0)
        stop_probability = stop_probability.clone()
        stop_probability[:, -1:] = stop_probability[:, -1:] + leftover
        return hazard, stop_probability

    def rollout(
        self,
        evidence_samples: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if evidence_samples.ndim != 3:
            raise ValueError("evidence_samples must have shape [batch, time_steps, n_classes]")
        batch_size, time_steps, evidence_dim = evidence_samples.shape
        if evidence_dim != self.config.evidence_dim:
            raise ValueError(f"Expected evidence_dim={self.config.evidence_dim}, got {evidence_dim}")
        if time_steps > self.config.time_steps:
            raise ValueError(f"Configured time_steps={self.config.time_steps} but received {time_steps}")

        device = evidence_samples.device
        hidden = torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=evidence_samples.dtype)
        accumulator = torch.zeros(batch_size, self.config.n_classes, device=device, dtype=evidence_samples.dtype)
        threshold = F.softplus(self.global_threshold_raw) + self.config.threshold_min
        threshold = threshold.expand(batch_size)
        t0 = F.softplus(self.global_t0_raw).expand(batch_size)
        choice_temperature = F.softplus(self.choice_temperature_raw).expand(batch_size)
        if not bool(self.config.temperature_calibration):
            choice_temperature = torch.full_like(choice_temperature, float(self.config.choice_temperature))
        leak = torch.sigmoid(self.leak_raw)
        competition_gain = F.softplus(self.competition_raw)

        hidden_states = []
        evidence_increments = []
        evidence_trajectory = []

        for t in range(time_steps):
            e_t = evidence_samples[:, t, :]
            projected_evidence = self.evidence_projection(e_t)
            projected_accumulator = self.acc_projection(accumulator)
            time_embed = self.time_embedding(torch.full((batch_size,), t, device=device, dtype=torch.long))
            hidden = self.gru(torch.cat([projected_evidence, projected_accumulator, time_embed], dim=1), hidden)
            raw_delta = self.delta_head(torch.cat([hidden, e_t], dim=1)) + self.class_bias.unsqueeze(0)
            if float(self.config.stage2_noise_scale) > 0.0:
                stage2_noise = torch.randn(
                    raw_delta.shape,
                    generator=generator,
                    device=device,
                    dtype=raw_delta.dtype,
                )
                raw_delta = raw_delta + float(self.config.stage2_noise_scale) * stage2_noise
            raw_delta = torch.nan_to_num(raw_delta, nan=0.0, posinf=20.0, neginf=-20.0)
            positive_delta = F.softplus(raw_delta)
            competition = competition_gain * (accumulator.sum(dim=1, keepdim=True) - accumulator)
            accumulator = F.relu(leak * accumulator + positive_delta - competition)
            hidden_states.append(hidden)
            evidence_increments.append(positive_delta)
            evidence_trajectory.append(accumulator)

        hidden_states_t = torch.stack(hidden_states, dim=1)
        evidence_increments_t = torch.stack(evidence_increments, dim=1)
        evidence_trajectory_t = torch.stack(evidence_trajectory, dim=1)
        stopping_hazard, stop_probability = self.compute_soft_stopping(evidence_trajectory_t, threshold)
        hard_stop_step = stop_probability.argmax(dim=1)
        first_crosser_step = _compute_first_crosser_step(evidence_trajectory_t, threshold)
        configured_mode = normalize_choice_readout_mode(self.config.choice_readout_mode)
        choice_probs, readout_evidence = _readout_choice_probs(
            evidence_trajectory=evidence_trajectory_t,
            stop_probability=stop_probability,
            hard_stop_step=hard_stop_step,
            first_crosser_step=first_crosser_step,
            sampled_stop_step=None,
            choice_temperature=choice_temperature,
            mode=configured_mode,
        )
        choice_logits = torch.nan_to_num(readout_evidence / choice_temperature.clamp_min(1e-6).unsqueeze(1), nan=0.0, posinf=50.0, neginf=-50.0)
        weighted_choice_probs, weighted_evidence = _readout_choice_probs(
            evidence_trajectory=evidence_trajectory_t,
            stop_probability=stop_probability,
            hard_stop_step=hard_stop_step,
            first_crosser_step=first_crosser_step,
            sampled_stop_step=None,
            choice_temperature=choice_temperature,
            mode="weighted_evidence",
        )
        hard_stop_choice_probs, hard_stop_evidence = _readout_choice_probs(
            evidence_trajectory=evidence_trajectory_t,
            stop_probability=stop_probability,
            hard_stop_step=hard_stop_step,
            first_crosser_step=first_crosser_step,
            sampled_stop_step=None,
            choice_temperature=choice_temperature,
            mode="hard_stop_time_evidence",
        )
        first_crosser_choice_probs, first_crosser_evidence = _readout_choice_probs(
            evidence_trajectory=evidence_trajectory_t,
            stop_probability=stop_probability,
            hard_stop_step=hard_stop_step,
            first_crosser_step=first_crosser_step,
            sampled_stop_step=None,
            choice_temperature=choice_temperature,
            mode="first_crosser",
        )
        no_crossing_mask = evidence_trajectory_t.amax(dim=2).amax(dim=1) < threshold
        pred_choice_hard = choice_probs.argmax(dim=1)
        pred_rt_hard = t0 + hard_stop_step.to(evidence_samples.dtype) * (self.config.dt_ms / 1000.0)
        time_axis = torch.arange(time_steps, device=device, dtype=evidence_samples.dtype) * (self.config.dt_ms / 1000.0)
        pred_rt_soft = t0 + (stop_probability * time_axis.unsqueeze(0)).sum(dim=1)

        return {
            "hidden_states": hidden_states_t,
            "evidence_increments": evidence_increments_t,
            "evidence_trajectory": evidence_trajectory_t,
            "stopping_hazard": stopping_hazard,
            "stop_probability": stop_probability,
            "hard_stop_step": hard_stop_step,
            "first_crosser_step": first_crosser_step,
            "pred_choice_hard": pred_choice_hard,
            "pred_rt_hard": pred_rt_hard,
            "pred_rt_soft": pred_rt_soft,
            "choice_logits": choice_logits,
            "choice_probs": choice_probs,
            "weighted_choice_probs": weighted_choice_probs,
            "weighted_evidence": weighted_evidence,
            "hard_stop_choice_probs": hard_stop_choice_probs,
            "hard_stop_evidence": hard_stop_evidence,
            "first_crosser_choice_probs": first_crosser_choice_probs,
            "first_crosser_evidence": first_crosser_evidence,
            "threshold": threshold,
            "t0": t0,
            "choice_temperature_scalar": choice_temperature,
            "configured_choice_readout_mode": torch.full((batch_size,), list(READOUT_MODE_ALIASES).index(configured_mode) if configured_mode in READOUT_MODE_ALIASES else 0, device=device, dtype=evidence_samples.dtype),
            "no_crossing_mask": no_crossing_mask,
        }


def sample_spea_readout(
    predictions: Mapping[str, np.ndarray | torch.Tensor],
    *,
    dt_ms: int,
    seed: int,
    choice_readout_mode: str = "sampled_stop_time_evidence",
    choice_temperature: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    stop_probability = _to_numpy(predictions["stop_probability"]).astype(np.float64)
    t0 = _to_numpy(predictions["t0"]).astype(np.float32)
    stop_probability = stop_probability / np.clip(stop_probability.sum(axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed)
    sampled_stop_step = np.array([rng.choice(stop_probability.shape[1], p=row) for row in stop_probability], dtype=np.int64)
    readout_temperature = choice_temperature
    if readout_temperature is None:
        if "choice_temperature_scalar" in predictions:
            readout_temperature = float(np.asarray(_to_numpy(predictions["choice_temperature_scalar"])).reshape(-1)[0])
        else:
            readout_temperature = 0.10
    if "evidence_trajectory" not in predictions:
        choice_probs = _to_numpy(predictions["choice_probs"]).astype(np.float64)
        choice_probs = choice_probs / np.clip(choice_probs.sum(axis=1, keepdims=True), 1e-12, None)
        pred_choice_sampled = np.array([rng.choice(choice_probs.shape[1], p=row) for row in choice_probs], dtype=np.int64)
        pred_rt_sampled = t0 + sampled_stop_step.astype(np.float32) * (float(dt_ms) / 1000.0)
        return {
            "pred_choice_sampled": pred_choice_sampled,
            "sampled_stop_step": sampled_stop_step,
            "pred_rt_sampled": pred_rt_sampled.astype(np.float32),
            "sampled_choice_probs": choice_probs.astype(np.float32),
            "sampled_readout_evidence": np.zeros((choice_probs.shape[0], choice_probs.shape[1]), dtype=np.float32),
            "choice_readout_mode": np.asarray("weighted_evidence"),
            "readout_seed": np.asarray(seed, dtype=np.int64),
        }
    evidence_trajectory = _to_numpy(predictions["evidence_trajectory"]).astype(np.float32)
    hard_stop_step = _to_numpy(predictions["hard_stop_step"]).astype(np.int64)
    if "first_crosser_step" in predictions:
        first_crosser_step = _to_numpy(predictions["first_crosser_step"]).astype(np.int64)
    else:
        threshold = _to_numpy(predictions["threshold"]).astype(np.float32)
        crossed = evidence_trajectory.max(axis=2) >= threshold[:, None]
        first_crosser_step = np.where(crossed.any(axis=1), crossed.argmax(axis=1), evidence_trajectory.shape[1] - 1).astype(np.int64)
    choice_probs, readout_evidence = _numpy_choice_probs_from_mode(
        evidence_trajectory=evidence_trajectory,
        stop_probability=stop_probability.astype(np.float32),
        hard_stop_step=hard_stop_step,
        first_crosser_step=first_crosser_step,
        sampled_stop_step=sampled_stop_step,
        choice_temperature=float(readout_temperature),
        mode=choice_readout_mode,
    )
    pred_choice_sampled = np.array([rng.choice(choice_probs.shape[1], p=row) for row in choice_probs], dtype=np.int64)
    pred_rt_sampled = t0 + sampled_stop_step.astype(np.float32) * (float(dt_ms) / 1000.0)
    return {
        "pred_choice_sampled": pred_choice_sampled,
        "sampled_stop_step": sampled_stop_step,
        "pred_rt_sampled": pred_rt_sampled.astype(np.float32),
        "sampled_choice_probs": choice_probs.astype(np.float32),
        "sampled_readout_evidence": readout_evidence.astype(np.float32),
        "choice_readout_mode": np.asarray(normalize_choice_readout_mode(choice_readout_mode)),
        "readout_seed": np.asarray(seed, dtype=np.int64),
    }


def compute_spea_losses(
    *,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: SemiSupervisedSPEAConfig,
) -> Dict[str, torch.Tensor]:
    response_labels = targets["response_labels"].long()
    target_labels = targets["target_labels"].long()
    true_rt = torch.nan_to_num(targets["rts"].float(), nan=0.0, posinf=20.0, neginf=0.0)
    choice_probs = torch.nan_to_num(predictions["choice_probs"], nan=1e-8, posinf=1.0, neginf=1e-8).clamp_min(1e-8)
    pred_rt_soft = torch.nan_to_num(predictions["pred_rt_soft"], nan=0.0, posinf=20.0, neginf=0.0)
    pred_rt_hard = torch.nan_to_num(predictions["pred_rt_hard"], nan=0.0, posinf=20.0, neginf=0.0)
    choice_log_probs = torch.log(choice_probs)
    choice_loss = F.nll_loss(choice_log_probs, response_labels)
    rt_loss = F.mse_loss(pred_rt_soft, true_rt)

    pred_rt_sorted = torch.sort(pred_rt_soft)[0]
    true_rt_sorted = torch.sort(true_rt)[0]
    quantile_losses = []
    n = pred_rt_sorted.shape[0]
    for q in config.quantiles:
        idx = min(n - 1, max(0, int(round((n - 1) * float(q)))))
        quantile_losses.append(torch.abs(pred_rt_sorted[idx] - true_rt_sorted[idx]))
    quantile_loss = torch.stack(quantile_losses).mean() if quantile_losses else torch.zeros((), device=true_rt.device)

    center_loss = torch.abs(pred_rt_soft.mean() - true_rt.mean())

    target_probs = choice_probs.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    model_accuracy_soft = target_probs.mean()
    human_accuracy = (response_labels == target_labels).float().mean()
    accuracy_loss = torch.abs(model_accuracy_soft - human_accuracy)
    model_error_rate = 1.0 - model_accuracy_soft
    human_error_rate = 1.0 - human_accuracy
    error_rate_loss = (model_error_rate - human_error_rate) ** 2

    hard_correct_mask = predictions["pred_choice_hard"] == target_labels
    hard_error_mask = ~hard_correct_mask
    if hard_correct_mask.any() and hard_error_mask.any():
        pred_error_minus_correct = pred_rt_hard[hard_error_mask].mean() - pred_rt_hard[hard_correct_mask].mean()
    else:
        pred_error_minus_correct = torch.zeros((), device=true_rt.device)
    human_correct_mask = response_labels == target_labels
    human_error_mask = ~human_correct_mask
    if human_correct_mask.any() and human_error_mask.any():
        human_error_minus_correct = true_rt[human_error_mask].mean() - true_rt[human_correct_mask].mean()
    else:
        human_error_minus_correct = torch.zeros((), device=true_rt.device)
    error_sign_loss = torch.abs(pred_error_minus_correct - human_error_minus_correct)

    rate_reg = torch.nan_to_num(predictions["evidence_increments"], nan=0.0, posinf=10.0, neginf=0.0).mean()

    loss = (
        config.lambda_behavior_choice * choice_loss
        + config.lambda_response_choice * choice_loss
        + config.lambda_behavior_rt * rt_loss
        + config.lambda_quantile * quantile_loss
        + config.lambda_center * center_loss
        + config.lambda_error_rate * error_rate_loss
        + config.lambda_error_sign * error_sign_loss
        + config.lambda_accuracy * accuracy_loss
        + config.lambda_accuracy_calibration * accuracy_loss
        + config.lambda_rate_reg * rate_reg
    )
    return {
        "loss": loss,
        "choice_loss": choice_loss,
        "rt_loss": rt_loss,
        "quantile_loss": quantile_loss,
        "center_loss": center_loss,
        "error_rate_loss": error_rate_loss,
        "error_sign_loss": error_sign_loss,
        "accuracy_loss": accuracy_loss,
        "rate_reg": rate_reg,
    }


def infer_spea_predictions_from_params(
    *,
    params: Dict[str, np.ndarray],
    evidence_samples: np.ndarray,
    config: SemiSupervisedSPEAConfig,
    device: str,
) -> Dict[str, np.ndarray]:
    model = SemiSupervisedSPEA(config)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in params:
            state_dict[key] = torch.as_tensor(params[key], dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model.rollout(torch.as_tensor(evidence_samples, dtype=torch.float32, device=device))
    return {key: _to_numpy(value) for key, value in predictions.items()}


def evaluate_spea_predictions(
    *,
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    metrics = evaluate_joint_behavior(
        pred_rt=np.asarray(predictions["pred_rt_hard"], dtype=np.float32),
        pred_choice=np.asarray(predictions["pred_choice_hard"], dtype=np.int64),
        true_rt=np.asarray(targets["rts"], dtype=np.float32),
        target_labels=np.asarray(targets["target_labels"], dtype=np.int64),
        response_labels=np.asarray(targets["response_labels"], dtype=np.int64),
        congruency=np.asarray(targets["congruency"], dtype=np.int64),
        human_stats=compute_human_stats_from_rts(np.asarray(targets["rts"], dtype=np.float32)),
        rt_shape_focus=True,
    )
    for key, value in list(metrics.items()):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            metrics[key] = 0.0
    metrics["early_stop_rate"] = float((np.asarray(predictions["hard_stop_step"], dtype=np.int64) < 10).mean())
    metrics["no_crossing_rate"] = float(np.asarray(predictions["no_crossing_mask"], dtype=bool).mean())
    return metrics


def fit_spea_from_stage1_inputs(
    *,
    train_bundle: Dict[str, np.ndarray],
    test_bundle: Dict[str, np.ndarray],
    config: SemiSupervisedSPEAConfig,
    output_dir: str,
    device: str,
    random_seed: int,
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    model = SemiSupervisedSPEA(config).to(device)
    optimizer = Adam(model.parameters(), lr=float(config.learning_rate))

    train_dataset = TensorDataset(
        torch.as_tensor(train_bundle["evidence_samples"], dtype=torch.float32),
        torch.as_tensor(train_bundle["response_labels"], dtype=torch.long),
        torch.as_tensor(train_bundle["target_labels"], dtype=torch.long),
        torch.as_tensor(train_bundle["rts"], dtype=torch.float32),
        torch.as_tensor(train_bundle["congruency"], dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True)

    best: Optional[Dict[str, Any]] = None
    start_epoch = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        model_state = checkpoint["model_state"].item() if "model_state" in checkpoint.files else None
        optimizer_state = checkpoint["optimizer_state"].item() if "optimizer_state" in checkpoint.files else None
        if isinstance(model_state, dict):
            model.load_state_dict({k: torch.as_tensor(v, dtype=model.state_dict()[k].dtype) for k, v in model_state.items()}, strict=False)
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        if "epoch" in checkpoint.files:
            start_epoch = int(np.asarray(checkpoint["epoch"]).item()) + 1

    for epoch in range(start_epoch, int(config.epochs)):
        model.train()
        for evidence_samples, response_labels, target_labels, rts, congruency in train_loader:
            evidence_samples = evidence_samples.to(device)
            targets = {
                "response_labels": response_labels.to(device),
                "target_labels": target_labels.to(device),
                "rts": rts.to(device),
                "congruency": congruency.to(device),
            }
            predictions = model.rollout(evidence_samples)
            losses = compute_spea_losses(predictions=predictions, targets=targets, config=config)
            if not torch.isfinite(losses["loss"]):
                continue
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model.rollout(torch.as_tensor(test_bundle["evidence_samples"], dtype=torch.float32, device=device))
        prediction_np = {key: _to_numpy(value) for key, value in predictions.items()}
        passthrough_keys = [
            "evidence_samples",
            "evidence_sample_mu",
            "evidence_sample_sigma",
            "stage1_uncertainty_scalar",
            "entropy_mean",
            "entropy_var",
            "margin_mean",
            "margin_var",
            "target_labels",
            "response_labels",
            "congruency",
        ]
        for key in passthrough_keys:
            if key in test_bundle:
                prediction_np[key] = np.asarray(test_bundle[key])
        if "rts" in test_bundle:
            prediction_np["true_rt"] = np.asarray(test_bundle["rts"], dtype=np.float32)
        metrics = evaluate_spea_predictions(predictions=prediction_np, targets=test_bundle)
        score = float(metrics["behavior_optimal_score"])
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "results": metrics,
                "predictions": prediction_np,
                "params": {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
            }

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                checkpoint_path,
                epoch=np.array(int(epoch), dtype=np.int32),
                model_state=np.array([{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}], dtype=object),
                optimizer_state=np.array([optimizer.state_dict()], dtype=object),
            )

    if best is None:
        raise RuntimeError("No SPEA checkpoints evaluated")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "best_config.json").write_text(json.dumps({"config": to_jsonable(config.__dict__), "epoch": int(best["epoch"])}, indent=2), encoding="utf-8")
    (output_path / "metrics_smoke.json").write_text(json.dumps(to_jsonable(best["results"]), indent=2), encoding="utf-8")
    np.savez_compressed(output_path / "predictions_smoke.npz", **best["predictions"])
    np.savez(output_path / "best_model_params.npz", **best["params"])
    return {"best": best}


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()

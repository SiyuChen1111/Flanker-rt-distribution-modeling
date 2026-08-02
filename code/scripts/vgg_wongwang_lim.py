"""
VGG-WongWang LIM Model

Two-stage model for Lost in Migration task:
- Stage 1: VGG16 feature extraction + classification
- Stage 2: Wong-Wang decision module for RT prediction

Architecture:
    Input Image (128x128x3)
        ↓
    VGG16 (pretrained)
        ↓
    FC Layer → logits (4 classes: L/R/U/D)
        ↓
    Linear Transform
        ↓
    WongWangMultiClassDecision (4 competing neural populations)
        ↓
    decision_times → RT prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional, Dict, Any, cast, Union
import torchvision.models as models


def apply_stage2_input_transform(
    logits: torch.Tensor,
    scale: torch.Tensor,
    transform_mode: str = 'relu',
    softplus_center: float = 0.0,
) -> torch.Tensor:
    """Map Stage-1 logits into non-negative Stage-2 input.

    Modes:
        relu             – original  ReLU(logits * scale)  (default)
        softplus_centered – softplus(logits * scale - center)
                           No dead zone; center shifts the
                           operating point away from saturation.
    """
    scaled = logits * scale
    if transform_mode == 'relu':
        return F.relu(scaled)
    if transform_mode == 'softplus_centered':
        return F.softplus(scaled - softplus_center)
    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def _build_alpha_pulse(time_axis: torch.Tensor, peak_s: float) -> torch.Tensor:
    peak = max(float(peak_s), 1e-6)
    scaled_time = time_axis / peak
    return scaled_time * torch.exp(1.0 - scaled_time)


def build_dynamic_stage2_input(
    logits: torch.Tensor,
    scale: torch.Tensor,
    time_steps: int,
    config: Optional[Dict[str, Any]] = None,
    target_labels: Optional[torch.Tensor] = None,
    flanker_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    config = config or {}
    transform_mode = str(config.get('transform_mode', 'relu'))
    softplus_center = float(config.get('softplus_center', 0.0))
    transformed = apply_stage2_input_transform(
        logits, scale,
        transform_mode=transform_mode,
        softplus_center=softplus_center,
    )
    selection_mode = str(config.get('selection_mode', 'baseline'))
    if selection_mode == 'baseline':
        return transformed, {}
    if selection_mode not in {'dynamic_flanker_suppression', 'dynamic_flanker_dmc_like'}:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    if target_labels is None or flanker_labels is None:
        raise ValueError(f'{selection_mode} requires target_labels and flanker_labels')

    dt_ms = float(config.get('dt_ms', 10.0))
    selection_strength = float(config.get('selection_strength', 0.35))
    selection_midpoint_s = float(config.get('selection_midpoint_s', 0.18))
    selection_tau_s = max(float(config.get('selection_tau_s', 0.06)), 1e-6)
    target_boost = float(config.get('target_boost', 0.0))
    selection_apply_to = str(config.get('selection_apply_to', 'incongruent_only'))
    auto_strength = float(config.get('auto_strength', 0.0))
    auto_peak_s = float(config.get('auto_peak_s', 0.06))
    # Optional minimal mechanism: time-localized conflict capture.
    # Default disabled (capture_strength=0.0) to preserve baseline behavior.
    capture_strength = float(config.get('capture_strength', 0.0))
    capture_midpoint_s = float(config.get('capture_midpoint_s', 0.05))
    capture_tau_s = max(float(config.get('capture_tau_s', 0.03)), 1e-6)

    batch_size, n_classes = transformed.shape
    if target_labels.shape[0] != batch_size or flanker_labels.shape[0] != batch_size:
        raise ValueError('target_labels/flanker_labels must align with logits batch size')

    time_axis = torch.arange(time_steps, device=transformed.device, dtype=transformed.dtype) * (dt_ms / 1000.0)
    selection_gate = torch.sigmoid((time_axis - selection_midpoint_s) / selection_tau_s)
    auto_pulse_t = torch.zeros_like(selection_gate)
    capture_pulse_t = torch.zeros_like(selection_gate)
    selection_mode_id = 1
    if selection_mode == 'dynamic_flanker_dmc_like':
        auto_pulse_t = _build_alpha_pulse(time_axis, peak_s=auto_peak_s)
        selection_mode_id = 2
    if selection_mode == 'dynamic_flanker_suppression' and abs(capture_strength) > 0.0:
        # Symmetric, bounded pulse peaking near capture_midpoint_s.
        capture_pulse_t = torch.exp(-torch.abs(time_axis - capture_midpoint_s) / capture_tau_s)
    flanker_multiplier_t = (
        1.0
        + capture_strength * capture_pulse_t
        + auto_strength * auto_pulse_t
        - selection_strength * selection_gate
    ).clamp_min(0.0)
    target_multiplier_t = (1.0 - auto_strength * auto_pulse_t + target_boost * selection_gate).clamp_min(0.0)

    dynamic_input = transformed.unsqueeze(1).repeat(1, time_steps, 1)
    if selection_apply_to == 'incongruent_only':
        selection_trial_mask = target_labels != flanker_labels
    elif selection_apply_to == 'all_trials':
        selection_trial_mask = torch.ones(batch_size, device=transformed.device, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown selection_apply_to: {selection_apply_to}")

    selected_idx = torch.nonzero(selection_trial_mask, as_tuple=False).squeeze(1)
    if selected_idx.numel() > 0:
        time_idx = torch.arange(time_steps, device=transformed.device).unsqueeze(0)
        flanker_idx = flanker_labels[selected_idx].unsqueeze(1)
        dynamic_input[selected_idx.unsqueeze(1), time_idx, flanker_idx] = (
            dynamic_input[selected_idx.unsqueeze(1), time_idx, flanker_idx]
            * flanker_multiplier_t.unsqueeze(0)
        )
        if abs(target_boost) > 0.0:
            target_idx = target_labels[selected_idx].unsqueeze(1)
            dynamic_input[selected_idx.unsqueeze(1), time_idx, target_idx] = (
                dynamic_input[selected_idx.unsqueeze(1), time_idx, target_idx]
                * target_multiplier_t.unsqueeze(0)
            )

    traces = {
        'selection_mode': torch.tensor(selection_mode_id, device=transformed.device, dtype=torch.int32),
        'selection_gate': selection_gate,
        'auto_pulse_t': auto_pulse_t,
        'capture_pulse_t': capture_pulse_t,
        'flanker_multiplier_t': flanker_multiplier_t,
        'target_multiplier_t': target_multiplier_t,
        'selection_trial_mask': selection_trial_mask.to(torch.int32),
    }
    return dynamic_input, traces


def _first_crossing_times(evidence_traj: torch.Tensor, dt_ms: float) -> Tuple[torch.Tensor, torch.Tensor]:
    max_time = evidence_traj.shape[1]
    crossing_mask = evidence_traj > 0
    decision_indices = crossing_mask.float().argmax(dim=1).long()
    no_cross = ~crossing_mask.any(dim=1)
    decision_indices = decision_indices.masked_fill(no_cross, max_time - 1)
    decision_times = decision_indices.to(evidence_traj.dtype) * (float(dt_ms) / 1000.0)
    return decision_indices, decision_times


def compute_legacy_choice_logits(evidence_traj: torch.Tensor, choice_temperature: float) -> torch.Tensor:
    class_strength = evidence_traj.amax(dim=1)
    temperature = max(float(choice_temperature), 1e-6)
    return class_strength / temperature


def compute_baseline_readout(
    evidence_traj: torch.Tensor,
    readout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    config = readout_config or {}
    dt_ms = float(config.get('dt_ms', 10.0))
    decision_indices, decision_times = _first_crossing_times(evidence_traj, dt_ms=dt_ms)
    pred_rt, winner_idx = decision_times.min(dim=1)
    return {
        'pred_rt': pred_rt,
        'decision_indices': decision_indices,
        'decision_times': decision_times,
        'winner_idx': winner_idx,
    }


def _resolve_t0_seconds(
    config: Optional[Dict[str, Any]],
    reference_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    cfg = config or {}
    t0_mode = str(cfg.get('t0_mode', 'disabled'))
    valid_modes = {'disabled', 'fixed_global', 'fit_global', 'fit_age_group'}
    if t0_mode not in valid_modes:
        raise ValueError(f"Unknown t0_mode: {t0_mode}")

    raw_value = cfg.get('t0_seconds', 0.0)
    if isinstance(raw_value, torch.Tensor):
        t0_seconds = raw_value.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    else:
        t0_seconds = torch.as_tensor(raw_value, device=reference_tensor.device, dtype=reference_tensor.dtype)
    if t0_mode == 'disabled':
        t0_seconds = torch.zeros_like(t0_seconds)
    else:
        t0_seconds = torch.clamp(t0_seconds, min=0.0)
    return t0_seconds, t0_mode


def apply_t0_readout_shift(
    pred_rt: torch.Tensor,
    readout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0_seconds, t0_mode = _resolve_t0_seconds(readout_config, pred_rt)
    shifted_rt = pred_rt + t0_seconds
    return {
        'pred_rt': shifted_rt,
        't0_seconds': t0_seconds,
        't0_mode': t0_mode,
    }


def extract_decision_variable(evidence_traj: torch.Tensor, config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    del config
    top2 = torch.topk(evidence_traj, k=min(2, evidence_traj.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        return top2[..., 0]
    return top2[..., 0] - top2[..., 1]


def compute_soft_hazard_readout(
    evidence_traj: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    config = config or {}
    dt_ms = float(config.get('dt_ms', 10.0))
    alpha = float(config.get('alpha', 12.0))
    beta = float(config.get('beta', 0.15))
    eps = float(config.get('eps', 1e-6))

    baseline = compute_baseline_readout(evidence_traj, readout_config=config)
    dv_t = extract_decision_variable(evidence_traj, config)
    hazard_t = torch.sigmoid(alpha * (dv_t - beta)).clamp(min=eps, max=1.0 - eps)

    one_minus = (1.0 - hazard_t).clamp(min=eps, max=1.0)
    survival_prev = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :1]), one_minus[:, :-1]], dim=1),
        dim=1,
    )
    rt_mass = survival_prev * hazard_t
    leftover = (1.0 - rt_mass.sum(dim=1, keepdim=True)).clamp(min=0.0)
    rt_mass = rt_mass.clone()
    rt_mass[:, -1:] = rt_mass[:, -1:] + leftover

    time_axis = torch.arange(
        evidence_traj.shape[1],
        device=evidence_traj.device,
        dtype=evidence_traj.dtype,
    ) * (dt_ms / 1000.0)
    pred_rt = (rt_mass * time_axis.unsqueeze(0)).sum(dim=1)
    decision_index = rt_mass.argmax(dim=1)
    winner_idx = evidence_traj[
        torch.arange(evidence_traj.shape[0], device=evidence_traj.device),
        decision_index,
    ].argmax(dim=1)

    return {
        'pred_rt': pred_rt,
        'decision_indices': baseline['decision_indices'],
        'decision_times': baseline['decision_times'],
        'winner_idx': winner_idx,
        'hazard_t': hazard_t,
        'rt_mass': rt_mass,
        'dv_t': dv_t,
    }


def compute_urgency_readout(
    evidence_traj: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    config = config or {}
    dt_ms = float(config.get('dt_ms', 10.0))
    urgency_type = str(config.get('urgency_type', 'additive_urgency'))
    urgency_start = float(config.get('urgency_start', 0.80))
    urgency_slope = float(config.get('urgency_slope', 0.25))
    urgency_floor = float(config.get('urgency_floor', 0.0))

    baseline = compute_baseline_readout(evidence_traj, readout_config=config)
    dv_t = extract_decision_variable(evidence_traj, config)
    time_axis = torch.arange(
        evidence_traj.shape[1],
        device=evidence_traj.device,
        dtype=evidence_traj.dtype,
    ) * (dt_ms / 1000.0)
    urgency_gain = torch.clamp(time_axis - urgency_start, min=0.0) * urgency_slope
    if urgency_floor > 0.0:
        active = time_axis >= urgency_start
        urgency_gain = torch.where(active, torch.clamp(urgency_gain, min=urgency_floor), urgency_gain)

    baseline_index = baseline['pred_rt'].div(dt_ms / 1000.0).round().long().clamp(max=evidence_traj.shape[1] - 1)
    baseline_threshold = dv_t[
        torch.arange(evidence_traj.shape[0], device=evidence_traj.device),
        baseline_index,
    ]

    if urgency_type == 'additive_urgency':
        adjusted_signal = dv_t + urgency_gain.unsqueeze(0)
        commit_mask = adjusted_signal >= baseline_threshold.unsqueeze(1)
    elif urgency_type == 'collapsing_bound':
        bound_t = torch.clamp(
            baseline_threshold.unsqueeze(1) - urgency_gain.unsqueeze(0),
            min=urgency_floor,
        )
        adjusted_signal = dv_t
        commit_mask = adjusted_signal >= bound_t
    else:
        raise ValueError(f"Unknown urgency_type: {urgency_type}")

    urgency_index = commit_mask.float().argmax(dim=1).long()
    no_commit = ~commit_mask.any(dim=1)
    urgency_index = torch.where(no_commit, baseline_index, urgency_index)
    pred_rt = urgency_index.to(evidence_traj.dtype) * (dt_ms / 1000.0)
    winner_idx = evidence_traj[
        torch.arange(evidence_traj.shape[0], device=evidence_traj.device),
        urgency_index,
    ].argmax(dim=1)

    return {
        'pred_rt': pred_rt,
        'decision_indices': baseline['decision_indices'],
        'decision_times': baseline['decision_times'],
        'winner_idx': winner_idx,
        'dv_t': dv_t,
        'urgency_gain': urgency_gain,
        'baseline_index': baseline_index,
        'urgency_index': urgency_index,
        'baseline_threshold': baseline_threshold,
    }


def compute_soft_index_readout(
    evidence_traj: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Soft-index selection readout — decouples *which* class from *when*.

    Inspired by the RNN/LSTM RTify pipeline (RNN_decision.py /
    AlexNet_BN_LSTM_sup_2.py).  For each class we place a Gaussian bump
    around its first-crossing time, then read out the evidence at that
    soft-indexed position.  Choice probabilities come from softmax over
    class evidence; RT is a probability-weighted average of class decision
    times.

    Key parameters (via config):
        sigma_s: width of the Gaussian soft-index in seconds (default 0.05)
        choice_temperature: temperature for the choice softmax (default 0.10)
    """
    config = config or {}
    dt_ms = float(config.get('dt_ms', 10.0))
    sigma_s = float(config.get('sigma_s', 0.05))
    sigma_steps = max(sigma_s / (dt_ms / 1000.0), 0.5)  # at least half a step
    choice_temperature = max(float(config.get('choice_temperature', 0.10)), 1e-6)

    baseline = compute_baseline_readout(evidence_traj, readout_config=config)
    decision_indices = baseline['decision_indices']        # [B, C]  integer steps
    decision_times_class = baseline['decision_times']      # [B, C]  seconds

    B, T, C = evidence_traj.shape
    device = evidence_traj.device
    dtype = evidence_traj.dtype

    time_axis = torch.arange(T, device=device, dtype=dtype)  # [T]

    # decision_indices: [B, C] → expand to [B, C, T]
    center = decision_indices.unsqueeze(-1).to(dtype)       # [B, C, 1]
    # Gaussian soft-index per (batch, class) around its crossing time
    soft_index = torch.exp(-0.5 * ((time_axis - center) / sigma_steps) ** 2)  # [B, C, T]
    soft_index = soft_index / soft_index.sum(dim=2, keepdim=True).clamp_min(1e-8)

    # class_evidence[b, c] = Σ_t soft_index[b, c, t] · evidence_traj[b, t, c]
    class_evidence = (soft_index * evidence_traj.permute(0, 2, 1)).sum(dim=2)  # [B, C]

    choice_probs = torch.softmax(class_evidence / choice_temperature, dim=1)
    winner_idx = choice_probs.argmax(dim=1)

    # RT: probability-weighted average of per-class decision times
    pred_rt = (choice_probs * decision_times_class).sum(dim=1)

    return {
        'pred_rt': pred_rt,
        'decision_indices': decision_indices,
        'decision_times': decision_times_class,
        'winner_idx': winner_idx,
        'choice_probs': choice_probs,
        'class_evidence': class_evidence,
        'soft_index': soft_index,
    }


def compute_rt_readout(
    mode: str,
    evidence_traj: torch.Tensor,
    readout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if mode == 'baseline':
        readout = compute_baseline_readout(evidence_traj, readout_config=readout_config)
    elif mode == 'soft_hazard':
        readout = compute_soft_hazard_readout(evidence_traj, config=readout_config)
    elif mode == 'urgency':
        readout = compute_urgency_readout(evidence_traj, config=readout_config)
    elif mode == 'soft_index':
        readout = compute_soft_index_readout(evidence_traj, config=readout_config)
    elif mode == 'noisy_readout':
        raise NotImplementedError(f"RT readout mode '{mode}' is reserved for later experiments.")
    else:
        raise ValueError(f"Unknown RT readout mode: {mode}")

    shifted = apply_t0_readout_shift(readout['pred_rt'], readout_config=readout_config)
    readout = dict(readout)
    readout['pred_rt'] = shifted['pred_rt']
    readout['t0_seconds'] = shifted['t0_seconds']
    readout['t0_mode'] = shifted['t0_mode']
    return readout


def compute_behavioral_losses(
    *,
    pred_rt: torch.Tensor,
    pred_choice: torch.Tensor,
    choice_probs: torch.Tensor,
    target_labels: torch.Tensor,
    response_labels: torch.Tensor,
    true_rt: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """SPEA-style behavioral loss suite adapted for Wong-Wang training.

    Provides fine-grained behavioral constraints beyond simple RT MSE:
      - error_rate_loss: squared error between model and human error rates
      - error_sign_loss:  absolute error between model and human error-correct RT gaps
      - accuracy_loss:    absolute error between model and human accuracy
      - response_nll:     NLL on response labels (supervised choice)
      - rt_mse:           MSE on RT

    Weight keys (via config) with defaults:
      lambda_error_rate   (0.0)
      lambda_error_sign   (0.0)
      lambda_accuracy     (0.0)
      lambda_response_nll (1.0)
      lambda_rt_mse       (1.0)
    """
    config = config or {}
    zero = pred_rt.new_zeros(())

    # --- error rate loss ---
    lambda_error_rate = float(config.get('lambda_error_rate', 0.0))
    if lambda_error_rate > 0.0:
        human_accuracy = (response_labels == target_labels).float().mean()
        model_accuracy = (pred_choice == target_labels).float().mean()
        error_rate_loss = (1.0 - model_accuracy - (1.0 - human_accuracy)).pow(2)
    else:
        error_rate_loss = zero

    # --- error sign loss ---
    lambda_error_sign = float(config.get('lambda_error_sign', 0.0))
    if lambda_error_sign > 0.0:
        pred_correct_mask = pred_choice == target_labels
        pred_error_mask = ~pred_correct_mask
        human_correct_mask = response_labels == target_labels
        human_error_mask = ~human_correct_mask

        if pred_correct_mask.any() and pred_error_mask.any():
            pred_gap = pred_rt[pred_error_mask].mean() - pred_rt[pred_correct_mask].mean()
        else:
            pred_gap = zero
        if human_correct_mask.any() and human_error_mask.any():
            human_gap = true_rt[human_error_mask].mean() - true_rt[human_correct_mask].mean()
        else:
            human_gap = zero
        error_sign_loss = (pred_gap - human_gap).abs()
    else:
        error_sign_loss = zero

    # --- accuracy loss ---
    lambda_accuracy = float(config.get('lambda_accuracy', 0.0))
    if lambda_accuracy > 0.0:
        human_acc = (response_labels == target_labels).float().mean()
        model_acc = (pred_choice == target_labels).float().mean()
        accuracy_loss = (model_acc - human_acc).abs()
    else:
        accuracy_loss = zero

    # --- response NLL ---
    lambda_response_nll = float(config.get('lambda_response_nll', 1.0))
    if lambda_response_nll > 0.0:
        response_nll = F.nll_loss(
            (choice_probs + 1e-8).log(),
            response_labels.long(),
        )
    else:
        response_nll = zero

    # --- RT MSE ---
    lambda_rt_mse = float(config.get('lambda_rt_mse', 1.0))
    if lambda_rt_mse > 0.0:
        rt_mse = F.mse_loss(pred_rt, true_rt)
    else:
        rt_mse = zero

    total = (
        lambda_error_rate * error_rate_loss
        + lambda_error_sign * error_sign_loss
        + lambda_accuracy * accuracy_loss
        + lambda_response_nll * response_nll
        + lambda_rt_mse * rt_mse
    )

    return {
        'loss': total,
        'error_rate_loss': error_rate_loss,
        'error_sign_loss': error_sign_loss,
        'accuracy_loss': accuracy_loss,
        'response_nll': response_nll,
        'rt_mse': rt_mse,
    }


class DiffDecisionMultiClass(Function):
    """Differentiable decision time computation for multi-class decisions.

    Uses implicit-function-theorem gradient: d(crossing_time)/d(trajectory)
    = -1 / dsdt at the crossing point.  The forward returns crossing_time * dt
    so the backward multiplies through by dt for correct chain-rule scaling.

    Batch elements where *no* class crosses the threshold receive zero
    gradient (the clamped fallback index carries no signal).
    """

    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory, dt, max_time):
        mask = trajectory > 0                                 # [B, T, C]
        decision_times = mask.float().argmax(dim=1).float()   # [B, C]
        no_cross_class = ~mask.any(dim=1)                     # [B, C] – class never crossed
        decision_times[no_cross_class] = max_time - 1
        ctx.save_for_backward(dsdt_trajectory, decision_times, no_cross_class)
        ctx.dt = float(dt)
        return decision_times * dt

    @staticmethod
    def backward(ctx, *grad_outputs):
        dsdt_trajectory, decision_times, no_cross_class = ctx.saved_tensors
        dt = ctx.dt
        grads = torch.zeros_like(dsdt_trajectory)
        grad_output = grad_outputs[0]

        decision_indices = decision_times.long()

        batch_indices, class_indices = torch.meshgrid(
            torch.arange(decision_times.size(0), device=decision_times.device),
            torch.arange(decision_times.size(1), device=decision_times.device),
            indexing='ij',
        )

        # dsdt at the detected crossing step for each (batch, class)
        dsdt_cross = dsdt_trajectory[
            batch_indices,
            decision_indices[batch_indices, class_indices],
            class_indices,
        ]
        # clamp near-zero dsdt to avoid extreme / NaN gradients
        safe_dsdt = dsdt_cross.abs().clamp_min(1e-6)

        grads[batch_indices, decision_indices[batch_indices, class_indices], class_indices] = (
            -1.0 / safe_dsdt
        )

        # chain-rule: forward returns decision_times * dt
        grads = grads * (grad_output.unsqueeze(1).expand_as(grads) * dt)

        # A clamped fallback time carries no crossing-time gradient.
        grads = grads.masked_fill(no_cross_class.unsqueeze(1), 0.0)

        return grads, None, None, None


class WongWangMultiClassDecision(nn.Module):
    """
    Multi-class Wong-Wang decision model with RTify.
    
    Implements competing neural populations for decision making.
    Based on: Wong, K. F., & Wang, X. J. (2006). Journal of Neuroscience.
    
    Parameters:
        n_classes: Number of competing choices (4 for LIM: L/R/U/D)
        dt: Time step in ms
        time_steps: Total simulation time steps
        t_stimulus: Duration of stimulus presentation
        normalize_competition: Keep total off-diagonal inhibition approximately
            constant when the number of alternatives changes. Disabled by default
            to preserve existing checkpoints and result packages.
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        dt: int = 10,
        time_steps: int = 500,
        t_stimulus: int = 500,
        normalize_competition: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        
        self.a = nn.Parameter(torch.tensor(270.0, dtype=torch.float32), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(108.0, dtype=torch.float32), requires_grad=False)
        self.d = nn.Parameter(torch.tensor(0.1540, dtype=torch.float32), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.641, dtype=torch.float32), requires_grad=False)
        self.tau_s = nn.Parameter(torch.tensor(100.0, dtype=torch.float32), requires_grad=False)
        
        cross_inhibition = -0.0497
        if normalize_competition and n_classes > 1:
            cross_inhibition /= n_classes - 1
        self.J_matrix = nn.Parameter(
            torch.ones(n_classes, n_classes, dtype=torch.float32) * cross_inhibition,
            requires_grad=True,
        )
        self.J_matrix.data[range(n_classes), range(n_classes)] = 0.2609
        self.J_ext = nn.Parameter(torch.tensor(0.0156, dtype=torch.float32), requires_grad=True)
        self.I_0 = nn.Parameter(torch.tensor(0.3255, dtype=torch.float32), requires_grad=True)
        self.noise_ampa = nn.Parameter(torch.tensor(0.02, dtype=torch.float32), requires_grad=True)
        self.tau_ampa = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        
        self.dt = dt
        self.time_steps = time_steps
        self.t_stimulus = t_stimulus
    
    def _sample_noise(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.n_classes,
            device=device,
            generator=generator,
        )

    def forward(self, input_signal: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Forward pass through Wong-Wang dynamics.
        
        Args:
            input_signal: Input logits [batch, n_classes] or [batch, time, n_classes]
        
        Returns:
            decision_times: Decision time for each class [batch, n_classes]
        """
        batch_size = input_signal.shape[0]
        device = input_signal.device
        
        s = torch.ones(batch_size, self.n_classes, requires_grad=False, device=device) / 10.0
        I_noise = self._sample_noise(batch_size, device, generator=generator) * self.noise_ampa
        
        trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        dsdt_trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        
        for t in range(self.time_steps):
            if t < self.t_stimulus:
                if input_signal.dim() == 3:
                    I = self.J_ext * input_signal[:, t, :]
                else:
                    I = self.J_ext * input_signal
            else:
                I = torch.zeros(batch_size, self.n_classes, requires_grad=False, device=device)
            
            x = torch.matmul(s, self.J_matrix) + self.I_0 + I + I_noise
            
            H = F.relu((self.a * x - self.b) / (1 - torch.exp(-self.d * (self.a * x - self.b)) + 1e-6))
            
            dsdt = - (s / self.tau_s) + (1 - s) * H * self.gamma / 1000.0
            
            I_noise = I_noise.clone() * torch.exp(-self.dt / self.tau_ampa) + \
                self.noise_ampa * torch.sqrt((1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * \
                self._sample_noise(batch_size, device, generator=generator)
            
            s = s.clone() + dsdt * self.dt
            
            trajectory[:, t, :] = s.clone()
            dsdt_trajectory[:, t, :] = dsdt.clone()
        
        decision_times_class = cast(torch.Tensor, DiffDecisionMultiClass.apply(
            trajectory - self.threshold, dsdt_trajectory, self.dt, self.time_steps
        ))
        
        return decision_times_class / 1000.0
    
    def inference(
        self,
        input_signal: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference mode returning trajectory for visualization.
        
        Returns:
            decision_times: Decision time for each class
            trajectory: Full trajectory of neural activity
            threshold: Current threshold value
        """
        batch_size = input_signal.shape[0]
        device = input_signal.device
        
        s = torch.ones(batch_size, self.n_classes, requires_grad=False, device=device) / 10.0
        I_noise = self._sample_noise(batch_size, device, generator=generator) * self.noise_ampa
        
        trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        dsdt_trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        
        for t in range(self.time_steps):
            if t < self.t_stimulus:
                if input_signal.dim() == 3:
                    I = self.J_ext * input_signal[:, t, :]
                else:
                    I = self.J_ext * input_signal
            else:
                I = torch.zeros(batch_size, self.n_classes, requires_grad=False, device=device)
            
            x = torch.matmul(s, self.J_matrix) + self.I_0 + I + I_noise
            H = F.relu((self.a * x - self.b) / (1 - torch.exp(-self.d * (self.a * x - self.b)) + 1e-6))
            dsdt = - (s / self.tau_s) + (1 - s) * H * self.gamma / 1000.0
            
            I_noise = I_noise.clone() * torch.exp(-self.dt / self.tau_ampa) + \
                self.noise_ampa * torch.sqrt((1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * \
                self._sample_noise(batch_size, device, generator=generator)
            
            s = s.clone() + dsdt * self.dt
            
            trajectory[:, t, :] = s.clone()
            dsdt_trajectory[:, t, :] = dsdt.clone()
        
        decision_times_class = cast(torch.Tensor, DiffDecisionMultiClass.apply(
            trajectory - self.threshold, dsdt_trajectory, self.dt, self.time_steps
        ))
        
        return decision_times_class / 1000.0, trajectory, self.threshold


class VGGFeatureExtractor(nn.Module):
    """
    VGG16-based feature extractor for LIM task.
    
    Uses pretrained VGG16 with custom classification head.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_features: bool = False,
        n_classes: int = 4,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG.
        
        Args:
            x: Input images [batch, 3, H, W]
        
        Returns:
            logits: Classification logits [batch, n_classes]
        """
        x = self.features(x)
        
        original_device = x.device
        if original_device.type == 'mps':
            x = self.avgpool(x.cpu())
            x = x.to(original_device)
        else:
            x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def mc_forward(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC Dropout forward — keeps dropout active at inference time.

        Multiple stochastic forward passes with dropout enabled produce a
        distribution of logits.  The variance captures epistemic uncertainty
        (model uncertainty about its own prediction).

        Args:
            x: Input images [batch, 3, H, W]
            n_samples: Number of MC samples (default 30)
            seed: RNG seed for reproducibility

        Returns:
            mean:   Mean logits across samples    [batch, n_classes]
            var:    Variance across samples        [batch, n_classes]
            samples: All raw samples              [n_samples, batch, n_classes]
        """
        was_training = self.training
        self.train()  # keep dropout active
        with torch.no_grad():
            g = torch.Generator(device=x.device)
            if seed >= 0:
                g.manual_seed(seed)
            samples = []
            for _ in range(n_samples):
                # Re-seed per sample to get independent dropout masks
                samples.append(self.forward(x))
            samples = torch.stack(samples, dim=0)  # [S, B, C]
        if not was_training:
            self.eval()
        mean = samples.mean(dim=0)        # [B, C]
        var = samples.var(dim=0)          # [B, C]
        return mean, var, samples


class WWWrapper(nn.Module):
    """
    Wrapper for Wong-Wang decision module.
    
    Applies fixed scaling to logits before Wong-Wang dynamics.
    """
    
    def __init__(self, n_classes: int = 4, dt: int = 10, time_steps: int = 500):
        super().__init__()
        
        self.ww = WongWangMultiClassDecision(n_classes=n_classes, dt=dt, time_steps=time_steps)
        
        self.register_buffer('scale', torch.tensor(0.25, dtype=torch.float32))
        self.t0_seconds = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Wong-Wang.
        
        Args:
            logits: Classification logits [batch, n_classes]
        
        Returns:
            decision_times: Decision time for each class [batch, n_classes]
        """
        # Apply fixed scaling and ReLU to ensure positive input
        x = apply_stage2_input_transform(logits, cast(torch.Tensor, self.scale))
        decision_times = self.ww(x)
        return decision_times


class VGGWongWangLIM(nn.Module):
    """
    Complete VGG-WongWang model for LIM task.
    
    Two-stage model:
    - Stage 1: VGG feature extraction + classification
    - Stage 2: Wong-Wang decision module for RT prediction
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_features: bool = False,
        n_classes: int = 4,
        dropout_rate: float = 0.5,
        dt: int = 10,
        time_steps: int = 500
    ):
        super().__init__()
        
        self.feature_extractor = VGGFeatureExtractor(
            pretrained=pretrained,
            freeze_features=freeze_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate
        )
        
        self.ww_wrapper = WWWrapper(n_classes=n_classes, dt=dt, time_steps=time_steps)
    
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through complete model.
        
        Args:
            x: Input images [batch, 3, H, W]
            return_logits: Whether to return intermediate logits
        
        Returns:
            decision_logits: Classification logits based on decision time
            decision_time: Final decision time (min across classes)
            rt_pred: Predicted RT (linearly transformed decision time)
        """
        logits = self.feature_extractor(x)
        
        decision_times = self.ww_wrapper(logits)
        
        final_decision_time, decision_direction = decision_times.min(dim=1)
        
        # Use raw decision time as RT (already in seconds)
        rt_pred = final_decision_time
        
        if return_logits:
            return logits, decision_times, final_decision_time, rt_pred
        
        return logits, final_decision_time, rt_pred
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification logits only (for stage 1 training)."""
        return self.feature_extractor(x)
    
    def get_decision_times(self, logits: torch.Tensor) -> torch.Tensor:
        """Get decision times from logits (for stage 2 training)."""
        return self.ww_wrapper(logits)
    
    def freeze_vgg(self):
        """Freeze VGG feature extractor for stage 2 training."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_vgg(self):
        """Unfreeze VGG feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


class NegativePearsonCorrelationLoss(nn.Module):
    """Negative Pearson correlation loss for RT fitting."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = -torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return cost


def create_model(
    pretrained: bool = True,
    freeze_features: bool = False,
    n_classes: int = 4,
    **kwargs
) -> VGGWongWangLIM:
    """Create VGG-WongWang model."""
    return VGGWongWangLIM(
        pretrained=pretrained,
        freeze_features=freeze_features,
        n_classes=n_classes,
        **kwargs
    )


if __name__ == '__main__':
    model = VGGWongWangLIM(pretrained=False, n_classes=4)
    x = torch.randn(2, 3, 128, 128)
    logits, decision_time, rt_pred = model(x)
    print(
        {
            "input_shape": list(x.shape),
            "logits_shape": list(logits.shape),
            "decision_time_shape": list(decision_time.shape),
            "rt_prediction_shape": list(rt_pred.shape),
        }
    )

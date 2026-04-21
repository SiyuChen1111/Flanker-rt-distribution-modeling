"""Stage-2 backend: noisy recurrent accumulator (AccumulatorRaceDecisionV2) on cached logits.

This module provides reusable helpers for:
  - fitting the Stage-2 accumulator head from cached Stage-1 logits
  - evaluating fitted parameters on held-out cached logits

It is intentionally scoped to the bounded single-subject feasibility workflow.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from train_age_groups_efficient import compute_human_stats_from_rts, evaluate_joint_behavior, to_jsonable
from vgg_accumulator_rnn_v2 import AccumulatorRaceDecisionV2


COUPLED_CHOICE_READOUT = "first_crosser_coupled.v1"
COUPLED_MAX_EXTRA_STEPS = 4000


def set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_torch_generator(seed: int, device: torch.device) -> Optional[torch.Generator]:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    try:
        generator = torch.Generator(device=generator_device)
    except (RuntimeError, TypeError):
        if generator_device != "cpu":
            generator = torch.Generator()
        else:
            return None
    generator.manual_seed(int(seed))
    return generator


def _ranking_key(results: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    # Mirrors the accumulator RNN training script: stable, monotone preference ordering.
    return (
        1.0,
        float(results.get("rt_shape_score", 0.0)),
        float(results.get("response_agreement", 0.0)),
        float(results.get("congruency_score", 0.0)),
        float(results.get("mean_median_score", 0.0)),
        float(results.get("accuracy_score", 0.0)),
    )


def _decision_time_indices(
    decision_times: torch.Tensor,
    dt_ms: int,
    max_time: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_choice_idx = decision_times.argmin(dim=1)
    chosen_time = decision_times[torch.arange(decision_times.size(0), device=decision_times.device), pred_choice_idx]
    time_idx = torch.clamp((chosen_time * 1000.0 / float(dt_ms)).long(), min=0, max=max_time - 1)
    return pred_choice_idx, time_idx


def pick_choice_state(
    traj: torch.Tensor,
    decision_times: torch.Tensor,
    dt_ms: int,
    *,
    choice_window: int = 3,
) -> torch.Tensor:
    """Return a (window-averaged) choice state for each sample.

    We first compute the predicted decision time index (argmin over boundary times),
    then average trajectory states over a short backward window:

        mean(traj[:, t-k : t+1, :])

    where k = choice_window. Early indices clamp the window start to 0.

    Args:
        traj: [batch, time, n_classes]
        decision_times: [batch, n_classes] boundary-crossing times in seconds
        dt_ms: timestep size in milliseconds
        choice_window: backward window size (k). k=3 averages over 4 steps.

    Returns:
        choice_state: [batch, n_classes]
    """

    if int(choice_window) < 0:
        raise ValueError("choice_window must be >= 0")

    _pred_choice_idx, time_idx = _decision_time_indices(decision_times, dt_ms, traj.size(1))

    # Average over [start, ..., end] using a per-sample prefix-sum window.
    end_idx = time_idx
    start_idx = torch.clamp(end_idx - int(choice_window), min=0)

    cumsum = traj.cumsum(dim=1)
    n_classes = traj.size(2)

    end_gather = end_idx.view(-1, 1, 1).expand(-1, 1, n_classes)
    sum_end = cumsum.gather(1, end_gather).squeeze(1)

    start_minus1 = start_idx - 1
    start_minus1_clamped = torch.clamp(start_minus1, min=0)
    start_gather = start_minus1_clamped.view(-1, 1, 1).expand(-1, 1, n_classes)
    sum_before = cumsum.gather(1, start_gather).squeeze(1)
    sum_before = torch.where(start_idx.view(-1, 1) == 0, torch.zeros_like(sum_before), sum_before)

    window_sum = sum_end - sum_before
    denom = (end_idx - start_idx + 1).to(dtype=traj.dtype).view(-1, 1)
    denom = torch.clamp(denom, min=1.0)
    return window_sum / denom


def pick_choice_state_gaussian(
    traj: torch.Tensor,
    decision_times: torch.Tensor,
    dt_ms: int,
    *,
    radius_steps: int = 6,
    sigma_steps: float = 2.0,
) -> torch.Tensor:
    if int(radius_steps) < 0:
        raise ValueError("radius_steps must be >= 0")
    if float(sigma_steps) <= 0:
        raise ValueError("sigma_steps must be > 0")

    _pred_choice_idx, time_idx = _decision_time_indices(decision_times, dt_ms, traj.size(1))
    offsets = torch.arange(-int(radius_steps), int(radius_steps) + 1, device=traj.device)
    gather_idx = torch.clamp(time_idx.unsqueeze(1) + offsets.unsqueeze(0), min=0, max=traj.size(1) - 1)
    gathered = traj.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, traj.size(2)))
    weights = torch.exp(-0.5 * (offsets.to(dtype=traj.dtype) / float(sigma_steps)) ** 2)
    weights = weights / torch.clamp(weights.sum(), min=1e-6)
    return (gathered * weights.view(1, -1, 1)).sum(dim=1)


def compute_choice_logits(
    *,
    traj: torch.Tensor,
    decision_times: torch.Tensor,
    threshold_t: torch.Tensor,
    dt_ms: int,
    choice_temperature: float,
    choice_readout: str,
    choice_window: int,
    gaussian_radius_steps: int,
    gaussian_sigma_steps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    temperature = max(float(choice_temperature), 1e-6)
    crossed_mask = (traj > threshold_t.view(1, 1, 1)).any(dim=1)

    if choice_readout == "windowed_state_at_decision.v1":
        choice_state = pick_choice_state(traj, decision_times, dt_ms, choice_window=int(choice_window))
        choice_logits = choice_state / temperature
    elif choice_readout == "first_crosser_coupled.v1":
        large_time = torch.full_like(decision_times, fill_value=float(traj.size(1) * dt_ms * 2) / 1000.0)
        true_crossing_times = torch.where(crossed_mask, decision_times, large_time)
        choice_state = -true_crossing_times
        choice_logits = choice_state / temperature
    elif choice_readout == "threshold_relative_windowed_state_at_decision.v1":
        choice_state = pick_choice_state(traj, decision_times, dt_ms, choice_window=int(choice_window))
        choice_logits = (choice_state - threshold_t) / temperature
    elif choice_readout == "gaussian_pooled_state_at_decision.v1":
        choice_state = pick_choice_state_gaussian(
            traj,
            decision_times,
            dt_ms,
            radius_steps=int(gaussian_radius_steps),
            sigma_steps=float(gaussian_sigma_steps),
        )
        choice_logits = choice_state / temperature
    else:
        raise ValueError(f"UNKNOWN_CHOICE_READOUT: {choice_readout}")

    return choice_state, choice_logits


def evaluate_model(
    *,
    model: AccumulatorRaceDecisionV2,
    cached: Dict[str, np.ndarray],
    human_stats: Dict[str, float],
    device: str,
    choice_temperature: float,
    eval_seed: Optional[int] = None,
    choice_readout: str = "windowed_state_at_decision.v1",
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    model.eval()
    logits_tensor = torch.tensor(cached["logits"], dtype=torch.float32, device=device)
    generator = None
    if eval_seed is not None:
        generator = build_torch_generator(int(eval_seed), logits_tensor.device)
    coupled_mode = str(choice_readout) == COUPLED_CHOICE_READOUT

    with torch.no_grad():
        decision_times, traj, threshold_t = model.rollout(
            logits_tensor,
            generator=generator,
            ensure_crossing=bool(coupled_mode),
            max_extra_steps=COUPLED_MAX_EXTRA_STEPS,
            require_crossing=bool(coupled_mode),
        )
        _choice_state, choice_logits_t = compute_choice_logits(
            traj=traj,
            decision_times=decision_times,
            threshold_t=torch.as_tensor(threshold_t, device=traj.device, dtype=traj.dtype),
            dt_ms=model.dt,
            choice_temperature=float(choice_temperature),
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
        )
        pred_choice = choice_logits_t.argmax(dim=1).cpu().numpy()
        batch_idx = torch.arange(decision_times.size(0), device=decision_times.device)
        pred_rt_t = decision_times[batch_idx, torch.as_tensor(pred_choice, device=decision_times.device, dtype=torch.long)]
        pred_rt = pred_rt_t.cpu().numpy()
        choice_logits = choice_logits_t.cpu().numpy()
        decision_times_np = decision_times.cpu().numpy()
        traj_np = traj.cpu().numpy().astype(np.float32)

    results = evaluate_joint_behavior(
        pred_rt=pred_rt,
        pred_choice=pred_choice,
        true_rt=cached["rts"],
        target_labels=cached["target_labels"],
        response_labels=cached["response_labels"],
        congruency=cached["congruency"],
        human_stats=human_stats,
        rt_shape_focus=True,
    )
    predictions = {
        "pred_rt": pred_rt.astype(np.float32),
        "pred_choice": pred_choice.astype(np.int64),
        "choice_logits": choice_logits.astype(np.float32),
        "decision_times_class": decision_times_np.astype(np.float32),
        "crossed_mask_class": (traj > torch.as_tensor(threshold_t, device=traj.device, dtype=traj.dtype).view(1, 1, 1)).any(dim=1).detach().cpu().numpy().astype(bool),
        "traj": traj_np,
        "threshold": np.array(float(torch.as_tensor(threshold_t).detach().cpu().item()), dtype=np.float32),
    }
    return results, predictions


def train_with_scale(
    *,
    scale: float,
    time_steps: int,
    train_cached: Dict[str, np.ndarray],
    eval_cached: Dict[str, np.ndarray],
    eval_human_stats: Dict[str, float],
    epochs: int,
    device: str,
    noise_std: float,
    threshold: float,
    choice_temperature: float,
    rt_loss_weight: float,
    response_loss_weight: float,
    accuracy_calib_weight: float,
    congruency_loss_weight: float,
    learning_rate: float,
    batch_size: int,
    train_seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
    choice_readout: str = "windowed_state_at_decision.v1",
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
    competition_mix: float = 0.0,
    log_prefix: str = "",
) -> Dict[str, Any]:
    model = AccumulatorRaceDecisionV2(
        n_classes=4,
        dt=10,
        time_steps=int(time_steps),
        threshold=float(threshold),
        noise_std=float(noise_std),
        competition_mix=float(competition_mix),
    )
    # Scale search is treated as a hyperparameter for this backend.
    model.input_scale.data.fill_(float(scale))
    model.input_scale.requires_grad_(False)
    model = model.to(device)

    if train_seed is not None:
        set_random_seed(int(train_seed))

    logits_tensor = torch.tensor(train_cached["logits"], dtype=torch.float32)
    rts_tensor = torch.tensor(train_cached["rts"], dtype=torch.float32)
    response_tensor = torch.tensor(train_cached["response_labels"], dtype=torch.long)
    congruency_tensor = torch.tensor(train_cached["congruency"], dtype=torch.long)
    target_tensor = torch.tensor(train_cached["target_labels"], dtype=torch.long)

    dataset = TensorDataset(logits_tensor, rts_tensor, response_tensor, congruency_tensor, target_tensor)
    dataloader_generator = torch.Generator()
    if train_seed is not None:
        dataloader_generator.manual_seed(int(train_seed))
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=dataloader_generator)

    optimizer = Adam(model.parameters(), lr=float(learning_rate))
    mse = torch.nn.MSELoss()

    checkpoint_history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, float, float, float]] = None

    for epoch in range(int(epochs)):
        model.train()
        coupled_mode = str(choice_readout) == COUPLED_CHOICE_READOUT
        for batch_logits, batch_rt, batch_response, batch_cong, batch_target in dataloader:
            batch_logits = batch_logits.to(device)
            batch_rt = batch_rt.to(device)
            batch_response = batch_response.to(device)
            batch_cong = batch_cong.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            decision_times, _traj, _threshold_t = model.rollout(
                batch_logits,
                ensure_crossing=bool(coupled_mode),
                max_extra_steps=COUPLED_MAX_EXTRA_STEPS,
                require_crossing=bool(coupled_mode),
            )
            _choice_state, choice_logits = compute_choice_logits(
                traj=_traj,
                decision_times=decision_times,
                threshold_t=torch.as_tensor(_threshold_t, device=_traj.device, dtype=_traj.dtype),
                dt_ms=model.dt,
                choice_temperature=float(choice_temperature),
                choice_readout=str(choice_readout),
                choice_window=int(choice_window),
                gaussian_radius_steps=int(gaussian_radius_steps),
                gaussian_sigma_steps=float(gaussian_sigma_steps),
            )
            pred_choice_train = choice_logits.argmax(dim=1)
            pred_rt = decision_times[torch.arange(decision_times.size(0), device=decision_times.device), pred_choice_train]
            choice_loss = F.cross_entropy(choice_logits, batch_response)
            rt_loss = mse(pred_rt, batch_rt)

            if float(accuracy_calib_weight) > 0:
                probs = F.softmax(choice_logits, dim=1)
                p_correct_batch = probs.gather(1, batch_target.unsqueeze(1)).mean()
                human_acc_batch = (batch_response == batch_target).float().mean()
                accuracy_calib_loss = (p_correct_batch - human_acc_batch).pow(2)
            else:
                accuracy_calib_loss = torch.tensor(0.0, device=device)

            if float(congruency_loss_weight) > 0 and (batch_cong == 0).any() and (batch_cong == 1).any():
                mean_cong = pred_rt[batch_cong == 0].mean()
                mean_incong = pred_rt[batch_cong == 1].mean()
                congruency_loss = F.relu(0.01 - (mean_incong - mean_cong))
            else:
                congruency_loss = torch.tensor(0.0, device=device)

            loss = (
                float(response_loss_weight) * choice_loss
                + float(rt_loss_weight) * rt_loss
                + float(accuracy_calib_weight) * accuracy_calib_loss
                + float(congruency_loss_weight) * congruency_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        results, predictions = evaluate_model(
            model=model,
            cached=eval_cached,
            human_stats=eval_human_stats,
            device=device,
            choice_temperature=float(choice_temperature),
            eval_seed=eval_seed,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
        )
        key = _ranking_key(results)
        history_item = {
            "epoch": int(epoch + 1),
            "selected": False,
            "ranking_key": list(key),
            "metrics": {
                "behavior_optimal_score": float(results["behavior_optimal_score"]),
                "rt_shape_score": float(results["rt_shape_score"]),
                "quantile_score": float(results["quantile_score"]),
                "response_agreement": float(results["response_agreement"]),
                "mean_median_score": float(results["mean_median_score"]),
                "accuracy_score": float(results["accuracy_score"]),
                "congruency_score": float(results["congruency_score"]),
                "error_minus_correct_rt": float(results["error_minus_correct_rt"]),
                "human_error_minus_correct_rt": float(results["human_error_minus_correct_rt"]),
                "model_congruency_rt_gap": float(results["model_congruency_rt_gap"]),
                "human_congruency_rt_gap": float(results["human_congruency_rt_gap"]),
                "pred_mean": float(results["pred_mean"]),
                "pred_median": float(results["pred_median"]),
                "pred_q95": float(results["pred_q95"]),
                "pred_q99": float(results["pred_q99"]),
            },
        }
        checkpoint_history.append(history_item)

        print(
            f"{log_prefix}Eval epoch {epoch + 1:02d}: behavior={results['behavior_optimal_score']:.4f} "
            f"rt_shape={results['rt_shape_score']:.4f} resp_agree={results['response_agreement']:.4f} "
            f"cong_gap={results['model_congruency_rt_gap']:.4f}/{results['human_congruency_rt_gap']:.4f} "
            f"err-corr={results['error_minus_correct_rt']:.4f}"
        )

        if best_key is None or key > best_key:
            best_key = key
            best = {
                "epoch": int(epoch + 1),
                "results": results,
                "params": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
                "predictions": predictions,
                "selection_details": {
                    "best_epoch": int(epoch + 1),
                    "ranking_key": list(key),
                    "checkpoint_history": checkpoint_history,
                },
            }

    if best is None:
        raise RuntimeError("No best checkpoint found")
    for item in best["selection_details"]["checkpoint_history"]:
        item["selected"] = item["epoch"] == best["epoch"]
    return best


def fit_stage2_accumrnn_from_logits(
    *,
    age_group: str,
    output_dir: str,
    human_stats: Dict[str, float],
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
    device: str = "cpu",
    scales: Optional[np.ndarray] = None,
    epochs: int = 8,
    time_steps: Optional[int] = None,
    time_steps_factor: float = 1.0,
    threshold: float = 0.5,
    noise_std: float = 0.02,
    choice_temperature: float = 0.05,
    rt_loss_weight: float = 2.0,
    response_loss_weight: float = 1.0,
    accuracy_calib_weight: float = 0.0,
    congruency_loss_weight: float = 0.10,
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    choice_readout: str = "windowed_state_at_decision.v1",
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
    competition_mix: float = 0.0,
    random_seed: Optional[int] = None,
    eval_random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if scales is None:
        scales = np.array([0.10, 0.30, 0.50], dtype=np.float32)
    if np.asarray(scales).size == 0:
        raise ValueError("SCALES_EMPTY")

    if time_steps is None:
        # Match WW convention: dt=10ms => 100 steps per second.
        inferred = int(np.ceil(float(human_stats["percentile_99"]) * 100.0 * float(time_steps_factor)))
        time_steps = max(40, int(inferred))
    time_steps = int(time_steps)

    print(f"\n{'='*60}")
    print(f"[accumrnn] Processing cached logits: {age_group}")
    print(f"{'='*60}")
    print(f"Human stats: Mean={human_stats['mean']:.3f}s, Median={human_stats['median']:.3f}s")
    print(f"Device: {device}")
    print(f"Epochs: {int(epochs)}")
    print(f"Time steps: {time_steps} (max RT={time_steps * 10 / 1000:.2f}s)")
    print(f"Scale grid: {[float(x) for x in np.asarray(scales).tolist()]}")
    print(f"Accumulator noise_std={float(noise_std)} threshold={float(threshold)}")

    eval_human_stats = compute_human_stats_from_rts(np.asarray(test_cached["rts"], dtype=np.float32))
    best_overall: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, float, float, float]] = None
    results_list: List[Dict[str, Any]] = []

    for idx, scale in enumerate(np.asarray(scales, dtype=np.float32), start=1):
        log_prefix = f"[accumrnn {age_group} scale {idx}/{len(scales)}] "
        scale_train_seed = None if random_seed is None else int(random_seed + idx * 1000)
        scale_eval_seed = None if eval_random_seed is None else int(eval_random_seed)
        best = train_with_scale(
            scale=float(scale),
            time_steps=time_steps,
            train_cached=train_cached,
            eval_cached=test_cached,
            eval_human_stats=eval_human_stats,
            epochs=int(epochs),
            device=device,
            noise_std=float(noise_std),
            threshold=float(threshold),
            choice_temperature=float(choice_temperature),
            rt_loss_weight=float(rt_loss_weight),
            response_loss_weight=float(response_loss_weight),
            accuracy_calib_weight=float(accuracy_calib_weight),
            congruency_loss_weight=float(congruency_loss_weight),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            train_seed=scale_train_seed,
            eval_seed=scale_eval_seed,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
            log_prefix=log_prefix,
        )
        key = tuple(best["selection_details"]["ranking_key"])  # type: ignore[assignment]
        results_list.append({"scale": float(scale), "selection_details": best["selection_details"]})
        if best_key is None or key > best_key:
            best_key = key
            best_overall = {"scale": float(scale), **best}

    if best_overall is None:
        raise RuntimeError("No overall best checkpoint found")

    os.makedirs(output_dir, exist_ok=True)
    config_payload: Dict[str, Any] = {
        "schema_version": "stage2_accumrnn.best_config.v1",
        "stage2_backend": "AccumulatorRaceDecisionV2",
        "model_family": "VGGAccumulatorRNNLIMV2",
        "choice_readout": str(choice_readout),
        "choice_window": int(choice_window),
        "gaussian_radius_steps": int(gaussian_radius_steps),
        "gaussian_sigma_steps": float(gaussian_sigma_steps),
        "competition_mix": float(competition_mix),
        "age_group": str(age_group),
        "scale": float(best_overall["scale"]),
        "best_epoch": int(best_overall["epoch"]),
        "score": float(best_overall["results"]["behavior_optimal_score"]),
        "time_steps": int(time_steps),
        "dt_ms": 10,
        "epochs": int(epochs),
        "threshold": float(threshold),
        "noise_std": float(noise_std),
        "choice_temperature": float(choice_temperature),
        "accuracy_calib_weight": float(accuracy_calib_weight),
        "scales": [float(x) for x in np.asarray(scales, dtype=np.float32).tolist()],
        "results": best_overall["results"],
        "selection_details": best_overall["selection_details"],
        "train_random_seed": None if random_seed is None else int(random_seed),
        "eval_random_seed": None if eval_random_seed is None else int(eval_random_seed),
    }

    with open(os.path.join(output_dir, "best_config.json"), "w") as handle:
        json.dump(to_jsonable(config_payload), handle, indent=2)
    np.savez(os.path.join(output_dir, "best_model_params.npz"), **best_overall["params"])

    return {"best_config": config_payload, "best": best_overall, "ranking_summary": results_list}


def infer_predictions_from_params(
    *,
    params: Dict[str, np.ndarray],
    time_steps: int,
    logits: np.ndarray,
    device: str,
    choice_temperature: float,
    random_seed: Optional[int] = None,
    choice_readout: str = "windowed_state_at_decision.v1",
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
    competition_mix: float = 0.0,
) -> Dict[str, np.ndarray]:
    model = AccumulatorRaceDecisionV2(n_classes=4, dt=10, time_steps=int(time_steps), competition_mix=float(competition_mix))
    state = model.state_dict()
    for key in state:
        if key in params:
            state[key] = torch.tensor(params[key])
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    logits_t = torch.tensor(np.asarray(logits, dtype=np.float32), dtype=torch.float32, device=device)
    generator = None
    if random_seed is not None:
        generator = build_torch_generator(int(random_seed), logits_t.device)
    coupled_mode = str(choice_readout) == COUPLED_CHOICE_READOUT
    with torch.no_grad():
        decision_times, traj, threshold_t = model.rollout(
            logits_t,
            generator=generator,
            ensure_crossing=bool(coupled_mode),
            max_extra_steps=COUPLED_MAX_EXTRA_STEPS,
            require_crossing=bool(coupled_mode),
        )
        _choice_state, choice_logits_t = compute_choice_logits(
            traj=traj,
            decision_times=decision_times,
            threshold_t=torch.as_tensor(threshold_t, device=traj.device, dtype=traj.dtype),
            dt_ms=model.dt,
            choice_temperature=float(choice_temperature),
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
        )
        pred_choice = choice_logits_t.argmax(dim=1).cpu().numpy()
        batch_idx = torch.arange(decision_times.size(0), device=decision_times.device)
        pred_rt_t = decision_times[batch_idx, torch.as_tensor(pred_choice, device=decision_times.device, dtype=torch.long)]
        pred_rt = pred_rt_t.cpu().numpy()
        choice_logits = choice_logits_t.cpu().numpy()
        crossed_mask = (traj > torch.as_tensor(threshold_t, device=traj.device, dtype=traj.dtype).view(1, 1, 1)).any(dim=1)
    return {
        "pred_rt": pred_rt.astype(np.float32),
        "pred_choice": pred_choice.astype(np.int64),
        "choice_logits": choice_logits.astype(np.float32),
        "decision_times_class": decision_times.cpu().numpy().astype(np.float32),
        "crossed_mask_class": crossed_mask.detach().cpu().numpy().astype(bool),
        "traj": traj.cpu().numpy().astype(np.float32),
        "threshold": np.array(float(torch.as_tensor(threshold_t).detach().cpu().item()), dtype=np.float32),
    }


def evaluate_cached_stage2_accumrnn_params(
    *,
    params: Dict[str, np.ndarray],
    time_steps: int,
    cached: Dict[str, np.ndarray],
    device: str,
    choice_temperature: float,
    random_seed: Optional[int] = None,
    rt_shape_focus: bool = True,
    choice_readout: str = "windowed_state_at_decision.v1",
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
    competition_mix: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    predictions = infer_predictions_from_params(
        params=params,
        time_steps=int(time_steps),
        logits=cached["logits"],
        device=device,
        choice_temperature=float(choice_temperature),
        random_seed=random_seed,
        choice_readout=str(choice_readout),
        choice_window=int(choice_window),
        gaussian_radius_steps=int(gaussian_radius_steps),
        gaussian_sigma_steps=float(gaussian_sigma_steps),
        competition_mix=float(competition_mix),
    )
    metrics = evaluate_joint_behavior(
        pred_rt=predictions["pred_rt"],
        pred_choice=predictions["pred_choice"],
        true_rt=cached["rts"],
        target_labels=cached["target_labels"],
        response_labels=cached["response_labels"],
        congruency=cached["congruency"],
        human_stats=compute_human_stats_from_rts(np.asarray(cached["rts"], dtype=np.float32)),
        rt_shape_focus=bool(rt_shape_focus),
    )
    return predictions, metrics

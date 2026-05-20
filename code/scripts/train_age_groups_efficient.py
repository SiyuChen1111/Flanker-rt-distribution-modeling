"""
Efficient Stage 1 + Stage 2 training for age groups.

This script:
1. Uses existing Stage 1 model to extract logits for age groups
2. Trains Stage 2 Wong-Wang model with automatic scale search
3. Compares parameters between age groups
"""

import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as transforms
from scipy import stats
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any, List
from time import perf_counter

from project_paths import (
    CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT,
    CHECKPOINTS_AGE_GROUPS_ROOT,
    CHECKPOINTS_TEST_ROOT,
    DATA_AGE_GROUPS_ROOT,
    rel_to_root,
)

from vgg_wongwang_lim import (
    VGGFeatureExtractor,
    WWWrapper,
    apply_stage2_input_transform,
    build_dynamic_stage2_input,
    compute_behavioral_losses,
    compute_legacy_choice_logits,
    compute_rt_readout,
)


DIRECTION_MAP = {'L': 0, 'R': 1, 'U': 2, 'D': 3}


def to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def compute_human_stats_from_rts(rts: np.ndarray) -> Dict[str, float]:
    rts = np.asarray(rts, dtype=np.float32)
    return {
        'mean': float(rts.mean()),
        'median': float(np.median(rts)),
        'skewness': float(stats.skew(rts)),
        'min': float(rts.min()),
        'max': float(rts.max()),
        'percentile_99': float(np.quantile(rts, 0.99)),
    }


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_torch_generator(seed: int, device: torch.device) -> Optional[torch.Generator]:
    if device.type not in {'cpu', 'cuda'}:
        return None
    generator_device = 'cuda' if device.type == 'cuda' else 'cpu'
    try:
        generator = torch.Generator(device=generator_device)
    except (RuntimeError, TypeError):
        if generator_device != 'cpu':
            generator = torch.Generator()
        else:
            return None
    generator.manual_seed(int(seed))
    return generator


def subset_cached_stage2_inputs(cached: Dict[str, np.ndarray], fraction: float, seed: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    total_rows = len(cached['logits'])
    subset_rows = max(1, int(np.ceil(total_rows * fraction)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total_rows, size=subset_rows, replace=False))
    subset = {key: value[indices] for key, value in cached.items()}
    return subset, indices


def attach_flanker_labels_from_csv(cached: Dict[str, np.ndarray], csv_path: str) -> Dict[str, np.ndarray]:
    if 'flanker_labels' in cached:
        return cached
    df = pd.read_csv(csv_path)
    if len(df) != len(cached['logits']):
        raise ValueError(f"Length mismatch while attaching flanker_labels: csv={len(df)} logits={len(cached['logits'])}")
    flanker_labels = df['flanker_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
    enriched = dict(cached)
    enriched['flanker_labels'] = flanker_labels
    return enriched


def subset_smoke_eval_inputs(
    cached: Dict[str, np.ndarray],
    fraction: float,
    seed: int,
    mode: str = 'random',
    min_errors: int = 0,
    balance_congruency: bool = False,
    max_trials: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    total_rows = int(len(cached['logits']))
    requested_rows = max(1, int(np.ceil(total_rows * fraction)))
    if max_trials is not None and max_trials > 0:
        requested_rows = min(requested_rows, int(max_trials))
    subset_rows = min(total_rows, requested_rows)

    response_labels = np.asarray(cached['response_labels'])
    target_labels = np.asarray(cached['target_labels'])
    congruency = np.asarray(cached['congruency'])
    error_mask = response_labels != target_labels
    congruent_mask = congruency == 0
    incongruent_mask = congruency == 1

    available_error_indices = np.flatnonzero(error_mask)
    available_congruent_indices = np.flatnonzero(congruent_mask)
    available_incongruent_indices = np.flatnonzero(incongruent_mask)

    if mode == 'random':
        subset, indices = subset_cached_stage2_inputs(cached, subset_rows / max(total_rows, 1), seed)
    elif mode == 'behavior_balanced':
        rng = np.random.default_rng(seed)
        selected: List[int] = []
        selected_set = set()

        def add_candidates(candidates: np.ndarray, limit: int):
            if limit <= 0 or len(candidates) == 0 or len(selected) >= subset_rows:
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
                if len(selected) >= subset_rows or added >= limit:
                    break

        requested_error_rows = min(max(int(min_errors), 0), subset_rows)
        add_candidates(available_error_indices, requested_error_rows)

        if balance_congruency and len(selected) < subset_rows:
            selected_array = np.array(selected, dtype=np.int64)
            has_congruent = bool(selected) and bool(np.any(congruent_mask[selected_array]))
            has_incongruent = bool(selected) and bool(np.any(incongruent_mask[selected_array]))
            if not has_congruent:
                add_candidates(available_congruent_indices, 1)
            if not has_incongruent:
                add_candidates(available_incongruent_indices, 1)

        if len(selected) < subset_rows:
            remaining = np.setdiff1d(np.arange(total_rows, dtype=np.int64), np.array(selected, dtype=np.int64), assume_unique=False)
            add_candidates(remaining, subset_rows - len(selected))

        indices = np.sort(np.array(selected[:subset_rows], dtype=np.int64))
        subset = {key: value[indices] for key, value in cached.items()}
    else:
        raise ValueError(f"Unknown smoke_eval_mode: {mode}")

    actual_error_rows = int(np.sum(subset['response_labels'] != subset['target_labels']))
    actual_congruent_rows = int(np.sum(subset['congruency'] == 0))
    actual_incongruent_rows = int(np.sum(subset['congruency'] == 1))
    required_error_rows = min(max(int(min_errors), 0), subset_rows)
    error_constraint_satisfied = actual_error_rows >= required_error_rows
    congruency_constraint_satisfied = (
        actual_congruent_rows > 0 and actual_incongruent_rows > 0
        if balance_congruency and subset_rows >= 2
        else True
    )
    subset_meta = {
        'subset_mode': mode,
        'total_eval_trials': total_rows,
        'requested_eval_trials': requested_rows,
        'selected_eval_trials': int(len(indices)),
        'requested_min_errors': int(min_errors),
        'actual_human_error_trials': actual_error_rows,
        'congruent_trials': actual_congruent_rows,
        'incongruent_trials': actual_incongruent_rows,
        'available_human_error_trials': int(len(available_error_indices)),
        'available_congruent_trials': int(len(available_congruent_indices)),
        'available_incongruent_trials': int(len(available_incongruent_indices)),
        'eval_sampling_seed': int(seed),
        'requested_balance_congruency': bool(balance_congruency),
        'requested_max_trials': None if max_trials is None else int(max_trials),
        'error_constraint_satisfied': bool(error_constraint_satisfied),
        'congruency_constraint_satisfied': bool(congruency_constraint_satisfied),
        'balance_constraints_satisfied': bool(error_constraint_satisfied and congruency_constraint_satisfied),
        'selected_indices': indices.tolist(),
    }
    return subset, indices, subset_meta


class StimulusDataset(Dataset):
    """Dataset for loading pre-generated stimulus images."""
    
    def __init__(self, csv_path: str, image_size: int = 128):
        """
        Args:
            csv_path: Path to CSV file with stimulus_image_path column
            image_size: Image size for resizing
        """
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size
        
        # Load RT stats for normalization
        data_dir = os.path.dirname(csv_path)
        with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
            self.rt_stats = json.load(f)
        
        # Calculate log RT for normalization
        self.log_rt_min = np.log(self.rt_stats['min'] + 0.001)
        self.log_rt_max = np.log(self.rt_stats['max'])
        self.log_rt_range = self.log_rt_max - self.log_rt_min
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_paths = self.data['stimulus_image_path'].astype(str).to_numpy()
        self.rts = (self.data['response_time'].to_numpy(dtype=np.float32) / 1000.0)
        self.rt_normalized = ((np.log(self.rts + 0.001) - self.log_rt_min) / self.log_rt_range).astype(np.float32)
        self.target_labels = self.data['target_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.response_labels = self.data['response_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.flanker_labels = self.data['flanker_direction'].map(lambda x: DIRECTION_MAP[x]).to_numpy(dtype=np.int64)
        self.congruency = (self.target_labels != self.flanker_labels).astype(np.int64)
        self.correct = (self.target_labels == self.response_labels).astype(np.float32)
        self.missing_image_paths = []

        unique_paths = sorted(set(self.image_paths.tolist()))
        self.image_cache = {}
        zero_image = torch.zeros(3, self.image_size, self.image_size)
        for path in unique_paths:
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                self.image_cache[path] = self.transform(image)
            else:
                self.image_cache[path] = zero_image
                self.missing_image_paths.append(path)
        self.missing_image_count = int(len(self.missing_image_paths))
        self.missing_image_rate = float(self.missing_image_count / max(len(unique_paths), 1))

        print(f"Loaded {len(self.data)} samples from {csv_path} with {len(self.image_cache)} cached images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        return {
            'image': self.image_cache[image_path],
            'rt': torch.tensor(self.rts[idx], dtype=torch.float32),
            'rt_normalized': torch.tensor(self.rt_normalized[idx], dtype=torch.float32),
            'target_label': torch.tensor(self.target_labels[idx], dtype=torch.long),
            'response_label': torch.tensor(self.response_labels[idx], dtype=torch.long),
            'flanker_label': torch.tensor(self.flanker_labels[idx], dtype=torch.long),
            'congruency': torch.tensor(self.congruency[idx], dtype=torch.long),
            'correct': torch.tensor(self.correct[idx], dtype=torch.float32)
        }


def extract_logits(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    """Extract logits using Stage 1 model."""
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            images = batch['image'].to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)


def evaluate_joint_behavior(
    pred_rt: np.ndarray,
    pred_choice: np.ndarray,
    true_rt: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    human_stats: Dict,
    rt_shape_focus: bool = False,
) -> Dict:
    pred_mean = pred_rt.mean()
    pred_median = np.median(pred_rt)
    pred_skewness = stats.skew(pred_rt)
    pred_q90 = float(np.quantile(pred_rt, 0.90))
    pred_q95 = float(np.quantile(pred_rt, 0.95))
    pred_q99 = float(np.quantile(pred_rt, 0.99))
    
    true_mean = human_stats['mean']
    true_median = human_stats['median']
    true_skewness = human_stats['skewness']
    true_q90 = float(np.quantile(true_rt, 0.90))
    true_q95 = float(np.quantile(true_rt, 0.95))
    true_q99 = float(np.quantile(true_rt, 0.99))
    
    # Score components
    skewness_score = min(max(pred_skewness, 0.0) / max(true_skewness, 1.0), 1.0)
    mean_diff = abs(pred_mean - true_mean) / true_mean
    mean_score = max(0, 1 - mean_diff)
    median_diff = abs(pred_median - true_median) / true_median
    median_score = max(0, 1 - median_diff)
    pred_range = pred_rt.max() - pred_rt.min()
    true_range = human_stats['max'] - human_stats['min']
    coverage_score = min(pred_range / true_range, 1.0)
    q90_score = max(0, 1 - abs(pred_q90 - true_q90) / max(true_q90, 1e-6))
    q95_score = max(0, 1 - abs(pred_q95 - true_q95) / max(true_q95, 1e-6))
    q99_score = max(0, 1 - abs(pred_q99 - true_q99) / max(true_q99, 1e-6))
    quantile_score = (q90_score + q95_score + q99_score) / 3.0

    human_accuracy = float((response_labels == target_labels).mean())
    model_accuracy = float((pred_choice == target_labels).mean())
    response_agreement = float((pred_choice == response_labels).mean())
    pred_correct_mask = pred_choice == target_labels
    human_correct_mask = response_labels == target_labels
    pred_error_rt = float(pred_rt[~pred_correct_mask].mean()) if (~pred_correct_mask).any() else float('nan')
    pred_correct_rt = float(pred_rt[pred_correct_mask].mean()) if pred_correct_mask.any() else float('nan')
    human_error_rt = float(true_rt[~human_correct_mask].mean()) if (~human_correct_mask).any() else float('nan')
    human_correct_rt = float(true_rt[human_correct_mask].mean()) if human_correct_mask.any() else float('nan')
    acc_diff = abs(model_accuracy - human_accuracy) / max(human_accuracy, 1e-6)
    accuracy_score = max(0, 1 - acc_diff)

    human_cong_gap = float(true_rt[congruency == 1].mean() - true_rt[congruency == 0].mean())
    model_cong_gap = float(pred_rt[congruency == 1].mean() - pred_rt[congruency == 0].mean())
    if human_cong_gap > 0:
        cong_diff = abs(model_cong_gap - human_cong_gap) / human_cong_gap
        congruency_score = max(0, 1 - cong_diff)
    else:
        congruency_score = 1.0 if model_cong_gap <= 0 else 0.0
    
    if rt_shape_focus:
        rt_score = 0.30 * skewness_score + 0.10 * mean_score + 0.10 * median_score + 0.20 * coverage_score + 0.30 * quantile_score
        total_score = 0.60 * rt_score + 0.25 * accuracy_score + 0.15 * congruency_score
    else:
        rt_score = 0.3 * skewness_score + 0.3 * mean_score + 0.2 * median_score + 0.2 * coverage_score
        total_score = 0.45 * rt_score + 0.35 * accuracy_score + 0.20 * congruency_score

    mean_median_score = 0.5 * mean_score + 0.5 * median_score
    rt_shape_score = 0.35 * skewness_score + 0.35 * quantile_score + 0.30 * coverage_score
    behavior_optimal_score = (
        0.40 * rt_shape_score
        + 0.30 * response_agreement
        + 0.15 * congruency_score
        + 0.10 * mean_median_score
        + 0.05 * accuracy_score
    )
    
    return {
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_skewness': pred_skewness,
        'true_mean': true_mean,
        'true_median': true_median,
        'true_skewness': true_skewness,
        'pred_q90': pred_q90,
        'pred_q95': pred_q95,
        'pred_q99': pred_q99,
        'true_q90': true_q90,
        'true_q95': true_q95,
        'true_q99': true_q99,
        'skewness_score': skewness_score,
        'mean_score': mean_score,
        'median_score': median_score,
        'coverage_score': coverage_score,
        'q90_score': q90_score,
        'q95_score': q95_score,
        'q99_score': q99_score,
        'quantile_score': quantile_score,
        'mean_median_score': mean_median_score,
        'rt_shape_score': rt_shape_score,
        'rt_score': rt_score,
        'human_accuracy': human_accuracy,
        'model_accuracy': model_accuracy,
        'response_agreement': response_agreement,
        'pred_error_rt': pred_error_rt,
        'pred_correct_rt': pred_correct_rt,
        'human_error_rt': human_error_rt,
        'human_correct_rt': human_correct_rt,
        'error_minus_correct_rt': float(pred_error_rt - pred_correct_rt),
        'human_error_minus_correct_rt': float(human_error_rt - human_correct_rt),
        'accuracy_score': accuracy_score,
        'human_congruency_rt_gap': human_cong_gap,
        'model_congruency_rt_gap': model_cong_gap,
        'congruency_score': congruency_score,
        'total_score': total_score,
        'behavior_optimal_score': behavior_optimal_score,
    }


def compute_stage2_outputs(
    model: WWWrapper,
    logits_batch: torch.Tensor,
    choice_temperature: float,
    rt_readout_mode: str = 'baseline',
    readout_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    target_labels: Optional[torch.Tensor] = None,
    flanker_labels: Optional[torch.Tensor] = None,
    random_seed: Optional[int] = None,
    ):
    eval_generator = None
    if random_seed is not None:
        eval_generator = build_torch_generator(int(random_seed), logits_batch.device)
    scale_tensor = model.state_dict()['scale']
    ww_input, selection_traces = build_dynamic_stage2_input(
        logits_batch,
        scale_tensor,
        time_steps=model.ww.time_steps,
        config=selection_config,
        target_labels=target_labels,
        flanker_labels=flanker_labels,
    )
    decision_times, trajectory, threshold = model.ww.inference(ww_input, generator=eval_generator)
    evidence_traj = trajectory - threshold
    choice_logits = compute_legacy_choice_logits(evidence_traj, choice_temperature)

    effective_readout_config = dict(readout_config or {})
    effective_readout_config.setdefault('t0_mode', 'disabled')
    effective_readout_config['t0_seconds'] = model.t0_seconds
    readout = compute_rt_readout(rt_readout_mode, evidence_traj, readout_config=effective_readout_config)
    final_dt = readout['pred_rt']

    readout = dict(readout)
    readout['traj'] = trajectory
    readout['threshold'] = threshold
    for key, value in selection_traces.items():
        readout[key] = value

    return final_dt, choice_logits, decision_times, readout


def compute_behavior_loss(
    final_dt: torch.Tensor,
    choice_logits: torch.Tensor,
    target_labels: torch.Tensor,
    behavior_loss_mode: str = 'baseline',
) -> torch.Tensor:
    zero = final_dt.new_zeros(())

    if behavior_loss_mode == 'baseline':
        return zero
    if behavior_loss_mode != 'error_ordering':
        raise ValueError(f"Unknown behavior_loss_mode: {behavior_loss_mode}")

    pred_choice = choice_logits.argmax(dim=1)
    pred_correct_mask = pred_choice == target_labels
    pred_error_mask = ~pred_correct_mask

    if not pred_correct_mask.any() or not pred_error_mask.any():
        return zero

    pred_correct_rt = final_dt[pred_correct_mask].mean()
    pred_error_rt = final_dt[pred_error_mask].mean()
    return F.relu(pred_correct_rt - pred_error_rt)


def compute_rt_distribution_loss(
    pred_rt: torch.Tensor,
    true_rt: torch.Tensor,
    loss_mode: str = 'baseline',
    n_bins: int = 24,
    sigma: float = 0.05,
) -> torch.Tensor:
    zero = pred_rt.new_zeros(())

    if loss_mode == 'baseline':
        return zero
    if loss_mode == 'cdf_wasserstein':
        pred_sorted = torch.sort(pred_rt.clamp(0.0, 1.0))[0]
        true_sorted = torch.sort(true_rt.clamp(0.0, 1.0))[0].detach()
        if pred_sorted.numel() != true_sorted.numel():
            n = min(pred_sorted.numel(), true_sorted.numel())
            pred_sorted = pred_sorted[:n]
            true_sorted = true_sorted[:n]
        return F.l1_loss(pred_sorted, true_sorted)

    if loss_mode != 'soft_hist_kl':
        raise ValueError(f"Unknown rt_distribution_loss_mode: {loss_mode}")

    if pred_rt.numel() < 2 or true_rt.numel() < 2:
        return zero

    pred_vals = pred_rt.clamp(0.0, 1.0)
    true_vals = true_rt.clamp(0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, n_bins, device=pred_rt.device, dtype=pred_rt.dtype)

    def soft_hist(values: torch.Tensor) -> torch.Tensor:
        distances = (values.unsqueeze(1) - centers.unsqueeze(0)) / max(float(sigma), 1e-6)
        weights = torch.exp(-0.5 * distances.square())
        hist = weights.mean(dim=0)
        hist = hist / hist.sum().clamp_min(1e-8)
        return hist

    pred_hist = soft_hist(pred_vals)
    true_hist = soft_hist(true_vals).detach()
    return F.kl_div(pred_hist.clamp_min(1e-8).log(), true_hist, reduction='batchmean')


def compute_conditional_rt_distribution_loss(
    pred_rt: torch.Tensor,
    true_rt: torch.Tensor,
    condition_labels: torch.Tensor,
    loss_mode: str = 'baseline',
    n_bins: int = 24,
    sigma: float = 0.05,
) -> torch.Tensor:
    zero = pred_rt.new_zeros(())

    if loss_mode == 'baseline':
        return zero
    if loss_mode != 'congruency_cdf_wasserstein':
        raise ValueError(f"Unknown conditional_rt_distribution_loss_mode: {loss_mode}")

    losses = []
    for condition_value in (0, 1):
        mask = condition_labels == condition_value
        if mask.sum() < 2:
            continue
        losses.append(
            compute_rt_distribution_loss(
                pred_rt=pred_rt[mask],
                true_rt=true_rt[mask],
                loss_mode='cdf_wasserstein',
                n_bins=n_bins,
                sigma=sigma,
            )
        )

    if not losses:
        return zero
    return torch.stack(losses).mean()


def compute_rt_moment_anchor_loss(
    pred_rt: torch.Tensor,
    true_rt: torch.Tensor,
    loss_mode: str = 'baseline',
) -> torch.Tensor:
    zero = pred_rt.new_zeros(())

    if loss_mode == 'baseline':
        return zero
    if loss_mode != 'mean_median_anchor':
        raise ValueError(f"Unknown rt_moment_anchor_loss_mode: {loss_mode}")
    if pred_rt.numel() < 2 or true_rt.numel() < 2:
        return zero

    pred_mean = pred_rt.mean()
    true_mean = true_rt.mean().detach()
    pred_median = pred_rt.median()
    true_median = true_rt.median().detach()
    return F.l1_loss(pred_mean, true_mean) + F.l1_loss(pred_median, true_median)


def _checkpoint_tail_focus_gate(results: Dict) -> bool:
    return (
        float(results['mean_score']) >= 0.80
        and float(results['median_score']) >= 0.75
        and float(results['response_agreement']) >= 0.93
        and float(results['q95_score']) >= 0.65
        and float(results['q99_score']) >= 0.95
        and float(results['model_congruency_rt_gap']) <= 0.065
    )


def behavior_optimal_key(
    results: Dict,
    behavior_smoke_mode: str = 'baseline',
    rt_shape_checkpoint_bias: float = 0.45,
    tail_spread_weight: float = 0.18,
    error_slower_weight: float = 0.0,
    congruency_rt_weight: float = 0.18,
):
    if behavior_smoke_mode == 'baseline':
        return (
            1,
            float(results['rt_shape_score']),
            float(results['response_agreement']),
            float(results['congruency_score']),
            float(results['mean_median_score']),
            float(results['accuracy_score']),
        )

    if behavior_smoke_mode == 'rt_response_only':
        return (
            1,
            float(results['mean_median_score']),
            float(results['response_agreement']),
            float(results['quantile_score']),
            float(results['rt_shape_score']),
            -abs(float(results['pred_mean']) - float(results['true_mean'])),
            -abs(float(results['pred_median']) - float(results['true_median'])),
        )

    if behavior_smoke_mode != 'checkpoint_tail_focus':
        raise ValueError(f"Unknown behavior_smoke_mode: {behavior_smoke_mode}")

    error_score = 0.0
    if error_slower_weight > 0:
        target_gap = float(results['human_error_minus_correct_rt'])
        pred_gap = float(results['error_minus_correct_rt'])
        error_score = max(0.0, 1.0 - abs(pred_gap - target_gap) / max(abs(target_gap), 0.1))

    tail_focus_score = (
        float(rt_shape_checkpoint_bias) * float(results['rt_shape_score'])
        + float(tail_spread_weight) * float(results['quantile_score'])
        + float(error_slower_weight) * error_score
        + float(congruency_rt_weight) * float(results['congruency_score'])
        + 0.25 * float(results['response_agreement'])
        + 0.08 * float(results['mean_median_score'])
        + 0.04 * float(results['accuracy_score'])
    )

    return (
        1 if _checkpoint_tail_focus_gate(results) else 0,
        tail_focus_score,
        float(results['q99_score']),
        float(results['q95_score']),
        float(results['coverage_score']),
        float(results['response_agreement']),
        float(results['mean_median_score']),
    )


def build_checkpoint_candidate_summary(
    *,
    scale: float,
    epoch: int,
    ranking_key: Tuple,
    results: Dict[str, Any],
    selected: bool,
) -> Dict[str, Any]:
    return {
        'scale': float(scale),
        'epoch': int(epoch),
        'selected': bool(selected),
        'ranking_key': list(ranking_key),
        'metrics': {
            'behavior_optimal_score': float(results['behavior_optimal_score']),
            'rt_shape_score': float(results['rt_shape_score']),
            'quantile_score': float(results['quantile_score']),
            'response_agreement': float(results['response_agreement']),
            'mean_median_score': float(results['mean_median_score']),
            'accuracy_score': float(results['accuracy_score']),
            'congruency_score': float(results['congruency_score']),
            'error_minus_correct_rt': float(results['error_minus_correct_rt']),
            'human_error_minus_correct_rt': float(results['human_error_minus_correct_rt']),
            'model_congruency_rt_gap': float(results['model_congruency_rt_gap']),
            'human_congruency_rt_gap': float(results['human_congruency_rt_gap']),
            'pred_mean': float(results['pred_mean']),
            'pred_median': float(results['pred_median']),
            'pred_q95': float(results['pred_q95']),
            'pred_q99': float(results['pred_q99']),
        },
    }


def train_stage2_with_scale(
    scale: float,
    time_steps: int,
    logits: np.ndarray,
    rts: np.ndarray,
    rts_normalized: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    flanker_labels: Optional[np.ndarray],
    human_stats: Dict,
    epochs: int = 20,
    lr: float = 1e-4,
    dt: int = 10,
    choice_temperature: float = 0.05,
    lambda_rt: float = 1.0,
    lambda_choice: float = 2.0,
    lambda_cong: float = 0.3,
    lambda_tail: float = 0.0,
    lambda_pileup: float = 0.0,
    behavioral_loss_config: Optional[Dict[str, Any]] = None,
    fixed_noise_ampa: Optional[float] = None,
    fixed_threshold: Optional[float] = None,
    fixed_competition_scale: Optional[float] = None,
    t0_mode: str = 'disabled',
    t0_seconds: float = 0.0,
    congruency_margin: float = 0.01,
    device: str = 'cpu',
    log_prefix: str = '',
    rt_shape_focus: bool = False,
    tail_quantiles: Optional[np.ndarray] = None,
    rt_readout_mode: str = 'baseline',
    readout_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    behavior_smoke_mode: str = 'baseline',
    behavior_loss_mode: str = 'baseline',
    behavior_loss_weight: float = 0.0,
    rt_distribution_loss_mode: str = 'baseline',
    rt_distribution_loss_weight: float = 0.0,
    conditional_rt_distribution_loss_mode: str = 'baseline',
    conditional_rt_distribution_loss_weight: float = 0.0,
    rt_moment_anchor_loss_mode: str = 'baseline',
    rt_moment_anchor_loss_weight: float = 0.0,
    rt_distribution_bins: int = 24,
    rt_distribution_sigma: float = 0.05,
    rt_shape_checkpoint_bias: float = 0.45,
    tail_spread_weight: float = 0.18,
    error_slower_weight: float = 0.0,
    congruency_rt_weight: float = 0.18,
    random_seed: Optional[int] = None,
    eval_random_seed: Optional[int] = None,
) -> Tuple[Dict, float, Dict, Dict[str, Any]]:
    """Train Stage 2 model with a specific scale value."""
    if random_seed is not None:
        set_random_seed(int(random_seed))

    
    # Create model
    model = WWWrapper(n_classes=4, dt=dt, time_steps=time_steps)
    model.scale = torch.tensor(scale, dtype=torch.float32)
    valid_t0_modes = {'disabled', 'fixed_global', 'fit_global', 'fit_age_group'}
    if t0_mode not in valid_t0_modes:
        raise ValueError(f"Unknown t0_mode: {t0_mode}")
    if float(t0_seconds) < 0.0:
        raise ValueError('t0_seconds must be non-negative.')
    with torch.no_grad():
        model.t0_seconds.copy_(torch.tensor(float(t0_seconds), dtype=torch.float32))
    model.t0_seconds.requires_grad_(t0_mode in {'fit_global', 'fit_age_group'})
    if fixed_noise_ampa is not None:
        with torch.no_grad():
            model.ww.noise_ampa.copy_(torch.tensor(float(fixed_noise_ampa), dtype=torch.float32))
        model.ww.noise_ampa.requires_grad_(False)
    if fixed_threshold is not None:
        with torch.no_grad():
            model.ww.threshold.copy_(torch.tensor(float(fixed_threshold), dtype=torch.float32))
        model.ww.threshold.requires_grad_(False)
    if fixed_competition_scale is not None:
        with torch.no_grad():
            j_matrix = model.ww.J_matrix.detach().clone()
            eye_mask = torch.eye(j_matrix.size(0), dtype=torch.bool)
            j_matrix[~eye_mask] = j_matrix[~eye_mask] * float(fixed_competition_scale)
            model.ww.J_matrix.copy_(j_matrix)
        model.ww.J_matrix.requires_grad_(False)
    model = model.to(device)
    
    # Create tensor dataset
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    rts_tensor = torch.tensor(rts_normalized, dtype=torch.float32)
    target_tensor = torch.tensor(target_labels, dtype=torch.long)
    response_tensor = torch.tensor(response_labels, dtype=torch.long)
    congruency_tensor = torch.tensor(congruency, dtype=torch.long)
    flanker_tensor = None if flanker_labels is None else torch.tensor(flanker_labels, dtype=torch.long)

    if flanker_tensor is None:
        dataset = TensorDataset(logits_tensor, rts_tensor, target_tensor, response_tensor, congruency_tensor)
    else:
        dataset = TensorDataset(logits_tensor, rts_tensor, target_tensor, response_tensor, congruency_tensor, flanker_tensor)
    dataloader_generator = torch.Generator()
    if random_seed is not None:
        dataloader_generator.manual_seed(int(random_seed))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, generator=dataloader_generator)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if tail_quantiles is None:
        tail_quantiles = np.array([0.90, 0.95, 0.99], dtype=np.float32)
    tail_quantiles_t = torch.tensor(tail_quantiles, dtype=torch.float32, device=device)
    
    best_score = 0.0
    best_results = None
    best_params = None
    best_key = None
    best_epoch = None
    checkpoint_history: List[Dict[str, Any]] = []

    print(f"{log_prefix}Starting scale={scale:.3f} for {epochs} epochs on {len(dataset)} samples")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = perf_counter()
        
        for batch in dataloader:
            if flanker_tensor is None:
                batch_logits, batch_rt, batch_target, batch_response, batch_congruency = batch
                batch_flanker = None
            else:
                batch_logits, batch_rt, batch_target, batch_response, batch_congruency, batch_flanker = batch
            batch_logits = batch_logits.to(device)
            batch_rt = batch_rt.to(device)
            batch_target = batch_target.to(device)
            batch_response = batch_response.to(device)
            batch_congruency = batch_congruency.to(device)
            if batch_flanker is not None:
                batch_flanker = batch_flanker.to(device)
            
            optimizer.zero_grad()
            final_dt, choice_logits, _, _ = compute_stage2_outputs(
                model,
                batch_logits,
                choice_temperature,
                rt_readout_mode=rt_readout_mode,
                readout_config=readout_config,
                selection_config=selection_config,
                target_labels=batch_target,
                flanker_labels=batch_flanker,
            )
            rt_loss = criterion(final_dt, batch_rt)
            choice_loss = F.cross_entropy(choice_logits, batch_response)

            if (batch_congruency == 0).any() and (batch_congruency == 1).any():
                mean_rt_cong = final_dt[batch_congruency == 0].mean()
                mean_rt_incong = final_dt[batch_congruency == 1].mean()
                cong_loss = F.relu(congruency_margin - (mean_rt_incong - mean_rt_cong))
            else:
                cong_loss = torch.tensor(0.0, device=device)

            if lambda_tail > 0:
                pred_q = torch.quantile(final_dt, tail_quantiles_t)
                true_q = torch.quantile(batch_rt, tail_quantiles_t)
                tail_loss = F.l1_loss(pred_q, true_q)
            else:
                tail_loss = torch.tensor(0.0, device=device)

            if lambda_pileup > 0:
                boundary_start = 0.95 * batch_rt.max().detach()
                pred_pile = (final_dt >= boundary_start).float().mean()
                true_pile = (batch_rt >= boundary_start).float().mean()
                pileup_loss = F.l1_loss(pred_pile, true_pile)
            else:
                pileup_loss = torch.tensor(0.0, device=device)

            if behavior_loss_weight > 0:
                behavior_loss = compute_behavior_loss(
                    final_dt=final_dt,
                    choice_logits=choice_logits,
                    target_labels=batch_target,
                    behavior_loss_mode=behavior_loss_mode,
                )
            else:
                behavior_loss = torch.tensor(0.0, device=device)

            if rt_distribution_loss_weight > 0:
                rt_distribution_loss = compute_rt_distribution_loss(
                    pred_rt=final_dt,
                    true_rt=batch_rt,
                    loss_mode=rt_distribution_loss_mode,
                    n_bins=rt_distribution_bins,
                    sigma=rt_distribution_sigma,
                )
            else:
                rt_distribution_loss = torch.tensor(0.0, device=device)

            if conditional_rt_distribution_loss_weight > 0:
                conditional_rt_distribution_loss = compute_conditional_rt_distribution_loss(
                    pred_rt=final_dt,
                    true_rt=batch_rt,
                    condition_labels=batch_congruency,
                    loss_mode=conditional_rt_distribution_loss_mode,
                    n_bins=rt_distribution_bins,
                    sigma=rt_distribution_sigma,
                )
            else:
                conditional_rt_distribution_loss = torch.tensor(0.0, device=device)

            if rt_moment_anchor_loss_weight > 0:
                rt_moment_anchor_loss = compute_rt_moment_anchor_loss(
                    pred_rt=final_dt,
                    true_rt=batch_rt,
                    loss_mode=rt_moment_anchor_loss_mode,
                )
            else:
                rt_moment_anchor_loss = torch.tensor(0.0, device=device)

            if behavioral_loss_config is not None:
                behavioral_losses = compute_behavioral_losses(
                    pred_rt=final_dt,
                    pred_choice=choice_logits.argmax(dim=1),
                    choice_probs=F.softmax(choice_logits, dim=1),
                    target_labels=batch_target,
                    response_labels=batch_response,
                    true_rt=batch_rt,
                    config=behavioral_loss_config,
                )
                behavioral_total = behavioral_losses['loss']
            else:
                behavioral_total = torch.tensor(0.0, device=device)

            loss = (
                lambda_rt * rt_loss
                + lambda_choice * choice_loss
                + lambda_cong * cong_loss
                + lambda_tail * tail_loss
                + lambda_pileup * pileup_loss
                + behavior_loss_weight * behavior_loss
                + rt_distribution_loss_weight * rt_distribution_loss
                + conditional_rt_distribution_loss_weight * conditional_rt_distribution_loss
                + rt_moment_anchor_loss_weight * rt_moment_anchor_loss
                + behavioral_total
            )

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        epoch_duration = perf_counter() - epoch_start
        print(f"{log_prefix}Epoch {epoch + 1:02d}/{epochs} avg_loss={avg_loss:.6f} duration={epoch_duration:.2f}s")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_rt_t, choice_logits_t, _, _ = compute_stage2_outputs(
                    model,
                    logits_tensor.to(device),
                    choice_temperature,
                    rt_readout_mode=rt_readout_mode,
                    readout_config=readout_config,
                    selection_config=selection_config,
                    target_labels=target_tensor.to(device),
                    flanker_labels=None if flanker_tensor is None else flanker_tensor.to(device),
                    random_seed=None if eval_random_seed is None else int(eval_random_seed + epoch + 1),
                )
                pred_rt = pred_rt_t.cpu().numpy()
                pred_choice = choice_logits_t.argmax(dim=1).cpu().numpy()
            
            results = evaluate_joint_behavior(
                pred_rt=pred_rt,
                pred_choice=pred_choice,
                true_rt=rts,
                target_labels=target_labels,
                response_labels=response_labels,
                congruency=congruency,
                human_stats=human_stats,
                rt_shape_focus=rt_shape_focus,
            )
            score = results['total_score']
            print(
                f"{log_prefix}Eval epoch {epoch + 1:02d}: "
                f"score={score:.4f} rt_score={results['rt_score']:.4f} "
                f"acc={results['model_accuracy']:.4f}/{results['human_accuracy']:.4f} "
                f"resp_agree={results['response_agreement']:.4f} "
                f"cong_gap={results['model_congruency_rt_gap']:.4f}/{results['human_congruency_rt_gap']:.4f} "
                f"rt_shape={results['rt_shape_score']:.4f} behavior_opt={results['behavior_optimal_score']:.4f}"
            )
            
            current_key = behavior_optimal_key(
                results,
                behavior_smoke_mode=behavior_smoke_mode,
                rt_shape_checkpoint_bias=rt_shape_checkpoint_bias,
                tail_spread_weight=tail_spread_weight,
                error_slower_weight=error_slower_weight,
                congruency_rt_weight=congruency_rt_weight,
            )
            checkpoint_history.append(build_checkpoint_candidate_summary(
                scale=scale,
                epoch=epoch + 1,
                ranking_key=current_key,
                results=results,
                selected=False,
            ))
            if best_key is None or current_key > best_key:
                best_score = score
                best_results = results.copy()
                best_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                best_key = current_key
                best_epoch = epoch + 1
                print(f"{log_prefix}New best score={best_score:.4f} at epoch {epoch + 1:02d}")
    
    if best_results is None or best_params is None:
        raise RuntimeError(f"No valid Stage 2 result found for scale={scale}")

    print(f"{log_prefix}Completed scale={scale:.3f} with best_score={best_score:.4f}")

    if checkpoint_history and best_epoch is not None:
        for candidate in checkpoint_history:
            if candidate['epoch'] == int(best_epoch):
                candidate['selected'] = True
                break

    return best_results, best_score, best_params, {
        'best_epoch': None if best_epoch is None else int(best_epoch),
        'checkpoint_history': checkpoint_history,
    }


def load_cached_logits_npz(npz_path: str) -> Dict[str, np.ndarray]:
    cached = np.load(npz_path)
    required_keys = ['logits', 'rts', 'rts_normalized', 'target_labels', 'response_labels', 'congruency']
    missing = [key for key in required_keys if key not in cached.files]
    if missing:
        raise KeyError(f"Cached logits file missing required keys {missing}: {npz_path}")

    optional_keys = ['flanker_labels']
    output = {key: cached[key] for key in required_keys}
    for key in optional_keys:
        if key in cached.files:
            output[key] = cached[key]
    return output


def validate_cached_stage2_inputs(age_group: str, data_dir: str, train_logits_path: str, test_logits_path: str):
    required_files = [
        os.path.join(data_dir, 'train_data.csv'),
        os.path.join(data_dir, 'test_data.csv'),
        os.path.join(data_dir, 'rt_stats.json'),
        train_logits_path,
        test_logits_path,
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required cached Stage 2 inputs for {age_group}: {missing}")

    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    train_cached = load_cached_logits_npz(train_logits_path)
    test_cached = load_cached_logits_npz(test_logits_path)

    if len(train_df) != len(train_cached['logits']):
        raise ValueError(f"Train length mismatch for {age_group}: csv={len(train_df)} logits={len(train_cached['logits'])}")
    if len(test_df) != len(test_cached['logits']):
        raise ValueError(f"Test length mismatch for {age_group}: csv={len(test_df)} logits={len(test_cached['logits'])}")

    metadata_path = os.path.join(data_dir, 'matched_branch_metadata.json')
    if 'matched' in Path(data_dir).parts and not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Matched branch metadata missing for {age_group}: {metadata_path}")

    return train_cached, test_cached


def save_partial_best_snapshot(
    output_dir: str,
    age_group: str,
    scale: float,
    epoch: int,
    score: float,
    results: Dict,
    params: Dict,
    pred_rt: np.ndarray,
    pred_choice: np.ndarray,
    target_labels: np.ndarray,
    response_labels: np.ndarray,
    congruency: np.ndarray,
    time_steps: int,
    data_root: str,
    output_root: str,
    supervision_type: str,
    extra_config: Optional[Dict[str, Any]] = None,
):
    partial_dir = os.path.join(output_dir, 'partial_best')
    os.makedirs(partial_dir, exist_ok=True)
    extra_config = extra_config or {}
    with open(os.path.join(partial_dir, 'best_config.partial.json'), 'w') as f:
        json.dump(to_jsonable({
            'age_group': age_group,
            'scale': float(scale),
            'score': float(score),
            'epoch': int(epoch),
            'time_steps': int(time_steps),
            'results': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results.items()},
            **extra_config,
        }), f, indent=2)
    np.savez(os.path.join(partial_dir, 'best_model_params.partial.npz'), **params)
    np.savez_compressed(
        os.path.join(partial_dir, 'best_test_predictions.partial.npz'),
        pred_rt=pred_rt,
        pred_choice=pred_choice,
        target_labels=target_labels,
        response_labels=response_labels,
        congruency=congruency,
    )
    with open(os.path.join(partial_dir, 'best_checkpoint_meta.json'), 'w') as f:
        json.dump(to_jsonable({
            'age_group': age_group,
            'data_root': data_root,
            'output_root': output_root,
            'supervision_type': supervision_type,
            'scale': float(scale),
            'epoch': int(epoch),
            'score': float(score),
            'rt_score': float(results['rt_score']),
            'model_accuracy': float(results['model_accuracy']),
            'human_accuracy': float(results['human_accuracy']),
            'response_agreement': float(results['response_agreement']),
            'model_congruency_rt_gap': float(results['model_congruency_rt_gap']),
            'human_congruency_rt_gap': float(results['human_congruency_rt_gap']),
            'rt_shape_score': float(results['rt_shape_score']),
            'behavior_optimal_score': float(results['behavior_optimal_score']),
            **extra_config,
        }), f, indent=2)


def build_checkpoint_ranking_summary(
    results_list: List[Dict[str, Any]],
    behavior_smoke_mode: str,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []

    for result in results_list:
        scale = float(result['scale'])
        selection_details = result.get('selection_details', {})
        for candidate in selection_details.get('checkpoint_history', []):
            candidate_copy = dict(candidate)
            candidate_copy['scale'] = scale
            candidate_copy['best_within_scale'] = bool(candidate_copy.get('selected', False))
            candidate_copy['selected'] = False
            candidates.append(candidate_copy)

    candidates_sorted = sorted(candidates, key=lambda item: tuple(item['ranking_key']), reverse=True)
    selected_checkpoint = {'scale': None, 'epoch': None}
    if candidates_sorted:
        candidates_sorted[0]['selected'] = True
        selected_checkpoint = {
            'scale': candidates_sorted[0]['scale'],
            'epoch': candidates_sorted[0]['epoch'],
        }

    return {
        'behavior_smoke_mode': behavior_smoke_mode,
        'selected_checkpoint': selected_checkpoint,
        'candidate_checkpoints': candidates_sorted,
    }


def infer_predictions_from_params(
    params: Dict,
    scale: float,
    time_steps: int,
    logits: np.ndarray,
    device: str,
    choice_temperature: float = 0.05,
    rt_readout_mode: str = 'baseline',
    readout_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    target_labels: Optional[np.ndarray] = None,
    flanker_labels: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
        set_random_seed(int(random_seed))
    model = WWWrapper(n_classes=4, dt=10, time_steps=time_steps)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in params:
            state_dict[key] = torch.tensor(params[key], dtype=torch.float32)
    state_dict['scale'] = torch.tensor(scale, dtype=torch.float32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_rt_t, choice_logits_t, decision_times_t, readout = compute_stage2_outputs(
            model,
            torch.tensor(logits, dtype=torch.float32).to(device),
            choice_temperature,
            rt_readout_mode=rt_readout_mode,
            readout_config=readout_config,
            selection_config=selection_config,
            target_labels=None if target_labels is None else torch.tensor(target_labels, dtype=torch.long).to(device),
            flanker_labels=None if flanker_labels is None else torch.tensor(flanker_labels, dtype=torch.long).to(device),
            random_seed=random_seed,
        )
    output = {
        'pred_rt': pred_rt_t.cpu().numpy(),
        'pred_choice': choice_logits_t.argmax(dim=1).cpu().numpy(),
        'choice_logits': choice_logits_t.cpu().numpy(),
        'decision_times_class': decision_times_t.cpu().numpy(),
    }
    for key, value in readout.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu().numpy()
    return output


def evaluate_cached_stage2_params(
    *,
    params: Dict,
    scale: float,
    time_steps: int,
    cached: Dict[str, np.ndarray],
    device: str,
    choice_temperature: float = 0.05,
    rt_readout_mode: str = 'baseline',
    readout_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    rt_shape_focus: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    predictions = infer_predictions_from_params(
        params=params,
        scale=scale,
        time_steps=time_steps,
        logits=cached['logits'],
        device=device,
        choice_temperature=choice_temperature,
        rt_readout_mode=rt_readout_mode,
        readout_config=readout_config,
        selection_config=selection_config,
        target_labels=cached.get('target_labels'),
        flanker_labels=cached.get('flanker_labels'),
        random_seed=random_seed,
    )
    canonical_results = evaluate_joint_behavior(
        pred_rt=predictions['pred_rt'],
        pred_choice=predictions['pred_choice'],
        true_rt=cached['rts'],
        target_labels=cached['target_labels'],
        response_labels=cached['response_labels'],
        congruency=cached['congruency'],
        human_stats=compute_human_stats_from_rts(cached['rts']),
        rt_shape_focus=rt_shape_focus,
    )
    return predictions, canonical_results


def save_ww_trajectory_samples(
    output_dir: str,
    predictions: Dict[str, np.ndarray],
    cached: Dict[str, np.ndarray],
    dt_ms: int,
):
    np.savez_compressed(
        os.path.join(output_dir, 'trajectory_samples.npz'),
        traj=predictions['traj'].astype(np.float32),
        pred_choice=predictions['pred_choice'].astype(np.int64),
        target_labels=cached['target_labels'].astype(np.int64),
        flanker_labels=cached['flanker_labels'].astype(np.int64) if 'flanker_labels' in cached else np.full(len(cached['target_labels']), -1, dtype=np.int64),
        response_labels=cached['response_labels'].astype(np.int64),
        pred_rt=predictions['pred_rt'].astype(np.float32),
        true_rt=cached['rts'].astype(np.float32),
        congruency=cached['congruency'].astype(np.int64),
        decision_times_class=predictions['decision_times_class'].astype(np.float32),
        choice_logits=predictions['choice_logits'].astype(np.float32),
        threshold=predictions['threshold'].astype(np.float32),
        dt_ms=np.array(int(dt_ms), dtype=np.int32),
    )


def fit_stage2_from_logits(
    age_group: str,
    output_dir: str,
    human_stats: Dict,
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
    device: str = 'cpu',
    scales: Optional[np.ndarray] = None,
    data_root: str = 'data_age_groups',
    output_root: str = 'checkpoints_age_groups',
    supervision_type: str = 'response_labels',
    epochs: int = 20,
    lambda_rt: float = 1.0,
    lambda_choice: float = 2.0,
    lambda_cong: float = 0.3,
    choice_temperature: float = 0.05,
    lambda_tail: float = 0.0,
    lambda_pileup: float = 0.0,
    fixed_noise_ampa: Optional[float] = None,
    fixed_threshold: Optional[float] = None,
    fixed_competition_scale: Optional[float] = None,
    t0_mode: str = 'disabled',
    t0_seconds: float = 0.0,
    time_steps_factor: float = 1.0,
    rt_shape_focus: bool = False,
    tail_quantiles: Optional[np.ndarray] = None,
    rt_readout_mode: str = 'baseline',
    readout_config: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
    smoke_metadata: Optional[Dict[str, Any]] = None,
    behavior_smoke_mode: str = 'baseline',
    behavior_loss_mode: str = 'baseline',
    behavior_loss_weight: float = 0.0,
    rt_distribution_loss_mode: str = 'baseline',
    rt_distribution_loss_weight: float = 0.0,
    conditional_rt_distribution_loss_mode: str = 'baseline',
    conditional_rt_distribution_loss_weight: float = 0.0,
    rt_moment_anchor_loss_mode: str = 'baseline',
    rt_moment_anchor_loss_weight: float = 0.0,
    rt_distribution_bins: int = 24,
    rt_distribution_sigma: float = 0.05,
    rt_shape_checkpoint_bias: float = 0.45,
    tail_spread_weight: float = 0.18,
    error_slower_weight: float = 0.0,
    congruency_rt_weight: float = 0.18,
    behavioral_loss_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    eval_random_seed: Optional[int] = None,
):
    print(f"\n{'='*60}")
    print(f"Processing age group from cached logits: {age_group}")
    print(f"{'='*60}")
    print(f"Human stats: Mean={human_stats['mean']:.3f}s, Median={human_stats['median']:.3f}s")
    print(f"Choice supervision: {supervision_type}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Epochs: {epochs}")
    print(f"lambda_rt={lambda_rt} lambda_choice={lambda_choice} lambda_cong={lambda_cong} lambda_tail={lambda_tail} lambda_pileup={lambda_pileup} choice_temperature={choice_temperature}")
    print(f"RT readout mode: {rt_readout_mode}")
    print(f"Behavior smoke mode: {behavior_smoke_mode}")
    print(f"Behavior loss mode: {behavior_loss_mode} (weight={behavior_loss_weight})")
    print(f"RT distribution loss mode: {rt_distribution_loss_mode} (weight={rt_distribution_loss_weight}, bins={rt_distribution_bins}, sigma={rt_distribution_sigma})")
    print(f"Conditional RT distribution loss mode: {conditional_rt_distribution_loss_mode} (weight={conditional_rt_distribution_loss_weight})")
    print(f"RT moment anchor loss mode: {rt_moment_anchor_loss_mode} (weight={rt_moment_anchor_loss_weight})")
    if fixed_noise_ampa is not None:
        print(f"Fixed noise_ampa: {fixed_noise_ampa}")
    if fixed_threshold is not None:
        print(f"Fixed threshold: {fixed_threshold}")
    if fixed_competition_scale is not None:
        print(f"Fixed competition scale: {fixed_competition_scale}")
    print(f"t0 mode: {t0_mode}")
    print(f"t0 seconds: {float(t0_seconds):.6f}")

    smoke_metadata = smoke_metadata or {}
    is_smoke = bool(smoke_metadata.get('smoke_test', False))

    time_steps = int(np.ceil(human_stats['percentile_99'] * 100 * time_steps_factor))
    print(f"Using time_steps={time_steps} (max RT={time_steps*10/1000:.2f}s)")
    print(f"RT shape focus: {rt_shape_focus}")

    train_logits = train_cached['logits']
    train_rts = train_cached['rts']
    train_rts_norm = train_cached['rts_normalized']
    train_target = train_cached['target_labels']
    train_response = train_cached['response_labels']
    train_congruency = train_cached['congruency']
    train_flanker = train_cached.get('flanker_labels')
    test_target = test_cached['target_labels']
    test_response = test_cached['response_labels']
    test_congruency = test_cached['congruency']

    if scales is None:
        scales = np.linspace(0.1, 0.5, 5)

    print("\nSearching for optimal scale using cached logits...")
    results_list = []
    total_scales = len(scales)
    global_best_key = None

    for scale_idx, scale in enumerate(scales, start=1):
        scale_start = perf_counter()
        log_prefix = f"[{age_group} scale {scale_idx}/{total_scales}] "
        scale_complete_marker = Path(output_dir) / "scales" / f"scale_{scale_idx:02d}_complete.json"
        if scale_complete_marker.exists():
            print(f"{log_prefix}Scale already complete, skipping.")
            try:
                prior = json.loads(scale_complete_marker.read_text(encoding="utf-8"))
                test_predictions_path = Path(output_dir) / "scales" / f"scale_{scale_idx:02d}_predictions.npz"
                params_path = Path(output_dir) / "scales" / f"scale_{scale_idx:02d}_params.npz"
                if test_predictions_path.exists() and params_path.exists():
                    params = dict(np.load(params_path))
                    test_predictions = dict(np.load(test_predictions_path))
                    canonical_results = prior.get("results", {})
                    ranking_key = tuple(prior.get("ranking_key", []))
                    results_list.append({
                        'scale': scale,
                        'score': float(canonical_results.get('total_score', 0.0)),
                        'ranking_key': ranking_key,
                        'results': canonical_results,
                        'params': params,
                        'selection_details': prior.get('selection_details', {}),
                        'selection_results': prior.get('selection_results', {}),
                    })
                    if global_best_key is None or ranking_key > global_best_key:
                        global_best_key = ranking_key
                    continue
            except Exception:
                print(f"{log_prefix}Scale marker exists but corrupted, re-running.")
        print(f"{log_prefix}Beginning optimization")
        scale_train_seed = None if random_seed is None else int(random_seed + scale_idx * 1000)
        scale_eval_seed = None if eval_random_seed is None else int(eval_random_seed)
        results, score, params, selection_details = train_stage2_with_scale(
            scale=scale,
            time_steps=time_steps,
            logits=train_logits,
            rts=train_rts,
            rts_normalized=train_rts_norm,
            target_labels=train_target,
            response_labels=train_response,
            congruency=train_congruency,
            flanker_labels=train_flanker,
            human_stats=human_stats,
            epochs=epochs,
            lambda_rt=lambda_rt,
            lambda_choice=lambda_choice,
            lambda_cong=lambda_cong,
            lambda_tail=lambda_tail,
            lambda_pileup=lambda_pileup,
            fixed_noise_ampa=fixed_noise_ampa,
            fixed_threshold=fixed_threshold,
            fixed_competition_scale=fixed_competition_scale,
            t0_mode=t0_mode,
            t0_seconds=t0_seconds,
            choice_temperature=choice_temperature,
            rt_shape_focus=rt_shape_focus,
            tail_quantiles=tail_quantiles,
            rt_readout_mode=rt_readout_mode,
            readout_config=readout_config,
            selection_config=selection_config,
            behavior_smoke_mode=behavior_smoke_mode,
            behavior_loss_mode=behavior_loss_mode,
            behavior_loss_weight=behavior_loss_weight,
            rt_distribution_loss_mode=rt_distribution_loss_mode,
            rt_distribution_loss_weight=rt_distribution_loss_weight,
            conditional_rt_distribution_loss_mode=conditional_rt_distribution_loss_mode,
            conditional_rt_distribution_loss_weight=conditional_rt_distribution_loss_weight,
            rt_moment_anchor_loss_mode=rt_moment_anchor_loss_mode,
            rt_moment_anchor_loss_weight=rt_moment_anchor_loss_weight,
            rt_distribution_bins=rt_distribution_bins,
            rt_distribution_sigma=rt_distribution_sigma,
            rt_shape_checkpoint_bias=rt_shape_checkpoint_bias,
            tail_spread_weight=tail_spread_weight,
            error_slower_weight=error_slower_weight,
            congruency_rt_weight=congruency_rt_weight,
            behavioral_loss_config=behavioral_loss_config,
            random_seed=scale_train_seed,
            eval_random_seed=scale_eval_seed,
            device=device,
            log_prefix=log_prefix,
        )

        test_predictions, canonical_results = evaluate_cached_stage2_params(
            params=params,
            scale=scale,
            time_steps=time_steps,
            cached=test_cached,
            device=device,
            choice_temperature=choice_temperature,
            rt_readout_mode=rt_readout_mode,
            readout_config=readout_config,
            selection_config=selection_config,
            random_seed=scale_eval_seed,
            rt_shape_focus=rt_shape_focus,
        )

        ranking_key = behavior_optimal_key(
            canonical_results,
            behavior_smoke_mode=behavior_smoke_mode,
            rt_shape_checkpoint_bias=rt_shape_checkpoint_bias,
            tail_spread_weight=tail_spread_weight,
            error_slower_weight=error_slower_weight,
            congruency_rt_weight=congruency_rt_weight,
        )

        results_list.append({
            'scale': scale,
            'score': float(canonical_results['total_score']),
            'ranking_key': ranking_key,
            'results': canonical_results,
            'params': params,
            'selection_details': selection_details,
            'selection_results': results,
        })

        if global_best_key is None or ranking_key > global_best_key:
            global_best_key = ranking_key
            save_partial_best_snapshot(
                output_dir=output_dir,
                age_group=age_group,
                scale=scale,
                epoch=int(selection_details.get('best_epoch', epochs)),
                score=float(canonical_results['total_score']),
                results=canonical_results,
                params=params,
                pred_rt=test_predictions['pred_rt'],
                pred_choice=test_predictions['pred_choice'],
                target_labels=test_target,
                response_labels=test_response,
                congruency=test_congruency,
                time_steps=time_steps,
                data_root=data_root,
                output_root=output_root,
                supervision_type=supervision_type,
                extra_config={
                    'best_epoch': selection_details.get('best_epoch'),
                    'rt_readout_mode': rt_readout_mode,
                    'behavior_smoke_mode': behavior_smoke_mode,
                    'behavior_loss_mode': behavior_loss_mode,
                    'behavior_loss_weight': float(behavior_loss_weight),
                    'rt_distribution_loss_mode': rt_distribution_loss_mode,
                    'rt_distribution_loss_weight': float(rt_distribution_loss_weight),
                    'conditional_rt_distribution_loss_mode': conditional_rt_distribution_loss_mode,
                    'conditional_rt_distribution_loss_weight': float(conditional_rt_distribution_loss_weight),
                    'rt_moment_anchor_loss_mode': rt_moment_anchor_loss_mode,
                    'rt_moment_anchor_loss_weight': float(rt_moment_anchor_loss_weight),
                    'rt_distribution_bins': int(rt_distribution_bins),
                    'rt_distribution_sigma': float(rt_distribution_sigma),
                    'fixed_threshold': None if fixed_threshold is None else float(fixed_threshold),
                    'fixed_competition_scale': None if fixed_competition_scale is None else float(fixed_competition_scale),
                    't0_mode': t0_mode,
                    't0_seconds': float(t0_seconds),
                    'readout_config': readout_config or {},
                    'selection_config': selection_config or {},
                    'selection_results': results,
                    'train_random_seed': scale_train_seed,
                    'eval_random_seed': scale_eval_seed,
                    **smoke_metadata,
                },
            )

        scale_complete_marker.parent.mkdir(parents=True, exist_ok=True)
        with open(str(scale_complete_marker), 'w') as f:
            json.dump(to_jsonable({
                "scale": float(scale),
                "scale_idx": scale_idx,
                "score": float(canonical_results['total_score']),
                "ranking_key": list(ranking_key),
                "results": canonical_results,
                "selection_details": selection_details,
                "selection_results": results,
            }), f, indent=2)
        # Save per-scale params and predictions for resume
        np.savez(str(scale_complete_marker.parent / f"scale_{scale_idx:02d}_params.npz"), **params)
        np.savez_compressed(
            str(scale_complete_marker.parent / f"scale_{scale_idx:02d}_predictions.npz"),
            pred_rt=test_predictions['pred_rt'],
            pred_choice=test_predictions['pred_choice'],
        )
        scale_duration = perf_counter() - scale_start
        print(
            f"{log_prefix}Finished in {scale_duration:.2f}s | "
            f"Score={canonical_results['total_score']:.4f}, PredMean={canonical_results['pred_mean']:.3f}s, "
            f"Acc={canonical_results['model_accuracy']:.4f}, Cong={canonical_results['model_congruency_rt_gap']:.4f}"
        )

    best_idx = max(range(len(results_list)), key=lambda idx: results_list[idx]['ranking_key'])
    best_config = results_list[best_idx]

    print(f"\n{'='*50}")
    print(f"Best scale for {age_group}: {best_config['scale']:.3f}")
    print(f"Score: {best_config['score']:.4f}")
    print(f"Pred Mean RT: {best_config['results']['pred_mean']:.3f}s")
    print(f"Human Mean RT: {human_stats['mean']:.3f}s")
    print(f"Model Acc: {best_config['results']['model_accuracy']:.4f}")
    print(f"Human Acc: {best_config['results']['human_accuracy']:.4f}")
    print(f"Response agreement: {best_config['results']['response_agreement']:.4f}")
    print(f"RT-shape score: {best_config['results']['rt_shape_score']:.4f}")
    print(f"Behavior-optimal score: {best_config['results']['behavior_optimal_score']:.4f}")
    print(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)

    best_predictions, smoke_metrics = evaluate_cached_stage2_params(
        params=best_config['params'],
        scale=float(best_config['scale']),
        time_steps=time_steps,
        cached=test_cached,
        device=device,
        choice_temperature=choice_temperature,
        rt_readout_mode=rt_readout_mode,
        readout_config=readout_config,
        random_seed=eval_random_seed,
        rt_shape_focus=rt_shape_focus,
    )
    config_payload = {
        'scale': float(best_config['scale']),
        'best_epoch': best_config.get('selection_details', {}).get('best_epoch'),
        'score': float(best_config['score']),
        'time_steps': time_steps,
        'results': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in smoke_metrics.items()},
        'selection_results': {k: float(v) if isinstance(v, (np.floating, float)) else v
                              for k, v in best_config.get('selection_results', best_config['results']).items()},
        'rt_readout_mode': rt_readout_mode,
        'behavior_smoke_mode': behavior_smoke_mode,
        'behavior_loss_mode': behavior_loss_mode,
        'behavior_loss_weight': float(behavior_loss_weight),
        'rt_distribution_loss_mode': rt_distribution_loss_mode,
        'rt_distribution_loss_weight': float(rt_distribution_loss_weight),
        'conditional_rt_distribution_loss_mode': conditional_rt_distribution_loss_mode,
        'conditional_rt_distribution_loss_weight': float(conditional_rt_distribution_loss_weight),
        'rt_moment_anchor_loss_mode': rt_moment_anchor_loss_mode,
        'rt_moment_anchor_loss_weight': float(rt_moment_anchor_loss_weight),
        'rt_distribution_bins': int(rt_distribution_bins),
        'rt_distribution_sigma': float(rt_distribution_sigma),
        'fixed_noise_ampa': None if fixed_noise_ampa is None else float(fixed_noise_ampa),
        'fixed_threshold': None if fixed_threshold is None else float(fixed_threshold),
        'fixed_competition_scale': None if fixed_competition_scale is None else float(fixed_competition_scale),
        't0_mode': t0_mode,
        't0_seconds': float(t0_seconds),
        'readout_config': readout_config or {},
        'selection_config': selection_config or {},
        'trajectory_artifact': 'trajectory_samples.npz' if is_smoke else None,
        'random_seed': None if random_seed is None else int(random_seed),
        'eval_random_seed': None if eval_random_seed is None else int(eval_random_seed),
        **smoke_metadata,
    }

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(to_jsonable(config_payload), f, indent=2)

    if is_smoke:
        with open(os.path.join(output_dir, 'config_smoke.json'), 'w') as f:
            json.dump(to_jsonable(config_payload), f, indent=2)

        with open(os.path.join(output_dir, 'metrics_smoke.json'), 'w') as f:
            json.dump(to_jsonable(smoke_metrics), f, indent=2)

        smoke_eval_subset_meta = smoke_metadata.get('smoke_eval_subset_meta')
        if smoke_eval_subset_meta is not None:
            with open(os.path.join(output_dir, 'smoke_eval_subset_meta.json'), 'w') as f:
                json.dump(to_jsonable(smoke_eval_subset_meta), f, indent=2)

        with open(os.path.join(output_dir, 'checkpoint_ranking_summary.json'), 'w') as f:
            json.dump(to_jsonable(build_checkpoint_ranking_summary(results_list, behavior_smoke_mode)), f, indent=2)

    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_config['params'])

    if is_smoke:
        np.savez_compressed(
            os.path.join(output_dir, 'predictions_smoke.npz'),
            pred_rt=best_predictions['pred_rt'],
            pred_choice=best_predictions['pred_choice'],
            choice_logits=best_predictions['choice_logits'],
            decision_times_class=best_predictions['decision_times_class'],
            true_rt=test_cached['rts'],
            target_labels=test_target,
            flanker_labels=test_cached['flanker_labels'] if 'flanker_labels' in test_cached else np.full(len(test_target), -1, dtype=np.int64),
            response_labels=test_response,
            congruency=test_congruency,
            rt_readout_mode=np.array(rt_readout_mode),
            scale=np.array(float(best_config['scale']), dtype=np.float32),
        )
        save_ww_trajectory_samples(output_dir, best_predictions, test_cached, dt_ms=10)

    np.savez(
        os.path.join(output_dir, 'train_logits.npz'),
        **train_cached,
    )

    np.savez(
        os.path.join(output_dir, 'test_logits.npz'),
        **test_cached,
    )

    return best_config


def process_age_group(
    age_group: str,
    stage1_model_path: str,
    data_dir: str,
    output_dir: str,
    device: str = 'cpu'
):
    """Process a single age group: extract logits and train Stage 2."""
    
    print(f"\n{'='*60}")
    print(f"Processing age group: {age_group}")
    print(f"{'='*60}")
    
    # Load human stats
    with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
        human_stats = json.load(f)
    
    print(f"Human stats: Mean={human_stats['mean']:.3f}s, Median={human_stats['median']:.3f}s")
    
    # Determine time_steps based on human RT
    time_steps = int(np.ceil(human_stats['percentile_99'] * 100))
    print(f"Using time_steps={time_steps} (max RT={time_steps*10/1000:.2f}s)")
    
    # Load Stage 1 model
    print("Loading Stage 1 model...")
    model = VGGFeatureExtractor(pretrained=False, n_classes=4)
    checkpoint = torch.load(stage1_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Create datasets
    train_dataset = StimulusDataset(os.path.join(data_dir, 'train_data.csv'))
    test_dataset = StimulusDataset(os.path.join(data_dir, 'test_data.csv'))
    
    loader_kwargs = {
        'batch_size': 64,
        'shuffle': False,
        'num_workers': min(4, os.cpu_count() or 1),
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    
    # Extract logits
    print("\nExtracting train logits...")
    train_logits = extract_logits(model, train_loader, device)
    
    print("Extracting test logits...")
    test_logits = extract_logits(model, test_loader, device)
    
    # Get RTs
    train_rts = train_dataset.rts
    train_rts_norm = train_dataset.rt_normalized
    train_target = train_dataset.target_labels
    train_response = train_dataset.response_labels
    train_congruency = train_dataset.congruency
    train_flanker = train_dataset.flanker_labels
    
    # Search for optimal scale
    print("\nSearching for optimal scale...")
    scales = np.linspace(0.1, 0.5, 5)
    results_list = []
    
    for scale in scales:
        results, score, params, _ = train_stage2_with_scale(
            scale=scale,
            time_steps=time_steps,
            logits=train_logits,
            rts=train_rts,
            rts_normalized=train_rts_norm,
            target_labels=train_target,
            response_labels=train_response,
            congruency=train_congruency,
            flanker_labels=train_flanker,
            human_stats=human_stats,
            epochs=20,
            device=device
        )
        
        results_list.append({
            'scale': scale,
            'score': score,
            'results': results,
            'params': params
        })
        
        print(f"  Scale={scale:.3f}: Score={score:.4f}, PredMean={results['pred_mean']:.3f}s")
    
    # Find best scale
    best_idx = np.argmax([r['score'] for r in results_list])
    best_config = results_list[best_idx]
    
    print(f"\n{'='*50}")
    print(f"Best scale for {age_group}: {best_config['scale']:.3f}")
    print(f"Score: {best_config['score']:.4f}")
    print(f"Pred Mean RT: {best_config['results']['pred_mean']:.3f}s")
    print(f"Human Mean RT: {human_stats['mean']:.3f}s")
    print(f"{'='*50}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump({
            'scale': float(best_config['scale']),
            'score': float(best_config['score']),
            'time_steps': time_steps,
            'results': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in best_config['results'].items()}
        }, f, indent=2)
    
    # Save model params
    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_config['params'])
    
    # Save logits
    np.savez(
        os.path.join(output_dir, 'train_logits.npz'),
        logits=train_logits,
        rts=train_rts,
        rts_normalized=train_rts_norm,
        target_labels=train_target,
        response_labels=train_response,
        congruency=train_congruency,
        flanker_labels=train_flanker,
    )

    np.savez(
        os.path.join(output_dir, 'test_logits.npz'),
        logits=test_logits,
        rts=test_dataset.rts,
        rts_normalized=test_dataset.rt_normalized,
        target_labels=test_dataset.target_labels,
        response_labels=test_dataset.response_labels,
        congruency=test_dataset.congruency,
        flanker_labels=test_dataset.flanker_labels,
    )
    
    return best_config


def main():
    parser = argparse.ArgumentParser(description='Train age-group Stage 2 models')
    parser.add_argument('--age_group', choices=['20-29', '80-89'], help='Run only one age group')
    parser.add_argument('--cached_only', action='store_true', help='Use cached logits instead of extracting from Stage 1')
    parser.add_argument('--train_logits_path', type=str, help='Path to cached train logits npz')
    parser.add_argument('--test_logits_path', type=str, help='Path to cached test logits npz')
    parser.add_argument('--data_root', default=rel_to_root(DATA_AGE_GROUPS_ROOT), help='Root directory containing age-group data')
    parser.add_argument('--output_root', default=rel_to_root(CHECKPOINTS_AGE_GROUPS_ROOT), help='Root directory for Stage 2 outputs')
    parser.add_argument('--scale_values', type=str, help='Comma-separated scale values, e.g. 0.1 or 0.1,0.3,0.5')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per scale')
    parser.add_argument('--lambda_rt', type=float, default=1.0, help='Weight for RT loss')
    parser.add_argument('--lambda_choice', type=float, default=2.0, help='Weight for choice loss')
    parser.add_argument('--lambda_cong', type=float, default=0.3, help='Weight for congruency loss')
    parser.add_argument('--lambda_tail', type=float, default=0.0, help='Weight for tail quantile loss')
    parser.add_argument('--lambda_pileup', type=float, default=0.0, help='Weight for anti-boundary-pileup loss')
    parser.add_argument('--fixed_noise_ampa', type=float, help='If set, override and freeze Wong-Wang noise_ampa')
    parser.add_argument('--fixed_threshold', type=float, help='If set, override and freeze Wong-Wang threshold')
    parser.add_argument('--fixed_competition_scale', type=float, help='If set, scale and freeze Wong-Wang off-diagonal competition weights')
    parser.add_argument('--t0_mode', choices=['disabled', 'fixed_global', 'fit_global', 'fit_age_group'], default='disabled')
    parser.add_argument('--t0_seconds', type=float, default=0.0, help='Explicit non-decision-time term in seconds')
    parser.add_argument('--choice_temperature', type=float, default=0.05, help='Choice temperature for logits')
    parser.add_argument('--time_steps_factor', type=float, default=1.0, help='Multiplier on time_steps derived from human percentile_99')
    parser.add_argument('--rt_shape_focus', action='store_true', help='Upweight skewness and tail coverage in checkpoint selection')
    parser.add_argument('--tail_quantiles', type=str, default='0.9,0.95,0.99', help='Comma-separated quantiles for tail loss')
    parser.add_argument('--rt_readout_mode', choices=['baseline', 'soft_hazard', 'urgency', 'soft_index', 'noisy_readout'], default='baseline')
    parser.add_argument('--urgency_type', choices=['collapsing_bound', 'additive_urgency'], default='additive_urgency')
    parser.add_argument('--urgency_start', type=float, default=0.80)
    parser.add_argument('--urgency_slope', type=float, default=0.25)
    parser.add_argument('--urgency_floor', type=float, default=0.0)
    parser.add_argument('--behavior_smoke_mode', choices=['baseline', 'checkpoint_tail_focus'], default='baseline')
    parser.add_argument('--behavior_loss_mode', choices=['baseline', 'error_ordering'], default='baseline')
    parser.add_argument('--behavior_loss_weight', type=float, default=0.0)
    parser.add_argument('--rt_distribution_loss_mode', choices=['baseline', 'soft_hist_kl', 'cdf_wasserstein'], default='baseline')
    parser.add_argument('--rt_distribution_loss_weight', type=float, default=0.0)
    parser.add_argument('--conditional_rt_distribution_loss_mode', choices=['baseline', 'congruency_cdf_wasserstein'], default='baseline')
    parser.add_argument('--conditional_rt_distribution_loss_weight', type=float, default=0.0)
    parser.add_argument('--rt_moment_anchor_loss_mode', choices=['baseline', 'mean_median_anchor'], default='baseline')
    parser.add_argument('--rt_moment_anchor_loss_weight', type=float, default=0.0)
    parser.add_argument('--rt_distribution_bins', type=int, default=24)
    parser.add_argument('--rt_distribution_sigma', type=float, default=0.05)
    parser.add_argument('--rt_shape_checkpoint_bias', type=float, default=0.45)
    parser.add_argument('--tail_spread_weight', type=float, default=0.18)
    parser.add_argument('--error_slower_weight', type=float, default=0.0)
    parser.add_argument('--congruency_rt_weight', type=float, default=0.18)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--eval_seed', type=int, default=None)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--smoke_eval', action='store_true', help='Alias for smoke_test to match branch smoke commands')
    parser.add_argument('--smoke_fraction', type=float, default=0.15)
    parser.add_argument('--smoke_eval_fraction', type=float, default=0.25)
    parser.add_argument('--smoke_epochs', type=int, default=5)
    parser.add_argument('--smoke_seed', type=int, default=7)
    parser.add_argument('--smoke_eval_mode', choices=['random', 'behavior_balanced'], default='random')
    parser.add_argument('--smoke_eval_min_errors', type=int, default=0)
    parser.add_argument('--smoke_eval_balance_congruency', action='store_true')
    parser.add_argument('--smoke_eval_max_trials', type=int, default=None)
    parser.add_argument('--smoke_eval_seed', type=int, default=None)
    parser.add_argument('--experiment_name', type=str, default=None, help='Explicit smoke experiment directory name')
    args = parser.parse_args()

    if args.smoke_eval:
        args.smoke_test = True
    if float(args.t0_seconds) < 0.0:
        raise ValueError('t0_seconds must be non-negative.')

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    set_random_seed(args.seed)
    
    stage1_model_path = str(CHECKPOINTS_TEST_ROOT / 'stage1' / 'best_model.pth')
    if not args.cached_only and not os.path.exists(stage1_model_path):
        print(f"Stage 1 model not found at {stage1_model_path}")
        return

    if args.smoke_test:
        if not args.cached_only:
            raise ValueError('Smoke test mode requires --cached_only.')
        if args.age_group != '20-29':
            raise ValueError('Smoke test mode is restricted to age_group=20-29.')
        if 'matched' not in str(Path(args.data_root)):
            raise ValueError('Smoke test mode requires the matched 20-29 branch via data_root containing matched.')
        if args.output_root == rel_to_root(CHECKPOINTS_AGE_GROUPS_ROOT):
            args.output_root = rel_to_root(CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT)
        if not (0 < args.smoke_fraction <= 1.0 and 0 < args.smoke_eval_fraction <= 1.0):
            raise ValueError('Smoke fractions must be in (0, 1].')
        if args.smoke_eval_min_errors < 0:
            raise ValueError('smoke_eval_min_errors must be non-negative.')
        if args.smoke_eval_max_trials is not None and args.smoke_eval_max_trials <= 0:
            raise ValueError('smoke_eval_max_trials must be positive when provided.')
        set_random_seed(args.smoke_seed)

    eval_seed = int(args.eval_seed if args.eval_seed is not None else args.seed + 1)
    
    age_groups = [args.age_group] if args.age_group else ['20-29', '80-89']
    custom_scales = None
    if args.scale_values:
        custom_scales = np.array([float(x.strip()) for x in args.scale_values.split(',') if x.strip()], dtype=np.float32)
    tail_quantiles = np.array([float(x.strip()) for x in args.tail_quantiles.split(',') if x.strip()], dtype=np.float32)
    if args.smoke_test:
        args.epochs = args.smoke_epochs
    all_results = {}
    
    for age_group in age_groups:
        data_dir = os.path.join(args.data_root, age_group)
        ww_variant = args.behavior_smoke_mode if args.behavior_loss_mode == 'baseline' else args.behavior_loss_mode
        generated_experiment_name = (
            f"WW_rt_dist_{args.rt_distribution_loss_mode}_noise_anchor{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode != 'baseline'
            and args.fixed_noise_ampa is not None
            and args.rt_moment_anchor_loss_mode != 'baseline'
            else
            f"WW_rt_dist_{args.rt_distribution_loss_mode}_noise_threshold_probe{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode != 'baseline'
            and args.fixed_noise_ampa is not None
            and args.fixed_threshold is not None
            else
            f"WW_rt_dist_{args.rt_distribution_loss_mode}_noise_probe{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode != 'baseline'
            and args.fixed_noise_ampa is not None
            else
            f"WW_competition_probe{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode == 'baseline'
            and args.fixed_competition_scale is not None
            else
            f"WW_threshold_probe{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode == 'baseline'
            and args.fixed_threshold is not None
            else
            f"WW_noise_probe{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode == 'baseline'
            and args.fixed_noise_ampa is not None
            else
            f"WW_rt_dist_{args.rt_distribution_loss_mode}{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.rt_distribution_loss_mode != 'baseline'
            and args.conditional_rt_distribution_loss_mode == 'baseline'
            else
            f"WW_rt_dist_conditional{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test
            and args.rt_readout_mode == 'baseline'
            and args.behavior_smoke_mode == 'baseline'
            and args.behavior_loss_mode == 'baseline'
            and args.conditional_rt_distribution_loss_mode != 'baseline'
            else
            f"WW_{ww_variant}{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test and args.rt_readout_mode == 'baseline'
            else
            f"B_urgency{'' if args.smoke_eval_mode == 'random' else f'_{args.smoke_eval_mode}'}"
            if args.smoke_test and args.rt_readout_mode == 'urgency'
            else args.rt_readout_mode if args.smoke_test else 'stage2'
        )
        experiment_name = args.experiment_name or generated_experiment_name
        output_dir = (
            os.path.join(args.output_root, experiment_name)
            if args.smoke_test and args.experiment_name
            else os.path.join(args.output_root, age_group, 'smoke', experiment_name)
            if args.smoke_test
            else os.path.join(args.output_root, age_group, 'stage2')
        )
        
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue
        
        if args.cached_only:
            matched_default_dir = str(CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT / age_group / 'stage2')
            default_cached_dir = matched_default_dir if args.smoke_test and 'matched' in str(Path(args.data_root)) else os.path.join(args.output_root, age_group, 'stage2')
            train_logits_path = args.train_logits_path or os.path.join(default_cached_dir, 'train_logits.npz')
            test_logits_path = args.test_logits_path or os.path.join(default_cached_dir, 'test_logits.npz')

            with open(os.path.join(data_dir, 'rt_stats.json'), 'r') as f:
                human_stats = json.load(f)

            print(f"Using cached logits for {age_group}:")
            print(f"  train: {train_logits_path}")
            print(f"  test: {test_logits_path}")
            print("  choice supervision: response_labels")

            train_cached, test_cached = validate_cached_stage2_inputs(
                age_group=age_group,
                data_dir=data_dir,
                train_logits_path=train_logits_path,
                test_logits_path=test_logits_path,
            )

            smoke_metadata: Dict[str, Any] = {}
            if args.smoke_test:
                source_train_rows = int(len(train_cached['logits']))
                source_test_rows = int(len(test_cached['logits']))
                train_cached, train_indices = subset_cached_stage2_inputs(train_cached, args.smoke_fraction, args.smoke_seed)
                smoke_eval_seed = int(args.smoke_eval_seed if args.smoke_eval_seed is not None else args.smoke_seed + 1)
                test_cached, test_indices, smoke_eval_subset_meta = subset_smoke_eval_inputs(
                    test_cached,
                    fraction=args.smoke_eval_fraction,
                    seed=smoke_eval_seed,
                    mode=args.smoke_eval_mode,
                    min_errors=args.smoke_eval_min_errors,
                    balance_congruency=args.smoke_eval_balance_congruency,
                    max_trials=args.smoke_eval_max_trials,
                )
                smoke_metadata = {
                    'smoke_test': True,
                    'experiment_name': experiment_name,
                    'fixed_noise_ampa': None if args.fixed_noise_ampa is None else float(args.fixed_noise_ampa),
                    'fixed_threshold': None if args.fixed_threshold is None else float(args.fixed_threshold),
                    'fixed_competition_scale': None if args.fixed_competition_scale is None else float(args.fixed_competition_scale),
                    't0_mode': args.t0_mode,
                    't0_seconds': float(args.t0_seconds),
                    'smoke_seed': int(args.smoke_seed),
                    'smoke_fraction': float(args.smoke_fraction),
                    'smoke_eval_fraction': float(args.smoke_eval_fraction),
                    'smoke_epochs': int(args.smoke_epochs),
                    'smoke_eval_mode': args.smoke_eval_mode,
                    'smoke_eval_seed': smoke_eval_seed,
                    'smoke_eval_min_errors': int(args.smoke_eval_min_errors),
                    'smoke_eval_balance_congruency': bool(args.smoke_eval_balance_congruency),
                    'smoke_eval_max_trials': None if args.smoke_eval_max_trials is None else int(args.smoke_eval_max_trials),
                    'behavior_smoke_mode': args.smoke_eval_mode,
                    'behavior_loss_mode': 'response_labels_cached_logits',
                    'behavior_loss_weight': float(args.behavior_loss_weight),
                    'rt_distribution_loss_mode': args.rt_distribution_loss_mode,
                    'rt_distribution_loss_weight': float(args.rt_distribution_loss_weight),
                    'conditional_rt_distribution_loss_mode': args.conditional_rt_distribution_loss_mode,
                    'conditional_rt_distribution_loss_weight': float(args.conditional_rt_distribution_loss_weight),
                    'rt_moment_anchor_loss_mode': args.rt_moment_anchor_loss_mode,
                    'rt_moment_anchor_loss_weight': float(args.rt_moment_anchor_loss_weight),
                    'rt_distribution_bins': int(args.rt_distribution_bins),
                    'rt_distribution_sigma': float(args.rt_distribution_sigma),
                    'source_train_rows': source_train_rows,
                    'source_test_rows': source_test_rows,
                    'effective_train_rows': int(len(train_cached['logits'])),
                    'effective_test_rows': int(len(test_cached['logits'])),
                    'train_indices': train_indices.tolist(),
                    'test_indices': test_indices.tolist(),
                    'train_logits_path': train_logits_path,
                    'test_logits_path': test_logits_path,
                    'smoke_eval_subset_meta': smoke_eval_subset_meta,
                }

            result = fit_stage2_from_logits(
                age_group=age_group,
                output_dir=output_dir,
                human_stats=human_stats,
                train_cached=train_cached,
                test_cached=test_cached,
                device=device,
                scales=custom_scales,
                data_root=args.data_root,
                output_root=args.output_root,
                supervision_type='response_labels',
                epochs=args.epochs,
                lambda_rt=args.lambda_rt,
                lambda_choice=args.lambda_choice,
                lambda_cong=args.lambda_cong,
                lambda_tail=args.lambda_tail,
                lambda_pileup=args.lambda_pileup,
                fixed_noise_ampa=args.fixed_noise_ampa,
                fixed_threshold=args.fixed_threshold,
                fixed_competition_scale=args.fixed_competition_scale,
                t0_mode=args.t0_mode,
                t0_seconds=float(args.t0_seconds),
                choice_temperature=args.choice_temperature,
                time_steps_factor=args.time_steps_factor,
                rt_shape_focus=args.rt_shape_focus,
                tail_quantiles=tail_quantiles,
                rt_readout_mode=args.rt_readout_mode,
                readout_config={
                    'dt_ms': 10.0,
                    'alpha': 12.0,
                    'beta': 0.15,
                    'eps': 1e-6,
                    'choice_temperature': float(args.choice_temperature),
                    'urgency_type': args.urgency_type,
                    'urgency_start': float(args.urgency_start),
                    'urgency_slope': float(args.urgency_slope),
                    'urgency_floor': float(args.urgency_floor),
                    't0_mode': args.t0_mode,
                },
                smoke_metadata=smoke_metadata,
                behavior_smoke_mode=args.behavior_smoke_mode,
                behavior_loss_mode=args.behavior_loss_mode,
                behavior_loss_weight=float(args.behavior_loss_weight),
                rt_distribution_loss_mode=args.rt_distribution_loss_mode,
                rt_distribution_loss_weight=float(args.rt_distribution_loss_weight),
                conditional_rt_distribution_loss_mode=args.conditional_rt_distribution_loss_mode,
                conditional_rt_distribution_loss_weight=float(args.conditional_rt_distribution_loss_weight),
                rt_moment_anchor_loss_mode=args.rt_moment_anchor_loss_mode,
                rt_moment_anchor_loss_weight=float(args.rt_moment_anchor_loss_weight),
                rt_distribution_bins=int(args.rt_distribution_bins),
                rt_distribution_sigma=float(args.rt_distribution_sigma),
                rt_shape_checkpoint_bias=float(args.rt_shape_checkpoint_bias),
                tail_spread_weight=float(args.tail_spread_weight),
                error_slower_weight=float(args.error_slower_weight),
                congruency_rt_weight=float(args.congruency_rt_weight),
                random_seed=int(args.seed),
                eval_random_seed=eval_seed,
            )
        else:
            result = process_age_group(
                age_group=age_group,
                stage1_model_path=stage1_model_path,
                data_dir=data_dir,
                output_dir=output_dir,
                device=device
            )
        
        all_results[age_group] = result
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN AGE GROUPS")
    print(f"{'='*60}")
    
    for age_group, result in all_results.items():
        print(f"\n{age_group}:")
        print(f"  Best scale: {result['scale']:.3f}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Pred Mean RT: {result['results']['pred_mean']:.3f}s")
        print(f"  Pred Skewness: {result['results']['pred_skewness']:.3f}")


if __name__ == "__main__":
    main()

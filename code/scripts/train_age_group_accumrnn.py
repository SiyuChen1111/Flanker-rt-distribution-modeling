import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from train_age_groups_efficient import (
    build_checkpoint_ranking_summary,
    compute_human_stats_from_rts,
    evaluate_joint_behavior,
    save_partial_best_snapshot,
    subset_cached_stage2_inputs,
    subset_smoke_eval_inputs,
    to_jsonable,
    validate_cached_stage2_inputs,
)
from vgg_accumulator_rnn_v2 import AccumulatorRaceDecisionV2


def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_choice_state(
    traj: torch.Tensor,
    decision_times: torch.Tensor,
    dt_ms: int,
    *,
    choice_window: int = 3,
) -> torch.Tensor:
    """Windowed state-at-decision readout (legacy trainer).

    Mirrors the canonical Stage-2 backend readout:
      choice_state = mean(traj[:, t-k : t+1, :])
    where t is the discretized decision index, and k=choice_window.
    """

    if int(choice_window) < 0:
        raise ValueError('choice_window must be >= 0')

    pred_choice_idx = decision_times.argmin(dim=1)
    chosen_time = decision_times[torch.arange(traj.size(0), device=traj.device), pred_choice_idx]
    time_idx = torch.clamp((chosen_time * 1000.0 / float(dt_ms)).long(), min=0, max=traj.size(1) - 1)

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


def evaluate_model(
    model: AccumulatorRaceDecisionV2,
    cached: Dict[str, np.ndarray],
    human_stats: Dict[str, float],
    device: str,
    choice_temperature: float,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    model.eval()
    logits_tensor = torch.tensor(cached['logits'], dtype=torch.float32, device=device)
    with torch.no_grad():
        decision_times, traj, threshold_t = model.rollout(logits_tensor)
        choice_state = pick_choice_state(traj, decision_times, model.dt, choice_window=3)
        pred_choice = choice_state.argmax(dim=1).cpu().numpy()
        pred_rt = decision_times.min(dim=1)[0].cpu().numpy()
        temperature = max(float(choice_temperature), 1e-6)
        choice_logits = (choice_state / temperature).cpu().numpy()
        decision_times_np = decision_times.cpu().numpy()
        traj_np = traj.cpu().numpy().astype(np.float32)
    results = evaluate_joint_behavior(
        pred_rt=pred_rt,
        pred_choice=pred_choice,
        true_rt=cached['rts'],
        target_labels=cached['target_labels'],
        response_labels=cached['response_labels'],
        congruency=cached['congruency'],
        human_stats=human_stats,
    )
    predictions = {
        'pred_rt': pred_rt.astype(np.float32),
        'pred_choice': pred_choice.astype(np.int64),
        'choice_logits': choice_logits.astype(np.float32),
        'decision_times_class': decision_times_np.astype(np.float32),
        'traj': traj_np,
        'threshold': np.array(float(torch.as_tensor(threshold_t).detach().cpu().item()), dtype=np.float32),
    }
    return results, predictions


def ranking_key(results: Dict[str, Any]) -> List[float]:
    return [
        1.0,
        float(results['rt_shape_score']),
        float(results['response_agreement']),
        float(results['congruency_score']),
        float(results['mean_median_score']),
        float(results['accuracy_score']),
    ]


def train_with_scale(
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
    congruency_loss_weight: float,
    learning_rate: float,
    batch_size: int,
    log_prefix: str = '',
) -> Dict[str, Any]:
    model = AccumulatorRaceDecisionV2(
        n_classes=4,
        dt=10,
        time_steps=time_steps,
        threshold=threshold,
        noise_std=noise_std,
    )
    model.input_scale.data.fill_(scale)
    model = model.to(device)

    logits_tensor = torch.tensor(train_cached['logits'], dtype=torch.float32)
    rts_tensor = torch.tensor(train_cached['rts'], dtype=torch.float32)
    response_tensor = torch.tensor(train_cached['response_labels'], dtype=torch.long)
    congruency_tensor = torch.tensor(train_cached['congruency'], dtype=torch.long)
    dataloader = DataLoader(
        TensorDataset(logits_tensor, rts_tensor, response_tensor, congruency_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()
    checkpoint_history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for epoch in range(epochs):
        model.train()
        for batch_logits, batch_rt, batch_response, batch_cong in dataloader:
            batch_logits = batch_logits.to(device)
            batch_rt = batch_rt.to(device)
            batch_response = batch_response.to(device)
            batch_cong = batch_cong.to(device)

            optimizer.zero_grad()
            decision_times, traj, _threshold_t = model.rollout(batch_logits)
            pred_rt = decision_times.min(dim=1)[0]
            choice_state = pick_choice_state(traj, decision_times, model.dt, choice_window=3)
            temperature = max(float(choice_temperature), 1e-6)
            choice_logits = choice_state / temperature
            choice_loss = F.cross_entropy(choice_logits, batch_response)
            rt_loss = mse(pred_rt, batch_rt)

            if congruency_loss_weight > 0 and (batch_cong == 0).any() and (batch_cong == 1).any():
                mean_cong = pred_rt[batch_cong == 0].mean()
                mean_incong = pred_rt[batch_cong == 1].mean()
                congruency_loss = F.relu(0.01 - (mean_incong - mean_cong))
            else:
                congruency_loss = torch.tensor(0.0, device=device)

            loss = (
                response_loss_weight * choice_loss
                + rt_loss_weight * rt_loss
                + congruency_loss_weight * congruency_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        results, predictions = evaluate_model(model, eval_cached, eval_human_stats, device, choice_temperature)
        history_item = {
            'epoch': epoch + 1,
            'selected': False,
            'ranking_key': ranking_key(results),
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
        checkpoint_history.append(history_item)
        print(
            f"{log_prefix}Eval epoch {epoch + 1:02d}: behavior={results['behavior_optimal_score']:.4f} "
            f"rt_shape={results['rt_shape_score']:.4f} resp_agree={results['response_agreement']:.4f} "
            f"cong_gap={results['model_congruency_rt_gap']:.4f}/{results['human_congruency_rt_gap']:.4f} "
            f"err-corr={results['error_minus_correct_rt']:.4f}"
        )
        if best is None or tuple(history_item['ranking_key']) > tuple(best['selection_details']['ranking_key']):
            best = {
                'epoch': epoch + 1,
                'results': results,
                'params': {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
                'predictions': predictions,
                'selection_details': {
                    'best_epoch': epoch + 1,
                    'ranking_key': history_item['ranking_key'],
                    'checkpoint_history': checkpoint_history,
                },
            }

    if best is None:
        raise RuntimeError('No best checkpoint found')

    for item in best['selection_details']['checkpoint_history']:
        item['selected'] = item['epoch'] == best['epoch']
    return best


def save_trajectory_samples(
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
        response_labels=cached['response_labels'].astype(np.int64),
        pred_rt=predictions['pred_rt'].astype(np.float32),
        true_rt=cached['rts'].astype(np.float32),
        congruency=cached['congruency'].astype(np.int64),
        decision_times_class=predictions['decision_times_class'].astype(np.float32),
        choice_logits=predictions['choice_logits'].astype(np.float32),
        threshold=predictions['threshold'].astype(np.float32),
        dt_ms=np.array(int(dt_ms), dtype=np.int32),
    )


def build_config_payload(
    best_overall: Dict[str, Any],
    time_steps: int,
    experiment_name: str,
    smoke_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        'scale': float(best_overall['scale']),
        'best_epoch': int(best_overall['epoch']),
        'score': float(best_overall['results']['behavior_optimal_score']),
        'time_steps': int(time_steps),
        'results': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in best_overall['results'].items()},
        'experiment_name': experiment_name,
        'dt_ms': 10,
        'model_family': 'VGGAccumulatorRNNLIMV2',
        'trajectory_artifact': 'trajectory_samples.npz',
        **smoke_metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data_age_groups_matched')
    parser.add_argument('--output_root', default='checkpoints_age_groups_rtreadout')
    parser.add_argument('--age_group', default='20-29')
    parser.add_argument('--train_logits_path')
    parser.add_argument('--test_logits_path')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--scale_values', type=str, default='0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--time_steps', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--choice_temperature', type=float, default=0.10)
    parser.add_argument('--rt_loss_weight', type=float, default=2.0)
    parser.add_argument('--response_loss_weight', type=float, default=1.0)
    parser.add_argument('--congruency_loss_weight', type=float, default=0.10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--smoke_fraction', type=float, default=0.15)
    parser.add_argument('--smoke_eval_fraction', type=float, default=0.25)
    parser.add_argument('--smoke_epochs', type=int, default=5)
    parser.add_argument('--smoke_seed', type=int, default=7)
    parser.add_argument('--smoke_eval_mode', choices=['random', 'behavior_balanced'], default='behavior_balanced')
    parser.add_argument('--smoke_eval_seed', type=int, default=8)
    parser.add_argument('--smoke_eval_min_errors', type=int, default=512)
    parser.add_argument('--smoke_eval_balance_congruency', action='store_true')
    parser.add_argument('--smoke_eval_max_trials', type=int, default=4096)
    parser.add_argument('--experiment_name', default='accumulator_rnn_behavior_balanced')
    args = parser.parse_args()

    if args.smoke_test:
        if args.age_group != '20-29':
            raise ValueError('Smoke test mode is restricted to age_group=20-29.')
        if 'matched' not in str(Path(args.data_root)):
            raise ValueError('Smoke test mode requires matched data_root.')
        args.epochs = args.smoke_epochs
        set_random_seed(args.smoke_seed)

    experiment_name = args.experiment_name
    output_dir = (
        os.path.join(args.output_root, args.age_group, 'smoke', experiment_name)
        if args.smoke_test
        else os.path.join(args.output_root, args.age_group, 'stage2')
    )
    data_dir = os.path.join(args.data_root, args.age_group)
    default_cached_dir = (
        os.path.join('checkpoints_age_groups_matched', args.age_group, 'stage2')
        if args.smoke_test and 'matched' in str(Path(args.data_root))
        else os.path.join(args.output_root, args.age_group, 'stage2')
    )
    train_logits_path = args.train_logits_path or os.path.join(default_cached_dir, 'train_logits.npz')
    test_logits_path = args.test_logits_path or os.path.join(default_cached_dir, 'test_logits.npz')

    train_cached, test_cached = validate_cached_stage2_inputs(args.age_group, data_dir, train_logits_path, test_logits_path)
    device = get_device()
    scales = np.array([float(x.strip()) for x in args.scale_values.split(',') if x.strip()], dtype=np.float32)

    smoke_metadata: Dict[str, Any] = {
        'train_logits_path': train_logits_path,
        'test_logits_path': test_logits_path,
        'behavior_smoke_mode': 'balanced',
        'behavior_loss_mode': 'balanced',
        'behavior_loss_weight': float(args.rt_loss_weight),
    }
    if args.smoke_test:
        source_train_rows = int(len(train_cached['logits']))
        source_test_rows = int(len(test_cached['logits']))
        train_cached, train_indices = subset_cached_stage2_inputs(train_cached, args.smoke_fraction, args.smoke_seed)
        test_cached, test_indices, smoke_eval_subset_meta = subset_smoke_eval_inputs(
            test_cached,
            fraction=args.smoke_eval_fraction,
            seed=args.smoke_eval_seed,
            mode=args.smoke_eval_mode,
            min_errors=args.smoke_eval_min_errors,
            balance_congruency=args.smoke_eval_balance_congruency,
            max_trials=args.smoke_eval_max_trials,
        )
        smoke_metadata.update({
            'smoke_test': True,
            'smoke_seed': int(args.smoke_seed),
            'smoke_fraction': float(args.smoke_fraction),
            'smoke_eval_fraction': float(args.smoke_eval_fraction),
            'smoke_epochs': int(args.smoke_epochs),
            'smoke_eval_mode': args.smoke_eval_mode,
            'smoke_eval_seed': int(args.smoke_eval_seed),
            'smoke_eval_min_errors': int(args.smoke_eval_min_errors),
            'smoke_eval_balance_congruency': bool(args.smoke_eval_balance_congruency),
            'smoke_eval_max_trials': int(args.smoke_eval_max_trials),
            'source_train_rows': source_train_rows,
            'source_test_rows': source_test_rows,
            'effective_train_rows': int(len(train_cached['logits'])),
            'effective_test_rows': int(len(test_cached['logits'])),
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist(),
            'smoke_eval_subset_meta': smoke_eval_subset_meta,
        })

    eval_human_stats = compute_human_stats_from_rts(test_cached['rts'])
    best_overall: Optional[Dict[str, Any]] = None
    results_list: List[Dict[str, Any]] = []

    for idx, scale in enumerate(scales, start=1):
        best = train_with_scale(
            scale=float(scale),
            time_steps=args.time_steps,
            train_cached=train_cached,
            eval_cached=test_cached,
            eval_human_stats=eval_human_stats,
            epochs=args.epochs,
            device=device,
            noise_std=args.noise_std,
            threshold=args.threshold,
            choice_temperature=args.choice_temperature,
            rt_loss_weight=args.rt_loss_weight,
            response_loss_weight=args.response_loss_weight,
            congruency_loss_weight=args.congruency_loss_weight,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            log_prefix=f'[accumrnn balanced scale {idx}/{len(scales)}] ',
        )
        results_list.append({'scale': float(scale), 'selection_details': best['selection_details']})
        if best_overall is None or tuple(best['selection_details']['ranking_key']) > tuple(best_overall['selection_details']['ranking_key']):
            best_overall = {'scale': float(scale), **best}
            save_partial_best_snapshot(
                output_dir,
                args.age_group,
                float(scale),
                best['epoch'],
                float(best['results']['behavior_optimal_score']),
                best['results'],
                best['params'],
                best['predictions']['pred_rt'],
                best['predictions']['pred_choice'],
                test_cached['target_labels'],
                test_cached['response_labels'],
                test_cached['congruency'],
                args.time_steps,
                args.data_root,
                args.output_root,
                'response_labels',
                extra_config={'best_epoch': int(best['epoch']), 'experiment_name': experiment_name},
            )

    if best_overall is None:
        raise RuntimeError('No overall best checkpoint found')

    os.makedirs(output_dir, exist_ok=True)
    config_payload = build_config_payload(best_overall, args.time_steps, experiment_name, smoke_metadata)

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(to_jsonable(config_payload), f, indent=2)
    with open(os.path.join(output_dir, 'config_smoke.json'), 'w') as f:
        json.dump(to_jsonable(config_payload), f, indent=2)
    with open(os.path.join(output_dir, 'metrics_smoke.json'), 'w') as f:
        json.dump(to_jsonable(best_overall['results']), f, indent=2)
    subset_meta = smoke_metadata.get('smoke_eval_subset_meta')
    if subset_meta is not None:
        with open(os.path.join(output_dir, 'smoke_eval_subset_meta.json'), 'w') as f:
            json.dump(to_jsonable(subset_meta), f, indent=2)
    with open(os.path.join(output_dir, 'checkpoint_ranking_summary.json'), 'w') as f:
        json.dump(to_jsonable(build_checkpoint_ranking_summary(results_list, 'balanced')), f, indent=2)

    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_overall['params'])
    np.savez_compressed(
        os.path.join(output_dir, 'predictions_smoke.npz'),
        pred_rt=best_overall['predictions']['pred_rt'].astype(np.float32),
        pred_choice=best_overall['predictions']['pred_choice'].astype(np.int64),
        choice_logits=best_overall['predictions']['choice_logits'].astype(np.float32),
        decision_times_class=best_overall['predictions']['decision_times_class'].astype(np.float32),
        true_rt=test_cached['rts'].astype(np.float32),
        target_labels=test_cached['target_labels'].astype(np.int64),
        response_labels=test_cached['response_labels'].astype(np.int64),
        congruency=test_cached['congruency'].astype(np.int64),
        rt_readout_mode=np.array('accumulator_rnn_balanced'),
        scale=np.array(float(best_overall['scale']), dtype=np.float32),
    )
    save_trajectory_samples(output_dir, best_overall['predictions'], test_cached, dt_ms=10)
    np.savez(os.path.join(output_dir, 'train_logits.npz'), **train_cached)
    np.savez(os.path.join(output_dir, 'test_logits.npz'), **test_cached)

    print(f'Best scale for {args.age_group}: {best_overall["scale"]:.3f}')
    print(f'Behavior score: {best_overall["results"]["behavior_optimal_score"]:.4f}')
    print(f'Pred mean RT: {best_overall["results"]["pred_mean"]:.3f}s')
    print(f'Trajectory shape: {best_overall["predictions"]["traj"].shape}')


if __name__ == '__main__':
    main()

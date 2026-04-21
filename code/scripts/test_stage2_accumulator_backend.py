"""Synthetic smoke test for the Stage-2 accumulator backend.

This test is intentionally lightweight and self-contained:
  - it builds a tiny synthetic cached-logits dataset (train/test)
  - it fits the AccumRNN Stage-2 head for 1-2 epochs
  - it verifies that expected artifacts are written and evaluation works
  - it checks that evaluation is stochastic across different RNG seeds

Run:
  python code/scripts/test_stage2_accumulator_backend.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from stage2_accumulator_backend import evaluate_cached_stage2_accumrnn_params, fit_stage2_accumrnn_from_logits
from train_age_groups_efficient import compute_human_stats_from_rts
from vgg_accumulator_rnn_v2 import VGGAccumulatorRNNLIMV2


def _make_synthetic_cached(
    *,
    n: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n_classes = 4

    logits = rng.normal(loc=0.0, scale=1.0, size=(int(n), n_classes)).astype(np.float32)
    target_labels = logits.argmax(axis=1).astype(np.int64)

    # Make responses mostly correct but include errors.
    response_labels = target_labels.copy()
    error_mask = rng.random(int(n)) < 0.25
    for i in np.where(error_mask)[0].tolist():
        choices = [c for c in range(n_classes) if c != int(target_labels[i])]
        response_labels[i] = int(rng.choice(choices))
    response_labels = response_labels.astype(np.int64)

    congruency = (rng.random(int(n)) < 0.5).astype(np.int64)

    # Synthetic RTs in seconds with plausible structure:
    #  - incongruent slower
    #  - errors slower
    #  - lognormal noise
    base = 0.35
    incong_penalty = 0.10
    error_penalty = 0.07
    noise = rng.lognormal(mean=-2.5, sigma=0.35, size=int(n)).astype(np.float32)
    is_error = (response_labels != target_labels).astype(np.float32)
    rts = (
        base
        + incong_penalty * congruency.astype(np.float32)
        + error_penalty * is_error
        + noise
    ).astype(np.float32)
    rts = np.clip(rts, 0.15, 2.0).astype(np.float32)

    return {
        "logits": logits,
        "rts": rts,
        "target_labels": target_labels,
        "response_labels": response_labels,
        "congruency": congruency,
    }


def _load_npz_params(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def _assert_finite_dict(d: Dict[str, object], *, name: str) -> None:
    bad = []
    for k, v in d.items():
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
            bad.append(k)
    if bad:
        raise AssertionError(f"Non-finite values in {name}: {bad}")


def _fit_and_eval_once(
    *,
    seed: int,
    choice_readout: str,
    choice_window: int = 3,
    gaussian_radius_steps: int = 6,
    gaussian_sigma_steps: float = 2.0,
    competition_mix: float = 0.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    train_cached = _make_synthetic_cached(n=96, seed=int(seed) + 1)
    test_cached = _make_synthetic_cached(n=48, seed=int(seed) + 2)
    human_stats = compute_human_stats_from_rts(train_cached["rts"])

    with tempfile.TemporaryDirectory(prefix="accumrnn-smoke-") as tmp:
        out_dir = Path(tmp)

        fit_stage2_accumrnn_from_logits(
            age_group="synthetic/user_0",
            output_dir=str(out_dir),
            human_stats=human_stats,
            train_cached=train_cached,
            test_cached=test_cached,
            device="cpu",
            scales=np.array([0.10], dtype=np.float32),
            epochs=2,
            time_steps=60,
            threshold=float(threshold),
            choice_temperature=0.05,
            accuracy_calib_weight=1.0,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
            random_seed=int(seed),
            eval_random_seed=int(seed) + 13,
            batch_size=32,
            learning_rate=1e-3,
        )

        cfg_path = out_dir / "best_config.json"
        params_path = out_dir / "best_model_params.npz"
        if not cfg_path.exists():
            raise AssertionError(f"Missing {cfg_path}")
        if not params_path.exists():
            raise AssertionError(f"Missing {params_path}")

        cfg = json.loads(cfg_path.read_text())
        if cfg.get("schema_version") != "stage2_accumrnn.best_config.v1":
            raise AssertionError(f"Unexpected best_config schema_version: {cfg.get('schema_version')}")
        if float(cfg.get("accuracy_calib_weight", float("nan"))) != 1.0:
            raise AssertionError(f"Unexpected accuracy_calib_weight in best_config: {cfg.get('accuracy_calib_weight')}")
        if cfg.get("choice_readout") != str(choice_readout):
            raise AssertionError(f"Unexpected choice_readout in best_config: {cfg.get('choice_readout')}")
        if int(cfg.get("choice_window", -1)) != int(choice_window):
            raise AssertionError(f"Unexpected choice_window in best_config: {cfg.get('choice_window')}")
        if int(cfg.get("gaussian_radius_steps", -1)) != int(gaussian_radius_steps):
            raise AssertionError(f"Unexpected gaussian_radius_steps in best_config: {cfg.get('gaussian_radius_steps')}")
        if float(cfg.get("gaussian_sigma_steps", float("nan"))) != float(gaussian_sigma_steps):
            raise AssertionError(f"Unexpected gaussian_sigma_steps in best_config: {cfg.get('gaussian_sigma_steps')}")
        if float(cfg.get("competition_mix", float("nan"))) != float(competition_mix):
            raise AssertionError(f"Unexpected competition_mix in best_config: {cfg.get('competition_mix')}")
        time_steps = int(cfg["time_steps"])
        params = _load_npz_params(params_path)

        preds_a, metrics = evaluate_cached_stage2_accumrnn_params(
            params=params,
            time_steps=time_steps,
            cached=test_cached,
            device="cpu",
            choice_temperature=0.05,
            random_seed=int(seed) + 31,
            rt_shape_focus=True,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
        )

        preds_b, _ = evaluate_cached_stage2_accumrnn_params(
            params=params,
            time_steps=time_steps,
            cached=test_cached,
            device="cpu",
            choice_temperature=0.05,
            random_seed=int(seed) + 32,
            rt_shape_focus=True,
            choice_readout=str(choice_readout),
            choice_window=int(choice_window),
            gaussian_radius_steps=int(gaussian_radius_steps),
            gaussian_sigma_steps=float(gaussian_sigma_steps),
            competition_mix=float(competition_mix),
        )

        if preds_a["pred_rt"].shape[0] != test_cached["rts"].shape[0]:
            raise AssertionError("pred_rt has unexpected shape")
        if preds_a["pred_choice"].shape[0] != test_cached["response_labels"].shape[0]:
            raise AssertionError("pred_choice has unexpected shape")
        if preds_a["choice_logits"].shape != (test_cached["logits"].shape[0], 4):
            raise AssertionError("choice_logits has unexpected shape")

        if str(choice_readout) == "first_crosser_coupled.v1":
            decision_times = preds_a["decision_times_class"]
            crossed_mask = preds_a["crossed_mask_class"]
            pred_choice = preds_a["pred_choice"]
            pred_rt = preds_a["pred_rt"]
            horizon_rt = time_steps * 10 / 1000.0
            expected_choice = decision_times.argmin(axis=1)
            expected_rt = decision_times[np.arange(decision_times.shape[0]), expected_choice]
            if not np.array_equal(pred_choice, expected_choice.astype(np.int64)):
                raise AssertionError("Coupled invariant failed: pred_choice != argmin(decision_times)")
            if not np.allclose(pred_rt, expected_rt.astype(np.float32), atol=1e-6, rtol=0.0):
                raise AssertionError("Coupled invariant failed: pred_rt != decision_time[pred_choice]")
            chosen_crossed = crossed_mask[np.arange(crossed_mask.shape[0]), pred_choice]
            if not np.all(chosen_crossed):
                raise AssertionError("Coupled invariant failed: predicted class did not truly cross threshold")
            if np.any(pred_rt >= horizon_rt - 1e-6):
                raise AssertionError("Coupled invariant failed: some trials appear to use a horizon fallback instead of a real crossing")

        _assert_finite_dict(metrics, name="metrics")
        return preds_a["pred_rt"].copy(), preds_b["pred_rt"].copy(), metrics


def main() -> None:
    variants = [
        {
            "name": "baseline-windowed",
            "choice_readout": "windowed_state_at_decision.v1",
            "choice_window": 3,
            "gaussian_radius_steps": 6,
            "gaussian_sigma_steps": 2.0,
            "competition_mix": 0.0,
        },
        {
            "name": "first-crosser-coupled",
            "choice_readout": "first_crosser_coupled.v1",
            "choice_window": 3,
            "gaussian_radius_steps": 6,
            "gaussian_sigma_steps": 2.0,
            "competition_mix": 0.0,
            "threshold": 0.2,
        },
        {
            "name": "threshold-relative",
            "choice_readout": "threshold_relative_windowed_state_at_decision.v1",
            "choice_window": 3,
            "gaussian_radius_steps": 6,
            "gaussian_sigma_steps": 2.0,
            "competition_mix": 0.0,
        },
        {
            "name": "gaussian-pooling",
            "choice_readout": "gaussian_pooled_state_at_decision.v1",
            "choice_window": 3,
            "gaussian_radius_steps": 6,
            "gaussian_sigma_steps": 2.0,
            "competition_mix": 0.0,
        },
        {
            "name": "competition-normalized",
            "choice_readout": "windowed_state_at_decision.v1",
            "choice_window": 3,
            "gaussian_radius_steps": 6,
            "gaussian_sigma_steps": 2.0,
            "competition_mix": 1.0,
        },
    ]

    for idx, variant in enumerate(variants, start=1):
        variant_kwargs = {k: v for k, v in variant.items() if k != "name"}
        pred_rt_a, pred_rt_b, metrics_a = _fit_and_eval_once(seed=123 + idx * 10, **variant_kwargs)
        if not np.isfinite(pred_rt_a).all():
            raise AssertionError(f"Non-finite pred_rt for variant {variant['name']}")
        if np.allclose(pred_rt_a, pred_rt_b, atol=0.0, rtol=0.0):
            raise AssertionError(f"Expected stochastic pred_rt across different seeds for variant {variant['name']}; got identical outputs")
        required_metric_keys = {"behavior_optimal_score", "rt_shape_score", "response_agreement"}
        missing = sorted(required_metric_keys - set(metrics_a.keys()))
        if missing:
            raise AssertionError(f"Missing expected metric keys for variant {variant['name']}: {missing}")

    # Direct model-level coupled invariant check.
    model = VGGAccumulatorRNNLIMV2(pretrained=False, freeze_features=False, n_classes=4, choice_readout="first_crosser_coupled.v1")
    dummy_x = torch.randn(4, 3, 224, 224)
    logits, decision_times, final_dt, traj, threshold, pred_choice = model(dummy_x, return_logits=True)
    crossed_mask = (traj > threshold.view(1, 1, 1)).any(dim=1)
    chosen_crossed = crossed_mask[torch.arange(crossed_mask.size(0)), pred_choice]
    if not torch.all(chosen_crossed):
        raise AssertionError("Model-level coupled invariant failed: predicted class did not truly cross")
    expected_rt = decision_times[torch.arange(decision_times.size(0)), pred_choice]
    if not torch.allclose(final_dt, expected_rt, atol=1e-6, rtol=0.0):
        raise AssertionError("Model-level coupled invariant failed: final_dt != decision_times[pred_choice]")

    print("OK: stage2_accumulator_backend synthetic smoke test passed for baseline + 3 variants")


if __name__ == "__main__":
    main()

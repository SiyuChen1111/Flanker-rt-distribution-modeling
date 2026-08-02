from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from vgg_wongwang_lim import WongWangMultiClassDecision  # noqa: E402


def test_inference_reports_horizon_for_each_class_that_never_crosses() -> None:
    """Per-class decision times must not turn a never-crossed class into a 0 ms decision."""
    model = WongWangMultiClassDecision(n_classes=4, dt=10, time_steps=200, t_stimulus=200)
    model.eval()
    with torch.no_grad():
        model.noise_ampa.fill_(0.0)
        model.threshold.fill_(0.5)

    input_signal = torch.tensor([[1.6, 0.4, 0.4, 0.4]], dtype=torch.float32)
    with torch.no_grad():
        decision_times, trajectory, threshold = model.inference(input_signal)

    crossed = (trajectory > threshold).any(dim=1).squeeze(0)
    assert crossed.tolist() == [True, False, False, False]
    assert decision_times[0, 0].item() == pytest.approx(0.43, abs=0.02)
    assert decision_times[0, 1:].tolist() == pytest.approx([1.99, 1.99, 1.99])


def test_normalized_competition_preserves_four_choice_weak_evidence_decisions() -> None:
    """Adding alternatives must not triple the total lateral inhibition per population."""
    model = WongWangMultiClassDecision(
        n_classes=4,
        dt=10,
        time_steps=200,
        t_stimulus=200,
        normalize_competition=True,
    )
    model.eval()
    with torch.no_grad():
        model.noise_ampa.fill_(0.02)
        model.threshold.fill_(0.5)

    input_signal = torch.ones(256, 4, dtype=torch.float32)
    input_signal[:, 0] += 0.1
    input_signal[:, 1:] -= 0.1 / 3.0
    generator = torch.Generator().manual_seed(20260802)
    with torch.no_grad():
        _, trajectory, threshold = model.inference(input_signal, generator=generator)

    crossed = (trajectory > threshold).any(dim=2)
    crossing_rate = crossed.any(dim=1).float().mean().item()
    first_step = torch.where(
        crossed,
        torch.arange(trajectory.shape[1]).view(1, -1),
        trajectory.shape[1],
    ).amin(dim=1)
    crossing_state = trajectory[
        torch.arange(trajectory.shape[0]),
        first_step.clamp_max(trajectory.shape[1] - 1),
    ]
    target_choice_rate = (crossing_state.argmax(dim=1)[first_step < trajectory.shape[1]] == 0).float().mean().item()

    assert crossing_rate > 0.95
    assert target_choice_rate > 0.50

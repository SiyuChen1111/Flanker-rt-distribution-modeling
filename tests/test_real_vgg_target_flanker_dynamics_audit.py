from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_real_vgg_target_flanker_dynamics_audit import (  # noqa: E402
    channel_gap,
    first_stable_recovery,
    temporal_pattern,
)


def test_channel_gap_handles_layer_and_time_arrays() -> None:
    target = np.array([0, 1])
    flanker = np.array([1, 0])
    layer = np.array([[3.0, 1.0], [2.0, 5.0]])
    assert np.allclose(channel_gap(layer, target, flanker), [2.0, 3.0])

    timed = np.stack([layer, layer + 1.0], axis=1)
    assert np.allclose(channel_gap(timed, target, flanker), [[2.0, 2.0], [3.0, 3.0]])


def test_recovery_requires_prior_flanker_advantage_and_sustained_target_advantage() -> None:
    gap = np.array(
        [
            [-1.0, -0.5, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [-1.0, 0.2, -0.1, 0.3, 0.4],
            [-1.0, -0.5, -0.2, -0.1, -0.3],
        ]
    )
    recovered = first_stable_recovery(gap, sustained_k=2)
    assert recovered[0] == 2
    assert np.isnan(recovered[1])
    assert recovered[2] == 3
    assert np.isnan(recovered[3])

    patterns = temporal_pattern(gap[:, :2].mean(axis=1), gap[:, -2:].mean(axis=1))
    assert patterns.tolist() == [
        "early_flanker_late_target",
        "target_throughout",
        "early_flanker_late_target",
        "flanker_throughout",
    ]

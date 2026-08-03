import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_r5_choice_rule_alignment_audit import ordered_bins, semantic_outcome
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout


def test_semantic_outcome_keeps_other_choices_distinct():
    choice = np.array([0, 1, 3])
    target = np.array([0, 0, 0])
    flanker = np.array([2, 1, 2])
    assert semantic_outcome(choice, target, flanker).tolist() == ["target", "flanker", "other"]


def test_ordered_bins_are_equal_count_and_deterministic_with_ties():
    values = np.ones(10)
    first = ordered_bins(values, n_bins=5)
    second = ordered_bins(values, n_bins=5)
    assert np.array_equal(first, second)
    assert np.bincount(first)[1:].tolist() == [2, 2, 2, 2, 2]


def test_default_readout_choice_is_bound_to_the_rt_step():
    # Flanker wins when the boundary is first reached, but target peaks later.
    trajectory = np.array([[[0.60, 0.80], [0.70, 0.75], [0.90, 0.70]]], dtype=np.float32)
    outputs = {"trajectory": trajectory, "evidence_traj": trajectory - 0.50}
    base = pd.DataFrame({"pred_choice": [0], "target_label": [0]})
    result = apply_readout(
        base,
        outputs,
        cfg=ReadoutConfig("sustained_crossing", sustained_k=1, margin=0.0),
        threshold=0.50,
        dt_ms=10,
        t0_seconds=0.0,
    )
    assert int(result.loc[0, "pred_choice"]) == 1
    assert int(result.loc[0, "readout_step"]) == 0
    assert bool(result.loc[0, "crossed"])

    legacy = apply_readout(
        base,
        outputs,
        cfg=ReadoutConfig("sustained_crossing", sustained_k=1, margin=0.0),
        threshold=0.50,
        dt_ms=10,
        t0_seconds=0.0,
        choice_rule="trajectory_max_choice",
    )
    assert int(legacy.loc[0, "pred_choice"]) == 0


def test_no_crossing_is_flagged_and_uses_deadline_only_as_a_sentinel():
    trajectory = np.array([[[0.10, 0.20], [0.15, 0.25], [0.20, 0.30]]], dtype=np.float32)
    outputs = {"trajectory": trajectory, "evidence_traj": trajectory - 0.90}
    base = pd.DataFrame({"pred_choice": [1], "target_label": [0]})
    result = apply_readout(
        base,
        outputs,
        cfg=ReadoutConfig("sustained_crossing", sustained_k=1, margin=0.0),
        threshold=0.90,
        dt_ms=10,
        t0_seconds=0.0,
    )
    assert not bool(result.loc[0, "crossed"])
    assert int(result.loc[0, "readout_step"]) == 2
    assert np.isclose(float(result.loc[0, "decision_time"]), 0.02)

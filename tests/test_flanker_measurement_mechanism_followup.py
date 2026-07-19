from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code/scripts/run_flanker_measurement_mechanism_followup.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("flanker_followup", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_presentation_floor_uses_left_closed_grid():
    delta = 1 / 60
    values = np.array([0.0, delta - 1e-8, delta, 2 * delta + 0.001])
    got = mod.presentation_floor(values, 60)
    np.testing.assert_allclose(got, [0.0, 0.0, delta, 2 * delta], atol=1e-10)


def test_keyboard_center_and_combined_order():
    values = np.array([0.500, 0.508, 0.516])
    centered = mod.keyboard_center(values, 125)
    expected = np.rint(values / 0.008) * 0.008 - 0.004
    np.testing.assert_allclose(centered, expected)
    np.testing.assert_allclose(
        mod.combined_hardware_bin(values, 60, 125),
        mod.presentation_floor(expected, 60),
    )


def test_first_passage_couples_choice_and_time_and_marks_deadline():
    trajectory = np.array(
        [
            [[0.0, 0.0], [0.2, 0.1], [0.3, 0.4]],
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.3]],
        ]
    )
    result = mod.first_passage_readout(trajectory, 0.35, dt=0.01)
    assert result.step.tolist() == [2, 2]
    assert result.choice.tolist() == [1, 1]
    assert result.deadline_response.tolist() == [False, True]
    np.testing.assert_allclose(result.time, [0.02, 0.02])


def test_first_passage_respects_minimum_decision_time():
    trajectory = np.array([[[0.5, 0.1], [0.2, 0.6], [0.8, 0.1]]])
    result = mod.first_passage_readout(
        trajectory, 0.4, dt=0.01, min_decision_time=0.01
    )
    assert result.step.item() == 1
    assert result.choice.item() == 1


def test_error_sources_are_mutually_exclusive_and_cover_errors():
    # Trial 0 recovers after an early wrong stop; trial 1 never recovers;
    # trial 2 is correct at stop but later response is flipped.
    trajectory = np.array(
        [
            [[0.1, 0.4], [0.5, 0.2], [0.6, 0.1]],
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            [[0.5, 0.1], [0.6, 0.2], [0.7, 0.3]],
        ]
    )
    got = mod.classify_error_source(
        trajectory, stop_step=[0, 0, 0], final_choice=[1, 1, 1], target=[0, 0, 0]
    )
    assert got.tolist() == [
        "premature_commitment",
        "persistent_dynamics",
        "post_stop_flip",
    ]


def test_subject_equal_aggregation_does_not_weight_more_trials_more():
    subject = pd.DataFrame(
        {
            "source": ["human", "human"],
            "seed_id": ["observed", "observed"],
            "analysis_group": ["young_20_29"] * 2,
            "congruency": [1, 1],
            "rt_bin": [1, 1],
            "user_id": ["a", "b"],
            "n_trials": [100, 10],
            "mean_rt": [0.5, 0.7],
            "median_rt": [0.5, 0.7],
            "error_rate": [0.0, 1.0],
        }
    )
    got = mod.subject_equal_group_profile(subject).iloc[0]
    assert got.error_rate == pytest.approx(0.5)
    assert got.mean_rt == pytest.approx(0.6)
    assert got.n_trials == 110


def test_common_random_draws_are_trial_keyed_and_reproducible():
    human = pd.DataFrame(
        {
            "trial_uid": [7, 3, 9, 4],
            "analysis_group": ["young_20_29", "young_20_29", "older_80_89", "older_80_89"],
        }
    )
    first = mod.make_common_random_draws(human, 2).sort_values("trial_uid").reset_index(drop=True)
    second = mod.make_common_random_draws(human.sample(frac=1, random_state=4), 2).sort_values("trial_uid").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert first.trial_uid.is_unique
    assert first[[f"choice_noise_{c}" for c in range(4)]].nunique().min() > 1


def test_split_manifest_has_no_participant_overlap():
    split_path = mod.DUAL_DIR / "split_manifest.csv"
    if not split_path.exists():
        pytest.skip("formal split output not available")
    splits = pd.read_csv(split_path, dtype={"user_id": str})
    for fold in splits.fold.unique():
        train = set(splits[(splits.fold == fold) & (splits.role == "train")].user_id)
        test = set(splits[(splits.fold == fold) & (splits.role == "test")].user_id)
        assert not train & test
        assert len(test) == 4


def test_human_master_complete_join_key_is_unique():
    human = mod.load_human_master()
    assert not human.duplicated(mod.JOIN_KEYS).any()
    assert human.trial_uid.nunique() == len(human)


def test_full_mode_seed_gate_is_explicit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--mode", "full", "--seeds", "9", "--run-id", "never-created"],
    )
    with pytest.raises(ValueError, match="at least 10 seeds"):
        mod.main()

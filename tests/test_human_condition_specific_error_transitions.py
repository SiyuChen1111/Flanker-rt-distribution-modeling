from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))
SPEC = importlib.util.spec_from_file_location(
    "transition_audit", ROOT / "code/scripts/run_human_condition_specific_error_transitions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_lagged_frame_does_not_cross_sessions_or_cleaning_gaps() -> None:
    data = pd.DataFrame({
        "user_id": [1] * 7, "nth_play": [1, 1, 1, 1, 2, 2, 2],
        "trial": [1, 2, 4, 5, 1, 2, 3], "correct": [True, False, True, True, True, False, True],
        "congruency": ["congruent"] * 7,
    })
    lag1 = MODULE.lagged_frame(data, 1)
    assert list(zip(lag1.nth_play, lag1.trial)) == [(1, 2), (1, 5), (2, 2), (2, 3)]
    lag2 = MODULE.lagged_frame(data, 2)
    assert list(zip(lag2.nth_play, lag2.trial)) == [(2, 3)]


def test_effect_estimation_detects_positive_risk_difference() -> None:
    rows = []
    for uid in range(1, 9):
        rows.extend([
            {"user_id": uid, "previous_error": False, "n_trials": 1000, "n_errors": 20 + uid},
            {"user_id": uid, "previous_error": True, "n_trials": 100, "n_errors": 10 + uid},
        ])
    detail, summary = MODULE.estimate_effect_from_counts(pd.DataFrame(rows), np.random.default_rng(1))
    assert len(detail) == 8
    assert summary["population_mean_risk_difference"] > 0
    assert summary["n_positive"] == 8


def test_pattern_classifier_identifies_general_positive_state() -> None:
    summary = pd.DataFrame({
        "transition": ["C_to_C", "I_to_C", "C_to_I", "I_to_I"],
        "population_mean_risk_difference": [.07, .04, .06, .05],
    })
    classification, _ = MODULE.classify_pattern(summary)
    assert classification.startswith("T1")

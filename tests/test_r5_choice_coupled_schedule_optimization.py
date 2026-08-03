import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_natural_layer_to_time_var_ww_diagnostic import schedule_weights
from run_r5_choice_coupled_schedule_optimization import compressed_schedule, ordered_bins


def test_uncompressed_schedule_matches_original_mapping():
    expected = schedule_weights("natural_smooth_5stage", 80).to_numpy()
    actual = compressed_schedule(1.0, 0.0, 1.0).to_numpy()
    assert np.allclose(actual, expected, atol=1e-6)


def test_compression_moves_final_layer_weight_earlier():
    full = compressed_schedule(1.0)
    compressed = compressed_schedule(0.4)
    full_final_dominance = int(np.flatnonzero(full.to_numpy().argmax(axis=1) == 4)[0])
    compressed_final_dominance = int(np.flatnonzero(compressed.to_numpy().argmax(axis=1) == 4)[0])
    assert compressed_final_dominance < full_final_dominance
    assert compressed.iloc[40:]["final"].mean() > compressed.iloc[40:]["conv3"].mean()


def test_ordered_bins_are_balanced_and_deterministic():
    values = np.ones(10)
    first = ordered_bins(values)
    second = ordered_bins(values)
    assert np.array_equal(first, second)
    assert np.bincount(first)[1:].tolist() == [2, 2, 2, 2, 2]

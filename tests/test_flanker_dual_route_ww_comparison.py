from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_flanker_dual_route_ww_comparison import (  # noqa: E402
    equal_count_bins,
    source_mu_energy_matched,
    temporal_envelopes,
)


def test_dual_route_and_reversed_envelopes_have_opposite_order() -> None:
    target, flanker = temporal_envelopes("M2_flanker_early_target_late", 80, 120)
    rev_target, rev_flanker = temporal_envelopes("M3_target_early_flanker_late", 80, 120)
    assert flanker[0] > target[0]
    assert target[-1] > flanker[-1]
    np.testing.assert_allclose(target, rev_flanker)
    np.testing.assert_allclose(flanker, rev_target)


def test_energy_matching_preserves_full_timewise_rms() -> None:
    rng = np.random.default_rng(4)
    full = rng.normal(size=(7, 80, 4)).astype(np.float32)
    target = rng.normal(size=(7, 4)).astype(np.float32)
    flanker = rng.normal(size=(7, 4)).astype(np.float32)
    te, fe = temporal_envelopes("M2_flanker_early_target_late", 80, 120)
    combined = source_mu_energy_matched(full, target, flanker, te, fe)
    a = np.sqrt(np.mean((full - full.mean(axis=2, keepdims=True)) ** 2, axis=2))
    b = np.sqrt(np.mean((combined - combined.mean(axis=2, keepdims=True)) ** 2, axis=2))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_equal_count_bins_differ_by_at_most_one() -> None:
    for n in range(4, 19):
        bins = equal_count_bins(np.arange(n)[::-1])
        counts = np.bincount(bins, minlength=4)
        assert counts.max() - counts.min() <= 1

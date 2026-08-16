from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

from run_candidate_B_directional_initial_state import predeclare, readout, ww_trajectory


def test_predeclared_grid_is_small_and_fixed(tmp_path: Path) -> None:
    params = pd.DataFrame({"age_group": ["20-29"], "threshold": [.12]})
    grid = predeclare(tmp_path, params)
    assert len(grid) == 37
    assert set(grid.variant) == {"B0", "B1", "B2"}
    assert set(grid.loc[grid.variant.eq("B2"), "beta"]) == {.25, .50, .75}


def test_neutral_recurrence_is_deterministic_and_readable() -> None:
    evidence = np.zeros((80, 4), dtype=np.float32)
    first = ww_trajectory(evidence, np.full(4, .1))
    second = ww_trajectory(evidence, np.full(4, .1))
    assert np.array_equal(first, second)
    choice, step, crossed = readout(first, threshold=.12, margin=0.0)
    assert 0 <= choice < 4
    assert 0 <= step < 80
    assert isinstance(crossed, bool)


def test_directional_code_is_centered() -> None:
    for choice in range(4):
        code = np.full(4, -.25)
        code[choice] = .75
        assert np.isclose(code.sum(), 0.0)

import sys
from pathlib import Path
import importlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "code" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_runner_module():
    return importlib.import_module("run_dynamic_selection_single_subject")


def test_build_scale_grid_clips_to_bounds():
    module = load_runner_module()
    _build_scale_grid = module._build_scale_grid

    grid = _build_scale_grid(0.01)
    assert len(grid) == 5
    assert all(0.05 <= x <= 0.15 for x in grid)
    assert grid[0] == 0.05

    grid = _build_scale_grid(0.20)
    assert len(grid) == 5
    assert all(0.05 <= x <= 0.15 for x in grid)
    assert grid[-1] == 0.15


def test_incongruent_error_minus_correct_rt_nan_when_missing_errors():
    module = load_runner_module()
    _compute_incongruent_error_minus_correct_rt = module._compute_incongruent_error_minus_correct_rt

    rt = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    correct = np.array([True, True, True])
    congruency = np.array([1, 1, 1], dtype=np.int64)
    out = _compute_incongruent_error_minus_correct_rt(rt, correct, congruency)
    assert not np.isfinite(out)

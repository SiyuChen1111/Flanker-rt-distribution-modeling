import sys
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "code" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_train_helper():
    module = importlib.import_module("train_age_groups_efficient")
    return module.attach_flanker_labels_from_csv


def load_selection_helper():
    module = importlib.import_module("vgg_wongwang_lim")
    return module.build_dynamic_stage2_input


def load_alignment_helper():
    module = importlib.import_module("run_dynamic_selection_single_subject")
    return module.build_alignment_report


def test_dynamic_selection_noop_when_disabled():
    build_dynamic_stage2_input = load_selection_helper()
    logits = torch.tensor([[0.5, 0.2, 0.1, 0.0]], dtype=torch.float32)
    scale = torch.tensor(0.1, dtype=torch.float32)
    output, traces = build_dynamic_stage2_input(logits, scale, time_steps=5, config=None)
    expected = torch.relu(logits * scale)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected)
    assert traces == {}


def test_dynamic_selection_suppresses_incongruent_flanker_over_time():
    build_dynamic_stage2_input = load_selection_helper()
    logits = torch.tensor([[1.0, 0.8, 0.2, 0.1]], dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    target_labels = torch.tensor([0], dtype=torch.long)
    flanker_labels = torch.tensor([1], dtype=torch.long)
    config = {
        "selection_mode": "dynamic_flanker_suppression",
        "selection_strength": 0.5,
        "selection_midpoint_s": 0.02,
        "selection_tau_s": 0.01,
        "target_boost": 0.0,
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }
    output, traces = build_dynamic_stage2_input(
        logits,
        scale,
        time_steps=6,
        config=config,
        target_labels=target_labels,
        flanker_labels=flanker_labels,
    )
    assert output.shape == (1, 6, 4)
    flanker_series = output[0, :, 1].detach().cpu().numpy()
    assert flanker_series[0] > flanker_series[-1]
    assert int(traces["selection_trial_mask"][0].item()) == 1


def test_dynamic_selection_dmc_like_adds_early_capture_then_late_control():
    build_dynamic_stage2_input = load_selection_helper()
    logits = torch.tensor([[1.0, 0.9, 0.2, 0.1]], dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    target_labels = torch.tensor([0], dtype=torch.long)
    flanker_labels = torch.tensor([1], dtype=torch.long)
    config = {
        "selection_mode": "dynamic_flanker_dmc_like",
        "selection_strength": 0.5,
        "selection_midpoint_s": 0.06,
        "selection_tau_s": 0.02,
        "target_boost": 0.1,
        "auto_strength": 0.25,
        "auto_peak_s": 0.02,
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }
    output, traces = build_dynamic_stage2_input(
        logits,
        scale,
        time_steps=8,
        config=config,
        target_labels=target_labels,
        flanker_labels=flanker_labels,
    )
    flanker_series = output[0, :, 1].detach().cpu().numpy()
    target_series = output[0, :, 0].detach().cpu().numpy()
    auto_pulse = traces["auto_pulse_t"].detach().cpu().numpy()
    assert output.shape == (1, 8, 4)
    assert auto_pulse.max() > 0.0
    assert flanker_series[1] > flanker_series[0]
    assert flanker_series[-1] < flanker_series[1]
    assert target_series[1] < target_series[0]
    assert target_series[-1] > target_series[1]


def test_attach_flanker_labels_from_csv(tmp_path):
    attach_flanker_labels_from_csv = load_train_helper()
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "stimulus_image_path": ["a.png", "b.png"],
        "response_time": [500, 600],
        "target_direction": ["L", "R"],
        "response_direction": ["L", "R"],
        "flanker_direction": ["R", "U"],
    }).to_csv(csv_path, index=False)
    cached = {
        "logits": np.zeros((2, 4), dtype=np.float32),
        "rts": np.array([0.5, 0.6], dtype=np.float32),
        "rts_normalized": np.array([0.1, 0.2], dtype=np.float32),
        "target_labels": np.array([0, 1], dtype=np.int64),
        "response_labels": np.array([0, 1], dtype=np.int64),
        "congruency": np.array([1, 1], dtype=np.int64),
    }
    enriched = attach_flanker_labels_from_csv(cached, str(csv_path))
    assert "flanker_labels" in enriched
    assert enriched["flanker_labels"].tolist() == [1, 2]


def test_alignment_report_detects_row_order_mismatch():
    build_alignment_report = load_alignment_helper()
    df = pd.DataFrame(
        {
            "user_id": [10, 10, 11],
            "target_direction": ["L", "R", "U"],
            "response_direction": ["L", "R", "U"],
            "flanker_direction": ["R", "R", "D"],
            "response_time": [500, 600, 700],
        }
    )
    cached = {
        "logits": np.zeros((3, 4), dtype=np.float32),
        "rts": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "rts_normalized": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "target_labels": np.array([0, 1, 2], dtype=np.int64),
        "response_labels": np.array([0, 1, 2], dtype=np.int64),
        "flanker_labels": np.array([1, 1, 3], dtype=np.int64),
        "congruency": np.array([1, 0, 1], dtype=np.int64),
    }

    report = build_alignment_report(df, cached)
    assert report["alignment_ok"].all()

    permuted = df.iloc[[1, 0, 2]].reset_index(drop=True)
    try:
        build_alignment_report(permuted, cached)
        assert False, "Expected alignment mismatch ValueError"
    except ValueError as exc:
        assert "CSV_NPZ_ALIGNMENT_MISMATCH" in str(exc)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_flanker_dual_source_followup import (  # noqa: E402
    SourceSeparatedRenderer,
    balanced_manifest_sample,
    directional_margin,
    source_separation_pass,
)


def example_row() -> pd.Series:
    return pd.Series(
        {
            "xpos": 320,
            "ypos": 240,
            "layout": 0,
            "target_label": 0,
            "flanker_label": 1,
        }
    )


def test_source_variants_reconstruct_clean_background_components() -> None:
    renderer = SourceSeparatedRenderer()
    background = np.asarray(renderer.background.convert("RGB"), dtype=np.int16)
    full = np.asarray(renderer.render_image(example_row(), "full"), dtype=np.int16)
    target = np.asarray(renderer.render_image(example_row(), "target_only"), dtype=np.int16)
    flanker = np.asarray(renderer.render_image(example_row(), "flanker_only"), dtype=np.int16)
    assert np.any(target != background)
    assert np.any(flanker != background)
    assert np.any(full != target)
    assert np.any(full != flanker)
    # Full is exactly the union of source-specific changes over the same background.
    target_changed = np.any(target != background, axis=2)
    flanker_changed = np.any(flanker != background, axis=2)
    full_changed = np.any(full != background, axis=2)
    assert np.array_equal(full_changed, target_changed | flanker_changed)


def test_balanced_sample_is_deterministic_and_unique() -> None:
    rows = []
    idx = 0
    for congruency in [0, 1]:
        for target in range(4):
            for flanker in range(4):
                for _ in range(3):
                    rows.append({"subset_stimulus_id": idx, "congruency": congruency, "target_label": target, "flanker_label": flanker})
                    idx += 1
    df = pd.DataFrame(rows)
    a = balanced_manifest_sample(df, 40)
    b = balanced_manifest_sample(df, 40)
    assert len(a) == 40
    assert a["subset_stimulus_id"].is_unique
    assert a["subset_stimulus_id"].tolist() == b["subset_stimulus_id"].tolist()


def test_directional_margin_uses_best_competing_direction() -> None:
    evidence = np.array([[0.7, 0.5, 0.1, 0.2], [0.4, 0.2, 0.6, 0.3]])
    labels = np.array([0, 1])
    margin = directional_margin(evidence, labels)
    np.testing.assert_allclose(margin, [0.2, -0.4])


def test_source_gate_requires_reliable_nonfinal_layer_for_each_source() -> None:
    rows = []
    for variant, best in [("target_only", 0.91), ("flanker_only", 0.94)]:
        rows.extend(
            [
                {"variant": variant, "layer": "conv3", "condition": "all", "direction_accuracy": best, "mean_expected_margin": 0.2},
                {"variant": variant, "layer": "final", "condition": "all", "direction_accuracy": 1.0, "mean_expected_margin": 0.3},
            ]
        )
    audit = pd.DataFrame(rows)
    assert source_separation_pass(audit)
    audit.loc[(audit["variant"] == "target_only") & (audit["layer"] == "conv3"), "direction_accuracy"] = 0.89
    assert not source_separation_pass(audit)

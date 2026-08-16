#!/usr/bin/env python3
"""Causal canonical choice/RT readout with no legacy choice-rule switch."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout


CANONICAL_CHOICE_RULE = "winner_at_commitment_completion"


def apply_canonical_readout(
    base_df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    *,
    cfg: ReadoutConfig,
    threshold: float,
    dt_ms: int,
    t0_seconds: float,
) -> pd.DataFrame:
    """Bind choice and time to the completed sustained-crossing event.

    ``apply_readout`` retains the historical convention in which ``readout_step``
    is the start of a qualifying sustained window.  The canonical API preserves
    that value as ``window_start_step`` and records the causal commitment at the
    inclusive window end: ``start + sustained_k - 1``.  Indices are zero-based.
    No-crossing trials retain the final-step censoring sentinel.
    """
    if cfg.readout_rule not in {"sustained_crossing", "sustained_margin"}:
        raise ValueError("The causal canonical API requires a sustained-crossing readout rule.")
    if int(cfg.sustained_k) < 1:
        raise ValueError("sustained_k must be at least 1.")
    out = apply_readout(
        base_df,
        outputs,
        cfg=cfg,
        threshold=threshold,
        dt_ms=dt_ms,
        t0_seconds=t0_seconds,
        choice_rule="winner_at_readout",
    )
    trajectory = np.asarray(outputs["trajectory"])
    rows = np.arange(len(out))
    window_start = out["readout_step"].to_numpy(dtype=np.int64)
    crossed = out["crossed"].to_numpy(dtype=bool)
    commitment = window_start.copy()
    commitment[crossed] += int(cfg.sustained_k) - 1
    if np.any(commitment >= trajectory.shape[1]):
        raise RuntimeError("Commitment completion falls outside the available trajectory.")
    expected_choice = trajectory[rows, commitment].argmax(axis=1)
    expected_decision_time = commitment.astype(float) * (float(dt_ms) / 1000.0)
    out["window_start_step"] = window_start
    out["commitment_step"] = commitment
    out["readout_step"] = commitment
    out["pred_choice"] = expected_choice
    out["decision_time"] = expected_decision_time
    out["pred_rt"] = expected_decision_time + float(t0_seconds)
    out["model_correct"] = expected_choice == out["target_label"].to_numpy(dtype=np.int64)
    out["choice_rule"] = CANONICAL_CHOICE_RULE
    out["commitment_timestamp_semantics"] = "sustained_window_completion"
    if not np.array_equal(out["pred_choice"].to_numpy(dtype=np.int64), expected_choice):
        raise RuntimeError("Canonical choice is not the winner at commitment completion.")
    if not np.allclose(out["decision_time"].to_numpy(dtype=float), expected_decision_time):
        raise RuntimeError("Canonical decision time is not derived from commitment completion.")
    if not out["choice_rule"].eq(CANONICAL_CHOICE_RULE).all():
        raise RuntimeError("Legacy choice semantics entered the canonical readout path.")
    return out

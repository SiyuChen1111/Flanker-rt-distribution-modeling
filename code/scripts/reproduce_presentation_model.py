#!/usr/bin/env python3
"""Small, non-destructive entry point for the retained presentation model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CONFIG = PROJECT_ROOT / "configs/presentation_model.json"
ROOT = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
CORRECTED = PROJECT_ROOT / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["young", "older", "all"], default="all")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts/results/presentation_model_reproduction"))
    return p.parse_args()


def selected_groups(value: str) -> set[str]:
    return {"young_20_29", "older_80_89"} if value == "all" else {"young_20_29" if value == "young" else "older_80_89"}


def validate_tables(groups: set[str]) -> None:
    original = pd.read_csv(ROOT / "best_model_R5_combined_best/results/best_model_trial_level_predictions.csv")
    corrected = pd.read_csv(CORRECTED / "selected_trial_level_predictions.csv")
    if len(original) != 10000 or len(corrected) != 10000:
        raise RuntimeError("Expected 10,000 trials in both retained result tables.")
    for frame, name in [(original, "original R5"), (corrected, "corrected-equivalent")]:
        counts = frame["analysis_group"].value_counts().to_dict()
        if any(counts.get(group, 0) != 5000 for group in groups):
            raise RuntimeError(f"{name} is missing a 5,000-trial group: {counts}")
    if "crossed" in corrected and not corrected["crossed"].astype(bool).all():
        raise RuntimeError("Corrected-equivalent table contains censored trials in a plotted result.")
    print(json.dumps({"original_rows": len(original), "corrected_rows": len(corrected), "groups": sorted(groups)}, indent=2))


def smoke_test() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from vgg_wongwang_lim import DiffDecisionMultiClass, WongWangMultiClassDecision  # noqa: WPS433
    import torch

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["vgg_layers"] != ["conv3", "conv4", "conv5", "pooled", "final"]:
        raise RuntimeError("Presentation layer fingerprint changed.")
    model = WongWangMultiClassDecision(n_classes=4, dt=10, time_steps=4, t_stimulus=40)
    model.noise_ampa.data.zero_()
    signal = torch.zeros((2, 4, 4), dtype=torch.float32)
    out = model(signal)
    if out.shape != (2, 4):
        raise RuntimeError(f"Unexpected Wong-Wang smoke output shape: {tuple(out.shape)}")
    if not issubclass(DiffDecisionMultiClass, torch.autograd.Function):
        raise RuntimeError("DiffDecisionMultiClass import failed.")
    print("model import and Wong-Wang smoke test passed")


def plot(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, str(SCRIPT_DIR / "plot_r5_caf_and_delta_curves.py"), "--output-dir", str(output_dir / "caf_delta")],
        [sys.executable, str(SCRIPT_DIR / "plot_r5_rt_distribution_kde.py"), "--output-dir", str(output_dir / "rt_distribution")],
        [sys.executable, str(SCRIPT_DIR / "run_real_vgg_target_flanker_dynamics_audit.py"), "--output-dir", str(output_dir / "mechanism")],
    ]
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    validate_tables(selected_groups(args.group))
    if args.smoke_test:
        smoke_test()
    if args.plot_only:
        plot(Path(args.output_dir))
    elif not args.analysis_only and not args.smoke_test:
        print("Validated retained results. Use --plot-only for figures or the documented full run for refitting.")


if __name__ == "__main__":
    main()

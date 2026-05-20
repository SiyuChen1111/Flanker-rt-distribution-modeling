from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT, RESULTS_ROOT
from run_subject_level_dmc_var_ww import ALLOWED_ARMS, analyze_panel_mode


DEFAULT_INPUT_ROOT = RESULTS_ROOT / "per_subject_age_comparison_plan" / "01_subject_level_dmc_var_ww"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze comparator-arm subject outputs and export locked Stage-1 signature/gate artifacts."
    )
    parser.add_argument("--input_root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--comparator_arm", default=None, choices=ALLOWED_ARMS)
    parser.add_argument("--comparator_arms", default=",".join(ALLOWED_ARMS))
    return parser.parse_args()


def _resolve_input_root(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _resolve_arms(args: argparse.Namespace) -> tuple[str, ...]:
    if args.comparator_arm:
        return (str(args.comparator_arm),)
    arms = tuple(str(arm).strip() for arm in str(args.comparator_arms).split(",") if str(arm).strip())
    if not arms:
        raise ValueError("NO_COMPARATOR_ARMS_REQUESTED")
    for arm in arms:
        if arm not in ALLOWED_ARMS:
            raise ValueError(f"UNSUPPORTED_COMPARATOR_ARM: {arm}")
    return arms


def main() -> None:
    args = parse_args()
    input_root = _resolve_input_root(args.input_root)
    comparator_arms = _resolve_arms(args)
    analyze_panel_mode(output_root=input_root, arm_names=comparator_arms)


if __name__ == "__main__":
    main()

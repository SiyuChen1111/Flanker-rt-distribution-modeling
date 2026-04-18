import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from project_paths import RESULTS_ROOT
from train_age_groups_efficient import to_jsonable


AGE_GROUPS = ("20-29", "80-89")
DEFAULT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "true_single_subject_feasibility"
EVIDENCE_ROOT = Path(".sisyphus") / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze true single-subject feasibility outputs and write scorecard, verdict, and summary.")
    parser.add_argument("--input_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _write_evidence(task: int, slug: str, payload: dict) -> Path:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_ROOT / f"task-{task}-{slug}.json"
    out.write_text(json.dumps(to_jsonable(payload), indent=2))
    return out


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _load_subject_summaries(input_root: Path) -> pd.DataFrame:
    panel_path = input_root / "subject_panel.csv"
    allowed: Optional[Set[Tuple[str, str]]] = None
    if panel_path.exists():
        panel_df = pd.read_csv(panel_path)
        allowed = set(
            (
                panel_df["age_group"].astype(str).tolist()[i],
                panel_df["user_id"].astype(str).tolist()[i],
            )
            for i in range(len(panel_df))
        )

    rows: List[dict[str, Any]] = []
    for age_group in AGE_GROUPS:
        age_dir = input_root / age_group
        if not age_dir.exists():
            continue
        for subject_dir in sorted(age_dir.glob("user_*")):
            uid = subject_dir.name.replace("user_", "")
            if allowed is not None and (age_group, str(uid)) not in allowed:
                continue
            summary_path = subject_dir / "subject_eval_summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json(summary_path)
            fit = summary.get("fit", {})
            metrics = summary.get("test_metrics", {})

            subset_path = subject_dir / "fit_subset_indices.json"
            subset = _load_json(subset_path) if subset_path.exists() else {}

            pred_skew = float(metrics.get("pred_skewness", float("nan")))
            true_skew = float(metrics.get("true_skewness", float("nan")))
            model_acc = float(metrics.get("model_accuracy", float("nan")))
            human_acc = float(metrics.get("human_accuracy", float("nan")))
            pred_err_rt = float(metrics.get("pred_error_rt", float("nan")))
            pred_corr_rt = float(metrics.get("pred_correct_rt", float("nan")))
            human_err_rt = float(metrics.get("human_error_rt", float("nan")))
            human_corr_rt = float(metrics.get("human_correct_rt", float("nan")))
            model_gap = float(metrics.get("error_minus_correct_rt", float("nan")))
            human_gap = float(metrics.get("human_error_minus_correct_rt", float("nan")))

            # "Slow-error" direction: errors slower than correct (gap > 0)
            model_slow_error = bool(np.isfinite(model_gap) and model_gap > 0)
            human_slow_error = bool(np.isfinite(human_gap) and human_gap > 0)

            model_has_errors = bool(np.isfinite(pred_err_rt) and np.isfinite(pred_corr_rt))
            human_has_errors = bool(np.isfinite(human_err_rt) and np.isfinite(human_corr_rt))
            if model_has_errors and human_has_errors and np.isfinite(model_gap) and np.isfinite(human_gap):
                error_direction_match = bool(np.sign(model_gap) == np.sign(human_gap))
            else:
                error_direction_match = False

            accuracy_gap = float(abs(model_acc - human_acc)) if (np.isfinite(model_acc) and np.isfinite(human_acc)) else float("nan")

            rows.append(
                {
                    "age_group": str(summary.get("age_group", age_group)),
                    "user_id": str(summary.get("user_id", subject_dir.name.replace("user_", ""))),
                    "selected_scale": float(fit.get("scale", float("nan"))),
                    "time_steps": int(fit.get("time_steps", -1)),
                    "epochs": int(fit.get("epochs", -1)),
                    "choice_temperature": float(fit.get("choice_temperature", float("nan"))),
                    "n_train": int(subset.get("n_train", -1)),
                    "n_test": int(subset.get("n_test", int(summary.get("test_n_trials", -1)))),
                    "pred_skewness": pred_skew,
                    "true_skewness": true_skew,
                    "skewness_gap": float(pred_skew - true_skew) if (np.isfinite(pred_skew) and np.isfinite(true_skew)) else float("nan"),
                    "skewness_ratio": float(pred_skew / true_skew) if (np.isfinite(pred_skew) and np.isfinite(true_skew) and abs(true_skew) > 1e-9) else float("nan"),
                    "model_accuracy": model_acc,
                    "human_accuracy": human_acc,
                    "accuracy_gap": accuracy_gap,
                    "model_has_errors": model_has_errors,
                    "human_has_errors": human_has_errors,
                    "model_error_minus_correct_rt": model_gap,
                    "human_error_minus_correct_rt": human_gap,
                    "model_slow_error": model_slow_error,
                    "human_slow_error": human_slow_error,
                    "error_direction_match": error_direction_match,
                    "total_score": float(metrics.get("total_score", float("nan"))),
                    "rt_shape_score": float(metrics.get("rt_shape_score", float("nan"))),
                    "behavior_optimal_score": float(metrics.get("behavior_optimal_score", float("nan"))),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No subject_eval_summary.json files found under {input_root}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reagg_dir = output_root / "reaggregated"
    reagg_dir.mkdir(parents=True, exist_ok=True)

    heterogeneity_path = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_single_subject" / "reaggregated" / "success_bar.json"
    capture_path = RESULTS_ROOT / "repro_legacy_interim" / "minimal_conflict_capture_probe" / "reaggregated" / "success_bar.json"
    heterogeneity_verdict = _load_json(heterogeneity_path).get("verdict") if heterogeneity_path.exists() else "UNKNOWN"
    capture_verdict = _load_json(capture_path).get("verdict") if capture_path.exists() else "UNKNOWN"

    scorecard = _load_subject_summaries(input_root)
    scorecard["skew_present"] = scorecard["pred_skewness"].astype(float) > 0.5
    scorecard["model_has_errors"] = scorecard["model_has_errors"].astype(bool)
    scorecard["error_direction_match"] = scorecard["error_direction_match"].astype(bool)
    scorecard["nondegenerate_accuracy"] = scorecard["accuracy_gap"].astype(float) <= 0.05
    scorecard["subject_feasible"] = (
        scorecard["skew_present"]
        & scorecard["model_has_errors"]
        & scorecard["error_direction_match"]
        & scorecard["nondegenerate_accuracy"]
    )

    # Convenience summaries for Task-6 narrative
    pass_counts = {
        "skew_present": int(scorecard["skew_present"].sum()),
        "model_has_errors": int(scorecard["model_has_errors"].sum()),
        "error_direction_match": int(scorecard["error_direction_match"].sum()),
        "nondegenerate_accuracy": int(scorecard["nondegenerate_accuracy"].sum()),
    }
    fail_reasons = {
        "missing_model_errors": int((~scorecard["model_has_errors"]).sum()),
        "wrong_error_direction": int((scorecard["model_has_errors"] & ~scorecard["error_direction_match"]).sum()),
        "degenerate_accuracy": int((~scorecard["nondegenerate_accuracy"]).sum()),
        "low_skew": int((~scorecard["skew_present"]).sum()),
    }
    scorecard_path = reagg_dir / "feasibility_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)

    total_subjects = int(len(scorecard))
    feasible_subjects = int(scorecard["subject_feasible"].sum())
    age_support = {
        age_group: int(scorecard.loc[(scorecard["age_group"] == age_group) & (scorecard["subject_feasible"] == True)].shape[0])
        for age_group in AGE_GROUPS
    }
    verdict = (
        "VGGWW-SINGLE-SUBJECT-FEASIBLE"
        if feasible_subjects >= max(1, total_subjects // 2 + total_subjects % 2) and all(v >= 1 for v in age_support.values())
        else "VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE"
    )
    keep_recommendation = "keep_investigating" if verdict == "VGGWW-SINGLE-SUBJECT-FEASIBLE" else "deprioritize_vgg_ww"
    verdict_payload = {
        "verdict": verdict,
        "keep_recommendation": keep_recommendation,
        "scope": "bounded_panel_bounded_trial_budget_screen",
        "scope_note": "This verdict is based on a bounded representative panel and capped per-subject train/test trial budgets; treat it as a framework screen rather than an uncapped final proof.",
        "total_subjects": total_subjects,
        "feasible_subjects": feasible_subjects,
        "age_group_feasible_counts": age_support,
        "subject_feasibility_rule": "subject_feasible = skew_present AND model_has_errors AND error_direction_match AND nondegenerate_accuracy",
        "panel_feasibility_rule": "at least half of the panel feasible and at least one feasible subject in each age group",
    }
    verdict_path = reagg_dir / "feasibility_verdict.json"
    verdict_path.write_text(json.dumps(to_jsonable(verdict_payload), indent=2))
    ev5 = _write_evidence(5, "feasibility-scorecard", {"scorecard": _safe_rel(scorecard_path), "verdict": _safe_rel(verdict_path)})
    print(f"Wrote evidence: {ev5}")

    summary_lines = [
        "# True single-subject feasibility summary",
        "",
        "This workflow tested a **bounded true single-subject feasibility screen** for `VGG + WW`: each fit was learned from one subject's own trials rather than from group parameters plus a small tweak, but on a bounded representative panel and bounded per-subject trial budgets.",
        "",
        "## Why this branch exists",
        f"- Previous heterogeneity result: `{heterogeneity_verdict}`",
        f"- Previous minimal mechanism result: `{capture_verdict}`",
        "- Those results ruled out aggregation artifacts and tiny mechanism patches as the main explanation.",
        "- This branch asks the framework question directly: is `VGG + WW` viable at the individual-subject level?",
        "",
        "## Panel-wide result",
        f"- final verdict: `{verdict}`",
        f"- recommendation: `{keep_recommendation}`",
        "- scope: `bounded_panel_bounded_trial_budget_screen`",
        f"- feasible subjects: `{feasible_subjects}/{total_subjects}`",
        f"- age-group feasible counts: `{age_support}`",
        "- interpretation rule: use this as a screen for whether `VGG + WW` still looks worth pursuing, not as a full uncapped proof over every subject trial.",
        "",
        "## Subject-level feasibility rule",
        "A subject counts as feasible only if all of the following are true:",
        "- predicted RTs remain right-skewed (`pred_skewness > 0.5`)",
        "- the fitted model actually produces errors on held-out trials",
        "- model error-vs-correct RT direction matches the subject's own direction",
        "- model accuracy is not degenerate relative to the subject (`|model_accuracy - human_accuracy| <= 0.05`)",
        "",
        "## Scorecard highlights",
        "| age_group | user_id | selected_scale | pred_skewness | true_skewness | human_error_minus_correct_rt | model_error_minus_correct_rt | human_slow_error | model_slow_error | subject_feasible |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for _, row in scorecard.sort_values(by=["age_group", "user_id"], ascending=[True, True]).iterrows():
        summary_lines.append(
            f"| {row['age_group']} | {row['user_id']} | {float(row['selected_scale']):.3f} | {float(row['pred_skewness']):.3f} | {float(row['true_skewness']):.3f} | {float(row['human_error_minus_correct_rt']):.6g} | {float(row['model_error_minus_correct_rt']):.6g} | {'yes' if bool(row['human_slow_error']) else 'no'} | {'yes' if bool(row['model_slow_error']) else 'no'} | {'yes' if bool(row['subject_feasible']) else 'no'} |"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "This is the direct framework-retention screen under bounded compute. If the verdict is `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`, the current evidence supports deprioritizing `VGG + WW` rather than continuing to patch it indirectly, while still acknowledging that this was a bounded panel / bounded trial-budget screen rather than an uncapped final proof.",
            "",
            "**Panel diagnostics (counts over the bounded panel):**",
            f"- `skew_present (pred_skewness > 0.5)`: {pass_counts['skew_present']}/{total_subjects}",
            f"- `model_has_errors` (non-empty predicted error regime on held-out): {pass_counts['model_has_errors']}/{total_subjects}",
            f"- `error_direction_match` (sign(model_gap) == sign(human_gap)): {pass_counts['error_direction_match']}/{total_subjects}",
            f"- `nondegenerate_accuracy` (|model_acc - human_acc| <= 0.05): {pass_counts['nondegenerate_accuracy']}/{total_subjects}",
            "",
            "**Most common failure modes (not mutually exclusive):**",
            f"- missing model errors on held-out: {fail_reasons['missing_model_errors']}/{total_subjects}",
            f"- wrong error-vs-correct RT direction given errors: {fail_reasons['wrong_error_direction']}/{total_subjects}",
            f"- degenerate accuracy relative to subject: {fail_reasons['degenerate_accuracy']}/{total_subjects}",
            f"- insufficient predicted RT skew: {fail_reasons['low_skew']}/{total_subjects}",
            "",
            "**Slow-error direction:** we define slow-error as `error_minus_correct_rt > 0`. The scorecard includes both the human and model slow-error booleans explicitly.",
        ]
    )
    summary_path = output_root / "true_single_subject_feasibility_summary.md"
    summary_path.write_text("\n".join(summary_lines))
    ev6 = _write_evidence(6, "framework-summary", {"summary": _safe_rel(summary_path), "verdict": verdict, "recommendation": keep_recommendation})
    print(f"Wrote evidence: {ev6}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_dynamic_selection_single_subject import compute_conditional_error_rt, compute_tail_summary, safe_skew
from project_paths import RESULTS_ROOT
from train_age_groups_efficient import attach_flanker_labels_from_csv, compute_human_stats_from_rts, evaluate_cached_stage2_params, evaluate_joint_behavior, to_jsonable, validate_cached_stage2_inputs


AGE_GROUPS = ("20-29", "80-89")
DEFAULT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "true_single_subject_feasibility_rt_response_only"
EVIDENCE_ROOT = Path(".sisyphus") / "evidence"
SUBJECT_SKEW_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze true single-subject feasibility outputs and write scorecard, verdict, and summary.")
    parser.add_argument("--input_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--plot_subject", default=None, help="User ID for a single-subject RT plot export")
    parser.add_argument("--plot_age_group", default=None, choices=AGE_GROUPS, help="Age group for --plot_subject")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--choice_temperature", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=31)
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


def _load_panel_df(input_root: Path) -> pd.DataFrame:
    panel_path = input_root / "subject_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing bounded-panel definition: {panel_path}")
    panel_df = pd.read_csv(panel_path)
    panel_df["age_group"] = panel_df["age_group"].astype(str)
    panel_df["user_id"] = panel_df["user_id"].astype(str)
    return panel_df


def _load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _concat_cached_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in sorted(keys):
        if key not in a or key not in b:
            raise ValueError(f"CACHED_CONCAT_KEY_MISMATCH: key={key} a={key in a} b={key in b}")
        out[key] = np.concatenate([a[key], b[key]], axis=0)
    return out


def _filter_cached_by_indices(cached: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in cached.items():
        if isinstance(value, np.ndarray):
            out[key] = value[indices]
    return out


def _load_combined_cached(age_group: str) -> Dict[str, np.ndarray]:
    from project_paths import age_group_data_dir, age_group_stage2_dir

    data_dir = age_group_data_dir(age_group, matched=False)
    stage2_dir = age_group_stage2_dir(age_group, matched=False)
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
    train_npz = stage2_dir / "train_logits.npz"
    test_npz = stage2_dir / "test_logits.npz"
    train_cached, test_cached = validate_cached_stage2_inputs(age_group, str(data_dir), str(train_npz), str(test_npz))
    train_cached = attach_flanker_labels_from_csv(train_cached, str(train_csv))
    test_cached = attach_flanker_labels_from_csv(test_cached, str(test_csv))
    return _concat_cached_dicts(train_cached, test_cached)


def _load_subject_test_cached(input_root: Path, age_group: str, user_id: str) -> Dict[str, np.ndarray]:
    user_dir = input_root / age_group / f"user_{user_id}"
    subset_path = user_dir / "fit_subset_indices.json"
    split_path = user_dir / "subject_split.json"
    if subset_path.exists():
        subset = _load_json(subset_path)
        test_indices = np.array(subset["test_indices"], dtype=np.int64)
    elif split_path.exists():
        subset = _load_json(split_path)
        test_indices = np.array(subset["test_indices"], dtype=np.int64)
    else:
        raise FileNotFoundError(f"Missing test-index metadata for {age_group}/user_{user_id}")
    combined_cached = _load_combined_cached(age_group)
    return _filter_cached_by_indices(combined_cached, test_indices)


def _rt_metric_bundle(rt_s: np.ndarray, correct_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    rt_s = np.asarray(rt_s, dtype=np.float32)
    if rt_s.size == 0:
        return {
            "n_trials": 0,
            "mean_rt": float("nan"),
            "median_rt": float("nan"),
            "std_rt": float("nan"),
            "skewness": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "p90_over_p50": float("nan"),
            "p95_over_p50": float("nan"),
            "correct_rt": float("nan"),
            "error_rt": float("nan"),
            "error_minus_correct_rt": float("nan"),
            "n_correct": 0,
            "n_error": 0,
        }

    median_rt = float(np.median(rt_s))
    q90 = float(np.quantile(rt_s, 0.90))
    q95 = float(np.quantile(rt_s, 0.95))
    out: Dict[str, float] = {
        "n_trials": int(rt_s.size),
        "mean_rt": float(rt_s.mean()),
        "median_rt": median_rt,
        "std_rt": float(rt_s.std(ddof=0)),
        "skewness": float(safe_skew(rt_s)),
        "q90": q90,
        "q95": q95,
        "p90_over_p50": float(q90 / median_rt) if np.isfinite(median_rt) and abs(median_rt) > 1e-9 else float("nan"),
        "p95_over_p50": float(q95 / median_rt) if np.isfinite(median_rt) and abs(median_rt) > 1e-9 else float("nan"),
    }

    if correct_mask is None:
        out.update(
            {
                "correct_rt": float("nan"),
                "error_rt": float("nan"),
                "error_minus_correct_rt": float("nan"),
                "n_correct": 0,
                "n_error": 0,
            }
        )
        return out

    correct_mask = np.asarray(correct_mask, dtype=bool)
    correct_vals = rt_s[correct_mask]
    error_vals = rt_s[~correct_mask]
    correct_rt = float(correct_vals.mean()) if correct_vals.size else float("nan")
    error_rt = float(error_vals.mean()) if error_vals.size else float("nan")
    out.update(
        {
            "correct_rt": correct_rt,
            "error_rt": error_rt,
            "error_minus_correct_rt": float(error_rt - correct_rt) if correct_vals.size and error_vals.size else float("nan"),
            "n_correct": int(correct_vals.size),
            "n_error": int(error_vals.size),
        }
    )
    return out


def _subject_dir(input_root: Path, age_group: str, user_id: str) -> Path:
    return input_root / age_group / f"user_{user_id}"


def _single_subject_metrics_row(
    *,
    age_group: str,
    user_id: str,
    n_test: int,
    metrics: Dict[str, Any],
) -> pd.DataFrame:
    pred_median = float(metrics.get("pred_median", float("nan")))
    true_median = float(metrics.get("true_median", float("nan")))
    pred_q90 = float(metrics.get("pred_q90", float("nan")))
    pred_q95 = float(metrics.get("pred_q95", float("nan")))
    true_q90 = float(metrics.get("true_q90", float("nan")))
    true_q95 = float(metrics.get("true_q95", float("nan")))
    return pd.DataFrame(
        [
            {
                "age_group": age_group,
                "user_id": user_id,
                "n_test": int(n_test),
                "pred_mean": float(metrics.get("pred_mean", float("nan"))),
                "true_mean": float(metrics.get("true_mean", float("nan"))),
                "pred_median": pred_median,
                "true_median": true_median,
                "pred_skewness": float(metrics.get("pred_skewness", float("nan"))),
                "true_skewness": float(metrics.get("true_skewness", float("nan"))),
                "pred_p90_over_p50": float(pred_q90 / pred_median) if np.isfinite(pred_q90) and np.isfinite(pred_median) and abs(pred_median) > 1e-9 else float("nan"),
                "true_p90_over_p50": float(true_q90 / true_median) if np.isfinite(true_q90) and np.isfinite(true_median) and abs(true_median) > 1e-9 else float("nan"),
                "pred_p95_over_p50": float(pred_q95 / pred_median) if np.isfinite(pred_q95) and np.isfinite(pred_median) and abs(pred_median) > 1e-9 else float("nan"),
                "true_p95_over_p50": float(true_q95 / true_median) if np.isfinite(true_q95) and np.isfinite(true_median) and abs(true_median) > 1e-9 else float("nan"),
                "pred_correct_rt": float(metrics.get("pred_correct_rt", float("nan"))),
                "pred_error_rt": float(metrics.get("pred_error_rt", float("nan"))),
                "human_correct_rt": float(metrics.get("human_correct_rt", float("nan"))),
                "human_error_rt": float(metrics.get("human_error_rt", float("nan"))),
                "model_accuracy": float(metrics.get("model_accuracy", float("nan"))),
                "human_accuracy": float(metrics.get("human_accuracy", float("nan"))),
                "rt_shape_score": float(metrics.get("rt_shape_score", float("nan"))),
                "total_score": float(metrics.get("total_score", float("nan"))),
            }
        ]
    )


def _human_subject_level_metrics(input_root: Path) -> pd.DataFrame:
    panel_df = _load_panel_df(input_root)
    rows: List[Dict[str, Any]] = []
    for _, row in panel_df.iterrows():
        test_cached = _load_subject_test_cached(input_root, str(row["age_group"]), str(row["user_id"]))
        trial_metrics = _rt_metric_bundle(
            np.asarray(test_cached["rts"], dtype=np.float32),
            np.asarray(test_cached["response_labels"], dtype=np.int64) == np.asarray(test_cached["target_labels"], dtype=np.int64),
        )
        rows.append(
            {
                "age_group": str(row["age_group"]),
                "user_id": str(row["user_id"]),
                **trial_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["age_group", "user_id"]).reset_index(drop=True)


def _condition_label(condition: Optional[str]) -> str:
    if condition is None:
        return "all"
    return str(condition)


def _pooled_metric_rows(*, source: str, age_group: str, condition: Optional[str], rt_s: np.ndarray, correct_mask: np.ndarray, congruency: np.ndarray) -> List[Dict[str, Any]]:
    if condition is None:
        subset_rt = np.asarray(rt_s, dtype=np.float32)
        subset_correct = np.asarray(correct_mask, dtype=bool)
    else:
        desired = 0 if condition == "congruent" else 1
        mask = np.asarray(congruency, dtype=np.int64) == desired
        subset_rt = np.asarray(rt_s, dtype=np.float32)[mask]
        subset_correct = np.asarray(correct_mask, dtype=bool)[mask]
    metrics = _rt_metric_bundle(subset_rt, subset_correct)
    return [
        {
            "source": source,
            "age_group": age_group,
            "condition": _condition_label(condition),
            **metrics,
        }
    ]


def _human_pooled_metrics(input_root: Path) -> pd.DataFrame:
    panel_df = _load_panel_df(input_root)
    rows: List[Dict[str, Any]] = []
    for age_group in list(AGE_GROUPS) + ["all"]:
        scoped_panel = panel_df if age_group == "all" else panel_df.loc[panel_df["age_group"] == age_group]
        rts_parts: List[np.ndarray] = []
        correct_parts: List[np.ndarray] = []
        congruency_parts: List[np.ndarray] = []
        for _, panel_row in scoped_panel.iterrows():
            test_cached = _load_subject_test_cached(input_root, str(panel_row["age_group"]), str(panel_row["user_id"]))
            rts_parts.append(np.asarray(test_cached["rts"], dtype=np.float32))
            correct_parts.append(np.asarray(test_cached["response_labels"], dtype=np.int64) == np.asarray(test_cached["target_labels"], dtype=np.int64))
            congruency_parts.append(np.asarray(test_cached["congruency"], dtype=np.int64))
        pooled_rts = np.concatenate(rts_parts, axis=0)
        pooled_correct = np.concatenate(correct_parts, axis=0)
        pooled_congruency = np.concatenate(congruency_parts, axis=0)
        for condition in (None, "congruent", "incongruent"):
            rows.extend(
                _pooled_metric_rows(
                    source="human",
                    age_group=str(age_group),
                    condition=condition,
                    rt_s=pooled_rts,
                    correct_mask=pooled_correct,
                    congruency=pooled_congruency,
                )
            )
    return pd.DataFrame(rows)


def _build_trial_df(rt_s: np.ndarray, choice: np.ndarray, target_labels: np.ndarray, congruency: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "rt_s": np.asarray(rt_s, dtype=np.float32),
            "choice": np.asarray(choice, dtype=np.int64),
            "target": np.asarray(target_labels, dtype=np.int64),
            "congruency": np.asarray(congruency, dtype=np.int64),
        }
    )
    df["condition"] = np.where(df["congruency"] == 1, "incongruent", "congruent")
    df["correct"] = df["choice"] == df["target"]
    return df


def _plot_single_subject_rt(*, pred_rt: np.ndarray, true_rt: np.ndarray, age_group: str, user_id: str, output_path: Path) -> None:
    pred_rt = np.asarray(pred_rt, dtype=np.float32)
    true_rt = np.asarray(true_rt, dtype=np.float32)
    max_rt = float(max(np.max(pred_rt), np.max(true_rt)))
    bins = np.linspace(0.0, max_rt, 40)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    axes[0].hist(true_rt, bins=bins, density=True, alpha=0.55, label="human", color="#4C72B0")
    axes[0].hist(pred_rt, bins=bins, density=True, alpha=0.55, label="model", color="#DD8452")
    axes[0].set_title("RT distribution")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    for values, label, color in ((true_rt, "human", "#4C72B0"), (pred_rt, "model", "#DD8452")):
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1, dtype=np.float32) / float(len(sorted_vals))
        axes[1].plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    axes[1].set_title("RT cumulative distribution")
    axes[1].set_xlabel("RT (s)")
    axes[1].set_ylabel("CDF")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Single-subject RT fit: {age_group} / user_{user_id}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_single_subject_rt_and_response(
    *,
    pred_rt: np.ndarray,
    true_rt: np.ndarray,
    pred_choice: np.ndarray,
    response_labels: np.ndarray,
    age_group: str,
    user_id: str,
    output_path: Path,
) -> None:
    pred_rt = np.asarray(pred_rt, dtype=np.float32)
    true_rt = np.asarray(true_rt, dtype=np.float32)
    pred_choice = np.asarray(pred_choice, dtype=np.int64)
    response_labels = np.asarray(response_labels, dtype=np.int64)

    max_rt = float(max(np.max(pred_rt), np.max(true_rt)))
    bins = np.linspace(0.0, max_rt, 40)
    categories = np.arange(4)
    category_labels = ["L", "R", "U", "D"]
    human_counts = np.bincount(response_labels, minlength=4).astype(np.float32)
    model_counts = np.bincount(pred_choice, minlength=4).astype(np.float32)
    human_props = human_counts / max(float(human_counts.sum()), 1.0)
    model_props = model_counts / max(float(model_counts.sum()), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

    axes[0].hist(true_rt, bins=bins, density=True, alpha=0.55, label="human", color="#4C72B0")
    axes[0].hist(pred_rt, bins=bins, density=True, alpha=0.55, label="model", color="#DD8452")
    axes[0].set_title("RT histogram")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    width = 0.36
    axes[1].bar(categories - width / 2, human_props, width=width, label="human", color="#4C72B0", alpha=0.8)
    axes[1].bar(categories + width / 2, model_props, width=width, label="model", color="#DD8452", alpha=0.8)
    axes[1].set_xticks(categories)
    axes[1].set_xticklabels(category_labels)
    axes[1].set_title("Response distribution")
    axes[1].set_xlabel("Response")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Single-subject RT + response fit: {age_group} / user_{user_id}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_single_subject_plot(
    *,
    input_root: Path,
    output_root: Path,
    age_group: str,
    user_id: str,
    device: str,
    choice_temperature: float,
    random_seed: int,
) -> Dict[str, str]:
    user_dir = _subject_dir(input_root, age_group, user_id)
    best_config_path = user_dir / "best_config.json"
    params_path = user_dir / "best_model_params.npz"
    summary_path = user_dir / "subject_eval_summary.json"
    required = [best_config_path, params_path, summary_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required single-subject plotting inputs: {missing}")

    best_cfg = _load_json(best_config_path)
    params = _load_npz_dict(params_path)
    test_cached = _load_subject_test_cached(input_root, age_group, user_id)
    eval_seed = int(best_cfg.get("eval_random_seed", random_seed))
    predictions, metrics = evaluate_cached_stage2_params(
        params=params,
        scale=float(best_cfg["scale"]),
        time_steps=int(best_cfg["time_steps"]),
        cached=test_cached,
        device=device,
        choice_temperature=float(best_cfg.get("choice_temperature", choice_temperature)),
        rt_readout_mode=str(best_cfg.get("rt_readout_mode", "baseline")),
        readout_config=best_cfg.get("readout_config") or {},
        selection_config=best_cfg.get("selection_config") or {},
        random_seed=eval_seed,
        rt_shape_focus=True,
    )

    export_dir = output_root / age_group / f"user_{user_id}" / "plot_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = _single_subject_metrics_row(
        age_group=age_group,
        user_id=user_id,
        n_test=int(len(test_cached["rts"])),
        metrics=metrics,
    )
    metrics_path = export_dir / "subject_rt_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = export_dir / "subject_rt_predictions.npz"
    np.savez_compressed(
        predictions_path,
        pred_rt=np.asarray(predictions["pred_rt"], dtype=np.float32),
        true_rt=np.asarray(test_cached["rts"], dtype=np.float32),
        pred_choice=np.asarray(predictions["pred_choice"], dtype=np.int64),
        target_labels=np.asarray(test_cached["target_labels"], dtype=np.int64),
        response_labels=np.asarray(test_cached["response_labels"], dtype=np.int64),
        congruency=np.asarray(test_cached["congruency"], dtype=np.int64),
    )

    plot_path = export_dir / "rt_distribution.png"
    _plot_single_subject_rt(
        pred_rt=np.asarray(predictions["pred_rt"], dtype=np.float32),
        true_rt=np.asarray(test_cached["rts"], dtype=np.float32),
        age_group=age_group,
        user_id=user_id,
        output_path=plot_path,
    )

    comparison_plot_path = export_dir / "rt_histogram_response_barplot.png"
    _plot_single_subject_rt_and_response(
        pred_rt=np.asarray(predictions["pred_rt"], dtype=np.float32),
        true_rt=np.asarray(test_cached["rts"], dtype=np.float32),
        pred_choice=np.asarray(predictions["pred_choice"], dtype=np.int64),
        response_labels=np.asarray(test_cached["response_labels"], dtype=np.int64),
        age_group=age_group,
        user_id=user_id,
        output_path=comparison_plot_path,
    )

    note_path = export_dir / "subject_rt_summary.md"
    note_lines = [
        f"# Single-subject RT fit summary: {age_group} / user_{user_id}",
        "",
        f"- n_test: `{int(len(test_cached['rts']))}`",
        f"- pred_skewness: `{float(metrics.get('pred_skewness', float('nan'))):.3f}`",
        f"- true_skewness: `{float(metrics.get('true_skewness', float('nan'))):.3f}`",
        f"- pred_median: `{float(metrics.get('pred_median', float('nan'))):.3f}` s",
        f"- true_median: `{float(metrics.get('true_median', float('nan'))):.3f}` s",
        f"- rt_shape_score: `{float(metrics.get('rt_shape_score', float('nan'))):.3f}`",
        f"- total_score: `{float(metrics.get('total_score', float('nan'))):.3f}`",
        "",
        "This export is intended as a first-pass visual check of whether the current model can match one subject's RT distribution on held-out trials.",
    ]
    note_path.write_text("\n".join(note_lines))

    return {
        "plot": _safe_rel(plot_path),
        "comparison_plot": _safe_rel(comparison_plot_path),
        "metrics_csv": _safe_rel(metrics_path),
        "predictions_npz": _safe_rel(predictions_path),
        "summary_md": _safe_rel(note_path),
    }


def _evaluate_subject_model_predictions(
    *,
    input_root: Path,
    age_group: str,
    user_id: str,
    device: str,
    choice_temperature: float,
    random_seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    user_dir = _subject_dir(input_root, age_group, user_id)
    best_cfg = _load_json(user_dir / "best_config.json")
    params = _load_npz_dict(user_dir / "best_model_params.npz")
    test_cached = _load_subject_test_cached(input_root, age_group, user_id)
    eval_seed = int(best_cfg.get("eval_random_seed", random_seed))
    predictions, canonical_metrics = evaluate_cached_stage2_params(
        params=params,
        scale=float(best_cfg["scale"]),
        time_steps=int(best_cfg["time_steps"]),
        cached=test_cached,
        device=device,
        choice_temperature=float(best_cfg.get("choice_temperature", choice_temperature)),
        rt_readout_mode=str(best_cfg.get("rt_readout_mode", "baseline")),
        readout_config=best_cfg.get("readout_config") or {},
        selection_config=best_cfg.get("selection_config") or {},
        random_seed=eval_seed,
        rt_shape_focus=True,
    )
    model_df = _build_trial_df(predictions["pred_rt"], predictions["pred_choice"], test_cached["target_labels"], test_cached["congruency"])
    human_df = _build_trial_df(test_cached["rts"], test_cached["response_labels"], test_cached["target_labels"], test_cached["congruency"])
    return predictions, canonical_metrics, model_df, human_df


def _model_subject_level_metrics(
    *,
    input_root: Path,
    device: str,
    choice_temperature: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_df = _load_panel_df(input_root)
    human_subject_df = _human_subject_level_metrics(input_root)
    rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for _, panel_row in panel_df.iterrows():
        age_group = str(panel_row["age_group"])
        user_id = str(panel_row["user_id"])
        predictions, canonical_metrics, _, human_df = _evaluate_subject_model_predictions(
            input_root=input_root,
            age_group=age_group,
            user_id=user_id,
            device=device,
            choice_temperature=choice_temperature,
            random_seed=random_seed,
        )
        model_bundle = _rt_metric_bundle(
            np.asarray(predictions["pred_rt"], dtype=np.float32),
            np.asarray(predictions["pred_choice"], dtype=np.int64) == np.asarray(human_df["target"], dtype=np.int64),
        )
        human_row = human_subject_df.loc[(human_subject_df["age_group"] == age_group) & (human_subject_df["user_id"] == user_id)].iloc[0]
        human_accuracy = float(human_row["n_correct"] / human_row["n_trials"]) if float(human_row["n_trials"]) > 0 else float("nan")
        rows.append(
            {
                "age_group": age_group,
                "user_id": user_id,
                **model_bundle,
                "model_accuracy": float(canonical_metrics.get("model_accuracy", float("nan"))),
                "human_accuracy": human_accuracy,
                "rt_shape_score": float(canonical_metrics.get("rt_shape_score", float("nan"))),
                "total_score": float(canonical_metrics.get("total_score", float("nan"))),
            }
        )
        comparison_rows.append(
            {
                "age_group": age_group,
                "user_id": user_id,
                "human_skewness": float(human_row["skewness"]),
                "model_skewness": float(model_bundle["skewness"]),
                "human_p95_over_p50": float(human_row["p95_over_p50"]),
                "model_p95_over_p50": float(model_bundle["p95_over_p50"]),
                "human_error_minus_correct_rt": float(human_row["error_minus_correct_rt"]),
                "model_error_minus_correct_rt": float(model_bundle["error_minus_correct_rt"]),
                "human_accuracy": human_accuracy,
                "model_accuracy": float(canonical_metrics.get("model_accuracy", float("nan"))),
                "skew_match": bool(np.isfinite(model_bundle["skewness"]) and model_bundle["skewness"] > SUBJECT_SKEW_THRESHOLD),
                "error_direction_match": bool(np.isfinite(model_bundle["error_minus_correct_rt"]) and np.isfinite(human_row["error_minus_correct_rt"]) and np.sign(model_bundle["error_minus_correct_rt"]) == np.sign(float(human_row["error_minus_correct_rt"]))),
                "accuracy_gap": abs(float(canonical_metrics.get("model_accuracy", float("nan"))) - human_accuracy),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["age_group", "user_id"]).reset_index(drop=True),
        pd.DataFrame(comparison_rows).sort_values(["age_group", "user_id"]).reset_index(drop=True),
    )


def _model_pooled_metrics(
    *,
    input_root: Path,
    device: str,
    choice_temperature: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_df = _load_panel_df(input_root)
    pooled_model: Dict[str, Dict[str, List[np.ndarray]]] = {scope: {"rt": [], "correct": [], "congruency": []} for scope in [*AGE_GROUPS, "all"]}
    pooled_human: Dict[str, Dict[str, List[np.ndarray]]] = {scope: {"rt": [], "correct": [], "congruency": []} for scope in [*AGE_GROUPS, "all"]}
    condition_rows: List[Dict[str, Any]] = []

    for _, panel_row in panel_df.iterrows():
        age_group = str(panel_row["age_group"])
        user_id = str(panel_row["user_id"])
        predictions, _, model_df, human_df = _evaluate_subject_model_predictions(
            input_root=input_root,
            age_group=age_group,
            user_id=user_id,
            device=device,
            choice_temperature=choice_temperature,
            random_seed=random_seed,
        )
        model_correct = np.asarray(predictions["pred_choice"], dtype=np.int64) == np.asarray(human_df["target"], dtype=np.int64)
        human_correct = np.asarray(human_df["correct"], dtype=bool)
        model_congruency = np.asarray(model_df["congruency"], dtype=np.int64)
        human_congruency = np.asarray(human_df["congruency"], dtype=np.int64)
        for scope in (age_group, "all"):
            pooled_model[scope]["rt"].append(np.asarray(predictions["pred_rt"], dtype=np.float32))
            pooled_model[scope]["correct"].append(model_correct)
            pooled_model[scope]["congruency"].append(model_congruency)
            pooled_human[scope]["rt"].append(np.asarray(human_df["rt_s"], dtype=np.float32))
            pooled_human[scope]["correct"].append(human_correct)
            pooled_human[scope]["congruency"].append(human_congruency)

        human_err_wide, human_err_long = compute_conditional_error_rt(human_df, "human", "rt_s", "correct")
        model_err_wide, model_err_long = compute_conditional_error_rt(model_df, "model", "rt_s", "correct")
        human_tail = compute_tail_summary(human_df, "human", "rt_s", "correct")
        model_tail = compute_tail_summary(model_df, "model", "rt_s", "correct")
        for source_name, err_df, tail_df in (("human", human_err_long, human_tail), ("model", model_err_long, model_tail)):
            for _, cond_row in err_df.iterrows():
                tail_match = tail_df.loc[tail_df["group"] == f"{cond_row['condition']}_error"]
                condition_rows.append(
                    {
                        "age_group": age_group,
                        "user_id": user_id,
                        "source": source_name,
                        "condition": str(cond_row["condition"]),
                        "correct_rt": float(cond_row["correct_rt"]),
                        "error_rt": float(cond_row["error_rt"]),
                        "error_minus_correct_rt": float(cond_row["error_minus_correct_rt"]),
                        "n_correct": int(cond_row["n_correct"]),
                        "n_error": int(cond_row["n_error"]),
                        "error_q95": float(tail_match["q95"].iloc[0]) if not tail_match.empty else float("nan"),
                    }
                )

    pooled_rows: List[Dict[str, Any]] = []
    for source_name, store in (("human", pooled_human), ("model", pooled_model)):
        for scope, arrays in store.items():
            pooled_rt = np.concatenate(arrays["rt"], axis=0)
            pooled_correct = np.concatenate(arrays["correct"], axis=0)
            pooled_congruency = np.concatenate(arrays["congruency"], axis=0)
            for condition in (None, "congruent", "incongruent"):
                pooled_rows.extend(
                    _pooled_metric_rows(
                        source=source_name,
                        age_group=scope,
                        condition=condition,
                        rt_s=pooled_rt,
                        correct_mask=pooled_correct,
                        congruency=pooled_congruency,
                    )
                )
    return pd.DataFrame(pooled_rows), pd.DataFrame(condition_rows)


def _mixture_reconstruction_summary(
    human_subject_df: pd.DataFrame,
    human_pooled_df: pd.DataFrame,
    model_subject_df: pd.DataFrame,
    model_pooled_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope in [*AGE_GROUPS, "all"]:
        human_subject_scope = human_subject_df if scope == "all" else human_subject_df.loc[human_subject_df["age_group"] == scope]
        model_subject_scope = model_subject_df if scope == "all" else model_subject_df.loc[model_subject_df["age_group"] == scope]
        human_pooled_row = human_pooled_df.loc[(human_pooled_df["source"] == "human") & (human_pooled_df["age_group"] == scope) & (human_pooled_df["condition"] == "all")].iloc[0]
        model_pooled_row = model_pooled_df.loc[(model_pooled_df["source"] == "model") & (model_pooled_df["age_group"] == scope) & (model_pooled_df["condition"] == "all")].iloc[0]
        mean_subject_skew = float(human_subject_scope["skewness"].mean()) if not human_subject_scope.empty else float("nan")
        mean_model_subject_skew = float(model_subject_scope["skewness"].mean()) if not model_subject_scope.empty else float("nan")
        pooled_human_skew = float(human_pooled_row["skewness"])
        pooled_model_skew = float(model_pooled_row["skewness"])
        rows.append(
            {
                "scope": scope,
                "mean_subject_skewness": mean_subject_skew,
                "pooled_skewness": pooled_human_skew,
                "pooled_minus_mean_subject_skewness": float(pooled_human_skew - mean_subject_skew) if np.isfinite(pooled_human_skew) and np.isfinite(mean_subject_skew) else float("nan"),
                "mean_model_subject_skewness": mean_model_subject_skew,
                "model_pooled_skewness": pooled_model_skew,
                "model_pooled_minus_mean_subject_skewness": float(pooled_model_skew - mean_model_subject_skew) if np.isfinite(pooled_model_skew) and np.isfinite(mean_model_subject_skew) else float("nan"),
                "mean_subject_p95_over_p50": float(np.nanmean(human_subject_scope["p95_over_p50"].to_numpy(dtype=float))) if not human_subject_scope.empty else float("nan"),
                "pooled_p95_over_p50": float(human_pooled_row["p95_over_p50"]),
                "mixture_effect_flag": bool(np.isfinite(pooled_human_skew) and np.isfinite(mean_subject_skew) and (pooled_human_skew - mean_subject_skew) > 0.15),
            }
        )
    return pd.DataFrame(rows)


def _bounded_panel_verdict(comparison_df: pd.DataFrame, human_mixture_df: pd.DataFrame, model_pooled_df: pd.DataFrame) -> Dict[str, Any]:
    single_subject_robust_count = int(
        (
            comparison_df["skew_match"].astype(bool)
            & comparison_df["error_direction_match"].astype(bool)
            & (comparison_df["accuracy_gap"].astype(float) <= 0.05)
        ).sum()
    )
    total_subjects = int(len(comparison_df))
    human_all = human_mixture_df.loc[human_mixture_df["scope"] == "all"].iloc[0]
    model_all = model_pooled_df.loc[(model_pooled_df["source"] == "model") & (model_pooled_df["age_group"] == "all") & (model_pooled_df["condition"] == "all")].iloc[0]
    pooled_human_strong = bool(np.isfinite(float(human_all["pooled_skewness"])) and float(human_all["pooled_skewness"]) > SUBJECT_SKEW_THRESHOLD)
    pooled_model_strong = bool(np.isfinite(float(model_all["skewness"])) and float(model_all["skewness"]) > SUBJECT_SKEW_THRESHOLD)
    mostly_subject_level = single_subject_robust_count >= max(1, total_subjects // 2)
    if mostly_subject_level and pooled_model_strong:
        status = "single-subject robust"
    elif (not mostly_subject_level) and pooled_human_strong and pooled_model_strong:
        status = "pooled-only"
    else:
        status = "neither"
    if bool(human_all["mixture_effect_flag"]) and not mostly_subject_level:
        next_direction = "prioritize_subject_specific_or_hierarchical_heterogeneity"
    elif mostly_subject_level and not bool(human_all["mixture_effect_flag"]):
        next_direction = "prioritize_single_subject_generative_dynamics"
    else:
        next_direction = "require_both_subject_level_and_pooled_evaluation"
    return {
        "model_status": status,
        "single_subject_robust_count": single_subject_robust_count,
        "total_subjects": total_subjects,
        "human_pooled_skewness": float(human_all["pooled_skewness"]),
        "human_mean_subject_skewness": float(human_all["mean_subject_skewness"]),
        "human_mixture_effect_flag": bool(human_all["mixture_effect_flag"]),
        "model_pooled_skewness": float(model_all["skewness"]),
        "next_direction": next_direction,
    }



def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if args.plot_subject is not None or args.plot_age_group is not None:
        if args.plot_subject is None or args.plot_age_group is None:
            raise ValueError("--plot_subject and --plot_age_group must be provided together")
        exported = export_single_subject_plot(
            input_root=input_root,
            output_root=output_root,
            age_group=str(args.plot_age_group),
            user_id=str(args.plot_subject),
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
            random_seed=int(args.random_seed),
        )
        print(json.dumps({"single_subject_plot_export": exported}, indent=2))
        return

    reagg_dir = output_root / "reaggregated"
    reagg_dir.mkdir(parents=True, exist_ok=True)

    heterogeneity_path = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_single_subject" / "reaggregated" / "success_bar.json"
    capture_path = RESULTS_ROOT / "repro_legacy_interim" / "minimal_conflict_capture_probe" / "reaggregated" / "success_bar.json"
    heterogeneity_verdict = _load_json(heterogeneity_path).get("verdict") if heterogeneity_path.exists() else "UNKNOWN"
    capture_verdict = _load_json(capture_path).get("verdict") if capture_path.exists() else "UNKNOWN"

    human_subject_metrics = _human_subject_level_metrics(input_root)
    human_subject_metrics_path = reagg_dir / "human_subject_level_metrics.csv"
    human_subject_metrics.to_csv(human_subject_metrics_path, index=False)

    human_pooled_metrics = _human_pooled_metrics(input_root)
    human_pooled_metrics_path = reagg_dir / "human_pooled_metrics.csv"
    human_pooled_metrics.to_csv(human_pooled_metrics_path, index=False)

    model_subject_metrics, subject_level_comparison = _model_subject_level_metrics(
        input_root=input_root,
        device=str(args.device),
        choice_temperature=float(args.choice_temperature),
        random_seed=int(args.random_seed),
    )
    model_subject_metrics_path = reagg_dir / "model_subject_level_metrics.csv"
    model_subject_metrics.to_csv(model_subject_metrics_path, index=False)
    subject_level_comparison_path = reagg_dir / "subject_level_comparison.csv"
    subject_level_comparison.to_csv(subject_level_comparison_path, index=False)

    model_pooled_metrics, condition_level_summary = _model_pooled_metrics(
        input_root=input_root,
        device=str(args.device),
        choice_temperature=float(args.choice_temperature),
        random_seed=int(args.random_seed),
    )
    model_pooled_metrics_path = reagg_dir / "model_pooled_metrics.csv"
    model_pooled_metrics.to_csv(model_pooled_metrics_path, index=False)
    condition_level_summary_path = reagg_dir / "condition_level_error_summary.csv"
    condition_level_summary.to_csv(condition_level_summary_path, index=False)

    mixture_reconstruction = _mixture_reconstruction_summary(
        human_subject_metrics,
        human_pooled_metrics,
        model_subject_metrics,
        model_pooled_metrics,
    )
    mixture_reconstruction_path = reagg_dir / "mixture_reconstruction_summary.csv"
    mixture_reconstruction.to_csv(mixture_reconstruction_path, index=False)

    bounded_panel_verdict = _bounded_panel_verdict(subject_level_comparison, mixture_reconstruction, model_pooled_metrics)

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
        "bounded_panel_diagnostic": bounded_panel_verdict,
    }
    verdict_path = reagg_dir / "feasibility_verdict.json"
    verdict_path.write_text(json.dumps(to_jsonable(verdict_payload), indent=2))
    skew_verdict_path = reagg_dir / "single_subject_skew_verdict.json"
    skew_verdict_path.write_text(json.dumps(to_jsonable(bounded_panel_verdict), indent=2))
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
        f"- bounded-panel diagnostic status: `{bounded_panel_verdict['model_status']}`",
        f"- suggested next direction: `{bounded_panel_verdict['next_direction']}`",
        "- interpretation rule: use this as a screen for whether `VGG + WW` still looks worth pursuing, not as a full uncapped proof over every subject trial.",
        "",
        "## Bounded-panel skew diagnostic",
        f"- human pooled skewness: `{bounded_panel_verdict['human_pooled_skewness']:.3f}`",
        f"- human mean subject skewness: `{bounded_panel_verdict['human_mean_subject_skewness']:.3f}`",
        f"- mixture-effect flag: `{bounded_panel_verdict['human_mixture_effect_flag']}`",
        f"- model pooled skewness: `{bounded_panel_verdict['model_pooled_skewness']:.3f}`",
        f"- single-subject robust count: `{bounded_panel_verdict['single_subject_robust_count']}/{bounded_panel_verdict['total_subjects']}`",
        "",
        "## Subject-level feasibility rule",
        "A subject counts as feasible only if all of the following are true:",
        f"- predicted RTs remain right-skewed (`pred_skewness > {SUBJECT_SKEW_THRESHOLD}`)",
        "- the fitted model actually produces errors on held-out trials",
        "- model error-vs-correct RT direction matches the subject's own direction",
        "- model accuracy is not degenerate relative to the subject (`|model_accuracy - human_accuracy| <= 0.05`)",
        "",
        "## Generated diagnostic artifacts",
        f"- human subject-level metrics: `{_safe_rel(human_subject_metrics_path)}`",
        f"- human pooled metrics: `{_safe_rel(human_pooled_metrics_path)}`",
        f"- model subject-level metrics: `{_safe_rel(model_subject_metrics_path)}`",
        f"- model pooled metrics: `{_safe_rel(model_pooled_metrics_path)}`",
        f"- subject-level comparison: `{_safe_rel(subject_level_comparison_path)}`",
        f"- condition-level error summary: `{_safe_rel(condition_level_summary_path)}`",
        f"- mixture reconstruction summary: `{_safe_rel(mixture_reconstruction_path)}`",
        f"- single-subject skew verdict: `{_safe_rel(skew_verdict_path)}`",
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
            "## Decision on next modeling direction",
            f"- If the right-skew signal is mostly within-subject and the bounded-panel diagnostic is `single-subject robust`, prioritize stronger single-subject generative dynamics.",
            f"- If the pooled skew is stronger than the mean subject skew and the bounded-panel diagnostic is `pooled-only`, prioritize subject-specific or hierarchical heterogeneity modeling.",
            f"- Current study outcome: `{bounded_panel_verdict['model_status']}` -> `{bounded_panel_verdict['next_direction']}`.",
            "",
            "**Panel diagnostics (counts over the bounded panel):**",
            f"- `skew_present (pred_skewness > {SUBJECT_SKEW_THRESHOLD})`: {pass_counts['skew_present']}/{total_subjects}",
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
    decision_memo_path = output_root / "single_subject_skew_decision_memo.md"
    decision_memo_lines = [
        "# Single-subject skew decision memo",
        "",
        "## What is true in bounded-panel human data",
        f"- Human pooled skewness (all subjects): `{bounded_panel_verdict['human_pooled_skewness']:.3f}`.",
        f"- Mean within-subject human skewness: `{bounded_panel_verdict['human_mean_subject_skewness']:.3f}`.",
        f"- Mixture-effect flag from pooled-vs-mean-subject skew: `{bounded_panel_verdict['human_mixture_effect_flag']}`.",
        "",
        "## What the current true-single-subject VGG+WW workflow reproduces",
        f"- Bounded-panel model status: `{bounded_panel_verdict['model_status']}`.",
        f"- Single-subject robust count: `{bounded_panel_verdict['single_subject_robust_count']}/{bounded_panel_verdict['total_subjects']}`.",
        f"- Model pooled skewness: `{bounded_panel_verdict['model_pooled_skewness']:.3f}`.",
        "",
        "## Is pooled evaluation overstating current model success?",
        (
            "- Yes: pooled evaluation appears to overstate success when the bounded-panel diagnostic is `pooled-only`, "
            "because the model reproduces pooled shape better than it reproduces subject-level skew/error structure."
            if bounded_panel_verdict["model_status"] == "pooled-only"
            else "- Not clearly: the bounded-panel diagnostic does not indicate a pooled-only success regime."
        ),
        "",
        "## Recommended next direction",
        f"- `{bounded_panel_verdict['next_direction']}`",
        "",
        "## Supporting artifacts",
        f"- `{_safe_rel(human_subject_metrics_path)}`",
        f"- `{_safe_rel(human_pooled_metrics_path)}`",
        f"- `{_safe_rel(model_subject_metrics_path)}`",
        f"- `{_safe_rel(model_pooled_metrics_path)}`",
        f"- `{_safe_rel(subject_level_comparison_path)}`",
        f"- `{_safe_rel(mixture_reconstruction_path)}`",
        f"- `{_safe_rel(skew_verdict_path)}`",
    ]
    decision_memo_path.write_text("\n".join(decision_memo_lines))
    ev6 = _write_evidence(6, "framework-summary", {"summary": _safe_rel(summary_path), "verdict": verdict, "recommendation": keep_recommendation})
    print(f"Wrote evidence: {ev6}")


if __name__ == "__main__":
    main()

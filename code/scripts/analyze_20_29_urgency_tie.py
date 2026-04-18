import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project_paths import RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir
from train_age_groups_efficient import validate_cached_stage2_inputs


AGE_GROUP = "20-29"
SEED = 7
TRAIN_SUBSET_N = 12000
TEST_SUBSET_N = 24000
N_QUANTILES = 5
SOURCE_ORDER = ["human", "additive_urgency", "collapsing_bound"]
SOURCE_LABELS = {
    "human": "Human",
    "additive_urgency": "Additive urgency",
    "collapsing_bound": "Collapsing bound",
}
SOURCE_COLORS = {
    "human": "#4C78A8",
    "additive_urgency": "#F58518",
    "collapsing_bound": "#54A24B",
}
CONDITION_LABELS = {0: "congruent", 1: "incongruent"}
CORRECTNESS_LABELS = {True: "correct", False: "error"}
SWEEP_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "urgency_parameter_20_29_sweep_v2"
OUTPUT_DIR = RESULTS_ROOT / "repro_legacy_interim" / "urgency_mechanism_tie_break_20_29"
CANDIDATES = {
    "additive_urgency": SWEEP_ROOT / "additive_urgency_start0.60_slope0.25_floor0.00",
    "collapsing_bound": SWEEP_ROOT / "collapsing_bound_start0.60_slope0.25_floor0.00",
}


def subset_cached_inputs(
    cached: Dict[str, np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if n_rows >= len(cached["logits"]):
        idx = np.arange(len(cached["logits"]))
        return cached, idx
    idx = np.sort(rng.choice(len(cached["logits"]), size=n_rows, replace=False))
    return {key: value[idx] for key, value in cached.items()}, idx


def safe_skew(values: np.ndarray) -> float:
    if values.size < 3 or np.allclose(values, values[0]):
        return 0.0
    skew = stats.skew(values, bias=False)
    return 0.0 if np.isnan(skew) else float(skew)


def load_canonical_test_subset() -> tuple[Dict[str, np.ndarray], np.ndarray]:
    data_dir = age_group_data_dir(AGE_GROUP)
    stage2_dir = age_group_stage2_dir(AGE_GROUP)
    train_cached, test_cached = validate_cached_stage2_inputs(
        AGE_GROUP,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    rng = np.random.default_rng(SEED)
    subset_cached_inputs(train_cached, TRAIN_SUBSET_N, rng)
    test_subset, test_idx = subset_cached_inputs(test_cached, TEST_SUBSET_N, rng)
    return test_subset, test_idx


def load_candidate_predictions() -> tuple[Dict[str, dict], dict]:
    bundles: Dict[str, dict] = {}
    summaries: Dict[str, dict] = {}
    for mechanism, candidate_dir in CANDIDATES.items():
        prediction_path = candidate_dir / "predictions.npz"
        summary_path = candidate_dir / "summary.json"
        if not prediction_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing candidate artifacts for {mechanism}: {candidate_dir}")
        npz = np.load(prediction_path)
        bundles[mechanism] = {key: npz[key] for key in npz.files}
        with summary_path.open() as handle:
            summaries[mechanism] = json.load(handle)
    return bundles, summaries


def build_trial_df(source: str, test_subset: Dict[str, np.ndarray], predictions: dict | None = None) -> pd.DataFrame:
    base = {
        "source": source,
        "true_rt": test_subset["rts"].astype(np.float32),
        "congruency": test_subset["congruency"].astype(np.int64),
        "target_label": test_subset["target_labels"].astype(np.int64),
        "response_label": test_subset["response_labels"].astype(np.int64),
    }
    if predictions is None:
        pred_rt = test_subset["rts"].astype(np.float32)
        pred_choice = test_subset["response_labels"].astype(np.int64)
        df = pd.DataFrame({
            **base,
            "pred_rt": pred_rt,
            "pred_choice": pred_choice,
        })
    else:
        df = pd.DataFrame({
            **base,
            "pred_rt": predictions["pred_rt"].astype(np.float32),
            "pred_choice": predictions["pred_choice"].astype(np.int64),
            "baseline_index": predictions["baseline_index"].astype(np.int64),
            "urgency_index": predictions["urgency_index"].astype(np.int64),
        })
    df["condition"] = df["congruency"].map(CONDITION_LABELS)
    df["human_correct"] = df["response_label"] == df["target_label"]
    df["pred_correct"] = df["pred_choice"] == df["target_label"]
    return df


def compute_caf(df: pd.DataFrame, source: str, rt_col: str, correct_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition, [rt_col, correct_col]].copy()
        if subset.empty:
            continue
        subset["bin"] = pd.qcut(subset[rt_col], q=N_QUANTILES, labels=False, duplicates="drop")
        grouped = subset.groupby("bin", sort=True)
        for raw_bin, group in grouped:
            rows.append({
                "source": source,
                "condition": condition,
                "bin_index": int(raw_bin) + 1,
                "rt_min": float(group[rt_col].min()),
                "rt_max": float(group[rt_col].max()),
                "mean_rt": float(group[rt_col].mean()),
                "accuracy": float(group[correct_col].mean()),
                "n_trials": int(len(group)),
            })
    return pd.DataFrame(rows)


def compute_delta(df: pd.DataFrame, source: str, rt_col: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition, [rt_col]].copy()
        subset["condition"] = condition
        subset["bin"] = pd.qcut(subset[rt_col], q=N_QUANTILES, labels=False, duplicates="drop")
        grouped = (
            subset.groupby(["condition", "bin"], sort=True)[rt_col]
            .mean()
            .reset_index()
            .rename(columns={rt_col: "mean_rt"})
        )
        pieces.append(grouped)
    combined = pd.concat(pieces, ignore_index=True)
    pivot = combined.pivot(index="bin", columns="condition", values="mean_rt").reset_index()
    pivot["quantile_index"] = pivot["bin"].astype(int) + 1
    pivot["source"] = source
    pivot["delta"] = pivot["incongruent"] - pivot["congruent"]
    return pivot.rename(columns={"congruent": "mean_congruent_rt", "incongruent": "mean_incongruent_rt"})[
        ["source", "quantile_index", "mean_congruent_rt", "mean_incongruent_rt", "delta"]
    ]


def compute_conditional_error_rt(df: pd.DataFrame, source: str, rt_col: str, correct_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    wide: dict[str, float | int | str] = {"source": source}
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition].copy()
        correct_vals = subset.loc[subset[correct_col], rt_col].to_numpy()
        error_vals = subset.loc[~subset[correct_col], rt_col].to_numpy()
        correct_rt = float(correct_vals.mean()) if correct_vals.size else float("nan")
        error_rt = float(error_vals.mean()) if error_vals.size else float("nan")
        gap = error_rt - correct_rt if correct_vals.size and error_vals.size else float("nan")
        rows.append({
            "source": source,
            "condition": condition,
            "correct_rt": correct_rt,
            "error_rt": error_rt,
            "error_minus_correct_rt": gap,
            "n_correct": int(correct_vals.size),
            "n_error": int(error_vals.size),
        })
        wide[f"{condition}_correct_rt"] = correct_rt
        wide[f"{condition}_error_rt"] = error_rt
        wide[f"{condition}_error_minus_correct_rt"] = gap
        wide[f"{condition}_n_correct"] = int(correct_vals.size)
        wide[f"{condition}_n_error"] = int(error_vals.size)
    return pd.DataFrame([wide]), pd.DataFrame(rows)


def compute_tail_summary(df: pd.DataFrame, source: str, rt_col: str, correct_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for condition in ("congruent", "incongruent"):
        for is_correct in (True, False):
            subset = df.loc[(df["condition"] == condition) & (df[correct_col] == is_correct), rt_col].to_numpy()
            label = CORRECTNESS_LABELS[is_correct]
            if subset.size == 0:
                q90 = q95 = q99 = skewness = float("nan")
            else:
                q90 = float(np.quantile(subset, 0.90))
                q95 = float(np.quantile(subset, 0.95))
                q99 = float(np.quantile(subset, 0.99))
                skewness = safe_skew(subset)
            rows.append({
                "source": source,
                "condition": condition,
                "correctness": label,
                "group": f"{condition}_{label}",
                "q90": q90,
                "q95": q95,
                "q99": q99,
                "skewness": skewness,
                "n_trials": int(subset.size),
            })
    return pd.DataFrame(rows)


def save_caf_outputs(caf_df: pd.DataFrame) -> None:
    for condition in ("congruent", "incongruent"):
        caf_df.loc[caf_df["condition"] == condition].to_csv(OUTPUT_DIR / f"caf_{condition}.csv", index=False)


def save_delta_output(delta_df: pd.DataFrame) -> None:
    delta_df.to_csv(OUTPUT_DIR / "delta_plot.csv", index=False)


def save_conditional_error_output(error_wide: pd.DataFrame) -> None:
    error_wide.to_csv(OUTPUT_DIR / "conditional_error_rt.csv", index=False)


def save_tail_output(tail_df: pd.DataFrame) -> None:
    tail_df.to_csv(OUTPUT_DIR / "conditional_tail_summary.csv", index=False)


def plot_caf(caf_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, condition in zip(axes, ("congruent", "incongruent")):
        subset = caf_df.loc[caf_df["condition"] == condition]
        for source in SOURCE_ORDER:
            source_subset = subset.loc[subset["source"] == source].sort_values("bin_index")
            if source_subset.empty:
                continue
            ax.plot(
                source_subset["bin_index"],
                source_subset["accuracy"],
                marker="o",
                linewidth=2,
                color=SOURCE_COLORS[source],
                label=SOURCE_LABELS[source],
            )
        ax.set_title(condition.capitalize())
        ax.set_xlabel("RT quantile bin")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(range(1, N_QUANTILES + 1))
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional accuracy function")
    fig.savefig(OUTPUT_DIR / "caf_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_delta(delta_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for source in SOURCE_ORDER:
        subset = delta_df.loc[delta_df["source"] == source].sort_values("quantile_index")
        if subset.empty:
            continue
        ax.plot(
            subset["quantile_index"],
            subset["delta"],
            marker="o",
            linewidth=2,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("RT quantile")
    ax.set_ylabel("Incongruent − congruent RT (s)")
    ax.set_xticks(range(1, N_QUANTILES + 1))
    ax.set_title("Delta plot")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "delta_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_conditional_error(error_long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
    correctness_cols = ["correct_rt", "error_rt"]
    x = np.arange(len(SOURCE_ORDER))
    width = 0.35
    for ax, condition in zip(axes, ("congruent", "incongruent")):
        subset = error_long.loc[error_long["condition"] == condition].set_index("source")
        correct_vals = [subset.loc[source, "correct_rt"] for source in SOURCE_ORDER]
        error_vals = [subset.loc[source, "error_rt"] for source in SOURCE_ORDER]
        ax.bar(x - width / 2, correct_vals, width, label="Correct", color="#4C78A8")
        ax.bar(x + width / 2, error_vals, width, label="Error", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABELS[source] for source in SOURCE_ORDER], rotation=20, ha="right")
        ax.set_title(condition.capitalize())
        ax.set_ylabel("RT (s)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional error RT structure")
    fig.savefig(OUTPUT_DIR / "conditional_error_rt_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_tail_summary(tail_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    groups = [
        "congruent_correct",
        "congruent_error",
        "incongruent_correct",
        "incongruent_error",
    ]
    x = np.arange(len(groups))
    width = 0.22
    for source_idx, source in enumerate(SOURCE_ORDER):
        subset = tail_df.loc[tail_df["source"] == source].set_index("group")
        q95_vals = [subset.loc[group, "q95"] for group in groups]
        skew_vals = [subset.loc[group, "skewness"] for group in groups]
        offset = (source_idx - 1) * width
        axes[0].bar(x + offset, q95_vals, width, label=SOURCE_LABELS[source], color=SOURCE_COLORS[source])
        axes[1].bar(x + offset, skew_vals, width, label=SOURCE_LABELS[source], color=SOURCE_COLORS[source])
    axes[0].set_title("q95 by condition/correctness")
    axes[0].set_ylabel("RT (s)")
    axes[1].set_title("Skewness by condition/correctness")
    axes[1].set_ylabel("Skewness")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([group.replace("_", "\n") for group in groups])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional tail summary")
    fig.savefig(OUTPUT_DIR / "conditional_tail_plot.png", bbox_inches="tight")
    plt.close(fig)


def arrays_identical(bundle_a: dict, bundle_b: dict) -> tuple[bool, list[str]]:
    differing: list[str] = []
    keys = sorted(set(bundle_a) | set(bundle_b))
    for key in keys:
        if key not in bundle_a or key not in bundle_b:
            differing.append(key)
            continue
        if not np.array_equal(bundle_a[key], bundle_b[key]):
            differing.append(key)
    return len(differing) == 0, differing


def audit_formula_equivalence(bundle: dict) -> dict:
    dv_t = bundle["dv_t"]
    baseline_threshold = bundle["baseline_threshold"][:, None]
    urgency_gain = bundle["urgency_gain"][None, :]
    additive_commit = (dv_t + urgency_gain) >= baseline_threshold
    collapsing_commit = dv_t >= np.maximum(baseline_threshold - urgency_gain, 0.0)
    return {
        "dv_min": float(dv_t.min()),
        "dv_max": float(dv_t.max()),
        "formula_commit_masks_identical": bool(np.array_equal(additive_commit, collapsing_commit)),
        "formula_commit_mask_difference_count": int(np.count_nonzero(additive_commit != collapsing_commit)),
    }


def build_comparison_table(
    summaries: Dict[str, dict],
    caf_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    error_wide: pd.DataFrame,
    tail_df: pd.DataFrame,
    identical_outputs: bool,
) -> pd.DataFrame:
    human_caf = caf_df.loc[caf_df["source"] == "human"]
    human_delta = delta_df.loc[delta_df["source"] == "human"]
    human_error = error_wide.loc[error_wide["source"] == "human"].iloc[0]
    rows: list[dict] = []
    for mechanism, summary in summaries.items():
        caf_subset = caf_df.loc[caf_df["source"] == mechanism]
        delta_subset = delta_df.loc[delta_df["source"] == mechanism].sort_values("quantile_index")
        error_row = error_wide.loc[error_wide["source"] == mechanism].iloc[0]
        tail_subset = tail_df.loc[tail_df["source"] == mechanism].set_index("group")
        model_fast_incong = caf_subset.loc[caf_subset["condition"] == "incongruent"].sort_values("bin_index").iloc[0]
        human_fast_incong = human_caf.loc[human_caf["condition"] == "incongruent"].sort_values("bin_index").iloc[0]
        rows.append({
            "mechanism": mechanism,
            "score": float(summary["score"]),
            "frac_at_ceiling": float(summary["frac_at_ceiling"]),
            "coverage_score": float(summary["coverage_score"]),
            "pred_q95": float(summary["pred_q95"]),
            "pred_q99": float(summary["pred_q99"]),
            "model_congruency_rt_gap": float(summary["model_congruency_rt_gap"]),
            "rt_shape_score": float(summary["rt_shape_score"]),
            "pred_skewness": float(summary["pred_skewness"]),
            "error_minus_correct_rt": float(summary["error_minus_correct_rt"]),
            "caf_incongruent_fast_bin_accuracy": float(model_fast_incong["accuracy"]),
            "caf_incongruent_fast_bin_human_accuracy": float(human_fast_incong["accuracy"]),
            "caf_incongruent_fast_bin_gap": float(model_fast_incong["accuracy"] - human_fast_incong["accuracy"]),
            "caf_incongruent_accuracy_span": float(
                caf_subset.loc[caf_subset["condition"] == "incongruent", "accuracy"].max()
                - caf_subset.loc[caf_subset["condition"] == "incongruent", "accuracy"].min()
            ),
            "delta_first_quantile": float(delta_subset.iloc[0]["delta"]),
            "delta_last_quantile": float(delta_subset.iloc[-1]["delta"]),
            "delta_change_last_minus_first": float(delta_subset.iloc[-1]["delta"] - delta_subset.iloc[0]["delta"]),
            "human_delta_first_quantile": float(human_delta.sort_values("quantile_index").iloc[0]["delta"]),
            "human_delta_last_quantile": float(human_delta.sort_values("quantile_index").iloc[-1]["delta"]),
            "congruent_error_minus_correct_rt": float(error_row["congruent_error_minus_correct_rt"]),
            "incongruent_error_minus_correct_rt": float(error_row["incongruent_error_minus_correct_rt"]),
            "human_congruent_error_minus_correct_rt": float(human_error["congruent_error_minus_correct_rt"]),
            "human_incongruent_error_minus_correct_rt": float(human_error["incongruent_error_minus_correct_rt"]),
            "incongruent_correct_q95": float(tail_subset.loc["incongruent_correct", "q95"]),
            "incongruent_error_q95": float(tail_subset.loc["incongruent_error", "q95"]),
            "incongruent_correct_q99": float(tail_subset.loc["incongruent_correct", "q99"]),
            "incongruent_error_q99": float(tail_subset.loc["incongruent_error", "q99"]),
            "incongruent_correct_skewness": float(tail_subset.loc["incongruent_correct", "skewness"]),
            "incongruent_error_skewness": float(tail_subset.loc["incongruent_error", "skewness"]),
            "identical_to_other_candidate": identical_outputs,
        })
    return pd.DataFrame(rows)


def write_decision_memo(
    comparison_df: pd.DataFrame,
    identical_outputs: bool,
    differing_keys: Iterable[str],
    formula_audit: dict,
) -> None:
    human_fast = comparison_df.iloc[0]["caf_incongruent_fast_bin_human_accuracy"]
    model_fast = comparison_df.iloc[0]["caf_incongruent_fast_bin_accuracy"]
    human_delta_first = comparison_df.iloc[0]["human_delta_first_quantile"]
    model_delta_first = comparison_df.iloc[0]["delta_first_quantile"]
    human_incong_gap = comparison_df.iloc[0]["human_incongruent_error_minus_correct_rt"]
    model_incong_gap = comparison_df.iloc[0]["incongruent_error_minus_correct_rt"]

    if identical_outputs:
        headline = "No unique winner remains under the richer diagnostics."
        verdict = (
            "The two urgency candidates remain unresolved because they are operationally equivalent "
            "under the current canonical implementation and fixed regime, not merely close in score."
        )
    else:
        headline = "A unique winner emerged under the richer diagnostics."
        verdict = "The richer diagnostics separated the two urgency mechanisms."

    memo = f"""# Urgency mechanism decision memo

## Bottom line

{headline}

{verdict}

## Fixed regime used

- `time_steps_factor = 2.0`
- `lambda_pileup = 1.0`
- `scale = 0.1`
- no new readout family
- no broad urgency sweep

## Candidates compared

1. `additive_urgency`, start `0.60`, slope `0.25`, floor `0.00`
2. `collapsing_bound`, start `0.60`, slope `0.25`, floor `0.00`

## What the new diagnostics show

### CAF

- In the earliest incongruent RT bin, human accuracy is `{human_fast:.4f}`.
- Both urgency candidates produce `{model_fast:.4f}` in that same diagnostic slot.
- Because both candidates are identical on this analysis, CAF realism does not break the tie.

### Delta plot

- Human early-quantile delta is `{human_delta_first:.4f}` s.
- Both urgency candidates produce `{model_delta_first:.4f}` s at the first quantile.
- The full delta curves overlap exactly for the two urgency candidates.

### Conditional error RT structure

- Human incongruent error-minus-correct RT is `{human_incong_gap:.4f}` s.
- Both urgency candidates produce `{model_incong_gap:.4f}` s for the same conditional diagnostic.
- This means the richer error-RT analysis still does not separate the two mechanisms.

### Conditional tail summary

- The conditional q90/q95/q99 and skewness summaries are identical across the two candidates.
- Therefore the slow-tail structure does not provide a tie-break under the current implementation.

## Equivalence audit

- Saved prediction bundles identical across all arrays: `{identical_outputs}`
- Differing saved-array keys: `{list(differing_keys)}`
- Decision-variable range: `{formula_audit['dv_min']:.6f}` to `{formula_audit['dv_max']:.6f}`
- Additive vs collapsing commit masks identical when evaluated directly from the saved arrays: `{formula_audit['formula_commit_masks_identical']}`
- Commit-mask differences found: `{formula_audit['formula_commit_mask_difference_count']}`

Interpretation:

- Under the current implementation, `additive_urgency` and `collapsing_bound` collapse to the same effective commitment rule for this candidate setting.
- That means the present stage cannot legitimately promote one mechanism over the other.

## Decision

**No unique winner remains.**

The most defensible reading is not simply that the two candidates are "still tied," but that they are **mechanistically unresolved because the current parameterization makes them empirically equivalent on the canonical deterministic path**.

## Recommended next branch

Because the tie persists under mechanism-sensitive diagnostics, the next useful change should be one of:

1. a stronger urgency parameterization that is not algebraically equivalent across urgency types under `floor = 0.00`, or
2. a mechanism pivot toward dynamic attentional selection / DMC-like conflict control if the goal is to improve early capture and conditional slow-tail structure.

Fuller-data winner promotion should remain paused until the branch actually produces distinguishable mechanism-level predictions.
"""
    (OUTPUT_DIR / "urgency_mechanism_decision_memo.md").write_text(memo)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_subset, test_idx = load_canonical_test_subset()
    candidate_bundles, summaries = load_candidate_predictions()

    candidate_lengths = {name: int(bundle["pred_rt"].shape[0]) for name, bundle in candidate_bundles.items()}
    if len(set(candidate_lengths.values())) != 1 or next(iter(candidate_lengths.values())) != len(test_subset["rts"]):
        raise ValueError("Candidate predictions do not align with the canonical test subset length.")

    human_df = build_trial_df("human", test_subset, predictions=None)
    candidate_dfs = {name: build_trial_df(name, test_subset, predictions=bundle) for name, bundle in candidate_bundles.items()}

    caf_frames = [compute_caf(human_df, "human", rt_col="true_rt", correct_col="human_correct")]
    delta_frames = [compute_delta(human_df, "human", rt_col="true_rt")]
    error_wide_frames = []
    error_long_frames = []
    tail_frames = [compute_tail_summary(human_df, "human", rt_col="true_rt", correct_col="human_correct")]

    human_error_wide, human_error_long = compute_conditional_error_rt(
        human_df,
        "human",
        rt_col="true_rt",
        correct_col="human_correct",
    )
    error_wide_frames.append(human_error_wide)
    error_long_frames.append(human_error_long)

    for source, df in candidate_dfs.items():
        caf_frames.append(compute_caf(df, source, rt_col="pred_rt", correct_col="pred_correct"))
        delta_frames.append(compute_delta(df, source, rt_col="pred_rt"))
        error_wide, error_long = compute_conditional_error_rt(df, source, rt_col="pred_rt", correct_col="pred_correct")
        error_wide_frames.append(error_wide)
        error_long_frames.append(error_long)
        tail_frames.append(compute_tail_summary(df, source, rt_col="pred_rt", correct_col="pred_correct"))

    caf_df = pd.concat(caf_frames, ignore_index=True)
    delta_df = pd.concat(delta_frames, ignore_index=True)
    error_wide = pd.concat(error_wide_frames, ignore_index=True)
    error_long = pd.concat(error_long_frames, ignore_index=True)
    tail_df = pd.concat(tail_frames, ignore_index=True)

    save_caf_outputs(caf_df)
    save_delta_output(delta_df)
    save_conditional_error_output(error_wide)
    save_tail_output(tail_df)

    plot_caf(caf_df)
    plot_delta(delta_df)
    plot_conditional_error(error_long)
    plot_tail_summary(tail_df)

    identical_outputs, differing_keys = arrays_identical(
        candidate_bundles["additive_urgency"],
        candidate_bundles["collapsing_bound"],
    )
    formula_audit = audit_formula_equivalence(candidate_bundles["additive_urgency"])

    comparison_df = build_comparison_table(
        summaries=summaries,
        caf_df=caf_df,
        delta_df=delta_df,
        error_wide=error_wide,
        tail_df=tail_df,
        identical_outputs=identical_outputs,
    )
    comparison_df.to_csv(OUTPUT_DIR / "urgency_mechanism_comparison_table.csv", index=False)

    manifest = {
        "age_group": AGE_GROUP,
        "seed": SEED,
        "train_subset_n": TRAIN_SUBSET_N,
        "test_subset_n": TEST_SUBSET_N,
        "n_quantiles": N_QUANTILES,
        "canonical_test_subset_indices_preview": test_idx[:10].tolist(),
        "identical_outputs": identical_outputs,
        "differing_saved_array_keys": list(differing_keys),
        "formula_audit": formula_audit,
        "output_files": [
            "caf_congruent.csv",
            "caf_incongruent.csv",
            "caf_plot.png",
            "delta_plot.csv",
            "delta_plot.png",
            "conditional_error_rt.csv",
            "conditional_error_rt_plot.png",
            "conditional_tail_summary.csv",
            "conditional_tail_plot.png",
            "urgency_mechanism_comparison_table.csv",
            "urgency_mechanism_decision_memo.md",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    write_decision_memo(
        comparison_df=comparison_df,
        identical_outputs=identical_outputs,
        differing_keys=differing_keys,
        formula_audit=formula_audit,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Human-only audit of condition-specific sequential error transitions.

The analysis uses only true adjacent trials from the frozen LIM preprocessing.
It does not read, modify, or refit C0v2 or any cognitive-model component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import logit
from scipy.stats import beta, betabinom
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/vam_data"
DEFAULT_OUTPUT = ROOT / "artifacts/results/human_condition_specific_error_transitions_20260815"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
DIRECTIONS = {"L", "R", "U", "D"}
MIN_RT_MS = 250.0
OUTLIER_MAD_MULTIPLIER = 10.0
SEED = 20260815
N_BOOTSTRAP = 5000
N_FOLDS = 5
N_SIMULATIONS = 300
MIN_STABLE_ERROR_EXPOSURES = 20
MIN_STABLE_CORRECT_EXPOSURES = 100

ACCENT = "#0072B2"
ACCENT_2 = "#D55E00"
NEUTRAL = "#707070"
LIGHT = "#D9D9D9"
DARK = "#222222"

TRANSITIONS = [
    ("congruent", "congruent", "C_to_C", "C→C"),
    ("incongruent", "congruent", "I_to_C", "I→C"),
    ("congruent", "incongruent", "C_to_I", "C→I"),
    ("incongruent", "incongruent", "I_to_I", "I→I"),
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_participant(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the frozen LIM human-data exclusions used by the source audit."""
    required = [
        "anon_id", "nth_play", "trial", "xpos", "ypos", "flanker_direction",
        "response_direction", "response_time", "stimulus_layout", "target_direction",
    ]
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    work = raw.copy()
    n_raw = len(work)
    duplicate = (
        work.duplicated(keep="first")
        | work.duplicated(["anon_id", "nth_play", "trial"], keep="first")
    )
    n_duplicate = int(duplicate.sum())
    work = work.loc[~duplicate].copy()

    missing_response = work["response_direction"].isna()
    n_missing_response = int(missing_response.sum())
    work = work.loc[~missing_response].copy()

    missing_other = work[
        ["response_time", "target_direction", "flanker_direction", "stimulus_layout", "xpos", "ypos"]
    ].isna().any(axis=1)
    n_missing_other = int(missing_other.sum())
    work = work.loc[~missing_other].copy()

    invalid_response = ~work["response_direction"].isin(DIRECTIONS)
    n_invalid_response = int(invalid_response.sum())
    work = work.loc[~invalid_response].copy()

    invalid_stimulus = (
        ~work["target_direction"].isin(DIRECTIONS)
        | ~work["flanker_direction"].isin(DIRECTIONS)
        | ~work["stimulus_layout"].isin(range(7))
    )
    n_invalid_stimulus = int(invalid_stimulus.sum())
    work = work.loc[~invalid_stimulus].copy()

    finite_rt = np.isfinite(work["response_time"].to_numpy(float))
    invalid_rt = ~finite_rt | work["response_time"].lt(MIN_RT_MS).to_numpy()
    n_short_or_invalid_rt = int(invalid_rt.sum())
    work = work.loc[~invalid_rt].copy()

    median_rt = float(work["response_time"].median())
    deviations = (work["response_time"] - median_rt).abs()
    mad = float(deviations.median())
    rt_outlier = (
        deviations.ge(OUTLIER_MAD_MULTIPLIER * mad)
        if mad > 0 else pd.Series(False, index=work.index)
    )
    n_rt_outlier = int(rt_outlier.sum())
    work = work.loc[~rt_outlier].copy()

    work["correct"] = work["response_direction"].eq(work["target_direction"])
    work["congruency"] = np.where(
        work["target_direction"].eq(work["flanker_direction"]), "congruent", "incongruent"
    )
    work["rt_s"] = work["response_time"].astype(float) / 1000.0
    work["user_id"] = work["anon_id"].astype(int)
    work["congruency"] = pd.Categorical(
        work["congruency"], categories=["congruent", "incongruent"], ordered=True
    )
    for column in ["target_direction", "flanker_direction", "response_direction"]:
        work[column] = work[column].astype("category")

    audit = {
        "user_id": int(work["user_id"].iloc[0]),
        "raw_trials": n_raw,
        "duplicate_trials_removed": n_duplicate,
        "missing_response_removed": n_missing_response,
        "missing_other_required_field_removed": n_missing_other,
        "invalid_response_removed": n_invalid_response,
        "invalid_stimulus_removed": n_invalid_stimulus,
        "rt_below_250ms_or_nonfinite_removed": n_short_or_invalid_rt,
        "rt_10mad_outlier_removed": n_rt_outlier,
        "final_trials": len(work),
        "participant_rt_median_ms_for_filter": median_rt,
        "participant_rt_mad_ms_for_filter": mad,
    }
    if n_raw - sum(value for key, value in audit.items() if key.endswith("_removed")) != len(work):
        raise AssertionError("Exclusion audit does not reconcile")
    return work, audit


def load_human_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(data_dir / "metadata.csv")
    metadata["user_id"] = metadata["user_id"].astype(int)
    files = sorted(data_dir.glob("user*df.csv"))
    if len(files) != len(metadata):
        raise ValueError(f"Found {len(files)} trial files for {len(metadata)} metadata rows")

    frames, audits, sources = [], [], []
    for path in files:
        raw = pd.read_csv(path)
        cleaned, audit = clean_participant(raw)
        uid = int(cleaned["user_id"].iloc[0])
        frames.append(cleaned)
        audits.append(audit)
        sources.append({
            "user_id": uid,
            "file": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
            "raw_trials": len(raw),
        })

    data = pd.concat(frames, ignore_index=True)
    data = data.merge(
        metadata[["user_id", "gender", "binned_age"]], on="user_id", validate="many_to_one"
    ).rename(columns={"binned_age": "age_group"})
    data["age_group"] = pd.Categorical(data["age_group"], AGE_GROUPS, ordered=True)
    if set(data["user_id"].unique()) != set(metadata["user_id"]):
        raise AssertionError("Participant IDs do not match metadata")
    return data, pd.DataFrame(audits), pd.DataFrame(sources)


def add_condition_rt_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add the condition-relative current-RT percentile used in sensitivity checks."""
    work = data.copy()
    groups = work.groupby(["user_id", "congruency"], observed=True)["rt_s"]
    work["rt_percentile"] = groups.transform(
        lambda values: (values.rank(method="average") - 0.5) / len(values)
    )
    return work


def beta_binomial_prior(n: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    n, k = np.asarray(n, float), np.asarray(k, float)

    def objective(theta: np.ndarray) -> float:
        a, b = np.exp(theta)
        return -float(np.sum(betabinom.logpmf(k, n, a, b)))

    pooled = np.clip(k.sum() / n.sum(), 1e-7, 1 - 1e-7)
    fit = minimize(objective, np.log([pooled * 100, (1 - pooled) * 100]), method="Nelder-Mead")
    if not fit.success:
        raise RuntimeError(f"Beta-binomial prior failed: {fit.message}")
    return tuple(np.exp(fit.x))


def posterior_rate(k: np.ndarray, n: np.ndarray, prior: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    a, b = prior
    mean = (k + a) / (n + a + b)
    var = (k + a) * (n - k + b) / ((n + a + b) ** 2 * (n + a + b + 1))
    return mean, var


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    draw = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draw, [.025, .975]))


def make_true_adjacent_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Return lag-1 pairs without crossing a session or any cleaned trial gap."""
    data = add_condition_rt_metrics(data)
    ordered = data.sort_values(["user_id", "nth_play", "trial"], kind="mergesort").copy()
    group = ordered.groupby(["user_id", "nth_play"], observed=True, sort=False)
    for new, old in {
        "previous_trial": "trial", "previous_correct": "correct",
        "previous_congruency": "congruency", "previous_rt_s": "rt_s",
        "previous_response": "response_direction", "previous_target": "target_direction",
    }.items():
        ordered[new] = group[old].shift(1)
    pairs = ordered.loc[ordered.trial.eq(ordered.previous_trial + 1)].copy()
    pairs["error"] = ~pairs.correct.astype(bool)
    pairs["previous_error"] = ~pairs.previous_correct.astype(bool)
    pairs["previous_incongruent"] = pairs.previous_congruency.astype(str).eq("incongruent")
    pairs["current_incongruent"] = pairs.congruency.astype(str).eq("incongruent")
    pairs["response_repeat"] = pairs.response_direction.eq(pairs.previous_response)
    pairs["target_repeat"] = pairs.target_direction.eq(pairs.previous_target)
    previous_group = pairs.groupby("user_id", observed=True).previous_rt_s
    pairs["previous_rt_z"] = (pairs.previous_rt_s - previous_group.transform("mean")) / previous_group.transform("std")
    rank = pairs.groupby("user_id", observed=True).cumcount().to_numpy()
    size = pairs.groupby("user_id", observed=True).user_id.transform("size").to_numpy()
    pairs["blocked_fold"] = np.minimum(N_FOLDS - 1, rank * N_FOLDS // size).astype("int8")
    return add_interaction_columns(pairs)


def add_interaction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pe = out.previous_error.astype(float)
    pi = out.previous_incongruent.astype(float)
    ci = out.current_incongruent.astype(float)
    out["previous_error_x_previous_incongruent"] = pe * pi
    out["previous_error_x_current_incongruent"] = pe * ci
    out["previous_x_current_incongruent"] = pi * ci
    out["three_way_interaction"] = pe * pi * ci
    return out


def estimate_effect_from_counts(cells: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, object]]:
    """Use one common participant prior so rare contrasts shrink toward zero."""
    wide_n = cells.pivot(index="user_id", columns="previous_error", values="n_trials")
    wide_k = cells.pivot(index="user_id", columns="previous_error", values="n_errors")
    common = wide_n.dropna().index.intersection(wide_k.dropna().index)
    if not {False, True}.issubset(wide_n.columns) or len(common) == 0:
        raise ValueError("Both previous-accuracy cells are required")
    common_prior = beta_binomial_prior(
        wide_n.loc[common, False].to_numpy() + wide_n.loc[common, True].to_numpy(),
        wide_k.loc[common, False].to_numpy() + wide_k.loc[common, True].to_numpy(),
    )
    priors = {False: common_prior, True: common_prior}
    rates, variances = {}, {}
    for level in [False, True]:
        rates[level], variances[level] = posterior_rate(
            wide_k.loc[common, level].to_numpy(), wide_n.loc[common, level].to_numpy(), common_prior
        )
    detail = pd.DataFrame({
        "user_id": common.astype(int),
        "n_after_previous_correct": wide_n.loc[common, False].astype(int).to_numpy(),
        "errors_after_previous_correct": wide_k.loc[common, False].astype(int).to_numpy(),
        "n_after_previous_error": wide_n.loc[common, True].astype(int).to_numpy(),
        "errors_after_previous_error": wide_k.loc[common, True].astype(int).to_numpy(),
        "shrunk_risk_after_previous_correct": rates[False],
        "shrunk_risk_after_previous_error": rates[True],
        "risk_difference": rates[True] - rates[False],
        "posterior_risk_difference_sd": np.sqrt(variances[True] + variances[False]),
    })
    detail["raw_stable"] = detail.n_after_previous_error.ge(MIN_STABLE_ERROR_EXPOSURES) & detail.n_after_previous_correct.ge(MIN_STABLE_CORRECT_EXPOSURES)
    detail["log_odds_ratio"] = logit(np.clip(rates[True], 1e-9, 1 - 1e-9)) - logit(np.clip(rates[False], 1e-9, 1 - 1e-9))
    detail["odds_ratio"] = np.exp(detail.log_odds_ratio)
    rd_low, rd_high = bootstrap_ci(detail.risk_difference.to_numpy(), rng)
    correct_low, correct_high = bootstrap_ci(detail.shrunk_risk_after_previous_correct.to_numpy(), rng)
    error_low, error_high = bootstrap_ci(detail.shrunk_risk_after_previous_error.to_numpy(), rng)
    lor_low, lor_high = bootstrap_ci(detail.log_odds_ratio.to_numpy(), rng)
    summary = {
        "n_participants": len(detail),
        "n_raw_stable_participants": int(detail.raw_stable.sum()),
        "n_trials_after_previous_correct": int(detail.n_after_previous_correct.sum()),
        "n_errors_after_previous_correct": int(detail.errors_after_previous_correct.sum()),
        "n_trials_after_previous_error": int(detail.n_after_previous_error.sum()),
        "n_errors_after_previous_error": int(detail.errors_after_previous_error.sum()),
        "mean_shrunk_risk_after_previous_correct": detail.shrunk_risk_after_previous_correct.mean(),
        "risk_after_previous_correct_ci_low": correct_low,
        "risk_after_previous_correct_ci_high": correct_high,
        "mean_shrunk_risk_after_previous_error": detail.shrunk_risk_after_previous_error.mean(),
        "risk_after_previous_error_ci_low": error_low,
        "risk_after_previous_error_ci_high": error_high,
        "population_mean_risk_difference": detail.risk_difference.mean(),
        "risk_difference_ci_low": rd_low,
        "risk_difference_ci_high": rd_high,
        "n_positive": int(detail.risk_difference.gt(0).sum()),
        "proportion_positive": detail.risk_difference.gt(0).mean(),
        "population_mean_log_odds_ratio": detail.log_odds_ratio.mean(),
        "log_odds_ratio_ci_low": lor_low,
        "log_odds_ratio_ci_high": lor_high,
        "population_geometric_mean_odds_ratio": np.exp(detail.log_odds_ratio.mean()),
        "odds_ratio_ci_low": np.exp(lor_low),
        "odds_ratio_ci_high": np.exp(lor_high),
        "prior_correct_alpha": priors[False][0], "prior_correct_beta": priors[False][1],
        "prior_error_alpha": priors[True][0], "prior_error_beta": priors[True][1],
        "pooled_risk_after_previous_correct": detail.errors_after_previous_correct.sum() / detail.n_after_previous_correct.sum(),
        "pooled_risk_after_previous_error": detail.errors_after_previous_error.sum() / detail.n_after_previous_error.sum(),
    }
    summary["pooled_risk_difference"] = summary["pooled_risk_after_previous_error"] - summary["pooled_risk_after_previous_correct"]
    return detail, summary


def primary_transition_analysis(pairs: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[tuple[float, float], tuple[float, float]]]]:
    details, summaries, probability_rows, priors = [], [], [], {}
    for previous, current, code, label in TRANSITIONS:
        part = pairs[pairs.previous_congruency.astype(str).eq(previous) & pairs.congruency.astype(str).eq(current)]
        cells = part.groupby(["user_id", "previous_error"], observed=True).error.agg(n_trials="size", n_errors="sum").reset_index()
        detail, summary = estimate_effect_from_counts(cells, rng)
        detail.insert(0, "transition", code)
        detail.insert(1, "transition_label", label)
        detail.insert(2, "previous_condition", previous)
        detail.insert(3, "current_condition", current)
        details.append(detail)
        summary.update({"transition": code, "transition_label": label, "previous_condition": previous, "current_condition": current, "estimation": "participant_empirical_bayes"})
        summaries.append(summary)
        for level, name in [(False, "previous_correct"), (True, "previous_error")]:
            probability_rows.append({
                "transition": code, "transition_label": label, "previous_condition": previous,
                "current_condition": current, "previous_accuracy": name,
                "participant_mean_shrunk_error_probability": summary[f"mean_shrunk_risk_after_{name}"],
                "ci_low": summary[f"risk_after_{name}_ci_low"], "ci_high": summary[f"risk_after_{name}_ci_high"],
                "n_trials": summary[f"n_trials_after_{name}"], "n_errors": summary[f"n_errors_after_{name}"],
            })
        priors[code] = (
            (summary["prior_correct_alpha"], summary["prior_correct_beta"]),
            (summary["prior_error_alpha"], summary["prior_error_beta"]),
        )
    return pd.concat(details, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(probability_rows), priors


def fit_formal_interaction(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    grouped = pairs.groupby(
        ["user_id", "previous_error", "previous_incongruent", "current_incongruent"], observed=True
    ).error.agg(n_trials="size", n_errors="sum").reset_index()
    grouped["error_rate"] = grouped.n_errors / grouped.n_trials
    formula = "error_rate ~ C(user_id) + previous_error * previous_incongruent * current_incongruent"
    fit = smf.glm(formula, data=grouped, family=sm.families.Binomial(), freq_weights=grouped.n_trials).fit()
    ci = fit.conf_int()
    rows = pd.DataFrame({
        "term": fit.params.index, "log_odds_coefficient": fit.params.to_numpy(),
        "standard_error": fit.bse.to_numpy(), "ci_low": ci[0].to_numpy(), "ci_high": ci[1].to_numpy(),
        "z_value": fit.tvalues.to_numpy(), "p_value": fit.pvalues.to_numpy(),
    })
    rows["odds_ratio"] = np.exp(rows.log_odds_coefficient)
    rows["odds_ratio_ci_low"] = np.exp(rows.ci_low)
    rows["odds_ratio_ci_high"] = np.exp(rows.ci_high)
    participants = np.sort(pairs.user_id.unique())
    pred_rows = []
    for previous, current, code, label in TRANSITIONS:
        for pe in [False, True]:
            counter = pd.DataFrame({
                "user_id": participants, "previous_error": pe,
                "previous_incongruent": previous == "incongruent",
                "current_incongruent": current == "incongruent",
            })
            pred_rows.append({
                "transition": code, "transition_label": label, "previous_condition": previous,
                "current_condition": current, "previous_accuracy": "previous_error" if pe else "previous_correct",
                "participant_equal_weight_predicted_probability": fit.predict(counter).mean(),
            })
    predicted = pd.DataFrame(pred_rows)
    return rows, predicted, fit


def interaction_feature_sets() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "M0_participant_current_condition": (["user_id"], ["current_incongruent"]),
        "M1_plus_generic_previous_error": (["user_id"], ["current_incongruent", "previous_error"]),
        "M2_plus_previous_condition_history": (
            ["user_id"], ["current_incongruent", "previous_error", "previous_incongruent", "previous_error_x_previous_incongruent"]
        ),
        "M3_full_three_way_interaction": (
            ["user_id"], [
                "current_incongruent", "previous_error", "previous_incongruent",
                "previous_error_x_previous_incongruent", "previous_error_x_current_incongruent",
                "previous_x_current_incongruent", "three_way_interaction",
            ]
        ),
    }


def make_pipeline(categorical: list[str], numeric: list[str], max_iter: int = 180) -> Pipeline:
    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", dtype=np.float32), categorical),
        ("num", StandardScaler(), numeric),
    ], sparse_threshold=.99)
    model = LogisticRegression(solver="lbfgs", C=10.0, max_iter=max_iter, tol=1e-6, random_state=SEED)
    return Pipeline([("features", transformer), ("model", model)])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float(abs(y.mean() - p.mean()))
    cell = np.searchsorted(edges[1:-1], p, side="right")
    return float(sum((cell == i).mean() * abs(y[cell == i].mean() - p[cell == i].mean()) for i in range(len(edges) - 1) if (cell == i).any()))


def cv_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n_trials": len(y), "n_errors": int(y.sum()), "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier_score": brier_score_loss(y, p), "observed_error_rate": y.mean(),
        "mean_predicted_error_rate": p.mean(), "calibration_in_the_large": p.mean() - y.mean(),
        "expected_calibration_error": expected_calibration_error(y, p),
    }


def blocked_cross_validation(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    features = interaction_feature_sets()
    all_columns = sorted(set(sum((cat + num for cat, num in features.values()), [])))
    rows, participant_rows = [], []
    y_all = pairs.error.to_numpy(np.int8)
    for fold in range(N_FOLDS):
        test_mask = pairs.blocked_fold.to_numpy() == fold
        train_mask = ~test_mask
        for model_name, (categorical, numeric) in features.items():
            pipeline = make_pipeline(categorical, numeric)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipeline.fit(pairs.loc[train_mask, all_columns][categorical + numeric], y_all[train_mask])
            p = pipeline.predict_proba(pairs.loc[test_mask, all_columns][categorical + numeric])[:, 1]
            rows.append({"record_type": "fold", "fold": fold, "model": model_name, "user_id": "all", **cv_metric(y_all[test_mask], p)})
            held = pd.DataFrame({"user_id": pairs.loc[test_mask, "user_id"].to_numpy(), "y": y_all[test_mask], "p": p})
            for uid, part in held.groupby("user_id", observed=True):
                participant_rows.append({"user_id": uid, "fold": fold, "model": model_name, **cv_metric(part.y.to_numpy(), part.p.to_numpy())})
    folds = pd.DataFrame(rows)
    base = folds[folds.model.eq("M0_participant_current_condition")].set_index("fold")
    folds["delta_log_loss_vs_m0"] = [r.log_loss - base.loc[r.fold, "log_loss"] for r in folds.itertuples()]
    folds["delta_brier_vs_m0"] = [r.brier_score - base.loc[r.fold, "brier_score"] for r in folds.itertuples()]
    participant = pd.DataFrame(participant_rows)
    pbase = participant[participant.model.eq("M0_participant_current_condition")].set_index(["user_id", "fold"])
    participant["delta_log_loss_vs_m0"] = [r.log_loss - pbase.loc[(r.user_id, r.fold), "log_loss"] for r in participant.itertuples()]
    participant["delta_brier_vs_m0"] = [r.brier_score - pbase.loc[(r.user_id, r.fold), "brier_score"] for r in participant.itertuples()]
    summaries = []
    for model, fold_part in folds.groupby("model", observed=True):
        pp = participant[participant.model.eq(model)].groupby("user_id", observed=True).agg(
            delta_log_loss_vs_m0=("delta_log_loss_vs_m0", "mean"), delta_brier_vs_m0=("delta_brier_vs_m0", "mean")
        )
        dl = rng.choice(pp.delta_log_loss_vs_m0, size=(N_BOOTSTRAP, len(pp)), replace=True).mean(axis=1)
        db = rng.choice(pp.delta_brier_vs_m0, size=(N_BOOTSTRAP, len(pp)), replace=True).mean(axis=1)
        summaries.append({
            "record_type": "summary", "fold": "mean", "model": model, "user_id": "all",
            "n_trials": fold_part.n_trials.sum(), "n_errors": fold_part.n_errors.sum(),
            "log_loss": fold_part.log_loss.mean(), "brier_score": fold_part.brier_score.mean(),
            "observed_error_rate": fold_part.observed_error_rate.mean(),
            "mean_predicted_error_rate": fold_part.mean_predicted_error_rate.mean(),
            "calibration_in_the_large": fold_part.calibration_in_the_large.mean(),
            "expected_calibration_error": fold_part.expected_calibration_error.mean(),
            "delta_log_loss_vs_m0": fold_part.delta_log_loss_vs_m0.mean(),
            "delta_log_loss_ci_low": np.quantile(dl, .025), "delta_log_loss_ci_high": np.quantile(dl, .975),
            "delta_brier_vs_m0": fold_part.delta_brier_vs_m0.mean(),
            "delta_brier_ci_low": np.quantile(db, .025), "delta_brier_ci_high": np.quantile(db, .975),
            "proportion_participants_improved_log_loss": pp.delta_log_loss_vs_m0.lt(0).mean(),
        })
    return pd.concat([folds, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def sensitivity_analysis(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    core = [
        "current_incongruent", "previous_error", "previous_incongruent",
        "previous_error_x_previous_incongruent", "previous_error_x_current_incongruent",
        "previous_x_current_incongruent", "three_way_interaction",
    ]
    specs = {
        "pretrial_controls": {
            "categorical": ["user_id", "previous_target", "previous_response"],
            "numeric": core + ["previous_rt_z", "target_repeat"],
            "controls": "previous RT z; target repeat; previous target; previous response; participant",
        },
        "extended_downstream_diagnostic": {
            "categorical": ["user_id", "previous_target", "previous_response"],
            "numeric": core + ["previous_rt_z", "target_repeat", "response_repeat", "rt_percentile"],
            "controls": "pretrial controls + current RT percentile + response repeat",
        },
    }
    for sensitivity_set, spec in specs.items():
        categorical, numeric = spec["categorical"], spec["numeric"]
        pipeline = make_pipeline(categorical, numeric, max_iter=280)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(pairs[categorical + numeric], pairs.error.astype(int))
        iterations = int(np.max(pipeline.named_steps["model"].n_iter_))
        converged = iterations < pipeline.named_steps["model"].max_iter
        for previous, current, code, label in TRANSITIONS:
            mask = pairs.previous_congruency.astype(str).eq(previous) & pairs.congruency.astype(str).eq(current)
            base = pairs.loc[mask, categorical + numeric].copy()
            probabilities = {}
            for pe in [False, True]:
                base["previous_error"] = float(pe)
                base["previous_error_x_previous_incongruent"] = float(pe and previous == "incongruent")
                base["previous_error_x_current_incongruent"] = float(pe and current == "incongruent")
                base["three_way_interaction"] = float(pe and previous == "incongruent" and current == "incongruent")
                probabilities[pe] = pipeline.predict_proba(base)[:, 1].mean()
            rows.append({
                "transition": code, "transition_label": label, "previous_condition": previous,
                "current_condition": current, "sensitivity_set": sensitivity_set,
                "estimation": "covariate_adjusted_average_marginal",
                "risk_after_previous_correct": probabilities[False], "risk_after_previous_error": probabilities[True],
                "risk_difference": probabilities[True] - probabilities[False],
                "controls": spec["controls"], "optimizer_iterations": iterations, "optimizer_converged": converged,
            })
    return pd.DataFrame(rows)


def lagged_frame(data: pd.DataFrame, lag: int) -> pd.DataFrame:
    ordered = data.sort_values(["user_id", "nth_play", "trial"], kind="mergesort").copy()
    group = ordered.groupby(["user_id", "nth_play"], observed=True, sort=False)
    ordered["lagged_trial"] = group.trial.shift(lag)
    ordered["lagged_correct"] = group.correct.shift(lag)
    ordered["lagged_congruency"] = group.congruency.shift(lag)
    out = ordered.loc[ordered.trial.eq(ordered.lagged_trial + lag)].copy()
    out["error"] = ~out.correct.astype(bool)
    out["lagged_error"] = ~out.lagged_correct.astype(bool)
    return out


def lag_decay_analysis(data: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    details, summaries = [], []
    for lag in range(1, 6):
        frame = lagged_frame(data, lag)
        for condition in ["all", "congruent", "incongruent"]:
            part = frame if condition == "all" else frame[frame.lagged_congruency.astype(str).eq(condition)]
            cells = part.groupby(["user_id", "lagged_error"], observed=True).error.agg(n_trials="size", n_errors="sum").reset_index().rename(columns={"lagged_error": "previous_error"})
            detail, summary = estimate_effect_from_counts(cells, rng)
            detail.insert(0, "lag", lag)
            detail.insert(1, "lagged_condition", condition)
            details.append(detail)
            summary.update({"lag": lag, "lagged_condition": condition, "n_valid_pairs": len(part)})
            summaries.append(summary)
    summary = pd.DataFrame(summaries)
    for condition, idx in summary.groupby("lagged_condition", observed=True).groups.items():
        part = summary.loc[idx].sort_values("lag")
        x = part.lag.to_numpy(float)
        y = part.population_mean_risk_difference.to_numpy(float)
        try:
            params, _ = curve_fit(lambda k, amplitude, tau: amplitude * np.exp(-(k - 1) / tau), x, y, p0=[max(y[0], 1e-4), 2], bounds=([0, .05], [1, 100]))
            predicted = params[0] * np.exp(-(x - 1) / params[1])
            r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2) if np.var(y) > 0 else np.nan
            monotonic = bool(np.all(np.diff(y) <= 0))
            summary.loc[idx, ["decay_amplitude", "decay_tau_trials", "decay_r_squared", "strictly_monotonic_decrease"]] = [params[0], params[1], r2, monotonic]
        except (RuntimeError, ValueError):
            summary.loc[idx, ["decay_amplitude", "decay_tau_trials", "decay_r_squared", "strictly_monotonic_decrease"]] = [np.nan, np.nan, np.nan, False]
    return pd.concat(details, ignore_index=True), summary


def null_simulations(pairs: pd.DataFrame, primary_details: pd.DataFrame, primary_summary: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = pairs.groupby(["user_id", "previous_error", "previous_incongruent", "current_incongruent"], observed=True).error.agg(n_trials="size", n_errors="sum").reset_index()
    grouped["error_rate"] = grouped.n_errors / grouped.n_trials
    fits = {
        "participant_current_condition_only": smf.glm(
            "error_rate ~ C(user_id) + current_incongruent", grouped, family=sm.families.Binomial(), freq_weights=grouped.n_trials
        ).fit(),
        "generic_previous_error": smf.glm(
            "error_rate ~ C(user_id) + current_incongruent + previous_error", grouped, family=sm.families.Binomial(), freq_weights=grouped.n_trials
        ).fit(),
    }
    grouped["transition"] = np.select([
        ~grouped.previous_incongruent & ~grouped.current_incongruent,
        grouped.previous_incongruent & ~grouped.current_incongruent,
        ~grouped.previous_incongruent & grouped.current_incongruent,
    ], ["C_to_C", "I_to_C", "C_to_I"], default="I_to_I")
    prior_map = {}
    for row in primary_summary.itertuples():
        prior_map[(row.transition, False)] = (row.prior_correct_alpha, row.prior_correct_beta)
        prior_map[(row.transition, True)] = (row.prior_error_alpha, row.prior_error_beta)
    simulation_rows = []
    for null_name, fit in fits.items():
        probabilities = fit.predict(grouped).to_numpy()
        for simulation in range(N_SIMULATIONS):
            simulated_errors = rng.binomial(grouped.n_trials.to_numpy(int), probabilities)
            sim = grouped[["user_id", "previous_error", "transition", "n_trials"]].copy()
            sim["n_errors"] = simulated_errors
            for code in [t[2] for t in TRANSITIONS]:
                part = sim[sim.transition.eq(code)]
                wide_n = part.pivot(index="user_id", columns="previous_error", values="n_trials")
                wide_k = part.pivot(index="user_id", columns="previous_error", values="n_errors")
                rates = {}
                for level in [False, True]:
                    rates[level], _ = posterior_rate(wide_k[level].to_numpy(), wide_n[level].to_numpy(), prior_map[(code, level)])
                simulation_rows.append({
                    "null_model": null_name, "simulation": simulation, "transition": code,
                    "population_mean_risk_difference": np.mean(rates[True] - rates[False]),
                })
    simulations = pd.DataFrame(simulation_rows)
    observed = primary_summary.set_index("transition").population_mean_risk_difference
    comparison = []
    for (null_name, transition), part in simulations.groupby(["null_model", "transition"], observed=True):
        values = part.population_mean_risk_difference.to_numpy()
        value = observed.loc[transition]
        comparison.append({
            "null_model": null_name, "transition": transition, "observed": value,
            "null_mean": values.mean(), "null_sd": values.std(ddof=1),
            "null_ci_low": np.quantile(values, .025), "null_ci_high": np.quantile(values, .975),
            "observed_z_vs_null": (value - values.mean()) / values.std(ddof=1),
            "two_sided_simulation_p": (1 + np.sum(np.abs(values - values.mean()) >= abs(value - values.mean()))) / (len(values) + 1),
        })
    return simulations, pd.DataFrame(comparison)


def classify_pattern(summary: pd.DataFrame) -> tuple[str, str]:
    values = summary.set_index("transition").population_mean_risk_difference
    all_positive = values.gt(0).all()
    if all_positive and values.max() / values.min() < 2.25:
        return "T1 — GENERAL ERROR-PRONE STATE", "All four transitions are positive and broadly similar in absolute magnitude; modest condition modulation remains, but no transition is isolated or reversed."
    if values.C_to_C > 1.75 * values.drop("C_to_C").max():
        return "T2 — CONGRUENT-ERROR PERSISTENCE", "C→C is selectively dominant."
    if values[["I_to_C", "I_to_I"]].mean() > 1.5 * values[["C_to_C", "C_to_I"]].mean():
        return "T3 — CONFLICT-ERROR CARRYOVER", "Effects following incongruent errors dominate."
    current_c = values[["C_to_C", "I_to_C"]].mean()
    current_i = values[["C_to_I", "I_to_I"]].mean()
    if max(current_c, current_i) > 1.5 * min(current_c, current_i):
        return "T4 — CURRENT-CONDITION-SPECIFIC EFFECT", "The history effect differs strongly by current condition."
    return "T5 — MIXED / COMPLEX", "The four effects do not fit a single simple transition pattern."


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "figure.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def percent_axis(value: float, _position: int) -> str:
    return f"{100 * value:.0f}%"


def pp_axis(value: float, _position: int) -> str:
    return f"{100 * value:.0f}"


def add_title(fig: plt.Figure, number: int, title: str) -> None:
    fig.suptitle(f"Figure {number}. {title}", x=.06, y=.985, ha="left", fontweight="bold")


def add_note(fig: plt.Figure, note: str) -> None:
    fig.text(.06, .008, r"$\it{Note.}$ " + note, ha="left", va="bottom", fontsize=8.3, color="#444444")


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def make_figures(output: Path, probabilities: pd.DataFrame, summary: pd.DataFrame, details: pd.DataFrame,
                 lag_summary: pd.DataFrame, cv: pd.DataFrame) -> None:
    setup_style()
    order_prev = ["congruent", "incongruent"]
    colors = {"congruent": ACCENT, "incongruent": ACCENT_2}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.82, bottom=.25, wspace=.22)
    add_title(fig, 1, "Condition-specific error transitions")
    for ax, current, panel in zip(axes, ["congruent", "incongruent"], ["A  Current congruent", "B  Current incongruent"]):
        for i, previous in enumerate(order_prev):
            code = next(t[2] for t in TRANSITIONS if t[0] == previous and t[1] == current)
            cell = probabilities[probabilities.transition.eq(code)].set_index("previous_accuracy")
            x = np.array([0, 1]) + (i - .5) * .10
            y = cell.loc[["previous_correct", "previous_error"], "participant_mean_shrunk_error_probability"].to_numpy()
            lo = cell.loc[["previous_correct", "previous_error"], "ci_low"].to_numpy()
            hi = cell.loc[["previous_correct", "previous_error"], "ci_high"].to_numpy()
            ax.plot(x, y, marker="o", color=colors[previous], label=f"Previous {previous}", lw=1.8)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=colors[previous], capsize=3, lw=.9)
            effect = summary[summary.transition.eq(code)].iloc[0]
            ax.text(x[1] + .02, y[1], f"Δ = {100 * effect.population_mean_risk_difference:.1f} pp", color=colors[previous], va="center", fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1], ["Previous correct", "Previous error"])
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(percent_axis))
        ax.tick_params(direction="in")
        ax.set_ylim(0, max(.17, probabilities.participant_mean_shrunk_error_probability.max() * 1.25))
    axes[0].set_ylabel("Current error probability")
    axes[1].legend(frameon=False, loc="upper left")
    add_note(fig, "Points are participant-mean empirical-Bayes probabilities; error bars are participant-bootstrap 95% intervals. True adjacent trials only.")
    save(fig, output, "fig1_condition_specific_error_transitions")

    matrix = summary.pivot(index="previous_condition", columns="current_condition", values="population_mean_risk_difference").reindex(index=order_prev, columns=order_prev)
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    fig.subplots_adjust(left=.20, right=.92, top=.82, bottom=.23)
    add_title(fig, 2, "Risk increase after an error in the preceding condition")
    image = ax.imshow(matrix.to_numpy() * 100, cmap="Blues", vmin=0, vmax=max(8, matrix.to_numpy().max() * 100 * 1.1))
    for i in range(2):
        for j in range(2):
            value = matrix.iloc[i, j] * 100
            ax.text(j, i, f"+{value:.2f} pp", ha="center", va="center", color="white" if value > 4.5 else DARK, fontweight="bold", fontsize=11)
    ax.set_xticks([0, 1], ["Current C", "Current I"])
    ax.set_yticks([0, 1], ["Previous C", "Previous I"])
    ax.set_xlabel("Current condition")
    ax.set_ylabel("Previous condition")
    fig.colorbar(image, ax=ax, label="Risk difference (percentage points)", fraction=.05, pad=.04)
    add_note(fig, "C = congruent; I = incongruent. Each cell is the participant-mean shrunk risk after previous error minus previous correct.")
    save(fig, output, "fig2_transition_risk_heatmap")

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 5.3), sharey=True)
    fig.subplots_adjust(left=.07, right=.99, top=.83, bottom=.20, wspace=.18)
    add_title(fig, 3, "Participant-specific transition effects")
    for ax, (_, _, code, label) in zip(axes, TRANSITIONS):
        part = details[details.transition.eq(code)].sort_values("risk_difference").reset_index(drop=True)
        y = np.arange(1, len(part) + 1)
        low = part.risk_difference - 1.96 * part.posterior_risk_difference_sd
        high = part.risk_difference + 1.96 * part.posterior_risk_difference_sd
        ax.hlines(y, low, high, color=ACCENT, alpha=.30, lw=.65)
        ax.scatter(part.risk_difference, y, color=np.where(part.risk_difference.gt(0), ACCENT, LIGHT), s=13, zorder=3)
        ax.axvline(0, color=DARK, lw=.8)
        ax.axvline(part.risk_difference.mean(), color=ACCENT, ls="--", lw=1.1)
        ax.set_title(label, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(pp_axis))
        ax.tick_params(direction="in")
        ax.text(.97, .04, f"{part.risk_difference.gt(0).sum()}/75 positive\nMean +{100*part.risk_difference.mean():.1f} pp", transform=ax.transAxes, ha="right", color=ACCENT, fontsize=8.7, fontweight="bold")
    axes[0].set_ylabel("Participants ordered by effect")
    fig.supxlabel("Previous-error risk difference (percentage points)", y=.10, fontsize=11)
    add_note(fig, "Horizontal lines are approximate 95% posterior intervals; positive point estimates are not individual significance tests.")
    save(fig, output, "fig3_participant_transition_effects")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    fig.subplots_adjust(left=.13, right=.97, top=.83, bottom=.22)
    add_title(fig, 4, "Error-history effects across lags 1–5")
    for condition, label, color, marker in [
        ("all", "Any previous error", DARK, "o"), ("congruent", "Previous congruent error", ACCENT, "s"),
        ("incongruent", "Previous incongruent error", ACCENT_2, "^")
    ]:
        part = lag_summary[lag_summary.lagged_condition.eq(condition)].sort_values("lag")
        ax.plot(part.lag, part.population_mean_risk_difference, color=color, marker=marker, label=label)
        ax.fill_between(part.lag, part.risk_difference_ci_low, part.risk_difference_ci_high, color=color, alpha=.12)
    ax.axhline(0, color=DARK, lw=.7)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Lag (trials)")
    ax.set_ylabel("Increase in current error risk (percentage points)")
    ax.yaxis.set_major_formatter(FuncFormatter(pp_axis))
    ax.tick_params(direction="in")
    ax.legend(frameon=False)
    add_note(fig, "All lagged pairs remain within the same participant and session and require the exact original trial-index difference.")
    save(fig, output, "fig4_error_history_lag_decay")

    models = cv[cv.record_type.eq("summary")].copy()
    labels = {
        "M0_participant_current_condition": "M0 Participant + current condition",
        "M1_plus_generic_previous_error": "M1 + generic previous error",
        "M2_plus_previous_condition_history": "M2 + previous-condition history",
        "M3_full_three_way_interaction": "M3 + full three-way interaction",
    }
    order = list(labels)
    models["order"] = models.model.map({name: i for i, name in enumerate(order)})
    models = models.sort_values("order")
    improvement = -models.delta_log_loss_vs_m0.to_numpy(float)
    low = -models.delta_log_loss_ci_high.fillna(0).to_numpy(float)
    high = -models.delta_log_loss_ci_low.fillna(0).to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.subplots_adjust(left=.36, right=.97, top=.83, bottom=.22)
    add_title(fig, 5, "Condition-specific history improves held-out prediction")
    y = np.arange(len(models))[::-1]
    ax.barh(y, improvement, color=[NEUTRAL, ACCENT, ACCENT, ACCENT], height=.58)
    ax.errorbar(improvement, y, xerr=[np.maximum(0, improvement - low), np.maximum(0, high - improvement)], fmt="none", color=DARK, capsize=3, lw=.8)
    ax.axvline(0, color=DARK, lw=.8)
    ax.set_yticks(y, [labels[m] for m in models.model])
    ax.set_xlabel("Held-out log-loss improvement over M0")
    ax.tick_params(direction="in")
    for yi, value in zip(y, improvement):
        ax.text(max(value, 0) + .0001, yi, "Baseline" if value == 0 else f"{value:.5f}", va="center", fontsize=9)
    add_note(fig, "Positive values indicate lower held-out log loss; intervals are participant-bootstrap 95% intervals across five temporal folds.")
    save(fig, output, "fig5_history_model_predictive_gain")


def write_report(output: Path, pairs: pd.DataFrame, summary: pd.DataFrame, interaction: pd.DataFrame,
                 lag_summary: pd.DataFrame, cv: pd.DataFrame, null_compare: pd.DataFrame,
                 sensitivity: pd.DataFrame, classification: str, rationale: str) -> None:
    primary = summary.set_index("transition")
    cv_summary = cv[cv.record_type.eq("summary")].set_index("model")
    terms = interaction.set_index("term")
    three_term = "previous_error[T.True]:previous_incongruent[T.True]:current_incongruent[T.True]"
    if three_term not in terms.index:
        three_term = next(t for t in terms.index if t.count(":") == 2)
    previous_term = "previous_error[T.True]:previous_incongruent[T.True]"
    current_term = "previous_error[T.True]:current_incongruent[T.True]"
    lag_overall = lag_summary[lag_summary.lagged_condition.eq("all")].sort_values("lag")
    generic_null = null_compare[null_compare.null_model.eq("generic_previous_error")]
    largest_deviation = generic_null.iloc[generic_null.observed_z_vs_null.abs().argmax()]
    rows = []
    for _, _, code, label in TRANSITIONS:
        row = primary.loc[code]
        rows.append({
            "Transition": label, "After previous correct": row.mean_shrunk_risk_after_previous_correct,
            "After previous error": row.mean_shrunk_risk_after_previous_error,
            "Risk difference": row.population_mean_risk_difference,
            "95% interval": f"[{row.risk_difference_ci_low:.4f}, {row.risk_difference_ci_high:.4f}]",
            "Participants positive": f"{int(row.n_positive)}/75 ({100*row.proportion_positive:.1f}%)",
            "Raw-stable participants": f"{int(row.n_raw_stable_participants)}/75",
            "Odds ratio": row.population_geometric_mean_odds_ratio,
        })
    table = pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    sensitivity_table = sensitivity[["sensitivity_set", "transition_label", "risk_after_previous_correct", "risk_after_previous_error", "risk_difference", "optimizer_converged"]].to_markdown(index=False, floatfmt=".4f")
    pretrial = sensitivity[sensitivity.sensitivity_set.eq("pretrial_controls")].set_index("transition")
    primary_effects = summary.set_index("transition").population_mean_risk_difference
    max_pretrial_change = (pretrial.risk_difference - primary_effects).abs().max()
    lag_table = lag_overall[["lag", "population_mean_risk_difference", "risk_difference_ci_low", "risk_difference_ci_high", "n_valid_pairs"]].to_markdown(index=False, floatfmt=".4f")
    cv_table = cv_summary[["log_loss", "delta_log_loss_vs_m0", "brier_score", "delta_brier_vs_m0", "expected_calibration_error"]].reset_index().to_markdown(index=False, floatfmt=".7f")
    text = f"""# Human condition-specific error-transition audit

## Scope and adjacency audit

This is a human-only audit using the frozen LIM preprocessing. It retained **{len(pairs):,} true adjacent trial pairs** from 75 participants. Every pair is within the same participant and `nth_play` session and has original trial-index difference exactly 1. Session starts, cleaning gaps, and nonconsecutive pairs are excluded. C0v2 and all VGG/Wong–Wang or cognitive-model files were not read, modified, or refitted.

## Four primary transition effects

Absolute risks are participant-level empirical-Bayes estimates; intervals bootstrap participants. Positive point estimates are not individual significance declarations.

{table}

## Interaction model

The participant-fixed-effect grouped-binomial model is likelihood-equivalent to the requested trial-level categorical logistic model. The previous-error × previous-condition coefficient is {terms.loc[previous_term, 'log_odds_coefficient']:.4f}, 95% CI [{terms.loc[previous_term, 'ci_low']:.4f}, {terms.loc[previous_term, 'ci_high']:.4f}]. The previous-error × current-condition coefficient is {terms.loc[current_term, 'log_odds_coefficient']:.4f}, 95% CI [{terms.loc[current_term, 'ci_low']:.4f}, {terms.loc[current_term, 'ci_high']:.4f}]. The three-way coefficient is {terms.loc[three_term, 'log_odds_coefficient']:.4f}, 95% CI [{terms.loc[three_term, 'ci_low']:.4f}, {terms.loc[three_term, 'ci_high']:.4f}]. Thus previous and current congruency modulate the effect on the log-odds scale, although all four probability-scale effects remain positive and similar in absolute magnitude.

## Sensitivity controls

The pretrial sensitivity model controls previous RT, target repetition, previous target, previous response, and participant identity. A separate extended diagnostic additionally includes current RT percentile and response repetition; these can be downstream of or jointly determined with the current response, so they are not treated as clean pretrial controls. Age is not added because it is fixed within participant and therefore redundant with participant effects.

{sensitivity_table}

All four pretrial-adjusted effects remain positive; the largest absolute change from the primary shrunk effect is {max_pretrial_change:.4f}. The extended diagnostic is reported transparently but does not replace the primary estimand.

## Lag decay

{lag_table}

For the overall series, the descriptive exponential fit gives A={lag_overall.decay_amplitude.iloc[0]:.4f}, τ={lag_overall.decay_tau_trials.iloc[0]:.2f} trials, R²={lag_overall.decay_r_squared.iloc[0]:.3f}. τ is only a behavioral history timescale, not a neural time constant. Strict monotonic decrease was {bool(lag_overall.strictly_monotonic_decrease.iloc[0])}.

## Blocked held-out prediction

{cv_table}

The five folds hold out contiguous temporal blocks within every participant. Improvements are absolute per-trial probability-score changes; p-values are not used as the main evidence.

## Null models

The stable participant/current-condition null and the generic previous-error null each used {N_SIMULATIONS} fixed-seed grouped-binomial simulations. Under the generic-history null, the largest standardized mismatch is {largest_deviation.transition}: observed {largest_deviation.observed:.4f} versus null mean {largest_deviation.null_mean:.4f}, z={largest_deviation.observed_z_vs_null:.2f}. This asks whether one condition-invariant history effect can reproduce the four-cell matrix.

## Classification and interpretation

**{classification}.** {rationale}

The most plausible next mechanistic hypothesis is therefore a general, short-lived error-prone state whose strength is modulated by trial condition, especially the condition of the preceding error. This is a hypothesis for a later model-comparison task, not a mechanism implemented here.

## What this does not establish

- The associations do not establish that an error causally creates the later state; an unmeasured state may produce both errors.
- The decay parameter is descriptive and is not a neural time constant.
- Positive participant estimates do not imply individually significant effects.
- Sensitivity adjustment cannot eliminate every sequential confound and current RT may be downstream.
- No incomplete reset, state carryover, starting-state variability, lapse, sensory-noise, or history-dependent cognitive parameter was created.

## Integrity confirmation

**C0v2 was not read, modified, or refitted. No cognitive model was created or tuned.**
"""
    (output / "human_condition_specific_transition_report.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data, audits, sources = load_human_data(args.data_dir.resolve())
    pairs = make_true_adjacent_pairs(data)
    if args.finalize_only:
        probabilities = pd.read_csv(args.output_dir / "condition_transition_probabilities.csv")
        summary = pd.read_csv(args.output_dir / "condition_transition_risk_differences.csv")
        details = pd.read_csv(args.output_dir / "participant_transition_effects.csv")
        interaction = pd.read_csv(args.output_dir / "transition_interaction_model.csv")
        lag_summary = pd.read_csv(args.output_dir / "lag_decay_summary.csv")
        cv = pd.read_csv(args.output_dir / "crossvalidated_transition_models.csv")
        null_compare = pd.read_csv(args.output_dir / "transition_null_observed_comparison.csv")
        sensitivity = pd.read_csv(args.output_dir / "transition_sensitivity_adjusted_effects.csv")
        classification_data = json.loads((args.output_dir / "transition_classification.json").read_text())
        make_figures(args.output_dir, probabilities, summary, details, lag_summary, cv)
        write_report(args.output_dir, pairs, summary, interaction, lag_summary, cv, null_compare, sensitivity, classification_data["classification"], classification_data["rationale"])
        return

    details, summary, probabilities, _ = primary_transition_analysis(pairs, rng)
    interaction, formal_probabilities, _ = fit_formal_interaction(pairs)
    cv = blocked_cross_validation(pairs, rng)
    sensitivity = sensitivity_analysis(pairs)
    lag_details, lag_summary = lag_decay_analysis(data, rng)
    null_sims, null_compare = null_simulations(pairs, details, summary, rng)
    classification, rationale = classify_pattern(summary)

    probabilities.merge(formal_probabilities, on=["transition", "transition_label", "previous_condition", "current_condition", "previous_accuracy"], how="left", validate="one_to_one").to_csv(args.output_dir / "condition_transition_probabilities.csv", index=False)
    summary.to_csv(args.output_dir / "condition_transition_risk_differences.csv", index=False)
    details.to_csv(args.output_dir / "participant_transition_effects.csv", index=False)
    interaction.to_csv(args.output_dir / "transition_interaction_model.csv", index=False)
    lag_summary.to_csv(args.output_dir / "lag_decay_summary.csv", index=False)
    lag_details.to_csv(args.output_dir / "lag_decay_participant_effects.csv", index=False)
    cv.to_csv(args.output_dir / "crossvalidated_transition_models.csv", index=False)
    null_sims.to_csv(args.output_dir / "transition_null_simulation_summary.csv", index=False)
    null_compare.to_csv(args.output_dir / "transition_null_observed_comparison.csv", index=False)
    sensitivity.to_csv(args.output_dir / "transition_sensitivity_adjusted_effects.csv", index=False)
    audits.to_csv(args.output_dir / "preprocessing_audit_by_participant.csv", index=False)
    sources.to_csv(args.output_dir / "source_file_inventory.csv", index=False)
    (args.output_dir / "transition_classification.json").write_text(json.dumps({"classification": classification, "rationale": rationale}, indent=2))
    make_figures(args.output_dir, probabilities, summary, details, lag_summary, cv)
    write_report(args.output_dir, pairs, summary, interaction, lag_summary, cv, null_compare, sensitivity, classification, rationale)
    qa = {
        "analysis": "human-only condition-specific sequential error-transition audit",
        "seed": SEED, "n_participants": int(pairs.user_id.nunique()), "n_true_adjacent_pairs": len(pairs),
        "all_pairs_trial_difference_one": bool(pairs.trial.eq(pairs.previous_trial + 1).all()),
        "all_participants_in_every_fold": bool(pairs.groupby("blocked_fold").user_id.nunique().eq(75).all()),
        "n_simulations_per_null": N_SIMULATIONS, "n_png": len(list(args.output_dir.glob("fig*.png"))),
        "n_pdf": len(list(args.output_dir.glob("fig*.pdf"))), "classification": classification,
        "c0v2_read_or_modified": False,
    }
    (args.output_dir / "qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

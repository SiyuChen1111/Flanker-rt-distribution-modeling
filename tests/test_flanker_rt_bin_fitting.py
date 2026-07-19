from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_flanker_rt_bin_fitting import (  # noqa: E402
    CandidateSpec,
    aggregate_equal_count_profiles,
    assign_rt_bins,
    assign_equal_count_bins,
    compute_bin_edges,
    default_candidate_specs,
    equal_count_subject_profiles,
    make_join_columns,
    make_subject_folds,
    load_human_master,
    normalize_candidate,
    profile_frame,
    seed_metrics,
    shape_seed_metrics,
)


def test_edges_and_edge_assignment_are_shared_and_right_closed() -> None:
    rt = np.arange(1.0, 9.0)
    edges = compute_bin_edges(rt)
    np.testing.assert_allclose(edges, [2.75, 4.5, 6.25])
    assigned = assign_rt_bins([1.0, 2.75, 3.0, 4.5, 5.0, 6.25, 7.0], edges)
    assert assigned.tolist() == [1, 1, 2, 2, 3, 3, 4]


def test_profile_uses_condition_denominator_and_keeps_empty_cells() -> None:
    rows = []
    for group in ("young_20_29", "older_80_89"):
        for congruency in (0, 1):
            for rt, correct in zip([0.1, 0.2, 0.3, 0.4], [True, False, True, True]):
                rows.append(
                    {
                        "analysis_group": group,
                        "congruency": congruency,
                        "rt": rt,
                        "correct": correct,
                    }
                )
    profile = profile_frame(pd.DataFrame(rows), "rt", "correct", [0.15, 0.25, 0.35])
    assert len(profile) == 16
    assert np.allclose(profile["bin_proportion"], 0.25)
    young_congruent_bin2 = profile.query(
        "analysis_group == 'young_20_29' and congruency == 0 and rt_bin == 2"
    ).iloc[0]
    assert young_congruent_bin2.error_rate == 1.0


def test_four_folds_hold_out_one_old_and_three_young_without_overlap() -> None:
    rows = []
    for group, n_subjects in (("young_20_29", 12), ("older_80_89", 4)):
        for subject in range(n_subjects):
            rows.append({"analysis_group": group, "user_id": f"{group}_{subject}"})
    folds = make_subject_folds(pd.DataFrame(rows), seed=10)
    for fold in range(4):
        test = folds.query("fold == @fold and role == 'test'")
        assert (test.analysis_group == "young_20_29").sum() == 3
        assert (test.analysis_group == "older_80_89").sum() == 1
        train_ids = set(folds.query("fold == @fold and role == 'train'").user_id)
        assert train_ids.isdisjoint(set(test.user_id))


def test_candidate_join_distinguishes_repeated_raw_trial_ids(tmp_path: Path) -> None:
    human_raw = pd.DataFrame(
        {
            "row_index": [40, 40],
            "analysis_group": ["older_80_89", "older_80_89"],
            "user_id": [1, 2],
            "true_rt": [0.51, 0.72],
            "human_correct": [True, False],
            "congruency": [0, 1],
            "target_label": [1, 2],
            "flanker_label": [1, 3],
        }
    )
    human = make_join_columns(human_raw, "row_index")
    human["trial_uid"] = [0, 1]
    candidate = pd.DataFrame(
        {
            "trial_id": [40, 40],
            "analysis_group": ["older_80_89", "older_80_89"],
            "true_rt": [0.72, 0.51],
            "human_correct": [False, True],
            "congruency": [1, 0],
            "target_label": [2, 1],
            "flanker_label": [3, 1],
            "model": ["m", "m"],
            "model_rt": [0.7, 0.5],
            "model_correct": [False, True],
        }
    )
    path = tmp_path / "candidate.csv"
    candidate.to_csv(path, index=False)
    out = normalize_candidate(
        CandidateSpec("test", path, "model", None, "model_rt", "model_correct"), human
    )
    assert out["trial_uid"].tolist() == [1, 0]
    assert out["user_id"].tolist() == [2, 1]


def test_seed_gate_rejects_perfect_congruent_accuracy_and_small_cells() -> None:
    human_rows = []
    model_rows = []
    for group in ("young_20_29", "older_80_89"):
        for congruency in (0, 1):
            for rt in [0.1, 0.2, 0.3, 0.4]:
                human_rows.append(
                    {
                        "analysis_group": group,
                        "congruency": congruency,
                        "true_rt": rt,
                        "human_correct": not (congruency == 1 and rt == 0.1),
                    }
                )
                model_rows.append(
                    {
                        "analysis_group": group,
                        "congruency": congruency,
                        "model_rt": rt,
                        "model_correct": True,
                    }
                )
    human = pd.DataFrame(human_rows)
    model = pd.DataFrame(model_rows)
    human_profile = profile_frame(human, "true_rt", "human_correct", [0.15, 0.25, 0.35])
    human_q = []
    for (group, congruency), part in human.groupby(["analysis_group", "congruency"]):
        for quantile in (0.25, 0.5, 0.75):
            human_q.append(
                {
                    "analysis_group": group,
                    "congruency": congruency,
                    "quantile": quantile,
                    "rt_quantile": part.true_rt.quantile(quantile),
                }
            )
    tolerances = human_profile[["analysis_group", "congruency", "rt_bin"]].assign(
        error_tolerance=0.1, proportion_tolerance=0.1
    )
    q_tolerances = pd.DataFrame(human_q)[
        ["analysis_group", "congruency", "quantile"]
    ].assign(rt_tolerance=0.1)
    metrics, _ = seed_metrics(
        model,
        human_profile,
        pd.DataFrame(human_q),
        tolerances,
        q_tolerances,
        [0.15, 0.25, 0.35],
        min_cell_trials=5,
    )
    assert not metrics["cell_count_pass"]
    assert not metrics["congruent_nonzero_pass"]
    assert not metrics["seed_behavior_pass"]


def test_repository_baseline_reproduces_known_edges_and_failure_pattern() -> None:
    human = load_human_master()
    edges = compute_bin_edges(human["true_rt"])
    np.testing.assert_allclose(edges, [0.578, 0.709, 0.898], atol=1e-9)
    spec = CandidateSpec(
        "R5_baseline",
        SCRIPT_DIR.parents[1]
        / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
        / "fitting/representative_trial_level_predictions.csv",
        "model_name",
        "seed",
        "pred_rt",
        "model_correct",
        "pred_choice",
    )
    model = normalize_candidate(spec, human)
    hp = profile_frame(human, "true_rt", "human_correct", edges)
    mp = profile_frame(model, "model_rt", "model_correct", edges)
    merged = hp.merge(mp, on=["analysis_group", "congruency", "rt_bin"], suffixes=("_human", "_model"))
    young = merged[merged.analysis_group.eq("young_20_29")]
    inc_rmse = np.sqrt(np.mean((young.query("congruency == 1").error_rate_model - young.query("congruency == 1").error_rate_human) ** 2))
    cong_rmse = np.sqrt(np.mean((young.query("congruency == 0").error_rate_model - young.query("congruency == 0").error_rate_human) ** 2))
    assert inc_rmse > cong_rmse
    old_fast = merged.query("analysis_group == 'older_80_89' and rt_bin == 1")
    assert (old_fast.error_rate_human > old_fast.error_rate_model).all()


def test_equal_count_bins_are_independent_and_balanced_within_subject_condition() -> None:
    frame = pd.DataFrame(
        {
            "analysis_group": ["young_20_29"] * 18,
            "user_id": ["a"] * 9 + ["b"] * 9,
            "congruency": [0] * 5 + [1] * 4 + [0] * 5 + [1] * 4,
            "human_rt": np.arange(18, dtype=float),
            "model_rt": np.arange(18, dtype=float)[::-1],
        }
    )
    human_bins = assign_equal_count_bins(frame, "human_rt")
    model_bins = assign_equal_count_bins(frame, "model_rt")
    assert not human_bins.equals(model_bins)
    work = frame.assign(rt_bin=human_bins)
    for _, part in work.groupby(["user_id", "congruency"]):
        counts = part.rt_bin.value_counts()
        assert counts.max() - counts.min() <= 1


def test_equal_count_aggregation_weights_subjects_not_trials() -> None:
    subject_profiles = pd.DataFrame(
        {
            "analysis_group": ["young_20_29", "young_20_29"],
            "user_id": ["few", "many"],
            "congruency": [1, 1],
            "rt_bin": [1, 1],
            "n_trials": [5, 100],
            "mean_rt": [0.4, 0.5],
            "accuracy": [0.0, 1.0],
            "error_rate": [1.0, 0.0],
        }
    )
    aggregated = aggregate_equal_count_profiles(subject_profiles).iloc[0]
    assert aggregated.error_rate == 0.5
    assert aggregated.n_subjects == 2


def test_shape_can_pass_while_fixed_time_distribution_fails() -> None:
    rows = []
    for group in ("young_20_29", "older_80_89"):
        for congruency in (0, 1):
            for index, rt in enumerate(np.linspace(0.1, 0.8, 8)):
                rows.append(
                    {
                        "analysis_group": group,
                        "user_id": group,
                        "congruency": congruency,
                        "true_rt": rt,
                        "human_correct": index != 0,
                        "model_rt": rt + 10.0,
                        "model_correct": index != 0,
                    }
                )
    frame = pd.DataFrame(rows)
    human_subject = equal_count_subject_profiles(frame, "true_rt", "human_correct")
    model_subject = equal_count_subject_profiles(frame, "model_rt", "model_correct")
    human_profile = aggregate_equal_count_profiles(human_subject)
    shape_tol = human_profile[["analysis_group", "congruency", "rt_bin"]].assign(
        error_tolerance=0.01
    )
    contrast_tol = pd.DataFrame(
        [(group, congruency, 0.01) for group in ("young_20_29", "older_80_89") for congruency in (0, 1)],
        columns=["analysis_group", "congruency", "contrast_tolerance"],
    )
    shape = shape_seed_metrics(
        model_subject, human_profile, shape_tol, contrast_tol, min_cell_trials=1
    )
    fixed_human = profile_frame(frame, "true_rt", "human_correct", [0.25, 0.45, 0.65])
    fixed_q = []
    for (group, congruency), part in frame.groupby(["analysis_group", "congruency"]):
        for quantile in (0.25, 0.5, 0.75):
            fixed_q.append(
                {
                    "analysis_group": group,
                    "congruency": congruency,
                    "quantile": quantile,
                    "rt_quantile": part.true_rt.quantile(quantile),
                }
            )
    fixed_tol = fixed_human[["analysis_group", "congruency", "rt_bin"]].assign(
        error_tolerance=0.1, proportion_tolerance=0.1
    )
    fixed_q_tol = pd.DataFrame(fixed_q)[
        ["analysis_group", "congruency", "quantile"]
    ].assign(rt_tolerance=0.1)
    fixed, _ = seed_metrics(
        frame,
        fixed_human,
        pd.DataFrame(fixed_q),
        fixed_tol,
        fixed_q_tol,
        [0.25, 0.45, 0.65],
        min_cell_trials=1,
    )
    assert shape["seed_shape_pass"]
    assert not fixed["occupancy_pass"]
    assert not fixed["seed_behavior_pass"]


def test_gate_model_equal_count_profile_matches_read_only_reference() -> None:
    human = load_human_master()
    gate_spec = next(spec for spec in default_candidate_specs() if spec.family == "gate_execution")
    gate = normalize_candidate(gate_spec, human)
    human_subject = equal_count_subject_profiles(human, "true_rt", "human_correct")
    human_profile = aggregate_equal_count_profiles(human_subject)
    seed_profiles = []
    for _, seed_part in gate.groupby("seed_id"):
        seed_profiles.append(
            aggregate_equal_count_profiles(
                equal_count_subject_profiles(seed_part, "model_rt", "model_correct")
            ).assign(seed_id=seed_part.seed_id.iloc[0])
        )
    gate_profile = (
        pd.concat(seed_profiles)
        .groupby(["analysis_group", "congruency", "rt_bin"])
        .error_rate.mean()
        .reset_index()
    )
    merged = human_profile.merge(
        gate_profile,
        on=["analysis_group", "congruency", "rt_bin"],
        suffixes=("_human", "_model"),
    )
    older_inc = merged.query("analysis_group == 'older_80_89' and congruency == 1")
    rmse = np.sqrt(np.mean((older_inc.error_rate_model - older_inc.error_rate_human) ** 2))
    assert rmse == pytest.approx(0.025, abs=0.002)
    young_fast = merged.query(
        "analysis_group == 'young_20_29' and congruency == 1 and rt_bin == 1"
    ).iloc[0]
    assert young_fast.error_rate_human == pytest.approx(0.160, abs=0.002)
    assert young_fast.error_rate_model == pytest.approx(0.304, abs=0.002)

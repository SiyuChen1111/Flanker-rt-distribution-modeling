from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/results/all_age_groups_20260806/all_age_model_update_20260807"
AGE_GROUPS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]


def test_shared_scale_and_trial_contract():
    qa = pd.read_json(OUT / "qa.json", typ="series")
    pred = pd.read_csv(OUT / "results/updated_model_trial_level_predictions.csv", low_memory=False)
    assert float(qa["selected_shared_decision_time_scale"]) == 0.27
    assert len(pred) == 35000
    assert pred.groupby("age_group").size().eq(5000).all()
    assert pred["decision_time_scale"].nunique() == 1
    assert pred["decision_time_scale"].iloc[0] == 0.27


def test_choice_crossing_and_condition_alignment():
    pred = pd.read_csv(OUT / "results/updated_model_trial_level_predictions.csv", low_memory=False)
    assert (pred["pred_choice"] == pred["winner_at_readout"]).all()
    assert pred.loc[~pred["crossed"].astype(bool), "pred_rt"].isna().all()
    assert (~pred["crossed"].astype(bool)).sum() == 1
    alignment = pd.read_csv(OUT / "results/condition_rt_alignment_before_after.csv")
    current = alignment[alignment["source"] == "current_model"]["mean_error_vs_human"].abs().mean()
    refined = alignment[alignment["source"] == "refined_model"]["mean_error_vs_human"].abs().mean()
    assert refined < current
    assert refined < 0.01


def test_refined_distribution_and_caf_tables_cover_all_groups():
    summary = pd.read_csv(OUT / "results/updated_model_rt_distribution_summary.csv")
    caf = pd.read_csv(OUT / "results/updated_model_caf.csv")
    assert set(summary["age_group"]) == set(AGE_GROUPS)
    assert summary.groupby("age_group").size().eq(4).all()
    assert caf.groupby(["age_group", "source", "congruency"]).size().eq(5).all()
    assert summary["fraction_within_display"].min() > 0.99


def test_publication_formats_exist():
    stems = [
        "all_age_caf_updated_model",
        "all_age_rt_distribution_updated_model",
        "shared_timing_calibration_selection",
        "condition_rt_alignment_before_after",
    ]
    for stem in stems:
        for extension in ["png", "pdf", "svg", "tiff"]:
            assert (OUT / "figures_publication" / f"{stem}.{extension}").exists()

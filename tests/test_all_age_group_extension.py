from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/results/all_age_groups_20260806"

def test_age_groups_and_subset_counts():
    inventory = pd.read_csv(OUT / "audits/age_group_inventory.csv")
    assert inventory["age_group"].tolist() == ["20-29","30-39","40-49","50-59","60-69","70-79","80-89"]
    subset = pd.read_csv(OUT / "manifests/all_age_group_trial_manifest.csv")
    assert subset.groupby("age_group").size().eq(5000).all()
    assert subset["trial_id"].is_unique
    assert subset.groupby("age_group")["subject_id"].nunique().to_dict() == dict(zip(inventory.age_group, inventory.n_subjects))

def test_corrected_choice_and_censoring_contract():
    pred = pd.read_csv(OUT / "results/all_age_group_trial_level_predictions.csv", low_memory=False)
    assert (pred["pred_choice"] == pred["winner_at_readout"]).all()
    assert pred.loc[~pred["crossed"].astype(bool), "pred_rt"].isna().all()
    assert pred.groupby("age_group").size().eq(5000).all()
    assert len(pred) == 35000
    assert not pred.duplicated(["age_group", "user_id", "row_index"]).any()
    assert (~pred["crossed"].astype(bool)).sum() == 1

def test_caf_uses_real_rt_coordinates_and_delta_is_participant_first():
    caf = pd.read_csv(OUT / "results/all_age_group_caf.csv")
    assert caf["median_rt"].notna().all()
    assert caf.groupby(["age_group","source","congruency"]).size().eq(5).all()
    delta = pd.read_csv(OUT / "results/all_age_group_subject_delta.csv")
    assert {"user_id","rt_bin","delta_rt"}.issubset(delta.columns)
    assert delta.groupby(["age_group","source","user_id","rt_bin"]).size().le(1).all()

def test_run_status_is_evidence_based():
    status = pd.read_csv(OUT / "audits/age_group_run_status.csv").set_index("age_group")
    assert status["run_status"].eq("completed_with_corrected_equivalent_model").all()
    assert status["existing_evidence_cache"].all()
    assert status["existing_trial_predictions"].all()
    assert status["n_existing_predictions"].eq(5000).all()


def test_middle_age_vgg_caches_are_complete_and_finite():
    for age_group in ["30-39", "40-49", "50-59", "60-69", "70-79"]:
        path = OUT / "evidence_cache" / age_group / "full_age_group_layerwise_evidence.npz"
        with np.load(path, allow_pickle=True) as cache:
            assert len(cache["evidence_available"]) == 5000
            assert cache["evidence_available"].astype(bool).all()
            for layer in ["conv3", "conv4", "conv5", "pooled", "final"]:
                values = cache[f"evidence_{layer}"]
                assert values.shape == (5000, 4)
                assert np.isfinite(values).all()
        readiness = json.loads(
            (OUT / "evidence_cache" / age_group / "age_group_fitting_readiness.json").read_text()
        )
        assert readiness["can_run_age_group_restricted_fitting"]
        assert "blocking_reasons" not in readiness


def test_all_age_fit_and_crossing_tables_cover_seven_groups():
    scores = pd.read_csv(OUT / "results/all_age_group_model_fit_scores.csv")
    crossing = pd.read_csv(OUT / "results/all_age_group_crossing_audit.csv")
    assert set(scores["age_group"]) == {"20-29","30-39","40-49","50-59","60-69","70-79","80-89"}
    assert scores["crossing_gate_passed"].all()
    assert crossing["winner_readout_consistency"].eq(1.0).all()
    assert crossing["n_no_crossing"].sum() == 1

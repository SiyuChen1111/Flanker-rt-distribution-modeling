#!/usr/bin/env python3
"""Audit age groups, trial validity, existing model artifacts, and model identity."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT

OUT_DEFAULT = PROJECT_ROOT / "artifacts/results/all_age_groups_20260806"
META = PROJECT_ROOT / "data/vam_data/metadata.csv"
HUMAN_FILES = sorted(PROJECT_ROOT.glob("data/vam_data/user*df.csv"))
REQUIRED_LABELS = {"L", "R", "U", "D"}
MODEL_ROOT = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
LEGACY_ROOT = MODEL_ROOT / "best_model_R5_combined_best"
COUPLED_ROOT = PROJECT_ROOT / "artifacts/results/r5_choice_coupled_schedule_optimization_20260803"
MECHANISM_ROOT = PROJECT_ROOT / "artifacts/results/r5_real_vgg_target_flanker_audit_20260803"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_DEFAULT))
    return p.parse_args()


def read_trials() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(META, dtype={"user_id": str})
    meta["user_id"] = meta["user_id"].astype(str)
    frames: list[pd.DataFrame] = []
    for path in HUMAN_FILES:
        d = pd.read_csv(path)
        if d.empty:
            continue
        d["subject_id"] = d["anon_id"].astype(str)
        d["source_file"] = str(path)
        d["source_row_index"] = np.arange(len(d), dtype=np.int64)
        d = d.merge(meta[["user_id", "binned_age"]], left_on="subject_id", right_on="user_id", how="left")
        d["age_group"] = d["binned_age"].astype(str)
        d["human_rt"] = pd.to_numeric(d["response_time"], errors="coerce") / 1000.0
        d["human_correct"] = d["response_direction"].astype(str).eq(d["target_direction"].astype(str))
        d["target_valid"] = d["target_direction"].astype(str).isin(REQUIRED_LABELS)
        d["flanker_valid"] = d["flanker_direction"].astype(str).isin(REQUIRED_LABELS)
        d["response_valid"] = d["response_direction"].astype(str).isin(REQUIRED_LABELS)
        d["rt_valid"] = d["human_rt"].between(0.15, 10.0, inclusive="both")
        d["trial_key"] = (
            d["subject_id"].astype(str) + "|" + d["nth_play"].astype(str) + "|" + d["trial"].astype(str)
        )
        d["duplicate_trial"] = d.duplicated("trial_key", keep=False)
        d["valid_trial"] = d[["target_valid", "flanker_valid", "response_valid", "rt_valid"]].all(axis=1)
        d["congruency"] = (d["target_direction"].astype(str) != d["flanker_direction"].astype(str)).astype(int)
        frames.append(d)
    if not frames:
        raise RuntimeError("No user trial files were found.")
    return pd.concat(frames, ignore_index=True), meta


def existing_artifacts(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = MODEL_ROOT / "manifests/representative_subset_trial_manifest.csv"
    legacy_pred = LEGACY_ROOT / "results/best_model_trial_level_predictions.csv"
    coupled_pred = COUPLED_ROOT / "selected_trial_level_predictions.csv"
    selected = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    pred_paths = [p for p in [legacy_pred, coupled_pred] if p.exists()]
    pred = pd.concat([pd.read_csv(p).assign(_prediction_file=str(p)) for p in pred_paths], ignore_index=True) if pred_paths else pd.DataFrame()
    model_rows: list[dict[str, Any]] = []
    for group in sorted(trials["age_group"].dropna().unique()):
        src = selected[selected.get("original_age_group", pd.Series(dtype=str)).astype(str).eq(group)] if not selected.empty else pd.DataFrame()
        n_selected = int(len(src))
        n_pred = 0
        if not pred.empty:
            for col in ["original_age_group", "age_group", "binned_age"]:
                if col in pred.columns:
                    n_pred = int(pred[col].astype(str).eq(group).sum())
                    break
        model_rows.append({
            "age_group": group,
            "existing_subset": bool(n_selected),
            "n_existing_selected": n_selected,
            "existing_trial_predictions": bool(n_pred),
            "n_existing_predictions": n_pred,
            "existing_evidence_cache": bool(n_selected and (MODEL_ROOT / "evidence_cache/representative_subset_layerwise_evidence.npz").exists()),
            "existing_figures": bool(group in {"20-29", "80-89"} and LEGACY_ROOT.exists()),
        })
    return pd.DataFrame(model_rows), pred


def subject_rows(trials: pd.DataFrame, model_info: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_path = MODEL_ROOT / "manifests/representative_subset_trial_manifest.csv"
    selected = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    coupled = pd.read_csv(COUPLED_ROOT / "selected_trial_level_predictions.csv") if (COUPLED_ROOT / "selected_trial_level_predictions.csv").exists() else pd.DataFrame()
    if not coupled.empty:
        coupled["age_group"] = coupled["analysis_group"].map({"young_20_29": "20-29", "older_80_89": "80-89"})
        coupled["user_id"] = coupled["user_id"].astype(str)
    for (age, subject), d in trials.groupby(["age_group", "subject_id"], dropna=False, sort=True):
        valid = d[d["valid_trial"]]
        row = {
            "age_group": age, "subject_id": subject,
            "n_trials_raw": len(d), "n_trials_valid": len(valid),
            "n_trials_model_input": 0, "n_trials_selected_subset": 0,
            "n_congruent": int((valid["congruency"] == 0).sum()),
            "n_incongruent": int((valid["congruency"] == 1).sum()),
            "n_correct": int(valid["human_correct"].sum()),
            "n_error": int((~valid["human_correct"]).sum()),
            "n_congruent_correct": int(((valid["congruency"] == 0) & valid["human_correct"]).sum()),
            "n_congruent_error": int(((valid["congruency"] == 0) & ~valid["human_correct"]).sum()),
            "n_incongruent_correct": int(((valid["congruency"] == 1) & valid["human_correct"]).sum()),
            "n_incongruent_error": int(((valid["congruency"] == 1) & ~valid["human_correct"]).sum()),
            "n_missing_rt": int(d["human_rt"].isna().sum()),
            "n_missing_response": int(d["response_direction"].isna().sum()),
            "n_invalid_label": int((~d[["target_valid", "flanker_valid", "response_valid"]].all(axis=1)).sum()),
            "n_duplicate": int(d["duplicate_trial"].sum()),
            "n_stimulus_unmatched": 0, "n_evidence_cache_missing": 0,
            "n_model_prediction_missing": 0, "n_crossed": 0, "n_no_crossing": 0,
            "included_in_previous_extreme_analysis": age in {"20-29", "80-89"},
            "exclusion_reason": "" if len(valid) else "no_valid_trials",
        }
        # The retained extreme manifest uses a synthetic trial_id; use subject and source row
        # only when a direct source-row mapping is available.
        if not selected.empty and "subject_id" in selected.columns:
            match = selected[selected["subject_id"].astype(str).eq(str(subject))]
            row["n_trials_selected_subset"] = int(len(match))
            row["n_trials_model_input"] = int(len(match))
        if not coupled.empty:
            cp = coupled[coupled["age_group"].eq(str(age)) & coupled["user_id"].eq(str(subject))]
            row["n_model_prediction_missing"] = max(row["n_trials_model_input"] - len(cp), 0)
            row["n_crossed"] = int(cp.get("crossed", pd.Series([True] * len(cp))).astype(bool).sum())
            row["n_no_crossing"] = int(len(cp) - row["n_crossed"])
        rows.append(row)
    return pd.DataFrame(rows)


def fingerprint(out: Path) -> dict[str, Any]:
    param_path = LEGACY_ROOT / "results/best_model_parameter_estimates.csv"
    params = pd.read_csv(param_path).to_dict(orient="records") if param_path.exists() else []
    return {
        "presentation_model_name": "retained R5 representative extreme-age model (historical presentation candidate)",
        "presentation_model_result_dir": str(LEGACY_ROOT),
        "corrected_model_name": "choice-coupled schedule optimization with winner_at_readout",
        "corrected_model_result_dir": str(COUPLED_ROOT),
        "selected_model_for_age_extension": "to_be_decided_after_smoke_test",
        "model_python_file": str(PROJECT_ROOT / "code/scripts/vgg_wongwang_lim.py"),
        "model_class_name": "VGGWongWangLIM",
        "decision_class_name": "DiffDecisionMultiClass",
        "run_script": str(PROJECT_ROOT / "code/scripts/run_representative_extreme_age_subset_fitting.py"),
        "choice_coupled_run_script": str(PROJECT_ROOT / "code/scripts/run_r5_choice_coupled_schedule_optimization.py"),
        "plotting_scripts": [str(PROJECT_ROOT / "code/scripts/make_representative_extreme_age_figures.py"), str(PROJECT_ROOT / "code/scripts/plot_r5_caf_and_delta_curves.py"), str(PROJECT_ROOT / "code/scripts/run_real_vgg_target_flanker_dynamics_audit.py")],
        "source_trial_prediction_file": str(LEGACY_ROOT / "results/best_model_trial_level_predictions.csv"),
        "corrected_trial_prediction_file": str(COUPLED_ROOT / "selected_trial_level_predictions.csv"),
        "selected_config_file": str(param_path),
        "vgg_architecture": "VGG16",
        "vgg_layers_used": ["conv3", "conv4", "conv5", "pooled", "final"],
        "evidence_definition": "four-direction target/flanker layerwise evidence; target-minus-flanker gap",
        "evidence_normalization": "per_layer_gap_scale",
        "layer_to_time_mapping": "natural_smooth_5stage; corrected candidate additionally tests compressed schedule",
        "ww_architecture": "four-channel recurrent competitive Wong-Wang",
        "choice_rule": "legacy trajectory maximum versus corrected winner_at_readout, audited separately",
        "rt_rule": "first sustained crossing plus group non-decision time",
        "crossing_rule": "sustained crossing with threshold and margin",
        "no_crossing_rule": "deadline censoring sentinel with explicit crossing flag",
        "random_seed": 20260530,
        "parameter_estimates": params,
        "mechanism_figure_source": str(MECHANISM_ROOT / "05_natural_emergence_evidence_chain.svg"),
        "model_identity_status": "candidate fingerprint; final identity requires smoke-test evidence",
    }


def main() -> None:
    a = args(); out = Path(a.output_dir)
    for name in ["audits", "configs", "manifests", "logs", "summaries", "results", "figures_publication", "tests"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    trials, meta = read_trials()
    model, pred = existing_artifacts(trials)
    subjects = subject_rows(trials, model)
    valid = trials[trials["valid_trial"]].copy()
    inventory = trials.groupby("age_group", sort=True).agg(
        n_subjects=("subject_id", "nunique"), n_trials_raw=("trial_key", "size")
    ).reset_index()
    valid_inv = valid.groupby("age_group", sort=True).agg(n_trials_valid=("trial_key", "size")).reset_index()
    inventory = inventory.merge(valid_inv, on="age_group", how="left").merge(model, on="age_group", how="left")
    inventory["mean_trials_per_subject"] = inventory["n_trials_raw"] / inventory["n_subjects"]
    stats = trials.groupby(["age_group", "subject_id"], sort=True).size().groupby(level=0).agg(["mean", "median", "std", "min", "max"]).reset_index()
    stats = stats.rename(columns={"mean": "raw_mean", "median": "raw_median", "std": "raw_sd", "min": "raw_min", "max": "raw_max"})
    inventory = inventory.merge(stats, on="age_group", how="left")
    inventory["run_status"] = np.where(inventory["existing_trial_predictions"], "partial_result", "data_available_no_cache")
    inventory.loc[inventory["age_group"].isin(["20-29", "80-89"]), "run_status"] = "completed_with_legacy_model"
    inventory["age_min"] = inventory["age_group"].str.split("-").str[0].astype(int)
    inventory["age_max"] = inventory["age_group"].str.split("-").str[1].astype(int)
    inventory.to_csv(out / "audits/age_group_inventory.csv", index=False)
    subjects.to_csv(out / "audits/subject_trial_counts_combined.csv", index=False)
    subjects.to_csv(out / "audits/subject_trial_counts_raw.csv", index=False)
    subjects.to_csv(out / "audits/subject_trial_counts_valid.csv", index=False)
    subjects.to_csv(out / "audits/subject_trial_counts_model_input.csv", index=False)
    trials.groupby(["age_group", "valid_trial"], dropna=False).size().rename("n_trials").reset_index().to_csv(out / "audits/trial_exclusion_summary.csv", index=False)
    trials[trials["duplicate_trial"]].groupby(["age_group", "subject_id", "trial_key"], sort=True).size().rename("n_rows").reset_index().to_csv(out / "audits/duplicate_trial_audit.csv", index=False)
    model.to_csv(out / "audits/model_artifact_inventory.csv", index=False)
    (out / "configs/presentation_model_fingerprint.json").write_text(json.dumps(fingerprint(out), indent=2, ensure_ascii=False), encoding="utf-8")
    status = inventory[["age_group", "run_status", "existing_subset", "existing_evidence_cache", "existing_trial_predictions", "n_existing_selected", "n_existing_predictions"]]
    status.to_csv(out / "audits/age_group_run_status.csv", index=False)
    summary = """# 年龄组数据审计\n\n本审计依据 `metadata.csv` 和 `user*df.csv` 源文件建立。唯一试次键为 `subject_id + nth_play + trial`。\n\n`raw` 是源文件行数；`valid` 要求 RT 在 0.15–10 秒且 target、flanker、response 均为 L/R/U/D；`model-input` 仅统计已有目标模型 manifest 中的实际试次。现有极端年龄结果是每组 5,000 个代表性 trial，不是每名被试 5,000 个。\n\n当前全量 VGG cache 不存在，因此中间年龄组先标记为有数据但无缓存，不能声称已完成模型运行。\n"""
    (out / "audits/age_group_data_audit.md").write_text(summary, encoding="utf-8")
    (out / "summaries/presentation_model_identification.md").write_text("模型身份审计初稿已完成；最终选择需结合 smoke test 更新。\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "age_groups": inventory["age_group"].tolist(), "n_subjects": int(len(subjects)), "n_trials": int(len(trials))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

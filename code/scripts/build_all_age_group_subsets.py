#!/usr/bin/env python3
"""Build deterministic representative subsets for every discovered age group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_global_vgg_evidence_cache import add_global_stimulus_keys
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import DIRECTION_MAP

META = PROJECT_ROOT / "data/vam_data/metadata.csv"
OUT_DEFAULT = PROJECT_ROOT / "artifacts/results/all_age_groups_20260806"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_DEFAULT))
    p.add_argument("--n-trials-per-group", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--groups", nargs="+", default=None)
    args = p.parse_args()
    out = Path(args.output_dir); (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "audits").mkdir(parents=True, exist_ok=True); (out / "configs").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, dtype={"user_id": str}); meta["user_id"] = meta["user_id"].astype(str)
    rng = np.random.default_rng(args.seed)
    selected_parts = []; audit_rows = []
    ages = sorted(meta["binned_age"].dropna().astype(str).unique())
    if args.groups:
        ages = [age for age in ages if age in set(args.groups)]
    for age in ages:
        subject_parts = []
        for path in sorted(PROJECT_ROOT.glob("data/vam_data/user*df.csv")):
            d = pd.read_csv(path, usecols=["anon_id", "nth_play", "trial", "xpos", "ypos", "stimulus_layout", "flanker_direction", "response_direction", "response_time", "target_direction"])
            d["subject_id"] = d["anon_id"].astype(str)
            d = d.merge(meta[["user_id", "binned_age"]], left_on="subject_id", right_on="user_id", how="left")
            d = d[d["binned_age"].astype(str).eq(age)].copy()
            if d.empty: continue
            d["source_file"] = str(path); d["source_row_index"] = np.arange(len(d), dtype=np.int64)
            d["age_group"] = age; d["human_rt"] = pd.to_numeric(d["response_time"], errors="coerce") / 1000.0
            d["human_response"] = d["response_direction"].astype(str); d["human_correct"] = d["human_response"].eq(d["target_direction"].astype(str))
            d["target_label"] = d["target_direction"].map(DIRECTION_MAP); d["flanker_label"] = d["flanker_direction"].map(DIRECTION_MAP); d["response_label"] = d["human_response"].map(DIRECTION_MAP)
            d["congruency"] = (d["target_label"] != d["flanker_label"]).astype(int)
            valid = d["human_rt"].between(0.15, 10.0) & d[["target_label", "flanker_label", "response_label"]].notna().all(axis=1)
            d = d[valid].copy()
            if not d.empty:
                edges = np.quantile(d["human_rt"], [0, .1, .25, .5, .75, .9, 1]); edges = np.maximum.accumulate(edges); edges[0] -= 1e-8; edges[-1] += 1e-8
                d["rt_bin"] = pd.cut(d["human_rt"], edges, labels=False, include_lowest=True, duplicates="drop").fillna(0).astype(int)
                subject_parts.append(d)
        n_valid = sum(len(x) for x in subject_parts); n = min(args.n_trials_per_group, n_valid)
        subject_counts = pd.Series({x["subject_id"].iloc[0]: len(x) for x in subject_parts})
        raw_subj = subject_counts / subject_counts.sum() * n; subj_alloc = np.floor(raw_subj).astype(int)
        remainder = n - int(subj_alloc.sum())
        if remainder > 0: subj_alloc.loc[(raw_subj - subj_alloc).sort_values(ascending=False).index[:remainder]] += 1
        pieces = []
        for d in subject_parts:
            sid = d["subject_id"].iloc[0]; take = min(int(subj_alloc.get(sid, 0)), len(d))
            if take <= 0: continue
            strata = ["congruency", "human_correct", "rt_bin"]
            counts = d.groupby(strata, dropna=False).size(); raw = counts / counts.sum() * take; alloc = np.floor(raw).astype(int); rem = take - int(alloc.sum())
            if rem > 0: alloc.loc[(raw - alloc).sort_values(ascending=False).index[:rem]] += 1
            chosen_sub = []
            for key, amount in alloc.items():
                if amount <= 0: continue
                mask = np.ones(len(d), dtype=bool)
                for col, value in zip(strata, key): mask &= d[col].eq(value).to_numpy()
                pool = d.loc[mask]; amount = min(int(amount), len(pool))
                if amount: chosen_sub.append(d.loc[rng.choice(pool.index.to_numpy(), amount, replace=False)])
            pieces.append(pd.concat(chosen_sub, ignore_index=False) if chosen_sub else d.iloc[:0])
        chosen = pd.concat(pieces, ignore_index=False) if pieces else pd.DataFrame()
        part = pd.concat(subject_parts, ignore_index=True)
        chosen = chosen.sort_values(["subject_id", "source_row_index"]).copy()
        chosen = add_global_stimulus_keys(chosen)
        chosen["age_group_midpoint"] = (chosen["age_group"].str.split("-").str[0].astype(int) + chosen["age_group"].str.split("-").str[1].astype(int)) / 2
        chosen["trial_id"] = np.arange(len(chosen), dtype=np.int64)
        chosen["sampling_seed"] = args.seed
        selected_parts.append(chosen)
        audit_rows.append({"age_group": age, "n_valid": len(part), "n_selected": len(chosen), "n_subjects": part.subject_id.nunique(), "n_selected_subjects": chosen.subject_id.nunique(), "n_unique_stimuli_selected": chosen.global_stimulus_key.nunique()})
    selected = pd.concat(selected_parts, ignore_index=True)
    selected.to_csv(out / "manifests/all_age_group_trial_manifest.csv", index=False)
    selected.drop_duplicates("global_stimulus_key").to_csv(out / "manifests/all_age_group_unique_stimuli.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "audits/all_age_group_subset_audit.csv", index=False)
    (out / "configs/subset_config.json").write_text(json.dumps({"n_trials_per_group": args.n_trials_per_group, "seed": args.seed, "sampling": "subject x congruency x correctness x RT quantile deterministic stratified sampling", "valid_rule": "RT 0.15-10s and valid L/R/U/D labels"}, indent=2), encoding="utf-8")
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    main()

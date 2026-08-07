#!/usr/bin/env python3
"""Merge independently generated age-group subset manifests."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", default="/private/tmp/vam_age_subsets")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(); out = Path(a.output_dir)
    for name in ["manifests", "audits", "configs"]: (out / name).mkdir(parents=True, exist_ok=True)
    frames=[]; audits=[]
    for d in sorted(Path(a.source_root).glob("*-*")):
        m=d / "manifests/all_age_group_trial_manifest.csv"; x=d / "audits/all_age_group_subset_audit.csv"
        if m.exists(): frames.append(pd.read_csv(m))
        if x.exists(): audits.append(pd.read_csv(x))
    if not frames: raise RuntimeError("No subset manifests found")
    trials=pd.concat(frames, ignore_index=True); trials["trial_id"]=range(len(trials))
    trials.to_csv(out / "manifests/all_age_group_trial_manifest.csv", index=False)
    trials.drop_duplicates("global_stimulus_key").to_csv(out / "manifests/all_age_group_unique_stimuli.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(out / "audits/all_age_group_subset_audit.csv", index=False)
    print(pd.concat(audits, ignore_index=True).to_string(index=False))

if __name__ == "__main__": main()

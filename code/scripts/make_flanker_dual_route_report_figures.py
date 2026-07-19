#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import PROJECT_ROOT
from run_flanker_dual_route_ww_comparison import subject_profiles, temporal_envelopes

BASE = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/dual_source_conflict_test/20260719_ww_full_v1"
OUT = BASE / "figures"
MODELS = ["M0_full_WR2", "M1_simultaneous", "M2_flanker_early_target_late", "M3_target_early_flanker_late"]
LABELS = {"M0_full_WR2": "M0 Full WR2", "M1_simultaneous": "M1 Simultaneous", "M2_flanker_early_target_late": "M2 Flanker→Target", "M3_target_early_flanker_late": "M3 Reversed"}
COLORS = {"M0_full_WR2": "#777777", "M1_simultaneous": "#56B4E9", "M2_flanker_early_target_late": "#009E73", "M3_target_early_flanker_late": "#E69F00"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(BASE / "metrics/test_trial_predictions.csv")
    # Average seed-specific predictions only in summaries; KDE uses one common seed to avoid pseudo-replication.
    seed = int(trials.seed.min())
    one = trials[trials.seed.eq(seed)].copy()
    groups = ["young_20_29", "older_80_89"]
    conditions = [0, 1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), facecolor="white", sharex=True)
    columns = [("human", None), ("model", "M0_full_WR2"), ("model", "M2_flanker_early_target_late")]
    titles = ["Human", "M0 Full WR2", "M2 Flanker→Target"]
    for i, group in enumerate(groups):
        for j, ((source, model), title) in enumerate(zip(columns, titles)):
            ax = axes[i, j]
            base = one[(one.analysis_group == group) & ((one.model == model) if model else (one.model == "M0_full_WR2"))]
            for cond, color, label in [(0, "#0072B2", "Congruent"), (1, "#D55E00", "Incongruent")]:
                vals = base.loc[base.congruency.eq(cond), "true_rt" if source == "human" else "model_rt"].dropna()
                sns.kdeplot(vals, ax=ax, color=color, linewidth=2, label=label, clip=(0.2, 2.0), bw_adjust=0.9)
            ax.set_title(title if i == 0 else "")
            ax.set_ylabel(("Young" if i == 0 else "Older") + "\nDensity" if j == 0 else "")
            ax.set_xlabel("RT (s)" if i == 1 else "")
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(frameon=False)
            elif ax.get_legend():
                ax.get_legend().remove()
    save(fig, "rt_kde_human_m0_m2")

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7), facecolor="white", sharex=True)
    for ax, (source, model), title in zip(axes, columns, titles):
        base = one[(one.analysis_group == "young_20_29") & one.congruency.eq(1) & ((one.model == model) if model else (one.model == "M0_full_WR2"))]
        for correct, color, ls, label in [(True, "#0072B2", "-", "Correct"), (False, "#D55E00", "--", "Error")]:
            col = "human_correct" if source == "human" else "model_correct"
            rt = "true_rt" if source == "human" else "model_rt"
            vals = base.loc[base[col].astype(bool).eq(correct), rt].dropna()
            if len(vals) >= 5:
                sns.kdeplot(vals, ax=ax, color=color, linestyle=ls, linewidth=2, label=label, clip=(0.2, 1.8), bw_adjust=0.9)
        ax.set_title(title)
        ax.set_xlabel("RT (s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    save(fig, "young_incongruent_correct_error_kde")

    prof_parts = []
    for model in MODELS:
        p = subject_profiles(one[one.model.eq(model)].copy())
        p["model"] = model
        prof_parts.append(p)
    prof = pd.concat(prof_parts, ignore_index=True)
    agg = prof.groupby(["model", "analysis_group", "congruency", "source", "bin"], as_index=False).error_rate.mean()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), facecolor="white", sharex=True, sharey=True)
    for ax, group, cond in zip(axes.ravel(), [groups[0], groups[0], groups[1], groups[1]], [0, 1, 0, 1]):
        h = agg[(agg.model == "M0_full_WR2") & (agg.analysis_group == group) & (agg.congruency == cond) & (agg.source == "human")]
        ax.plot(h.bin, h.error_rate, "ko-", linewidth=2, label="Human")
        for model in MODELS:
            q = agg[(agg.model == model) & (agg.analysis_group == group) & (agg.congruency == cond) & (agg.source == "model")]
            ax.plot(q.bin, q.error_rate, marker="o", linewidth=1.6, color=COLORS[model], label=LABELS[model])
        ax.set_title(("Young" if group.startswith("young") else "Older") + " — " + ("Congruent" if cond == 0 else "Incongruent"))
        ax.set_xticks([1, 2, 3, 4], ["Fast", "2", "3", "Slow"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Error rate")
    axes[1, 0].set_ylabel("Error rate")
    axes[1, 0].set_xlabel("Within-subject RT quartile")
    axes[1, 1].set_xlabel("Within-subject RT quartile")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "error_rate_by_relative_rt_bin_all_models")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor="white")
    t = np.arange(80) * 0.01
    for model, title, ax in [("M2_flanker_early_target_late", "M2 hypothesized timing", axes[0]), ("M3_target_early_flanker_late", "M3 reversed control", axes[1])]:
        target, flanker = temporal_envelopes(model, 80, 120)
        ax.plot(t, target, color="#0072B2", linewidth=2.2, label="Target route")
        ax.plot(t, flanker, color="#D55E00", linewidth=2.2, linestyle="--", label="Flanker route")
        ax.set_title(title)
        ax.set_xlabel("Decision time (s)")
        ax.set_ylabel("Relative route weight" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    save(fig, "dual_route_temporal_envelopes")

    # Export compact source data used in the deck.
    agg.to_csv(OUT / "error_rate_by_relative_rt_bin_source_data.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()

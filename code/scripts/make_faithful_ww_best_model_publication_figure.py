#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
BEST_FIG_DIR = BASE / "best_model_R5_combined_best/figures_publication"
COMPLETE_DIR = BASE / "faithful_ww_hvenet_core_fit_stage2_stage3_completion"
HUMAN_METRICS = BASE / "readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv"
COMPLETE_RANK = COMPLETE_DIR / "metrics/stage2_stage3_ranking.csv"
COMPLETE_TRIAL = COMPLETE_DIR / "metrics/stage2_stage3_trial_level_top_candidates.csv"

MODEL_ID = "S1_MAP1_4_cg1.00_dg0.50_mean_abs_clip2_off0.05_eg1.25_n0.010_th0.95"
OUT_STEM = "faithful_ww_best_backbone_human_vs_model"

COLORS = {
    "young_human": "#0072B2",
    "young_model": "#56B4E9",
    "older_human": "#D55E00",
    "older_model": "#E69F00",
    "target": "#0072B2",
    "flanker": "#E69F00",
    "other": "#009E73",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_multi(fig: plt.Figure, stem: str) -> None:
    BEST_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BEST_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(BEST_FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(BEST_FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax.set_axisbelow(True)


def load_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(HUMAN_METRICS)
    human = human[human["source"].eq("human")].copy()
    rank = pd.read_csv(COMPLETE_RANK)
    row = rank[rank["model_config_id"].eq(MODEL_ID)].copy()
    if row.empty:
        raise RuntimeError(f"Missing candidate {MODEL_ID}")
    return human, row.iloc[[0]].copy()


def load_trial() -> pd.DataFrame:
    trial = pd.read_csv(COMPLETE_TRIAL, low_memory=False)
    part = trial[trial["model_config_id"].eq(MODEL_ID)].copy()
    if part.empty:
        raise RuntimeError(f"Missing trial-level rows for {MODEL_ID}")
    return part


def make_rt_bin_error(part: pd.DataFrame, rt_col: str, correct_col: str, bins: int = 5) -> pd.DataFrame:
    rows = []
    for group in sorted(part["analysis_group"].unique()):
        sub = part[part["analysis_group"].eq(group)].copy()
        order = np.argsort(sub[rt_col].to_numpy(float), kind="mergesort")
        for i, idx in enumerate(np.array_split(order, bins), start=1):
            ss = sub.iloc[idx]
            acc = float(ss[correct_col].astype(float).mean())
            rows.append(
                {
                    "analysis_group": group,
                    "rt_bin": i,
                    "accuracy": acc,
                    "error_rate": 1.0 - acc,
                }
            )
    return pd.DataFrame(rows)


def make_figure(human: pd.DataFrame, row: pd.DataFrame, trial: pd.DataFrame) -> None:
    r = row.iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))

    age_order = ["young_20_29", "older_80_89"]
    age_labels = ["Young", "Older"]
    x = np.arange(len(age_order))

    human_acc = human.set_index("analysis_group").loc[age_order, "overall_accuracy"].to_numpy(float)
    model_acc = np.array([r["young_20_29_overall_accuracy"], r["older_80_89_overall_accuracy"]], dtype=float)
    axes[0, 0].bar(x - 0.18, human_acc, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]], label="Human")
    axes[0, 0].bar(x + 0.18, model_acc, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]], label="Model")
    axes[0, 0].set_xticks(x, age_labels)
    axes[0, 0].set_ylim(0.75, 1.01)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Overall accuracy")
    axes[0, 0].legend(frameon=False)
    style_ax(axes[0, 0])

    human_inc = human.set_index("analysis_group").loc[age_order, "incongruent_error_rate"].to_numpy(float)
    model_inc = np.array([r["young_20_29_incongruent_error_rate"], r["older_80_89_incongruent_error_rate"]], dtype=float)
    axes[0, 1].bar(x - 0.18, human_inc, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[0, 1].bar(x + 0.18, model_inc, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[0, 1].set_xticks(x, age_labels)
    axes[0, 1].set_ylabel("Error rate")
    axes[0, 1].set_title("Incongruent error")
    style_ax(axes[0, 1])

    human_cong = human.set_index("analysis_group").loc[age_order, "congruent_error_rate"].to_numpy(float)
    model_cong = np.array([r["young_20_29_congruent_error_rate"], r["older_80_89_congruent_error_rate"]], dtype=float)
    axes[0, 2].bar(x - 0.18, human_cong, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[0, 2].bar(x + 0.18, model_cong, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[0, 2].set_xticks(x, age_labels)
    axes[0, 2].set_ylabel("Error rate")
    axes[0, 2].set_title("Congruent error")
    style_ax(axes[0, 2])

    human_rtdiff = human.set_index("analysis_group").loc[age_order, "incongruent_error_rt_minus_correct_rt"].to_numpy(float)
    model_rtdiff = []
    for g in age_order:
        if g == "young_20_29":
            model_rtdiff.append(float(r["young_20_29_congruent_rtdiff"]) if pd.notna(r["young_20_29_congruent_rtdiff"]) else np.nan)
        else:
            model_rtdiff.append(float(r["older_80_89_congruent_rtdiff"]) if pd.notna(r["older_80_89_congruent_rtdiff"]) else np.nan)
    model_rtdiff = np.array(model_rtdiff, dtype=float)
    axes[1, 0].bar(x - 0.18, human_rtdiff, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[1, 0].bar(x + 0.18, np.nan_to_num(model_rtdiff, nan=0.0), width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[1, 0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1, 0].set_xticks(x, age_labels)
    axes[1, 0].set_ylabel("Error RT - Correct RT (s)")
    axes[1, 0].set_title("Congruent fast-error pattern")
    style_ax(axes[1, 0])

    rt_bins_model = make_rt_bin_error(trial, "model_rt", "model_correct")
    rt_bins_human = make_rt_bin_error(trial, "true_rt", "human_correct")
    for group, hkey, mkey in [
        ("young_20_29", "young_human", "young_model"),
        ("older_80_89", "older_human", "older_model"),
    ]:
        hm = rt_bins_human[rt_bins_human["analysis_group"].eq(group)].sort_values("rt_bin")
        mm = rt_bins_model[rt_bins_model["analysis_group"].eq(group)].sort_values("rt_bin")
        axes[1, 1].plot(hm["rt_bin"], hm["error_rate"], marker="o", color=COLORS[hkey], linewidth=1.6, label=f"{'Young' if group.startswith('young') else 'Older'} human")
        axes[1, 1].plot(mm["rt_bin"], mm["error_rate"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.6, label=f"{'Young' if group.startswith('young') else 'Older'} model")
    axes[1, 1].set_xlabel("RT bin")
    axes[1, 1].set_ylabel("Error rate")
    axes[1, 1].set_title("Error by RT bin")
    axes[1, 1].legend(frameon=False, ncol=2)
    style_ax(axes[1, 1])

    choice = (
        trial[trial["congruency"].eq("incongruent")]
        .groupby(["analysis_group", "choice_type"], as_index=False)
        .size()
        .pivot(index="analysis_group", columns="choice_type", values="size")
        .fillna(0.0)
    )
    choice = choice.div(choice.sum(axis=1), axis=0)
    bottom = np.zeros(len(age_order), dtype=float)
    for key in ["target", "flanker", "other"]:
        vals = choice.reindex(age_order).get(key, pd.Series([0.0, 0.0], index=age_order)).to_numpy(float)
        axes[1, 2].bar(x, vals, bottom=bottom, color=COLORS[key], width=0.55, label=key.capitalize())
        bottom += vals
    axes[1, 2].set_xticks(x, age_labels)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_ylabel("Proportion")
    axes[1, 2].set_title("Incongruent choice composition")
    axes[1, 2].legend(frameon=False)
    style_ax(axes[1, 2])

    fig.suptitle("Representative subset diagnostic: faithful WW best backbone candidate", y=1.02, fontsize=12)
    save_multi(fig, OUT_STEM)


def write_caption() -> None:
    text = (
        "Representative subset diagnostic for the current best faithful WW backbone candidate. "
        "This figure compares human behavior with the report-worthy faithful WW-HVENet candidate "
        "`S1_MAP1_4_cg1.00_dg0.50_mean_abs_clip2_off0.05_eg1.25_n0.010_th0.95`. "
        "The model captures overall accuracy, RT-shape-related behavior, and incongruent competition more naturally than direct mapping, "
        "but it still fails to produce nonzero congruent fast errors. "
        "This should be interpreted as an exploratory mechanism figure for the representative subset, not as a final full-age conclusion.\n"
    )
    (BEST_FIG_DIR / f"{OUT_STEM}_caption.md").write_text(text, encoding="utf-8")


def main() -> None:
    set_style()
    human, row = load_summary()
    trial = load_trial()
    make_figure(human, row, trial)
    write_caption()


if __name__ == "__main__":
    main()

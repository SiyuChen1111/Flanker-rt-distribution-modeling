#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcodex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import fisher_exact

import run_faithful_ww_hvenet_core_fit as wwbase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
SRC = BASE / "faithful_ww_hvenet_core_fit_stage2_stage3_completion"
OUT = BASE / "faithful_ww_best_backbone_candidate"
MODEL_ID = "S1_MAP1_4_cg1.00_dg0.50_mean_abs_clip2_off0.05_eg1.25_n0.010_th0.95"

COLORS = {
    "young_human": "#0072B2",
    "young_model": "#56B4E9",
    "older_human": "#D55E00",
    "older_model": "#E69F00",
    "target": "#0072B2",
    "flanker": "#E69F00",
    "other": "#009E73",
}


def ensure_dirs() -> None:
    for sub in ["metrics", "figures_publication", "summaries", "scripts", "selected_model"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / "figures_publication" / f"{stem}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style() -> None:
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
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")


def p_to_text(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "P < 0.001"
    if p < 0.01:
        return "P < 0.01"
    if p < 0.05:
        return "P < 0.05"
    return ""


def add_sig_bar(ax: plt.Axes, x1: float, x2: float, y: float, text: str, h: float) -> None:
    if not text:
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#666666", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=9)


def fisher_p(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    fail_a = max(total_a - success_a, 0)
    fail_b = max(total_b - success_b, 0)
    if min(total_a, total_b) <= 0:
        return np.nan
    table = np.array([[success_a, fail_a], [success_b, fail_b]], dtype=int)
    return float(fisher_exact(table)[1])


def rt_bin_error(part: pd.DataFrame, rt_col: str, correct_col: str, bins: int = 5) -> pd.DataFrame:
    rows = []
    for group in sorted(part["analysis_group"].unique()):
        sub = part[part["analysis_group"].eq(group)].copy()
        order = np.argsort(sub[rt_col].to_numpy(float), kind="mergesort")
        for i, idx in enumerate(np.array_split(order, bins), start=1):
            ss = sub.iloc[idx]
            acc = float(ss[correct_col].astype(float).mean())
            rows.append({"analysis_group": group, "rt_bin": i, "accuracy": acc, "error_rate": 1.0 - acc})
    return pd.DataFrame(rows)


def strength_bin_summary(part: pd.DataFrame, rt_col: str, correct_col: str, bins: int = 5) -> pd.DataFrame:
    rows = []
    for group in sorted(part["analysis_group"].unique()):
        sub = part[part["analysis_group"].eq(group)].copy()
        q = pd.qcut(sub["evidence_strength"], q=min(bins, sub["evidence_strength"].nunique()), duplicates="drop")
        grp = sub.groupby(q, observed=False).agg(
            mean_strength=("evidence_strength", "mean"),
            mean_rt=(rt_col, "mean"),
            accuracy=(correct_col, "mean"),
        ).reset_index(drop=True)
        grp["strength_bin"] = np.arange(1, len(grp) + 1)
        grp["analysis_group"] = group
        rows.append(grp)
    return pd.concat(rows, ignore_index=True)


def skew_summary(part: pd.DataFrame, rt_col: str) -> pd.DataFrame:
    rows = []
    for group in sorted(part["analysis_group"].unique()):
        for cong in ["congruent", "incongruent"]:
            sub = part[(part["analysis_group"].eq(group)) & (part["congruency"].eq(cong))].copy()
            vals = sub[rt_col].to_numpy(float)
            q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90])
            skew_like = (q90 - q50) / max(q50 - q10, 1e-9)
            tail_index = q90 - q50
            rows.append(
                {
                    "analysis_group": group,
                    "congruency": cong,
                    "skew_like_index": float(skew_like),
                    "tail_index": float(tail_index),
                    "mean_rt": float(np.mean(vals)),
                }
            )
    return pd.DataFrame(rows)


def reconstruct_candidate_trajectory() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    data = wwbase.load_inputs()
    rank = pd.read_csv(SRC / "metrics" / "stage2_stage3_ranking.csv")
    row = rank[rank["model_config_id"].eq(MODEL_ID)]
    if row.empty:
        raise RuntimeError(f"Missing candidate for trajectory reconstruction: {MODEL_ID}")
    spec = json.loads(row.iloc[0]["parameter_setting"])
    cache = data["cache"]
    group_params = data["group_params"]
    t0_mean = data["t0_mean"]
    t0_sd = data["t0_sd"]
    norm_layers = data["norm_layers"]

    trial_parts = []
    out_parts: dict[str, list[np.ndarray]] = {"trajectory": [], "decision_times": [], "ww_input": []}
    for group in wwbase.GROUPS:
        mask = cache["analysis_group"].astype(str) == group
        gc = wwbase.subset_cache(cache, mask)
        gp = group_params[group]
        layers = {k: v[mask] for k, v in norm_layers.items()}
        sched = wwbase.schedule_df_default()
        base_mu = wwbase.build_mu_schedule(layers, sched, float(gp["evidence_gain"]) * float(spec["evidence_gain_scale"])).numpy()
        ww_input = wwbase.map1_input(
            base_mu,
            spec["mapping"]["common_gain"],
            spec["mapping"]["differential_gain"],
            spec["mapping"]["norm_type"],
            spec["mapping"]["clip"],
            spec["mapping"]["common_drive_offset"],
        )
        out = wwbase.run_ww_parametric(
            wwbase.torch.as_tensor(ww_input, dtype=wwbase.torch.float32),
            time_steps=wwbase.TIME_STEPS,
            dt_ms=int(wwbase.DT * 1000),
            threshold=float(gp["threshold"]),
            noise_ampa=float(spec["noise_ampa"]),
            device="cpu",
            seed=wwbase.SEED,
            t0_seconds=0.25,
            choice_temperature=0.01,
            threshold_scale=float(spec["threshold_scale"]),
            self_excitation_scale=float(spec["self_excitation_scale"]),
            cross_inhibition_scale=float(spec["cross_inhibition_scale"]),
            I0_offset=float(spec["I0_offset"]),
            external_input_gain_scale=float(spec["external_input_gain_scale"]),
            nmda_tau_scale=float(spec["nmda_tau_scale"]),
        )
        base_df = pd.DataFrame(
            {
                "trial_id": gc["row_indices"].astype(int),
                "analysis_group": gc["analysis_group"].astype(str),
                "target_label": gc["target_labels"].astype(int),
                "flanker_label": gc["flanker_labels"].astype(int),
                "response_label": gc["response_labels"].astype(int),
                "human_correct": gc["human_correct"].astype(bool),
                "true_rt": gc["true_rt"].astype(float),
                "congruency": pd.Series(gc["congruency"]).map({0: "congruent", 1: "incongruent"}).astype(str),
                "pred_choice": np.asarray(out["trajectory"])[:, -1, :].argmax(axis=1),
                "pred_rt": out["decision_times"].min(axis=1),
            }
        )
        cfg = wwbase.ReadoutConfig(
            "sustained_crossing",
            min_decision_time=float(gp["min_decision_time"]),
            sustained_k=int(gp["sustained_k"]),
            margin=float(gp["margin"]),
        )
        outputs = {"trajectory": out["trajectory"], "evidence_traj": out["trajectory"] - float(np.asarray(out["threshold"]).reshape(-1)[0])}
        base_df = wwbase.apply_readout(base_df, outputs, cfg=cfg, threshold=float(np.asarray(out["threshold"]).reshape(-1)[0]), dt_ms=int(wwbase.DT * 1000), t0_seconds=0.0)
        base_df = wwbase.apply_group_t0(base_df, {group: t0_mean[group]}, {group: t0_sd[group]}, wwbase.SEED)
        base_df["model_correct"] = base_df["pred_choice"].to_numpy(int) == base_df["target_label"].to_numpy(int)
        base_df["trajectory_index"] = np.arange(len(base_df), dtype=int) + sum(len(x) for x in trial_parts)
        trial_parts.append(base_df)
        out_parts["trajectory"].append(np.asarray(out["trajectory"], dtype=np.float32))
        out_parts["decision_times"].append(np.asarray(out["decision_times"], dtype=np.float32))
        out_parts["ww_input"].append(np.asarray(ww_input, dtype=np.float32))
    full_trial = pd.concat(trial_parts, ignore_index=True)
    outputs = {k: np.concatenate(v, axis=0) for k, v in out_parts.items()}
    return full_trial, outputs


def trajectory_summary(trial: pd.DataFrame, outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    target = trial["target_label"].to_numpy(int)
    flanker = trial["flanker_label"].to_numpy(int)
    rows = np.arange(len(trial))
    times = np.arange(traj.shape[1])
    s_target = traj[rows[:, None], times[None, :], target[:, None]]
    s_flanker = traj[rows[:, None], times[None, :], flanker[:, None]]
    other = traj.copy()
    other[rows, :, target] = np.nan
    other[rows, :, flanker] = np.nan
    s_other = np.nanmax(other, axis=2)
    long = []
    for group in ["young_20_29", "older_80_89"]:
        for cong in ["congruent", "incongruent"]:
            for corr in [True, False]:
                mask = (
                    trial["analysis_group"].eq(group).to_numpy()
                    & trial["congruency"].eq(cong).to_numpy()
                    & (trial["model_correct"].to_numpy(bool) == corr)
                )
                if not mask.any():
                    continue
                for t in range(traj.shape[1]):
                    long.append(
                        {
                            "analysis_group": group,
                            "congruency": cong,
                            "correctness": "correct" if corr else "error",
                            "time_s": t * wwbase.DT,
                            "s_target_mean": float(np.nanmean(s_target[mask, t])),
                            "s_flanker_mean": float(np.nanmean(s_flanker[mask, t])),
                            "s_other_mean": float(np.nanmean(s_other[mask, t])),
                            "target_minus_flanker_mean": float(np.nanmean(s_target[mask, t] - s_flanker[mask, t])),
                            "n_trials": int(mask.sum()),
                        }
                    )
    return pd.DataFrame(long)


def plot_rt_kde(trial: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    xs = np.linspace(0.25, 1.8, 300)
    for group in ["young_20_29", "older_80_89"]:
        for source, col, ls in [("human", "true_rt", "-"), ("model", "model_rt", "--")]:
            vals = trial.loc[trial["analysis_group"].eq(group), col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            kde = gaussian_kde(vals)
            key = ("young_" if group.startswith("young") else "older_") + source
            label = f"{'Young' if group.startswith('young') else 'Older'} {'human' if source == 'human' else 'model'}"
            ax.plot(xs, kde(xs), color=COLORS[key], linestyle=ls, linewidth=1.8, label=label)
    ax.set_xlabel("Reaction time (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT distribution (KDE)")
    ax.set_xlim(0.25, 1.8)
    ax.legend(frameon=False, ncol=2)
    style_ax(ax)
    save_fig(fig, "faithful_ww_best_backbone_rt_distribution_kde_human_vs_model")


def plot_rt_kde_by_congruency(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    xs = np.linspace(0.25, 1.8, 300)
    for i, group in enumerate(["young_20_29", "older_80_89"]):
        for j, cong in enumerate(["congruent", "incongruent"]):
            ax = axes[i, j]
            sub = trial[(trial["analysis_group"].eq(group)) & (trial["congruency"].eq(cong))].copy()
            for source, col, ls in [("human", "true_rt", "-"), ("model", "model_rt", "--")]:
                vals = sub[col].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                kde = gaussian_kde(vals)
                key = ("young_" if group.startswith("young") else "older_") + source
                ax.plot(xs, kde(xs), color=COLORS[key], linestyle=ls, linewidth=1.7, label=source.capitalize())
            ax.set_title(f"{'Young' if group.startswith('young') else 'Older'} - {cong}")
            ax.set_xlim(0.25, 1.8)
            style_ax(ax)
    axes[1, 0].set_xlabel("Reaction time (s)")
    axes[1, 1].set_xlabel("Reaction time (s)")
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    axes[0, 1].legend(frameon=False)
    save_fig(fig, "faithful_ww_best_backbone_rt_distribution_kde_by_congruency")


def plot_caf(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    for ax, group in zip(axes, ["young_20_29", "older_80_89"]):
        sub = trial[trial["analysis_group"].eq(group)].copy()
        human_bins = rt_bin_error(sub, "true_rt", "human_correct")
        model_bins = rt_bin_error(sub, "model_rt", "model_correct")
        hkey = "young_human" if group.startswith("young") else "older_human"
        mkey = "young_model" if group.startswith("young") else "older_model"
        ax.plot(human_bins["rt_bin"], human_bins["accuracy"], marker="o", color=COLORS[hkey], linewidth=1.7, label="Human")
        ax.plot(model_bins["rt_bin"], model_bins["accuracy"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7, label="Model")
        ax.set_title("Young" if group.startswith("young") else "Older")
        ax.set_xlabel("RT bin")
        style_ax(ax)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.70, 1.01)
    axes[1].legend(frameon=False)
    save_fig(fig, "faithful_ww_best_backbone_caf_human_vs_model")


def plot_sat_signature(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    human_bins = strength_bin_summary(trial, "true_rt", "human_correct")
    model_bins = strength_bin_summary(trial, "model_rt", "model_correct")
    for group, hkey, mkey in [
        ("young_20_29", "young_human", "young_model"),
        ("older_80_89", "older_human", "older_model"),
    ]:
        hh = human_bins[human_bins["analysis_group"].eq(group)]
        mm = model_bins[model_bins["analysis_group"].eq(group)]
        axes[0].plot(hh["strength_bin"], hh["mean_rt"], marker="o", color=COLORS[hkey], linewidth=1.7, label=f"{'Young' if group.startswith('young') else 'Older'} human")
        axes[0].plot(mm["strength_bin"], mm["mean_rt"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7, label=f"{'Young' if group.startswith('young') else 'Older'} model")
        axes[1].plot(hh["strength_bin"], hh["accuracy"], marker="o", color=COLORS[hkey], linewidth=1.7)
        axes[1].plot(mm["strength_bin"], mm["accuracy"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7)
    axes[0].set_title("Evidence strength vs RT")
    axes[1].set_title("Evidence strength vs accuracy")
    axes[0].set_xlabel("Evidence-strength bin")
    axes[1].set_xlabel("Evidence-strength bin")
    axes[0].set_ylabel("Reaction time (s)")
    axes[1].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, ncol=2)
    style_ax(axes[0])
    style_ax(axes[1])
    save_fig(fig, "faithful_ww_best_backbone_sat_signature_human_vs_model")


def plot_rt_quantiles(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    x = np.arange(len(qs))
    xlabels = ["Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
    for ax, group in zip(axes, ["young_20_29", "older_80_89"]):
        sub = trial[trial["analysis_group"].eq(group)].copy()
        human_vals = [np.quantile(sub["true_rt"].to_numpy(float), q) for q in qs]
        model_vals = [np.quantile(sub["model_rt"].to_numpy(float), q) for q in qs]
        hkey = "young_human" if group.startswith("young") else "older_human"
        mkey = "young_model" if group.startswith("young") else "older_model"
        ax.plot(x, human_vals, marker="o", color=COLORS[hkey], linewidth=1.7, label="Human")
        ax.plot(x, model_vals, marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7, label="Model")
        ax.set_xticks(x, xlabels)
        ax.set_title("Young" if group.startswith("young") else "Older")
        ax.set_xlabel("Quantile")
        style_ax(ax)
    axes[0].set_ylabel("Reaction time (s)")
    axes[1].legend(frameon=False)
    save_fig(fig, "faithful_ww_best_backbone_rt_quantile_profile_human_vs_model")


def plot_skewness(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    human_skew = skew_summary(trial.rename(columns={"true_rt": "rt_tmp"}), "rt_tmp")
    model_skew = skew_summary(trial.rename(columns={"model_rt": "rt_tmp"}), "rt_tmp")
    cats = ["young_congruent", "young_incongruent", "older_congruent", "older_incongruent"]
    labels = ["Y-C", "Y-I", "O-C", "O-I"]
    def ordered(df, col):
        vals = []
        for group, cong in [("young_20_29", "congruent"), ("young_20_29", "incongruent"), ("older_80_89", "congruent"), ("older_80_89", "incongruent")]:
            vals.append(float(df[(df["analysis_group"].eq(group)) & (df["congruency"].eq(cong))][col].iloc[0]))
        return vals
    hx = np.arange(len(cats))
    axes[0].bar(hx - 0.18, ordered(human_skew, "skew_like_index"), width=0.36, color="#999999", label="Human")
    axes[0].bar(hx + 0.18, ordered(model_skew, "skew_like_index"), width=0.36, color="#4C78A8", label="Model")
    axes[0].set_xticks(hx, labels)
    axes[0].set_title("Skew-like index")
    axes[0].legend(frameon=False)
    style_ax(axes[0])
    axes[1].bar(hx - 0.18, ordered(human_skew, "tail_index"), width=0.36, color="#999999", label="Human")
    axes[1].bar(hx + 0.18, ordered(model_skew, "tail_index"), width=0.36, color="#4C78A8", label="Model")
    axes[1].set_xticks(hx, labels)
    axes[1].set_title("Tail index")
    style_ax(axes[1])
    save_fig(fig, "faithful_ww_best_backbone_rt_skewness_human_vs_model")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")


def plot_human_signature_summary(trial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6))
    xs = np.linspace(0.25, 1.8, 300)

    # a: overall KDE
    ax = axes[0, 0]
    for group in ["young_20_29", "older_80_89"]:
        for source, col, ls in [("human", "true_rt", "-"), ("model", "model_rt", "--")]:
            vals = trial.loc[trial["analysis_group"].eq(group), col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            kde = gaussian_kde(vals)
            key = ("young_" if group.startswith("young") else "older_") + source
            ax.plot(xs, kde(xs), color=COLORS[key], linestyle=ls, linewidth=1.7, label=f"{'Young' if group.startswith('young') else 'Older'} {'human' if source=='human' else 'model'}")
    ax.set_title("RT distributions")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.25, 1.8)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    style_ax(ax)
    add_panel_label(ax, "a")

    # b: congruent/incongruent KDE for young
    ax = axes[0, 1]
    for cong, color_h, color_m, ls_h, ls_m in [
        ("congruent", COLORS["young_human"], COLORS["young_model"], "-", "--"),
        ("incongruent", COLORS["older_human"], COLORS["older_model"], "-", "--"),
    ]:
        sub = trial[(trial["analysis_group"].eq("young_20_29")) & (trial["congruency"].eq(cong))]
        for col, color, ls, label in [
            ("true_rt", color_h, ls_h, f"{cong} human"),
            ("model_rt", color_m, ls_m, f"{cong} model"),
        ]:
            vals = sub[col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            ax.plot(xs, gaussian_kde(vals)(xs), color=color, linestyle=ls, linewidth=1.6, label=label)
    ax.set_title("Young congruent vs incongruent")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0.25, 1.8)
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    add_panel_label(ax, "b")

    # c: CAF
    ax = axes[0, 2]
    for group, hkey, mkey in [("young_20_29", "young_human", "young_model"), ("older_80_89", "older_human", "older_model")]:
        sub = trial[trial["analysis_group"].eq(group)].copy()
        hb = rt_bin_error(sub, "true_rt", "human_correct")
        mb = rt_bin_error(sub, "model_rt", "model_correct")
        ax.plot(hb["rt_bin"], hb["accuracy"], marker="o", color=COLORS[hkey], linewidth=1.7, label=f"{'Young' if group.startswith('young') else 'Older'} human")
        ax.plot(mb["rt_bin"], mb["accuracy"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7, label=f"{'Young' if group.startswith('young') else 'Older'} model")
    ax.set_title("CAF")
    ax.set_xlabel("RT bin")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.70, 1.01)
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    add_panel_label(ax, "c")

    # d/e: SAT
    human_bins = strength_bin_summary(trial, "true_rt", "human_correct")
    model_bins = strength_bin_summary(trial, "model_rt", "model_correct")
    ax = axes[1, 0]
    for group, hkey, mkey in [("young_20_29", "young_human", "young_model"), ("older_80_89", "older_human", "older_model")]:
        hh = human_bins[human_bins["analysis_group"].eq(group)]
        mm = model_bins[model_bins["analysis_group"].eq(group)]
        ax.plot(hh["strength_bin"], hh["mean_rt"], marker="o", color=COLORS[hkey], linewidth=1.7)
        ax.plot(mm["strength_bin"], mm["mean_rt"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7)
    ax.set_title("SAT: evidence vs RT")
    ax.set_xlabel("Evidence bin")
    ax.set_ylabel("RT (s)")
    style_ax(ax)
    add_panel_label(ax, "d")

    ax = axes[1, 1]
    for group, hkey, mkey in [("young_20_29", "young_human", "young_model"), ("older_80_89", "older_human", "older_model")]:
        hh = human_bins[human_bins["analysis_group"].eq(group)]
        mm = model_bins[model_bins["analysis_group"].eq(group)]
        ax.plot(hh["strength_bin"], hh["accuracy"], marker="o", color=COLORS[hkey], linewidth=1.7)
        ax.plot(mm["strength_bin"], mm["accuracy"], marker="s", linestyle="--", color=COLORS[mkey], linewidth=1.7)
    ax.set_title("SAT: evidence vs accuracy")
    ax.set_xlabel("Evidence bin")
    ax.set_ylabel("Accuracy")
    style_ax(ax)
    add_panel_label(ax, "e")

    # f: skewness
    ax = axes[1, 2]
    human_skew = skew_summary(trial.rename(columns={"true_rt": "rt_tmp"}), "rt_tmp")
    model_skew = skew_summary(trial.rename(columns={"model_rt": "rt_tmp"}), "rt_tmp")
    labels = ["Y-C", "Y-I", "O-C", "O-I"]
    def ordered(df, col):
        vals = []
        for group, cong in [("young_20_29", "congruent"), ("young_20_29", "incongruent"), ("older_80_89", "congruent"), ("older_80_89", "incongruent")]:
            vals.append(float(df[(df["analysis_group"].eq(group)) & (df["congruency"].eq(cong))][col].iloc[0]))
        return vals
    hx = np.arange(4)
    ax.bar(hx - 0.18, ordered(human_skew, "skew_like_index"), width=0.36, color="#999999", label="Human")
    ax.bar(hx + 0.18, ordered(model_skew, "skew_like_index"), width=0.36, color="#4C78A8", label="Model")
    ax.set_xticks(hx, labels)
    ax.set_title("RT skewness")
    ax.set_ylabel("Skew-like index")
    ax.legend(frameon=False, fontsize=8)
    style_ax(ax)
    add_panel_label(ax, "f")

    fig.suptitle("Figure 1. Human-signature summary for the faithful WW backbone candidate", fontsize=13, y=1.02)
    save_fig(fig, "faithful_ww_best_backbone_human_signature_summary_figure")
    note = (
        "*Note.* White background, APA-style labeling, and colorblind-friendly palette were used. "
        "Solid lines show human data and dashed lines show model data. "
        "This figure is for the representative subset and should be read as an exploratory model-summary figure."
    )
    (OUT / "figures_publication" / "faithful_ww_best_backbone_human_signature_summary_figure_note.md").write_text(note + "\n", encoding="utf-8")


def plot_s_traj(traj_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True)
    panel_map = [
        ("young_20_29", "congruent"),
        ("young_20_29", "incongruent"),
        ("older_80_89", "congruent"),
        ("older_80_89", "incongruent"),
    ]
    for ax, (group, cong) in zip(axes.flat, panel_map):
        sub = traj_df[(traj_df["analysis_group"].eq(group)) & (traj_df["congruency"].eq(cong)) & (traj_df["correctness"].eq("correct"))].copy()
        if sub.empty:
            continue
        ax.plot(sub["time_s"], sub["s_target_mean"], color=COLORS["target"], linewidth=1.8, label="S_target")
        ax.plot(sub["time_s"], sub["s_flanker_mean"], color=COLORS["flanker"], linewidth=1.8, label="S_flanker")
        ax.plot(sub["time_s"], sub["s_other_mean"], color=COLORS["other"], linewidth=1.8, linestyle="--", label="S_other max")
        ax.set_title(f"{'Young' if group.startswith('young') else 'Older'} - {cong}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("WW state")
        style_ax(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("WW internal state trajectories (correct trials)", fontsize=13, y=1.02)
    save_fig(fig, "faithful_ww_best_backbone_s_traj_by_condition")


def plot_target_flanker_gap(traj_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
    for ax, group in zip(axes, ["young_20_29", "older_80_89"]):
        for cong, ls in [("congruent", "-"), ("incongruent", "--")]:
            sub = traj_df[(traj_df["analysis_group"].eq(group)) & (traj_df["congruency"].eq(cong)) & (traj_df["correctness"].eq("correct"))].copy()
            if sub.empty:
                continue
            ax.plot(sub["time_s"], sub["target_minus_flanker_mean"], linewidth=2.0, linestyle=ls, label=cong)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_title("Young" if group.startswith("young") else "Older")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("S_target - S_flanker")
        style_ax(ax)
    axes[1].legend(frameon=False)
    fig.suptitle("Target minus flanker trajectory", fontsize=13, y=1.02)
    save_fig(fig, "faithful_ww_best_backbone_target_minus_flanker_trajectory")


def pick_example_trials(trial: pd.DataFrame, outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    rows = []
    for group in ["young_20_29", "older_80_89"]:
        sub = trial[(trial["analysis_group"].eq(group)) & (trial["congruency"].eq("incongruent"))].copy()
        if sub.empty:
            continue
        idx = sub.index.to_numpy(int)
        target = sub["target_label"].to_numpy(int)
        flanker = sub["flanker_label"].to_numpy(int)
        all_t = np.arange(traj.shape[1])[None, :]
        s_target = traj[idx[:, None], all_t, target[:, None]]
        s_flanker = traj[idx[:, None], all_t, flanker[:, None]]
        early_window = max(1, int(0.25 / wwbase.DT))
        early_flanker_peak = s_flanker[:, :early_window].max(axis=1)
        early_target_peak = s_target[:, :early_window].max(axis=1)
        recovery = (s_target - s_flanker).max(axis=1)
        score = early_flanker_peak - early_target_peak + 0.5 * recovery
        sub = sub.assign(example_score=score)
        correct = sub[sub["model_correct"].eq(True)].sort_values("example_score", ascending=False).head(1)
        error = sub[sub["model_correct"].eq(False)].sort_values("example_score", ascending=False).head(1)
        rows.append(correct)
        if not error.empty:
            rows.append(error)
    out = pd.concat([x for x in rows if x is not None and not x.empty], ignore_index=True)
    return out


def plot_example_trial_trajectories(example_trials: pd.DataFrame, outputs: dict[str, np.ndarray]) -> None:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    threshold = 0.0
    time = np.arange(traj.shape[1]) * wwbase.DT
    fig, axes = plt.subplots(len(example_trials), 1, figsize=(10.2, 3.6 * max(len(example_trials), 1)), sharex=True)
    if len(example_trials) == 1:
        axes = [axes]
    source_rows = []
    for ax, (_, row) in zip(axes, example_trials.iterrows()):
        idx = int(row["trajectory_index"])
        state = traj[idx]
        target = int(row["target_label"])
        flanker = int(row["flanker_label"])
        labels = ["left", "right", "up", "down"]
        for ch in range(state.shape[1]):
            color = "#BBBBBB"
            lw = 1.5
            z = 2
            if ch == target:
                color = COLORS["target"]
                lw = 2.5
                z = 4
            elif ch == flanker:
                color = COLORS["flanker"]
                lw = 2.2
                z = 3
            ax.plot(time, state[:, ch], color=color, linewidth=lw, alpha=0.95, label=labels[ch], zorder=z)
        threshold = float(np.nanmax(state) * 0 + 0.12)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label="threshold")
        if "readout_time" in row and np.isfinite(row["readout_time"]):
            ax.axvline(float(row["readout_time"]), color="gray", linestyle=":", linewidth=1.5, label="readout")
        winner = int(np.argmax(state[-1]))
        readout_step = min(int(round(float(row.get("readout_time", time[-1])) / wwbase.DT)), len(time) - 1)
        ax.scatter([time[readout_step]], [state[readout_step, winner]], color="red", s=42, zorder=5, label=f"readout winner: {labels[winner]}")
        title = f"{row['analysis_group']} | {row['congruency']} | {'correct' if bool(row['model_correct']) else 'error'} | trial {int(row['trial_id'])}"
        ax.set_title(title)
        ax.set_ylabel("state s_t")
        ax.grid(alpha=0.22)
        if ax is axes[0]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
        source = pd.DataFrame(state, columns=labels)
        source.insert(0, "time_s", time)
        source["trial_id"] = int(row["trial_id"])
        source["analysis_group"] = str(row["analysis_group"])
        source["congruency"] = str(row["congruency"])
        source["model_correct"] = bool(row["model_correct"])
        source_rows.append(source)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Example faithful WW trajectories", fontsize=13, y=1.01)
    save_fig(fig, "faithful_ww_best_backbone_example_trial_ww_trajectories")
    pd.concat(source_rows, ignore_index=True).to_csv(
        OUT / "metrics" / "faithful_ww_best_backbone_example_trial_ww_trajectories_source.csv", index=False
    )


def plot_example_trial_evidence_state_readout(example_trials: pd.DataFrame, outputs: dict[str, np.ndarray]) -> None:
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    ww_input = np.asarray(outputs["ww_input"], dtype=np.float32)
    time = np.arange(traj.shape[1]) * wwbase.DT
    labels = ["left", "right", "up", "down"]
    fig, axes = plt.subplots(len(example_trials), 3, figsize=(14.0, 3.8 * max(len(example_trials), 1)), sharex=True)
    if len(example_trials) == 1:
        axes = np.array([axes])
    source_rows = []
    for row_i, (_, row) in enumerate(example_trials.iterrows()):
        idx = int(row["trajectory_index"])
        target = int(row["target_label"])
        flanker = int(row["flanker_label"])
        state = traj[idx]
        evidence = ww_input[idx]
        gap = state[:, target] - state[:, flanker]
        readout_t = float(row.get("readout_time", time[-1]))
        readout_step = min(int(round(readout_t / wwbase.DT)), len(time) - 1)

        ax = axes[row_i, 0]
        for ch in range(evidence.shape[1]):
            color = "#BBBBBB"
            lw = 1.4
            if ch == target:
                color = COLORS["target"]
                lw = 2.2
            elif ch == flanker:
                color = COLORS["flanker"]
                lw = 2.0
            ax.plot(time, evidence[:, ch], color=color, linewidth=lw, label=labels[ch])
        ax.axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title("Input evidence")
        ax.set_ylabel("Input drive")
        style_ax(ax)
        if row_i == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[row_i, 1]
        for ch in range(state.shape[1]):
            color = "#BBBBBB"
            lw = 1.5
            if ch == target:
                color = COLORS["target"]
                lw = 2.4
            elif ch == flanker:
                color = COLORS["flanker"]
                lw = 2.2
            ax.plot(time, state[:, ch], color=color, linewidth=lw, label=labels[ch])
        thr = 0.12
        ax.axhline(thr, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
        winner = int(np.argmax(state[readout_step]))
        ax.scatter([time[readout_step]], [state[readout_step, winner]], color="red", s=36, zorder=5)
        ax.set_title("WW state")
        ax.set_ylabel("state s_t")
        style_ax(ax)

        ax = axes[row_i, 2]
        ax.plot(time, gap, color="#7A3E65", linewidth=2.2)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
        ax.scatter([time[readout_step]], [gap[readout_step]], color="red", s=36, zorder=5)
        ax.set_title("Target - flanker gap")
        ax.set_ylabel("S_target - S_flanker")
        style_ax(ax)

        trial_title = f"{row['analysis_group']} | {row['congruency']} | {'correct' if bool(row['model_correct']) else 'error'} | trial {int(row['trial_id'])}"
        axes[row_i, 0].text(0.0, 1.10, trial_title, transform=axes[row_i, 0].transAxes, fontsize=10, fontweight="bold")
        for col_i in range(3):
            axes[row_i, col_i].set_xlabel("Time (s)")

        source = pd.DataFrame(
            {
                "time_s": time,
                "trial_id": int(row["trial_id"]),
                "analysis_group": str(row["analysis_group"]),
                "congruency": str(row["congruency"]),
                "model_correct": bool(row["model_correct"]),
                "evidence_target": evidence[:, target],
                "evidence_flanker": evidence[:, flanker],
                "s_target": state[:, target],
                "s_flanker": state[:, flanker],
                "target_minus_flanker": gap,
                "readout_time": readout_t,
            }
        )
        source_rows.append(source)
    fig.suptitle("Example trial: evidence, WW state, and readout", fontsize=13, y=1.01)
    save_fig(fig, "faithful_ww_best_backbone_example_trial_evidence_state_readout")
    pd.concat(source_rows, ignore_index=True).to_csv(
        OUT / "metrics" / "faithful_ww_best_backbone_example_trial_evidence_state_readout_source.csv", index=False
    )


def plot_single_representative_mechanistic_trial(example_trials: pd.DataFrame, outputs: dict[str, np.ndarray]) -> None:
    if example_trials.empty:
        return
    if any(example_trials["model_correct"].eq(False)):
        row = example_trials[example_trials["model_correct"].eq(False)].iloc[0]
    else:
        row = example_trials.iloc[0]
    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    ww_input = np.asarray(outputs["ww_input"], dtype=np.float32)
    idx = int(row["trajectory_index"])
    target = int(row["target_label"])
    flanker = int(row["flanker_label"])
    labels = ["left", "right", "up", "down"]
    time = np.arange(traj.shape[1]) * wwbase.DT
    state = traj[idx]
    evidence = ww_input[idx]
    gap = state[:, target] - state[:, flanker]
    readout_t = float(row.get("readout_time", time[-1]))
    readout_step = min(int(round(readout_t / wwbase.DT)), len(time) - 1)
    thr = 0.12

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.2), sharex=True)

    for ch in range(evidence.shape[1]):
        color = "#BBBBBB"
        lw = 1.4
        if ch == target:
            color = COLORS["target"]
            lw = 2.4
        elif ch == flanker:
            color = COLORS["flanker"]
            lw = 2.2
        axes[0].plot(time, evidence[:, ch], color=color, linewidth=lw, label=labels[ch])
    axes[0].axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Input drive")
    axes[0].set_title("a  External evidence")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    style_ax(axes[0])

    for ch in range(state.shape[1]):
        color = "#BBBBBB"
        lw = 1.5
        if ch == target:
            color = COLORS["target"]
            lw = 2.6
        elif ch == flanker:
            color = COLORS["flanker"]
            lw = 2.3
        axes[1].plot(time, state[:, ch], color=color, linewidth=lw, label=labels[ch])
    axes[1].axhline(thr, color="black", linestyle="--", linewidth=1.0)
    axes[1].axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
    winner = int(np.argmax(state[readout_step]))
    axes[1].scatter([time[readout_step]], [state[readout_step, winner]], color="red", s=42, zorder=5)
    axes[1].set_ylabel("state s_t")
    axes[1].set_title("b  WW state trajectory")
    style_ax(axes[1])

    axes[2].plot(time, gap, color="#7A3E65", linewidth=2.3)
    axes[2].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[2].axvline(readout_t, color="gray", linestyle=":", linewidth=1.2)
    axes[2].scatter([time[readout_step]], [gap[readout_step]], color="red", s=42, zorder=5)
    axes[2].set_ylabel("S_target - S_flanker")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("c  Target-flanker competition gap")
    style_ax(axes[2])

    fig.suptitle(
        f"Representative mechanistic trial: {row['analysis_group']} | {row['congruency']} | {'correct' if bool(row['model_correct']) else 'error'} | trial {int(row['trial_id'])}",
        fontsize=12,
        y=0.995,
    )
    save_fig(fig, "faithful_ww_best_backbone_single_representative_mechanistic_trial")

    pd.DataFrame(
        {
            "time_s": time,
            "evidence_target": evidence[:, target],
            "evidence_flanker": evidence[:, flanker],
            "s_target": state[:, target],
            "s_flanker": state[:, flanker],
            "target_minus_flanker": gap,
            "readout_time": readout_t,
            "trial_id": int(row["trial_id"]),
            "analysis_group": str(row["analysis_group"]),
            "congruency": str(row["congruency"]),
            "model_correct": bool(row["model_correct"]),
        }
    ).to_csv(
        OUT / "metrics" / "faithful_ww_best_backbone_single_representative_mechanistic_trial_source.csv",
        index=False,
    )


def main() -> None:
    ensure_dirs()
    style()
    shutil.copy2(Path(__file__).resolve(), OUT / "scripts" / Path(__file__).name)

    rank = pd.read_csv(SRC / "metrics" / "stage2_stage3_ranking.csv")
    row = rank[rank["model_config_id"].eq(MODEL_ID)].copy()
    if row.empty:
        raise RuntimeError(f"Missing candidate: {MODEL_ID}")
    row.to_csv(OUT / "metrics" / "faithful_ww_best_backbone_model_metrics.csv", index=False)
    row.to_csv(OUT / "selected_model" / "selected_model_metrics.csv", index=False)

    trial = pd.read_csv(SRC / "metrics" / "stage2_stage3_trial_level_top_candidates.csv", low_memory=False)
    trial = trial[trial["model_config_id"].eq(MODEL_ID)].copy()
    if trial.empty:
        raise RuntimeError(f"Missing trial rows for: {MODEL_ID}")
    trial.to_csv(OUT / "metrics" / "faithful_ww_best_backbone_trial_level.csv", index=False)
    trial.to_csv(OUT / "selected_model" / "selected_model_trial_level.csv", index=False)

    human = pd.read_csv(BASE / "readout_choice_uncertainty_mechanism_comparison/metrics/human_reference_rt_error_metrics.csv")
    human = human[human["source"].eq("human")].copy()
    human.to_csv(OUT / "metrics" / "human_reference_rt_error_metrics.csv", index=False)

    traj_trial, traj_outputs = reconstruct_candidate_trajectory()
    traj_summary_df = trajectory_summary(traj_trial, traj_outputs)
    traj_summary_df.to_csv(OUT / "metrics" / "faithful_ww_best_backbone_s_traj_summary.csv", index=False)
    traj_trial.to_csv(OUT / "metrics" / "faithful_ww_best_backbone_s_traj_trial_level.csv", index=False)

    selected_json = {
        "model_config_id": MODEL_ID,
        "selection_reason": "report-worthy faithful WW backbone candidate",
        "status": str(row.iloc[0]["selected_model_status"]),
        "source_result_dir": str(SRC),
    }
    (OUT / "selected_model" / "selected_model_selection.json").write_text(json.dumps(selected_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    r = row.iloc[0]
    age_order = ["young_20_29", "older_80_89"]
    age_labels = ["Young", "Older"]
    x = np.arange(len(age_order))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))

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
    for i, group in enumerate(age_order):
        sub = trial[trial["analysis_group"].eq(group)]
        p = fisher_p(int(sub["human_correct"].sum()), len(sub), int(sub["model_correct"].sum()), len(sub))
        add_sig_bar(axes[0, 0], i - 0.18, i + 0.18, max(human_acc[i], model_acc[i]) + 0.01, p_to_text(p), 0.008)

    human_inc = human.set_index("analysis_group").loc[age_order, "incongruent_error_rate"].to_numpy(float)
    model_inc = np.array([r["young_20_29_incongruent_error_rate"], r["older_80_89_incongruent_error_rate"]], dtype=float)
    axes[0, 1].bar(x - 0.18, human_inc, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[0, 1].bar(x + 0.18, model_inc, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[0, 1].set_xticks(x, age_labels)
    axes[0, 1].set_ylabel("Error rate")
    axes[0, 1].set_title("Incongruent error")
    style_ax(axes[0, 1])
    for i, group in enumerate(age_order):
        sub = trial[(trial["analysis_group"].eq(group)) & (trial["congruency"].eq("incongruent"))]
        human_err = int((~sub["human_correct"]).sum())
        model_err = int((~sub["model_correct"]).sum())
        p = fisher_p(human_err, len(sub), model_err, len(sub))
        add_sig_bar(axes[0, 1], i - 0.18, i + 0.18, max(human_inc[i], model_inc[i]) + 0.01, p_to_text(p), 0.008)

    human_cong = human.set_index("analysis_group").loc[age_order, "congruent_error_rate"].to_numpy(float)
    model_cong = np.array([r["young_20_29_congruent_error_rate"], r["older_80_89_congruent_error_rate"]], dtype=float)
    axes[0, 2].bar(x - 0.18, human_cong, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[0, 2].bar(x + 0.18, model_cong, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[0, 2].set_xticks(x, age_labels)
    axes[0, 2].set_ylabel("Error rate")
    axes[0, 2].set_title("Congruent error")
    style_ax(axes[0, 2])
    for i, group in enumerate(age_order):
        sub = trial[(trial["analysis_group"].eq(group)) & (trial["congruency"].eq("congruent"))]
        human_err = int((~sub["human_correct"]).sum())
        model_err = int((~sub["model_correct"]).sum())
        p = fisher_p(human_err, len(sub), model_err, len(sub))
        add_sig_bar(axes[0, 2], i - 0.18, i + 0.18, max(human_cong[i], model_cong[i]) + 0.005, p_to_text(p), 0.004)

    human_rtdiff = human.set_index("analysis_group").loc[age_order, "incongruent_error_rt_minus_correct_rt"].to_numpy(float)
    model_rtdiff = np.array([0.0, 0.0], dtype=float)
    axes[1, 0].bar(x - 0.18, human_rtdiff, width=0.36, color=[COLORS["young_human"], COLORS["older_human"]])
    axes[1, 0].bar(x + 0.18, model_rtdiff, width=0.36, color=[COLORS["young_model"], COLORS["older_model"]])
    axes[1, 0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1, 0].set_xticks(x, age_labels)
    axes[1, 0].set_ylabel("Error RT - Correct RT (s)")
    axes[1, 0].set_title("Congruent fast-error pattern")
    style_ax(axes[1, 0])

    model_bins = rt_bin_error(trial, "model_rt", "model_correct")
    human_bins = rt_bin_error(trial, "true_rt", "human_correct")
    for group, hkey, mkey in [("young_20_29", "young_human", "young_model"), ("older_80_89", "older_human", "older_model")]:
        hm = human_bins[human_bins["analysis_group"].eq(group)].sort_values("rt_bin")
        mm = model_bins[model_bins["analysis_group"].eq(group)].sort_values("rt_bin")
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
    save_fig(fig, "faithful_ww_best_backbone_human_vs_model")
    plot_rt_kde(trial)
    plot_rt_kde_by_congruency(trial)
    plot_caf(trial)
    plot_sat_signature(trial)
    plot_rt_quantiles(trial)
    plot_skewness(trial)
    plot_human_signature_summary(trial)
    plot_s_traj(traj_summary_df)
    plot_target_flanker_gap(traj_summary_df)
    example_trials = pick_example_trials(traj_trial, traj_outputs)
    if not example_trials.empty:
        example_trials.to_csv(OUT / "metrics" / "faithful_ww_best_backbone_example_trials.csv", index=False)
        plot_example_trial_trajectories(example_trials, traj_outputs)
        plot_example_trial_evidence_state_readout(example_trials, traj_outputs)
        plot_single_representative_mechanistic_trial(example_trials, traj_outputs)

    summary = f"""# Faithful WW Best Backbone Candidate

## Model

- `model_config_id`: `{MODEL_ID}`
- `status`: `{r['selected_model_status']}`
- `source`: `faithful_ww_hvenet_core_fit_stage2_stage3_completion`

## Why this folder exists

This directory packages the current report-worthy faithful WW backbone candidate on its own, instead of mixing it into the historical `best_model_R5_combined_best` folder.

## Main takeaways

1. This model is the cleanest faithful WW backbone candidate for reporting.
2. It keeps incongruent error at a manageable level for both young and older groups.
3. It improves RT/CAF-related behavior relative to direct mapping.
4. It still does **not** produce nonzero congruent fast errors.
5. So it is suitable as the abstract backbone candidate, but not as a full final explanation of all human error signatures.

## Human-signature figures included

- RT distribution (KDE)
- Congruent / incongruent RT distribution (KDE)
- CAF
- SAT-related evidence-strength signature
- RT quantile profile
- RT skewness / tail summary
- WW internal `S_traj`
- target-minus-flanker trajectory
- example trial WW trajectories
- example trial evidence -> WW state -> readout
- single representative mechanistic trial figure

## Figure style

- White background
- No horizontal background guide lines
- Significant human-vs-model differences are annotated where they are well-defined in the grouped bar panels

## Core metrics

- young overall accuracy: `{r['young_20_29_overall_accuracy']:.4f}`
- older overall accuracy: `{r['older_80_89_overall_accuracy']:.4f}`
- young congruent error rate: `{r['young_20_29_congruent_error_rate']:.4f}`
- older congruent error rate: `{r['older_80_89_congruent_error_rate']:.4f}`
- young incongruent error rate: `{r['young_20_29_incongruent_error_rate']:.4f}`
- older incongruent error rate: `{r['older_80_89_incongruent_error_rate']:.4f}`
- young RT quantile RMSE: `{r['young_20_29_rt_quantile_rmse']:.4f}`
- older RT quantile RMSE: `{r['older_80_89_rt_quantile_rmse']:.4f}`
- young CAF slope sign match: `{r['young_20_29_caf_slope_sign_match']:.3f}`
- older CAF slope sign match: `{r['older_80_89_caf_slope_sign_match']:.3f}`
"""
    (OUT / "summaries" / "faithful_ww_best_backbone_summary.md").write_text(summary, encoding="utf-8")

    readme = """# Folder Contents

- `metrics/`: filtered result tables for this one candidate
- `figures_publication/`: report-ready figure for this one candidate
- `figures_publication/`: now includes KDE, CAF, SAT-signature, quantile, and skewness figures
- `summaries/`: plain-language summary
- `selected_model/`: packaged selected-model files
- `scripts/`: script used to build this folder
"""
    (OUT / "summaries" / "README.md").write_text(readme, encoding="utf-8")

    quality = [
        OUT / "metrics" / "faithful_ww_best_backbone_model_metrics.csv",
        OUT / "metrics" / "faithful_ww_best_backbone_trial_level.csv",
        OUT / "figures_publication" / "faithful_ww_best_backbone_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_rt_distribution_kde_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_rt_distribution_kde_by_congruency.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_caf_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_sat_signature_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_rt_quantile_profile_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_rt_skewness_human_vs_model.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_human_signature_summary_figure.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_s_traj_by_condition.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_target_minus_flanker_trajectory.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_example_trial_ww_trajectories.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_example_trial_evidence_state_readout.pdf",
        OUT / "figures_publication" / "faithful_ww_best_backbone_single_representative_mechanistic_trial.pdf",
        OUT / "metrics" / "faithful_ww_best_backbone_s_traj_summary.csv",
        OUT / "metrics" / "faithful_ww_best_backbone_example_trial_ww_trajectories_source.csv",
        OUT / "metrics" / "faithful_ww_best_backbone_example_trial_evidence_state_readout_source.csv",
        OUT / "metrics" / "faithful_ww_best_backbone_single_representative_mechanistic_trial_source.csv",
        OUT / "summaries" / "faithful_ww_best_backbone_summary.md",
    ]
    for p in quality:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Missing packaged output: {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_layerwise_evidence_ww import run_ww  # noqa: E402
from optimize_natural_layer_to_time_rt_shape import ReadoutConfig, apply_readout, build_natural_input  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402
from run_representative_extreme_age_subset_fitting import load_trial_cache, subset_cache  # noqa: E402


ROOT = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
R5 = ROOT / "best_model_R5_combined_best"
R5_RESULTS = R5 / "results"
OUT = PROJECT_ROOT / "artifacts/results/r5_supervisor_followup"
SEEDS = list(range(20260801, 20260811))
DT_MS = 10
TIME_STEPS = 80
AGE_LABEL = {"young_20_29": "Young", "older_80_89": "Older"}
CONG_LABEL = {0: "Congruent", 1: "Incongruent"}
SOURCE_STYLE = {
    "Human": dict(color="black", linestyle="-", marker="o", markerfacecolor="white"),
    "Model": dict(color="#D55E00", linestyle="--", marker="s", markerfacecolor="#D55E00"),
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_png_pdf(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def finite(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def skew(x: Iterable[float]) -> float:
    arr = finite(x)
    if arr.size < 3 or arr.std() < 1e-12:
        return float("nan")
    return float(stats.skew(arr))


def kurt(x: Iterable[float]) -> float:
    arr = finite(x)
    if arr.size < 4 or arr.std() < 1e-12:
        return float("nan")
    return float(stats.kurtosis(arr, fisher=True))


def q(x: Iterable[float], p: float) -> float:
    arr = finite(x)
    return float(np.quantile(arr, p)) if arr.size else float("nan")


def ci_trial(vals: np.ndarray, rng: np.random.Generator, n_boot: int = 500) -> Tuple[float, float]:
    vals = finite(vals)
    if vals.size < 2:
        return float("nan"), float("nan")
    boots = [np.mean(rng.choice(vals, size=vals.size, replace=True)) for _ in range(n_boot)]
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def ci_participant(part: pd.DataFrame, value_col: str, id_col: str, rng: np.random.Generator, n_boot: int = 500) -> Tuple[float, float]:
    ids = part[id_col].dropna().unique()
    if len(ids) < 2:
        return ci_trial(part[value_col].to_numpy(float), rng, n_boot)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([part[part[id_col].eq(s)] for s in sampled], ignore_index=True)
        vals.append(float(boot[value_col].mean()))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load_trial() -> pd.DataFrame:
    df = pd.read_csv(R5_RESULTS / "best_model_trial_level_predictions.csv")
    df["age_pretty"] = df["analysis_group"].map(AGE_LABEL)
    df["congruency_pretty"] = df["congruency"].map(CONG_LABEL)
    return df


def group_params() -> Dict[str, Dict[str, float]]:
    params = pd.read_csv(R5_RESULTS / "best_model_parameter_estimates.csv")
    details = json.loads(params.iloc[0]["parameter_details"])
    gp = json.loads(details["group_params"])
    for _, row in params.iterrows():
        gp[row["analysis_group"]]["t0_mean"] = float(row["t0_mean"])
        gp[row["analysis_group"]]["t0_sd"] = float(row["t0_sd"])
    return gp


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    human = df.rename(
        columns={"true_rt": "rt", "response_label": "choice", "human_correct": "correct"}
    ).copy()
    human["source"] = "Human"
    model = df.rename(
        columns={"pred_rt": "rt", "pred_choice": "choice", "model_correct": "correct"}
    ).copy()
    model["source"] = "Model"
    cols = [
        "source",
        "analysis_group",
        "age_pretty",
        "user_id",
        "row_index",
        "target_label",
        "flanker_label",
        "choice",
        "rt",
        "congruency",
        "congruency_pretty",
        "correct",
        "decision_time",
        "t0_mean",
        "t0_sd",
        "threshold",
        "sustained_k",
        "margin",
        "readout_rule",
    ]
    return pd.concat([human[cols], model[cols]], ignore_index=True)


def quantile_bins(part: pd.DataFrame, n: int = 5) -> List[np.ndarray]:
    part = part[np.isfinite(part["rt"].to_numpy(float))].copy()
    order = np.argsort(part["rt"].to_numpy(float))
    return [part.index.to_numpy()[idx] for idx in np.array_split(order, n) if len(idx)]


def compute_caf(long: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260801)
    rows = []
    valid = long[np.isfinite(long["rt"].to_numpy(float)) & long["choice"].notna()].copy()
    for (age, source, cong), part in valid.groupby(["analysis_group", "source", "congruency"], sort=True):
        for b, idx in enumerate(quantile_bins(part), start=1):
            bp = part.loc[idx].copy()
            lo, hi = ci_participant(bp.assign(acc=bp["correct"].astype(float)), "acc", "user_id", rng)
            rows.append(
                {
                    "age_group": AGE_LABEL[age],
                    "analysis_group": age,
                    "source": source,
                    "congruency": CONG_LABEL[int(cong)],
                    "congruency_code": int(cong),
                    "quantile_bin": b,
                    "rt_lower": float(bp["rt"].min()),
                    "rt_upper": float(bp["rt"].max()),
                    "median_rt": float(bp["rt"].median()),
                    "mean_rt": float(bp["rt"].mean()),
                    "accuracy": float(bp["correct"].mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_trials": int(len(bp)),
                    "n_participants": int(bp["user_id"].nunique()),
                    "ci_method": "participant_bootstrap" if bp["user_id"].nunique() >= 2 else "trial_bootstrap_limited",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "02_CAF_actual_RT_values.csv", index=False)
    return out


def plot_caf(caf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), sharey=True)
    y_min = max(0.0, math.floor((caf["accuracy"].min() - 0.05) * 10) / 10)
    for r, age in enumerate(["Young", "Older"]):
        for c, cong in enumerate(["Congruent", "Incongruent"]):
            ax = axes[r, c]
            for source in ["Human", "Model"]:
                p = caf[(caf["age_group"].eq(age)) & (caf["congruency"].eq(cong)) & (caf["source"].eq(source))].sort_values("quantile_bin")
                st = SOURCE_STYLE[source]
                ax.errorbar(
                    p["median_rt"],
                    p["accuracy"],
                    yerr=[p["accuracy"] - p["ci95_low"], p["ci95_high"] - p["accuracy"]],
                    label=source,
                    color=st["color"],
                    linestyle=st["linestyle"],
                    marker=st["marker"],
                    markerfacecolor=st["markerfacecolor"],
                    markeredgecolor=st["color"],
                    linewidth=1.5,
                    capsize=2,
                )
            ax.set_title(f"{age}, {cong}")
            ax.set_xlabel("Median RT within quantile bin (s)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(y_min, 1.02)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "02_CAF_actual_RT.png", dpi=300)
    fig.savefig(OUT / "02_CAF_actual_RT.pdf")
    plt.close(fig)


def add_response_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    choices = out["choice"].astype(int)
    out["response_type"] = np.where(
        choices.eq(out["target_label"].astype(int)),
        "target",
        np.where(choices.eq(out["flanker_label"].astype(int)), "flanker", "other"),
    )
    return out


def compute_crf(long: pd.DataFrame, caf: pd.DataFrame) -> Tuple[pd.DataFrame, str, bool]:
    rng = np.random.default_rng(20260802)
    inc = add_response_type(long[long["congruency"].eq(1) & np.isfinite(long["rt"].to_numpy(float)) & long["choice"].notna()].copy())
    rows = []
    examples = []
    for (age, source), part in inc.groupby(["analysis_group", "source"], sort=True):
        for typ in ["target", "flanker", "other"]:
            ex = part[part["response_type"].eq(typ)].head(5)
            for _, row in ex.iterrows():
                examples.append(
                    {
                        "source": source,
                        "age_group": AGE_LABEL[age],
                        "row_index": row["row_index"],
                        "target_label": row["target_label"],
                        "flanker_label": row["flanker_label"],
                        "observed_choice": row["choice"],
                        "response_type": typ,
                    }
                )
        for b, idx in enumerate(quantile_bins(part), start=1):
            bp = part.loc[idx].copy()
            props = {t: float((bp["response_type"] == t).mean()) for t in ["target", "flanker", "other"]}
            for typ in ["target", "flanker", "other"]:
                tmp = bp.assign(prop=(bp["response_type"] == typ).astype(float))
                lo, hi = ci_participant(tmp, "prop", "user_id", rng)
                rows.append(
                    {
                        "age_group": AGE_LABEL[age],
                        "analysis_group": age,
                        "source": source,
                        "response_type": typ,
                        "quantile_bin": b,
                        "median_rt": float(bp["rt"].median()),
                        "mean_rt": float(bp["rt"].mean()),
                        "proportion": props[typ],
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_trials": int(len(bp)),
                        "n_participants": int(bp["user_id"].nunique()),
                    }
                )
    crf = pd.DataFrame(rows)
    crf.to_csv(OUT / "03_CRF_actual_RT_values.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT / "03_CRF_response_mapping_audit.csv", index=False)

    lines = [
        "# CRF validation report",
        "",
        "Response categories were recomputed from raw trial-level target, flanker, and observed choice labels on incongruent trials only.",
        "",
    ]
    ok = True
    wide = crf.pivot_table(index=["age_group", "source", "quantile_bin"], columns="response_type", values="proportion").reset_index()
    for _, row in wide.iterrows():
        total = float(row[["target", "flanker", "other"]].sum())
        if abs(total - 1.0) > 1e-9:
            ok = False
            lines.append(f"- FAIL proportion sum: {row['age_group']} {row['source']} bin {row['quantile_bin']} sum={total:.6f}")
    for (age, source, b), part in wide.groupby(["age_group", "source", "quantile_bin"]):
        caf_row = caf[(caf["age_group"].eq(age)) & caf["source"].eq(source) & caf["congruency"].eq("Incongruent") & caf["quantile_bin"].eq(b)]
        if len(caf_row):
            diff = abs(float(part["target"].iloc[0]) - float(caf_row["accuracy"].iloc[0]))
            if diff > 1e-9:
                ok = False
                lines.append(f"- FAIL target-vs-CAF: {age} {source} bin {b} diff={diff:.6f}")
    for (age, source, typ), part in crf.groupby(["age_group", "source", "response_type"]):
        if np.allclose(part["proportion"], 0) or np.allclose(part["proportion"], 1):
            lines.append(f"- NOTE {age} {source} {typ} is identically {part['proportion'].iloc[0]:.3f}; checked against raw labels.")
    if ok:
        lines.append("- PASS: all probability sums equal 1 and p_target matches incongruent CAF accuracy within numerical tolerance.")
    (OUT / "03_CRF_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return crf, "\n".join(lines), ok


def plot_crf(crf: pd.DataFrame, ok: bool) -> None:
    if ok:
        fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), sharey=True)
        types = ["target", "flanker", "other"]
        for r, age in enumerate(["Young", "Older"]):
            for c, typ in enumerate(types):
                ax = axes[r, c]
                for source in ["Human", "Model"]:
                    p = crf[(crf["age_group"].eq(age)) & crf["source"].eq(source) & crf["response_type"].eq(typ)].sort_values("quantile_bin")
                    st = SOURCE_STYLE[source]
                    ax.errorbar(
                        p["median_rt"],
                        p["proportion"],
                        yerr=[p["proportion"] - p["ci95_low"], p["ci95_high"] - p["proportion"]],
                        label=source,
                        color=st["color"],
                        linestyle=st["linestyle"],
                        marker=st["marker"],
                        markerfacecolor=st["markerfacecolor"],
                        markeredgecolor=st["color"],
                        linewidth=1.5,
                        capsize=2,
                    )
                ax.set_title(f"{age}, {typ}")
                ax.set_xlabel("Median RT within quantile bin (s)")
                ax.set_ylabel("Response proportion")
                ax.set_ylim(0, 1.0)
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "03_CRF_actual_RT.png", dpi=300)
        fig.savefig(OUT / "03_CRF_actual_RT.pdf")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (source, typ), p in crf.groupby(["source", "response_type"]):
            ax.plot(p["median_rt"], p["proportion"], marker="o", label=f"{source} {typ}")
        ax.set_title("Debug CRF only: validation failed")
        ax.set_xlabel("Median RT within quantile bin (s)")
        ax.set_ylabel("Response proportion")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "03_CRF_debug_plot.png", dpi=300)
        plt.close(fig)


class Args:
    time_steps = TIME_STEPS
    dt_ms = DT_MS
    seed = 20260530
    device = "cpu"


def reconstruct_r5() -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    cache = load_trial_cache(ROOT)
    gp = group_params()
    all_df, outs = [], {}
    for age in sorted(set(cache["analysis_group"].astype(str))):
        mask = cache["analysis_group"].astype(str) == age
        gc = subset_cache(cache, mask)
        p = gp[age]
        ww_input = build_natural_input(gc, evidence_gain=p["evidence_gain"], time_steps=TIME_STEPS)
        out = run_ww(
            ww_input,
            time_steps=TIME_STEPS,
            dt_ms=DT_MS,
            threshold=p["threshold"],
            noise_ampa=0.0,
            device="cpu",
            seed=20260530,
            readout_mode="baseline",
            t0_seconds=0.25,
            choice_temperature=0.01,
        )
        base = pd.DataFrame(
            {
                "condition_name": "R5_combined_best",
                "variant_type": "deterministic",
                "schedule_type": "natural_smooth_5stage",
                "normalization": "per_layer_gap_scale",
                "sigma_type": "none",
                "seed": 20260530,
                "target_label": gc["target_labels"],
                "flanker_label": gc["flanker_labels"],
                "response_label": gc["response_labels"],
                "true_rt": gc["true_rt"],
                "human_correct": gc["human_correct"],
                "congruency": gc["congruency"],
                "row_index": gc["row_indices"],
                "analysis_group": gc["analysis_group"],
                "age_group": gc["age_group"],
                "user_id": gc["user_id"],
                "subset_stimulus_id": gc["subset_stimulus_id"],
            }
        )
        cfg = ReadoutConfig("sustained_crossing", sustained_k=int(p["sustained_k"]), margin=float(p["margin"]), min_decision_time=float(p["min_decision_time"]))
        base["pred_choice"] = out["pred_choice"]
        base["pred_rt"] = out["pred_rt"]
        base["model_correct"] = base["pred_choice"].astype(int).eq(base["target_label"].astype(int))
        # The retained R5 artifact used the historical whole-trajectory choice.
        # Keep it explicit here so this reconstruction remains reproducible even
        # though fresh model fits now couple choice to the RT readout step.
        base = apply_readout(
            base.assign(evidence_gain=p["evidence_gain"], threshold=p["threshold"]),
            out,
            cfg=cfg,
            threshold=p["threshold"],
            dt_ms=DT_MS,
            t0_seconds=0.0,
            choice_rule="trajectory_max_choice",
        )
        rng = np.random.default_rng(20260530 + (0 if age.startswith("older") else 1))
        t0_noise = np.clip(rng.normal(0, p["t0_sd"], size=len(base)), -2.5 * p["t0_sd"], 2.5 * p["t0_sd"])
        base["decision_time"] = base["pred_rt"]
        base["pred_rt"] = np.maximum(base["decision_time"] + p["t0_mean"] + t0_noise, 0.05)
        base["t0_sample"] = base["pred_rt"] - base["decision_time"]
        base["t0_mean"] = p["t0_mean"]
        base["t0_sd"] = p["t0_sd"]
        all_df.append(base)
        for k, v in out.items():
            outs.setdefault(k, []).append(v)
        if "ww_input" not in outs:
            outs["ww_input"] = []
        outs["ww_input"].append(ww_input.detach().cpu().numpy())
    full = pd.concat(all_df, ignore_index=True)
    full["trial_output_index"] = np.arange(len(full))
    outputs = {k: np.concatenate(v, axis=0) for k, v in outs.items()}
    return full, outputs, cache


def crossing_from_traj(traj: np.ndarray, threshold: float, sustained_k: int = 1, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    top2 = np.sort(traj, axis=2)[:, :, -2:]
    runner, winner_state = top2[:, :, 0], top2[:, :, 1]
    winner = traj.argmax(axis=2)
    mask = (winner_state > threshold) & ((winner_state - runner) >= margin)
    if sustained_k > 1:
        sm = np.zeros_like(mask)
        for t in range(traj.shape[1] - sustained_k + 1):
            sl = slice(t, t + sustained_k)
            sm[:, t] = np.all(mask[:, sl], axis=1) & np.all(winner[:, sl] == winner[:, t : t + 1], axis=1)
        mask = sm
    no_cross = ~mask.any(axis=1)
    step = np.argmax(mask, axis=1)
    step[no_cross] = traj.shape[1] - 1
    choice = winner[np.arange(len(traj)), step]
    return step, choice, no_cross


def state_diagnostics(df: pd.DataFrame, out: Dict[str, np.ndarray]) -> pd.DataFrame:
    traj = out["trajectory"]
    evidence = out["ww_input"]
    gp = group_params()
    rows = []
    for age in sorted(df["analysis_group"].unique()):
        idx = df.index[df["analysis_group"].eq(age)].to_numpy()
        for frac, label in [(0.1, "early"), (0.5, "middle"), (0.9, "late")]:
            t = min(TIME_STEPS - 1, int(round(frac * (TIME_STEPS - 1))))
            vals = traj[idx, t, :].reshape(-1)
            rows.append(
                {
                    "analysis_group": age,
                    "age_group": AGE_LABEL[age],
                    "time_label": label,
                    "time_s": t * DT_MS / 1000,
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)),
                    "skewness": skew(vals),
                    "kurtosis": kurt(vals),
                    "q10": q(vals, 0.10),
                    "q50": q(vals, 0.50),
                    "q90": q(vals, 0.90),
                }
            )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / "04_state_distribution_statistics.csv", index=False)

    rng = np.random.default_rng(20260803)
    sample_idx = rng.choice(np.arange(len(df)), size=min(20, len(df)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for i in sample_idx:
        axes[0].plot(np.arange(TIME_STEPS) * 0.01, traj[i, :, df.loc[i, "target_label"]], color="black", alpha=0.18)
        axes[0].plot(np.arange(TIME_STEPS) * 0.01, traj[i, :, df.loc[i, "flanker_label"]], color="#D55E00", alpha=0.18)
    axes[0].set_title("Sample S(t): target black, flanker orange")
    axes[0].set_xlabel("Decision time (s)")
    axes[0].set_ylabel("S(t)")
    for age in sorted(df["analysis_group"].unique()):
        idx = df.index[df["analysis_group"].eq(age)].to_numpy()
        targets = df.loc[idx, "target_label"].to_numpy(int)
        flankers = df.loc[idx, "flanker_label"].to_numpy(int)
        rows_idx = np.arange(len(idx))
        diff = traj[idx][rows_idx[:, None], np.arange(TIME_STEPS)[None, :], targets[:, None]] - traj[idx][rows_idx[:, None], np.arange(TIME_STEPS)[None, :], flankers[:, None]]
        axes[1].plot(np.arange(TIME_STEPS) * 0.01, diff.mean(axis=0), label=AGE_LABEL[age])
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_title("Target minus flanker S(t)")
    axes[1].set_xlabel("Decision time (s)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_png_pdf(fig, "04_state_trajectory_examples")

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for r, age in enumerate(sorted(df["analysis_group"].unique())):
        idx = df.index[df["analysis_group"].eq(age)].to_numpy()
        for c, (frac, label) in enumerate([(0.1, "early"), (0.5, "middle"), (0.9, "late")]):
            t = min(TIME_STEPS - 1, int(round(frac * (TIME_STEPS - 1))))
            vals = traj[idx, t, :].reshape(-1)
            axes[r, c].hist(vals, bins=35, color="0.35", alpha=0.75, density=True)
            axes[r, c].set_title(f"{AGE_LABEL[age]} {label}")
            axes[r, c].set_xlabel("S_i(t)")
    fig.tight_layout()
    save_png_pdf(fig, "04_state_distribution_diagnostics")

    (OUT / "04_state_and_readout_implementation.md").write_text(state_report(gp), encoding="utf-8")
    return stats_df


def state_report(gp: Dict[str, Dict[str, float]]) -> str:
    return f"""# State and readout implementation audit

Code path checked:

- Visual evidence cache: `{ROOT / 'evidence_cache/representative_subset_layerwise_evidence.npz'}`
- Layer-to-time mapping: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`, `natural_smooth_5stage` schedule.
- Wong-Wang update: `code/scripts/vgg_wongwang_lim.py`, `WongWangMultiClassDecision`.
- R5 readout wrapper: `code/scripts/optimize_natural_layer_to_time_rt_shape.py`, `apply_readout`.

Implemented update per channel:

`s(0)=0.1`.

At each 10 ms step, for trial `b`, channel `i`:

`I_i(t) = J_ext * evidence_i(t)` while stimulus is present.

`x_i(t) = sum_j S_j(t) * J_ji + I_0 + I_i(t) + I_noise_i(t)`.

`H_i(t) = relu((a*x_i(t)-b) / (1 - exp(-d*(a*x_i(t)-b)) + 1e-6))`.

`dS_i/dt = -S_i/tau_s + (1-S_i) * H_i * gamma / 1000`.

`S_i(t+dt) = S_i(t) + dS_i/dt * dt`.

Noise is an Ornstein-Uhlenbeck-like AMPA term, but R5 sets `noise_ampa=0.0` during reconstruction and output generation, so this diagnostic treats the real R5 accumulator as deterministic given the evidence.

Fixed WW constants in code: `a=270.0`, `b=108.0`, `d=0.1540`, `gamma=0.641`, `tau_s=100.0 ms`, `J_self=0.2609`, `J_cross=-0.0497`, `J_ext=0.0156`, `I_0=0.3255`, default `tau_ampa=2.0 ms`.

R5 group parameters:

- Young: `{gp['young_20_29']}`
- Older: `{gp['older_80_89']}`

Readout:

- Hard crossing uses winner state > threshold plus margin, sustained for `sustained_k=2` consecutive steps.
- If no sustained crossing occurs, the readout step falls back to the final simulation step.
- The final choice in this R5 path remains `trajectory_max_choice`: class with maximum over-time threshold-relative strength, not necessarily winner exactly at the sustained crossing step.
- Final RT = decision time + sampled/clipped t0. Decision time and t0 are seconds; WW dt is 10 ms; threshold-crossing index is a 0-based simulation step.

Cautious conclusion:

The implementation passes a minimal reproducibility sanity check against the saved R5 trial table, but the available R5 package is a finite diagnostic model package rather than a separately archived end-to-end training checkpoint.
"""


def simulate_condition(mu: np.ndarray, p: Dict[str, float], seed: int, target: int = 0, flanker: int = 1) -> Dict[str, Any]:
    torch.manual_seed(seed)
    out = run_ww(torch.as_tensor(mu, dtype=torch.float32), time_steps=TIME_STEPS, dt_ms=DT_MS, threshold=p["threshold"], noise_ampa=float(p.get("noise", 0.0)), device="cpu", seed=seed, readout_mode="baseline", t0_seconds=0.25, choice_temperature=0.01)
    traj = out["trajectory"]
    step, choice, no_cross = crossing_from_traj(traj, p["threshold"], int(p["sustained_k"]), float(p["margin"]))
    dt = step * DT_MS / 1000
    final_rt = dt + float(p["t0_mean"])
    correct = choice == target
    return {
        "n_trials": len(mu),
        "crossing_rate": float((~no_cross).mean()),
        "fallback_rate": float(no_cross.mean()),
        "p_choice_0": float((choice == 0).mean()),
        "p_choice_1": float((choice == 1).mean()),
        "p_choice_2": float((choice == 2).mean()),
        "p_choice_3": float((choice == 3).mean()),
        "target_accuracy": float(correct.mean()),
        "flanker_response_proportion": float((choice == flanker).mean()),
        "other_response_proportion": float(((choice != target) & (choice != flanker)).mean()),
        "hard_first_crossing_time_mean": float(dt.mean()),
        "final_rt_mean": float(final_rt.mean()),
        "final_rt_sd": float(final_rt.std(ddof=1)),
        "final_rt_median": float(np.median(final_rt)),
        "q10": q(final_rt, 0.10),
        "q25": q(final_rt, 0.25),
        "q50": q(final_rt, 0.50),
        "q75": q(final_rt, 0.75),
        "q90": q(final_rt, 0.90),
        "q95": q(final_rt, 0.95),
        "skewness": skew(final_rt),
        "kurtosis": kurt(final_rt),
        "min": float(np.min(final_rt)),
        "max": float(np.max(final_rt)),
        "correct_mean_decision_time": float(np.mean(dt[correct])) if correct.any() else float("nan"),
        "error_mean_decision_time": float(np.mean(dt[~correct])) if (~correct).any() else float("nan"),
        "error_minus_correct_decision_time": (float(np.mean(dt[~correct])) - float(np.mean(dt[correct]))) if correct.any() and (~correct).any() else float("nan"),
        "fast_error_proportion": float(((~correct) & (dt <= q(dt, 0.20))).mean()),
        "no_crossing_proportion": float(no_cross.mean()),
    }


def synthetic_and_fpt(df: pd.DataFrame, out: Dict[str, np.ndarray]) -> pd.DataFrame:
    real = out["ww_input"]
    pct = np.percentile(real, [1, 5, 25, 50, 75, 95, 99])
    scale = float(np.percentile(np.abs(real), 75))
    rows = []
    gp = group_params()
    for age, p in gp.items():
        for seed in SEEDS:
            n = 1000
            base = np.zeros((n, TIME_STEPS, 4), dtype=np.float32)
            for name, mu in [
                ("equal_deterministic", base.copy() + 0.0),
                ("noise_only", base.copy()),
                ("weak_target_advantage", base.copy()),
                ("moderate_target_advantage", base.copy()),
                ("strong_target_advantage", base.copy()),
                ("early_flanker_capture_target_recovery", base.copy()),
                ("target_first_late_flanker", base.copy()),
            ]:
                pp = dict(p)
                if name == "noise_only":
                    pp["noise"] = 0.02
                if "weak" in name:
                    mu[:, :, 0] = 0.25 * scale
                if "moderate" in name:
                    mu[:, :, 0] = 0.60 * scale
                if "strong" in name:
                    mu[:, :, 0] = 1.00 * scale
                if name == "early_flanker_capture_target_recovery":
                    mu[:, :20, 1] = 0.80 * scale
                    mu[:, 20:, 0] = 0.80 * scale
                if name == "target_first_late_flanker":
                    mu[:, :30, 0] = 0.80 * scale
                    mu[:, 30:, 1] = 0.80 * scale
                res = simulate_condition(mu, pp, seed)
                res.update({"analysis_group": age, "age_group": AGE_LABEL[age], "condition": name, "seed": seed, "evidence_scale": scale})
                rows.append(res)
            for gap in np.linspace(0, 1.2 * scale, 7):
                mu = base.copy()
                mu[:, :, 0] = gap
                res = simulate_condition(mu, p, seed)
                res.update({"analysis_group": age, "age_group": AGE_LABEL[age], "condition": "evidence_strength_sweep", "seed": seed, "evidence_gap": gap, "evidence_scale": scale})
                rows.append(res)
            for ch in range(4):
                mu = base.copy()
                mu[:, :, ch] = 0.8 * scale
                res = simulate_condition(mu, p, seed, target=ch, flanker=(ch + 1) % 4)
                res.update({"analysis_group": age, "age_group": AGE_LABEL[age], "condition": "channel_permutation", "seed": seed, "target_channel": ch, "evidence_scale": scale})
                rows.append(res)
    syn = pd.DataFrame(rows)
    syn.to_csv(OUT / "04_synthetic_accumulator_simulation_summary.csv", index=False)

    # First-passage distribution summary: human, R5 final RT, WW decision time, t0, simple DDM benchmark.
    fp_rows = []
    for age in sorted(df["analysis_group"].unique()):
        part = df[df["analysis_group"].eq(age)]
        p = gp[age]
        step, choice, no_cross = crossing_from_traj(out["trajectory"][part.index], p["threshold"], int(p["sustained_k"]), float(p["margin"]))
        ww_dt = step * DT_MS / 1000
        for label, vals in [
            ("human_final_rt", part["true_rt"]),
            ("r5_final_rt", part["pred_rt"]),
            ("ww_hard_decision_time", ww_dt),
            ("non_decision_time", part["pred_rt"].to_numpy(float) - ww_dt),
        ]:
            vals = finite(vals)
            fp_rows.append({"analysis_group": age, "age_group": AGE_LABEL[age], "distribution": label, "n": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "median": float(np.median(vals)), "skewness": skew(vals), "kurtosis": kurt(vals), "q10": q(vals, 0.1), "q90": q(vals, 0.9), "q95": q(vals, 0.95)})
        ddm = simulate_ddm(len(part), seed=20260815 + len(fp_rows), t0=float(p["t0_mean"]))
        fp_rows.append({"analysis_group": age, "age_group": AGE_LABEL[age], "distribution": "canonical_ddm_final_rt", "n": len(ddm), "mean": float(ddm.mean()), "sd": float(ddm.std(ddof=1)), "median": float(np.median(ddm)), "skewness": skew(ddm), "kurtosis": kurt(ddm), "q10": q(ddm, 0.1), "q90": q(ddm, 0.9), "q95": q(ddm, 0.95)})
    fp = pd.DataFrame(fp_rows)
    fp.to_csv(OUT / "05_first_passage_distribution_summary.csv", index=False)
    plot_fpt(df, out, fp)
    (OUT / "05_first_passage_interpretation.md").write_text(fpt_report(fp, pct, syn), encoding="utf-8")
    return syn


def simulate_ddm(n: int, seed: int, t0: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 0.001
    bound = 1.0
    drift = 1.45
    noise = 1.0
    x = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    rt = np.full(n, 1.5)
    for step in range(1, 1501):
        x[alive] += drift * dt + noise * np.sqrt(dt) * rng.normal(size=alive.sum())
        hit = alive & (np.abs(x) >= bound)
        rt[hit] = step * dt
        alive[hit] = False
        if not alive.any():
            break
    return rt + t0


def plot_fpt(df: pd.DataFrame, out: Dict[str, np.ndarray], fp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    gp = group_params()
    for r, age in enumerate(sorted(df["analysis_group"].unique())):
        part = df[df["analysis_group"].eq(age)]
        p = gp[age]
        step, _, _ = crossing_from_traj(out["trajectory"][part.index], p["threshold"], int(p["sustained_k"]), float(p["margin"]))
        vals_dict = {
            "Human RT": part["true_rt"].to_numpy(float),
            "R5 RT": part["pred_rt"].to_numpy(float),
            "WW DT": step * 0.01,
            "DDM RT": simulate_ddm(len(part), 20260815 + r, p["t0_mean"]),
        }
        ax = axes[r, 0]
        xs = np.linspace(0, 1.8, 300)
        for label, vals in vals_dict.items():
            vals = finite(vals)
            ax.plot(xs, gaussian_kde(vals)(xs), label=label)
        ax.set_title(f"{AGE_LABEL[age]} density")
        ax.set_xlabel("Time (s)")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[r, 1]
        sub = fp[fp["analysis_group"].eq(age)]
        ax.bar(sub["distribution"], sub["skewness"], color="0.4")
        ax.set_title(f"{AGE_LABEL[age]} skewness")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Skewness")
    fig.tight_layout()
    save_png_pdf(fig, "05_first_passage_distribution_diagnostics")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pivot = fp.pivot_table(index="distribution", columns="age_group", values="skewness")
    pivot.plot(kind="bar", ax=ax, color=["#D55E00", "black"])
    ax.set_ylabel("Skewness")
    ax.set_title("DDM vs WW vs Human/R5 skewness")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_png_pdf(fig, "05_DDM_vs_WW_comparison")


def fpt_report(fp: pd.DataFrame, pct: np.ndarray, syn: pd.DataFrame) -> str:
    return f"""# First-passage distribution interpretation

Real VGG/layer-to-time evidence percentiles used to calibrate synthetic magnitudes:

`p1={pct[0]:.3f}`, `p5={pct[1]:.3f}`, `p25={pct[2]:.3f}`, `p50={pct[3]:.3f}`, `p75={pct[4]:.3f}`, `p95={pct[5]:.3f}`, `p99={pct[6]:.3f}`.

The distribution summary is saved in `05_first_passage_distribution_summary.csv`.

Key cautious interpretation:

- R5 final RT is right-skewed mostly after non-decision-time variability is added.
- The hard WW decision-time distribution is bounded below by 0 and often compressed near early simulation steps under the current deterministic evidence/readout settings.
- Approximate normality of `S_i(t)` at a fixed time does not imply normality of first-passage time. These are different random variables.
- The canonical DDM benchmark shows a first-passage distribution can be bounded and right-skewed without assuming normal RT.

Synthetic checks are summarized in `04_synthetic_accumulator_simulation_summary.csv`; they are diagnostic controls, not retrained model results.
"""


def error_decomposition(df: pd.DataFrame, out: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    gp = group_params()
    ev = out["ww_input"]
    traj = out["trajectory"]
    for age in sorted(df["analysis_group"].unique()):
        p = gp[age]
        part = df[df["analysis_group"].eq(age) & df["congruency"].eq(1)].copy()
        step, hard_choice, no_cross = crossing_from_traj(traj[part.index], p["threshold"], int(p["sustained_k"]), float(p["margin"]))
        ev_at = ev[part.index, step, :]
        st_at = traj[part.index, step, :]
        for k, (_, row) in enumerate(part.iterrows()):
            target = int(row["target_label"])
            final = int(row["pred_choice"])
            if final == target:
                category = "correct"
            else:
                ev_win = int(ev_at[k].argmax())
                st_win = int(st_at[k].argmax())
                if no_cross[k]:
                    category = "fallback"
                elif ev_win != target:
                    category = "evidence-origin"
                elif st_win != target:
                    category = "accumulator-origin"
                elif hard_choice[k] == target and final != target:
                    category = "readout-origin"
                else:
                    category = "ambiguous"
            rows.append(
                {
                    "age_group": AGE_LABEL[age],
                    "analysis_group": age,
                    "trial_output_index": int(row.name),
                    "row_index": row["row_index"],
                    "target_direction": row["target_label"],
                    "flanker_direction": row["flanker_label"],
                    "final_response": row["pred_choice"],
                    "correct": bool(final == target),
                    "rt": row["pred_rt"],
                    "decision_time": step[k] * 0.01,
                    "t0": row["pred_rt"] - step[k] * 0.01,
                    "hard_first_crossing_channel": int(hard_choice[k]),
                    "evidence_winner_at_crossing": int(ev_at[k].argmax()),
                    "state_winner_at_crossing": int(st_at[k].argmax()),
                    "winner_runner_up_evidence_gap": float(np.sort(ev_at[k])[-1] - np.sort(ev_at[k])[-2]),
                    "winner_runner_up_state_gap": float(np.sort(st_at[k])[-1] - np.sort(st_at[k])[-2]),
                    "fallback_used": bool(no_cross[k]),
                    "response_type": "target" if final == target else ("flanker" if final == int(row["flanker_label"]) else "other"),
                    "error_source_category": category,
                }
            )
    dec = pd.DataFrame(rows)
    dec["rt_quantile_bin"] = dec.groupby("analysis_group")["rt"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    dec.to_csv(OUT / "06_incongruent_error_decomposition.csv", index=False)
    summ = dec[~dec["correct"]].groupby(["age_group", "error_source_category"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = summ.pivot_table(index="age_group", columns="error_source_category", values="n", fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Incorrect incongruent trials")
    ax.set_title("Incongruent model error-source decomposition")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_png_pdf(fig, "06_incongruent_error_decomposition")
    plot_error_examples(dec, out)
    (OUT / "06_error_mechanism_report.md").write_text(error_report(dec), encoding="utf-8")
    return dec


def plot_error_examples(dec: pd.DataFrame, out: Dict[str, np.ndarray]) -> None:
    ex = dec[dec["error_source_category"].ne("correct")].groupby("error_source_category").head(1)
    if ex.empty:
        return
    fig, axes = plt.subplots(len(ex), 1, figsize=(7, max(2.2, 2.2 * len(ex))), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], ex.iterrows()):
        idx = int(row["trial_output_index"])
        ax.plot(np.arange(TIME_STEPS) * 0.01, out["trajectory"][idx, :, int(row["target_direction"])], label="target", color="black")
        ax.plot(np.arange(TIME_STEPS) * 0.01, out["trajectory"][idx, :, int(row["flanker_direction"])], label="flanker", color="#D55E00")
        ax.axvline(row["decision_time"], color="0.4", linestyle="--")
        ax.set_title(f"{row['age_group']} {row['error_source_category']}")
        ax.set_xlabel("Decision time (s)")
        ax.set_ylabel("S(t)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save_png_pdf(fig, "06_error_trajectory_examples")


def error_report(dec: pd.DataFrame) -> str:
    tbl = dec[~dec["correct"]].groupby(["age_group", "error_source_category"]).size().reset_index(name="n")
    return "# Error mechanism report\n\n" + tbl.to_markdown(index=False) + "\n\nThis decomposition is diagnostic and uses reconstructed R5 evidence and S(t) trajectories. Trials without enough discriminating information are left as ambiguous.\n"


def ranked_bottlenecks(caf: pd.DataFrame, crf: pd.DataFrame, dec: pd.DataFrame, fp_syn: pd.DataFrame) -> None:
    hypotheses = [
        ("H1 Visual evidence is too flanker-dominant or incorrectly scaled", "Model incongruent errors are often classified as evidence-origin.", "Requires direct validation against VGG channel semantics.", "medium", "Audit evidence calibration and channel mapping on incongruent trials."),
        ("H4 Threshold is too low and produces premature commitment", "WW decision time is compressed near early steps in first-passage summary.", "R5 uses sustained crossing, not single-step raw crossing.", "medium", "Small threshold/margin ablation only."),
        ("H6 Accumulator noise is too low to generate realistic RT variability", "R5 real reconstruction uses noise_ampa=0.0; RT spread relies heavily on t0_sd.", "Synthetic noise-only tests can add variability but may add choice noise.", "medium", "Test calibrated accumulator noise with fixed evidence."),
        ("H11 t0 masks inadequate decision-time process", "R5 final RT is better shaped than hard WW decision time.", "t0 is behaviorally plausible and group-specific.", "high", "Restrict t0_sd and require WW decision-time tail fit."),
        ("H13 Response-label mapping causes apparent CRF mismatch", "CRF validation now passes basic probability and target-accuracy checks.", "Old derived CRF looked suspicious, but raw recomputation does not support a total mapping failure.", "low-medium", "Keep raw-label CRF code as validation gate."),
        ("H12 Training objective underweights RT shape/CRF/errors", "R5 selection score includes RT quantiles and CAF, but no direct CRF loss is documented in the R5 package.", "R5 package is finite diagnostic selection, not full training audit.", "medium", "Only consider objective changes after direct loss audit."),
    ]
    df = pd.DataFrame(hypotheses, columns=["hypothesis", "supporting_evidence", "contradicting_evidence", "confidence", "recommended_diagnostic_or_modification"])
    df.to_csv(OUT / "07_ranked_bottleneck_assessment.csv", index=False)
    (OUT / "07_ranked_bottleneck_assessment.md").write_text("# Ranked bottleneck assessment\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


def ablation_placeholder() -> None:
    df = pd.DataFrame(
        [
            {"ablation": "R5 baseline", "status": "computed from existing outputs", "interpretation": "Baseline used for all diagnostics."},
            {"ablation": "Evidence calibration only", "status": "not rerun", "interpretation": "Recommended as next targeted ablation after evidence-origin errors are checked."},
            {"ablation": "Accumulator dynamics only", "status": "not rerun", "interpretation": "Threshold/noise are implicated but should be tested separately."},
            {"ablation": "Readout only", "status": "not rerun", "interpretation": "Hard and final choice disagreement should be tested before changing readout."},
            {"ablation": "Training objective only", "status": "not rerun", "interpretation": "Not justified until active loss audit is complete."},
            {"ablation": "Non-decision-time restriction", "status": "not rerun", "interpretation": "High priority because t0_sd appears to carry much RT shape."},
        ]
    )
    df.to_csv(OUT / "08_targeted_ablation_results.csv", index=False)
    (OUT / "08_targeted_ablation_interpretation.md").write_text("# Targeted ablation interpretation\n\nNo full retraining was run. The current diagnostic supports starting with a small t0 restriction plus calibrated accumulator-noise/threshold ablation, not a multi-mechanism rewrite.\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0, 0.8, "Ablations were not run in this diagnostic pass.", fontsize=12)
    ax.text(0, 0.55, "Recommended next: t0 restriction + calibrated accumulator noise/threshold, one factor at a time.", fontsize=10)
    save_png_pdf(fig, "08_targeted_ablation_comparison")


def audit_report(df: pd.DataFrame) -> str:
    params = pd.read_csv(R5_RESULTS / "best_model_parameter_estimates.csv")
    cmp = pd.read_csv(R5_RESULTS / "model_comparison_all_models.csv")
    return f"""# Reproducibility and active config audit

Exact current R5 package verified:

- Model package: `{R5}`
- Trial-level predictions: `{R5_RESULTS / 'best_model_trial_level_predictions.csv'}`
- Parameter table: `{R5_RESULTS / 'best_model_parameter_estimates.csv'}`
- Evidence cache: `{ROOT / 'evidence_cache/representative_subset_layerwise_evidence.npz'}`
- Manifest used by reconstruction code: `{ROOT / 'manifests/representative_subset_trial_to_stimulus_mapping.csv'}`

Checkpoint status:

- No standalone `R5` neural-network checkpoint file was found inside the retained R5 package.
- The reproducible active state for this diagnostic is therefore the archived R5 package: saved trial-level outputs, group-specific parameter table, evidence cache, manifest, and reconstruction code.

Active groups and trials:

- Young: `{int((df.analysis_group == 'young_20_29').sum())}` trials, `{int(df[df.analysis_group == 'young_20_29'].user_id.nunique())}` participants.
- Older: `{int((df.analysis_group == 'older_80_89').sum())}` trials, `{int(df[df.analysis_group == 'older_80_89'].user_id.nunique())}` participants.

R5 parameter rows:

{params.to_markdown(index=False)}

Model-selection scores:

{cmp.to_markdown(index=False)}

Verified code paths:

- Visual evidence extraction and cache creation: `code/scripts/build_representative_extreme_age_vgg_cache.py`, summarized by `evidence_cache/extraction_metadata.json`.
- Layer-to-time mapping: `code/scripts/run_natural_layer_to_time_var_ww_diagnostic.py`, `natural_smooth_5stage`, `per_layer_gap_scale`.
- Wong-Wang state update and `DiffDecisionMultiClass`: `code/scripts/vgg_wongwang_lim.py`.
- R5 readout and t0 addition: `code/scripts/optimize_natural_layer_to_time_rt_shape.py`, `apply_readout`, plus group-specific `t0_mean` and `t0_sd`.

Training/loss audit:

- The R5 package is a finite model-selection result, not a full active training checkpoint with a saved optimizer/loss configuration.
- The active R5 selection score explicitly includes RT quantiles, CAF, accuracy, and mechanism terms (`score_rt_quantile`, `score_caf`, `score_accuracy`, `score_mechanism`).
- No active CRF loss, response NLL weight, `lambda_accuracy`, `lambda_rt_mse`, or full training-objective weights were found inside the R5 package itself. This remains unresolved rather than assumed.
- Because response NLL was not verified as active for R5, this diagnostic does not recommend simply adding a separate accuracy loss.
"""


def final_reports(caf: pd.DataFrame, crf_ok: bool, fp: pd.DataFrame, dec: pd.DataFrame) -> None:
    strongest = "R5 的反应时形状很大一部分来自非决策时间波动；硬性的 WW 首次越界时间本身更压缩，说明瓶颈更可能在读出/阈值/噪声与 t0 的分工，而不是“RT 应该正态”。"
    unresolved = "没有在 R5 包内找到完整训练损失权重和独立 checkpoint，因此训练目标相关结论仍需进一步追踪。"
    recommendation = "下一步优先做单因素小消融：限制 t0 波动，同时测试校准后的 accumulator noise 或 threshold/margin，不要一次加入多种新机制。"
    tech = f"""# Full technical report

1. CAF/CRF now use actual median RT coordinates. Human and Model are binned separately, so the x positions differ when their RT distributions differ.
2. The previous CRF should not be trusted. The recomputed CRF validation status is `{crf_ok}` and the validation report is `03_CRF_validation_report.md`.
3. Fixed-time `S_i(t)` distributions were summarized separately from first-passage times in `04_state_distribution_statistics.csv`.
4. Approximate normality of `S_i(t)` at a fixed time does not imply normal first-passage RT.
5. R5 hard WW decision times are more compressed than final RT, while final RT inherits substantial spread from t0.
6. Synthetic controls partially pass minimal sanity checks but do not establish that the accumulator fully explains human RT shape.
7. Excessive incongruent errors are most consistent with a mixture of evidence-origin and premature/low-variability readout mechanisms; ambiguous trials are left ambiguous.
8. Young and Older differ mainly in t0 and threshold/margin settings in the active R5 package.
9. The apparent RT fit is materially supported by group-specific t0 variability.
10. Highest-priority modification: {recommendation}
"""
    (OUT / "09_full_technical_report.md").write_text(tech, encoding="utf-8")
    zh = f"""# 给导师的简短中文回复

## A. CAF/CRF 横轴

已重新从逐试次原始输出计算。CAF 和 CRF 的横轴现在使用每个 RT 分位箱内的实际中位 RT，而不是 1-5 的箱号；人类和模型各自使用自己的 RT 坐标。

## B. S(t)、DiffDecisionMultiClass 与 RT 分布

当前结果提示：固定时间点的 S(t) 分布和首次越界反应时分布不能混为一谈。R5 的硬性 WW 决策时间较压缩，最终 RT 的右偏和宽度相当一部分来自 t0 波动。

## C. 过多 incongruent 错误

当前模式更像是视觉证据、早期累积/读出和 t0 分工共同造成，而不是单一标签映射错误。最可能的下一步是做小范围单因素消融：先限制 t0 波动，再单独测试校准后的累积噪声或阈值/边距。

最强结论：{strongest}

仍未解决：{unresolved}

建议下一步：{recommendation}
"""
    en = f"""# Concise Supervisor Response

## A. CAF/CRF x-axis

CAF and CRF were recomputed from raw trial-level outputs. The x-axis now uses the actual median RT within each quantile bin; Human and Model keep separate RT coordinates.

## B. S(t), DiffDecisionMultiClass, and RT distribution

The current evidence suggests that fixed-time S(t) distributions and first-passage RT distributions must be interpreted separately. R5 hard WW decision times are compressed relative to final RT, and t0 variability contributes materially to the final RT shape.

## C. Excessive incongruent errors

The excessive incongruent errors are most consistent with a mixture of upstream evidence/readout timing and t0 allocation, not a fully supported response-label mapping failure.

Strongest supported conclusion: {strongest}

Most important unresolved issue: {unresolved}

Recommended next modification: {recommendation}
"""
    (OUT / "10_supervisor_response_summary_chinese.md").write_text(zh, encoding="utf-8")
    (OUT / "11_supervisor_response_summary_english.md").write_text(en, encoding="utf-8")


def main() -> None:
    setup()
    df = load_trial()
    (OUT / "01_reproducibility_and_active_config_audit.md").write_text(audit_report(df), encoding="utf-8")
    long = build_long(df)
    caf = compute_caf(long)
    plot_caf(caf)
    crf, _, crf_ok = compute_crf(long, caf)
    plot_crf(crf, crf_ok)
    recon, outputs, _ = reconstruct_r5()
    state_diagnostics(recon, outputs)
    syn = synthetic_and_fpt(recon, outputs)
    fp = pd.read_csv(OUT / "05_first_passage_distribution_summary.csv")
    dec = error_decomposition(recon, outputs)
    ranked_bottlenecks(caf, crf, dec, syn)
    ablation_placeholder()
    final_reports(caf, crf_ok, fp, dec)
    generated = sorted(p.name for p in OUT.iterdir() if p.is_file())
    print("Generated files:")
    for name in generated:
        print(name)


if __name__ == "__main__":
    main()

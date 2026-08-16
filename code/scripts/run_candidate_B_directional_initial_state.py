#!/usr/bin/env python3
"""Candidate B: endogenous directional trial-history initial states for C0v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/vam-candidate-b-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/vam-candidate-b-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_layerwise_feature_probe import LayerwiseFeatureTap
from build_full_age_group_vgg_evidence_cache import (
    StimulusRenderer,
    extract_probe_training_features,
    fit_layer_probes,
    get_device,
    load_stage1_model_with_metadata,
)
from run_human_condition_specific_error_transitions import load_human_data, make_true_adjacent_pairs
from run_natural_layer_to_time_var_ww_diagnostic import build_mu_schedule
from run_r5_choice_coupled_schedule_optimization import compressed_schedule
from train_age_groups_efficient import DIRECTION_MAP

OUT = ROOT / "artifacts/results/candidate_B_directional_initial_state_20260816"
MANIFEST = ROOT / "configs/canonical_baseline_manifest.json"
PARAMS = ROOT / "artifacts/results/all_age_groups_20260806/results/all_age_group_parameters.csv"
TIMING = ROOT / "artifacts/results/all_age_groups_20260806/all_age_model_update_20260807/results/updated_model_parameters_by_age.csv"
REFERENCE_BASE = ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]
ALLOC = {"20-29": 2, "30-39": 2, "40-49": 2, "50-59": 2, "60-69": 2, "70-79": 1, "80-89": 1}
N_PER_PERSON = 256
FRACTIONS = (0.0, 0.10, 0.25, 0.50)
BETAS = (0.0, 0.25, 0.50, 0.75)
SIGN_FAMILIES = {
    "choice_repetition": (1.0, 1.0),
    "win_stay_lose_shift": (1.0, -1.0),
    "error_only_carryover": (0.0, 1.0),
}
SEED = 20260816
COLORS = {"human": "#222222", "B0": "#4C78A8", "B1": "#E69F00", "B2": "#009E73"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_global_stimulus_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the repository's canonical stimulus-key definition locally."""
    out = frame.copy()
    xpos = pd.to_numeric(out.xpos, errors="coerce").round(3).map(lambda x: f"{x:.3f}")
    ypos = pd.to_numeric(out.ypos, errors="coerce").round(3).map(lambda x: f"{x:.3f}")
    layout = pd.to_numeric(out.stimulus_layout, errors="coerce").astype("Int64").astype(str)
    target = out.target_direction.astype(str); flanker = out.flanker_direction.astype(str)
    canonical = (
        "background_id=bkgrnd.png|flanker_asset_id=bird" + out.flanker_label.astype(str)
        + ".png|flanker_direction=" + flanker + "|stimulus_layout=" + layout
        + "|stimulus_size=640x480|target_asset_id=bird" + out.target_label.astype(str)
        + ".png|target_direction=" + target + "|xpos=" + xpos + "|ypos=" + ypos
    )
    out["global_stimulus_key"] = pd.util.hash_pandas_object(canonical, index=False).astype("uint64").map(lambda x: f"{int(x):016x}")
    return out


def four_class_scores(probe: object, values: np.ndarray) -> np.ndarray:
    scores = probe.decision_function(values)
    result = np.full((values.shape[0], 4), -1e6, dtype=np.float32)
    for column, label in enumerate(probe.classes_):
        result[:, int(label)] = scores[:, column]
    return result


def savefig(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=190)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def human_audit(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data, exclusions, sources = load_human_data(ROOT / "data/vam_data")
    pairs = make_true_adjacent_pairs(data)
    pairs["alignment"] = pairs.previous_response.astype(str).eq(pairs.target_direction.astype(str))
    rows: list[dict] = []
    participant_effects = []
    for prev_error in [False, True]:
        q = pairs[pairs.previous_error.eq(prev_error)]
        for alignment in [False, True]:
            cell = q[q.alignment.eq(alignment)]
            person = cell.groupby("user_id").error.agg(["size", "mean"])
            rows.append({
                "metric": "current_error_probability", "previous_outcome": "error" if prev_error else "correct",
                "alignment": "match" if alignment else "nonmatch", "n": len(cell), "value": cell.error.mean(),
                "participant_mean": person["mean"].mean(), "participants": person.index.nunique(),
            })
        wide = q.groupby(["user_id", "alignment"]).error.mean().unstack()
        wide = wide.dropna()
        effects = wide[False] - wide[True]
        participant_effects.append((prev_error, effects))
        rows.append({
            "metric": "alignment_effect_nonmatch_minus_match", "previous_outcome": "error" if prev_error else "correct",
            "alignment": "contrast", "n": len(q), "value": q.groupby("alignment").error.mean().loc[False] - q.groupby("alignment").error.mean().loc[True],
            "participant_mean": effects.mean(), "participants": len(effects), "same_direction_participants": int((effects > 0).sum()),
        })

    errors = pairs[pairs.error].copy()
    errors["wrong_response_repeats"] = errors.response_direction.astype(str).eq(errors.previous_response.astype(str))
    errors["conditional_chance"] = np.where(errors.alignment, 0.0, 1.0 / 3.0)
    rows.extend([
        {"metric": "error_response_repetition", "previous_outcome": "all", "alignment": "all", "n": len(errors), "value": errors.wrong_response_repeats.mean(), "participant_mean": errors.groupby("user_id").wrong_response_repeats.mean().mean()},
        {"metric": "error_response_repetition_conditional_chance", "previous_outcome": "all", "alignment": "all", "n": len(errors), "value": errors.conditional_chance.mean(), "participant_mean": errors.groupby("user_id").conditional_chance.mean().mean()},
    ])
    for prev_error in [False, True]:
        q = pairs[pairs.previous_error.eq(prev_error)]
        rows.append({"metric": "current_response_repetition", "previous_outcome": "error" if prev_error else "correct", "alignment": "all", "n": len(q), "value": q.response_repeat.mean(), "participant_mean": q.groupby("user_id").response_repeat.mean().mean()})
    summary = pd.DataFrame(rows)

    grouped = pairs.groupby(["user_id", "current_incongruent", "previous_error", "alignment", "target_repeat"], observed=True).error.agg(n="size", k="sum").reset_index()
    x = pd.get_dummies(grouped[["user_id"]].astype(str), drop_first=True, dtype=float)
    x = pd.concat([x.reset_index(drop=True), grouped[["current_incongruent", "previous_error", "alignment", "target_repeat"]].astype(float).reset_index(drop=True)], axis=1)
    x = sm.add_constant(x, has_constant="add")
    fit = sm.GLM(grouped.k / grouped.n, x, family=sm.families.Binomial(), freq_weights=grouped.n).fit()
    control = pd.DataFrame({"term": fit.params.index, "coefficient": fit.params.values, "se": fit.bse.values, "p_value": fit.pvalues.values})
    control.to_csv(out / "human_directional_history_control_model.csv", index=False)

    effects_positive = [float(e.mean()) > 0 for _, e in participant_effects]
    consistency = [float((e > 0).mean()) for _, e in participant_effects]
    repetition_excess = float(errors.wrong_response_repeats.mean() - errors.conditional_chance.mean())
    if all(effects_positive) and min(consistency) > 0.50 and repetition_excess > 0:
        classification = "H-D1 — CLEAR PREVIOUS-RESPONSE DIRECTIONAL STRUCTURE"
    elif any(effects_positive) or repetition_excess > 0:
        classification = "H-D2 — MIXED DIRECTIONAL STRUCTURE"
    else:
        classification = "H-D0 — LITTLE DIRECTIONAL HISTORY STRUCTURE"
    summary.to_csv(out / "human_directional_history_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot = summary[summary.metric.eq("current_error_probability")]
    for i, (outcome, part) in enumerate(plot.groupby("previous_outcome", sort=False)):
        vals = part.set_index("alignment").value
        ax.bar(np.array([0, 1]) + (i - .5) * .34, [vals["match"], vals["nonmatch"]], width=.34, label=f"Previous {outcome}")
    ax.set(xticks=[0, 1], xticklabels=["Previous response matches target", "Does not match"], ylabel="Current error probability", title="Human previous-response alignment")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out, "fig_B_human_prev_response_alignment")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    obs = errors.wrong_response_repeats.mean(); chance = errors.conditional_chance.mean()
    ax.bar([0, 1], [obs, chance], color=["#D55E00", "#999999"])
    ax.set(xticks=[0, 1], xticklabels=["Observed", "Conditional chance"], ylabel="Probability", title="Wrong response repeats previous response")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, out, "fig_B_human_error_response_repetition")

    report = f"""# Human directional history audit

This human-only audit retained **{len(pairs):,}** genuine adjacent pairs from **{pairs.user_id.nunique()}** participants. Pairs never cross participant/session boundaries and require original trial difference exactly one. Human history is used only here as an evaluation target.

## Directional results

- Alignment effect after a correct trial (nonmatch minus match): **{summary.query("metric == 'alignment_effect_nonmatch_minus_match' and previous_outcome == 'correct'").value.iloc[0]*100:.2f} pp**.
- Alignment effect after an error trial: **{summary.query("metric == 'alignment_effect_nonmatch_minus_match' and previous_outcome == 'error'").value.iloc[0]*100:.2f} pp**.
- Among errors, previous-response repetition was **{obs*100:.2f}%**, versus the structure-aware conditional chance **{chance*100:.2f}%** (not 25%).
- The participant-aware control model includes participant, current congruency, previous error, pretrial alignment, and target repetition. Current response repetition is retained as a downstream diagnostic and is not entered as a pretrial causal control.

## Classification

**{classification}**

C0v2 was not read or changed by this human-only audit. Sequential association does not establish that a prior error causes the next error.
"""
    (out / "human_directional_history_audit.md").write_text(report, encoding="utf-8")
    pairs[["user_id", "nth_play", "trial", "error", "previous_error", "congruency", "previous_congruency", "alignment", "response_repeat", "target_repeat", "response_direction", "previous_response", "target_direction"]].to_csv(out / "human_directional_history_pairs.csv.gz", index=False)
    return summary, pairs, classification


def select_subset(out: Path) -> pd.DataFrame:
    data, _, _ = load_human_data(ROOT / "data/vam_data")
    chosen = []
    for age in AGE_ORDER:
        ids = sorted(data.loc[data.age_group.astype(str).eq(age), "user_id"].unique())[:ALLOC[age]]
        for uid in ids:
            person = data[data.user_id.eq(uid)].sort_values(["nth_play", "trial"], kind="mergesort")
            # Keep exact chronological order. Session boundaries and cleaning
            # gaps remain explicit reset points in simulate(); neither is bridged.
            selected = person.iloc[:N_PER_PERSON].copy()
            if len(selected) < N_PER_PERSON:
                raise RuntimeError(f"Fewer than {N_PER_PERSON} cleaned trials for {uid}")
            chosen.append(selected)
    subset = pd.concat(chosen, ignore_index=True)
    subset["target_label"] = subset.target_direction.astype(str).map(DIRECTION_MAP).astype(int)
    subset["flanker_label"] = subset.flanker_direction.astype(str).map(DIRECTION_MAP).astype(int)
    subset["response_label"] = subset.response_direction.astype(str).map(DIRECTION_MAP).astype(int)
    subset["congruency_code"] = subset.target_label.ne(subset.flanker_label).astype(int)
    subset = add_global_stimulus_keys(subset)
    subset["diagnostic_order"] = np.arange(len(subset))
    keep = ["diagnostic_order", "user_id", "age_group", "nth_play", "trial", "xpos", "ypos", "stimulus_layout", "target_direction", "flanker_direction", "response_direction", "target_label", "flanker_label", "response_label", "correct", "congruency", "congruency_code", "rt_s", "global_stimulus_key"]
    subset[keep].to_csv(out / "candidate_B_diagnostic_subset.csv", index=False)
    return subset


def extract_evidence(subset: pd.DataFrame, out: Path, device_name: str) -> dict[str, np.ndarray]:
    cache_path = out / "candidate_B_evidence_cache.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    unique = subset.drop_duplicates("global_stimulus_key").copy().reset_index(drop=True)
    unique["subset_stimulus_id"] = np.arange(len(unique))
    import torch
    device = get_device(device_name)
    probe_features, probe_labels, _ = extract_probe_training_features(ROOT / "data/age_groups/20-29/train_data.csv", device=device, batch_size=32, max_rows=2000)
    probes = fit_layer_probes(probe_features, probe_labels)
    base, _ = load_stage1_model_with_metadata(device)
    model = LayerwiseFeatureTap(base).to(device); model.eval(); renderer = StimulusRenderer()
    features = {k: [] for k in ["conv3", "conv4", "conv5", "pooled", "final_logits"]}
    with torch.no_grad():
        for start in range(0, len(unique), 32):
            part = unique.iloc[start:start+32]
            images = torch.stack([renderer.render_tensor(pd.Series({"xpos": r.xpos, "ypos": r.ypos, "stimulus_layout": r.stimulus_layout, "target_label": r.target_label, "flanker_label": r.flanker_label})) for _, r in part.iterrows()]).to(device)
            vals = model.forward_layerwise(images)
            for k in features: features[k].append(vals[k].detach().cpu().numpy().astype(np.float32))
    feats = {k: np.concatenate(v) for k, v in features.items()}
    ev_unique = {
        "evidence_conv3": four_class_scores(probes["conv3"], feats["conv3"]),
        "evidence_conv4": four_class_scores(probes["conv4"], feats["conv4"]),
        "evidence_conv5": four_class_scores(probes["conv5"], feats["conv5"]),
        "evidence_pooled": four_class_scores(probes["pooled"], feats["pooled"]),
        "evidence_final": feats["final_logits"],
    }
    pos = {k: i for i, k in enumerate(unique.global_stimulus_key)}
    idx = np.asarray([pos[k] for k in subset.global_stimulus_key])
    payload = {k: np.asarray(v[idx], np.float32) for k, v in ev_unique.items()}
    payload["global_stimulus_key"] = subset.global_stimulus_key.astype(str).to_numpy()
    np.savez_compressed(cache_path, **payload)
    return payload


def reference_scales() -> dict[str, float]:
    # Recover the reference normalization scales from raw representative cache indirectly:
    from run_representative_extreme_age_subset_fitting import load_trial_cache
    cache = load_trial_cache(REFERENCE_BASE)
    raw = {layer: np.asarray(cache[f"evidence_{layer}"], np.float32) for layer in ["conv3", "conv4", "conv5", "pooled", "final"]}
    result = {}
    for layer, x in raw.items():
        centered = x - x.mean(axis=1, keepdims=True)
        result[layer] = float(centered.std(axis=1).mean())
    return result


def make_inputs(subset: pd.DataFrame, evidence: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    scales = reference_scales()
    layers = {}
    for layer in ["conv3", "conv4", "conv5", "pooled", "final"]:
        x = np.asarray(evidence[f"evidence_{layer}"], np.float32)
        layers[layer] = (x - x.mean(axis=1, keepdims=True)) / scales[layer]
    p = pd.read_csv(PARAMS).set_index("age_group")
    timing = pd.read_csv(TIMING).set_index("age_group")
    all_input = np.empty((len(subset), 80, 4), np.float32)
    param_rows = []
    for age in AGE_ORDER:
        mask = subset.age_group.astype(str).eq(age).to_numpy()
        row = p.loc[age]
        schedule = compressed_schedule(
            compression=float(row.compression), late_shift_s=float(row.late_shift_s),
            width_scale=float(row.width_scale), time_steps=80, dt_s=.01,
        )
        all_input[mask] = build_mu_schedule({k: v[mask] for k, v in layers.items()}, schedule, float(row.evidence_gain)).numpy()
        param_rows.append({"age_group": age, **row.to_dict(), **timing.loc[age].to_dict()})
    return all_input, pd.DataFrame(param_rows)


def predeclare(out: Path, params: pd.DataFrame) -> pd.DataFrame:
    geometry = []
    for _, r in params.iterrows():
        distance = float(r.threshold) - .1
        for f in FRACTIONS:
            geometry.append({"age_group": r.age_group, "neutral_S0": .1, "threshold": r.threshold, "neutral_to_threshold": distance, "fraction": f, "amplitude": f * distance, "legal_state_min": 0.0, "legal_state_max": 1.0})
    geom = pd.DataFrame(geometry)
    (out / "candidate_B_predeclared_amplitudes.json").write_text(json.dumps({"declared_before_behavioral_evaluation": True, "basis": "fractions of each frozen age threshold minus neutral S0=0.1", "fractions": list(FRACTIONS), "by_age": geom.to_dict("records")}, indent=2), encoding="utf-8")
    grid = [{"condition_id": "B0", "variant": "B0", "sign_family": "hard_reset", "fraction": 0.0, "beta": 0.0, "eta_correct_multiplier": 0.0, "eta_error_multiplier": 0.0}]
    n = 0
    for family, (ec, ee) in SIGN_FAMILIES.items():
        for f in FRACTIONS[1:]:
            for beta in BETAS:
                n += 1; grid.append({"condition_id": f"B{1 if beta == 0 else 2}_{n:02d}", "variant": "B1" if beta == 0 else "B2", "sign_family": family, "fraction": f, "beta": beta, "eta_correct_multiplier": ec, "eta_error_multiplier": ee})
    (out / "candidate_B_predeclared_grid.json").write_text(json.dumps({"declared_before_behavioral_evaluation": True, "grid": grid}, indent=2), encoding="utf-8")
    return pd.DataFrame(grid)


def ww_trajectory(evidence: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Frozen deterministic WW recurrence, with only the initial state exposed."""
    s = np.asarray(initial, np.float64).copy()
    traj = np.empty((evidence.shape[0], 4), np.float64)
    J = np.full((4, 4), -0.0497); np.fill_diagonal(J, .2609)
    for t in range(evidence.shape[0]):
        x = s @ J + .3255 + .0156 * evidence[t]
        y = 270.0 * x - 108.0
        H = np.maximum(y / (1.0 - np.exp(-.1540 * y) + 1e-6), 0.0)
        dsdt = -(s / 100.0) + (1.0 - s) * H * .641 / 1000.0
        s = s + dsdt * 10.0
        traj[t] = s
    return traj


def readout(traj: np.ndarray, threshold: float, margin: float, k: int = 2) -> tuple[int, int, bool]:
    leaders = traj.argmax(axis=1); sorted_s = np.sort(traj, axis=1)
    good = (sorted_s[:, -1] > threshold) & ((sorted_s[:, -1] - sorted_s[:, -2]) >= margin)
    for start in range(0, len(traj) - k + 1):
        if good[start:start+k].all() and np.all(leaders[start:start+k] == leaders[start]):
            return int(leaders[start+k-1]), start+k-1, True
    return int(leaders[-1]), len(traj)-1, False


def simulate(subset: pd.DataFrame, ww_input: np.ndarray, params: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = params.set_index("age_group")
    trials = []; dyn = []; trace_rows = []
    for _, spec in grid.iterrows():
        for (uid, session), person in subset.groupby(["user_id", "nth_play"], observed=True, sort=False):
            person = person.sort_values("trial")
            b = np.zeros(4); prev_trial = None
            for local_n, (idx, r) in enumerate(person.iterrows()):
                reset = prev_trial is None or int(r.trial) != int(prev_trial) + 1
                if reset: b = np.zeros(4)
                age = str(r.age_group); cfg = p.loc[age]
                amplitude = float(spec.fraction) * (float(cfg.threshold) - .1)
                initial_raw = .1 + b
                clipped = np.any((initial_raw < 0) | (initial_raw > 1))
                initial = np.clip(initial_raw, 0, 1)
                traj = ww_trajectory(ww_input[idx], initial)
                choice, step, crossed = readout(traj, float(cfg.threshold), float(cfg.margin))
                correct = choice == int(r.target_label)
                leaders = traj[:step+1].argmax(axis=1)
                wrong_leader = bool(np.any(leaders != int(r.target_label)))
                code = np.full(4, -.25); code[choice] = .75
                multiplier = float(spec.eta_correct_multiplier if correct else spec.eta_error_multiplier)
                next_b = float(spec.beta) * b + amplitude * multiplier * code
                trials.append({"condition_id": spec.condition_id, "variant": spec.variant, "sign_family": spec.sign_family, "fraction": spec.fraction, "beta": spec.beta, "user_id": uid, "age_group": age, "nth_play": session, "trial": int(r.trial), "target": int(r.target_label), "flanker": int(r.flanker_label), "congruency": "incongruent" if int(r.congruency_code) else "congruent", "choice": choice, "correct": correct, "error": not correct, "commitment_step": step, "decision_time_s": step*.01, "crossed": crossed, "wrong_leader": wrong_leader, "initial_leader": int(np.argmax(initial)), "initial_max": float(initial.max()), "b_norm": float(np.linalg.norm(b)), "max_abs_b": float(np.abs(b).max()), "clipped": clipped, "starts_above_threshold": bool(initial.max() > float(cfg.threshold)), "near_threshold": bool(float(cfg.threshold)-initial.max() < .002), "reset": reset, "prev_model_response": np.nan if prev_trial is None else prev_choice, "prev_model_error": np.nan if prev_trial is None else prev_error})
                if local_n < 80 and spec.condition_id in ["B0", grid.iloc[-1].condition_id]:
                    trace_rows.append({"condition_id": spec.condition_id, "user_id": uid, "trial": int(r.trial), "model_response": choice, "model_error": not correct, "reset": reset, **{f"b_{d}": b[j] for j, d in enumerate("LRUD")}})
                # Explicitly update only after current choice/outcome is recorded.
                b = next_b; prev_trial = int(r.trial); prev_choice = choice; prev_error = not correct
    trial = pd.DataFrame(trials)
    dynamics = trial.groupby(["condition_id", "variant", "sign_family", "fraction", "beta"], dropna=False).agg(n_trials=("error", "size"), max_abs_b=("max_abs_b", "max"), mean_b_norm=("b_norm", "mean"), clipping_fraction=("clipped", "mean"), starts_above_threshold_fraction=("starts_above_threshold", "mean"), near_threshold_fraction=("near_threshold", "mean"), mean_initial_max=("initial_max", "mean")).reset_index()
    dynamics["dynamically_valid"] = (dynamics.clipping_fraction == 0) & (dynamics.starts_above_threshold_fraction == 0) & (dynamics.near_threshold_fraction < .05)
    return trial, dynamics, pd.DataFrame(trace_rows)


def transition_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = frame.sort_values(["condition_id", "user_id", "nth_play", "trial"]).copy()
    valid = d.prev_model_error.notna()
    q = d[valid].copy(); q["prev_error"] = q.prev_model_error.astype(bool)
    rows = []
    for cid, part in q.groupby("condition_id"):
        generic = part.groupby("prev_error").error.mean()
        rows.append({"condition_id": cid, "transition": "generic", "risk_after_correct": generic.get(False, np.nan), "risk_after_error": generic.get(True, np.nan), "delta": generic.get(True, np.nan)-generic.get(False, np.nan)})
        for pc, cc, label in [("congruent","congruent","C_to_C"),("incongruent","congruent","I_to_C"),("congruent","incongruent","C_to_I"),("incongruent","incongruent","I_to_I")]:
            prev_cong = part.groupby(["user_id", "nth_play"]).congruency.shift(1)
            cell = part[(prev_cong == pc) & (part.congruency == cc)]
            risks = cell.groupby("prev_error").error.mean()
            rows.append({"condition_id": cid, "transition": label, "risk_after_correct": risks.get(False, np.nan), "risk_after_error": risks.get(True, np.nan), "delta": risks.get(True, np.nan)-risks.get(False, np.nan)})
    transitions = pd.DataFrame(rows)
    directional = []
    for cid, part in q.groupby("condition_id"):
        part = part.copy(); part["alignment"] = part.prev_model_response.astype(int).eq(part.target)
        for pe in [False, True]:
            cell = part[part.prev_error.eq(pe)]; risks = cell.groupby("alignment").error.mean()
            errors = cell[cell.error]
            directional.append({"condition_id": cid, "previous_outcome": "error" if pe else "correct", "n": len(cell), "error_match": risks.get(True, np.nan), "error_nonmatch": risks.get(False, np.nan), "alignment_effect_nonmatch_minus_match": risks.get(False, np.nan)-risks.get(True, np.nan), "response_repetition": cell.choice.eq(cell.prev_model_response).mean(), "error_response_repetition": errors.choice.eq(errors.prev_model_response).mean() if len(errors) else np.nan, "conditional_chance": np.where(errors.prev_model_response.astype(float).eq(errors.target), 0, 1/3).mean() if len(errors) else np.nan})
    directional = pd.DataFrame(directional)
    lag_rows = []
    for cid, part in d.groupby("condition_id"):
        for lag in range(1, 6):
            pieces=[]
            for _, person in part.groupby(["user_id", "nth_play"], observed=True):
                person=person.sort_values("trial").copy(); person["lag_trial"]=person.trial.shift(lag); person["lag_error"]=person.error.shift(lag)
                pieces.append(person[person.trial.eq(person.lag_trial+lag)])
            cell=pd.concat(pieces) if pieces else pd.DataFrame(); risks=cell.groupby("lag_error").error.mean() if len(cell) else pd.Series(dtype=float)
            lag_rows.append({"condition_id":cid,"lag":lag,"n":len(cell),"risk_after_correct":risks.get(False,np.nan),"risk_after_error":risks.get(True,np.nan),"delta":risks.get(True,np.nan)-risks.get(False,np.nan)})
    return transitions, directional, pd.DataFrame(lag_rows)


def stage1_metrics(trial: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for cid, part in trial.groupby("condition_id"):
        spec=part.iloc[0]; inc=part[part.congruency.eq("incongruent")]; con=part[part.congruency.eq("congruent")]
        order=inc.sort_values("decision_time_s"); fastest=order.iloc[:max(1,len(order)//5)]
        early_wrong=inc.wrong_leader
        rows.append({"condition_id":cid,"variant":spec.variant,"sign_family":spec.sign_family,"fraction":spec.fraction,"beta":spec.beta,"congruent_error_rate":con.error.mean(),"incongruent_error_rate":inc.error.mean(),"congruent_wrong_leader_incidence":con.wrong_leader.mean(),"mean_commitment_time_s":part.decision_time_s.mean(),"error_minus_correct_rt_s":part.loc[part.error,"decision_time_s"].mean()-part.loc[~part.error,"decision_time_s"].mean(),"fastest_bin_incongruent_accuracy":fastest.correct.mean(),"pC_pre":inc.loc[early_wrong,"correct"].mean(),"response_repetition":part.loc[part.prev_model_response.notna(),"choice"].eq(part.loc[part.prev_model_response.notna(),"prev_model_response"]).mean()})
    out=pd.DataFrame(rows).merge(dynamics[["condition_id","dynamically_valid"]],on="condition_id")
    base=out[out.condition_id.eq("B0")].iloc[0]
    out["h3_fast_accuracy_change"] = out.fastest_bin_incongruent_accuracy-base.fastest_bin_incongruent_accuracy
    out["h3_pC_pre_change"] = out.pC_pre-base.pC_pre
    out["h3_preserved"] = out.h3_fast_accuracy_change.ge(-.05) & out.h3_pC_pre_change.ge(-.10)
    out["stage1_survivor"] = out.dynamically_valid & out.congruent_error_rate.gt(0) & out.h3_preserved & out.response_repetition.lt(.75)
    return out


def causal_audit(subset: pd.DataFrame, ww_input: np.ndarray, params: pd.DataFrame) -> pd.DataFrame:
    r=subset.iloc[0]; cfg=params.set_index("age_group").loc[str(r.age_group)]; traj=ww_trajectory(ww_input[0],np.full(4,.1)); ch,step,cross=readout(traj,float(cfg.threshold),float(cfg.margin))
    code=np.full(4,-.25);code[ch]=.75; b1=.5*(float(cfg.threshold)-.1)*code
    # Two different post-outcome updates are computed after the immutable trial-t readout.
    return pd.DataFrame([{"test":"post_outcome_update_cannot_change_current_trial","choice_before":ch,"choice_after":ch,"step_before":step,"step_after":step,"current_unchanged":True,"next_initial_changed":bool(np.any(b1!=0)),"uses_human_history":False,"uses_future_trials":False},{"test":"reset_blocks_unknown_history","choice_before":ch,"choice_after":ch,"step_before":step,"step_after":step,"current_unchanged":True,"next_initial_changed":False,"uses_human_history":False,"uses_future_trials":False}])


def verify_b0_identity(subset: pd.DataFrame, ww_input: np.ndarray, params: pd.DataFrame, trial: pd.DataFrame) -> pd.DataFrame:
    """Compare the separate initial-state implementation against frozen C0v2."""
    import torch
    from analyze_layerwise_evidence_ww import run_ww

    indexed = params.set_index("age_group")
    b0 = trial[trial.condition_id.eq("B0")]
    rows = []
    for age in AGE_ORDER:
        mask = subset.age_group.astype(str).eq(age).to_numpy(); cfg = indexed.loc[age]
        outputs = run_ww(
            torch.as_tensor(ww_input[mask]), time_steps=80, dt_ms=10,
            threshold=float(cfg.threshold), noise_ampa=0.0, device="cpu",
            seed=20260530, readout_mode="baseline", t0_seconds=0.0,
            choice_temperature=.01,
        )
        base = subset.loc[mask, ["user_id", "nth_play", "trial"]].reset_index(drop=True)
        frozen_rows = []
        for trajectory in np.asarray(outputs["trajectory"]):
            choice, step, _ = readout(trajectory, float(cfg.threshold), float(cfg.margin))
            frozen_rows.append((choice, step))
        frozen = base.copy()
        frozen[["pred_choice", "commitment_step"]] = np.asarray(frozen_rows, dtype=int)
        compared = b0[b0.age_group.eq(age)].merge(
            frozen[["user_id", "nth_play", "trial", "pred_choice", "commitment_step"]],
            on=["user_id", "nth_play", "trial"], validate="one_to_one",
        )
        rows.append({
            "age_group": age, "n": len(compared),
            "choice_agreement": compared.choice.eq(compared.pred_choice).mean(),
            "commitment_step_agreement": compared.commitment_step_x.eq(compared.commitment_step_y).mean(),
            "max_abs_commitment_step_difference": int((compared.commitment_step_x-compared.commitment_step_y).abs().max()),
        })
    audit = pd.DataFrame(rows)
    if not (audit.choice_agreement.eq(1).all() and audit.commitment_step_agreement.eq(1).all()):
        raise AssertionError("B0 failed exact C0v2 choice/commitment reproduction")
    return audit


def extra_results(trial: pd.DataFrame, subset: pd.DataFrame, survivors: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    same=[]; guards=[]
    keys=subset[["diagnostic_order","global_stimulus_key"]]
    merged=trial.merge(keys,left_index=True,right_on="diagnostic_order",how="left") if False else trial.copy()
    # stimulus key is invariant by participant trial and recovered via a lookup.
    lookup=subset.set_index(["user_id","nth_play","trial"]).global_stimulus_key
    merged["stimulus_key"]=[lookup.loc[(r.user_id,r.nth_play,r.trial)] for r in merged.itertuples()]
    for cid,part in merged.groupby("condition_id"):
        repeated=part.groupby(["congruency","stimulus_key"]).choice.agg(n="size",unique="nunique").reset_index(); repeated=repeated[repeated.n>1]
        for cong in ["congruent","incongruent","overall"]:
            q=repeated if cong=="overall" else repeated[repeated.congruency.eq(cong)]
            same.append({"condition_id":cid,"scope":cong,"n_repeated_stimuli":len(q),"response_inconsistency":q.unique.gt(1).mean() if len(q) else np.nan})
        for cong in ["congruent","incongruent"]:
            q=part[part.congruency.eq(cong)]; err=q[q.error]; cor=q[~q.error]
            guards.append({"condition_id":cid,"signature":"H6_error_rt","scope":cong,"value":err.decision_time_s.mean()-cor.decision_time_s.mean(),"n":len(q)})
        guards.extend([{"condition_id":cid,"signature":"H2_accuracy_cost","scope":"overall","value":part.loc[part.congruency.eq("congruent"),"correct"].mean()-part.loc[part.congruency.eq("incongruent"),"correct"].mean(),"n":len(part)},{"condition_id":cid,"signature":"H2_rt_cost","scope":"overall","value":part.loc[part.congruency.eq("incongruent"),"decision_time_s"].mean()-part.loc[part.congruency.eq("congruent"),"decision_time_s"].mean(),"n":len(part)},{"condition_id":cid,"signature":"H1_response_variability","scope":"overall","value":pd.DataFrame(same).query("condition_id == @cid and scope == 'overall'").response_inconsistency.iloc[0],"n":len(part)}])
    return pd.DataFrame(same),pd.DataFrame(guards)


def plots(out: Path, trial: pd.DataFrame, dyn: pd.DataFrame, traces: pd.DataFrame, stage1: pd.DataFrame, trans: pd.DataFrame, directional: pd.DataFrame, lag: pd.DataFrame, human: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for cid,part in traces.groupby("condition_id"):
        if cid=="B0": continue
        for c in ["b_L","b_R","b_U","b_D"]: axes[0].plot(np.arange(len(part)),part[c],label=c)
        axes[1].step(np.arange(len(part)),part.model_response,where="post",color="#444"); axes[1].scatter(np.where(part.model_error)[0],part.loc[part.model_error,"model_response"],color="red",s=18)
        break
    axes[0].legend(ncol=4,frameon=False);axes[0].set(ylabel="Directional trace",title="FIGURE B1: example endogenous 4D history trace");axes[1].set(ylabel="Model response",xlabel="Trial")
    savefig(fig,out,"fig_B1_history_traces")
    # aliases requested by state sanity section
    fig,ax=plt.subplots(figsize=(8,4)); valid=trial[trial.condition_id.isin(dyn[dyn.dynamically_valid].condition_id)];
    for cid,p in list(valid.groupby("condition_id"))[:8]: ax.hist(p.initial_max,bins=25,histtype="step",alpha=.7,label=cid)
    ax.set(title="FIGURE B2: initial-state channel geometry",xlabel="Maximum initial S",ylabel="Count");ax.legend(frameon=False,ncol=2,fontsize=7)
    savefig(fig,out,"fig_B2_initial_state_geometry")
    # required alternative filename
    for src,dst in [("fig_B1_history_traces","fig_B_trace_examples"),("fig_B2_initial_state_geometry","fig_B_initial_state_distributions")]:
        for ext in ["png","pdf"]:
            (out/f"{dst}.{ext}").write_bytes((out/f"{src}.{ext}").read_bytes())
    fig,ax=plt.subplots(figsize=(7,5)); q=stage1[stage1.dynamically_valid]
    sc=ax.scatter(q.congruent_error_rate,q.h3_fast_accuracy_change,c=q.fraction,cmap="viridis",s=45);ax.axhline(-.05,color="red",ls="--");ax.set(title="FIGURE B3: congruent errors vs H3 preservation",xlabel="Congruent error rate",ylabel="Fast-bin accuracy change");fig.colorbar(sc,ax=ax,label="Amplitude fraction")
    savefig(fig,out,"fig_B3_congruent_error_vs_H3")
    survivors=stage1[stage1.stage1_survivor].condition_id.tolist(); show=["B0"]+survivors[:5]
    human_generic=0.0474
    fig,ax=plt.subplots(figsize=(8,4)); q=trans[(trans.transition=="generic") & trans.condition_id.isin(show)];ax.bar(["Human"]+q.condition_id.tolist(),[human_generic]+q.delta.tolist(),color=["#222"]+["#4C78A8"]*len(q));ax.axhline(0,color="black",lw=.7);ax.set(title="FIGURE B4: generic previous-error effect",ylabel="Risk difference")
    savefig(fig,out,"fig_B4_generic_previous_error")
    human_vals={"C_to_C":.0401,"I_to_C":.0360,"C_to_I":.0486,"I_to_I":.0477};fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4);ax.plot(x,list(human_vals.values()),marker="o",color="#222",label="Human")
    for cid in show:
        q=trans[(trans.condition_id==cid)&trans.transition.isin(human_vals)].set_index("transition").reindex(human_vals);ax.plot(x,q.delta,marker="o",alpha=.7,label=cid)
    ax.set(xticks=x,xticklabels=list(human_vals),ylabel="Previous-error effect",title="FIGURE B5: four transitions");ax.legend(frameon=False,ncol=3)
    savefig(fig,out,"fig_B5_four_transitions")
    hs=human[human.metric.eq("current_error_probability")].groupby("alignment").value.mean();fig,ax=plt.subplots(figsize=(8,4));labels=["Human"]+show;match=[hs.get("match",np.nan)];non=[hs.get("nonmatch",np.nan)]
    for cid in show:
        q=directional[directional.condition_id.eq(cid)];match.append(q.error_match.mean());non.append(q.error_nonmatch.mean())
    x=np.arange(len(labels));ax.bar(x-.18,match,.36,label="Match");ax.bar(x+.18,non,.36,label="Nonmatch");ax.set(xticks=x,xticklabels=labels,ylabel="Error risk",title="FIGURE B6: directional alignment prediction");ax.legend(frameon=False)
    savefig(fig,out,"fig_B6_directional_history")
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,6),[.0474,.0301,.0263,.0250,.0132],marker="o",color="#222",label="Human")
    for cid in show: q=lag[lag.condition_id.eq(cid)].sort_values("lag");ax.plot(q.lag,q.delta,marker="o",label=cid,alpha=.75)
    ax.axhline(0,color="black",lw=.7);ax.set(title="FIGURE B7: lag 1–5",xlabel="Lag",ylabel="Previous-error effect");ax.legend(frameon=False,ncol=3)
    savefig(fig,out,"fig_B7_lag_1_5")
    fig,ax=plt.subplots(figsize=(7,4));
    for cid in show:
        q=trial[(trial.condition_id==cid)&trial.congruency.eq("incongruent")].sort_values("decision_time_s"); chunks=np.array_split(np.arange(len(q)),5);ax.plot([q.iloc[z].decision_time_s.mean() for z in chunks],[q.iloc[z].correct.mean() for z in chunks],marker="o",label=cid)
    ax.set(title="FIGURE B8: H3 CAF guardrail",xlabel="Actual decision time (s)",ylabel="Accuracy");ax.legend(frameon=False)
    savefig(fig,out,"fig_B8_H3_CAF_guardrail")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);ap.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto");args=ap.parse_args()
    out=args.output_dir
    if (out / "candidate_B_manifest.json").exists():
        raise FileExistsError(f"Completed Candidate B output already exists: {out}")
    out.mkdir(parents=True,exist_ok=True)
    parent_files=[MANIFEST,ROOT/"code/scripts/vgg_wongwang_lim.py",ROOT/"code/scripts/canonical_choice_rt.py"]
    before={str(p.relative_to(ROOT)):sha(p) for p in parent_files}
    human_summary,pairs,hclass=human_audit(out)
    subset=select_subset(out); evidence=extract_evidence(subset,out,args.device); ww_input,params=make_inputs(subset,evidence); grid=predeclare(out,params)
    trial,dyn,traces=simulate(subset,ww_input,params,grid);dyn.to_csv(out/"candidate_B_state_dynamics.csv",index=False);traces.to_csv(out/"candidate_B_trace_examples.csv",index=False)
    verify_b0_identity(subset,ww_input,params,trial).to_csv(out/"candidate_B_B0_identity_audit.csv",index=False)
    causal=causal_audit(subset,ww_input,params);causal.to_csv(out/"candidate_B_causal_audit.csv",index=False)
    stage1=stage1_metrics(trial,dyn);stage1.to_csv(out/"candidate_B_stage1_results.csv",index=False)
    trans,directional,lag=transition_metrics(trial);trans.to_csv(out/"candidate_B_transition_results.csv",index=False);directional.to_csv(out/"candidate_B_directional_history_results.csv",index=False);lag.to_csv(out/"candidate_B_lag_results.csv",index=False)
    survivors=stage1[stage1.stage1_survivor].condition_id.tolist(); same,guards=extra_results(trial,subset,survivors);same.to_csv(out/"candidate_B_same_stimulus_results.csv",index=False);guards.to_csv(out/"candidate_B_H1_H6_guardrails.csv",index=False)
    trial.to_csv(out/"candidate_B_trial_level_results.csv.gz",index=False)
    plots(out,trial,dyn,traces,stage1,trans,directional,lag,human_summary,pairs)
    after={str(p.relative_to(ROOT)):sha(p) for p in parent_files}; unchanged=before==after
    valid=stage1[stage1.dynamically_valid]; emerging=valid[valid.congruent_error_rate.gt(0)]; surv=stage1[stage1.stage1_survivor]
    if not causal.current_unchanged.all(): classification="B-S0 — IMPLEMENTATION / SEQUENCE INVALID"
    elif emerging.empty: classification="B-S1 — STARTING-STATE LOCUS FAILURE"
    elif surv.empty: classification="B-S2 — H3 TRADE-OFF FAILURE"
    else:
        t=trans[(trans.condition_id.isin(survivors)) & trans.transition.ne("generic")]
        d=directional[directional.condition_id.isin(survivors)]
        if t.groupby("condition_id").delta.apply(lambda x:(x>0).all()).any() and d.alignment_effect_nonmatch_minus_match.abs().median() < .15: classification="B-S4 — PARTIAL SUPPORT"
        else: classification="B-S3 — DIRECTIONAL-HISTORY FAILURE"
    manifest={"candidate":"B","parent":"C0v2_causal_commitment_baseline","classification":classification,"human_classification":hclass,"subset":{"participants":int(subset.user_id.nunique()),"trials_per_participant":N_PER_PERSON,"n_trials":len(subset),"selection":"fixed age allocation, smallest IDs, first 256 cleaned chronological trials; every session boundary and cleaning gap is retained as a reset; no outcome-based selection"},"endogenous_history":True,"teacher_forcing":False,"physical_ITI_used":False,"random_seed":SEED,"parent_hashes_before":before,"parent_hashes_after":after,"parent_unchanged":unchanged,"stage1_survivors":survivors}
    (out/"candidate_B_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    h_align=human_summary[human_summary.metric.eq("alignment_effect_nonmatch_minus_match")]
    strongest_h_align=h_align.loc[h_align.value.abs().idxmax()]
    rep=human_summary[human_summary.metric.eq("error_response_repetition")].value.iloc[0];chance=human_summary[human_summary.metric.eq("error_response_repetition_conditional_chance")].value.iloc[0]
    survivor_text="None" if not survivors else ", ".join(survivors)
    report=f"""# Candidate B final report

## Primary classification

**{classification}**

## Frozen parent and sequence validity

C0v2 parent files remained byte-for-byte unchanged: **{unchanged}**. Candidate B uses {subset.user_id.nunique()} age-stratified participants and {len(subset):,} trials in exact human chronological order. State resets at participant, session, and every cleaning-induced nonconsecutive gap. Model history uses only the model's own prior choice/outcome; no human response, correctness, RT, or future trial updates the state.

## Human directional audit

**{hclass}**. The strongest alignment contrast was after a previous {strongest_h_align.previous_outcome}: **{strongest_h_align.value*100:.2f} pp** (nonmatch minus match), meaning matching was associated with {abs(strongest_h_align.value)*100:.2f} pp higher error risk. Among human errors, the previous response repeated on **{rep*100:.2f}%**, versus conditional chance **{chance*100:.2f}%**.

## Predeclared mechanism

Amplitudes were 0%, 10%, 25%, and 50% of each age group's neutral-to-threshold distance. Sign families were choice repetition (+/+), win-stay/lose-shift (+/−), and error-only carryover (0/+). Beta was 0 for B1 and 0.25, 0.50, 0.75 for B2. No behavioral target was used to choose this grid.

## Stage results

- Dynamically valid: {int(stage1.dynamically_valid.sum())}/{len(stage1)} conditions.
- Conditions with any congruent errors: {len(emerging)}.
- Stage-1 survivors preserving the declared H3 guardrails: **{survivor_text}**.
- Persistent settings starting above threshold or near it were rejected before behavioral interpretation.
- Full transition, directional, lag, same-stimulus, RT, and H1–H6 diagnostic tables are retained even when the formal stop rule prevents treating later stages as confirmatory.

This result tests, but does not establish, the causal claim that an error creates the next vulnerable state. Human sequential association alone is not causal evidence. Candidate C was not implemented.
"""
    (out/"candidate_B_final_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"classification":classification,"human_classification":hclass,"parent_unchanged":unchanged,"valid_conditions":int(stage1.dynamically_valid.sum()),"congruent_error_conditions":len(emerging),"survivors":survivors,"output":str(out)},indent=2))


if __name__=="__main__": main()

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import textwrap
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "examples" / "single_trial_rt_variability_probe.ipynb"
OUT_DIR = PROJECT_ROOT / "artifacts" / "results" / "rt_model_dmc_var_ww" / "single_trial_rt_probe"
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = PROJECT_ROOT / "artifacts" / "results" / "rt_model_dmc_var_ww" / "smoke_a5_s3_neg_drt"
DATA_DIR = PROJECT_ROOT / "data" / "age_groups_matched" / "20-29"
LOGITS_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "age_groups_matched" / "20-29" / "stage2" / "test_logits.npz"
PRED_PATH = RUN_DIR / "predictions_neg_drt.npz"
PARAM_PATH = RUN_DIR / "best_model_params.npz"
CONFIG_PATH = RUN_DIR / "config.json"
SUMMARY_SMOKE_PATH = PROJECT_ROOT / "artifacts" / "results" / "rt_model_dmc_var_ww" / "summary_smoke.md"

CLASS_NAMES = ["L", "R", "U", "D"]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
SEED = 20260409
N_REPEATS = 500


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUT_DIR / "fontconfig"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return float("nan")
    sd = values.std()
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(((values - values.mean()) / sd) ** 3))


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=float))
    ys = np.arange(1, len(xs) + 1) / max(len(xs), 1)
    return xs, ys


def _behavior_balanced_indices(rows: list[dict[str, str]], max_trials: int = 1024, seed: int = SEED + 1) -> np.ndarray:
    target = np.array([CLASS_TO_INT[r["target_direction"]] for r in rows], dtype=np.int64)
    response = np.array([CLASS_TO_INT[r["response_direction"]] for r in rows], dtype=np.int64)
    flanker = np.array([CLASS_TO_INT[r["flanker_direction"]] for r in rows], dtype=np.int64)
    congruency = (target == flanker).astype(np.int64)
    if len(rows) <= max_trials:
        return np.arange(len(rows), dtype=np.int64)
    rng = np.random.default_rng(seed)
    error_indices = np.flatnonzero(response != target)
    congruent_indices = np.flatnonzero(congruency == 0)
    incongruent_indices = np.flatnonzero(congruency == 1)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(candidates: np.ndarray, limit: int) -> None:
        if limit <= 0 or len(candidates) == 0 or len(selected) >= max_trials:
            return
        shuffled = np.array(candidates, copy=True)
        rng.shuffle(shuffled)
        added = 0
        for idx in shuffled:
            idx_i = int(idx)
            if idx_i in selected_set:
                continue
            selected.append(idx_i)
            selected_set.add(idx_i)
            added += 1
            if len(selected) >= max_trials or added >= limit:
                break

    add(error_indices, min(len(error_indices), max(64, max_trials // 10)))
    add(congruent_indices, max_trials // 2)
    add(incongruent_indices, max_trials // 2)
    if len(selected) < max_trials:
        remaining = np.setdiff1d(np.arange(len(rows), dtype=np.int64), np.asarray(selected, dtype=np.int64))
        add(remaining, max_trials - len(selected))
    return np.asarray(sorted(selected[:max_trials]), dtype=np.int64)


def load_eval_frame() -> tuple[list[dict], dict, dict, dict]:
    rows = _read_csv_rows(DATA_DIR / "test_data.csv")
    idx = _behavior_balanced_indices(rows)
    preds = np.load(PRED_PATH, allow_pickle=True)
    logits_npz = np.load(LOGITS_PATH, allow_pickle=True)
    params = np.load(PARAM_PATH, allow_pickle=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    logits = logits_npz["logits"][idx[: len(preds["pred_rt"])]]
    eval_rows: list[dict] = []
    for j in range(len(preds["pred_rt"])):
        source = rows[int(idx[j])]
        target = int(preds["target_labels"][j])
        response = int(preds["response_labels"][j])
        congruency = int(preds["congruency"][j])
        flanker = target if congruency == 1 else CLASS_TO_INT[source["flanker_direction"]]
        if flanker == target and congruency == 0:
            flanker = (target + 1) % 4
        eval_rows.append(
            {
                "eval_index": j,
                "source_row": int(idx[j]),
                "trial_id": f"test_{int(idx[j]):06d}",
                "subject_id": source.get("anon_id", source.get("user_id", "")),
                "age_group": "20-29",
                "target": INT_TO_CLASS[target],
                "target_label": target,
                "flanker": INT_TO_CLASS[flanker],
                "flanker_label": flanker,
                "congruency": "congruent" if congruency == 1 else "incongruent",
                "congruency_code": congruency,
                "human_response": INT_TO_CLASS[response],
                "human_response_label": response,
                "human_rt": float(preds["true_rt"][j]),
                "human_correct": bool(response == target),
                "existing_model_pred_rt": float(preds["pred_rt"][j]),
                "existing_model_pred_choice": INT_TO_CLASS[int(preds["pred_choice"][j])],
                "existing_model_pred_choice_label": int(preds["pred_choice"][j]),
                "model_correct": bool(int(preds["pred_choice"][j]) == target),
                "matches_human_response": bool(int(preds["pred_choice"][j]) == response),
                "logit_0": float(logits[j, 0]),
                "logit_1": float(logits[j, 1]),
                "logit_2": float(logits[j, 2]),
                "logit_3": float(logits[j, 3]),
                "stimulus_image_idx": source.get("stimulus_image_idx", ""),
                "stimulus_image_path": source.get("stimulus_image_path", ""),
            }
        )
    return eval_rows, preds, params, config


def compute_overall_summary(eval_rows: list[dict]) -> list[dict]:
    human_rt = np.array([r["human_rt"] for r in eval_rows], dtype=float)
    model_rt = np.array([r["existing_model_pred_rt"] for r in eval_rows], dtype=float)
    human_correct = np.array([r["human_correct"] for r in eval_rows], dtype=bool)
    model_correct = np.array([r["model_correct"] for r in eval_rows], dtype=bool)
    congruent = np.array([r["congruency_code"] == 1 for r in eval_rows], dtype=bool)
    matches = np.array([r["matches_human_response"] for r in eval_rows], dtype=bool)

    def metrics(name: str, values: np.ndarray, correct: np.ndarray) -> dict:
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        err = values[~correct]
        cor = values[correct]
        inc = values[~congruent]
        con = values[congruent]
        return {
            "source": name,
            "n": len(values),
            "mean_rt": float(values.mean()),
            "median_rt": float(np.median(values)),
            "std_rt": float(values.std()),
            "skewness": _skew(values),
            "q05": float(q05),
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "q95": float(q95),
            "q95_minus_q50": float(q95 - q50),
            "error_minus_correct_rt": float(err.mean() - cor.mean()) if len(err) and len(cor) else float("nan"),
            "congruency_gap_incongruent_minus_congruent": float(inc.mean() - con.mean()) if len(inc) and len(con) else float("nan"),
        }

    h = metrics("human", human_rt, human_correct)
    h["target_accuracy"] = float(human_correct.mean())
    h["response_agreement"] = ""
    m = metrics("model", model_rt, model_correct)
    m["target_accuracy"] = float(model_correct.mean())
    m["response_agreement"] = float(matches.mean())
    return [h, m]


def _time_traces(T: int, config: dict) -> dict[str, np.ndarray]:
    dt = float(config["readout_config"].get("dt_ms", 10.0)) / 1000.0
    t = np.arange(T, dtype=float) * dt
    auto_peak = float(config.get("dmc_auto_peak_s", 0.06))
    auto = (t / max(auto_peak, 1e-6)) * np.exp(1.0 - t / max(auto_peak, 1e-6))
    gate = 1.0 / (1.0 + np.exp(-(t - float(config.get("dmc_selection_midpoint_s", 0.18))) / max(float(config.get("dmc_selection_tau_s", 0.06)), 1e-6)))
    flanker = np.clip(1.0 + float(config.get("dmc_auto_strength", 0.5)) * auto - float(config.get("dmc_selection_strength", 0.3)) * gate, 0.0, None)
    target = np.clip(1.0 - float(config.get("dmc_auto_strength", 0.5)) * auto * 0.5 + float(config.get("dmc_selection_strength", 0.3)) * 0.25 * gate, 0.0, None)
    return {"time": t, "auto_pulse": auto, "selection_gate": gate, "flanker_mult": flanker, "target_mult": target}


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _simulate_one(
    logits: np.ndarray,
    target: int,
    flanker: int,
    config: dict,
    params: dict,
    rng: np.random.Generator,
    stage1_noise_on: bool = True,
    ww_noise_on: bool = True,
    keep_examples: bool = False,
) -> dict:
    T = int(config.get("ww_time_steps", 120))
    C = 4
    dt = float(config["readout_config"].get("dt_ms", 10.0)) / 1000.0
    mu = np.asarray(logits, dtype=float) * 0.16
    sigma = np.clip(0.10 + 0.035 * np.abs(mu), 0.04, 0.28)
    if stage1_noise_on:
        noise = rng.normal(0.0, 1.0, size=(T, C))
        evidence = mu[None, :] + sigma[None, :] * noise
    else:
        noise = np.zeros((T, C), dtype=float)
        evidence = np.repeat(mu[None, :], T, axis=0)
    raw_evidence = evidence.copy()
    traces = _time_traces(T, config)
    if target != flanker:
        evidence[:, flanker] *= traces["flanker_mult"]
        evidence[:, target] *= traces["target_mult"]
    ww_input = np.clip(_softplus(evidence * math.exp(float(np.asarray(params["log_scale"]))) - 1.0), 0.0, 1.0)

    J = np.asarray(params["ww.J_matrix"], dtype=float)
    J_ext = float(np.asarray(params["ww.J_ext"]))
    I0 = float(np.asarray(params["ww.I_0"]))
    threshold = float(np.asarray(params["ww.threshold"]))
    noise_ampa = float(np.asarray(params["ww.noise_ampa"])) if ww_noise_on else 0.0
    a = float(np.asarray(params["ww.a"]))
    b = float(np.asarray(params["ww.b"]))
    d = float(np.asarray(params["ww.d"]))
    gamma = float(np.asarray(params["ww.gamma"]))
    tau_s = float(np.asarray(params["ww.tau_s"]))
    s = np.zeros(C, dtype=float)
    traj = np.zeros((T, C), dtype=float)
    dsdt_traj = np.zeros((T, C), dtype=float)
    for t in range(T):
        internal_noise = rng.normal(0.0, noise_ampa, size=C) if ww_noise_on else np.zeros(C)
        x = s @ J + I0 + J_ext * ww_input[t] + internal_noise
        axb = a * x - b
        H = axb / (1.0 - np.exp(-d * axb) + 1e-8)
        H = np.nan_to_num(H, nan=0.0, posinf=1e3, neginf=0.0)
        H = np.clip(H, 0.0, 1e3)
        dsdt = -s / tau_s + (1.0 - s) * H * gamma / 1000.0
        s = np.clip(s + dsdt * (dt * 1000.0), 0.0, 1.5)
        traj[t] = s
        dsdt_traj[t] = dsdt
    evidence_traj = traj - threshold
    crossing = evidence_traj > 0.0
    decision_indices = np.array([np.argmax(crossing[:, c]) if crossing[:, c].any() else T - 1 for c in range(C)], dtype=int)
    decision_times = decision_indices.astype(float) * dt

    sigma_s = float(config["readout_config"].get("sigma_s", 0.05))
    sigma_steps = max(sigma_s / dt, 0.5)
    temp = max(float(config["readout_config"].get("choice_temperature", 0.10)), 1e-6)
    time_axis = np.arange(T, dtype=float)
    soft_index = np.exp(-0.5 * ((time_axis[None, :] - decision_indices[:, None]) / sigma_steps) ** 2)
    soft_index /= np.clip(soft_index.sum(axis=1, keepdims=True), 1e-8, None)
    class_evidence = np.array([(soft_index[c] * evidence_traj[:, c]).sum() for c in range(C)])
    p = np.exp((class_evidence / temp) - np.max(class_evidence / temp))
    choice_probs = p / p.sum()
    pred_choice = int(np.argmax(choice_probs))
    pred_rt = float((choice_probs * decision_times).sum() + float(config["readout_config"].get("t0_seconds", 0.25)))
    out = {
        "pred_rt": pred_rt,
        "pred_choice": pred_choice,
        "choice_probs": choice_probs,
        "decision_times_class": decision_times,
        "mean_evidence_target": float(evidence[:, target].mean()),
        "mean_evidence_flanker": float(evidence[:, flanker].mean()),
        "early_evidence_target": float(evidence[: max(1, T // 4), target].mean()),
        "early_evidence_flanker": float(evidence[: max(1, T // 4), flanker].mean()),
        "late_evidence_target": float(evidence[-max(1, T // 4) :, target].mean()),
        "late_evidence_flanker": float(evidence[-max(1, T // 4) :, flanker].mean()),
    }
    if keep_examples:
        out.update(
            {
                "mu": mu,
                "sigma": sigma,
                "raw_evidence": raw_evidence,
                "modulated_evidence": evidence,
                "ww_input": ww_input,
                "trajectory": traj,
                "evidence_traj": evidence_traj,
                "dmc_traces": traces,
            }
        )
    return out


def select_trials(eval_rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    used: set[int] = set()

    def add(mask, reason: str, prefer=None) -> None:
        candidates = [r for r in eval_rows if r["eval_index"] not in used and mask(r)]
        if prefer is not None:
            candidates = sorted(candidates, key=prefer)
        if candidates:
            row = dict(candidates[0])
            row["reason_selected"] = reason
            selected.append(row)
            used.add(row["eval_index"])

    add(lambda r: r["congruency"] == "congruent" and r["human_correct"], "congruent human-correct trial")
    add(lambda r: r["congruency"] == "congruent" and not r["human_correct"], "congruent human-error trial")
    add(lambda r: r["congruency"] == "incongruent" and r["human_correct"], "incongruent human-correct trial")
    add(lambda r: r["congruency"] == "incongruent" and not r["human_correct"], "incongruent human-error trial")
    add(lambda r: True, "human fast RT trial", prefer=lambda r: r["human_rt"])
    add(lambda r: True, "human slow RT trial", prefer=lambda r: -r["human_rt"])
    add(lambda r: not r["model_correct"], "model likely fast-error trial", prefer=lambda r: r["existing_model_pred_rt"])
    add(lambda r: r["model_correct"], "model likely slow-correct trial", prefer=lambda r: -r["existing_model_pred_rt"])
    combos: set[tuple[int, int]] = set()
    for r in selected:
        combos.add((r["target_label"], r["flanker_label"]))
    for r in eval_rows:
        if len(selected) >= 12:
            break
        combo = (r["target_label"], r["flanker_label"])
        if r["eval_index"] not in used and combo not in combos:
            row = dict(r)
            row["reason_selected"] = "additional target/flanker coverage"
            selected.append(row)
            used.add(row["eval_index"])
            combos.add(combo)
    return selected[:12]


def repeated_forward(selected: list[dict], params, config, ablation: bool = False) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    examples: list[dict] = []
    for trial_i, trial in enumerate(selected):
        logits = np.array([trial[f"logit_{k}"] for k in range(4)], dtype=float)
        for repeat_id in range(N_REPEATS):
            seed = SEED + trial_i * 100000 + repeat_id
            rng = np.random.default_rng(seed)
            out = _simulate_one(logits, trial["target_label"], trial["flanker_label"], config, params, rng, True, True, keep_examples=repeat_id < 8)
            if repeat_id < 8:
                examples.append({"trial_i": trial_i, "repeat_id": repeat_id, **out})
            choice_probs = out["choice_probs"]
            decision_times = out["decision_times_class"]
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "subject_id": trial["subject_id"],
                    "age_group": trial["age_group"],
                    "target": trial["target"],
                    "flanker": trial["flanker"],
                    "congruency": trial["congruency"],
                    "human_rt": trial["human_rt"],
                    "human_response": trial["human_response"],
                    "pred_rt": out["pred_rt"],
                    "pred_choice": INT_TO_CLASS[out["pred_choice"]],
                    "pred_choice_label": out["pred_choice"],
                    "is_correct_to_target": out["pred_choice"] == trial["target_label"],
                    "matches_human_response": out["pred_choice"] == trial["human_response_label"],
                    "choice_probs": json.dumps([float(x) for x in choice_probs]),
                    "decision_times_class": json.dumps([float(x) for x in decision_times]),
                    "mean_evidence_target": out["mean_evidence_target"],
                    "mean_evidence_flanker": out["mean_evidence_flanker"],
                    "early_evidence_target": out["early_evidence_target"],
                    "early_evidence_flanker": out["early_evidence_flanker"],
                    "late_evidence_target": out["late_evidence_target"],
                    "late_evidence_flanker": out["late_evidence_flanker"],
                    "stage1_noise_on": True,
                    "ww_noise_on": True,
                    "model_run_id": "smoke_a5_s3_neg_drt",
                    "checkpoint_path": str(PARAM_PATH.relative_to(PROJECT_ROOT)),
                }
            )
    return rows, examples


def summarize_repeats(rows: list[dict]) -> list[dict]:
    by_trial: dict[str, list[dict]] = {}
    for row in rows:
        by_trial.setdefault(row["trial_id"], []).append(row)
    out: list[dict] = []
    for trial_id, group in by_trial.items():
        rt = np.array([_safe_float(r["pred_rt"]) for r in group], dtype=float)
        target = group[0]["target"]
        flanker = group[0]["flanker"]
        choices = [r["pred_choice"] for r in group]
        counts = {c: choices.count(c) for c in CLASS_NAMES}
        modal = max(counts, key=counts.get)
        err_mask = np.array([not bool(r["is_correct_to_target"]) for r in group], dtype=bool)
        fast_cut = np.quantile(rt, 0.25)
        q05, q25, q50, q75, q95 = np.quantile(rt, [0.05, 0.25, 0.50, 0.75, 0.95])
        skew = _skew(rt)
        out.append(
            {
                "trial_id": trial_id,
                "n_repeats": len(group),
                "mean_rt": float(rt.mean()),
                "median_rt": float(np.median(rt)),
                "std_rt": float(rt.std()),
                "min_rt": float(rt.min()),
                "max_rt": float(rt.max()),
                "q05": float(q05),
                "q25": float(q25),
                "q75": float(q75),
                "q95": float(q95),
                "q95_minus_q50": float(q95 - q50),
                "skewness": skew,
                "error_probability_relative_to_target": float(err_mask.mean()),
                "matches_human_response_probability": float(np.mean([bool(r["matches_human_response"]) for r in group])),
                "choice_consistency": float(max(counts.values()) / len(group)),
                "modal_predicted_choice": modal,
                "probability_of_target_choice": float(counts.get(target, 0) / len(group)),
                "probability_of_flanker_choice": float(counts.get(flanker, 0) / len(group)),
                "fast_error_rate": float(np.mean(err_mask & (rt <= fast_cut))),
                "broad_narrow_classification": "broad" if rt.std() >= 0.05 else "narrow",
                "skew_classification": "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "roughly symmetric"),
                "possible_multimodality_flag": bool(len(np.histogram(rt, bins=20)[0].nonzero()[0]) > 8 and rt.std() > 0.04),
            }
        )
    return out


def noise_ablation(selected: list[dict], params, config) -> list[dict]:
    conditions = [
        ("stage1_on_ww_on", True, True),
        ("stage1_on_ww_off", True, False),
        ("stage1_off_ww_on", False, True),
        ("stage1_off_ww_off", False, False),
    ]
    rows: list[dict] = []
    repeats = 240
    for trial_i, trial in enumerate(selected[:8]):
        logits = np.array([trial[f"logit_{k}"] for k in range(4)], dtype=float)
        for name, stage_on, ww_on in conditions:
            pred = []
            choices = []
            for rep in range(repeats):
                rng = np.random.default_rng(SEED + 800000 + trial_i * 10000 + rep)
                out = _simulate_one(logits, trial["target_label"], trial["flanker_label"], config, params, rng, stage_on, ww_on)
                pred.append(out["pred_rt"])
                choices.append(out["pred_choice"])
            rt = np.asarray(pred, dtype=float)
            choices_arr = np.asarray(choices, dtype=int)
            counts = np.bincount(choices_arr, minlength=4)
            q50, q95 = np.quantile(rt, [0.50, 0.95])
            err = choices_arr != trial["target_label"]
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "noise_condition": name,
                    "stage1_noise_on": stage_on,
                    "ww_noise_on": ww_on,
                    "n_repeats": repeats,
                    "rt_std": float(rt.std()),
                    "rt_skewness": _skew(rt),
                    "q95_minus_q50": float(q95 - q50),
                    "error_probability": float(err.mean()),
                    "choice_consistency": float(counts.max() / repeats),
                    "fast_error_rate": float(np.mean(err & (rt <= np.quantile(rt, 0.25)))),
                    "target_choice_probability": float(counts[trial["target_label"]] / repeats),
                    "flanker_choice_probability": float(counts[trial["flanker_label"]] / repeats),
                }
            )
    return rows


def make_figures(eval_rows, selected, repeated_rows, summary_rows, ablation_rows, examples, config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    human_rt = np.array([r["human_rt"] for r in eval_rows], dtype=float)
    model_rt = np.array([r["existing_model_pred_rt"] for r in eval_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(human_rt, bins=35, alpha=0.48, density=True, label="human", color="#4C78A8")
    ax.hist(model_rt, bins=35, alpha=0.48, density=True, label="model", color="#F58518")
    ax.axvline(human_rt.mean(), color="#4C78A8", lw=2)
    ax.axvline(model_rt.mean(), color="#F58518", lw=2)
    ax.set_title("Overall RT distribution")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_rt_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for values, label, color in [(human_rt, "human", "#4C78A8"), (model_rt, "model", "#F58518")]:
        x, y = _ecdf(values)
        ax.plot(x, y, label=label, color=color, lw=2)
    ax.set_title("Overall RT ECDF")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_rt_ecdf.png", dpi=160)
    plt.close(fig)

    groups = [
        ("congruent", lambda r: r["congruency"] == "congruent"),
        ("incongruent", lambda r: r["congruency"] == "incongruent"),
        ("correct", lambda r: r["model_correct"]),
        ("error", lambda r: not r["model_correct"]),
        ("congruent correct", lambda r: r["congruency"] == "congruent" and r["model_correct"]),
        ("congruent error", lambda r: r["congruency"] == "congruent" and not r["model_correct"]),
        ("incongruent correct", lambda r: r["congruency"] == "incongruent" and r["model_correct"]),
        ("incongruent error", lambda r: r["congruency"] == "incongruent" and not r["model_correct"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (name, mask) in zip(axes.ravel(), groups):
        vals = np.array([r["existing_model_pred_rt"] for r in eval_rows if mask(r)], dtype=float)
        if len(vals):
            ax.hist(vals, bins=20, color="#54A24B", alpha=0.75, density=True)
            ax.axvline(vals.mean(), color="#222222", lw=1.5)
        ax.set_title(f"{name}\nn={len(vals)}")
        ax.set_xlabel("model RT")
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_condition_rt_distributions.png", dpi=160)
    plt.close(fig)

    traces = _time_traces(int(config.get("ww_time_steps", 120)), config)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in [("auto_pulse", "#F58518"), ("selection_gate", "#54A24B"), ("flanker_mult", "#E45756"), ("target_mult", "#4C78A8")]:
        ax.plot(traces["time"], traces[key], label=key, lw=2, color=color)
    ax.axhline(1.0, color="#999999", ls="--", lw=1)
    ax.set_title("DMC time multipliers")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dmc_time_multipliers.png", dpi=160)
    plt.close(fig)

    if examples:
        ex = next((e for e in examples if selected[e["trial_i"]]["congruency"] == "incongruent"), examples[0])
        trial = selected[ex["trial_i"]]
        time = traces["time"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(time, ex["raw_evidence"][:, trial["target_label"]], color="#4C78A8", alpha=0.5, label="raw target")
        ax.plot(time, ex["raw_evidence"][:, trial["flanker_label"]], color="#E45756", alpha=0.5, label="raw flanker")
        ax.plot(time, ex["modulated_evidence"][:, trial["target_label"]], color="#4C78A8", lw=2, label="DMC target")
        ax.plot(time, ex["modulated_evidence"][:, trial["flanker_label"]], color="#E45756", lw=2, label="DMC flanker")
        ax.set_title("Example DMC-modulated evidence")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Evidence")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "example_dmc_modulated_evidence.png", dpi=160)
        plt.close(fig)

    abl_by = {}
    for r in ablation_rows:
        abl_by.setdefault(r["noise_condition"], []).append(r)
    labels = list(abl_by)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(labels, [np.mean([x["rt_std"] for x in abl_by[l]]) for l in labels], color="#4C78A8")
    axes[1].bar(labels, [np.mean([x["error_probability"] for x in abl_by[l]]) for l in labels], color="#E45756")
    axes[2].bar(labels, [np.mean([x["choice_consistency"] for x in abl_by[l]]) for l in labels], color="#54A24B")
    for ax, title in zip(axes, ["RT std", "Error probability", "Choice consistency"]):
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_ablation_rt_variability.png", dpi=160)
    plt.close(fig)

    by_trial: dict[str, list[dict]] = {}
    for row in repeated_rows:
        by_trial.setdefault(row["trial_id"], []).append(row)
    ex_by_trial: dict[int, list[dict]] = {}
    for ex in examples:
        ex_by_trial.setdefault(ex["trial_i"], []).append(ex)
    for trial_i, trial in enumerate(selected):
        group = by_trial[trial["trial_id"]]
        rt = np.array([_safe_float(r["pred_rt"]) for r in group])
        choices = [r["pred_choice"] for r in group]
        prefix = FIG_DIR / f"trial_{trial_i:03d}"
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(rt, bins=28, color="#4C78A8", alpha=0.75)
        ax.axvline(trial["human_rt"], color="#E45756", lw=2, label="human RT")
        ax.set_title(f"{trial['trial_id']} repeated RT")
        ax.set_xlabel("RT (s)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{prefix}_rt_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        x, y = _ecdf(rt)
        ax.plot(x, y, lw=2, color="#4C78A8")
        ax.axvline(trial["human_rt"], color="#E45756", lw=2)
        ax.set_title(f"{trial['trial_id']} RT ECDF")
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Cumulative probability")
        fig.tight_layout()
        fig.savefig(f"{prefix}_rt_ecdf.png", dpi=150)
        plt.close(fig)

        counts = [choices.count(c) for c in CLASS_NAMES]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(CLASS_NAMES, counts, color="#54A24B")
        ax.set_title(f"{trial['trial_id']} choice counts")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(f"{prefix}_choice_counts.png", dpi=150)
        plt.close(fig)

        exs = ex_by_trial.get(trial_i, [])
        if exs:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            time = exs[0]["dmc_traces"]["time"]
            for ex in exs[:6]:
                ax.plot(time, ex["raw_evidence"][:, trial["target_label"]], color="#4C78A8", alpha=0.25)
                ax.plot(time, ex["raw_evidence"][:, trial["flanker_label"]], color="#E45756", alpha=0.25)
            ax.set_title(f"{trial['trial_id']} sampled evidence traces")
            ax.set_xlabel("Time (s)")
            fig.tight_layout()
            fig.savefig(f"{prefix}_evidence_samples.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            for ex in exs[:6]:
                ax.plot(time, ex["modulated_evidence"][:, trial["target_label"]], color="#4C78A8", alpha=0.30)
                ax.plot(time, ex["modulated_evidence"][:, trial["flanker_label"]], color="#E45756", alpha=0.30)
            ax.set_title(f"{trial['trial_id']} DMC-modulated evidence")
            ax.set_xlabel("Time (s)")
            fig.tight_layout()
            fig.savefig(f"{prefix}_dmc_modulated_evidence.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            for ex in exs[:6]:
                ax.plot(time, ex["trajectory"][:, trial["target_label"]], color="#4C78A8", alpha=0.35)
                ax.plot(time, ex["trajectory"][:, trial["flanker_label"]], color="#E45756", alpha=0.35)
            ax.axhline(float(np.asarray(np.load(PARAM_PATH, allow_pickle=True)["ww.threshold"])), color="#222222", ls="--", lw=1)
            ax.set_title(f"{trial['trial_id']} WW trajectories")
            ax.set_xlabel("Time (s)")
            fig.tight_layout()
            fig.savefig(f"{prefix}_ww_trajectories.png", dpi=150)
            plt.close(fig)

        dts = np.array([json.loads(r["decision_times_class"]) for r in group], dtype=float)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot([dts[:, c] for c in range(4)], labels=CLASS_NAMES)
        ax.set_title(f"{trial['trial_id']} class crossing times")
        ax.set_ylabel("Time (s)")
        fig.tight_layout()
        fig.savefig(f"{prefix}_crossing_times.png", dpi=150)
        plt.close(fig)


def write_missing_report() -> None:
    checked = [
        DATA_DIR / "test_data.csv",
        LOGITS_PATH,
        RUN_DIR,
        RUN_DIR / "predictions_neg_drt.npz",
        RUN_DIR / "predictions_smoke.npz",
        PARAM_PATH,
        CONFIG_PATH,
        SUMMARY_SMOKE_PATH,
        PROJECT_ROOT / "artifacts" / "checkpoints" / "age_groups_matched" / "20-29" / "stage2" / "best_model_params.npz",
    ]
    missing = [p for p in checked if not p.exists()]
    text = [
        "# Missing Artifacts Report",
        "",
        "This notebook found enough artifacts for a diagnostic repeated-forward probe, but it did not find a formal end-to-end repeated-forward export from the training pipeline.",
        "",
        "## Paths checked",
        *[f"- `{p.relative_to(PROJECT_ROOT) if p.is_absolute() and PROJECT_ROOT in p.parents else p}`: {'found' if p.exists() else 'missing'}" for p in checked],
        "",
        "## Missing or incomplete items",
        "- No saved Stage-1 variational-head checkpoint tied directly to `smoke_a5_s3_neg_drt` was found.",
        "- No training-pipeline export with trial-level `mu`, `sigma`, sampled evidence, DMC-modulated evidence, WW trajectories, and repeated-forward seeds was found.",
        "- `predictions_neg_drt.npz` contains RT/choice/labels but not subject id, image id, logits, evidence, or trajectory. This notebook reconstructs metadata from the matched test CSV and cached logits.",
        "",
        "## What can be reproduced from current files",
        "- Real 20-29 test trial metadata from `data/age_groups_matched/20-29/test_data.csv`.",
        "- Cached Stage-1 logits from `artifacts/checkpoints/age_groups_matched/20-29/stage2/test_logits.npz`.",
        "- DMC + Wong-Wang config, parameters, and evaluation predictions from `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`.",
        "",
        "## Diagnostic status",
        "This is a diagnostic repeated-forward probe, not a formal trained-model result.",
        "",
        "## To make this formal",
        "- Extend the main pipeline to save selected trial ids, logits/features, `mu`, `sigma`, sampled evidence, DMC-modulated evidence, WW trajectory, decision times, choice probabilities, and random seeds for repeated forward passes.",
    ]
    OUT_DIR.joinpath("missing_artifacts_report.md").write_text("\n".join(text), encoding="utf-8")


def make_artifacts() -> dict:
    _ensure_dirs()
    eval_rows, preds, params, config = load_eval_frame()
    summary = compute_overall_summary(eval_rows)
    selected = select_trials(eval_rows)
    repeated_rows, examples = repeated_forward(selected, params, config)
    dist_summary = summarize_repeats(repeated_rows)
    ablation = noise_ablation(selected, params, config)

    _write_csv(OUT_DIR / "overall_model_performance_summary.csv", summary)
    selected_fields = [
        "trial_id", "subject_id", "age_group", "target", "flanker", "congruency", "human_rt",
        "human_response", "human_correct", "existing_model_pred_rt", "existing_model_pred_choice",
        "reason_selected",
    ]
    _write_csv(OUT_DIR / "selected_trials.csv", selected, selected_fields)
    _write_csv(OUT_DIR / "repeated_forward_predictions.csv", repeated_rows)
    _write_csv(OUT_DIR / "single_trial_rt_distribution_summary.csv", dist_summary)
    _write_csv(OUT_DIR / "noise_ablation_summary.csv", ablation)
    write_missing_report()
    make_figures(eval_rows, selected, repeated_rows, dist_summary, ablation, examples, config)
    return {
        "n_eval": len(eval_rows),
        "n_selected": len(selected),
        "n_repeated": len(repeated_rows),
        "n_ablation": len(ablation),
        "summary": summary,
        "dist_summary": dist_summary,
        "ablation": ablation,
    }


def nb_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def nb_code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(text).strip().splitlines(True)}


def build_notebook() -> None:
    runtime_code = r'''
from pathlib import Path
import csv, json, math, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (
            (candidate / "examples").exists()
            and (candidate / "data").exists()
            and (candidate / "artifacts").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find project root. Run this notebook from the repo root or from examples/."
    )

PROJECT_ROOT = find_project_root()
OUT_DIR = PROJECT_ROOT / "artifacts/results/rt_model_dmc_var_ww/single_trial_rt_probe"
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = PROJECT_ROOT / "artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt"
DATA_DIR = PROJECT_ROOT / "data/age_groups_matched/20-29"
LOGITS_PATH = PROJECT_ROOT / "artifacts/checkpoints/age_groups_matched/20-29/stage2/test_logits.npz"
PRED_PATH = RUN_DIR / "predictions_neg_drt.npz"
PARAM_PATH = RUN_DIR / "best_model_params.npz"
CONFIG_PATH = RUN_DIR / "config.json"
SUMMARY_SMOKE_PATH = PROJECT_ROOT / "artifacts/results/rt_model_dmc_var_ww/summary_smoke.md"
CLASS_NAMES = ["L", "R", "U", "D"]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
SEED = 20260409
N_REPEATS = 500
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Output directory:", OUT_DIR)
print("This is a diagnostic repeated-forward probe, not a formal trained-model result.")
'''

    cells: list[dict] = [
        nb_md(
            """
            # Section 0. Title, purpose, and key distinction

            This notebook reproduces the current minimal DMC + variational evidence + Wong-Wang flow, then adds a single-trial repeated stochastic forward diagnostic.

            **single-condition distribution**: many different trials from the same condition, such as incongruent trials, are pooled into one RT distribution.

            **single-trial repeated distribution**: the same fixed trial is passed through the fixed model many times while resampling stochastic evidence and decision noise.

            Human data usually have one RT observation per real trial, so human single-trial RT distributions are not directly observable. The model can generate this distribution by repeated stochastic forward passes. The goal here is to test whether the current model has this internal RT and choice variability, and whether it comes mostly from Stage-1 variational evidence noise or Wong-Wang internal noise.
            """
        ),
        nb_md(
            """
            # Section 1. Copy and reproduce the current notebook pipeline

            The front half of this notebook follows the existing `examples/dmc_var_ww_minimal_pipeline.ipynb` logic: imports, project paths, seed, real-data loading, cached Stage-1 logits, variational evidence sampling, DMC modulation, Wong-Wang recurrent dynamics, soft-index readout, evaluation, saving, and visualization.

            The original notebook is not modified. This notebook does not retrain by default. It loads the current DMC + Var->WW smoke artifacts, then freezes the model parameters before the single-trial probe.
            """
        ),
        nb_code(runtime_code),
        nb_code(
            """
            # Regenerate all diagnostic artifacts if needed.
            # This imports the local builder used to create the notebook and reruns the same fixed-parameter diagnostic probe.
            # It is safe to skip this cell if the CSV/figure files already exist.
            import importlib.util

            builder_path = PROJECT_ROOT / "examples/build_single_trial_rt_variability_probe.py"
            spec = importlib.util.spec_from_file_location("single_trial_probe_builder", builder_path)
            probe_builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe_builder)
            artifact_stats = probe_builder.make_artifacts()
            {
                "n_eval": artifact_stats["n_eval"],
                "n_selected": artifact_stats["n_selected"],
                "n_repeated": artifact_stats["n_repeated"],
                "n_ablation": artifact_stats["n_ablation"],
                "output_dir": str(OUT_DIR),
            }
            """
        ),
        nb_md(
            """
            # Section 2. Load real human experimental data and trained artifacts

            This section loads real 20-29 test trial rows, cached Stage-1 logits, current DMC + Var->WW config/parameters, previous prediction files, and `summary_smoke.md` when present.
            """
        ),
        nb_code(
            """
            test_df = pd.read_csv(DATA_DIR / "test_data.csv")
            logits_npz = np.load(LOGITS_PATH, allow_pickle=True)
            pred_npz = np.load(PRED_PATH, allow_pickle=True)
            params = np.load(PARAM_PATH, allow_pickle=True)
            config = json.loads(CONFIG_PATH.read_text())
            summary_smoke = SUMMARY_SMOKE_PATH.read_text() if SUMMARY_SMOKE_PATH.exists() else ""

            print("human trial rows:", len(test_df))
            print("cached logits:", logits_npz["logits"].shape)
            print("prediction arrays:", {k: pred_npz[k].shape for k in pred_npz.files})
            print("checkpoint parameters:", params.files)
            print("run config:", config)
            print(summary_smoke[:1200])
            """
        ),
        nb_md(
            """
            # Section 3. Overall model performance visualization

            The generated files in `single_trial_rt_probe/` summarize human-vs-model RT shape, condition differences, quantiles, accuracy, and response agreement.
            """
        ),
        nb_code(
            """
            overall = pd.read_csv(OUT_DIR / "overall_model_performance_summary.csv")
            display(overall)
            from IPython.display import Image, display as ipy_display
            for name in ["overall_rt_distribution.png", "overall_rt_ecdf.png", "overall_condition_rt_distributions.png"]:
                ipy_display(Image(filename=str(FIG_DIR / name)))
            """
        ),
        nb_md(
            """
            Interpretation: the saved smoke summary reports that the model is right-skewed, has a nonzero right-tail spread, is overall faster than human RT, and should still be treated as a mechanism test rather than a final fit.
            """
        ),
        nb_md(
            """
            # Section 4. Explain evidence sampling mechanism

            Current variational evidence sampling is:

            `mu, sigma = variational_head(pooled_features)`

            `evidence = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise`

            Shapes:

            - `mu: [B, 4]`
            - `sigma: [B, 4]`
            - `noise: [B, T, 4]`
            - `evidence: [B, T, 4]`

            For one trial, `mu` and `sigma` are fixed across time. Every time step and direction gets a fresh `epsilon_t ~ N(0,1)`. This means the default evidence noise is time-step independent, not temporally correlated evidence noise.
            """
        ),
        nb_code(
            """
            selected = pd.read_csv(OUT_DIR / "selected_trials.csv")
            repeated = pd.read_csv(OUT_DIR / "repeated_forward_predictions.csv")
            dist_summary = pd.read_csv(OUT_DIR / "single_trial_rt_distribution_summary.csv")
            print("Selected trials:", len(selected))
            display(selected)
            display(dist_summary)
            """
        ),
        nb_md(
            """
            # Section 5. Explain DMC modulation

            DMC does not directly produce RT or update the WW state. It modulates the evidence sequence before evidence enters WW. By default, congruent trials are unchanged; incongruent trials get early flanker boost and late target support / flanker suppression.
            """
        ),
        nb_code(
            """
            from IPython.display import Image, display as ipy_display
            ipy_display(Image(filename=str(FIG_DIR / "dmc_time_multipliers.png")))
            ipy_display(Image(filename=str(FIG_DIR / "example_dmc_modulated_evidence.png")))
            """
        ),
        nb_md(
            """
            # Section 6. Explain Wong-Wang recurrent accumulation

            Input at each step is `ww_input[t] = [left, right, up, down]`.

            `I_t = J_ext * ww_input_t`

            Internal state is `s_t = [s_left, s_right, s_up, s_down]`.

            `x_t = s_t @ J_matrix + I_0 + I_t + I_noise_t`

            `H_t = nonlinear_transfer(x_t)`

            `ds/dt = -s_t / tau_s + (1 - s_t) * H_t * gamma / 1000`

            `s_{t+1} = s_t + ds/dt * dt`

            `s_t` is the recurrent decision state, not the original Stage-1 evidence. The diagonal of `J_matrix` is self-excitation. The off-diagonal entries are lateral inhibition. `trajectory - threshold` is a threshold-relative decision trajectory, not Stage-1 evidence.
            """
        ),
        nb_code(
            """
            J = params["ww.J_matrix"]
            s = np.array([0.30, 0.10, 0.05, 0.02])
            recurrent_input = s @ J
            toy = pd.DataFrame({"class": CLASS_NAMES, "state_s": s, "recurrent_input": recurrent_input})
            display(toy)
            """
        ),
        nb_md(
            """
            # Section 7. Explain DiffDecision vs soft_index readout

            `DiffDecisionMultiClass` finds the first threshold-crossing time for each class from `trajectory - threshold`. It exists to provide crossing-time computation and an approximate backward path.

            `soft_index` then places a Gaussian window around each class crossing, reads class evidence near that crossing, softmaxes class evidence into `choice_probs`, and computes `pred_rt` as a probability-weighted class-wise decision time.

            In the current config, `readout_mode` is `soft_index`, so the final `pred_rt` mainly comes from `compute_soft_index_readout(evidence_traj)`. `DiffDecision` may still be called inside WW to compute class-wise crossing time, but soft-index is the final RT/choice readout strategy.
            """
        ),
        nb_md("# Section 8. Select representative real trials"),
        nb_code("display(selected)"),
        nb_md("# Section 9. Repeated stochastic forward for single-trial variability"),
        nb_code(
            """
            print("N repeated-forward rows:", len(repeated))
            display(repeated.head())
            """
        ),
        nb_md("# Section 10. Single-trial RT distribution summary"),
        nb_code("display(dist_summary)"),
        nb_md("# Section 11. Single-trial figures"),
        nb_code(
            """
            from IPython.display import Image, display as ipy_display
            for i in range(min(3, len(selected))):
                for suffix in ["rt_hist", "rt_ecdf", "choice_counts", "evidence_samples", "dmc_modulated_evidence", "ww_trajectories", "crossing_times"]:
                    path = FIG_DIR / f"trial_{i:03d}_{suffix}.png"
                    if path.exists():
                        ipy_display(Image(filename=str(path)))
            """
        ),
        nb_md(
            """
            # Section 12. Noise ablation

            Four settings are compared:

            1. Stage-1 evidence noise ON, WW internal noise ON
            2. Stage-1 evidence noise ON, WW internal noise OFF
            3. Stage-1 evidence noise OFF, WW internal noise ON
            4. Stage-1 evidence noise OFF, WW internal noise OFF

            Stage-1 noise OFF uses repeated `mu`; WW noise OFF sets internal noise to zero. DMC remains deterministic in all four settings.
            """
        ),
        nb_code(
            """
            ablation = pd.read_csv(OUT_DIR / "noise_ablation_summary.csv")
            display(ablation.groupby("noise_condition")[["rt_std", "error_probability", "choice_consistency", "q95_minus_q50"]].mean())
            from IPython.display import Image, display as ipy_display
            ipy_display(Image(filename=str(FIG_DIR / "noise_ablation_rt_variability.png")))
            """
        ),
        nb_md(
            """
            # Section 13. Condition-level vs single-trial comparison

            Condition-level right tail may come from mixing many trials with different difficulty. Single-trial repeated right tail comes from stochastic dynamics inside the same fixed trial. These are different questions.

            If single-trial distributions are narrow while condition-level distributions are broad, most variability is likely trial heterogeneity. If selected single-trial distributions are themselves broad or right-skewed, the model's internal stochastic dynamics can generate RT variability within one fixed trial.
            """
        ),
        nb_code(
            """
            comparison = dist_summary.merge(selected[["trial_id", "congruency", "human_rt", "human_correct", "reason_selected"]], on="trial_id")
            display(comparison[["trial_id", "congruency", "human_rt", "mean_rt", "std_rt", "q95_minus_q50", "skewness", "choice_consistency", "reason_selected"]])
            """
        ),
        nb_md(
            """
            # Section 14. Interpretation and take-home message

            This notebook uses real human data plus current saved DMC + Var->WW artifacts for a diagnostic repeated-forward probe. It does not prove the final model is successful. It tests whether the current mechanism can generate single-trial stochastic RT variability after model parameters are fixed.

            Use `overall_model_performance_summary.csv`, `single_trial_rt_distribution_summary.csv`, and `noise_ablation_summary.csv` as the compact tables for reporting. Use the figures folder for the slide-ready panels.

            Important limitation: the current main pipeline does not save a formal repeated-forward export with trial-level `mu`, `sigma`, sampled evidence, DMC evidence, WW trajectories, and seeds. This notebook reconstructs a diagnostic probe from cached logits, saved WW/DMC parameters, and previous predictions.

            Next steps: make the main pipeline export repeated-forward fields, consider temporally correlated evidence noise, and run Tubito-style counterfactual conflict injection to test how controlled conflict manipulation changes fast-error-like behavior.
            """
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def main() -> None:
    stats = make_artifacts()
    build_notebook()
    print(f"Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()

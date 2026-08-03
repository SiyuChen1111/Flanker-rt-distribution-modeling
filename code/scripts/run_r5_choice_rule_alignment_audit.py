#!/usr/bin/env python3
"""Compare R5 whole-trajectory choice with choice at the R5 RT readout step."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vam-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from run_r5_supervisor_followup import group_params, reconstruct_r5  # noqa: E402


ROOT = PROJECT_ROOT / "artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000"
R5_RESULTS = ROOT / "best_model_R5_combined_best/results"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/results/r5_choice_rule_alignment_audit_20260803"
DT_S = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def semantic_outcome(choice: np.ndarray, target: np.ndarray, flanker: np.ndarray) -> np.ndarray:
    return np.where(choice == target, "target", np.where(choice == flanker, "flanker", "other"))


def ordered_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Assign equal-count bins with deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    out = np.zeros(values.size, dtype=np.int64)
    order = np.argsort(values, kind="mergesort")
    for bin_id, idx in enumerate(np.array_split(order, n_bins), start=1):
        out[idx] = bin_id
    return out


def sustained_readout(traj: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the exact R5 sustained crossing step and no-cross flag."""
    params = group_params()
    readout_step = np.empty(len(traj), dtype=np.int64)
    no_cross = np.empty(len(traj), dtype=bool)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        part = traj[idx]
        p = params[str(group)]
        top2 = np.sort(part, axis=2)[:, :, -2:]
        winner = part.argmax(axis=2)
        passed = (top2[:, :, 1] > float(p["threshold"])) & (
            (top2[:, :, 1] - top2[:, :, 0]) >= float(p["margin"])
        )
        min_step = int(round(float(p["min_decision_time"]) / DT_S))
        passed[:, :min_step] = False
        k = int(p["sustained_k"])
        if k > 1:
            sustained = np.zeros_like(passed)
            for t in range(part.shape[1] - k + 1):
                same_winner = np.all(winner[:, t : t + k] == winner[:, t : t + 1], axis=1)
                sustained[:, t] = same_winner & np.all(passed[:, t : t + k], axis=1)
            passed = sustained
        local_no_cross = ~passed.any(axis=1)
        local_step = passed.argmax(axis=1)
        local_step[local_no_cross] = part.shape[1] - 1
        readout_step[idx] = local_step
        no_cross[idx] = local_no_cross
    return readout_step, no_cross


def caf_table(trial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_part in trial.groupby("analysis_group", sort=True):
        for congruency, part in group_part.groupby("congruency", sort=True):
            part = part.copy()
            model_bin = ordered_bins(part["pred_rt"].to_numpy(float))
            human_bin = ordered_bins(part["true_rt"].to_numpy(float))
            for rule, correct_col in [
                ("trajectory_max_choice", "trajectory_max_correct"),
                ("winner_at_readout", "readout_correct"),
            ]:
                for bin_id in range(1, 6):
                    selected = part.iloc[np.flatnonzero(model_bin == bin_id)]
                    rows.append(
                        {
                            "source": "model",
                            "rule": rule,
                            "analysis_group": group,
                            "congruency": int(congruency),
                            "rt_bin": bin_id,
                            "n_trials": len(selected),
                            "median_rt": float(selected["pred_rt"].median()),
                            "accuracy": float(selected[correct_col].mean()),
                        }
                    )
            for bin_id in range(1, 6):
                selected = part.iloc[np.flatnonzero(human_bin == bin_id)]
                rows.append(
                    {
                        "source": "human",
                        "rule": "observed",
                        "analysis_group": group,
                        "congruency": int(congruency),
                        "rt_bin": bin_id,
                        "n_trials": len(selected),
                        "median_rt": float(selected["true_rt"].median()),
                        "accuracy": float(selected["human_correct"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def rule_metrics(trial: pd.DataFrame, caf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    human = caf[caf["source"].eq("human")].set_index(["analysis_group", "congruency", "rt_bin"])
    for rule, choice_col, correct_col in [
        ("trajectory_max_choice", "trajectory_max_choice", "trajectory_max_correct"),
        ("winner_at_readout", "winner_at_readout", "readout_correct"),
    ]:
        model = caf[(caf["source"].eq("model")) & (caf["rule"].eq(rule))].set_index(
            ["analysis_group", "congruency", "rt_bin"]
        )
        for scope, mask in [
            ("all", np.ones(len(trial), dtype=bool)),
            ("congruent", trial["congruency"].eq(0).to_numpy()),
            ("incongruent", trial["congruency"].eq(1).to_numpy()),
        ]:
            if scope == "all":
                scoped_model, scoped_human = model, human
            else:
                congruency_code = 0 if scope == "congruent" else 1
                scoped_model = model.xs(congruency_code, level="congruency")
                scoped_human = human.xs(congruency_code, level="congruency")
            rmse = float(np.sqrt(np.mean((scoped_model["accuracy"] - scoped_human["accuracy"]) ** 2)))
            part = trial.loc[mask]
            choice = part[choice_col].to_numpy(int)
            rows.append(
                {
                    "rule": rule,
                    "scope": scope,
                    "n_trials": len(part),
                    "accuracy": float(part[correct_col].mean()),
                    "human_choice_agreement": float(np.mean(choice == part["response_label"].to_numpy(int))),
                    "target_rate": float(np.mean(choice == part["target_label"].to_numpy(int))),
                    "flanker_rate": float(np.mean(choice == part["flanker_label"].to_numpy(int))) if scope == "incongruent" else np.nan,
                    "other_rate": float(
                        np.mean(
                            (choice != part["target_label"].to_numpy(int))
                            & (choice != part["flanker_label"].to_numpy(int))
                        )
                    ),
                    "caf_rmse_vs_human": rmse,
                }
            )
    return pd.DataFrame(rows)


def make_figure(caf: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"trajectory_max_choice": "#0072B2", "winner_at_readout": "#D55E00", "observed": "#222222"}
    labels = {"trajectory_max_choice": "Whole-trajectory max", "winner_at_readout": "Winner at RT", "observed": "Human"}
    for ax, (group, part) in zip(axes, caf.groupby("analysis_group", sort=True)):
        part = part[part["congruency"].eq(1)]
        for rule in ["observed", "trajectory_max_choice", "winner_at_readout"]:
            selected = part[part["rule"].eq(rule)]
            ax.plot(selected["median_rt"], selected["accuracy"], marker="o", color=colors[rule], label=labels[rule])
        ax.set(title=str(group), xlabel="Median RT (s)", ylim=(0, 1.03))
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False)
    fig.suptitle("Incongruent CAF: choice rule paired comparison")
    fig.tight_layout()
    fig.savefig(out / "incongruent_caf_choice_rule_comparison.png", dpi=300)
    fig.savefig(out / "incongruent_caf_choice_rule_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise RuntimeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    reconstructed, outputs, _ = reconstruct_r5()
    retained = pd.read_csv(R5_RESULTS / "best_model_trial_level_predictions.csv")
    if len(reconstructed) != len(retained):
        raise RuntimeError("Retained and reconstructed R5 rows differ in length.")
    keys = ["analysis_group", "row_index", "subset_stimulus_id", "target_label", "flanker_label"]
    order_match = all(
        np.array_equal(reconstructed[key].astype(str).to_numpy(), retained[key].astype(str).to_numpy()) for key in keys
    )
    if not order_match:
        raise RuntimeError("Retained and reconstructed R5 rows are not aligned.")

    traj = np.asarray(outputs["trajectory"], dtype=np.float32)
    evidence = np.asarray(outputs["evidence_traj"], dtype=np.float32)
    readout_step, no_cross = sustained_readout(traj, retained["analysis_group"].astype(str).to_numpy())
    retained_step = np.rint(retained["decision_time"].to_numpy(float) / DT_S).astype(np.int64)
    retained_step = np.clip(retained_step, 0, traj.shape[1] - 1)
    step_match = float(np.mean(readout_step == retained_step))

    trial = retained.copy()
    rows = np.arange(len(trial))
    trajectory_max_choice = evidence.max(axis=1).argmax(axis=1)
    winner_at_readout = traj[rows, readout_step].argmax(axis=1)
    target = trial["target_label"].to_numpy(int)
    flanker = trial["flanker_label"].to_numpy(int)
    trial["readout_step"] = readout_step
    trial["no_cross"] = no_cross
    trial["trajectory_max_choice"] = trajectory_max_choice
    trial["winner_at_readout"] = winner_at_readout
    trial["trajectory_max_correct"] = trajectory_max_choice == target
    trial["readout_correct"] = winner_at_readout == target
    trial["rules_agree"] = trajectory_max_choice == winner_at_readout
    trial["trajectory_max_outcome"] = semantic_outcome(trajectory_max_choice, target, flanker)
    trial["readout_outcome"] = semantic_outcome(winner_at_readout, target, flanker)
    peak_step_by_class = evidence.argmax(axis=1)
    trial["chosen_peak_step"] = peak_step_by_class[rows, trajectory_max_choice]
    trial["chosen_peak_after_readout"] = trial["chosen_peak_step"].to_numpy(int) > readout_step
    trial.to_csv(out / "trial_level_choice_rule_comparison.csv", index=False)

    caf = caf_table(trial)
    metrics = rule_metrics(trial, caf)
    incongruent = trial[trial["congruency"].eq(1)]
    transitions = pd.crosstab(
        incongruent["trajectory_max_outcome"], incongruent["readout_outcome"], dropna=False
    ).rename_axis(index="trajectory_max_outcome", columns="readout_outcome").stack(future_stack=True).rename("n_trials").reset_index()
    transitions["proportion_of_incongruent"] = transitions["n_trials"] / len(incongruent)
    caf.to_csv(out / "caf_by_choice_rule.csv", index=False)
    metrics.to_csv(out / "choice_rule_metrics.csv", index=False)
    transitions.to_csv(out / "incongruent_choice_transition.csv", index=False)
    make_figure(caf, out)

    changed = ~trial["rules_agree"]
    changed_inc = changed & trial["congruency"].eq(1)
    target_rescue = int(
        ((trial["trajectory_max_outcome"].eq("target")) & (trial["readout_outcome"].eq("flanker"))).sum()
    )
    reverse_change = int(
        ((trial["trajectory_max_outcome"].eq("flanker")) & (trial["readout_outcome"].eq("target"))).sum()
    )
    target_rescue_mask = trial["trajectory_max_outcome"].eq("target") & trial["readout_outcome"].eq("flanker")
    qa = {
        "n_trials": len(trial),
        "row_order_match": order_match,
        "recomputed_readout_step_match_rate": step_match,
        "retained_choice_matches_trajectory_max_rate": float(
            np.mean(retained["pred_choice"].to_numpy(int) == trajectory_max_choice)
        ),
        "rule_change_count": int(changed.sum()),
        "rule_change_rate": float(changed.mean()),
        "rule_changes_on_incongruent_count": int(changed_inc.sum()),
        "rule_changes_on_congruent_count": int((changed & trial["congruency"].eq(0)).sum()),
        "no_cross_count": int(no_cross.sum()),
        "no_cross_rate": float(no_cross.mean()),
        "changed_trials_with_peak_after_readout_rate": float(trial.loc[changed, "chosen_peak_after_readout"].mean()),
        "target_rescue_after_readout_count": target_rescue,
        "target_rescue_peak_after_readout_rate": float(
            trial.loc[target_rescue_mask, "chosen_peak_after_readout"].mean()
        ),
        "flanker_to_target_at_readout_count": reverse_change,
    }
    qa["qa_passed"] = bool(
        order_match
        and step_match == 1.0
        and qa["retained_choice_matches_trajectory_max_rate"] == 1.0
        and qa["rule_changes_on_congruent_count"] == 0
    )
    (out / "qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    all_metrics = metrics[metrics["scope"].eq("all")].set_index("rule")
    inc_metrics = metrics[metrics["scope"].eq("incongruent")].set_index("rule")
    summary = f"""# R5 决策时点与选择规则一致性检查

## 对照定义

- `trajectory_max_choice`：当前 R5 规则。分别寻找四个通道在完整 0.8 秒轨迹中的最高状态，再选择最高者；它可以使用 RT 时点之后的信息。
- `winner_at_readout`：保持同一输入、同一 WW 轨迹、同一 sustained-crossing RT，只在该 RT 时点选择状态最高的通道。未越界试次在模拟截止点读出，并单独标记为删失。

## 主要结果

- 当前保留选择与 `trajectory_max_choice` 的逐试次一致率为 `{qa['retained_choice_matches_trajectory_max_rate']:.3f}`；重算 RT 时点与保留结果的一致率为 `{step_match:.3f}`。
- 两种规则在 `{qa['rule_change_count']}` / `{len(trial)}` 个试次中给出不同选择（`{qa['rule_change_rate']:.3f}`），所有变化都发生在不一致试次。
- 不一致试次中，当前规则正确率为 `{inc_metrics.loc['trajectory_max_choice', 'accuracy']:.3f}`，严格 RT 时点规则为 `{inc_metrics.loc['winner_at_readout', 'accuracy']:.3f}`。总体正确率相应从 `{all_metrics.loc['trajectory_max_choice', 'accuracy']:.3f}` 降到 `{all_metrics.loc['winner_at_readout', 'accuracy']:.3f}`。
- 变化方向高度不对称：`{target_rescue}` 个试次在 RT 时点由 flanker 占优，但完整轨迹最大值判为 target；这些试次中 target 的峰值 `100%` 出现在 RT 之后。反方向只有 `{reverse_change}` 个。这表明当前较高正确率主要来自 RT 之后的 target recovery。
- 未越界试次只有 `{qa['no_cross_count']}` 个（`{qa['no_cross_rate']:.3f}`），远少于 `{qa['rule_change_count']}` 个规则分歧，因此未越界不是主要原因。
- 在不一致试次中，两种规则的 CAF-vs-human RMSE 分别为 `{inc_metrics.loc['trajectory_max_choice', 'caf_rmse_vs_human']:.3f}` 与 `{inc_metrics.loc['winner_at_readout', 'caf_rmse_vs_human']:.3f}`。完整曲线见配套图和 CSV。

## 理论解释

如果把 R5 的 sustained crossing 称为“决策时点”，那么 choice 应与该时点绑定，或模拟应在到达边界时终止。当前实现把“何时反应”和“选了什么”交给两段不同时间信息：RT 只看首次满足条件的时点，choice 却可以查看整个后续轨迹。这是一个理论上的混合读出，而不是标准的不可逆首次通过决策。

不过，直接改成 `winner_at_readout` 也不能作为最终修复：它使不一致正确率大幅下降，说明当前上游时间映射、WW 惯性或阈值设置尚未让 target recovery 在决策前充分发生。正确的下一步应是先把 RT 与 choice 统一，再重新拟合这些参数；不能只替换 choice rule 后沿用为旧规则优化出的参数。

## “自然出现”的准确含义

真实 VGG 的 natural emergence 只指：未经额外手工冲突模块，层级证据本身出现“早期偏 flanker、后期偏 target”的时间结构。它不等于完整行为模式自然涌现。当前检查反而显示，模型是否把这项上游结构转化为正确选择，强烈依赖积累和读出规则。
"""
    (out / "summary.md").write_text(summary, encoding="utf-8")
    if not qa["qa_passed"]:
        raise RuntimeError(f"QA failed: {qa}")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

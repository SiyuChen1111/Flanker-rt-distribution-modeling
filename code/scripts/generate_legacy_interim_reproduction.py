import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as F

from project_paths import CHECKPOINTS_AGE_GROUPS_ROOT, DATA_AGE_GROUPS_ROOT, RESULTS_ROOT
from run_age_group_post_analysis import (
    DIR_MAP,
    build_matched_sets,
    build_model,
    pca_fit,
    project,
    set_apa_style,
    summarize_group,
)


DEFAULT_OUT_DIR = RESULTS_ROOT / "repro_legacy_interim"
DEFAULT_REFERENCE_DIR = RESULTS_ROOT / "organized" / "legacy_interim_reference"
DEFAULT_YOUNG_LOG_PATH = CHECKPOINTS_AGE_GROUPS_ROOT / "20-29" / "stage2" / "test_logits.npz"
DEFAULT_OLD_LOGITS_PATH = CHECKPOINTS_AGE_GROUPS_ROOT / "80-89" / "stage2" / "test_logits.npz"
DEFAULT_YOUNG_LOG_FILE = Path("logs/train_20_29_cached_unbuffered.log")
DEFAULT_OLD_CONFIG_PATH = Path("archive/response_label_refit_backup/80-89/best_config.target_supervision.json")
DEFAULT_OLD_PARAMS_PATH = Path("archive/response_label_refit_backup/80-89/best_model_params.target_supervision.npz")
PRIMARY_OUTPUTS = [
    "figureA2_80_89_rt_distributions.png",
    "figureA4_interim_trajectory_geometry.png",
    "figureA4_interim_trajectory_spread.csv",
]
SUPPORTING_OUTPUTS = [
    "figure_hybrid_legacy_parameter_comparison.png",
    "hybrid_legacy_parameter_comparison.csv",
    "hybrid_legacy_parameter_notes.md",
    "legacy_reference_comparison.md",
    "legacy_reference_image_comparison.csv",
    "legacy_reference_spread_comparison.csv",
    "legacy_reproduction_manifest.json",
    "README.md",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Replay the legacy interim report flow with pinned sources.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--young_logits", type=Path, default=DEFAULT_YOUNG_LOG_PATH)
    parser.add_argument("--young_log", type=Path, default=DEFAULT_YOUNG_LOG_FILE)
    parser.add_argument("--old_config", type=Path, default=DEFAULT_OLD_CONFIG_PATH)
    parser.add_argument("--old_params", type=Path, default=DEFAULT_OLD_PARAMS_PATH)
    parser.add_argument("--old_logits", type=Path, default=DEFAULT_OLD_LOGITS_PATH)
    parser.add_argument("--young_data", type=Path, default=DATA_AGE_GROUPS_ROOT / "20-29" / "test_data.csv")
    parser.add_argument("--young_rt_stats", type=Path, default=DATA_AGE_GROUPS_ROOT / "20-29" / "rt_stats.json")
    parser.add_argument("--old_data", type=Path, default=DATA_AGE_GROUPS_ROOT / "80-89" / "test_data.csv")
    parser.add_argument("--old_rt_stats", type=Path, default=DATA_AGE_GROUPS_ROOT / "80-89" / "rt_stats.json")
    return parser.parse_args()


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def parse_best_completed_scale_from_log(log_path: Path):
    text = log_path.read_text(errors="replace")
    pattern = re.compile(
        r"\[(?P<age>20-29) scale \d+/\d+\] Finished in [^|]+\| Score=(?P<score>[0-9.]+), "
        r"PredMean=(?P<pred_mean>[0-9.]+)s, Acc=(?P<model_acc>[0-9.]+), Cong=(?P<model_cong>[0-9.]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"No completed scale summaries found in {log_path}")

    best = None
    best_score = -np.inf
    for match in matches:
        score = float(match.group("score"))
        if score > best_score:
            best_score = score
            best = {
                "age_group": match.group("age"),
                "status": "interim_best_completed_scale",
                "best_score": score,
                "model_mean_rt": float(match.group("pred_mean")),
                "model_accuracy": float(match.group("model_acc")),
                "model_congruency_rt_gap": float(match.group("model_cong")),
            }

    scale_pattern = re.compile(
        r"\[(20-29) scale (?P<idx>\d+)/\d+\] Finished in [^|]+\| Score=(?P<score>[0-9.]+), "
        r"PredMean=(?P<pred_mean>[0-9.]+)s, Acc=(?P<model_acc>[0-9.]+), Cong=(?P<model_cong>[0-9.]+)"
    )
    scale_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    scale_lookup = {}
    for match in scale_pattern.finditer(text):
        idx = int(match.group("idx"))
        scale_lookup[float(match.group("score"))] = scale_values[idx - 1]

    if best is None:
        raise ValueError(f"Failed to extract best completed-scale summary from {log_path}")
    best["best_scale"] = float(scale_lookup.get(best["best_score"], np.nan))
    if not np.isclose(best["best_scale"], 0.1):
        raise ValueError(f"Expected legacy best_scale=0.1, got {best['best_scale']}")
    return best


def build_young_interim_summary(log_path: Path, data_path: Path, rt_stats_path: Path):
    summary = parse_best_completed_scale_from_log(log_path)
    with open(rt_stats_path, "r") as f:
        human_stats = json.load(f)
    df = pd.read_csv(data_path)
    target = df["target_direction"].map(lambda x: DIR_MAP[x])
    response = df["response_direction"].map(lambda x: DIR_MAP[x])
    flanker = df["flanker_direction"].map(lambda x: DIR_MAP[x])
    congruency = (target != flanker).astype(int)
    summary.update(
        {
            "human_mean_rt": float(human_stats["mean"]),
            "human_median_rt": float(human_stats["median"]),
            "human_skew": float(human_stats["skewness"]),
            "human_accuracy": float((target == response).mean()),
            "human_congruency_rt_gap": float(
                df.loc[congruency == 1, "response_time"].mean() / 1000.0
                - df.loc[congruency == 0, "response_time"].mean() / 1000.0
            ),
        }
    )
    return summary


def load_legacy_old_artifact(config_path: Path, params_path: Path, logits_path: Path, data_path: Path, rt_stats_path: Path):
    with open(config_path, "r") as f:
        best_config = json.load(f)
    params_npz = np.load(params_path)
    params = {k: params_npz[k] for k in params_npz.files}
    logits_npz = np.load(logits_path)
    test_df = pd.read_csv(data_path)
    with open(rt_stats_path, "r") as f:
        human_stats = json.load(f)
    if len(test_df) != len(logits_npz["logits"]):
        raise ValueError("Length mismatch for 80-89 legacy artifact: test csv vs logits")
    return {
        "age_group": "80-89",
        "best_config": best_config,
        "params": params,
        "test_logits": logits_npz["logits"].astype(np.float32),
        "test_rts": logits_npz["rts"].astype(np.float32),
        "test_rts_normalized": logits_npz["rts_normalized"].astype(np.float32),
        "test_df": test_df.copy(),
        "human_stats": human_stats,
    }


def run_hybrid_young_inference(interim_20: dict, artifact_80: dict, young_logits_path: Path, young_data_path: Path):
    best_scale = float(interim_20["best_scale"])
    test_logits_20 = np.load(young_logits_path)
    test_df_20 = pd.read_csv(young_data_path)
    df_20 = test_df_20.copy()
    df_20["target_dir_idx"] = df_20["target_direction"].map(lambda x: DIR_MAP[x])
    df_20["flanker_dir_idx"] = df_20["flanker_direction"].map(lambda x: DIR_MAP[x])
    df_20["response_dir_idx"] = df_20["response_direction"].map(lambda x: DIR_MAP[x])
    df_20["correct"] = (df_20["target_dir_idx"] == df_20["response_dir_idx"]).astype(int)
    df_20["congruency"] = (df_20["target_dir_idx"] != df_20["flanker_dir_idx"]).astype(int)
    df_20["condition"] = df_20["congruency"].map(lambda x: "Congruent" if x == 0 else "Incongruent")

    model_20 = build_model({"time_steps": 111, "scale": best_scale}, artifact_80["params"])
    state_dict = model_20.state_dict()
    state_dict["scale"] = torch.tensor(best_scale, dtype=torch.float32)
    model_20.load_state_dict(state_dict, strict=False)
    model_20.eval()

    x20 = torch.tensor(test_logits_20["logits"].astype(np.float32), dtype=torch.float32)
    with torch.no_grad():
        scaled20 = F.relu(x20 * model_20.state_dict()["scale"])
        decision_times_class_20, traj20, threshold20 = model_20.ww.inference(scaled20)
        pred_rt20, pred_choice20 = decision_times_class_20.min(dim=1)

    group_df_20 = df_20.copy()
    group_df_20["pred_rt"] = pred_rt20.cpu().numpy()
    group_df_20["pred_choice"] = pred_choice20.cpu().numpy()
    group_df_20["pred_correct"] = (group_df_20["pred_choice"] == group_df_20["target_dir_idx"]).astype(int)
    inf20 = {
        "trajectory": traj20.cpu().numpy(),
        "threshold": float(threshold20.detach().cpu().item()),
    }
    params20 = dict(artifact_80["params"])
    params20["scale"] = np.array(best_scale, dtype=np.float32)
    return group_df_20, inf20, params20


def make_dual_group_rt_distribution_plot(group_dfs: dict[str, pd.DataFrame], out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True, sharex=True, sharey=True)
    hist_bins = np.linspace(0, 2, 81)
    kde_x = np.linspace(0, 2, 400)
    panel_peaks = []

    def draw_panel(ax, series_a, series_b, title):
        hist_a, _ = np.histogram(series_a, bins=hist_bins, density=True)
        hist_b, _ = np.histogram(series_b, bins=hist_bins, density=True)
        peak = float(max(hist_a.max(initial=0.0), hist_b.max(initial=0.0)))
        ax.hist(series_a, bins=hist_bins, density=True, alpha=0.18, color="#4C78A8")
        ax.hist(series_b, bins=hist_bins, density=True, alpha=0.18, color="#E45756")
        if len(series_a) > 1:
            kde_a = gaussian_kde(series_a)
            kde_vals_a = kde_a(kde_x)
            peak = max(peak, float(np.max(kde_vals_a)))
            ax.plot(kde_x, kde_vals_a, color="#4C78A8", linewidth=2.2, linestyle="-", label="Congruent")
        if len(series_b) > 1:
            kde_b = gaussian_kde(series_b)
            kde_vals_b = kde_b(kde_x)
            peak = max(peak, float(np.max(kde_vals_b)))
            ax.plot(kde_x, kde_vals_b, color="#E45756", linewidth=2.2, linestyle="-", label="Incongruent")
        ax.set_title(title)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 2)
        ax.legend(frameon=False)
        panel_peaks.append(peak)

    panel_map = [
        ("80-89", axes[0, 0], "response_time", "80-89 Human RT distributions"),
        ("80-89", axes[0, 1], "pred_rt", "80-89 Model RT distributions"),
        ("20-29", axes[1, 0], "response_time", "20-29 Human RT distributions"),
        ("20-29", axes[1, 1], "pred_rt", "20-29 Model RT distributions"),
    ]
    for age_group, ax, column, title in panel_map:
        df = group_dfs[age_group]
        series_cong = df.loc[df["congruency"] == 0, column]
        series_incong = df.loc[df["congruency"] == 1, column]
        if column == "response_time":
            series_cong = series_cong / 1000.0
            series_incong = series_incong / 1000.0
        draw_panel(ax, series_cong, series_incong, title)

    common_ymax = max(panel_peaks) * 1.05 if panel_peaks else None
    if common_ymax is not None:
        for row in axes:
            for ax in row:
                ax.set_ylim(0, common_ymax)

    fig.suptitle(
        "Figure A2. Legacy hybrid RT distributions by age group and congruency "
        "(80-89 top row, 20-29 bottom row; Human vs Model, matched plot scales, step-hist + KDE)"
    )
    out_path = out_dir / "figureA2_80_89_rt_distributions.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_interim_trajectory_plot(group_df_20: pd.DataFrame, inf20: dict, group_df_80: pd.DataFrame, artifact_80: dict, out_dir: Path):
    matched = build_matched_sets({"20-29": group_df_20, "80-89": group_df_80})
    required_keys = [("20-29", "Congruent"), ("20-29", "Incongruent"), ("80-89", "Congruent"), ("80-89", "Incongruent")]
    if not all(key in matched for key in required_keys):
        raise ValueError("Failed to build matched sets for all required age-group/condition pairs")

    x80 = artifact_80["test_logits"].astype(np.float32)
    model_80 = build_model(artifact_80["best_config"], artifact_80["params"])
    with torch.no_grad():
        scaled80 = F.relu(torch.tensor(x80, dtype=torch.float32) * model_80.state_dict()["scale"])
        _, traj80, _ = model_80.ww.inference(scaled80)
    inf80 = {"trajectory": traj80.cpu().numpy()}

    state_blocks = []
    mean_trajs = {}
    spread_rows = []
    inference_by_group = {"20-29": inf20, "80-89": inf80}
    colors = {
        ("20-29", "Congruent"): "#4C78A8",
        ("20-29", "Incongruent"): "#72B7B2",
        ("80-89", "Congruent"): "#F58518",
        ("80-89", "Incongruent"): "#E45756",
    }

    for age, condition in required_keys:
        idx = matched[(age, condition)].index.to_numpy()
        traj = inference_by_group[age]["trajectory"][idx]
        state_blocks.append(traj.reshape(-1, traj.shape[-1]))
        mean_traj = traj.mean(axis=0)
        mean_trajs[(age, condition)] = mean_traj
        spread = np.linalg.norm(traj - mean_traj[None, :, :], axis=2).mean()
        spread_rows.append({"age_group": age, "condition": condition, "mean_state_space_spread": float(spread)})

    all_states = np.concatenate(state_blocks, axis=0)
    mean, comps = pca_fit(all_states)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for key, mean_traj in mean_trajs.items():
        proj = project(mean_traj, mean, comps)
        axes[0].plot(proj[:, 0], proj[:, 1], label=f"{key[0]} {key[1]}", color=colors[key], linewidth=2)
        axes[0].scatter(proj[0, 0], proj[0, 1], color=colors[key], s=18)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Interim trajectory geometry")
    axes[0].legend(frameon=False)

    spread_df = pd.DataFrame(spread_rows)
    x = np.arange(len(spread_df))
    bar_colors = [colors[(str(row["age_group"]), str(row["condition"]))] for _, row in spread_df.iterrows()]
    labels = [f"{row['age_group']}\n{row['condition']}" for _, row in spread_df.iterrows()]
    axes[1].bar(x, spread_df["mean_state_space_spread"], color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean trajectory spread")
    axes[1].set_title("Matched-trial spread")

    fig.suptitle("Figure A4. Interim trajectory geometry using 20-29 current-best scale and 80-89 formal fit")
    fig_path = out_dir / "figureA4_interim_trajectory_geometry.png"
    spread_path = out_dir / "figureA4_interim_trajectory_spread.csv"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    spread_df.to_csv(spread_path, index=False)
    return fig_path, spread_path, spread_df


def write_hybrid_parameter_comparison(out_dir: Path, params_20_hybrid: dict, artifact_80: dict):
    params_80 = dict(artifact_80["params"])
    params_80["scale"] = np.array(float(artifact_80["best_config"]["scale"]), dtype=np.float32)

    rows = []
    for label, key in [
        ("Scale", "scale"),
        ("Noise AMPA", "ww.noise_ampa"),
        ("Threshold", "ww.threshold"),
        ("J_ext", "ww.J_ext"),
        ("I_0", "ww.I_0"),
        ("Tau AMPA", "ww.tau_ampa"),
    ]:
        value_20 = float(np.asarray(params_20_hybrid[key]).item())
        value_80 = float(np.asarray(params_80[key]).item())
        rows.append(
            {
                "parameter": label,
                "parameter_key": key,
                "hybrid_20_29": value_20,
                "legacy_80_89": value_80,
                "abs_diff": abs(value_20 - value_80),
                "note": "shared_under_hybrid_setup" if np.isclose(value_20, value_80) else "different_under_hybrid_setup",
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_dir / "hybrid_legacy_parameter_comparison.csv", index=False)

    plot_df = comparison_df.copy()
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, plot_df["hybrid_20_29"], width, label="20-29 hybrid replay", color="#4C78A8")
    ax.bar(x + width / 2, plot_df["legacy_80_89"], width, label="80-89 legacy model", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["parameter"], rotation=20, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title("Hybrid legacy internal parameter comparison")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "figure_hybrid_legacy_parameter_comparison.png", bbox_inches="tight")
    plt.close(fig)

    notes = [
        "# Hybrid legacy parameter comparison notes",
        "",
        "This comparison uses the user-requested hybrid legacy setup.",
        "",
        "- `20-29` replay = 20-29 logits + parsed young scale (0.1) + reused 80-89 legacy parameter set.",
        "- `80-89` replay = archived legacy target-supervision config/params.",
        "- Consequence: noise-related parameters are expected to be the same across the two rows because the hybrid replay reuses the 80-89 parameter tensor block.",
        "- The main internal difference should therefore appear in `scale`, not in `ww.noise_ampa`.",
        "",
        comparison_df.to_markdown(index=False),
    ]
    (out_dir / "hybrid_legacy_parameter_notes.md").write_text("\n".join(notes))


def compare_images(candidate: Path, reference: Path) -> dict[str, object]:
    if not candidate.exists() or not reference.exists():
        return {"candidate_exists": candidate.exists(), "reference_exists": reference.exists()}
    with Image.open(candidate) as cand_img, Image.open(reference) as ref_img:
        cand_size = cand_img.size
        ref_size = ref_img.size
    return {
        "candidate_exists": True,
        "reference_exists": True,
        "candidate_bytes": candidate.stat().st_size,
        "reference_bytes": reference.stat().st_size,
        "candidate_width": cand_size[0],
        "candidate_height": cand_size[1],
        "reference_width": ref_size[0],
        "reference_height": ref_size[1],
        "same_dimensions": cand_size == ref_size,
    }


def write_comparison_report(out_dir: Path, reference_dir: Path, spread_df: pd.DataFrame):
    reference_spread_path = reference_dir / "figureA4_interim_trajectory_spread.csv"
    reference_spread_df = pd.read_csv(reference_spread_path)
    spread_compare = spread_df.merge(
        reference_spread_df,
        on=["age_group", "condition"],
        suffixes=("_repro", "_reference"),
    )
    spread_compare["abs_diff"] = (
        spread_compare["mean_state_space_spread_repro"] - spread_compare["mean_state_space_spread_reference"]
    ).abs()
    spread_compare.to_csv(out_dir / "legacy_reference_spread_comparison.csv", index=False)

    image_rows: list[dict[str, object]] = []
    for filename in ["figureA4_interim_trajectory_geometry.png"]:
        info = compare_images(out_dir / filename, reference_dir / filename)
        info["file"] = filename
        image_rows.append(info)
    image_compare_df = pd.DataFrame(image_rows)
    image_compare_df.to_csv(out_dir / "legacy_reference_image_comparison.csv", index=False)

    lines = [
        "# Legacy interim reproduction comparison",
        "",
        f"Reference directory: `{reference_dir}`",
        "",
        "## Spread comparison",
        spread_compare.to_markdown(index=False),
        "",
        "## Image comparison",
        "`figureA2_80_89_rt_distributions.png` is intentionally expanded in this bundle to include the extra 20-29 row, so legacy image-dimension matching is only enforced for A4.",
        "",
        image_compare_df.to_markdown(index=False),
        "",
    ]
    (out_dir / "legacy_reference_comparison.md").write_text("\n".join(lines))


def write_run_manifest(args, out_dir: Path, interim_20: dict, old_summary: dict):
    manifest = {
        "entrypoint": "code/scripts/generate_legacy_interim_reproduction.py",
        "purpose": "Formal reusable reproduction entrypoint for the pinned legacy interim A2/A4 replay.",
        "young_log": str(args.young_log),
        "young_logits": str(args.young_logits),
        "young_data": str(args.young_data),
        "young_rt_stats": str(args.young_rt_stats),
        "young_best_scale": float(interim_20["best_scale"]),
        "old_config": str(args.old_config),
        "old_params": str(args.old_params),
        "old_logits": str(args.old_logits),
        "old_data": str(args.old_data),
        "old_rt_stats": str(args.old_rt_stats),
        "old_backup_scale": float(old_summary["best_scale"]),
        "reference_dir": str(args.reference_dir),
        "output_dir": str(out_dir),
        "primary_outputs": PRIMARY_OUTPUTS,
        "supporting_outputs": SUPPORTING_OUTPUTS,
    }
    (out_dir / "legacy_reproduction_manifest.json").write_text(json.dumps(manifest, indent=2))


def write_bundle_readme(out_dir: Path):
    lines = [
        "# Legacy interim reproduction bundle",
        "",
        "This folder is the formal output bundle for the pinned legacy interim replay entrypoint:",
        "`code/scripts/generate_legacy_interim_reproduction.py`.",
        "",
        "## Primary figures/tables",
    ]
    lines.extend([f"- `{name}`" for name in PRIMARY_OUTPUTS])
    lines.extend(
        [
            "",
            "## Supporting files",
        ]
    )
    lines.extend([f"- `{name}`" for name in SUPPORTING_OUTPUTS if name != "README.md"])
    lines.extend(
        [
            "",
            "## Intended use",
            "- `figureA2_80_89_rt_distributions.png` = congruent/incongruent RT distribution replay in legacy context, with 80-89 on the top row and 20-29 on the bottom row. No RT mean alignment is applied; all four panels share the same plotting scale.",
            "- `figureA4_interim_trajectory_geometry.png` = legacy hybrid trajectory geometry replay.",
            "- `figureA4_interim_trajectory_spread.csv` = numeric A4 spread table for direct comparison with the legacy reference.",
            "- `figure_hybrid_legacy_parameter_comparison.png` / `hybrid_legacy_parameter_comparison.csv` = internal parameter comparison for the user-requested hybrid setup, showing that noise-related terms are shared while scale differs.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    set_apa_style()
    ensure_dirs(args.out_dir)
    stale_alignment_report = args.out_dir / "rt_mean_alignment_report.csv"
    if stale_alignment_report.exists():
        stale_alignment_report.unlink()

    interim_20 = build_young_interim_summary(args.young_log, args.young_data, args.young_rt_stats)
    artifact_80 = load_legacy_old_artifact(args.old_config, args.old_params, args.old_logits, args.old_data, args.old_rt_stats)
    group_df_80, _, summary_80 = summarize_group(artifact_80)
    group_df_20, inf20, params_20_hybrid = run_hybrid_young_inference(interim_20, artifact_80, args.young_logits, args.young_data)

    make_dual_group_rt_distribution_plot({"20-29": group_df_20, "80-89": group_df_80}, args.out_dir)
    _, _, spread_df = make_interim_trajectory_plot(group_df_20, inf20, group_df_80, artifact_80, args.out_dir)
    write_hybrid_parameter_comparison(args.out_dir, params_20_hybrid, artifact_80)
    write_comparison_report(args.out_dir, args.reference_dir, spread_df)
    write_run_manifest(args, args.out_dir, interim_20, summary_80)
    write_bundle_readme(args.out_dir)
    print(f"Saved legacy interim reproduction outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

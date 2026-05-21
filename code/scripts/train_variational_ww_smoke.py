"""
Variational Evidence → Wong-Wang Synthesis Smoke Test

Bridges the Stage-1 variational evidence sampler (time-varying, stochastic)
with the Wong-Wang multi-class decision dynamics (which CAN produce error regimes).

Key hypothesis:
  The SPEA GRU accumulator is the bottleneck (drops 98.4% → 22-44% accuracy).
  WW dynamics have produced error regimes with static logits (Phase 11).
  Variational evidence HAS uncertainty (Stage-1 head branch confirms).
  The combination may produce human-like error-direction behavior.

Design:
  1. Load warm-started variational evidence sampler (Stage 1)
  2. Sample time-varying evidence sequences [B, T_var, 4]
  3. Apply softplus transform + optional scaling
  4. Feed into WongWangMultiClassDecision with matching time_steps
  5. Use soft_index readout (best WW readout per Phase 14)
  6. Train with behavioral losses (error_rate, error_sign, accuracy)
  7. Evaluate against SPEA c0-c4 baselines

Run from project root:
  source .venv/bin/activate
  python code/scripts/train_variational_ww_smoke.py \
    --age_group 20-29 \
    --data_root data/age_groups_matched \
    --output_root artifacts/results/rt_model_variational_ww_synthesis/smoke \
    --epochs_stage1 5 --epochs_ww 15 \
    --smoke_eval --smoke_max_trials 256 \
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cache_vgg_stage2_features import load_stage1_model_with_metadata
from project_paths import PROJECT_ROOT
from stage1_evidence_sampler import (
    SemiSupervisedEvidenceSampler,
    Stage1EvidenceConfig,
)
from train_age_group_semisup_spea import (
    StimulusDataset,
    _build_behavior_balanced_subset,
    evaluate_stage1_head,
    to_jsonable,
    train_stage1_head,
)
from train_age_groups_efficient import (
    compute_human_stats_from_rts,
    evaluate_joint_behavior,
)
from vgg_wongwang_lim import (
    WongWangMultiClassDecision,
    apply_stage2_input_transform,
    compute_rt_readout,
    compute_behavioral_losses,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


class VariationalWWModel(nn.Module):
    """Variational evidence → Wong-Wang bridge model.

    Takes time-varying variational evidence sequences and feeds them
    into Wong-Wang multi-class dynamics.
    """

    def __init__(
        self,
        n_classes: int = 4,
        ww_time_steps: int = 500,
        ww_dt: int = 10,
        evidence_time_steps: int = 120,
        transform_mode: str = "softplus_centered",
        softplus_center: float = 1.0,
        noise_ampa: Optional[float] = None,
        threshold: Optional[float] = None,
        j_offdiag_scale: Optional[float] = None,
        j_ext: Optional[float] = None,
        freeze_ww_params: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ww_time_steps = ww_time_steps
        self.evidence_time_steps = evidence_time_steps
        self.transform_mode = transform_mode
        self.softplus_center = softplus_center

        self.ww = WongWangMultiClassDecision(
            n_classes=n_classes,
            dt=ww_dt,
            time_steps=ww_time_steps,
            t_stimulus=ww_time_steps,
        )

        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Apply noise/threshold overrides; optionally keep trainable
        if noise_ampa is not None:
            with torch.no_grad():
                self.ww.noise_ampa.copy_(torch.tensor(float(noise_ampa)))
            if freeze_ww_params:
                self.ww.noise_ampa.requires_grad_(False)
        if threshold is not None:
            with torch.no_grad():
                self.ww.threshold.copy_(torch.tensor(float(threshold)))
            if freeze_ww_params:
                self.ww.threshold.requires_grad_(False)
        # Scale off-diagonal inhibition (weaker inhibition → more cross-talk → more errors)
        if j_offdiag_scale is not None:
            with torch.no_grad():
                diag_mask = torch.eye(n_classes, dtype=torch.bool, device=self.ww.J_matrix.device)
                self.ww.J_matrix.data[~diag_mask] *= j_offdiag_scale
        # Override external input gain
        if j_ext is not None:
            with torch.no_grad():
                self.ww.J_ext.copy_(torch.tensor(float(j_ext)))
            self.ww.J_ext.requires_grad_(False)

    def _resample_evidence(
        self, evidence: torch.Tensor
    ) -> torch.Tensor:
        """Resample evidence from evidence_time_steps to ww_time_steps."""
        if evidence.shape[1] == self.ww_time_steps:
            return evidence
        B, T_in, C = evidence.shape
        # evidence: [B, T_in, C] → [B, C, T_in] → [B*C, 1, T_in]
        evidence_flat = evidence.permute(0, 2, 1).reshape(B * C, 1, T_in)
        # interpolate along time dimension: [B*C, 1, T_in] → [B*C, 1, ww_time_steps]
        resampled = F.interpolate(
            evidence_flat,
            size=self.ww_time_steps,
            mode="linear",
            align_corners=True,
        )
        # [B*C, 1, ww_time_steps] → [B, C, ww_time_steps] → [B, ww_time_steps, C]
        return resampled.reshape(B, C, self.ww_time_steps).permute(0, 2, 1)

    def forward(
        self,
        evidence_samples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: variational evidence → WW dynamics → decision.

        Args:
            evidence_samples: [B, T_evidence, 4] time-varying evidence
            generator: optional random generator for WW noise

        Returns:
            Dict with trajectory, decision_times, evidence_traj, etc.
        """
        B = evidence_samples.shape[0]
        device = evidence_samples.device

        # Apply transform: softplus(log_scale * evidence - center)
        scale = torch.exp(self.log_scale)
        transformed = apply_stage2_input_transform(
            evidence_samples.reshape(B, -1),
            scale.expand(1),
            transform_mode=self.transform_mode,
            softplus_center=self.softplus_center,
        ).reshape(B, self.evidence_time_steps, self.n_classes)

        # Resample to WW time steps
        ww_input = self._resample_evidence(transformed)

        # Run WW inference to get trajectory
        decision_times_class, trajectory, threshold = self.ww.inference(
            ww_input, generator=generator
        )

        evidence_traj = trajectory - threshold

        return {
            "trajectory": trajectory,
            "evidence_traj": evidence_traj,
            "decision_times_class": decision_times_class,
            "threshold": threshold,
            "ww_input": ww_input,
        }


def compute_ww_readout(
    ww_output: Dict[str, torch.Tensor],
    readout_mode: str = "soft_index",
    readout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute RT and choice from WW output using configured readout."""
    config = dict(readout_config or {})
    config.setdefault("dt_ms", 10.0)
    config.setdefault("choice_temperature", 0.10)
    config.setdefault("sigma_s", 0.05)

    readout = compute_rt_readout(
        readout_mode,
        ww_output["evidence_traj"],
        readout_config=config,
    )

    return readout


def train_variational_ww(
    *,
    sampler: SemiSupervisedEvidenceSampler,
    train_loader: DataLoader,
    test_loader: DataLoader,
    sampler_mode: str,
    evidence_time_steps: int,
    ww_time_steps: int,
    ww_dt: int,
    epochs_stage1: int,
    epochs_ww: int,
    readout_mode: str,
    checkpoint_mode: str = "beh_opt",
    readout_config: Dict[str, Any],
    behavioral_loss_config: Dict[str, Any],
    device: str,
    seed: int,
    output_dir: str,
    lr: float = 1e-4,
    lambda_rt: float = 2.0,
    noise_ampa: Optional[float] = None,
    threshold: Optional[float] = None,
    j_offdiag_scale: Optional[float] = None,
    j_ext: Optional[float] = None,
) -> Dict[str, Any]:
    set_seed(seed)

    # --- Stage 1: warm-start variational head ---
    print("Stage 1: Training variational evidence head...")
    stage1_metrics = train_stage1_head(
        sampler=sampler,
        dataset_loader=train_loader,
        sampler_mode=sampler_mode,
        epochs=epochs_stage1,
        lambda_cls=1.0,
        lambda_ssl=0.0,
        lambda_teacher=0.25,
        lambda_uncertainty_bound=0.05,
        device=device,
    )

    # --- Bundle evidence for WW training ---
    print("Bundling evidence sequences...")
    sampler.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def _bundle_evidence(loader: DataLoader, gen_seed: int) -> Dict[str, np.ndarray]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(gen_seed)
        rows: Dict[str, list] = {
            "evidence_samples": [],
            "target_labels": [],
            "response_labels": [],
            "rts": [],
            "congruency": [],
        }
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                payload = sampler.sample_from_images(
                    images=images,
                    time_steps=evidence_time_steps,
                    sampler_mode=sampler_mode,
                    generator=gen,
                )
                rows["evidence_samples"].append(
                    payload["evidence_samples"].detach().cpu().numpy()
                )
                rows["target_labels"].append(batch["target_label"].cpu().numpy())
                rows["response_labels"].append(batch["response_label"].cpu().numpy())
                rows["rts"].append(batch["rt"].cpu().numpy())
                rows["congruency"].append(batch["congruency"].cpu().numpy())

        return {
            key: np.concatenate([np.asarray(v) for v in values], axis=0)
            for key, values in rows.items()
        }

    train_bundle = _bundle_evidence(train_loader, seed)
    test_bundle = _bundle_evidence(test_loader, seed + 1)

    # --- Stage 2: Train WW on variational evidence ---
    print(f"Stage 2: Training WW on variational evidence ({epochs_ww} epochs)...")
    n_train = len(train_bundle["evidence_samples"])
    print(f"  Train trials: {n_train}, Test trials: {len(test_bundle['evidence_samples'])}")

    model = VariationalWWModel(
        n_classes=4,
        ww_time_steps=ww_time_steps,
        ww_dt=ww_dt,
        evidence_time_steps=evidence_time_steps,
        noise_ampa=noise_ampa,
        threshold=threshold,
        j_offdiag_scale=j_offdiag_scale,
        j_ext=j_ext,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    train_ev = torch.tensor(train_bundle["evidence_samples"], dtype=torch.float32)
    train_target = torch.tensor(train_bundle["target_labels"], dtype=torch.long)
    train_response = torch.tensor(train_bundle["response_labels"], dtype=torch.long)
    train_rt = torch.tensor(train_bundle["rts"], dtype=torch.float32)
    train_cong = torch.tensor(train_bundle["congruency"], dtype=torch.long)

    test_ev = torch.tensor(test_bundle["evidence_samples"], dtype=torch.float32)

    best_score = -float("inf")
    best_state: Optional[Dict[str, np.ndarray]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    all_epoch_metrics: list = []

    for epoch in range(epochs_ww):
        model.train()
        indices = torch.randperm(n_train)
        batch_size = 128
        total_loss_val = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]

            ev_batch = train_ev[idx].to(device)
            target_b = train_target[idx].to(device)
            response_b = train_response[idx].to(device)
            rt_b = train_rt[idx].to(device)

            ww_output = model(ev_batch)
            readout = compute_ww_readout(ww_output, readout_mode, readout_config)
            pred_rt = readout["pred_rt"]

            choice_probs = readout.get("choice_probs")
            if choice_probs is None:
                class_evidence = readout.get("class_evidence")
                if class_evidence is not None:
                    choice_probs = F.softmax(class_evidence, dim=1)
                else:
                    choice_logits = ww_output["evidence_traj"].amax(dim=1)
                    choice_probs = F.softmax(choice_logits, dim=1)

            behavioral_losses = compute_behavioral_losses(
                pred_rt=pred_rt,
                pred_choice=choice_probs.argmax(dim=1),
                choice_probs=choice_probs,
                target_labels=target_b,
                response_labels=response_b,
                true_rt=rt_b,
                config=behavioral_loss_config,
            )

            rt_mse = F.mse_loss(pred_rt, rt_b) * lambda_rt
            loss = behavioral_losses["loss"] + rt_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()
            n_batches += 1

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            ww_test = model(test_ev.to(device))
            readout_test = compute_ww_readout(ww_test, readout_mode, readout_config)
            pred_rt_np = readout_test["pred_rt"].cpu().numpy()
            choice_probs_test = readout_test.get("choice_probs")
            if choice_probs_test is not None:
                pred_choice_np = choice_probs_test.argmax(dim=1).cpu().numpy()
            else:
                class_ev = readout_test.get("class_evidence")
                pred_choice_np = (
                    class_ev.argmax(dim=1).cpu().numpy()
                    if class_ev is not None
                    else ww_test["evidence_traj"].amax(dim=1).argmax(dim=1).cpu().numpy()
                )

        human_stats = compute_human_stats_from_rts(test_bundle["rts"])
        metrics = evaluate_joint_behavior(
            pred_rt=pred_rt_np,
            pred_choice=pred_choice_np,
            true_rt=test_bundle["rts"],
            target_labels=test_bundle["target_labels"],
            response_labels=test_bundle["response_labels"],
            congruency=test_bundle["congruency"],
            human_stats=human_stats,
            rt_shape_focus=True,
        )

        score = float(metrics["behavior_optimal_score"])
        all_epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss_val / max(n_batches, 1),
                "beh_opt": score,
                "acc": float(metrics["model_accuracy"]),
                "resp_agree": float(metrics["response_agreement"]),
                "pred_mean": float(metrics["pred_mean"]),
                "err_correct_delta": float(metrics["error_minus_correct_rt"]),
                "cong_gap": float(metrics["model_congruency_rt_gap"]),
            }
        )

        print(
            f"  Epoch {epoch + 1:02d}/{epochs_ww}: "
            f"loss={total_loss_val / max(n_batches, 1):.4f}, "
            f"beh_opt={score:.4f}, acc={metrics['model_accuracy']:.4f}, "
            f"resp_agree={metrics['response_agreement']:.4f}, "
            f"ΔRT={metrics['error_minus_correct_rt']:.4f}, "
            f"pred_mean={metrics['pred_mean']:.3f}s"
        )

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().numpy()
                for k, v in model.state_dict().items()
            }
            best_metrics = metrics

    if best_state is None:
        raise RuntimeError("No WW checkpoints evaluated")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json(output_path / "metrics_smoke.json", best_metrics)
    _write_json(output_path / "epoch_history.json", all_epoch_metrics)
    _write_json(
        output_path / "config.json",
        {
            "readout_mode": readout_mode,
            "readout_config": readout_config,
            "behavioral_loss_config": behavioral_loss_config,
            "ww_time_steps": ww_time_steps,
            "evidence_time_steps": evidence_time_steps,
            "epochs_stage1": epochs_stage1,
            "epochs_ww": epochs_ww,
            "noise_ampa": noise_ampa,
            "threshold": threshold,
            "j_offdiag_scale": j_offdiag_scale,
            "j_ext": j_ext,
        },
    )
    np.savez(output_path / "best_model_params.npz", **best_state)
    np.savez_compressed(
        output_path / "predictions_smoke.npz",
        pred_rt=pred_rt_np,
        pred_choice=pred_choice_np,
        target_labels=test_bundle["target_labels"],
        response_labels=test_bundle["response_labels"],
        congruency=test_bundle["congruency"],
        true_rt=test_bundle["rts"],
    )

    _write_json(
        output_path / "run_complete.json",
        {
            "best_score": best_score,
            "best_epoch": all_epoch_metrics[-1]["epoch"],
            "variant": "variational_ww_synthesis",
        },
    )

    print(f"\nBest beh_opt={best_score:.4f}")
    print(f"  Accuracy: {best_metrics['model_accuracy']:.4f} (human: {best_metrics['human_accuracy']:.4f})")
    print(f"  Response agreement: {best_metrics['response_agreement']:.4f}")
    print(f"  ΔRT (err-corr): {best_metrics['error_minus_correct_rt']:.4f} (human: {best_metrics['human_error_minus_correct_rt']:.4f})")
    print(f"  Pred mean RT: {best_metrics['pred_mean']:.3f}s (human: {best_metrics['true_mean']:.3f}s)")

    return {"best_metrics": best_metrics, "best_score": best_score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variational evidence → Wong-Wang synthesis smoke test"
    )
    parser.add_argument("--age_group", default="20-29", choices=["20-29", "80-89"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_ww", type=int, default=15)
    parser.add_argument("--evidence_time_steps", type=int, default=120)
    parser.add_argument("--ww_time_steps", type=int, default=500)
    parser.add_argument("--ww_dt", type=int, default=10)
    parser.add_argument("--readout_mode", default="soft_index")
    parser.add_argument("--smoke_eval", action="store_true")
    parser.add_argument("--smoke_max_trials", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise_ampa", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--j_offdiag_scale", type=float, default=None)
    parser.add_argument("--j_ext", type=float, default=None)
    parser.add_argument("--t0_seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    data_dir = data_root / args.age_group
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"

    train_dataset_full = StimulusDataset(str(train_csv))
    test_dataset_full = StimulusDataset(str(test_csv))

    if args.smoke_eval:
        max_trials = args.smoke_max_trials
        train_indices = _build_behavior_balanced_subset(
            train_dataset_full, max_trials=max_trials, seed=args.seed
        )
        test_indices = _build_behavior_balanced_subset(
            test_dataset_full, max_trials=max_trials, seed=args.seed + 1
        )
        train_dataset = Subset(train_dataset_full, train_indices)
        test_dataset = Subset(test_dataset_full, test_indices)
    else:
        train_dataset = train_dataset_full
        test_dataset = test_dataset_full

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"Train trials: {len(train_dataset)}, Test trials: {len(test_dataset)}")

    # Load stage1 backbone
    stage1_backbone, _ = load_stage1_model_with_metadata(args.device)
    stage1_cfg = Stage1EvidenceConfig(
        n_classes=4, feature_dim=512, hidden_dim=128, dropout_rate=0.10
    )
    sampler = SemiSupervisedEvidenceSampler(
        stage1_cfg, stage1_backbone=stage1_backbone
    ).to(args.device)

    readout_config = {
        "dt_ms": float(args.ww_dt),
        "choice_temperature": 0.10,
        "sigma_s": 0.05,
        "t0_mode": "fixed_global" if args.t0_seconds > 0 else "disabled",
        "t0_seconds": float(args.t0_seconds),
    }

    behavioral_loss_config = {
        "lambda_error_rate": 0.75,
        "lambda_error_sign": 0.75,
        "lambda_accuracy": 0.0,
        "lambda_response_nll": 1.0,
        "lambda_rt_mse": 0.0,
    }

    train_variational_ww(
        sampler=sampler,
        train_loader=train_loader,
        test_loader=test_loader,
        sampler_mode="variational",
        evidence_time_steps=args.evidence_time_steps,
        ww_time_steps=args.ww_time_steps,
        ww_dt=args.ww_dt,
        epochs_stage1=args.epochs_stage1,
        epochs_ww=args.epochs_ww,
        readout_mode=args.readout_mode,
        readout_config=readout_config,
        behavioral_loss_config=behavioral_loss_config,
        device=args.device,
        seed=args.seed,
        output_dir=str(output_root),
        lr=args.lr,
        noise_ampa=args.noise_ampa,
        threshold=args.threshold,
        j_offdiag_scale=args.j_offdiag_scale,
        j_ext=args.j_ext,
    )


if __name__ == "__main__":
    main()

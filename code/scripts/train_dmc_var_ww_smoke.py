#!/usr/bin/env python3
"""
DMC + Variational Evidence → Wong-Wang Synthesis Smoke Test

Combines:
  1. Time-varying variational evidence (Stage-1 uncertainty)
  2. DMC-like early flanker capture + late cognitive control (directional bias)

Hypothesis (from Phase 14 + Phase 17):
  - Phase 14: DMC on static logits fails because evidence is too deterministic
  - Phase 17: Var→WW produces symmetric errors (ΔRT≈0) because noise is directionless
  → DMC + Var→WW resolves both: stochastic evidence enables errors,
    directional DMC bias makes them faster than correct responses.

Design:
  1. Sample time-varying variational evidence [B, T, 4]
  2. Apply DMC time-varying modulation:
     - Early (t ~ auto_peak): boost flanker-congruent class
     - Late (t > selection_midpoint): suppress flanker class
  3. Feed modulated evidence into WW dynamics
  4. Soft-index readout → RT, choice
  5. Train with behavioral losses

Run from project root:
  source .venv/bin/activate
  python code/scripts/train_dmc_var_ww_smoke.py \
    --age_group 20-29 --data_root data/age_groups_matched \
    --output_root artifacts/results/rt_model_dmc_var_ww/smoke \
    --epochs_stage1 5 --epochs_ww 15 --smoke_eval --smoke_max_trials 1024 \
    --auto_strength 0.3 --selection_strength 0.4 --noise_ampa 0.08 --threshold 0.22 \
    --t0_seconds 0.25 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    to_jsonable,
    train_stage1_head,
)
from train_age_groups_efficient import (
    compute_human_stats_from_rts,
    evaluate_joint_behavior,
)
from train_variational_ww_smoke import (
    VariationalWWModel,
    compute_ww_readout,
    set_seed,
)
from vgg_wongwang_lim import (
    _build_alpha_pulse,
    apply_stage2_input_transform,
    compute_behavioral_losses,
)


# ── DMC Evidence Modulation ──────────────────────────────────────────

def build_dmc_time_multipliers(
    time_steps: int,
    dt_ms: float,
    auto_strength: float,
    auto_peak_s: float,
    selection_strength: float,
    selection_midpoint_s: float,
    selection_tau_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build DMC time-varying multipliers for flanker and target populations.

    Returns:
        flanker_mult: [T] — multiplier for flanker-congruent class
        target_mult:  [T] — multiplier for target class
        traces:       diagnostic tensors
    """
    time_axis = torch.arange(time_steps, device=device, dtype=dtype) * (dt_ms / 1000.0)

    # Early automatic capture (alpha pulse)
    auto_pulse = _build_alpha_pulse(time_axis, peak_s=auto_peak_s)

    # Late cognitive control (sigmoid gate)
    selection_gate = torch.sigmoid(
        (time_axis - selection_midpoint_s) / max(selection_tau_s, 1e-6)
    )

    # Flanker: 1 + auto - selection (clamped ≥ 0)
    #   Early: 1 + auto (boosted) → favors flanker class
    #   Late:  1 - selection (suppressed) → flanker influence removed
    flanker_mult = (
        1.0 + auto_strength * auto_pulse - selection_strength * selection_gate
    ).clamp_min(0.0)

    # Target: 1 - auto (suppressed early) + boost late
    #   Early: 1 - auto → target slightly suppressed (flanker dominates)
    #   Late:  1 + boost → target dominates
    target_boost = float(selection_strength) * 0.25  # modest late boost to target
    target_mult = (
        1.0 - auto_strength * auto_pulse * 0.5 + target_boost * selection_gate
    ).clamp_min(0.0)

    traces: Dict[str, Any] = {
        "time_axis": time_axis,
        "auto_pulse": auto_pulse,
        "selection_gate": selection_gate,
        "flanker_mult": flanker_mult,
        "target_mult": target_mult,
    }
    return flanker_mult, target_mult, traces


def apply_dmc_modulation(
    evidence: torch.Tensor,
    flanker_labels: torch.Tensor,
    target_labels: torch.Tensor,
    auto_strength: float,
    auto_peak_s: float,
    selection_strength: float,
    selection_midpoint_s: float,
    selection_tau_s: float,
    dt_ms: float,
    apply_to: str = "incongruent_only",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply DMC time-varying modulation to variational evidence sequences.

    Args:
        evidence: [B, T, C] variational evidence sequences
        flanker_labels: [B] flanker class index
        target_labels: [B] target class index
        ...

    Returns:
        modulated_evidence: [B, T, C] with DMC directional bias
        traces: diagnostic info
    """
    B, T, C = evidence.shape
    device = evidence.device
    dtype = evidence.dtype

    flanker_mult, target_mult, traces = build_dmc_time_multipliers(
        time_steps=T,
        dt_ms=dt_ms,
        auto_strength=auto_strength,
        auto_peak_s=auto_peak_s,
        selection_strength=selection_strength,
        selection_midpoint_s=selection_midpoint_s,
        selection_tau_s=selection_tau_s,
        device=device,
        dtype=dtype,
    )

    modulated = evidence.clone()

    # Determine which trials get modulation
    if apply_to == "incongruent_only":
        trial_mask = target_labels != flanker_labels
    elif apply_to == "all_trials":
        trial_mask = torch.ones(B, device=device, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown apply_to: {apply_to}")

    mod_idx = torch.nonzero(trial_mask, as_tuple=False).squeeze(1)
    if mod_idx.numel() == 0:
        return modulated, traces

    # Apply flanker modulation: modulated[b, :, f] *= flanker_mult[:]
    flanker_idx = flanker_labels[mod_idx]  # [N_mod]
    target_idx = target_labels[mod_idx]    # [N_mod]

    # Build time index for broadcasting: [N_mod, T]
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(mod_idx.numel(), T)

    # Flanker class modulation
    modulated[mod_idx.unsqueeze(1), t_idx, flanker_idx.unsqueeze(1)] *= flanker_mult.unsqueeze(0)

    # Target class modulation
    modulated[mod_idx.unsqueeze(1), t_idx, target_idx.unsqueeze(1)] *= target_mult.unsqueeze(0)

    traces["n_modulated_trials"] = int(mod_idx.numel())
    traces["mod_trial_indices"] = mod_idx
    return modulated, traces


# ── DMC + Var→WW Model ──────────────────────────────────────────────

class DMCVarWWModel(VariationalWWModel):
    """Variational evidence → DMC modulation → Wong-Wang bridge model."""

    def __init__(
        self,
        n_classes: int = 4,
        ww_time_steps: int = 120,
        ww_dt: int = 10,
        evidence_time_steps: int = 120,
        transform_mode: str = "softplus_centered",
        softplus_center: float = 1.0,
        noise_ampa: Optional[float] = None,
        threshold: Optional[float] = None,
        j_offdiag_scale: Optional[float] = None,
        j_ext: Optional[float] = None,
        freeze_ww_params: bool = True,
        # DMC parameters
        dmc_auto_strength: float = 0.3,
        dmc_auto_peak_s: float = 0.06,
        dmc_selection_strength: float = 0.4,
        dmc_selection_midpoint_s: float = 0.18,
        dmc_selection_tau_s: float = 0.06,
        dmc_apply_to: str = "incongruent_only",
    ):
        super().__init__(
            n_classes=n_classes,
            ww_time_steps=ww_time_steps,
            ww_dt=ww_dt,
            evidence_time_steps=evidence_time_steps,
            transform_mode=transform_mode,
            softplus_center=softplus_center,
            noise_ampa=noise_ampa,
            threshold=threshold,
            j_offdiag_scale=j_offdiag_scale,
            j_ext=j_ext,
            freeze_ww_params=freeze_ww_params,
        )
        self.dmc_auto_strength = dmc_auto_strength
        self.dmc_auto_peak_s = dmc_auto_peak_s
        self.dmc_selection_strength = dmc_selection_strength
        self.dmc_selection_midpoint_s = dmc_selection_midpoint_s
        self.dmc_selection_tau_s = dmc_selection_tau_s
        self.dmc_apply_to = dmc_apply_to
        self.ww_input_max = 1.0

    def forward(
        self,
        evidence_samples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        *,
        flanker_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = evidence_samples.shape[0]
        device = evidence_samples.device

        scale = torch.exp(self.log_scale)
        transformed = apply_stage2_input_transform(
            evidence_samples.reshape(B, -1),
            scale.expand(1),
            transform_mode=self.transform_mode,
            softplus_center=self.softplus_center,
        ).reshape(B, self.evidence_time_steps, self.n_classes)

        ww_input = self._resample_evidence(transformed)
        ww_input = torch.nan_to_num(ww_input, nan=0.0, posinf=self.ww_input_max, neginf=0.0).clamp(
            min=0.0,
            max=float(self.ww_input_max),
        )

        # ── DMC modulation ──
        dmc_traces: Dict[str, Any] = {}
        if flanker_labels is not None and target_labels is not None:
            ww_input, dmc_traces = apply_dmc_modulation(
                evidence=ww_input,
                flanker_labels=flanker_labels,
                target_labels=target_labels,
                auto_strength=self.dmc_auto_strength,
                auto_peak_s=self.dmc_auto_peak_s,
                selection_strength=self.dmc_selection_strength,
                selection_midpoint_s=self.dmc_selection_midpoint_s,
                selection_tau_s=self.dmc_selection_tau_s,
                dt_ms=float(self.ww.dt),
                apply_to=self.dmc_apply_to,
            )

        decision_times_class, trajectory, threshold = self.ww.inference(
            ww_input, generator=generator
        )

        evidence_traj = trajectory - threshold

        result: Dict[str, Any] = {
            "trajectory": trajectory,
            "evidence_traj": evidence_traj,
            "decision_times_class": decision_times_class,
            "threshold": threshold,
            "ww_input": ww_input,
        }
        if dmc_traces:
            result["dmc_traces"] = dmc_traces
        return result


# ── Helpers ───────────────────────────────────────────────────────────

def _get_state_dict(model: nn.Module) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def _snapshot_ww_params(model) -> Dict[str, Any]:
    """Capture scalar WW diagnostics for epoch-level tracking."""
    ww = model.ww
    with torch.no_grad():
        j_mat = ww.J_matrix.detach().cpu().numpy()
        j_mat_safe = np.nan_to_num(j_mat, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            eigvals = np.linalg.eigvalsh(j_mat_safe).tolist()
        except np.linalg.LinAlgError:
            eigvals = []
        return {
            "j_ext": float(ww.J_ext.detach().cpu().numpy()),
            "I_0": float(ww.I_0.detach().cpu().numpy()),
            "threshold": float(ww.threshold.detach().cpu().numpy()),
            "noise_ampa": float(ww.noise_ampa.detach().cpu().numpy()),
            "j_diag_mean": float(np.mean(np.diag(j_mat_safe))),
            "j_offdiag_mean": float(np.mean(j_mat_safe[~np.eye(ww.n_classes, dtype=bool)])),
            "j_eigvals": eigvals,
            "log_scale": float(model.log_scale.detach().cpu().numpy()),
            "scale": float(torch.exp(model.log_scale).detach().cpu().numpy()),
        }


# ── Training ─────────────────────────────────────────────────────────

def train_dmc_variational_ww(
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
    freeze_ww_params: bool = True,
    dmc_auto_strength: float = 0.3,
    dmc_auto_peak_s: float = 0.06,
    dmc_selection_strength: float = 0.4,
    dmc_selection_midpoint_s: float = 0.18,
    dmc_selection_tau_s: float = 0.06,
    dmc_apply_to: str = "incongruent_only",
    sigma_evidence_noise: float = 0.0,
    stage1_uncertainty_gain: float = 1.0,
    cached_train_inputs: Optional[Dict[str, np.ndarray]] = None,
    cached_test_inputs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    # --- gate thresholds for ΔRT-aware checkpoint selection ---
    neg_drt_min_acc = float(behavioral_loss_config.get("neg_drt_min_acc", 0.75))
    neg_drt_min_resp = float(behavioral_loss_config.get("neg_drt_min_resp", 0.65))
    neg_drt_min_error = float(behavioral_loss_config.get("neg_drt_min_error", 0.02))
    set_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: warm-start variational head ---
    stage1_complete = output_path / "stage1_complete.json"
    if stage1_complete.exists():
        print("Stage 1 already complete, skipping warm-start.")
    else:
        _write_json(
            output_path / "stage1_started.json",
            {"phase": "stage1_warmstart", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"},
        )
        print("Stage 1: Training variational evidence head...")
        train_stage1_head(
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
        _write_json(
            output_path / "stage1_complete.json",
            {"phase": "stage1_warmstart", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "completed"},
        )

    # --- Bundle evidence for WW training ---
    bundle_complete = output_path / "bundle_complete.json"
    if bundle_complete.exists():
        print("Evidence already bundled, skipping.")
    else:
        _write_json(
            output_path / "bundle_started.json",
            {"phase": "evidence_bundling", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"},
        )
        print("Bundling evidence sequences (with flanker labels)...")
        sampler.eval()

    bundle_heartbeat_path = output_path / "bundle.heartbeat.json"

    def _write_bundle_heartbeat(payload: Dict[str, Any]) -> None:
        _write_json(bundle_heartbeat_path, payload)

    def _bundle_from_cached_inputs(cached: Dict[str, np.ndarray], gen_seed: int, split_name: str) -> Dict[str, np.ndarray]:
        feature_key = "pooled_features" if "pooled_features" in cached else "features"
        if feature_key not in cached:
            raise ValueError(f"CACHED_FEATURES_MISSING: split={split_name}")
        if "logits" not in cached:
            raise ValueError(f"CACHED_LOGITS_MISSING: split={split_name}")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(gen_seed)
        features = torch.tensor(np.asarray(cached[feature_key], dtype=np.float32), dtype=torch.float32, device=device)
        logits = torch.tensor(np.asarray(cached["logits"], dtype=np.float32), dtype=torch.float32, device=device)
        rows: Dict[str, list] = {"evidence_samples": []}
        batch_size = 512
        n_rows = int(features.shape[0])
        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                payload = sampler.sample_evidence_sequence(
                    pooled_features=features[start:end],
                    base_logits=logits[start:end],
                    time_steps=evidence_time_steps,
                    sampler_mode=sampler_mode,
                    uncertainty_gain=stage1_uncertainty_gain,
                    generator=gen,
                )
                rows["evidence_samples"].append(payload["evidence_samples"].detach().cpu().numpy())
                _write_bundle_heartbeat(
                    {
                        "phase": "evidence_bundling",
                        "status": "running",
                        "source": "cached_features",
                        "split": split_name,
                        "rows_done": int(end),
                        "rows_total": int(n_rows),
                    }
                )

        return {
            "evidence_samples": np.concatenate(rows["evidence_samples"], axis=0),
            "target_labels": np.asarray(cached["target_labels"], dtype=np.int64),
            "response_labels": np.asarray(cached["response_labels"], dtype=np.int64),
            "flanker_labels": np.asarray(cached["flanker_labels"], dtype=np.int64),
            "rts": np.asarray(cached["rts"], dtype=np.float32),
            "congruency": np.asarray(cached["congruency"], dtype=np.int64),
        }

    def _bundle_evidence(loader: DataLoader, gen_seed: int, split_name: str) -> Dict[str, np.ndarray]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(gen_seed)
        rows: Dict[str, list] = {
            "evidence_samples": [],
            "target_labels": [],
            "response_labels": [],
            "flanker_labels": [],
            "rts": [],
            "congruency": [],
        }
        with torch.no_grad():
            rows_done = 0
            try:
                total_rows = int(len(loader.dataset))
            except Exception:
                total_rows = 0
            for batch in loader:
                images = batch["image"].to(device)
                payload = sampler.sample_from_images(
                    images=images,
                    time_steps=evidence_time_steps,
                    sampler_mode=sampler_mode,
                    uncertainty_gain=stage1_uncertainty_gain,
                    generator=gen,
                )
                rows["evidence_samples"].append(
                    payload["evidence_samples"].detach().cpu().numpy()
                )
                rows["target_labels"].append(batch["target_label"].cpu().numpy())
                rows["response_labels"].append(batch["response_label"].cpu().numpy())
                rows["flanker_labels"].append(batch["flanker_label"].cpu().numpy())
                rows["rts"].append(batch["rt"].cpu().numpy())
                rows["congruency"].append(batch["congruency"].cpu().numpy())
                rows_done += int(images.shape[0])
                _write_bundle_heartbeat(
                    {
                        "phase": "evidence_bundling",
                        "status": "running",
                        "source": "images",
                        "split": split_name,
                        "rows_done": int(rows_done),
                        "rows_total": int(total_rows),
                    }
                )

        return {
            key: np.concatenate([np.asarray(v) for v in values], axis=0)
            for key, values in rows.items()
        }

    if cached_train_inputs is not None and cached_test_inputs is not None:
        print("Bundling evidence sequences from cached Stage-1 features...", flush=True)
        train_bundle = _bundle_from_cached_inputs(cached_train_inputs, seed, "train")
        test_bundle = _bundle_from_cached_inputs(cached_test_inputs, seed + 1, "test")
    else:
        train_bundle = _bundle_evidence(train_loader, seed, "train")
        test_bundle = _bundle_evidence(test_loader, seed + 1, "test")

    if not bundle_complete.exists():
        _write_json(
            output_path / "bundle_complete.json",
            {"phase": "evidence_bundling", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "completed",
             "n_train": int(len(train_bundle["evidence_samples"])),
             "n_test": int(len(test_bundle["evidence_samples"]))},
        )
    _write_bundle_heartbeat(
        {
            "phase": "evidence_bundling",
            "status": "completed",
            "n_train": int(len(train_bundle["evidence_samples"])),
            "n_test": int(len(test_bundle["evidence_samples"])),
        }
    )

    # --- Stage 2: Train DMC+WW on variational evidence ---
    _write_json(
        output_path / "stage2_started.json",
        {"phase": "stage2_ww_training", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running",
         "epochs_ww": epochs_ww},
    )
    stage2_heartbeat_path = output_path / "stage2.heartbeat.json"
    _write_json(
        stage2_heartbeat_path,
        {
            "phase": "stage2_ww_training",
            "status": "running",
            "epochs_requested": int(epochs_ww),
            "epoch": 0,
        },
    )
    print(f"Stage 2: Training DMC+WW ({epochs_ww} epochs)...", flush=True)
    if sigma_evidence_noise > 0:
        print(f"  Sensory noise: sigma={sigma_evidence_noise:.4f} (age-dependent evidence degradation)", flush=True)
    n_train = len(train_bundle["evidence_samples"])
    n_incongruent = int((train_bundle["target_labels"] != train_bundle["flanker_labels"]).sum())
    print(f"  Train trials: {n_train} ({n_incongruent} incongruent)", flush=True)

    model = DMCVarWWModel(
        n_classes=4,
        ww_time_steps=ww_time_steps,
        ww_dt=ww_dt,
        evidence_time_steps=evidence_time_steps,
        noise_ampa=noise_ampa,
        threshold=threshold,
        j_offdiag_scale=j_offdiag_scale,
        j_ext=j_ext,
        freeze_ww_params=freeze_ww_params,
        dmc_auto_strength=dmc_auto_strength,
        dmc_auto_peak_s=dmc_auto_peak_s,
        dmc_selection_strength=dmc_selection_strength,
        dmc_selection_midpoint_s=dmc_selection_midpoint_s,
        dmc_selection_tau_s=dmc_selection_tau_s,
        dmc_apply_to=dmc_apply_to,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    train_ev = torch.tensor(train_bundle["evidence_samples"], dtype=torch.float32)
    train_target = torch.tensor(train_bundle["target_labels"], dtype=torch.long)
    train_response = torch.tensor(train_bundle["response_labels"], dtype=torch.long)
    train_flanker = torch.tensor(train_bundle["flanker_labels"], dtype=torch.long)
    train_rt = torch.tensor(train_bundle["rts"], dtype=torch.float32)

    test_ev = torch.tensor(test_bundle["evidence_samples"], dtype=torch.float32)
    test_flanker = torch.tensor(test_bundle["flanker_labels"], dtype=torch.long)
    test_target = torch.tensor(test_bundle["target_labels"], dtype=torch.long)

    best_score = -float("inf")
    best_state: Optional[Dict[str, np.ndarray]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_epoch_drt = float("inf")
    best_neg_drt_gated = float("inf")          # most negative ΔRT passing gates
    best_neg_drt_state: Optional[Dict[str, np.ndarray]] = None
    best_neg_drt_metrics: Optional[Dict[str, Any]] = None
    best_neg_drt_preds: Optional[Dict[str, np.ndarray]] = None
    best_neg_drt_epoch: int = -1
    all_epoch_metrics: list = []
    pred_rt_np: Optional[np.ndarray] = None
    pred_choice_np: Optional[np.ndarray] = None

    for epoch in range(epochs_ww):
        # --- snapshot WW parameters before training step (for diagnostics) ---
        ww_param_snapshot = _snapshot_ww_params(model)

        # --- train ---
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
            flanker_b = train_flanker[idx].to(device)
            rt_b = train_rt[idx].to(device)

            if sigma_evidence_noise > 0:
                ev_batch = ev_batch + torch.randn_like(ev_batch) * sigma_evidence_noise

            ww_output = model(ev_batch, flanker_labels=flanker_b, target_labels=target_b)
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

            if not torch.isfinite(loss):
                print(f"  Skipping non-finite batch loss at epoch {epoch + 1}, batch_start={start}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_val += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(f"DMC_WW_ALL_BATCHES_NONFINITE: epoch={epoch + 1}")

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            test_ev_batch = test_ev.to(device)
            if sigma_evidence_noise > 0:
                test_ev_batch = test_ev_batch + torch.randn_like(test_ev_batch) * sigma_evidence_noise
            ww_test = model(test_ev_batch, flanker_labels=test_flanker.to(device), target_labels=test_target.to(device))
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
        if pred_choice_np is None:
            raise RuntimeError("DMC+WW evaluation did not produce predicted choices")
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
        drt = float(metrics["error_minus_correct_rt"])
        acc = float(metrics["model_accuracy"])

        all_epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss_val / max(n_batches, 1),
                "beh_opt": score,
                "acc": acc,
                "resp_agree": float(metrics["response_agreement"]),
                "pred_mean": float(metrics["pred_mean"]),
                "err_correct_delta": drt,
                "cong_gap": float(metrics["model_congruency_rt_gap"]),
                **{f"ww_{k}": v for k, v in ww_param_snapshot.items()},
            }
        )

        print(
            f"  Epoch {epoch + 1:02d}/{epochs_ww}: "
            f"loss={total_loss_val / max(n_batches, 1):.4f}, "
            f"beh_opt={score:.4f}, acc={metrics['model_accuracy']:.4f}, "
            f"resp_agree={metrics['response_agreement']:.4f}, "
            f"ΔRT={drt:+.4f}, "
            f"pred_mean={metrics['pred_mean']:.3f}s",
            flush=True,
        )
        _write_json(
            stage2_heartbeat_path,
            {
                "phase": "stage2_ww_training",
                "status": "running",
                "epochs_requested": int(epochs_ww),
                "epoch": int(epoch + 1),
                "loss": float(total_loss_val / max(n_batches, 1)),
                "beh_opt": float(score),
                "acc": float(acc),
                "response_agreement": float(metrics["response_agreement"]),
                "error_minus_correct_rt": float(drt),
                "pred_mean": float(metrics["pred_mean"]),
            },
        )

        # Track best by beh_opt
        if score > best_score:
            best_score = score
            best_state = _get_state_dict(model)
            best_metrics = metrics

        # Track best overall ΔRT
        if not np.isnan(drt) and drt < best_epoch_drt:
            best_epoch_drt = drt

        # Track best *gated* negative ΔRT checkpoint
        resp_agree = float(metrics["response_agreement"])
        model_error = 1.0 - acc
        if drt < best_neg_drt_gated and acc >= neg_drt_min_acc and resp_agree >= neg_drt_min_resp and model_error >= neg_drt_min_error:
            best_neg_drt_gated = drt
            best_neg_drt_epoch = epoch + 1
            best_neg_drt_state = _get_state_dict(model)
            best_neg_drt_metrics = metrics
            best_neg_drt_preds = {
                "pred_rt": pred_rt_np.copy(),
                "pred_choice": pred_choice_np.copy(),
                "target_labels": test_bundle["target_labels"].copy(),
                "response_labels": test_bundle["response_labels"].copy(),
                "congruency": test_bundle["congruency"].copy(),
                "true_rt": test_bundle["rts"].copy(),
            }

    if best_state is None:
        raise RuntimeError("No DMC+WW checkpoints evaluated")
    if best_metrics is None:
        raise RuntimeError("Best DMC+WW metrics were never recorded")
    if pred_rt_np is None or pred_choice_np is None:
        raise RuntimeError("Final DMC+WW prediction payload was never materialized")

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
            "dmc_auto_strength": dmc_auto_strength,
            "dmc_auto_peak_s": dmc_auto_peak_s,
            "dmc_selection_strength": dmc_selection_strength,
            "dmc_selection_midpoint_s": dmc_selection_midpoint_s,
            "dmc_selection_tau_s": dmc_selection_tau_s,
            "dmc_apply_to": dmc_apply_to,
            "sigma_evidence_noise": sigma_evidence_noise,
            "stage1_uncertainty_gain": float(stage1_uncertainty_gain),
            "neg_drt_gate_min_acc": neg_drt_min_acc,
            "neg_drt_gate_min_resp": neg_drt_min_resp,
            "neg_drt_gate_min_error": neg_drt_min_error,
        },
    )
    # beh_opt checkpoint
    np.savez(output_path / "best_model_params.npz", **best_state)
    # gated neg-ΔRT checkpoint (if any epoch passed gates)
    if best_neg_drt_state is not None:
        np.savez(output_path / "best_neg_drt_model_params.npz", **best_neg_drt_state)
        if best_neg_drt_metrics is None:
            raise RuntimeError("Missing gated negative-ΔRT metrics despite existing gated checkpoint")
        _write_json(output_path / "metrics_neg_drt.json", best_neg_drt_metrics)
    np.savez_compressed(
        output_path / "predictions_smoke.npz",
        pred_rt=pred_rt_np,
        pred_choice=pred_choice_np,
        target_labels=test_bundle["target_labels"],
        response_labels=test_bundle["response_labels"],
        congruency=test_bundle["congruency"],
        true_rt=test_bundle["rts"],
    )
    # Also save predictions from the most negative ΔRT epoch
    if best_neg_drt_preds is not None:
        np.savez_compressed(
            output_path / "predictions_neg_drt.npz",
            **best_neg_drt_preds,
        )
        print(f"  Saved gated neg-ΔRT predictions (ΔRT={best_neg_drt_gated:+.4f}, epoch={best_neg_drt_epoch})", flush=True)

    _write_json(
        output_path / "run_complete.json",
        {
            "variant": "dmc_variational_ww_synthesis",
            "best_score": best_score,
            "best_epoch": len(all_epoch_metrics),
            "best_acc": float(best_metrics["model_accuracy"]),
            "best_resp": float(best_metrics["response_agreement"]),
            "best_drt": best_epoch_drt,
            "best_neg_drt_gated": best_neg_drt_gated if best_neg_drt_gated < float("inf") else None,
            "best_neg_drt_epoch": best_neg_drt_epoch if best_neg_drt_epoch > 0 else None,
            "neg_drt_gate_config": {
                "min_acc": neg_drt_min_acc,
                "min_resp": neg_drt_min_resp,
                "min_error": neg_drt_min_error,
            },
        },
    )
    _write_json(
        stage2_heartbeat_path,
        {
            "phase": "stage2_ww_training",
            "status": "completed",
            "epochs_requested": int(epochs_ww),
            "epoch": int(len(all_epoch_metrics)),
            "best_score": float(best_score),
            "best_drt": float(best_epoch_drt),
        },
    )

    # --- final summary banner ---
    print(f"\n{'='*60}", flush=True)
    print(f"  DMC+Var→WW Training Complete", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Best beh_opt checkpoint (epoch {len(all_epoch_metrics)}):", flush=True)
    print(f"    score={best_score:.4f}", flush=True)
    print(f"    acc={best_metrics['model_accuracy']:.4f} (human: {best_metrics['human_accuracy']:.4f})", flush=True)
    print(f"    resp_agree={best_metrics['response_agreement']:.4f}", flush=True)
    print(f"    ΔRT={best_metrics['error_minus_correct_rt']:+.4f} (human: {best_metrics['human_error_minus_correct_rt']:+.4f})", flush=True)
    print(f"    pred_mean={best_metrics['pred_mean']:.3f}s (human: {best_metrics['true_mean']:.3f}s)", flush=True)
    print(f"  Best ΔRT (any epoch): {best_epoch_drt:+.4f}", flush=True)
    if best_neg_drt_state is not None:
        print(f"\n  ★ GATED NEG-ΔRT checkpoint (acc≥{neg_drt_min_acc}, resp≥{neg_drt_min_resp}, err≥{neg_drt_min_error}):", flush=True)
        print(f"    epoch={best_neg_drt_epoch}, ΔRT={best_neg_drt_gated:+.4f}", flush=True)
        if best_neg_drt_metrics is None:
            raise RuntimeError("Missing gated negative-ΔRT metrics at final reporting time")
        print(f"    acc={best_neg_drt_metrics['model_accuracy']:.4f}, resp={best_neg_drt_metrics['response_agreement']:.4f}", flush=True)
    else:
        print(f"\n  ⚠ No epoch passed neg-ΔRT gates (acc≥{neg_drt_min_acc}, resp≥{neg_drt_min_resp}, err≥{neg_drt_min_error})", flush=True)
    print(f"{'='*60}", flush=True)

    return {"best_metrics": best_metrics, "best_score": best_score, "best_drt": best_epoch_drt}


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DMC + Var→WW synthesis smoke test")
    parser.add_argument("--age_group", default="20-29")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_ww", type=int, default=15)
    parser.add_argument("--evidence_time_steps", type=int, default=120)
    parser.add_argument("--ww_time_steps", type=int, default=120)
    parser.add_argument("--ww_dt", type=int, default=10)
    parser.add_argument("--readout_mode", default="soft_index")
    parser.add_argument("--smoke_eval", action="store_true")
    parser.add_argument("--smoke_max_trials", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    # WW parameters
    parser.add_argument("--noise_ampa", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--j_offdiag_scale", type=float, default=None)
    parser.add_argument("--j_ext", type=float, default=None)
    parser.add_argument("--t0_seconds", type=float, default=0.0)
    # DMC parameters
    parser.add_argument("--auto_strength", type=float, default=0.3)
    parser.add_argument("--auto_peak_s", type=float, default=0.06)
    parser.add_argument("--selection_strength", type=float, default=0.4)
    parser.add_argument("--selection_midpoint_s", type=float, default=0.18)
    parser.add_argument("--selection_tau_s", type=float, default=0.06)
    parser.add_argument("--apply_to", default="incongruent_only",
                        choices=["incongruent_only", "all_trials"])
    parser.add_argument("--train_ww_params", action="store_true",
                        help="Keep noise_ampa and threshold trainable (instead of freezing)")
    # Sensory degradation: age-dependent Gaussian noise on Stage-1 evidence
    parser.add_argument("--sigma_evidence_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to evidence before WW (models sensory degradation)")
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
        "lambda_error_sign": 1.5,   # stronger penalty on ΔRT sign
        "lambda_accuracy": 0.0,
        "lambda_response_nll": 1.0,
        "lambda_rt_mse": 0.0,
    }

    train_dmc_variational_ww(
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
        freeze_ww_params=not args.train_ww_params,
        dmc_auto_strength=args.auto_strength,
        dmc_auto_peak_s=args.auto_peak_s,
        dmc_selection_strength=args.selection_strength,
        dmc_selection_midpoint_s=args.selection_midpoint_s,
        dmc_selection_tau_s=args.selection_tau_s,
        dmc_apply_to=args.apply_to,
        sigma_evidence_noise=args.sigma_evidence_noise,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vgg_wongwang_lim import VGGFeatureExtractor


@dataclass
class Stage1EvidenceConfig:
    n_classes: int = 4
    feature_dim: int = 512
    hidden_dim: int = 256
    dropout_rate: float = 0.10
    sigma_floor: float = 1e-3
    sigma_ceiling: float = 3.0


def build_weak_augmentation(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.10),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        ]
    )


def build_strong_augmentation(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
    )


class VGGFeatureTap(VGGFeatureExtractor):
    def forward_features_and_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        original_device = x.device
        if original_device.type == "mps":
            x = self.avgpool(x.cpu())
            x = x.to(original_device)
        else:
            x = self.avgpool(x)
        pooled = torch.flatten(x, 1)
        logits = self.classifier(pooled)
        return pooled, logits


class Stage1BackboneProtocol(Protocol):
    def forward_features_and_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...


class VariationalEvidenceHead(nn.Module):
    def __init__(self, config: Stage1EvidenceConfig):
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(config.hidden_dim, config.n_classes)
        self.log_sigma_head = nn.Linear(config.hidden_dim, config.n_classes)

    def forward(self, pooled_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(pooled_features)
        mu = self.mu_head(hidden)
        raw_sigma = F.softplus(self.log_sigma_head(hidden))
        sigma = raw_sigma.clamp(min=self.config.sigma_floor, max=self.config.sigma_ceiling)
        return mu, sigma


class MCDropoutEvidenceHead(nn.Module):
    def __init__(self, config: Stage1EvidenceConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.feature_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.n_classes)

    def _apply_seeded_dropout(
        self,
        tensor: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        keep_prob = 1.0 - float(self.config.dropout_rate)
        if keep_prob <= 0.0:
            raise ValueError("dropout_rate must be < 1.0")
        mask = torch.rand(
            tensor.shape,
            generator=generator,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        mask = (mask < keep_prob).to(dtype=tensor.dtype) / keep_prob
        return tensor * mask

    def forward(self, pooled_features: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        hidden = F.relu(self.fc1(pooled_features), inplace=True)
        hidden = self._apply_seeded_dropout(hidden, generator=generator)
        hidden = F.relu(self.fc2(hidden), inplace=True)
        hidden = self._apply_seeded_dropout(hidden, generator=generator)
        return self.out(hidden)


class SemiSupervisedEvidenceSampler(nn.Module):
    def __init__(
        self,
        config: Stage1EvidenceConfig,
        *,
        stage1_backbone: Optional[Stage1BackboneProtocol] = None,
    ):
        super().__init__()
        self.config = config
        self.stage1_backbone = stage1_backbone or VGGFeatureTap(pretrained=False, n_classes=config.n_classes)
        self.variational_head = VariationalEvidenceHead(config)
        self.mc_dropout_head = MCDropoutEvidenceHead(config)

    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stage1_backbone.forward_features_and_logits(images)

    def sample_evidence_sequence(
        self,
        *,
        pooled_features: torch.Tensor,
        base_logits: torch.Tensor,
        time_steps: int,
        sampler_mode: str,
        uncertainty_gain: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if sampler_mode == "deterministic":
            evidence = base_logits.unsqueeze(1).repeat(1, time_steps, 1)
            sigma = torch.zeros_like(base_logits)
            mu = base_logits
        elif sampler_mode == "variational":
            mu, sigma = self.variational_head(pooled_features)
            scaled_sigma = sigma * float(uncertainty_gain)
            randn_kwargs = {
                "device": pooled_features.device,
                "dtype": pooled_features.dtype,
            }
            if generator is not None:
                try:
                    randn_kwargs["generator"] = generator
                except Exception:
                    pass
            try:
                noise = torch.randn(
                    (pooled_features.shape[0], time_steps, self.config.n_classes),
                    **randn_kwargs,
                )
            except RuntimeError as exc:
                if generator is not None and pooled_features.device.type not in {"cpu", "cuda"}:
                    noise = torch.randn(
                        (pooled_features.shape[0], time_steps, self.config.n_classes),
                        device=pooled_features.device,
                        dtype=pooled_features.dtype,
                    )
                else:
                    raise exc
            evidence = mu.unsqueeze(1) + scaled_sigma.unsqueeze(1) * noise
            sigma = scaled_sigma
        elif sampler_mode == "mc_dropout":
            samples = []
            for _ in range(time_steps):
                samples.append(self.mc_dropout_head(pooled_features, generator=generator))
            evidence = torch.stack(samples, dim=1)
            mu = evidence.mean(dim=1)
            if float(uncertainty_gain) != 1.0:
                evidence = mu.unsqueeze(1) + (evidence - mu.unsqueeze(1)) * float(uncertainty_gain)
            sigma = evidence.std(dim=1, unbiased=False).clamp(min=self.config.sigma_floor)
        else:
            raise ValueError(f"Unknown sampler_mode: {sampler_mode}")

        metrics = compute_uncertainty_metrics(evidence_samples=evidence)
        return {
            "evidence_samples": evidence,
            "evidence_sample_mu": mu,
            "evidence_sample_sigma": sigma,
            **metrics,
        }

    def sample_from_images(
        self,
        *,
        images: torch.Tensor,
        time_steps: int,
        sampler_mode: str,
        uncertainty_gain: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled_features, base_logits = self.encode_images(images)
        payload = self.sample_evidence_sequence(
            pooled_features=pooled_features,
            base_logits=base_logits,
            time_steps=time_steps,
            sampler_mode=sampler_mode,
            uncertainty_gain=uncertainty_gain,
            generator=generator,
        )
        payload["pooled_features"] = pooled_features
        payload["base_logits"] = base_logits
        return payload


def load_stage1_backbone(
    *,
    checkpoint_path: Path,
    device: str,
    n_classes: int = 4,
) -> VGGFeatureTap:
    model = VGGFeatureTap(pretrained=False, n_classes=n_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def compute_uncertainty_metrics(evidence_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    if evidence_samples.ndim != 3:
        raise ValueError("evidence_samples must have shape [batch, time_steps, n_classes]")
    sample_mean = evidence_samples.mean(dim=1)
    sample_var = evidence_samples.var(dim=1, unbiased=False)
    probs = torch.softmax(evidence_samples, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[..., 0]
    else:
        margin = top2[..., 0] - top2[..., 1]
    stage1_uncertainty_scalar = sample_var.mean(dim=-1)
    return {
        "entropy_mean": entropy.mean(dim=1),
        "entropy_var": entropy.var(dim=1, unbiased=False),
        "margin_mean": margin.mean(dim=1),
        "margin_var": margin.var(dim=1, unbiased=False),
        "stage1_uncertainty_scalar": stage1_uncertainty_scalar,
        "sample_mean": sample_mean,
        "sample_var": sample_var,
    }

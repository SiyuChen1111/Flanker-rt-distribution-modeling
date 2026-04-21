import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from vgg_wongwang_lim import DiffDecisionMultiClass, VGGFeatureExtractor


COUPLED_CHOICE_READOUT = "first_crosser_coupled.v1"
COUPLED_MAX_EXTRA_STEPS = 4000


def coupled_choice_from_rollout(decision_times: torch.Tensor, traj: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    crossed_mask = (traj > threshold.view(1, 1, 1)).any(dim=1)
    large_time = decision_times.max(dim=1, keepdim=True).values + 1.0
    true_crossing_times = torch.where(crossed_mask, decision_times, large_time.expand_as(decision_times))
    return true_crossing_times.argmin(dim=1)


class AccumulatorRaceDecisionV2(nn.Module):
    def __init__(self, n_classes=4, dt=10, time_steps=120, threshold=0.5, noise_std=0.02, competition_mix=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.dt = dt
        self.time_steps = time_steps
        self.register_buffer('threshold', torch.tensor(float(threshold), dtype=torch.float32))
        self.register_buffer('competition_mix', torch.tensor(float(competition_mix), dtype=torch.float32))
        self.input_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.leak = nn.Parameter(torch.full((n_classes,), 0.08, dtype=torch.float32))
        self.self_excitation = nn.Parameter(torch.full((n_classes,), 0.12, dtype=torch.float32))
        self.inhibition = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.competition_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.noise_std = nn.Parameter(torch.tensor(float(noise_std), dtype=torch.float32))
        self.evidence_proj = nn.Linear(1, 1)

    def _momentary_evidence(self, logits):
        # Map logits -> strictly positive evidence without discarding negative logit structure.
        # Centering per trial keeps relative ordering informative while remaining shift-invariant.
        centered = logits * self.input_scale
        centered = centered - centered.mean(dim=1, keepdim=True)
        x = F.softplus(centered).unsqueeze(-1)
        return self.evidence_proj(x).squeeze(-1)

    def rollout(
        self,
        logits,
        generator: Optional[torch.Generator] = None,
        *,
        ensure_crossing: bool = False,
        max_extra_steps: int = 0,
        require_crossing: bool = False,
    ):
        batch, n_classes = logits.shape
        device = logits.device
        acc = torch.zeros(batch, n_classes, device=device)
        traj_steps = []
        dsdt_steps = []
        evidence = self._momentary_evidence(logits)
        leak = F.softplus(self.leak)
        self_exc = F.softplus(self.self_excitation)
        inhib = F.softplus(self.inhibition)
        competition_gain = F.softplus(self.competition_gain)
        noise_std = F.softplus(self.noise_std)
        threshold_value = torch.as_tensor(self.threshold, device=device, dtype=torch.float32)
        competition_mix_value = max(0.0, min(1.0, float(torch.as_tensor(self.competition_mix).detach().cpu().item())))
        total_steps = int(self.time_steps) + (int(max_extra_steps) if ensure_crossing else 0)

        def sample_noise() -> torch.Tensor:
            if generator is None:
                return torch.randn(batch, n_classes, device=device)
            try:
                return torch.randn(batch, n_classes, device=device, generator=generator)
            except TypeError:
                # Some torch backends do not accept a generator kwarg.
                return torch.randn(batch, n_classes, device=device)

        crossed_any = torch.zeros(batch, dtype=torch.bool, device=device)

        for t in range(total_steps):
            total_other = acc.sum(dim=1, keepdim=True) - acc
            total = acc.sum(dim=1, keepdim=True)
            normalized_other = total_other / (total + 1e-6)
            mix = competition_mix_value
            competition_term = (1.0 - mix) * total_other + mix * competition_gain * normalized_other
            drive = evidence + self_exc * acc - inhib * competition_term - leak * acc
            noise = sample_noise() * noise_std
            dsdt = F.softplus(drive + noise) - acc
            acc = torch.clamp(acc + 0.2 * dsdt, min=0.0)
            traj_steps.append(acc)
            dsdt_steps.append(dsdt)
            crossed_any = crossed_any | (acc > threshold_value).any(dim=1)
            if ensure_crossing and bool(crossed_any.all()) and t + 1 >= int(self.time_steps):
                break

        traj = torch.stack(traj_steps, dim=1)
        dsdt_traj = torch.stack(dsdt_steps, dim=1)

        if require_crossing and not bool(crossed_any.all()):
            missing = (~crossed_any).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            raise RuntimeError(
                f"COUPLED_MODE_REQUIRES_REAL_CROSSING: {len(missing)} samples never crossed threshold even after {traj.size(1)} steps; sample_indices={missing[:10]}"
            )

        threshold = threshold_value
        decision_times = torch.as_tensor(DiffDecisionMultiClass.apply(traj - threshold, dsdt_traj, self.dt, traj.size(1)))
        return decision_times / 1000.0, traj, threshold

    def forward(self, logits):
        decision_times, _, _ = self.rollout(logits)
        return decision_times

    def inference(self, logits):
        return self.rollout(logits)


class VGGAccumulatorRNNLIMV2(nn.Module):
    def __init__(
        self,
        pretrained=True,
        freeze_features=False,
        n_classes=4,
        dropout_rate=0.5,
        dt=10,
        time_steps=120,
        threshold=0.5,
        noise_std=0.02,
        choice_readout: str = COUPLED_CHOICE_READOUT,
    ):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(pretrained=pretrained, freeze_features=freeze_features, n_classes=n_classes, dropout_rate=dropout_rate)
        self.decision = AccumulatorRaceDecisionV2(n_classes=n_classes, dt=dt, time_steps=time_steps, threshold=threshold, noise_std=noise_std)
        self.choice_readout = str(choice_readout)

    def forward(self, x, return_logits=False):
        logits = self.feature_extractor(x)
        coupled_mode = self.choice_readout == COUPLED_CHOICE_READOUT
        decision_times, traj, threshold = self.decision.rollout(
            logits,
            ensure_crossing=bool(coupled_mode),
            require_crossing=bool(coupled_mode),
            max_extra_steps=COUPLED_MAX_EXTRA_STEPS,
        )
        if coupled_mode:
            pred_choice = coupled_choice_from_rollout(decision_times, traj, threshold)
        else:
            pred_choice = decision_times.argmin(dim=1)
        final_dt = decision_times[torch.arange(decision_times.size(0), device=decision_times.device), pred_choice]
        if return_logits:
            return logits, decision_times, final_dt, traj, threshold, pred_choice
        return logits, final_dt, pred_choice

    def get_logits(self, x):
        return self.feature_extractor(x)

    def get_decision_times(self, logits):
        return self.decision(logits)

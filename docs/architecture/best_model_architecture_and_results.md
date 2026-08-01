# Best Model Architecture, Implementation Flow, Results, and Limitations

> Current status: this document preserves an earlier architecture/result snapshot. For the current retained R5 representative-subset model and the latest supervisor follow-up diagnostics, start with `README.md`, `docs/model_framework_summary.md`, `docs/current_results_and_limitations.md`, and `docs/r5-supervisor-followup.md`.

**Date:** 2026-05-06 (updated with Phase 15 Var→WW synthesis + MC Dropout improvement plan)
**Scope:** Current best-performing model in the VGG → Wong-Wang → RT prediction pipeline for LIM/Flanker task
**Status:** Live document — update when a new configuration clears promotion gates

---

## 1. Architecture Overview

The model is a two-stage biologically-grounded pipeline for the Lost in Migration (LIM) four-choice task:

```
Input Image (128×128×3)
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: VGG16 Feature Extractor   │
│  (pretrained on ImageNet)          │
│  ├─ features (conv layers 1–13)    │
│  ├─ AdaptiveAvgPool2d(1,1)         │
│  └─ classifier: FC(512→4096)→ReLU  │
│       →Dropout→FC(4096→4096)→ReLU  │
│       →Dropout→FC(4096→4)          │
│  Output: logits (L, R, U, D)       │
└─────────────────────────────────┘
    │  [4-dim logit vector]
    ▼
┌─────────────────────────────────┐
│  Stage 2 Input Transform          │
│  softplus_centered:               │
│    softplus(logits × scale − c)   │
│  scale ∈ [0.15, 0.45] (tuned)     │
│  c = 0.0 (shift parameter)        │
└─────────────────────────────────┘
    │  [4-dim non-negative input]
    ▼
┌─────────────────────────────────┐
│  Wong-Wang Multi-Class Decision  │
│  4 competing neural populations  │
│  (L, R, U, D)                    │
│  Parameters:                     │
│  ├─ a=270, b=108, d=0.154       │
│  ├─ gamma=0.641, tau_s=100ms    │
│  ├─ J_matrix: 4×4 learned       │
│  │   diag=0.261 (自激励)        │
│  │   off-diag=−0.050 (侧抑制) │
│  ├─ J_ext=0.016 (input gain)    │
│  ├─ I_0=0.326 (baseline current)│
│  ├─ noise_ampa=0.02 (noise σ)   │
│  ├─ tau_ampa=2ms (noise filter) │
│  ├─ threshold=0.5 (learnable)   │
│  └─ dt=10ms, 500 time steps     │
│                                  │
│  Dynamics at each step t:        │
│    x_t = S_t · J_matrix + I_0   │
│          + J_ext·input_t + noise│
│    H_t = ReLU( (a·x - b) /      │
│          (1−exp(−d·(a·x−b))) )  │
│    dS/dt = −S/τ_s + (1−S)·H·γ   │
│    S_{t+1} = S_t + dS/dt · Δt   │
│                                  │
│  Output: evidence_trajectory     │
│    [B, 500, 4]                   │
└─────────────────────────────────┘
    │  [B, 500, 4] trajectory
    ▼
┌─────────────────────────────────┐
│  Soft-Index RT Readout           │
│  (decouples choice from RT)     │
│                                  │
│  1. Per-class first-crossing    │
│     time: t_c* = argmin_t       │
│       (trajectory[t,c] > θ)     │
│                                  │
│  2. Gaussian soft-index around  │
│     t_c*: w_c(t) ∝ exp(−0.5·   │
│     ((t−t_c*)/σ_steps)²)       │
│     σ_s = 0.05s (default)       │
│                                  │
│  3. Class evidence:             │
│     e_c = Σ_t w_c(t)·traj[t,c]  │
│                                  │
│  4. Choice: softmax(e_c / τ)    │
│     τ = 0.10 (temperature)      │
│                                  │
│  5. RT: Σ_c P(choice=c)·t_c*    │
│     + t0 shift                  │
│                                  │
│  Output: pred_rt, choice        │
└─────────────────────────────────┘
    │  [pred_rt, winner_idx, choice_probs]
    ▼
┌─────────────────────────────────┐
│  Behavioral Loss Suite           │
│  ├─ response_nll (λ=1.0)        │
│  ├─ rt_mse (λ=1.0)              │
│  ├─ error_rate_loss (λ=0.0)     │
│  └─ error_sign_loss (λ=0.0)     │
│                                  │
│  Note: behavioral penalties      │
│  (error_rate/sign) are active   │
│  only in experimental configs   │
└─────────────────────────────────┘
```

### Core implementation files

| File | Role |
|---|---|
| `code/scripts/vgg_wongwang_lim.py` | Full model architecture: VGG16, Wong-Wang dynamics, readout modes, behavioral losses, Stage-2 input transforms |
| `code/scripts/wong_wang.py` | Pure Wong-Wang dynamics implementation (standalone) |
| `code/scripts/train_age_groups_efficient.py` | Main training orchestration: data loading, Stage 1 freezing, Stage 2 fitting, scale search, evaluation |
| `code/scripts/evaluate_next_steps.py` | Targeted gap-closing evaluation: 16 configs across 4 phases |

---

## 2. Implementation Flow

### 2.1 Data Preparation

```
Raw LIM data (CSV)
  ├─ image_path → VGG16 feature extraction (128×128×3 → 25088-d features)
  ├─ stimulus_image → cached feature vector
  ├─ target_labels (ground truth: L/R/U/D)
  ├─ response_labels (human response: L/R/U/D)
  └─ true_rt (human reaction time, seconds)
         │
         ▼
  Data/age_groups_matched/20-29/
  ├─ train_data.csv (N≈94,737 rows)
  └─ test_data.csv  (N≈33,134 rows)
```

### 2.2 Training Pipeline

```
Phase 1: Stage 1 (VGG) Training
  ├─ If checkpoint exists → load and freeze
  ├─ If not → train from pretrained VGG16
  │   ├─ Loss: CrossEntropy(target_labels, logits)
  │   ├─ Epochs: 5–50 (task-dependent)
  │   └─ Output: best_model.pth
  └─ Result: ~95% classification accuracy

Phase 2: Stage 2 (Wong-Wang) Fitting
  ├─ Freeze Stage 1 VGG weights
  ├─ For each scale in [0.15, 0.25, 0.35, 0.45]:
  │   ├─ Transform logits → softplus_centered(logits × scale)
  │   ├─ Run Wong-Wang dynamics (500 steps, dt=10ms)
  │   ├─ Apply soft_index readout → pred_rt, choice
  │   ├─ Compute loss: response_nll + rt_mse (+ optional behavioral)
  │   └─ Backward through DiffDecisionMultiClass (implicit-function gradient)
  ├─ Select best scale by total_score or behavior_optimal_score
  └─ Output: best WW checkpoint + evaluation metrics
```

### 2.3 Key Technical Details

**Gradient flow through threshold crossing:**
- `DiffDecisionMultiClass` uses implicit function theorem: `d(crossing_time)/d(trajectory) = −1 / dsdt` at crossing point
- Batch elements where no class crosses threshold receive zero gradient
- The backward multiplies through by `dt` for correct chain-rule scaling

**Soft-index readout** (key innovation over baseline):
- Decouples "which class" from "when" — choice comes from amplitude-weighted evidence, not just crossing order
- Gaussian soft-index centered on each class's first-crossing time
- RT = probability-weighted average of per-class decision times
- Enabled by parameter: `sigma_s = 0.05s`, `choice_temperature = 0.10`

**t0 non-decision time shift:**
- Fixed t0 = 0.15s added to all predicted RTs
- Accounts for sensory encoding + motor execution latency
- `t0_mode = 'fixed_global'` avoids learned-t0 instability

---

## 3. Results: Best Configuration

### 3.1 Absolute Winner: B3 (fixed t0=0.15s)

**Configuration:**
- Stage 1: VGG16, pretrained, frozen
- Stage 2 input: `softplus_centered` transform, scale=0.183
- Readout: `soft_index`, sigma_s=0.05, choice_temperature=0.10
- t0: fixed_global = 0.15s
- Behavioral penalties: none (pure response_nll + rt_mse)
- Train: matched 20-29, 15 epochs WW, 0.15 train fraction

**Results (on matched 20-29 test set):**

| Metric | Model (B3) | Human | Gap |
|---|---|---|---|
| **Behavioral Optimal Score** | **0.592** | 1.0 (oracle) | — |
| Predicted Mean RT | 0.578s | 0.624s | −0.046s |
| Predicted Median RT | 0.571s | 0.601s | −0.030s |
| Model Accuracy | 1.000 | 0.950 | +0.050 |
| Response Agreement | 0.950 | 0.950 | 0.000 |
| Congruency RT Gap | 0.074s | 0.041s | +0.033s |
| Error−Correct ΔRT | **+0.389s** | **−0.058s** | **wrong sign** |
| Q90 | 0.715s | 0.700s | +0.015s |
| Q95 | 0.772s | 0.750s | +0.022s |
| Q99 | 0.887s | 1.001s | −0.114s |

**What B3 gets right:**
- RT mean and median close to human (~0.05s gap)
- RT distribution quantiles (Q90/Q95) well-aligned
- Response agreement equals human (0.95)
- Congruency gap in right ballpark (0.07 vs 0.04)
- Stable: fixed t0 avoids learned-parameter drift

**What B3 gets wrong:**
- Model accuracy = 1.0 (human = 0.95) — model never makes spontaneous errors
- Error−Correct ΔRT = +0.389s — model errors are **slower** than correct responses, opposite to human pattern (errors faster by 0.058s)
- Q99 saturated at 0.89s (insufficient tail)
- Congruency gap ~2× human magnitude

### 3.2 Second-Best: A1 (behavioral penalties 1×)

| Metric | Value |
|---|---|
| beh_opt | 0.550 |
| pred_mean | 0.590s |
| model_accuracy | 0.999 |
| response_agreement | 0.949 |
| error−correct ΔRT | +0.367s |
| cong_gap | 0.101s |

### 3.3 Cross-Phase Summary: Top 5 Configurations

| Rank | Config | beh_opt | Acc | ΔRT | Pred Mean | Key Mechanism |
|---|---|---|---|---|---|---|
| 1 | **B3** | **0.592** | 1.000 | +0.389 | 0.578s | Fixed t0=0.15s |
| 2 | C2 | 0.559 | 1.000 | +0.359 | 0.545s | Soft-index σ=0.15 |
| 3 | D1 | 0.559 | 1.000 | +0.460 | 0.599s | DMC moderate |
| 4 | C1 | 0.559 | 0.998 | +0.456 | 0.549s | Soft-index σ=0.02 |
| 5 | B1 | 0.555 | 0.996 | +0.401 | 0.510s | Fit t0=0.10s |
| — | **Human** | — | 0.950 | **−0.058** | 0.624s | — |

---

## 4. What Was Tried and Why It Didn't Surpass B3

### 4.1 Readout-Only Changes
- **soft_hazard**: collapsed RT scale (pred_mean ~0.2s)
- **urgency (additive/collapsing)**: preserved scale but worsened skew and error structure
- **Conclusion**: readout alone cannot fix the underlying evidence dynamics

### 4.2 Stage-2 Internal Parameter Probes
- **noise_ampa sweep** (0.02 → 0.06): created error regime but ΔRT sign remained wrong
- **threshold lowering**: preserved errors, worsened RT scale
- **competition scaling**: moved secondary metrics, no fundamental change
- **Conclusion**: single WW knobs insufficient

### 4.3 RT Distribution Shape Losses
- **soft_hist_kl, cdf_wasserstein, conditional CDF**: moved selected checkpoint, but no errors created
- **Conclusion**: shape-aware losses have signal, but not enough alone

### 4.4 Behavioral Penalty Weight Sweep (A1–A3)
- **1× penalties (A1)**: best single config with penalties, ΔRT=+0.367
- **2× penalties (A2)**: eliminated error regime entirely (acc=1.0, ΔRT=NaN)
- **4× penalties (A3)**: collapsed RT scale (pred_mean=0.473s)
- **Conclusion**: penalizing errors without a mechanism for error generation is counterproductive

### 4.5 DMC Psychological Mechanism (D1–D4)
- Tested early flanker capture + late cognitive control
- Stronger DMC parameters eliminated errors (not created more)
- Only D1 (moderate parameters) produced errors at all, with wrong sign
- **Conclusion**: Stage-2 selection mechanism needs Stage-1 evidence uncertainty to produce fast errors

### 4.6 SPEA GRU-Accumulator (c0–c4)
- 5 variants: weighted evidence, accuracy calibration, hard-stop readout, multi-sequence evidence, semi-supervised consistency
- All hit the same hard floor: accuracy 22–44%, response_agreement 20–41%
- **Conclusion**: GRU accumulator architecture is the bottleneck — it loses ~55% of Stage-1 evidence quality

### 4.7 Variational Evidence → Wong-Wang Synthesis (Phase 15)
- Fed time-varying variational (stochastic) evidence directly into WW dynamics, bypassing the SPEA accumulator
- **Breakthrough**: WW preserves evidence quality ~2.5× better than SPEA (64% vs 25% accuracy)
- Response agreement improves ~30% (0.54 vs 0.41) — WW's 4×4 learned J_matrix preserves evidence structure
- RT scale is calibratable (1.2–4.9s, needs threshold/noise tuning)
- Error regime still absent — variational sigma ~0.28 not yet sufficient for fast-error dynamics
- **Conclusion**: The GRU accumulator is definitively the architecture bottleneck. WW + stochastic evidence is the promising path forward; the remaining gaps (RT scale, error regime) are known-calibratable WW parameters.

---

## 5. Main Limitations and Root Causes

### 5.1 The Error-Direction Problem (HARDEST)

**Observation**: Across all 16 evaluated configurations and all prior experiments:
- Model error−correct ΔRT is **always positive** (+0.35 to +0.52)
- Human error−correct ΔRT is **negative** (−0.058s)
- This means: the model's errors are slow, deliberate mistakes; human errors are fast, automatic responses

**Root cause**: Stage 1 (VGG16) produces deterministic, high-quality logits that make the correct answer trivially distinguishable from distractors. Without evidence uncertainty at the input level, no Stage-2 mechanism can produce the human "fast error" pattern where flanker-driven automatic responses beat deliberate control.

**Evidence**: The DMC experiment (Phase D) directly tested this — even with explicit early flanker capture + late cognitive control, stronger DMC parameters eliminated errors instead of creating fast errors.

### 5.2 Accuracy Ceiling Effect

**Observation**: Model accuracy is almost always 0.995–1.000 (human = 0.950).
- The VGG logits are too clean — the correct class dominates by a wide margin
- Wong-Wang competition with learned J_matrix further amplifies the correct signal
- Without evidence noise or ambiguity, the model never makes spontaneous errors

### 5.3 RT Tail Truncation

**Observation**: Q99 (slowest 1% of RTs) is consistently below human:
- B3: 0.887s vs human 1.001s
- Model RT distribution lacks the heavy right tail characteristic of human RT distributions

**Possible cause**: Wong-Wang threshold crossing at 500 steps forces a hard time ceiling. The soft-index readout's Gaussian weighting around first-crossing time further suppresses late crossings.

### 5.4 Congruency Gap Over-Estimation

**Observation**: Model congruency gap ~0.07–0.10s vs human ~0.04s.
- The model shows stronger flanker interference than humans do
- This suggests the competition dynamics are too sensitive to input differences between congruent and incongruent trials

---

## 6. Promising Future Direction: Variational Evidence → WW

A recent synthesis smoke experiment (`2026-05-05`) demonstrated a breakthrough finding:

**Key result**: Feeding variational (stochastic) evidence sequences directly into Wong-Wang dynamics preserves evidence quality ~2.5× better than the SPEA GRU accumulator.

| Variant | Accuracy | Resp Agree | ΔRT | Pred Mean |
|---|---|---|---|---|
| Var→WW v2 (120 steps) | **0.637** | **0.539** | +0.009 | 1.18s |
| SPEA best (c1) | 0.438 | 0.408 | −0.006 | 0.370s |
| Static B3 (best) | 1.000 | 0.950 | +0.389 | 0.578s |

**Implications:**
1. WW's explicit 4-population competition with learned J_matrix preserves evidence structure much better than GRU-based accumulation
2. Variational evidence introduces the stochasticity that static VGG logits lack (sigma ~0.28)
3. RT scale needs calibration (threshold, J_ext, noise_ampa sweep) but is a known-calibratable parameter
4. Error regime direction remains unresolved — the variational uncertainty may or may not be sufficient to create fast-error dynamics in WW

**Remaining work:**
- WW parameter sweep on variational evidence (threshold × noise × competition)
- t0 calibration
- Behavioral-loss weight testing in combination
- If smoke gates clear → promote to full matched 20-29 surface

---

## 7. File Reference Index

### Core Architecture
| File | Lines | Description |
|---|---|---|
| `code/scripts/vgg_wongwang_lim.py` | ~850 | Full model: VGG16, WW dynamics, all readout modes, behavioral losses |
| `code/scripts/wong_wang.py` | ~200 | Standalone Wong-Wang implementation |

### Training & Evaluation
| File | Description |
|---|---|
| `code/scripts/train_age_groups_efficient.py` | Main training orchestration |
| `code/scripts/evaluate_next_steps.py` | 16-config gap-closing evaluation |
| `code/scripts/train_variational_ww_smoke.py` | Variational evidence → WW bridge |

### Results
| Path | Content |
|---|---|
| `artifacts/checkpoints/age_groups_matched/20-29/eval_next_steps/` | Phase A–D evaluation results |
| `artifacts/results/rt_model_variational_ww_synthesis/` | Var→WW synthesis results |
| `docs/history/logs.md` | Complete research timeline (Phases 0–15) |

### Key Memos
| Path | Content |
|---|---|
| `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md` | WW error-regime chain findings |
| `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md` | Successor screening verdict: `NO_SUCCESSOR_BRANCH_CLEARED_GATES` |
| `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md` | HSFA repair verdict: `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN` |

---

## 8. Executive Summary

The current best model (B3) achieves a behavioral optimal score of 0.592 — the highest across all 16 evaluated configurations and all prior experiment phases. It correctly reproduces human RT mean/median, response agreement, and distribution quantiles, but suffers from a fundamental limitation: **the model never makes spontaneous errors, and when forced into error regimes, its errors are slower than correct responses — the opposite of human behavior**.

After 15 phases of systematic experimentation — spanning readout redesigns, internal parameter probes, distribution-shape losses, behavioral penalty sweeps, DMC psychological mechanisms, SPEA accumulator calibration, successor-branch screening, and HSFA repair audits — the evidence converges on a single conclusion: **further local Stage-2 patching cannot resolve the error-direction problem**. The bottleneck is Stage 1: VGG16 produces evidence that is too deterministic, making the correct answer trivially distinguishable. Stage-1 uncertainty (variational/BNN-like feature sampling) is the justified next planning direction.

The variational evidence → Wong-Wang synthesis smoke provides the first direct evidence that combining stochastic Stage-1 evidence with WW's biologically-grounded competition dynamics is a promising path — accuracy jumps from 25% (SPEA) to 64% — but the error-regime problem remains unresolved even there.

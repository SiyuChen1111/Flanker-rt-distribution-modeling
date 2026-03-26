# VAM Study Notes - Siyu

**Study Date**: 2026-03-26  
**Learning Objectives**: Understand VAM model architecture, CNN drift rate generation mechanism, and the physical meaning of four-direction numerical values

---

## Conversation Records and Key Learning Points

### 1. Inquiry about AGENTS.md Line 9

**Siyu's Question**: `/Users/siyu/Documents/GitHub/VAM-studying/AGENTS.md#L9-9` What model is this? What is its implementation principle?

**AI's Key Points**:
- **Model**: VAM (Visual Attention Model)
- **Architecture**: Combines deep learning and cognitive computational modeling
- **Core Components**:
  1. **CNN**: Extracts features from visual stimuli and generates drift rates
  2. **LBAVI**: Uses variational inference to handle LBA model cognitive parameters

**Implementation Principles**:
- **ELBO Optimization Objective**: Maximizes Evidence Lower Bound
- **Reparameterization Trick**: Enables gradient propagation
- **LBA Model Integration**: Simulates information accumulation in decision-making processes

**Learning Outcomes**:
- VAM combines cognitive science and deep learning
- Implemented using JAX/Flax framework
- Simultaneously optimizes CNN and LBA parameters through ELBO

---

### 2. Understanding JAX/Flax

**Siyu's Question**: How to understand "VAM uses JAX/Flax for visual attention training"

**AI's Key Points**:
- **@jax.jit**: Just-in-time compilation accelerates training
- **Automatic Differentiation**: No need for manual gradient derivation
- **Flax Module System**: Declarative neural network definition
- **Deterministic Random Numbers**: Explicit management ensures reproducibility
- **Vectorized Operations (vmap)**: Avoids explicit loops

**Learning Outcomes**:
- JAX provides high-performance computing and automatic differentiation
- Flax offers modular neural network construction
- Suitable for complex statistical models like variational inference

---

### 3. About CNN Generating Drift Rates

**Siyu's Question**: Where is VAM's CNN reflected in the code?

**AI's Key Points**:
- **CNN Class Definition**: `vam/vam/models.py` lines 156-178
- **Integration Method**: As VAM's submodule `self.get_drifts`
- **Architecture**:
  - Convolutional layers: 3×3 kernels, ReLU, GroupNorm, MaxPool
  - Fully connected layers: Dense + ReLU + Dropout
  - Output layer: Generates n_acc drift rates

**Learning Outcomes**:
- CNN is named `get_drifts`, directly indicating its function
- Output layer feature count = accumulator count (4 directions)
- Each output corresponds to a drift rate for a decision option

---

### 4. About Model Input and Output

**Siyu's Question**: What are the model's inputs and outputs? How does the code reflect that the model generates cognitive model parameters (drift rates)?

**AI's Key Points**:
**Inputs**:
- `stimuli`: Visual stimulus images (batch_size, 128, 128, 3)
- `rts`: Reaction times
- `responses`: Choices/responses
- `key`: JAX random key

**Outputs**:
- `elbo`: Evidence Lower Bound (optimization objective)
- `drift_mean`: Drift rates (batch_size, 4)

**Reflection Methods**:
1. CNN output layer: `nn.Dense(features=self.config.n_acc)`
2. Semantic naming: `self.get_drifts = CNN(...)`
3. LBA model usage: `batch_drifts=drifts` as parameter v

**Learning Outcomes**:
- Drift rates are core parameters of the LBA model
- Represent information accumulation speed
- High values indicate strong support for that option

---

### 5. About How CNN Generates Drift Rates

**Siyu's Question**: How does CNN generate drift rates?

**AI's Key Points**:
**Complete Flow**:
```
Visual Stimulus (128×128×3)
  ↓
Convolutional Layers (6 layers: 64→64→128→128→128→256)
  ↓
Flatten (2×2×256 = 1024)
  ↓
Fully Connected Layer (1024)
  ↓
Output Layer (4 drift rates)
```

**Layer-by-layer Transformation**:
- Each layer: Conv → ReLU → GroupNorm → MaxPool
- Downsampling: 128→64→32→16→8→4→2
- Final output: (batch, 4)

**Learning Outcomes**:
- CNN gradually extracts visual features from local to global
- Finally maps to drift rates for 4 directions
- Each drift rate represents evidence accumulation speed for that direction

---

### 6. About Pre-trained Models and Flanker Task

**Siyu's Question**: Does this CNN use pre-trained models? How does it implement the flanker task? Does one stimulus include multiple objects?

**AI's Key Points**:
**Pre-trained Model**:
- Uses VGG16 pre-trained model!
- Transfers first 2 layers of VGG weights to CNN
- `n_pretrained_layers=2`

**Flanker Task Implementation**:
- Each stimulus contains 5 birds (1 target + 4 flankers)
- 7 layouts: horizontal, vertical, cross, V_left, V_right, V_down, V_up
- Congruency judgment: target_dir == flanker_dir

**Learning Outcomes**:
- VGG16 provides basic visual features
- CNN learns to handle multi-object scenes
- Handles competition between target and flankers through drift rate allocation

---

### 7. About Attention Mechanism

**Siyu's Question**: Attention mechanism: Learning to focus on central targets and suppress surrounding interference, how is this reflected in this model?

**AI's Key Points**:
**Implicit Attention Mechanism**:
1. **CNN Architecture**: Local receptive fields and pooling operations
2. **Feature Recording**: `self.sow("intermediates", "features", x)`
3. **Subspace Separation**: Decoder analysis of target vs. flankers
4. **Invariance Analysis**: Testing attention robustness
5. **Drift Rate Distribution**: Comparison of target vs. flanker drift rates

**Learning Outcomes**:
- VAM has no explicit attention module
- Naturally implements spatial attention through CNN architecture
- Drift rates reflect attention allocation and interference suppression

---

### 8. About ELBO

**Siyu's Question**: What is the role and principle of ELBO?

**AI's Key Points**:
**ELBO = Evidence Lower Bound**

**Four Components**:
1. **Reconstruction Error Term**: `log p(x|z)` - LBA model log-likelihood
2. **Jacobian Determinant Term**: Density adjustment for variable transformation
3. **Prior Term**: `log p(z)` - Standard normal distribution
4. **Entropy Term**: `log q(z)` - Variational posterior distribution

**Formula**:
```
ELBO = E[log p(x|z)] + E[log p(z)] - E[log q(z)] + Jacobian term
```

**Learning Outcomes**:
- ELBO connects vision and cognition
- Bayesian learning framework
- Simultaneously optimizes CNN and LBA parameters

---

### 9. About ELBO's Target

**Siyu's Question**: Can I understand that this ELBO acts on LBA?

**AI's Key Points**:
**Not entirely correct**! ELBO acts on the entire VAM model:

**Optimization Objective**:
- Simultaneously optimizes CNN parameters and LBA parameters
- Updates all parameters through gradient descent

**Gradient Propagation**:
```
ELBO loss → Backpropagation → Update CNN weights → Improve drift rate prediction
         → Update LBA parameters → Improve decision model
```

**Learning Outcomes**:
- ELBO is a unified optimization objective
- Connects deep learning and cognitive modeling
- Does not act on LBA alone

---

### 10. About Creating Notebook

**Siyu's Request**: Create an ipynb including image generation to VGG-only model training, then make the model output congruent/incongruent judgment

**Implemented Content**:
- Flanker stimulus generator
- VGG model outputting 4 drift rates
- Training process
- Congruency judgment mechanism

**Problems Encountered and Solutions**:
1. SSL download error → Manually copy model files
2. Feature dimension mismatch → 25088 vs 8192
3. Font display issue → Use matplotlib arrows
4. Inconsistent image generation → Use real bird images

**Learning Outcomes**:
- Complete deep learning workflow
- Problem troubleshooting and resolution skills
- Importance of consistency with original code

---

### 11. About Loss Function Differences

**Siyu's Question**: Is the loss function definition here the same as the original code?

**AI's Key Points**:
**Not the same!**

**Notebook (Simplified Version)**:
- Supervised learning
- Custom drift_rate_loss
- Directly optimizes drift rates

**Original VAM (ELBO)**:
- Variational inference
- Evidence Lower Bound
- Complete cognitive modeling

**Reasons for Differences**:
- Notebook purpose: Demonstrate VGG outputting drift rates
- Original VAM purpose: Complete cognitive computational model

**Learning Outcomes**:
- Understand supervised learning vs. variational inference
- Trade-offs between simplified vs. complete versions
- Choice between different frameworks (PyTorch vs. JAX)

---

## Key Concepts Summary

### 1. Drift Rate
- **Definition**: Information accumulation speed
- **Physical Meaning**: Evidence strength supporting a direction decision
- **Numerical Interpretation**: High value (>2.0)=strong support, Low value (<1.0)=weak support

### 2. LBA Model
- **Full Name**: Linear Ballistic Accumulator
- **Mechanism**: Multiple accumulators compete, fastest to reach threshold wins
- **Application**: Simulates decision process, predicts RT and choices

### 3. ELBO
- **Full Name**: Evidence Lower Bound
- **Role**: Optimization objective for variational inference
- **Composition**: Reconstruction error + Prior + Entropy + Jacobian term

### 4. VAM Architecture
- **CNN**: Visual feature extraction → Drift rates
- **LBA**: Cognitive model → Decision process
- **ELBO**: Unified optimization objective

---

## Next Steps Learning Plan

1. **Run Notebook**: Test complete VGG drift rate output workflow
2. **Deep Dive into ELBO**: Understand mathematical principles of variational inference
3. **Read Paper**: Jaffe et al. (2025) VAM paper
4. **Code Practice**: Try modifying loss functions and model architectures
5. **Comparison Experiments**: Compare effects of supervised learning vs. variational inference

---

## Reflections and Questions

### Resolved
- ✅ VAM model architecture understanding
- ✅ CNN drift rate generation mechanism
- ✅ Physical meaning of four-direction numerical values
- ✅ Flanker task implementation
- ✅ Attention mechanism reflection

### To be Explored
- 🤔 Mathematical derivation of ELBO
- 🤔 JAX/Flax vs. PyTorch choice
- 🤔 Practical applications of variational inference
- 🤔 Interpretability of cognitive models

---

**Record Time**: 2026-03-26  
**Recorder**: Siyu  
**AI Assistant**: Kimi-K2.5

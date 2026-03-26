# VAM Study Log - 2026-03-26

## Today's Updates Overview

### 1. Core Notebook Files Created

#### `vgg_drift_rate_complete.ipynb`
- **Purpose**: Complete implementation of VGG model outputting drift rates
- **Main Contents**:
  - Flanker stimulus generator (using real bird images)
  - VGG16 model outputting 4 drift rates
  - Detailed physical meaning explanation of four-direction numerical values
  - Association between drift rates and cognitive processes
  - Training workflow and visualization

#### `vgg_drift_rate_fixed.ipynb`
- **Purpose**: Version with SSL download issue fixed
- **Improvements**:
  - Added SSL certificate bypass
  - Updated torchvision API usage
  - Provided multiple model loading methods

### 2. Helper Scripts Created

#### `download_vgg16.py`
- **Function**: Manually download VGG16 pre-trained model
- **Solves**: SSL certificate verification failure issue

#### `vgg16_download_alternatives.py`
- **Function**: Provide multiple VGG16 download alternatives
- **Includes**: Domestic mirrors, manual download, no pre-training, etc.

#### `test_vgg16_loading.py`
- **Function**: Test VGG16 model loading
- **Verifies**: Whether pre-trained model is correctly installed

#### `test_vgg_dimensions.py`
- **Function**: Test VGG16 feature dimensions
- **Discovered**: Feature dimension is 25088 (512×7×7) for 128×128 input

#### `fix_vgg_dimensions.py`
- **Function**: Fix feature dimension mismatch issue
- **Modification**: Changed 8192 to 25088

#### `fix_notebook_complete.py`
- **Function**: Complete notebook fix
- **Includes**: Add shape display + fix feature dimension

#### `add_visualization.py`
- **Function**: Add Flanker stimulus visualization
- **Effect**: Display 8 samples, distinguish Congruent/Incongruent

#### `fix_visualization.py`
- **Function**: Fix visualization (using matplotlib arrows)
- **Solves**: Font display issue

#### `update_with_real_birds.py`
- **Function**: Update to use real bird images
- **Improvement**: Consistent with original code make_model_inputs.py

#### `add_shape_display.py`
- **Function**: Add test_images shape display cell

### 3. Key Discoveries and Fixes

#### Issue 1: SSL Download Error
- **Symptom**: `URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]>`
- **Solution**: Copy model file to `~/.cache/torch/hub/checkpoints/`

#### Issue 2: Feature Dimension Mismatch
- **Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x25088 and 8192x1024)`
- **Cause**: VGG16 produces 25088-dimensional features for 128×128 input, not 8192
- **Solution**: Modified `self.feature_dim = 512 * 7 * 7` (25088)

#### Issue 3: Font Display Issue
- **Symptom**: Arrows display as block garbled characters
- **Solution**: Use matplotlib's arrow function instead of fonts

#### Issue 4: Inconsistent Image Generation
- **Symptom**: Using arrows instead of real bird images
- **Solution**: Use bird0.png - bird3.png real images

### 4. Physical Meaning of Four-Direction Drift Rates

```
Output: [drift_rate_left, drift_rate_right, drift_rate_up, drift_rate_down]
Index:  [      0,            1,            2,            3]
Direction: [   L,            R,            U,            D]
```

**Drift Rate = Information Accumulation Speed**

- **High value (>2.0)**: Strongly supports this direction
- **Medium value (1.0-2.0)**: Moderate support
- **Low value (<1.0)**: Weak support or no support

**Association with Cognitive Processes**:
1. **Reaction Time**: RT ≈ threshold / drift_rate
2. **Choice Probability**: Calculated through Softmax
3. **Congruency Judgment**: Judged through drift rate concentration

### 5. Differences from Original VAM Code

#### Notebook Implementation
- **Framework**: PyTorch
- **Training Method**: Supervised learning (simplified loss function)
- **Loss Function**: Custom drift_rate_loss
- **Purpose**: Demonstrate VGG outputting drift rates

#### Original VAM Code
- **Framework**: JAX/Flax
- **Training Method**: Variational inference (ELBO optimization)
- **Loss Function**: Evidence Lower Bound (ELBO)
- **Purpose**: Complete cognitive computational model

**ELBO Composition**:
- Reconstruction error (LBA log-likelihood)
- Jacobian determinant
- Prior term
- Entropy term

### 6. File Management Suggestions

#### Core Files (Keep)
- `vgg_drift_rate_complete.ipynb` - Main learning material
- `vgg_drift_rate_fixed.ipynb` - Backup version
- `logs.md` - This log file
- `siyu_study.md` - Study notes

#### Helper Scripts (Can Archive)
- `download_vgg16.py`
- `vgg16_download_alternatives.py`
- `test_vgg16_loading.py`
- `test_vgg_dimensions.py`
- `fix_vgg_dimensions.py`
- `fix_notebook_complete.py`
- `add_visualization.py`
- `fix_visualization.py`
- `update_with_real_birds.py`
- `add_shape_display.py`

#### Pre-trained Model (Installed)
- `vgg16-397923af.pth` → `~/.cache/torch/hub/checkpoints/`

### 7. Next Steps Suggestions

1. **Run Notebook**: Test complete VGG drift rate output workflow
2. **Comparative Learning**: Understand difference between supervised learning and variational inference
3. **Deep Research**: Read original VAM paper and code
4. **Experimental Extension**: Try different loss functions and training strategies

---

**Record Time**: 2026-03-26
**Recorder**: AI Assistant
**Status**: Active Development

# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-26
**Commit:** (initial - no commits)
**Branch:** main

## OVERVIEW
Research repository containing VAM (Visual Attention Model) implementation and a skills library. VAM uses JAX/Flax for visual attention training with Lumosity cognitive task data.

## STRUCTURE
```
VAM-studying/
├── vam/                    # Visual Attention Model research code
│   ├── vam/              # Core module (training, models, metrics)
│   ├── manuscript/        # Analysis and reproduction scripts
│   ├── docs/             # Documentation
│   ├── *.py              # Data files (gameplay, metadata, graphics)
│   └── *.png             # Stimulus images
└── skills/                # Skills library (100+ items, flat structure)
    └── [skill-name]/      # Individual skill directories
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| VAM training | `vam/vam/training.py` | JAX/Flax Trainer class |
| VAM models | `vam/vam/models.py` | VAM, task_opt, binned_rt architectures |
| VAM config | `vam/vam/config.py` | ConfigDict pattern with CLI args |
| VAM metrics | `vam/vam/metrics.py` | Flax dataclasses, WandB logging |
| VAM analysis | `vam/vam/model_analysis.py` | Mixin-based statistical analysis |
| VAM data | `vam/*.zip`, `vam/*.csv` | Lumosity gameplay metadata |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|
| Trainer | class | vam/vam/training.py | high | Main training loop with JIT |
| ConfigDict | function | vam/vam/config.py | high | Configuration management |
| VAMMetrics | dataclass | vam/vam/metrics.py | medium | Evaluation metrics |
| ModelAnalysis | class | vam/vam/model_analysis.py | medium | Statistical analysis pipeline |

## CONVENTIONS
- **JAX/Flax framework**: Uses `@jax.jit` for compiled functions, `flax.struct.dataclass` for configs
- **ConfigDict pattern**: Uses `ml_collections.ConfigDict` with `d()` helper for nested configs
- **WandB integration**: Logs with structured prefixes (train/, val/, user/)
- **Multi-model support**: Handles vam, task_opt, binned_rt models via config
- **Data augmentation**: Uses augmax library (translate, warp, etc.)
- **Lazy loading**: "slow" mode loads images on-demand; "fast" mode caches
- **Mixin architecture**: Analysis uses multiple mixins (BasicAnalysisMixin, DimensionalityMixin, etc.)
- **CLI integration**: `get_config_from_cli()` for argument parsing

## ANTI-PATTERNS (THIS PROJECT)
- **Do not mix configs**: Use helper functions for different model types (get_default_config, get_task_opt_config, get_binned_rt_config)
- **Do not bypass JIT**: Train/eval functions use `@jax.jit` decorators
- **Do not ignore reproducibility**: Always use seeded random keys (`config.seed`)

## UNIQUE STYLES
- **JAX/Flax ecosystem**: Uses TrainState from flax.training, optax optimizers
- **Orbax checkpointing**: Custom checkpoint management not using standard Flax
- **Compositional metrics**: Metrics classes merge via `.merge()` and compute via `.compute()`
- **Statistical analysis pipeline**: Uses pandas DataFrames for all results with scipy for statistical tests (ANOVA, Tukey HSD)
- **Multi-dataset support**: gameplay_data.zip (75 users), metadata.csv (user demographics), graphics.zip (stimuli)

## COMMANDS
```bash
# Train VAM model
cd vam
python -m vam.vam.training --project test --expt_name exp001

# Train task-optimized model
python -m vam.vam.training --model_type task_opt --n_epochs 30

# Train binned RT model
python -m vam.vam.training --model_type binned_rt --n_rt_bins 5 --rt_bin 3
```

## NOTES
- **Data format**: game play_data.zip contains per-trial CSV files with columns (anon_id, nth_play, trial, xpos, ypos, flanker_direction, response_direction, response_time, stimulus_layout)
- **Model transfer**: VAM uses pretrained VGG layers (n_pretrained_layers=2)
- **Dual optimizers**: Separate CNN and VI optimizers with different learning rates
- **Metrics hierarchy**: Organized in wandb with train/, val/, user/ prefixes
- **Analysis flexibility**: Mixin system allows easy addition of new analysis methods

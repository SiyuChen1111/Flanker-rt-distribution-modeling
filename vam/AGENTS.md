# VAM PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-26
**Parent:** VAM-studying/AGENTS.md

## OVERVIEW
VAM (Visual Attention Model) - JAX/Flax implementation of visual accumulator model for Lumosity cognitive task data. Trains models on decision-making tasks with bird flock stimuli.

## STRUCTURE
```
vam/
├── vam/                    # Core module
│   ├── training.py        # Trainer class, main training loop
│   ├── models.py          # VAM, task_opt, binned_rt architectures
│   ├── config.py          # Configuration with ConfigDict pattern
│   ├── metrics.py         # Evaluation metrics (Flax dataclasses)
│   ├── task_data.py       # Data handling, augmentation, lazy loading
│   ├── model_analysis.py  # Mixin-based statistical analysis
│   ├── mixins.py          # Analysis mixins (BasicAnalysisMixin, etc.)
│   ├── model_outputs.py    # Model output structures
│   ├── transforms.py       # Image augmentation (augmax)
│   ├── lba.py            # Linear Ballistic Accumulator
│   └── __init__.py        # Module init
├── manuscript/              # Analysis and reproduction scripts
│   ├── train_model.py    # CLI entry point for training
│   ├── run_model_analysis.py
│   ├── get_model_outputs.py
│   ├── make_model_inputs.py
│   ├── make_manuscript.py
│   ├── figures.py
│   └── __init__.py
├── docs/                  # Documentation
├── *.csv                  # metadata.csv (user demographics)
├── *.zip                  # gameplay_data.zip (75 users)
├── *.png                  # Stimulus images (bird0-3, bkgrnd)
└── README.md
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training config | `config.py` | get_default_config(), get_task_opt_config(), get_binned_rt_config() |
| Training loop | `training.py` | Trainer class with JIT-compiled steps |
| Model architectures | `models.py` | vam (generative), task_opt (discriminative), binned_rt |
| Metrics | `metrics.py` | VAMMetrics dataclass with flax.metrics |
| Data loading | `task_data.py` | TaskData class with lazy/cached modes |
| Analysis | `model_analysis.py` | Mixin-based pipeline (ANOVA, Tukey HSD) |
| Train CLI | `manuscript/train_model.py` | Entry point with CLI args |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|
| Trainer | class | vam/vam/training.py | high | Main training loop |
| ConfigDict | function | vam/vam/config.py | high | Config management |
| VAMMetrics | dataclass | vam/vam/metrics.py | medium | Evaluation metrics |
| ModelAnalysis | class | vam/vam/model_analysis.py | medium | Statistical analysis |

## CONVENTIONS
- **JAX/Flax ecosystem**: Uses `@jax.jit` for compiled functions, `flax.struct.dataclass` for configs
- **ConfigDict pattern**: Uses `ml_collections.ConfigDict` with `d()` helper for nested configs
- **Multi-model support**: Select via `--model_type` flag (vam, task_opt, binned_rt)
- **Data modes**: "slow" (on-demand), "fast" (cached), "binned_rt" (filtered by RT quantile)
- **WandB integration**: Logs with structured prefixes (train/, val/, user/)
- **Dual optimizers**: Separate CNN and VI optimizers with different learning rates

## ANTI-PATTERNS (THIS PROJECT)
- **Do not mix configs**: Use helper functions for different model types
- **Do not bypass JIT**: Train/eval functions use `@jax.jit` decorators
- **Do not ignore reproducibility**: Always use seeded random keys (`config.seed`)
- **Do not mix structure and behavior**: Config helpers vs training logic

## UNIQUE STYLES
- **Orbax checkpointing**: Custom checkpoint management not using standard Flax
- **Compositional metrics**: Metrics classes merge via `.merge()` and compute via `.compute()`
- **Mixin analysis**: Analysis uses multiple mixins (BasicAnalysisMixin, DimensionalityMixin, SubspaceMixin, etc.)
- **Statistical analysis pipeline**: Uses pandas DataFrames for all results with scipy for statistical tests
- **Multi-dataset support**: gameplay_data.zip (75 users), metadata.csv (user demographics), graphics.zip (stimuli)
- **Data augmentation**: Uses augmax library (translate, warp, stochastic)

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
- **Pretrained models**: Set via `--pretrained_model` flag for transfer learning
- **Reproducibility**: All random operations use seeded JAX keys
- **Metrics**: Organized hierarchically in wandb with train/, val/, user/ prefixes

# Reproducing Results

## Environment

From the repository root:

```bash
python -m pip install -r config/requirements.txt
```

The retained analysis is CPU-capable. Full VGG evidence extraction additionally needs the local stimulus data and stage-1 checkpoint recorded in `evidence_cache/extraction_metadata.json`.

## Fast verification

```bash
python code/scripts/reproduce_presentation_model.py --group all --smoke-test
pytest -q tests/test_diffdecision_multiclass.py \
  tests/test_r5_choice_rule_alignment_audit.py \
  tests/test_r5_choice_coupled_schedule_optimization.py \
  tests/test_real_vgg_target_flanker_dynamics_audit.py
```

The smoke test loads the selected config, trial manifest, cached five-layer evidence, original R5 results, and corrected predictions. It executes a small Wong-Wang batch and checks that both requested age groups are present with 5,000 trials each.

## Analysis-only verification

```bash
python code/scripts/reproduce_presentation_model.py --group young --analysis-only
python code/scripts/reproduce_presentation_model.py --group older --analysis-only
```

This validates the retained tables and their choice/crossing fields without rerunning the full 10,000-trial schedule search.

## Figures

Write regenerated figures to a new result directory:

```bash
python code/scripts/reproduce_presentation_model.py \
  --plot-only \
  --output-dir artifacts/results/presentation_model_reproduction
```

This regenerates CAF/delta, RT distribution, and the three-stage VGG-evidence/Wong-Wang mechanism analysis without overwriting the retained bundles.

## Full corrected-equivalent run

```bash
python code/scripts/run_r5_choice_coupled_schedule_optimization.py \
  --output-dir artifacts/results/presentation_model_full_rerun_<date>
```

Use a new directory. The full search is substantially slower than the smoke and analysis-only checks.

## Interpretation

The original R5 package reproduces the model used for the presentation. The corrected-equivalent package should be used for new choice/RT interpretation because it binds the selected choice to the sustained-crossing readout step. Neither package is a held-out full-cohort fit.

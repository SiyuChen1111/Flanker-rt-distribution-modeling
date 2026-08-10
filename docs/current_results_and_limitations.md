# Current Results and Limitations

## Retained results

The presentation model is the two-group `best_model_R5_combined_best` diagnostic. It contains 5,000 trials for young adults aged 20-29 and 5,000 trials for older adults aged 80-89. Its exact mechanism figure demonstrates that early flanker preference and later target recovery are present in the VGG layers and survive the temporal mapping into Wong-Wang state dynamics.

The corrected-equivalent result keeps the same visual evidence and recurrent accumulator, chooses the response at the sustained-crossing readout step, and uses group-specific schedule compression. In the retained corrected predictions, all 10,000 trials crossed. Incongruent accuracy is lower among fast responses and recovers among slower responses in both age groups.

## What is supported

- The retained VGG representation contains a robust early-flanker/later-target reversal.
- The layer-to-time mapping transmits that reversal to the accumulator input.
- Wong-Wang dynamics preserve the reversal on many trials.
- A choice-coupled readout can reproduce the direction of the incongruent CAF recovery on these representative subsets.

These findings partially support the proposed mechanism and pass focused consistency checks. They do not prove that people use the same mechanism.

## Limitations

- Model selection and evaluation use the same representative trials; there is no held-out validation.
- The young group has 12 participants and the older group only four.
- Model RT tails remain shorter than human RT tails.
- Congruency-related RT shape and participant variability are not fully reproduced.
- The original presentation choice rule and RT rule used different trajectory horizons. New interpretation should use the corrected-equivalent result.
- Schedule compression was selected diagnostically and is not an independently validated biological timing estimate.

The model must therefore be described as a presentation-model diagnostic, not a final full-cohort fit or a complete account of human conflict control.

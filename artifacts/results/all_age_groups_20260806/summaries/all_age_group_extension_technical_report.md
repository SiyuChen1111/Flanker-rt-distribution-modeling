# Technical report

> This report describes the base seven-group extension before the final timing calibration. For current fit metrics and integrated figures, use `../all_age_model_update_20260807/`.

The historical presentation chain is the retained natural layer-to-time VGG16 plus four-choice Wong-Wang R5 package. The corrected-equivalent retains the five VGG evidence layers, per-layer normalization, recurrent four-choice accumulator, sustained crossing, and non-decision-time model while coupling choice to `winner_at_readout`.

All seven age groups now have 5,000 deterministic representative trials. The five intermediate groups use newly extracted complete VGG caches and the same 171-candidate schedule search. WW threshold and margin are linearly interpolated between the retained extreme-age anchors; schedule compression, late shift, width, and non-decision-time terms are selected with the original corrected-equivalent score. This is an exploratory age-structured rule, not an independently validated causal age model.

All derived accuracy, RT, CAF, CRF, crossing, and participant-first delta outputs are recomputed from the unified 35,000-row trial file. No-crossing rows are explicitly censored and have no model RT.

# Model Framework

The retained workflow has three scientific stages.

1. VGG16 produces four-direction evidence at five successive layers: `conv3`, `conv4`, `conv5`, `pooled`, and `final`.
2. `per_layer_gap_scale` normalizes the evidence and `natural_smooth_5stage` maps it onto an 80-step time sequence.
3. A four-choice recurrent Wong-Wang accumulator integrates the sequence and produces a sustained-crossing decision time.

For incongruent stimuli, the retained VGG evidence typically favors the flanker early and the target later. The presentation mechanism figure separately shows the layer evidence, evidence delivered to Wong-Wang, and resulting Wong-Wang state.

The original presentation R5 package used two different time horizons: RT came from the sustained-crossing step, while choice could use the maximum over the complete trajectory. The retained corrected-equivalent keeps the VGG evidence and accumulator equations, selects the choice at the crossing readout step, and adjusts only when the existing five-layer evidence is delivered.

The full identity, parameters, files, and correction boundary are recorded in `docs/PRESENTATION_MODEL_MANIFEST.md`.

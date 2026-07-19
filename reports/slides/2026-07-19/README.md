# WR2 Flanker mentor-report package

This folder contains the mentor-report deck for the WR2 Flanker model and a
map from each result shown in the deck to the code and committed result files
that support it.

## Deliverable

- `flanker_wr2_dual_source_update.pptx`: 11-slide English deck with Chinese
  speaker notes. The deck intentionally ends with open diagnostic questions
  rather than a predetermined next-step recommendation.

## Slide-to-source map

| Slides | Topic | Code | Committed result source |
| --- | --- | --- | --- |
| 2–5 | WR2 architecture, 240-candidate fine search, and current behavioral fit | `code/scripts/run_wr2_uncertainty_schedule_fine_search.py`; `code/scripts/make_wr2_mentor_report_figures.py` | `.../wr2_uncertainty_schedule_fine_search/{metrics,summaries}`; `.../wr2_mentor_report_figures_20260719_v2/` |
| 6 | RT measurement-error and binning audit | `code/scripts/run_flanker_rt_bin_fitting.py`; `code/scripts/run_flanker_measurement_mechanism_followup.py` | `.../flanker_measurement_mechanism_followup/full_20260717_v3/` |
| 7 | Target-only and flanker-only VGG representation audit | `code/scripts/run_flanker_dual_source_followup.py` | `.../dual_source_conflict_test/20260719_full_v1/` |
| 8–10 | M0–M3 controlled Wong–Wang comparison and RT/error figures | `code/scripts/run_flanker_dual_route_ww_comparison.py`; `code/scripts/make_flanker_dual_route_report_figures.py` | `.../dual_source_conflict_test/20260719_ww_full_v1/` |
| 11 | Open diagnosis: spatial readout versus decision dynamics | Combined evidence from the source audit and M0–M3 comparison | Same two dual-source result directories |

All abbreviated result paths above are under:

`artifacts/results/natural_layer_to_time_var_ww/representative_extreme_age_subset_5000/`

## Main reported results

- WR2 fine search evaluated 240 candidates and recommended
  `WR2_fine_3744359a`; it passed the main but not the strict criteria.
- Plausible 60-Hz and 60-Hz-plus-125-Hz RT discretization did not materially
  explain the human–model gap.
- The source audit covered 9,990 stimuli. Conv3 target-only decoding was
  97.9%, and flanker-only decoding was 100%.
- In the four-fold, ten-seed controlled comparison, no M0–M3 model passed.
  The absolute young-incongruent fast-error gaps were 17.7, 29.5, 47.7, and
  3.2 percentage points for M0–M3, respectively. M3's local improvement did
  not survive the joint error–RT criteria.

## Reproduction entry points

From the repository root:

```bash
python code/scripts/run_wr2_uncertainty_schedule_fine_search.py --mode full
python code/scripts/run_flanker_measurement_mechanism_followup.py --help
python code/scripts/run_flanker_dual_source_followup.py --run-id <new-id> --mode full
python code/scripts/run_flanker_dual_route_ww_comparison.py --run-id <new-id>
python code/scripts/make_flanker_dual_route_report_figures.py --help
```

The committed result package intentionally excludes large trial-level
prediction tables, model caches, NPZ evidence arrays, and repeated smoke runs.
Those files remain local run products and can be regenerated with the entry
points above. The committed CSV summaries, configurations, figures, and split
manifests are sufficient to audit the numerical claims in the deck.

## Validation scope

The results are internal participant-held-out diagnostics based on 12 young
and 4 older participants. They are not an independent external validation;
age-mechanism conclusions remain exploratory.

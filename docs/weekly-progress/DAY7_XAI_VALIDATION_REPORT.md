# ResQTwin — Day 7 XAI Validation Report

**Date:** 24 October 2026
**Workstream:** Satellite flood segmentation — Explainable AI
**Status:** Implementation and preliminary faithfulness validation completed; broader validation remains outstanding.

## 1. Objective

Review the existing Grad-CAM implementation, verify consistency between explanation generation and perturbation validation, consolidate the available evidence, and document limitations and next steps.

No model retraining or new inference experiment was performed for this report.

## 2. Existing implementation

Primary implementation:
- `ai/satellite/src/explain_chennai_gradcam.py`

Faithfulness validation:
- `ai/satellite/src/validate_chennai_gradcam_day5_v2.py`

Model checkpoint:
- `ai/satellite/checkpoints/positive_weight_5/best.pt`

Explanation outputs:
- `data/processed/chennai/explainability/`

The Grad-CAM implementation targets `model.dec1`, a decoder feature layer. It computes gradients of the mean raw flood logit over a selected flood-prediction region and uses the resulting feature-map activations and gradients to generate an attribution map.

The explanation indicates model sensitivity for the selected prediction target. It does not establish ground-truth correctness or the physical causes of flooding.

## 3. Configuration consistency

The following were checked in both scripts:

- Model checkpoint path: `positive_weight_5/best.pt`.
- Strict state-dictionary loading: `strict=True`.
- Input channels: VV and VH SAR imagery.
- Conversion from positive linear SAR values to decibels.
- VV normalization: mean -10.3929, standard deviation 4.0388.
- VH normalization: mean -17.2411, standard deviation 4.7544.
- Flood-logit target: mean logit over the selected flood region.

These checks establish consistency in the inspected configuration. They do not constitute a byte-for-byte comparison of checkpoint files or prove that every runtime dependency and input file is identical.

## 4. Day 5 v2 faithfulness experiment

### 4.1 Method

The validation script evaluates four selected tiles. It:

1. Excludes non-finite or non-positive SAR values and pixels with mask value 255.
2. Selects eligible pixels where the loaded flood mask equals 1.
3. Generates a Grad-CAM map for the selected flood target.
4. Selects high-attribution pixels from the top 10% and low-attribution pixels from the bottom 10% of attribution values within the eligible flood region.
5. Perturbs equal numbers of high- and low-attribution pixels, up to the configured maximum.
6. Perturbs a random control group sampled from the remaining eligible flood pixels.
7. Replaces selected values in both normalized VV and VH channels with 5x5 local-median values.
8. Measures absolute changes in the mean target flood-logit score.

The final tile has fewer eligible pixels, so its matched comparison uses 2,190 pixels per group rather than 3,000.

### 4.2 Results

| Tile (row, col) | Pixels per group | High-attribution change | Low-attribution change | Random-control change | High > low and random |
|---|---:|---:|---:|---:|---|
| (0, 1635) | 3,000 | 0.086403 | 0.033348 | 0.053226 | Yes |
| (384, 1536) | 3,000 | 0.115022 | 0.037918 | 0.053804 | Yes |
| (3072, 768) | 3,000 | 0.097636 | 0.033780 | 0.063112 | Yes |
| (768, 384) | 2,190 | 0.120044 | 0.042164 | 0.056617 | Yes |

All four rows completed with status `ok`.

### 4.3 Descriptive summary

| Group | Mean absolute score change |
|---|---:|
| High attribution | 0.104776 |
| Low attribution | 0.036803 |
| Random control | 0.056690 |

The mean high-attribution change is approximately 2.85 times the mean low-attribution change and 1.85 times the mean random-control change.

These are descriptive summaries across four selected tiles, not statistical significance tests. They should not be interpreted as population-level estimates.

### 4.4 Interpretation

In all four tested tiles, perturbing high-attribution pixels changed the target flood logit more than perturbing the same number of low-attribution pixels or random control pixels.

This is preliminary evidence that the Grad-CAM rankings are informative about model sensitivity for the chosen target on these tiles.

It does not prove causal correctness, establish that highlighted pixels are physically responsible for flooding, or demonstrate generalization across Chennai.

## 5. Earlier spatial and boundary diagnostics

The existing report `data/processed/chennai/explainability/day4/DAY4_XAI_DIAGNOSTIC_REPORT.md` documented:

- Higher mean CAM values within predicted-flood regions than predicted-non-flood regions on all four examined tiles.
- Potential external-boundary concentration in tiles `(0, 1635)` and `(384, 1536)`.
- Less boundary concentration in the other two tiles.
- Limitations from overlapping tiles, the small sample, and the use of the model's own flood mask.

The CAM-to-mask association is not independent validation because the explanation target and mask are derived from the model prediction. The boundary findings remain diagnostic flags; the statistics alone cannot determine whether responses are artifacts or genuine scene features.

The Day 4 report records the earlier zero-replacement experiment. Its numerical results must remain separate from the Day 5 v2 local-median results.

## 6. Limitations

1. **Limited sample:** Four selected tiles do not establish city-wide robustness.
2. **Target dependence:** Explanations concern a selected model flood-logit target, not ground-truth flood labels.
3. **Perturbation realism:** Local-median replacement can create inputs that differ from naturally observed SAR imagery.
4. **Boundary behavior:** Two tiles were flagged for potential external-boundary concentration; these findings have not been independently resolved here.
5. **Uncertainty:** Grad-CAM is not a calibrated uncertainty estimate. Confidence calibration and uncertainty validation require separate methods and metrics.
6. **Statistical strength:** No significance test or confidence interval was calculated for the four-tile comparison.
7. **Human-centered decisions:** These experiments do not evaluate whether emergency responders understand the explanations or make better decisions with them.
8. **Generalization:** The results do not establish performance across independent flood events, regions, seasons, or sensor conditions.

## 7. Reproducibility references

### Code
- `ai/satellite/src/explain_chennai_gradcam.py`
- `ai/satellite/src/validate_chennai_gradcam_day5_v2.py`

### Day 5 v2 outputs
- `data/processed/chennai/explainability/day5_v2/gradcam_day5_v2_robustness_summary.csv`
- `data/processed/chennai/explainability/day5_v2/gradcam_flood_r0_c1635.tif`
- `data/processed/chennai/explainability/day5_v2/gradcam_flood_r384_c1536.tif`
- `data/processed/chennai/explainability/day5_v2/gradcam_flood_r3072_c768.tif`
- `data/processed/chennai/explainability/day5_v2/gradcam_flood_r768_c384.tif`

### Earlier diagnostic
- `data/processed/chennai/explainability/day4/DAY4_XAI_DIAGNOSTIC_REPORT.md`
- `data/processed/chennai/explainability/day4/day4_spatial_audit_summary.csv`
- `data/processed/chennai/explainability/day4/day4_boundary_diagnostic_v2.csv`
- `data/processed/chennai/explainability/day3/gradcam_robustness_summary.csv`

## 8. Day 7 outcome

The existing Grad-CAM implementation and four-tile local-median perturbation validation have been reviewed and documented. All four Day 5 v2 comparisons favor high-attribution pixels over low-attribution and random controls.

**Day 7 XAI documentation outcome:** completed once this report is saved and its contents are verified in the repository.

**Overall XAI research status:** preliminary faithfulness evidence is available; comprehensive validation is not complete. Further work should prioritize independent evaluation, the flagged boundary cases, and a separate uncertainty-validation protocol rather than claiming that Grad-CAM alone establishes reliable emergency decisions.

## 9. Recommended future work

1. Visually inspect and document the two boundary-flagged tiles.
2. Expand faithfulness evaluation to a broader, justified sample of tiles and independent flood scenes.
3. Test sensitivity to the perturbation method and spatial sampling design.
4. Evaluate prediction quality against appropriate ground-truth labels separately from XAI faithfulness.
5. Define and validate uncertainty metrics independently of Grad-CAM.
6. Evaluate the interpretability and usefulness of explanations for human-centered emergency decision support.

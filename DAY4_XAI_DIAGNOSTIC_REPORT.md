# ResQTwin --- Day 4 XAI Diagnostic Report

**Date:** 21 October 2026\
**Workstream:** Satellite flood segmentation explainability (Grad-CAM)\
**Status:** Diagnostic review completed; findings are preliminary.

## 1. Objective

Consolidate the four-tile visual audit, spatial diagnostics,
boundary-concentration analysis, and Day 3 perturbation results. These
checks assess spatial consistency and sensitivity to perturbation; they
do **not** establish causal correctness or calibrated uncertainty.

## 2. Inputs and artifacts reviewed

-   `data/processed/chennai/explainability/day4/day4_four_tile_comparison.png`
-   `data/processed/chennai/explainability/day4/day4_spatial_audit_summary.csv`
-   `data/processed/chennai/explainability/day4/day4_boundary_diagnostic_v2.csv`
-   `data/processed/chennai/explainability/day3/gradcam_robustness_summary.csv`

The spatial audit reports equal valid SAR and CAM pixel counts for every
tile.

## 3. Spatial audit results

  --------------------------------------------------------------------------------------------------
  Tile         Valid   Predicted   Predicted  Mean CAM   CAM p90   Top-10%-CAM  Mean CAM Mean CAM in
  (row,       pixels       flood   non-flood                         pixels in  in flood   non-flood
  col)                                                               predicted
                                                                         flood
  -------- --------- ----------- ----------- --------- --------- ------------- --------- -----------
  (0,        151,011      45,023     105,988   0.12351   0.42268       99.927%   0.38203     0.01369
  1635)

  (384,      216,429      35,369     181,060   0.07316   0.37528       99.995%   0.40186     0.00895
  1536)

  (3072,     252,775      35,902     216,873   0.08696   0.43096       97.488%   0.50212     0.01823
  768)

  (768,      203,150      21,898     181,252   0.04625   0.25194       91.243%   0.34320     0.01037
  384)
  --------------------------------------------------------------------------------------------------

### Interpretation

-   Mean CAM values are higher within predicted-flood pixels than
    predicted-non-flood pixels in all four tiles.
-   Between 91.24% and 99.995% of top-10%-CAM pixels fall within
    predicted-flood regions. This is a consistency check, not
    independent validation, because the CAM target and mask come from
    the same model.
-   Raw CAM magnitudes should not be compared as calibrated importance
    across tiles, especially if maps were normalized independently.

## 4. Boundary-concentration diagnostic

The diagnostic used `filled_valid_footprint_distance`. The baseline is
the percentage of valid CAM pixels in the edge zone; the last column is
the percentage of selected top-CAM pixels located there.

  -----------------------------------------------------------------------
  Tile (row,       Edge-zone pixels  Edge zone as % of  Top-CAM pixels in
  col)                                    valid pixels          edge zone
  -------------- ------------------ ------------------ ------------------
  (0, 1635)                  16,179             10.71%             33.70%

  (384, 1536)                11,528              5.33%             22.06%

  (3072, 768)                 9,885              3.91%              0.26%

  (768, 384)                 13,354              6.57%              5.16%
  -----------------------------------------------------------------------

### Interpretation and flags

-   **Tiles (0, 1635) and (384, 1536):** top-CAM responses are
    concentrated in the edge zone beyond its share of valid pixels. Flag
    these for closer visual review. This may reflect boundary-related
    behavior, scene structure, or genuine features near the footprint
    edge; the statistic alone cannot distinguish these.
-   **Tile (3072, 768):** very little of the selected top-CAM response
    falls in the edge zone.
-   **Tile (768, 384):** the top-CAM edge-zone fraction is below the
    edge zone's baseline share.
-   The method uses a filled external valid-data footprint. It does not
    evaluate all internal invalid-patch boundaries, so it is not a
    complete nodata-artifact test.

## 5. Perturbation sensitivity results

The Day 3 test perturbed the same number of pixels in normalized VV and
VH inputs for each high/low comparison and measured the mean flood-logit
score. In all four tiles, the absolute score change after
high-attribution perturbation exceeded that after low-attribution
perturbation.

  ------------------------------------------------------------------------------------------------
  Tile          Pixels   Original   High-perturbed       High   Low-perturbed        Low    Change
  (row,      perturbed      score            score   absolute           score   absolute     ratio
  col)                                                 change                     change
  -------- ----------- ---------- ---------------- ---------- --------------- ---------- ---------
  (0,            3,000    3.37683          3.07018    0.30665         3.31963    0.05719     5.36×
  1635)

  (384,          3,000    2.97620          2.62595    0.35025         2.91334    0.06286     5.57×
  1536)

  (3072,         3,000    2.80252          2.48449    0.31803         2.74026    0.06226     5.11×
  768)

  (768,          2,190    3.26371          2.78446    0.47924         3.18711    0.07659     6.26×
  384)
  ------------------------------------------------------------------------------------------------

The final tile used 2,190 perturbed pixels rather than 3,000 because
fewer valid flood pixels were available. High/low perturbation counts
were matched within each tile, but not across all tiles.

### Interpretation

The result is consistent with high-attribution regions being more
influential on the selected flood-logit score than low-attribution
regions in these four tests. Change ratios range from approximately
5.11× to 6.26×.

### Limitations

-   Perturbations set normalized inputs to zero, which may create
    unrealistic or out-of-distribution inputs.
-   Four tiles are a small diagnostic sample, and two tiles overlap
    spatially.
-   Score changes are a sensitivity test, not proof of causal
    explanation or general attribution faithfulness.
-   The selected score is a flood logit, not a calibrated probability.

## 6. Overall assessment

**Positive evidence** 1. All four tiles passed the perturbation
comparison: high-attribution perturbations caused larger absolute
flood-logit changes than low-attribution perturbations. 2. CAM values
were higher in predicted-flood regions than predicted-non-flood regions
in all four tiles. 3. The boundary diagnostic revealed different
patterns across tiles rather than uniform edge concentration.

**Items requiring caution** 1. Two tiles show potential
external-boundary concentration and need visual review. 2. CAM-to-mask
association is not independent ground-truth validation. 3. Perturbations
may be unrealistic; results are preliminary sensitivity evidence. 4.
This audit does not validate or calibrate the project's uncertainty
outputs.

## 7. Conclusion

Day 4's comparison figure, spatial audit, boundary diagnostic, and
perturbation-summary review are complete. Tiles (0, 1635) and (384,
1536) are flagged for additional boundary-focused inspection. Findings
should be reported as preliminary diagnostic evidence---not proof of
causal correctness, calibrated uncertainty, or generalization.

## 8. Recommended follow-up

1.  Preserve the current model and Grad-CAM outputs.
2.  During Day 5, strengthen the robustness protocol and document the
    perturbation method and limitations.
3.  Evaluate uncertainty separately from Grad-CAM, with its own defined
    target, metric, and validation.
4.  Retain tile coordinates, parameters, code, and CSVs for
    reproducibility.

## 9. Reproducibility references

-   `ai/satellite/src/visualize_chennai_gradcam_day4.py`
-   `ai/satellite/src/audit_chennai_gradcam_boundaries_day4.py`
-   `ai/satellite/src/validate_chennai_gradcam_day3.py`
-   `data/processed/chennai/explainability/day3/gradcam_robustness_summary.csv`
-   `data/processed/chennai/explainability/day4/day4_spatial_audit_summary.csv`
-   `data/processed/chennai/explainability/day4/day4_boundary_diagnostic_v2.csv`

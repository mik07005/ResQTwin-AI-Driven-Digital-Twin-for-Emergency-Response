# ResQTwin AI — Environmental Flood Risk and Evidence Fusion

## 1. Model 2: Environmental Flood-Risk Baseline

### Purpose

Model 2 provides an experimental environmental risk assessment for the ResQTwin Chennai urban-flooding case study. It is intended to operate alongside the satellite-based inundation model.

The current implementation is a rule-based baseline, not a trained or calibrated flood-prediction model.

### Current inputs

- Recent rainfall observations, including timestamps and rainfall in mm.
- Weather forecast records with timestamps, precipitation, and optional precipitation probability.
- An optional location identifier.
- An optional timezone-aware prediction timestamp.

Rainfall telemetry is available in:

`data/raw/chennai/rtdas/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv`

### Current scoring approach

The implementation uses a provisional six-hour forecast horizon:

- Recent accepted rainfall contributes up to 40 points, scaled against 20 mm.
- Forecast precipitation within the six-hour horizon contributes up to 40 points, scaled against 30 mm.
- The maximum supplied precipitation probability contributes up to 20 points.

The score is clipped to 0–100. Provisional categories are:

- Low: below 25
- Moderate: 25 to below 50
- High: 50 to below 75
- Very high: 75 and above

These thresholds are engineering heuristics. They have not been calibrated against independent flood-event labels.

### Data-quality handling

- Missing or invalid rainfall observations do not count as zero rainfall.
- Stale rainfall observations make the score unavailable.
- Forecast records must fall strictly after the prediction time and within the six-hour horizon.
- Rainfall readings above the current 100 mm screening threshold are flagged and excluded from scoring pending verification.
- Missing inputs and warnings are returned explicitly.
- The source telemetry is preserved; the audit does not silently rewrite original readings.

### Current data limitations

The Chennai telemetry subset contains two stations: Chennai Mylapore (DGPOffice) and SGSWRDC Campus (Tharamani site). Their observation histories contain substantial gaps.

A review of 31 October through 4 November 2022 found seven Mylapore readings above 100 mm, with a maximum of 1,412.5 mm. Tharamani's maximum in that review window was 16.5 mm. This discrepancy warrants source-level review. It does not, by itself, establish a sensor failure or prove which reading is correct.

The source's accumulation interval, units, and handling of cumulative versus interval rainfall should be verified before operational use.

No independently verified historical Chennai flood-event label set has yet been established for validating Model 2's predictive performance.

### Output

The function `assess_environmental_risk` returns a timestamped result including:

- Environmental risk score, if calculable.
- Risk category or `unavailable`.
- Contributing factors.
- Uncertainty status.
- Data freshness.
- Missing-input messages and warnings.
- Count of suspicious readings excluded.
- Model version and an interpretation disclaimer.

The score is not a calibrated probability of flooding or a validated six-hour flood forecast.

---

## 2. Model 1: Satellite-Based Flood Inundation

### Purpose

Model 1 uses a U-Net-based deep-learning model to identify potential flood inundation from Sentinel-1 VV/VH satellite imagery.

It produces:

- A georeferenced flood-probability raster.
- A binary flood-inundation mask generated using a configurable threshold.

### Reusable inference

The reusable inference implementation is located at:

`ai/satellite/src/infer_sentinel1_scene.py`

The Chennai test scene is located at:

`data/processed/chennai/sentinel1/gcc/sentinel1_vv_vh_gcc_2025_utm44n.tif`

The reusable inference test outputs are:

- `data/processed/chennai/sentinel1/gcc/inference_reusable_test/flood_probability.tif`
- `data/processed/chennai/sentinel1/gcc/inference_reusable_test/flood_mask.tif`

### Model validation

The reusable inference implementation was compared against the original Chennai inference script using the same input scene.

- Binary masks matched across all 4,165,772 valid pixels.
- Valid probability values matched exactly.
- Maximum and mean absolute probability differences were both zero.

The held-out Sen1Floods11 test-set metrics were:

| Metric | Result |
|---|---:|
| IoU | 66.03% |
| Dice | 79.54% |
| Precision | 76.99% |
| Recall | 82.26% |

These metrics describe the held-out Sen1Floods11 evaluation, not an independent Chennai flood-event evaluation.

---

## 3. Uncertainty-Aware Evidence Fusion

### Purpose

The fusion layer integrates evidence from Model 1 and Model 2 while preserving the distinct meaning and uncertainty of each source.

It does not blindly average the satellite flood probability and environmental risk score.

### Implementation

Main implementation:

`ai/flood_prediction/src/fusion.py`

Main entry function:

`fuse_assessments(satellite_assessment, environmental_assessment, fusion_timestamp=None)`

Fusion contract:

`data/processed/chennai/fusion/fusion_io_contract.json`

### Satellite assessment processing

The fusion module validates:

- Existence of both probability and mask raster files.
- Single-band raster structure.
- Availability and consistency of CRS.
- Matching raster dimensions and geospatial transforms.
- Probability values within the range 0–1.
- Binary mask values restricted to 0 and 1.
- Consistency between the binary mask and probability threshold.
- Availability of overlapping valid pixels.

The satellite summary includes:

- Inference and acquisition timestamps.
- Age since inference.
- CRS and raster dimensions.
- Valid pixel count.
- Flooded pixel count and fraction.
- Mean and maximum flood probability.
- Segmentation threshold.
- Model version.
- Pixel area in map units squared.

Satellite freshness is explicitly marked as unassessed because a validated freshness threshold has not yet been configured.

### Environmental assessment processing

The fusion layer validates:

- Environmental risk score is numeric or null.
- Non-null scores fall between 0 and 100.
- Risk category matches the supplied score.
- Prediction timestamp is timezone-aware.

Environmental warnings and missing-input information are preserved.

### Source availability

The fusion layer distinguishes these states:

| Status | Meaning |
|---|---|
| `both_sources_available` | Both assessments are available |
| `satellite_only` | Satellite assessment is available |
| `environmental_only` | Environmental assessment is available |
| `insufficient_data` | Neither source is available |
| `temporal_validation_required` | Satellite inference timestamp is in the future |

Source availability does not establish spatial or temporal alignment between the two assessments.

### Uncertainty handling

The fusion layer follows these principles:

1. Satellite inundation and environmental risk remain separate evidence streams.
2. Missing or unavailable inputs are not silently replaced with zero.
3. Source-specific warnings and uncertainty are preserved.
4. Future satellite timestamps trigger temporal validation warnings.
5. No combined flood probability is generated.
6. No calibrated overall risk score is claimed.
7. Flooded pixel statistics are limited to overlapping valid raster coverage.

---

## 4. Real Chennai Satellite Integration Results

A real-data integration test was executed using the reusable Chennai satellite inference outputs.

**Input configuration**

- Probability raster: `flood_probability.tif`
- Binary mask: `flood_mask.tif`
- Threshold: 0.80
- CRS: EPSG:32644
- Environmental assessment: Not provided
- Acquisition timestamp: Unavailable

**Integration results**

| Metric | Result |
|---|---:|
| Overall status | `satellite_only` |
| Raster width | 2,147 pixels |
| Raster height | 4,233 pixels |
| Valid pixels | 4,165,772 |
| Flooded pixels | 308,654 |
| Flooded pixel fraction | 7.4093% |
| Mean flood probability | 0.160945 |
| Maximum flood probability | 0.994516 |
| Threshold | 0.80 |
| Pixel area | 100 m² |

Estimated flooded area within valid raster coverage:

30.8654 km².

This estimate is not the percentage or area of the entire Chennai study area. It describes predicted flooded pixels within the valid raster coverage only.

The integration completed successfully, including the probability-mask consistency validation.

### Timestamp limitation

The inference timestamp used in this integration test was derived from the probability raster's filesystem modification time.

It is a timestamp proxy, not a verified model inference timestamp or satellite acquisition timestamp. The acquisition timestamp remains unavailable.

### Integration interpretation

The integration confirms that the fusion implementation can process the real Chennai satellite outputs and produce a satellite-only assessment.

It does not establish:

- Model 2 predictive accuracy.
- Spatial alignment between rainfall stations and satellite coverage.
- Temporal alignment between rainfall observations and satellite acquisition.
- Operational flood-warning reliability.

---

## 5. Testing

Run the regression tests from the repository root:

```powershell
python -m unittest discover -s ai/flood_prediction/tests -v
```

### Current test results

| Test module | Tests passed |
|---|---:|
| Environmental risk | 8 |
| Fusion | 19 |
| **Total** | **27** |

The suite covers:

- Environmental input freshness and availability.
- Missing and extreme rainfall handling.
- Forecast horizon validation.
- Satellite raster validation.
- Geospatial alignment.
- Probability and mask consistency.
- Environmental score and category validation.
- Timestamp handling.
- Source availability combinations.
- Prevention of combined probability generation.

Passing these tests demonstrates expected software behaviour, not predictive accuracy or operational readiness.

---

## 6. Current Limitations

1. The environmental risk baseline remains experimental and rule-based.
2. Environmental scoring thresholds have not been calibrated against independent flood-event labels.
3. Satellite freshness has not been validated against a defined threshold.
4. A verified satellite acquisition timestamp is not currently available for the demonstrated integration.
5. Spatial and temporal alignment between Model 1 and Model 2 has not been established.
6. Historical Chennai flood-event ground truth is still required for independent evaluation.
7. Flooded pixel statistics cannot be interpreted as city-wide inundation estimates without appropriate spatial coverage analysis.

---

## 7. Validation and Development Roadmap

1. Verify rainfall measurement semantics and extreme readings against source documentation or provider confirmation.
2. Obtain independent, timestamped flood-event labels for Chennai.
3. Define event matching, spatial units, forecast horizon, and evaluation metrics before tuning thresholds.
4. Evaluate Model 2 against simple baselines and report false alarms and missed events.
5. Validate satellite freshness and acquisition-time handling.
6. Establish spatial and temporal alignment requirements for both evidence streams.
7. Evaluate fusion behaviour using independently validated environmental and satellite assessments.
8. Preserve uncertainty, missing-data status, timestamps, and source freshness throughout backend and frontend integration.
9. Document failure cases and limitations before considering operational emergency-response use.

---

**Project:** ResQTwin — AI-Driven Digital Twin for Emergency Response
**Study area:** Chennai, Tamil Nadu, India
**AI scope:** Satellite inundation, environmental flood-risk assessment, and uncertainty-aware evidence fusion

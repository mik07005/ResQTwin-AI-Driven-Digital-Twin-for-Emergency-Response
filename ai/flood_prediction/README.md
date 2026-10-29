# Model 2: Environmental Flood-Risk Baseline

## Purpose

Model 2 provides an experimental environmental risk assessment for the
ResQTwin Chennai urban-flooding case study. It is intended to operate
alongside the satellite-based inundation model.

The current implementation is a rule-based baseline, not a trained or
calibrated flood-prediction model.

## Current inputs

- Recent rainfall observations, including timestamps and rainfall in mm.
- Weather forecast records with timestamps, precipitation, and optional
  precipitation probability.
- An optional location identifier.
- An optional timezone-aware prediction timestamp.

Rainfall telemetry is available in
`data/raw/chennai/rtdas/rainfall_tel_hr_tamil_nadu_sw_gw_tn_2021_2025.csv`.

## Current scoring approach

The implementation uses a provisional six-hour forecast horizon:

- Recent accepted rainfall contributes up to 40 points, scaled against
  20 mm.
- Forecast precipitation within the six-hour horizon contributes up to
  40 points, scaled against 30 mm.
- The maximum supplied precipitation probability contributes up to
  20 points.

The score is clipped to 0-100. Provisional categories are:

- Low: below 25
- Moderate: 25 to below 50
- High: 50 to below 75
- Very high: 75 and above

These thresholds are engineering heuristics. They have not been calibrated
against independent flood-event labels.

## Data-quality handling

- Missing or invalid rainfall observations do not count as zero rainfall.
- Stale rainfall observations make the score unavailable.
- Forecast records must fall strictly after the prediction time and within
  the six-hour horizon.
- Rainfall readings above the current 100 mm screening threshold are flagged
  and excluded from scoring pending verification.
- Missing inputs and warnings are returned explicitly.
- The source telemetry is preserved; the audit does not silently rewrite
  original readings.

## Current data limitations

The Chennai telemetry subset contains two stations: Chennai Mylapore
(DGPOffice) and SGSWRDC Campus (Tharamani site). Their observation histories
contain substantial gaps.

A review of 31 October through 4 November 2022 found seven Mylapore readings
above 100 mm, with a maximum of 1,412.5 mm. Tharamani's maximum in that
review window was 16.5 mm. This discrepancy warrants source-level review.
It does not, by itself, establish a sensor failure or prove which reading
is correct.

The source's accumulation interval, units, and handling of cumulative
versus interval rainfall should be verified before operational use.

No independently verified historical Chennai flood-event label set has yet
been established for validating Model 2's predictive performance.

## Output

The function `assess_environmental_risk` returns a timestamped result
including:

- Environmental risk score, if calculable.
- Risk category or `unavailable`.
- Contributing factors.
- Uncertainty status.
- Data freshness.
- Missing-input messages and warnings.
- Count of suspicious readings excluded.
- Model version and an interpretation disclaimer.

The score is not a calibrated probability of flooding or a validated
six-hour flood forecast.

## Testing

Run the regression tests from the repository root:

```powershell
python -m unittest discover -s ai/flood_prediction/tests -v
```

The current suite covers eight baseline behaviours. Passing these tests
demonstrates expected software behaviour, not predictive accuracy.

## Validation plan

1. Verify the rainfall dataset's measurement semantics and extreme readings
   against source documentation or provider confirmation.
2. Obtain independent, timestamped flood-event labels for Chennai.
3. Define event matching, spatial units, forecast horizon, and evaluation
   metrics before tuning thresholds.
4. Evaluate against simple baselines and report false alarms and missed
   events, not only aggregate accuracy.
5. Calibrate thresholds only after sufficient independent validation data
   are available.
6. Preserve uncertainty, missing-data status, timestamps, and source
   freshness when integrating with Model 1.

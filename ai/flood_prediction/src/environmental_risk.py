"""Initial environmental risk baseline for ResQTwin.

This module provides an experimental rule-based score, not a calibrated
probability of flooding. Thresholds require domain review and validation.
"""

from datetime import datetime, timezone


MODEL_VERSION = "0.1.0-baseline"
MAX_OBSERVATION_AGE_HOURS = 3
EXTREME_RAINFALL_MM = 100.0


def _parse_timestamp(value):
    """Parse an ISO-8601 timestamp and require an explicit timezone."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def assess_environmental_risk(
    observations,
    forecast,
    prediction_timestamp=None,
    location_id=None,
):
    """Return an experimental environmental risk assessment.

    Observation records:
        {"timestamp": ISO-8601 datetime, "rainfall_mm": number or None}

    Forecast records:
        {"timestamp": ISO-8601 datetime, "precipitation_mm": number or None,
         "precipitation_probability_percent": number or None}

    The score is not a flood probability. It is unavailable when inputs
    are insufficient, stale, or invalid.
    """
    now = _parse_timestamp(prediction_timestamp) if prediction_timestamp else datetime.now(timezone.utc)
    if now is None:
        raise ValueError("prediction_timestamp must be timezone-aware ISO-8601")

    warnings = []
    missing_inputs = []
    factors = []
    valid_observations = []
    suspicious_count = 0

    for item in observations or []:
        timestamp = _parse_timestamp(item.get("timestamp"))
        value = item.get("rainfall_mm")

        if timestamp is None:
            warnings.append("Observation has missing or invalid timezone-aware timestamp.")
            continue
        if value is None:
            warnings.append("Missing rainfall observation; it was not treated as zero.")
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            warnings.append("Non-numeric rainfall observation ignored.")
            continue

        if value < 0:
            warnings.append("Negative rainfall observation ignored.")
            continue

        age_hours = (now - timestamp).total_seconds() / 3600
        if age_hours < 0:
            warnings.append("Future-dated rainfall observation ignored.")
            continue
        if value > EXTREME_RAINFALL_MM:
            suspicious_count += 1
            warnings.append(
                "Extreme rainfall reading flagged for verification; not used in score."
            )
            continue

        valid_observations.append((timestamp, value, age_hours))

    if not valid_observations:
        missing_inputs.append("No valid, non-extreme rainfall observations.")
        latest_observation_age = None
    else:
        latest = max(valid_observations, key=lambda row: row[0])
        latest_observation_age = latest[2]

    valid_forecasts = []
    for item in forecast or []:
        timestamp = _parse_timestamp(item.get("timestamp"))
        if timestamp is None:
            warnings.append("Forecast has missing or invalid timezone-aware timestamp.")
            continue

        precipitation = item.get("precipitation_mm")
        probability = item.get("precipitation_probability_percent")

        try:
            precipitation = None if precipitation is None else float(precipitation)
            probability = None if probability is None else float(probability)
        except (TypeError, ValueError):
            warnings.append("Non-numeric forecast value ignored.")
            continue

        if precipitation is not None and precipitation < 0:
            warnings.append("Negative forecast precipitation ignored.")
            continue
        if probability is not None and not 0 <= probability <= 100:
            warnings.append("Forecast precipitation probability outside 0-100 ignored.")
            probability = None

        if precipitation is not None or probability is not None:
            valid_forecasts.append((timestamp, precipitation, probability))

    # Only use forecast records strictly after now and within the
    # provisional six-hour prediction horizon.
    horizon_end = now + __import__("datetime").timedelta(hours=6)
    in_horizon_forecasts = [
        row for row in valid_forecasts
        if now < row[0] <= horizon_end
    ]

    if not in_horizon_forecasts:
        missing_inputs.append(
            "No usable weather forecast values inside the six-hour horizon."
        )

    if latest_observation_age is not None and latest_observation_age > MAX_OBSERVATION_AGE_HOURS:
        warnings.append("Latest rainfall observation is stale.")
        missing_inputs.append("Recent rainfall observation is stale.")

    score = None
    category = "unavailable"
    uncertainty_status = "high_or_unassessed"

    if valid_observations and in_horizon_forecasts and not missing_inputs:
        # Transparent provisional heuristic; not calibrated against flood labels.
        latest = max(valid_observations, key=lambda row: row[0])
        recent_rain = latest[1]
        forecast_rain = sum(
            precipitation or 0.0
            for _, precipitation, _ in in_horizon_forecasts
        )
        probabilities = [
            probability for _, _, probability in in_horizon_forecasts
            if probability is not None
        ]
        max_probability = max(probabilities) if probabilities else None

        score = 0.0
        score += min(recent_rain / 20.0, 1.0) * 40.0
        score += min(forecast_rain / 30.0, 1.0) * 40.0
        if max_probability is not None:
            score += (max_probability / 100.0) * 20.0

        score = round(max(0.0, min(score, 100.0)), 2)
        if score < 25:
            category = "low"
        elif score < 50:
            category = "moderate"
        elif score < 75:
            category = "high"
        else:
            category = "very_high"

        factors.append(f"Latest accepted rainfall: {recent_rain:.2f} mm.")
        factors.append(f"Forecast precipitation within six-hour horizon: {forecast_rain:.2f} mm.")
        if max_probability is not None:
            factors.append(f"Maximum supplied precipitation probability: {max_probability:.1f}%.")
        uncertainty_status = "high_unvalidated_heuristic"

    return {
        "prediction_timestamp": now.isoformat(),
        "target_horizon_hours": 6,
        "location_id": location_id,
        "environmental_risk_score": score,
        "risk_category": category,
        "contributing_factors": factors,
        "uncertainty_status": uncertainty_status,
        "data_freshness": {
            "latest_rainfall_age_hours": (
                round(latest_observation_age, 2)
                if latest_observation_age is not None else None
            ),
            "maximum_observation_age_hours": MAX_OBSERVATION_AGE_HOURS,
        },
        "missing_inputs": missing_inputs,
        "warnings": sorted(set(warnings)),
        "suspicious_readings_ignored": suspicious_count,
        "model_version": MODEL_VERSION,
        "interpretation": (
            "Experimental uncalibrated environmental score; not a flood probability "
            "or validated flood forecast."
        ),
    }

from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from environmental_risk import assess_environmental_risk


class EnvironmentalRiskTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)

    def ts(self, hours=0):
        return (self.now + timedelta(hours=hours)).isoformat()

    def observation(self, hours_ago=1, rainfall=8.0):
        return [{"timestamp": self.ts(-hours_ago), "rainfall_mm": rainfall}]

    def forecast(self, hours_ahead=3, rainfall=5.0, probability=60):
        return [{
            "timestamp": self.ts(hours_ahead),
            "precipitation_mm": rainfall,
            "precipitation_probability_percent": probability,
        }]

    def assess(self, observations, forecasts):
        return assess_environmental_risk(
            observations, forecasts,
            prediction_timestamp=self.now.isoformat(),
            location_id="test_location",
        )

    def test_fresh_inputs_produce_score(self):
        result = self.assess(self.observation(), self.forecast())
        self.assertIsNotNone(result["environmental_risk_score"])
        self.assertEqual(result["risk_category"], "moderate")

    def test_missing_rainfall_returns_unavailable(self):
        result = self.assess([], self.forecast())
        self.assertIsNone(result["environmental_risk_score"])
        self.assertEqual(result["risk_category"], "unavailable")

    def test_stale_rainfall_returns_unavailable(self):
        result = self.assess(self.observation(hours_ago=8), self.forecast())
        self.assertIsNone(result["environmental_risk_score"])
        self.assertTrue(result["missing_inputs"])

    def test_extreme_rainfall_is_flagged(self):
        result = self.assess(self.observation(rainfall=1412.5), self.forecast())
        self.assertIsNone(result["environmental_risk_score"])
        self.assertEqual(result["suspicious_readings_ignored"], 1)

    def test_outside_horizon_does_not_change_score(self):
        base = self.assess(self.observation(), self.forecast())
        outside = {
            "timestamp": self.ts(12),
            "precipitation_mm": 100.0,
            "precipitation_probability_percent": 100,
        }
        changed = self.assess(self.observation(), self.forecast() + [outside])
        self.assertEqual(
            base["environmental_risk_score"],
            changed["environmental_risk_score"],
        )

    def test_forecast_at_now_is_outside_future_window(self):
        result = self.assess(self.observation(), self.forecast(hours_ahead=0))
        self.assertIsNone(result["environmental_risk_score"])

    def test_forecast_beyond_horizon_alone_is_unavailable(self):
        result = self.assess(self.observation(), self.forecast(hours_ahead=12))
        self.assertIsNone(result["environmental_risk_score"])

    def test_naive_prediction_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_environmental_risk(
                self.observation(), self.forecast(),
                prediction_timestamp="2026-10-29T12:00:00",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

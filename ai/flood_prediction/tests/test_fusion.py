
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fusion import fuse_assessments


class FusionTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.now = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)

        self.probability_path = self.root / "probability.tif"
        self.mask_path = self.root / "mask.tif"

        self.create_rasters()

        self.satellite = {
            "probability_raster_path": str(self.probability_path),
            "mask_raster_path": str(self.mask_path),
            "acquisition_timestamp": "2026-10-31T18:00:00+00:00",
            "inference_timestamp": self.now.isoformat(),
            "crs": "EPSG:32644",
            "threshold": 0.80,
            "model_version": "test-unet-0.1",
        }

        self.environmental = {
            "prediction_timestamp": self.now.isoformat(),
            "environmental_risk_score": 34.67,
            "risk_category": "moderate",
            "uncertainty_status": "high_unvalidated_heuristic",
            "data_freshness": {
                "latest_rainfall_age_hours": 1.0
            },
            "missing_inputs": [],
            "warnings": [],
            "model_version": "0.1.0-baseline",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_rasters(self, probability=None, mask=None, transform=None):
        if probability is None:
            probability = np.array(
                [[0.95, 0.85, 0.20],
                 [0.10, 0.70, 0.90]],
                dtype=np.float32,
            )

        if mask is None:
            mask = np.array(
                [[1, 1, 0],
                 [0, 0, 1]],
                dtype=np.uint8,
            )

        if transform is None:
            transform = from_origin(500000, 1500000, 10, 10)

        for path, array, dtype, nodata in [
            (self.probability_path, probability, "float32", np.nan),
            (self.mask_path, mask, "uint8", 255),
        ]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=array.shape[0],
                width=array.shape[1],
                count=1,
                dtype=dtype,
                crs="EPSG:32644",
                transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(array, 1)

    def fuse(self, satellite=None, environmental=None):
        return fuse_assessments(
            self.satellite if satellite is None else satellite,
            self.environmental if environmental is None else environmental,
            fusion_timestamp=self.now.isoformat(),
        )

    # --------------------------------------------------
    # Satellite and environmental availability tests
    # --------------------------------------------------

    def test_both_sources_available(self):
        result = self.fuse()

        self.assertEqual(
            result["overall_status"],
            "both_sources_available",
        )

    def test_satellite_only(self):
        result = fuse_assessments(
            self.satellite,
            None,
            fusion_timestamp=self.now.isoformat(),
        )

        self.assertEqual(
            result["overall_status"],
            "satellite_only",
        )

    def test_environmental_only(self):
        result = fuse_assessments(
            None,
            self.environmental,
            fusion_timestamp=self.now.isoformat(),
        )

        self.assertEqual(
            result["overall_status"],
            "environmental_only",
        )

    def test_neither_source_available(self):
        result = fuse_assessments(
            None,
            None,
            fusion_timestamp=self.now.isoformat(),
        )

        self.assertEqual(
            result["overall_status"],
            "insufficient_data",
        )

    def test_unavailable_environmental_assessment(self):
        unavailable = dict(self.environmental)
        unavailable["environmental_risk_score"] = None
        unavailable["risk_category"] = "unavailable"

        result = self.fuse(environmental=unavailable)

        self.assertEqual(
            result["overall_status"],
            "satellite_only",
        )

        self.assertEqual(
            result["environmental_risk"]["availability"],
            "unavailable",
        )

    # --------------------------------------------------
    # Satellite timestamp and freshness tests
    # --------------------------------------------------

    def test_future_satellite_timestamp_requires_temporal_validation(self):
        future_satellite = dict(self.satellite)
        future_satellite["inference_timestamp"] = (
            "2026-11-01T01:00:00+00:00"
        )

        result = self.fuse(satellite=future_satellite)

        self.assertEqual(
            result["overall_status"],
            "temporal_validation_required",
        )

        self.assertEqual(
            result["satellite_inundation"]["availability"],
            "available",
        )

        self.assertLess(
            result["satellite_inundation"]["age_since_inference_hours"],
            0,
        )

        self.assertTrue(
            any(
                "Satellite inference timestamp is in the future" in warning
                for warning in result["warnings"]
            )
        )

    def test_satellite_freshness_is_explicitly_unassessed(self):
        result = self.fuse()

        satellite = result["satellite_inundation"]

        self.assertEqual(
            satellite["freshness_status"],
            "not_assessed_no_validated_threshold",
        )

        self.assertTrue(
            any(
                "Satellite freshness is not assessed" in warning
                for warning in result["warnings"]
            )
        )

    # --------------------------------------------------
    # Satellite raster statistics tests
    # --------------------------------------------------

    def test_satellite_statistics(self):
        result = self.fuse()
        satellite = result["satellite_inundation"]

        self.assertEqual(
            satellite["valid_pixel_count"],
            6,
        )

        self.assertEqual(
            satellite["flooded_pixel_count"],
            3,
        )

        self.assertAlmostEqual(
            satellite["flooded_pixel_fraction"],
            0.5,
        )

    def test_mask_probability_mismatch_rejected(self):
        inconsistent_mask = np.array(
            [[1, 0, 0],
             [0, 0, 1]],
            dtype=np.uint8,
        )

        self.create_rasters(mask=inconsistent_mask)

        with self.assertRaisesRegex(
            ValueError,
            "mask is inconsistent with the probability raster",
        ):
            self.fuse()

    def test_probability_equal_to_threshold_is_flooded(self):
        probability = np.array(
            [[0.80, 0.79, 0.20],
             [0.10, 0.70, 0.90]],
            dtype=np.float32,
        )

        expected_mask = np.array(
            [[1, 0, 0],
             [0, 0, 1]],
            dtype=np.uint8,
        )

        self.create_rasters(
            probability=probability,
            mask=expected_mask,
        )

        result = self.fuse()

        self.assertEqual(
            result["satellite_inundation"]["flooded_pixel_count"],
            2,
        )

    def test_invalid_probability_rejected(self):
        invalid_probability = np.array(
            [[0.95, 0.85, 1.20],
             [0.10, 0.70, 0.90]],
            dtype=np.float32,
        )

        self.create_rasters(probability=invalid_probability)

        with self.assertRaises(ValueError):
            self.fuse()

    def test_misaligned_rasters_rejected(self):
        with rasterio.open(self.mask_path, "r+") as dst:
            dst.transform = from_origin(
                500100,
                1500000,
                10,
                10,
            )

        with self.assertRaises(ValueError):
            self.fuse()

    def test_missing_satellite_file(self):
        broken = dict(self.satellite)
        broken["probability_raster_path"] = str(
            self.root / "missing.tif"
        )

        with self.assertRaises(FileNotFoundError):
            self.fuse(satellite=broken)

    # --------------------------------------------------
    # Environmental score and category tests
    # --------------------------------------------------

    def test_environmental_score_is_preserved(self):
        result = self.fuse()

        self.assertEqual(
            result["environmental_risk"]["environmental_risk_score"],
            34.67,
        )

    def test_invalid_environmental_score_rejected(self):
        invalid = dict(self.environmental)
        invalid["environmental_risk_score"] = 120

        with self.assertRaises(ValueError):
            self.fuse(environmental=invalid)

    def test_environmental_category_mismatch_rejected(self):
        invalid = dict(self.environmental)
        invalid["environmental_risk_score"] = 34.67
        invalid["risk_category"] = "very_high"

        with self.assertRaisesRegex(
            ValueError,
            "category does not match its score",
        ):
            self.fuse(environmental=invalid)

    def test_environmental_category_boundaries(self):
        test_cases = [
            (0, "low"),
            (24.99, "low"),
            (25, "moderate"),
            (49.99, "moderate"),
            (50, "high"),
            (74.99, "high"),
            (75, "very_high"),
            (100, "very_high"),
        ]

        for score, category in test_cases:
            with self.subTest(score=score):
                assessment = dict(self.environmental)
                assessment["environmental_risk_score"] = score
                assessment["risk_category"] = category

                result = self.fuse(environmental=assessment)

                self.assertEqual(
                    result["environmental_risk"]["risk_category"],
                    category,
                )

    # --------------------------------------------------
    # Input validation and fusion output tests
    # --------------------------------------------------

    def test_naive_fusion_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            fuse_assessments(
                self.satellite,
                self.environmental,
                fusion_timestamp="2026-11-01T00:00:00",
            )

    def test_no_combined_probability_is_produced(self):
        result = self.fuse()

        self.assertNotIn(
            "combined_flood_probability",
            result,
        )

        self.assertNotIn(
            "overall_risk_score",
            result,
        )

        self.assertEqual(
            result["uncertainty_status"],
            "source_uncertainties_preserved_no_combined_probability",
        )


if __name__ == "__main__":
    unittest.main()

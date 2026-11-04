from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from satellite_adapter import load_satellite_assessment
from environmental_risk import assess_environmental_risk
from fusion import fuse_assessments


class AIIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.now = datetime(
            2026, 11, 3, 19, 0, tzinfo=timezone.utc
        )

        self.probability_path = self.root / "flood_probability.tif"
        self.mask_path = self.root / "flood_mask.tif"
        self.metadata_path = self.root / "inference_metadata.json"

        self.create_rasters()
        self.create_metadata()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_rasters(self):
        probability = np.array(
            [[0.95, 0.85, 0.20],
             [0.10, 0.70, 0.90]],
            dtype=np.float32,
        )

        mask = np.array(
            [[1, 1, 0],
             [0, 0, 1]],
            dtype=np.uint8,
        )

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

    def create_metadata(self):
        metadata = {
            "inference_completed_at": self.now.isoformat(),
            "acquisition_timestamp": None,
            "threshold": 0.80,
            "model_version": "integration-test-unet-0.1",
            "outputs": {
                "probability_raster": str(self.probability_path),
                "mask_raster": str(self.mask_path),
            },
            "scene_metadata": {
                "crs": "EPSG:32644",
            },
        }

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

    def test_complete_ai_pipeline(self):
        # Model 1: Load satellite assessment through the adapter.
        satellite = load_satellite_assessment(self.metadata_path)

        # Model 2: Generate environmental risk assessment.
        observations = [
            {
                "timestamp": (
                    self.now - timedelta(hours=1)
                ).isoformat(),
                "rainfall_mm": 10.0,
            }
        ]

        forecast = [
            {
                "timestamp": (
                    self.now + timedelta(hours=2)
                ).isoformat(),
                "precipitation_mm": 10.0,
                "precipitation_probability_percent": 60.0,
            }
        ]

        environmental = assess_environmental_risk(
            observations,
            forecast,
            prediction_timestamp=self.now,
        )

        # Fusion: Process both assessment outputs.
        result = fuse_assessments(
            satellite,
            environmental,
            fusion_timestamp=self.now.isoformat(),
        )

        # Verify end-to-end integration.
        self.assertEqual(
            result["overall_status"],
            "both_sources_available",
        )

        self.assertTrue(
            result["data_quality"]["satellite_available"]
        )

        self.assertTrue(
            result["data_quality"]["environmental_assessment_provided"]
        )

        self.assertEqual(
            result["satellite_inundation"]["model_version"],
            "integration-test-unet-0.1",
        )

        self.assertEqual(
            result["environmental_risk"]["model_version"],
            "0.1.0-baseline",
        )

        self.assertIsNone(result.get("combined_flood_probability"))

        self.assertEqual(
            result["uncertainty_status"],
            "source_uncertainties_preserved_no_combined_probability",
        )


if __name__ == "__main__":
    unittest.main()
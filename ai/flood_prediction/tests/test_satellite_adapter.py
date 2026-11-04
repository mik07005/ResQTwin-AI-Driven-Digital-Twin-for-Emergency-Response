import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from satellite_adapter import load_satellite_assessment


class SatelliteAdapterTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.probability_path = self.root / "probability.tif"
        self.mask_path = self.root / "mask.tif"
        self.metadata_path = self.root / "metadata.json"

        self.probability_path.touch()
        self.mask_path.touch()

        self.metadata = {
            "inference_completed_at": "2026-11-03T19:00:00+00:00",
            "acquisition_timestamp": None,
            "threshold": 0.80,
            "model_version": "test-unet-0.1",
            "outputs": {
                "probability_raster": str(self.probability_path),
                "mask_raster": str(self.mask_path),
            },
            "scene_metadata": {
                "crs": "EPSG:32644",
            },
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_metadata(self, metadata):
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file)

    def test_valid_metadata_loads(self):
        self.write_metadata(self.metadata)

        result = load_satellite_assessment(self.metadata_path)

        self.assertEqual(result["model_version"], "test-unet-0.1")
        self.assertEqual(result["threshold"], 0.80)
        self.assertEqual(result["crs"], "EPSG:32644")

    def test_missing_required_field_rejected(self):
        metadata = self.metadata.copy()
        del metadata["model_version"]

        self.write_metadata(metadata)

        with self.assertRaisesRegex(ValueError, "model_version"):
            load_satellite_assessment(self.metadata_path)

    def test_invalid_json_rejected(self):
        self.metadata_path.write_text("{invalid json", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid inference metadata JSON"):
            load_satellite_assessment(self.metadata_path)

    def test_naive_timestamp_rejected(self):
        metadata = self.metadata.copy()
        metadata["inference_completed_at"] = "2026-11-03T19:00:00"

        self.write_metadata(metadata)

        with self.assertRaisesRegex(ValueError, "timezone"):
            load_satellite_assessment(self.metadata_path)

    def test_invalid_threshold_rejected(self):
        metadata = self.metadata.copy()
        metadata["threshold"] = 1.5

        self.write_metadata(metadata)

        with self.assertRaisesRegex(ValueError, "Threshold"):
            load_satellite_assessment(self.metadata_path)

    def test_missing_probability_path_rejected(self):
        metadata = self.metadata.copy()
        del metadata["outputs"]["probability_raster"]

        self.write_metadata(metadata)

        with self.assertRaisesRegex(ValueError, "probability_raster"):
            load_satellite_assessment(self.metadata_path)

    def test_missing_crs_rejected(self):
        metadata = self.metadata.copy()
        metadata["scene_metadata"] = {}

        self.write_metadata(metadata)

        with self.assertRaisesRegex(ValueError, "CRS"):
            load_satellite_assessment(self.metadata_path)


if __name__ == "__main__":
    unittest.main()
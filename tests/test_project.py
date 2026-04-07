from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modeling import run_pipeline


class FracPumpFailureForecastingTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_pipeline(temp_dir)
            self.assertEqual(summary["dataset_source"], "frac_pump_failure_sample_ai4i_style")
            self.assertEqual(summary["pump_count"], 8)
            self.assertGreaterEqual(summary["roc_auc"], 0.88)
            self.assertGreaterEqual(summary["average_precision"], 0.82)
            self.assertGreaterEqual(summary["f1"], 0.77)

            forecast = pd.read_csv(Path(summary["forecast_artifact"]))
            self.assertEqual(len(forecast), 8)
            self.assertTrue(forecast["health_score"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main()

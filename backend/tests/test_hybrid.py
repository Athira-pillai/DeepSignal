import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from backend import main
from backend import evaluation
from backend.hybrid import combine_probabilities


class HybridCombinationTests(unittest.TestCase):
    def test_known_component_probabilities_are_combined_and_explained(self):
        result = combine_probabilities(0.8, 0.6, 0.0, 0.9)

        self.assertAlmostEqual(result["probability_up"], 0.70)
        self.assertAlmostEqual(result["confidence"], 0.70)
        self.assertEqual(result["signal"], "strong_buy")
        self.assertEqual(result["target"]["horizon_sessions"], 1)
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0)
        self.assertAlmostEqual(
            sum(item["weighted_contribution"] for item in result["components"].values()),
            result["probability_up"],
        )

    def test_rejects_incompatible_probability_inputs(self):
        with self.assertRaises(ValueError):
            combine_probabilities(1.2, 0.5, 0.0, 0.5)

    def test_evaluation_reports_hybrid_and_components_on_same_holdout(self):
        rng = np.random.default_rng(7)
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, 420))
        raw = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=len(close), freq="B"),
            "open": close * (1 - 0.001),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, len(close)),
        })
        config = {
            "lookback": 30, "lstm_units": 8, "epochs": 1, "rf_trees": 50,
            "transaction_cost_pct": 0.06, "slippage_pct": 0.04,
        }
        neutral_lstm = np.full
        with tempfile.TemporaryDirectory() as directory:
            report_path = os.path.join(directory, "runs.json")
            with mock.patch.object(
                evaluation, "_lstm_holdout_probabilities",
                side_effect=lambda frame, train_end, fit_end, test_start, test_end, config:
                    (neutral_lstm(test_end - test_start, 0.5), "test_stub"),
            ):
                report = evaluation.evaluate_direction(
                    "TEST", raw, config, report_path, main.indicators
                )
            self.assertEqual(report["status"], "ok")
            self.assertIn("hybrid", report["models"])
            self.assertEqual(report["split"]["models_share_test_window"], True)
            self.assertEqual(
                report["models"]["hybrid"]["trading"]["start"],
                report["models"]["random_forest"]["trading"]["start"],
            )
            with open(report_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)[0]["run_id"], report["run_id"])


if __name__ == "__main__":
    unittest.main()

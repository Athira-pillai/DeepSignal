import json
import os
import sys
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
            self.assertEqual(report["split"]["models_share_test_dates"], True)
            self.assertTrue(report["split"]["final_holdout_untouched_for_selection"])
            self.assertGreaterEqual(report["split"]["walk_forward_folds"], 2)
            self.assertEqual(report["walk_forward"]["status"], "ok")
            for fold in report["walk_forward"]["folds"]:
                self.assertEqual(fold["status"], "ok")
                self.assertLess(fold["train_end"], fold["validation_end"])
                self.assertEqual(fold["validation_end"], fold["test_start"])
                self.assertEqual(fold["model_fit_end"], fold["train_end"])
                self.assertLessEqual(fold["test_end"], report["split"]["final_holdout_start_index"])
                self.assertIn("brier_score", fold["models"]["hybrid"]["classification"])
            self.assertEqual(
                report["models"]["hybrid"]["trading"]["start"],
                report["models"]["random_forest"]["trading"]["start"],
            )
            self.assertIn("balanced_accuracy", report["models"]["hybrid"]["classification"])
            self.assertIn("roc_auc", report["models"]["hybrid"]["classification"])
            self.assertEqual(report["component_status"]["xgboost"], "disabled_by_configuration")
            with open(report_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)[0]["run_id"], report["run_id"])

    def test_walk_forward_folds_are_strictly_chronological_and_exclude_final_holdout(self):
        plan = evaluation.generate_walk_forward_folds(600)
        self.assertEqual(plan["status"], "ok")
        self.assertGreaterEqual(len(plan["folds"]), 2)
        for fold in plan["folds"]:
            self.assertLess(fold["train_end"], fold["validation_end"])
            self.assertEqual(fold["train_end"], fold["validation_start"])
            self.assertEqual(fold["validation_end"], fold["test_start"])
            self.assertLess(fold["test_start"], fold["test_end"])
            self.assertLessEqual(fold["test_end"], plan["final_holdout_start"])
        self.assertEqual(plan["folds"][-1]["test_end"], plan["final_holdout_start"])
        real_size_plan = evaluation.generate_walk_forward_folds(703)
        fold_sizes = [fold["test_end"] - fold["test_start"] for fold in real_size_plan["folds"]]
        self.assertGreaterEqual(len(fold_sizes), 2)
        self.assertLessEqual(max(fold_sizes) - min(fold_sizes), 1)
        self.assertGreater(min(fold_sizes), 1)

    def test_target_is_next_close_direction_and_drops_unlabeled_tail(self):
        source = pd.DataFrame({"close": [100.0, 110.0, 99.0]})
        labeled = evaluation._add_next_session_target(source)
        self.assertEqual(labeled["target"].tolist(), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(labeled["forward_return"].iloc[0], 0.1)
        self.assertAlmostEqual(labeled["forward_return"].iloc[1], -0.1)
        self.assertTrue(np.isnan(labeled["forward_return"].iloc[2]))

    def test_scaler_is_fitted_on_training_feature_rows_only(self):
        frame = pd.DataFrame({
            column: np.r_[np.arange(10, dtype=float), np.full(5, 1000.0)]
            for column in evaluation.SEQUENCE_COLUMNS
        })
        scaler, transformed = evaluation._fit_scale_sequence_features(frame, 10)
        self.assertTrue(np.all(scaler.data_max_ == 9.0))
        self.assertTrue(np.all(transformed[-1] > 1.0))

    def test_probability_metrics_handle_single_class_without_auc(self):
        metrics = evaluation._metrics([1, 1, 1], [1, 0, 1], [0.8, 0.3, 0.6])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNotNone(metrics["brier_score"])
        self.assertEqual(metrics["class_counts"], {"down_or_flat": 0, "up": 3})

    def test_missing_optional_xgboost_is_reported_without_failing_window(self):
        rng = np.random.default_rng(12)
        rows = 180
        frame = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=rows, freq="B"),
            "close": 100 + np.arange(rows, dtype=float),
            "forward_return": rng.normal(0, 0.01, rows),
            "target": np.arange(rows) % 2,
        })
        for column in evaluation.FEATURE_COLUMNS:
            frame[column] = rng.normal(size=rows)
        frame["sma20"] = frame["close"]
        frame["sma50"] = frame["close"] - 1
        frame["macd_signal"] = rng.normal(size=rows)
        config = {
            "rf_trees": 50, "lookback": 10, "lstm_units": 8, "epochs": 1,
            "transaction_cost_pct": 0.06, "slippage_pct": 0.04,
            "include_xgboost": True,
        }
        with mock.patch.dict(sys.modules, {"xgboost": None}), mock.patch.object(
            evaluation, "_lstm_holdout_probabilities",
            return_value=(np.full(20, 0.5), "test_stub"),
        ):
            result = evaluation._evaluate_window(frame, 120, 80, 120, 120, 140, config)
        self.assertEqual(result["status"], "ok")
        self.assertIn("xgboost", result["component_status"])
        self.assertEqual(result["component_status"]["xgboost"], "unavailable_ModuleNotFoundError")


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from backend.calibration import (
    apply_calibrator,
    calibration_metrics,
    fit_calibrator,
    select_no_trade_thresholds,
    select_validation_calibrator,
)


class CalibrationTests(unittest.TestCase):
    def test_validation_prefix_selects_method_and_test_rows_are_not_inputs(self):
        probabilities = np.linspace(0.1, 0.9, 100)
        labels = (np.arange(len(probabilities)) % 2).astype(int)
        selected = select_validation_calibrator(probabilities, labels)
        self.assertEqual(selected["status"], "ok")
        self.assertIn(selected["selected_method"], {"platt", "isotonic"})
        self.assertEqual(selected["samples"], len(labels))
        first_test = np.asarray([0.2, 0.8])
        first_prediction = apply_calibrator(selected, first_test)
        changed_test_labels = np.asarray([1, 0])
        self.assertEqual(changed_test_labels.shape, first_test.shape)
        np.testing.assert_allclose(first_prediction, apply_calibrator(selected, first_test))

    def test_calibrators_require_sufficient_binary_validation_data(self):
        unavailable = fit_calibrator("isotonic", [0.2, 0.8] * 10, [0, 1] * 10)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIn("insufficient", unavailable["fallback_reason"])
        insufficient = select_validation_calibrator([0.4] * 30, [0, 1] * 15)
        self.assertEqual(insufficient["status"], "unavailable")
        self.assertIsNone(insufficient["estimator"])

    def test_isotonic_calibration_is_available_with_a_larger_validation_window(self):
        probabilities = np.linspace(0.01, 0.99, 80)
        labels = (np.arange(80) % 3 != 0).astype(int)
        fitted = fit_calibrator("isotonic", probabilities, labels)
        self.assertEqual(fitted["status"], "ok")
        output = apply_calibrator(fitted, [0.2, 0.5, 0.8])
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_no_trade_cutoffs_are_validation_selected_and_finite(self):
        probabilities = np.linspace(0.05, 0.95, 120)
        labels = (probabilities >= 0.5).astype(int)
        dispersion = np.full(len(labels), 0.08)
        selected = select_no_trade_thresholds(probabilities, dispersion, labels)
        self.assertEqual(selected["status"], "ok")
        self.assertGreaterEqual(selected["validation_coverage"], 0.35)
        self.assertTrue(np.isfinite(selected["neutral_margin"]))
        self.assertTrue(np.isfinite(selected["maximum_dispersion"]))

    def test_calibration_metrics_include_brier_and_reliability(self):
        metrics = calibration_metrics([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6])
        self.assertEqual(metrics["status"], "ok")
        self.assertAlmostEqual(metrics["brier_score"], 0.085)
        self.assertEqual(sum(row["count"] for row in metrics["reliability"]), 4)


if __name__ == "__main__":
    unittest.main()

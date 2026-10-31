import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from backend import evaluation
from backend.regime_detection import (
    REGIME_FEATURES,
    build_regime_features,
    estimate_current_regime,
    evaluate_regime_walk_forward,
    fit_regime_model,
    infer_regime_sequence,
)


def synthetic_prices(rows=360, seed=71):
    rng = np.random.default_rng(seed)
    states = np.repeat([0, 1, 2], int(np.ceil(rows / 3)))[:rows]
    means = np.choose(states, [-0.0008, 0.0001, 0.0010])
    scales = np.choose(states, [0.022, 0.009, 0.012])
    returns = means + rng.normal(0, scales, rows)
    close = 100 * np.cumprod(1 + returns)
    spread = rng.uniform(0.003, 0.02, rows)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=rows, freq="B"),
        "open": close * (1 - returns / 3),
        "high": close * (1 + spread), "low": close * (1 - spread),
        "close": close, "volume": 1_000_000 * np.exp(rng.normal(0, .15, rows)),
    })


class RegimeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_prices()
        self.features = build_regime_features(self.prices)

    def test_feature_rows_at_cutoff_do_not_use_later_prices(self):
        altered = self.prices.copy()
        altered.loc[250:, ["high", "low", "close", "volume"]] *= 7
        before = build_regime_features(self.prices).set_index("date")
        after = build_regime_features(altered).set_index("date")
        common_date = before.index[200]
        pd.testing.assert_series_equal(before.loc[common_date, REGIME_FEATURES], after.loc[common_date, REGIME_FEATURES])

    def test_hmm_scaler_and_state_mapping_use_training_window_only(self):
        train = self.features.iloc[:220].copy()
        validation = self.features.iloc[220:270].copy()
        fitted = fit_regime_model(train, validation, random_seed=17)

        self.assertEqual(fitted["status"], "ok")
        self.assertEqual(fitted["scaler_fit_rows"], len(train))
        np.testing.assert_allclose(
            fitted["scaler"].mean_, train[REGIME_FEATURES].mean().to_numpy(), rtol=1e-10, atol=1e-10
        )
        self.assertIn(fitted["selected_states"], (2, 3))
        self.assertEqual(set(fitted["state_labels"]), set(fitted["training_statistics"]))
        highest_return_state = max(
            fitted["training_statistics"],
            key=lambda state: fitted["training_statistics"][state]["mean_daily_return"],
        )
        self.assertEqual(fitted["state_labels"][highest_return_state], "bullish/trending")
        median_volatility = np.median([
            item["mean_volatility_20d"] for item in fitted["training_statistics"].values()
        ])
        for state, label in fitted["state_labels"].items():
            if label == "bearish/high-risk":
                self.assertGreaterEqual(fitted["training_statistics"][state]["mean_volatility_20d"], median_volatility)
        self.assertEqual(len(fitted["candidate_scores"]), 2)
        self.assertTrue(all(item["train_converged"] for item in fitted["candidate_scores"]))
        self.assertEqual(
            fitted["selected_states"],
            max(fitted["candidate_scores"], key=lambda item: item["selection_score"])["states"],
        )
        for state, label in fitted["state_labels"].items():
            expected = fitted["training_statistics"][state]
            self.assertIn("mean_daily_return", expected)
            self.assertIn("/", label)

    def test_filtered_posterior_for_cutoff_does_not_depend_on_future_rows(self):
        fitted = fit_regime_model(self.features.iloc[:220], self.features.iloc[220:270], random_seed=17)
        self.assertEqual(fitted["status"], "ok")
        observed = self.features.iloc[220:290].copy().reset_index(drop=True)
        original = infer_regime_sequence(fitted, observed)
        changed_future = observed.copy()
        changed_future.loc[40:, REGIME_FEATURES] = 1000.0
        altered = infer_regime_sequence(fitted, changed_future)

        self.assertEqual(original[25], altered[25])
        self.assertEqual(original[25]["state_probabilities"], altered[25]["state_probabilities"])
        self.assertAlmostEqual(sum(original[25]["state_probabilities"].values()), 1.0, places=8)

    def test_short_history_and_missing_optional_dependency_are_explicit(self):
        short = estimate_current_regime(self.prices.iloc[:80])
        self.assertEqual(short["status"], "insufficient_history")
        self.assertIsNone(short["regime"])

        with mock.patch.dict(sys.modules, {"pomegranate": None, "pomegranate.hmm": None, "pomegranate.distributions": None}):
            missing = fit_regime_model(self.features.iloc[:220], self.features.iloc[220:270])
        self.assertEqual(missing["status"], "unavailable_dependency")
        self.assertIn("pomegranate unavailable", missing["fallback_reason"])
        invalid = estimate_current_regime(pd.DataFrame({"date": ["2025-01-01"]}))
        self.assertEqual(invalid["status"], "invalid_data")

    def test_non_converged_hmm_candidates_return_explicit_fallback(self):
        class FakeHMM:
            def __init__(self, distributions, **_kwargs):
                self.distributions = distributions
                self.calls = 0

            def fit(self, _values):
                self.calls += 1
                return self

            def log_probability(self, _values):
                return __import__("torch").tensor([float(self.calls)])

        class FakeNormal:
            def __init__(self, **_kwargs):
                pass

        fake_root = types.ModuleType("pomegranate")
        fake_hmm = types.ModuleType("pomegranate.hmm")
        fake_distributions = types.ModuleType("pomegranate.distributions")
        fake_hmm.DenseHMM = FakeHMM
        fake_distributions.Normal = FakeNormal
        fake_root.hmm = fake_hmm
        fake_root.distributions = fake_distributions
        with mock.patch.dict(sys.modules, {
            "pomegranate": fake_root,
            "pomegranate.hmm": fake_hmm,
            "pomegranate.distributions": fake_distributions,
        }):
            result = fit_regime_model(self.features.iloc[:220], self.features.iloc[220:270])
        self.assertEqual(result["status"], "fit_failed")
        self.assertEqual([item["status"] for item in result["candidate_failures"]], ["non_converged", "non_converged"])

    def test_current_estimate_contains_causal_provenance_and_probabilities(self):
        result = estimate_current_regime(self.prices)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["inference_cutoff"], self.features.iloc[-1]["date"].strftime("%Y-%m-%d"))
        self.assertTrue(result["training_window"]["end"] < result["validation_window"]["start"])
        self.assertEqual(result["training_window"]["rows"], result["training_window"]["scaler_fit_rows"])
        self.assertAlmostEqual(sum(result["state_probabilities"].values()), 1.0, places=8)
        self.assertEqual(len(result["state_history"]), 60)
        self.assertIn("sha256", result["data"])

    def test_walk_forward_comparison_reports_pooled_and_regime_strata(self):
        rows = 240
        prices = synthetic_prices(rows, seed=23)

        def indicators(source):
            frame = source.copy()
            for column in evaluation.FEATURE_COLUMNS:
                frame[column] = 0.01
            frame["sma20"] = frame["close"] * .999
            frame["sma50"] = frame["close"] * .998
            frame["macd_signal"] = 0.0
            frame["bb_upper"] = frame["close"] * 1.01
            frame["bb_lower"] = frame["close"] * .99
            return frame

        def fake_evaluate(frame, fit_end, train_end, validation_end, test_start, test_end, config):
            count = test_end - test_start
            labels = frame.iloc[test_start:test_end]["target"].astype(int).to_numpy()
            probabilities = np.full(count, .55)
            predictions = (probabilities >= .5).astype(int)
            return {"status": "ok", "models": {"hybrid": {
                "probabilities": probabilities, "predictions": predictions,
                "classification": evaluation._metrics(labels, predictions, probabilities),
            }}}

        fake_fit = {"status": "ok", "selected_states": 2, "state_labels": {0: "bearish/high-risk", 1: "bullish/trending"}, "candidate_scores": []}

        def fake_infer(_fitted, observed):
            return [{"regime": "bearish/high-risk" if i % 2 else "bullish/trending"} for i in range(len(observed))]

        config = {"transaction_cost_pct": .06, "slippage_pct": .04, "rf_trees": 50,
                  "lookback": 10, "lstm_units": 8, "epochs": 1, "include_xgboost": False}
        with mock.patch.object(evaluation, "_evaluate_window", side_effect=fake_evaluate), \
             mock.patch("backend.regime_detection.fit_regime_model", return_value=fake_fit), \
             mock.patch("backend.regime_detection.infer_regime_sequence", side_effect=fake_infer):
            result = evaluate_regime_walk_forward("TEST", prices, config, indicators)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["split"]["final_holdout_excluded"])
        self.assertGreater(result["without_regime"]["samples"], 0)
        self.assertEqual(set(result["with_regime_context"]), {"bearish/high-risk", "bullish/trending"})
        self.assertEqual(len(result["folds"]), result["split"]["walk_forward_folds"])


if __name__ == "__main__":
    unittest.main()

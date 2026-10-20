import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient

from backend import main


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.original_config = main.TUNING_CONFIG.copy()

    def tearDown(self):
        main.TUNING_CONFIG.clear()
        main.TUNING_CONFIG.update(self.original_config)

    def test_health_response_shape(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("time", payload)

    def test_prediction_route_returns_signal_evidence(self):
        prediction = {
            "stock_symbol": "AAPL",
            "signal_type": "buy",
            "sentiment_score": 0.2,
            "lstm_probability_up": 0.6,
            "classifier_probability_up": 0.55,
            "signal_explanation": {"ensemble_probability_up": 0.57},
        }
        with mock.patch.object(main, "build_prediction", return_value=prediction):
            response = self.client.get("/api/predict/AAPL?timeframe=1D")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ("stock_symbol", "signal_type", "lstm_probability_up", "classifier_probability_up", "signal_explanation"):
            self.assertIn(key, payload)
        self.assertEqual(payload["signal_type"], "buy")

    def test_prediction_rejects_unsupported_timeframe(self):
        response = self.client.get("/api/predict/AAPL?timeframe=4M")
        self.assertEqual(response.status_code, 422)

    def test_signal_threshold_boundaries(self):
        cases = {
            0.64: "strong_buy",
            0.54: "buy",
            0.53: "hold",
            0.47: "hold",
            0.46: "sell",
            0.36: "strong_sell",
        }
        for score, expected in cases.items():
            with self.subTest(score=score):
                self.assertEqual(main.signal_from_score(score), expected)

    def test_saved_tuning_values_are_validated_and_persisted(self):
        values = {
            "lookback": 42,
            "lstm_units": 64,
            "epochs": 9,
            "rf_trees": 320,
            "transaction_cost_pct": 0.08,
            "slippage_pct": 0.03,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "tuning_config.json")
            with mock.patch.object(main, "_tuning_config_path", config_path):
                response = self.client.post("/api/config", json=values)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["config"], values)
                self.assertEqual(self.client.get("/api/config").json(), values)
                with open(config_path, encoding="utf-8") as config_file:
                    self.assertEqual(json.load(config_file), values)

    def test_tuning_endpoint_rejects_out_of_range_values(self):
        response = self.client.post("/api/config", json={
            "lookback": 2,
            "lstm_units": 48,
            "epochs": 6,
            "rf_trees": 250,
            "transaction_cost_pct": 0.06,
            "slippage_pct": 0.04,
        })
        self.assertEqual(response.status_code, 422)


class DependencyFallbackTests(unittest.TestCase):
    def test_missing_news_providers_return_empty_list(self):
        with mock.patch.dict(os.environ, {"NEWS_API_KEY": ""}), \
             mock.patch.object(main.requests, "get", side_effect=OSError("network unavailable")), \
             mock.patch.object(main.yf, "Ticker", side_effect=OSError("Yahoo unavailable")):
            self.assertEqual(main.get_news("AAPL"), [])

    def test_finbert_failure_returns_neutral_labeled_articles(self):
        articles = [{"title": "Example headline", "source": "Test"}]
        with mock.patch.object(main, "finbert_pipeline", side_effect=ImportError("transformers unavailable")):
            score, labeled = main.sentiment_score(articles)
        self.assertEqual(score, 0.0)
        self.assertEqual(labeled[0]["sentiment"], None)
        self.assertEqual(labeled[0]["title"], "Example headline")

    def test_missing_tensorflow_uses_neutral_lstm_probability(self):
        with mock.patch.dict(sys.modules, {"tensorflow": None}):
            self.assertEqual(main.lstm_direction(pd.DataFrame()), 0.5)


if __name__ == "__main__":
    unittest.main()

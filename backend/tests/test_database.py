import json
import os
import tempfile
import unittest
from unittest import mock

from backend import database


class FakeResponse:
    ok = True
    status_code = 200
    content = b'[{"id":"stored-id"}]'
    text = ""

    @staticmethod
    def json():
        return [{"id": "stored-id"}]


class JsonResponse:
    def __init__(self, payload):
        self.ok = True
        self.status_code = 200
        self.payload = payload
        self.content = json.dumps(payload).encode("utf-8") if payload is not None else b""
        self.text = ""

    def json(self):
        return self.payload


class StorageTests(unittest.TestCase):
    def test_local_prediction_and_experiment_survive_a_new_read(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"SUPABASE_URL": "", "SUPABASE_SECRET_KEY": ""}
        ):
            predictions_path = os.path.join(directory, "predictions.json")
            prediction = {
                "stock_symbol": "AAPL",
                "prediction_timestamp": "2026-10-24T10:00:00+00:00",
                "signal_type": "buy",
                "signal_explanation": {"hybrid_version": "hybrid-v1"},
                "analysis_summary": "Test signal",
                "market_data": {"sha256": "abc"},
            }
            result = database.save_prediction_bundle(prediction, predictions_path)
            self.assertEqual(result["status"], "saved_locally")
            history = database.list_predictions("AAPL", 10, predictions_path)
            self.assertEqual(history[0]["prediction"]["stock_symbol"], "AAPL")

            experiments_path = os.path.join(directory, "evaluation_runs.json")
            report = {"run_id": "run-1", "symbol": "AAPL", "data": {"sha256": "abc"}}
            database.save_experiment(report, "evaluation", experiments_path)
            restarted_read = database.list_experiments("evaluation", 10, directory)
            self.assertEqual(restarted_read[0]["run_id"], "run-1")

    def test_supabase_experiment_uses_upsert_and_backend_secret_header(self):
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test_only",
        }
        report = {
            "run_id": "repeatable-run-id",
            "symbol": "MSFT",
            "data": {"sha256": "data-hash", "start": "2025-01-01", "end": "2026-01-01"},
            "summary": {"strategy": {"total_return_pct": 4.2}},
            "assumptions": {"slippage_pct": 0.04},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, env), mock.patch.object(
            database.requests, "request", return_value=FakeResponse()
        ) as request:
            result = database.save_experiment(report, "backtest", os.path.join(directory, "runs.json"))

        self.assertEqual(result["status"], "saved")
        self.assertEqual(request.call_args.kwargs["params"]["on_conflict"], "run_id")
        self.assertIn("resolution=merge-duplicates", request.call_args.kwargs["headers"]["Prefer"])
        self.assertEqual(request.call_args.kwargs["headers"]["apikey"], "sb_secret_test_only")
        self.assertNotIn("Authorization", request.call_args.kwargs["headers"])
        self.assertNotIn("sb_secret_test_only", json.dumps(report))

    def test_supabase_prediction_saves_market_data_prediction_and_signal(self):
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test_only",
        }
        prediction = {
            "stock_symbol": "AAPL",
            "model_version": "deepsignal-model-v1-0123456789abcdef0123",
            "prediction_timestamp": "2026-10-24T10:00:00+00:00",
            "signal_type": "buy",
            "predicted_trend": "bullish",
            "predicted_change": 1.2,
            "signal_explanation": {
                "hybrid_version": "hybrid-v1",
                "target": {"horizon_sessions": 1},
                "ensemble_probability_up": 0.7,
            },
            "analysis_summary": "Test signal",
            "market_data": {
                "source": "Yahoo Finance", "interval": "1d",
                "market_timestamp": "2026-10-23T00:00:00+00:00",
                "retrieved_at": "2026-10-24T10:00:00+00:00",
                "sha256": "data-hash", "row_count": 500,
                "quality_status": "ok", "window_start": "2024-01-01",
                "window_end": "2026-10-23",
            },
        }
        responses = [
            JsonResponse([{"id": "market-id"}]),
            JsonResponse([{"id": "prediction-id"}]),
            JsonResponse(None),
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, env), mock.patch.object(
            database.requests, "request", side_effect=responses
        ) as request:
            result = database.save_prediction_bundle(prediction, os.path.join(directory, "predictions.json"))

        self.assertEqual(result["prediction_id"], "prediction-id")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[1].kwargs["json"]["market_data_run_id"], "market-id")
        self.assertEqual(request.call_args_list[1].kwargs["json"]["model_version"], prediction["model_version"])
        self.assertEqual(request.call_args_list[2].kwargs["json"]["prediction_id"], "prediction-id")

    def test_supabase_history_reader_uses_symbol_filter(self):
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test_only",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            database.requests, "request", return_value=JsonResponse([])
        ) as request:
            rows = database.list_predictions("AAPL", 20)

        self.assertEqual(rows, [])
        self.assertEqual(request.call_args.kwargs["params"]["symbol"], "eq.AAPL")


if __name__ == "__main__":
    unittest.main()

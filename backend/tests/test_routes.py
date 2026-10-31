import unittest
from unittest import mock

from backend import main
from backend.main import app


class HistoryRouteTests(unittest.TestCase):
    def test_saved_history_routes_precede_dynamic_symbol_route(self):
        route_paths = [getattr(route, "path", None) for route in app.routes]

        dynamic_history_index = route_paths.index("/api/history/{symbol}")
        for saved_history_path in (
            "/api/history/predictions",
            "/api/history/experiments",
        ):
            with self.subTest(path=saved_history_path):
                self.assertLess(route_paths.index(saved_history_path), dynamic_history_index)

    def test_technical_and_hybrid_backtest_routes_are_separate(self):
        route_paths = [getattr(route, "path", None) for route in app.routes]
        self.assertIn("/api/backtest/{symbol}", route_paths)
        self.assertIn("/api/backtest/{symbol}/hybrid", route_paths)

    def test_regime_route_returns_estimator_provenance(self):
        route_paths = [getattr(route, "path", None) for route in app.routes]
        self.assertIn("/api/regime/{symbol}", route_paths)
        with mock.patch.object(main, "download_history", return_value=object()), mock.patch.object(
            main, "estimate_current_regime", return_value={"status": "ok", "regime": "sideways/low-trend"}
        ) as estimate:
            response = main.market_regime("aapl")
        self.assertEqual(response["symbol"], "AAPL")
        self.assertEqual(response["regime"], "sideways/low-trend")
        estimate.assert_called_once()

    def test_regime_route_handles_market_data_failure_without_breaking_api(self):
        with mock.patch.object(main, "download_history", side_effect=RuntimeError("provider down")):
            response = main.market_regime("AAPL")
        self.assertEqual(response["status"], "unavailable_data")
        self.assertIsNone(response["regime"])
        self.assertIn("RuntimeError", response["fallback_reason"])

    def test_hybrid_backtest_route_allows_explicit_adaptive_mode(self):
        with mock.patch.object(main, "download_history", return_value=object()), mock.patch.object(
            main, "run_hybrid_backtest", return_value={"status": "ok", "selected_mode": "adaptive"}
        ) as run, mock.patch.object(main.storage, "save_experiment", return_value={"status": "saved"}):
            response = main.hybrid_backtest("AAPL", mode="adaptive")
        self.assertEqual(response["selected_mode"], "adaptive")
        self.assertEqual(run.call_args.kwargs["selected_mode"], "adaptive")


if __name__ == "__main__":
    unittest.main()

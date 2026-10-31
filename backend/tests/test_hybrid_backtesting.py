import unittest
from unittest import mock

import numpy as np
import pandas as pd

from backend import evaluation
from backend.hybrid_backtesting import _signal_action, run_hybrid_backtest


class HybridBacktestTests(unittest.TestCase):
    def test_no_trade_retains_position_without_opening_or_exiting(self):
        self.assertEqual(_signal_action("no_trade"), "HOLD")

    def test_walk_forward_only_hybrid_signals_share_dates_and_exclude_holdout(self):
        rows = 360
        dates = pd.date_range("2024-01-01", periods=rows, freq="B")
        close = 100 + np.arange(rows) * 0.05 + np.sin(np.arange(rows) / 3)
        raw = pd.DataFrame({
            "date": dates, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": np.full(rows, 1000),
        })

        def indicators(source):
            frame = source.copy()
            for column in evaluation.FEATURE_COLUMNS:
                frame[column] = 0.01
            frame["sma20"] = frame["close"] - 0.1
            frame["sma50"] = frame["close"] - 0.2
            frame["macd_signal"] = 0.0
            frame["bb_upper"] = frame["close"] + 1
            frame["bb_lower"] = frame["close"] - 1
            for column in evaluation.SEQUENCE_COLUMNS:
                if column not in frame:
                    frame[column] = frame["close"]
            return frame

        def fake_window(frame, fit_end, train_end, validation_end, test_start, test_end, config):
            count = test_end - test_start
            model = lambda p: {"probabilities": np.full(count, p), "predictions": np.full(count, int(p >= .5))}
            return {
                "status": "ok", "models": {"lstm": model(.8), "random_forest": model(.7)},
                "component_status": {"lstm": "trained"},
                "test_start": frame.iloc[test_start]["date"].strftime("%Y-%m-%d"),
                "test_end": frame.iloc[test_end - 1]["date"].strftime("%Y-%m-%d"),
            }

        config = {"transaction_cost_pct": .06, "slippage_pct": .04, "rf_trees": 50,
                  "lookback": 10, "lstm_units": 8, "epochs": 1, "include_xgboost": False}
        fake_regime = {"status": "ok", "selected_states": 2,
                       "state_labels": {0: "bearish/high-risk", 1: "bullish/trending"}}
        with mock.patch.object(evaluation, "_evaluate_window", side_effect=fake_window), \
             mock.patch("backend.hybrid_backtesting.fit_regime_model", return_value=fake_regime), \
             mock.patch("backend.hybrid_backtesting.infer_regime_sequence", side_effect=lambda _fit, frame: [
                 {"date": row["date"].strftime("%Y-%m-%d"), "regime": "bullish/trending"}
                 for _, row in frame.iterrows()
             ]):
            report = run_hybrid_backtest("TEST", raw, config, indicators)
            adaptive_report = run_hybrid_backtest("TEST", raw, config, indicators, selected_mode="adaptive")
            no_regime_config = {**config, "adaptive_use_regime": False}
            no_regime_report = run_hybrid_backtest("TEST", raw, no_regime_config, indicators)

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["data"]["final_holdout_excluded"])
        self.assertEqual(len(report["model_comparison"]), 5)
        self.assertEqual(report["selected_mode"], "fixed")
        self.assertEqual(report["summary"]["strategy"], report["summary"]["fixed_hybrid"])
        self.assertEqual(report["summary"]["strategy"], report["summary"]["hybrid"])
        self.assertEqual(len(report["fold_strategy_comparison"]), len(report["folds"]))
        for fold in report["fold_strategy_comparison"]:
            self.assertEqual(set(fold["strategies"]), {"fixed_hybrid", "adaptive_hybrid"})
            self.assertEqual(fold["signal_dates"], fold["executable_signals"] + 1)
            self.assertIn("total_return_pct", fold["strategies"]["fixed_hybrid"])
            self.assertIn("annualized_sharpe", fold["strategies"]["adaptive_hybrid"])
        self.assertEqual(adaptive_report["selected_mode"], "adaptive")
        self.assertEqual(adaptive_report["summary"]["strategy"], adaptive_report["summary"]["adaptive_hybrid"])
        self.assertEqual(report["summary"]["fixed_hybrid"], adaptive_report["summary"]["fixed_hybrid"])
        self.assertTrue(all(len(curve) == len(report["dates"]) for curve in report["model_curves"].values()))
        self.assertFalse(report["signal_audit"][-1]["executable"])
        self.assertEqual(report["signal_audit"][0]["hybrid_signal"], "strong_buy")
        self.assertAlmostEqual(sum(report["signal_audit"][0]["adaptive_weights"].values()), 1.0)
        self.assertEqual(report["signal_audit"][0]["regime"], "bullish/trending")
        self.assertGreaterEqual(report["signal_audit"][0]["validation_period"]["samples"], 20)
        self.assertEqual(report["signal_audit"][0]["adaptive_selection"]["status"], "ok")
        self.assertEqual(no_regime_report["configuration"]["adaptive_regime_context_enabled"], False)
        self.assertEqual(no_regime_report["signal_audit"][0]["adaptive_selection"]["scope"], "overall_validation")
        self.assertIn("raw_ensemble_probability_up", report["signal_audit"][0])
        self.assertIn("calibrated_probability_up", report["signal_audit"][0])
        self.assertIn("uncertainty", report["signal_audit"][0])
        self.assertTrue(np.isfinite(report["signal_audit"][0]["uncertainty"]["value"]))
        self.assertIn("calibrated_no_trade", report["model_comparison"])
        self.assertEqual(report["calibration_evaluation"]["pooled_test"]["raw_fixed"]["samples"],
                         len(report["signal_audit"]))
        self.assertTrue(all(
            t["execution_date"] > t["signal_date"]
            for t in report["model_trades"]["fixed_hybrid"] if t["reason"] == "signal"
        ))
        if report["model_trades"]["fixed_hybrid"]:
            trade = report["model_trades"]["fixed_hybrid"][0]
            self.assertIn("shares_before", trade)
            self.assertIn("shares_after", trade)
            self.assertIn("execution_timestamp", trade)


if __name__ == "__main__":
    unittest.main()

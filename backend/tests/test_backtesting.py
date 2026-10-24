import unittest

import pandas as pd

from backend.backtesting import run_backtest


class BacktestingEngineTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({
            "date": pd.date_range("2026-01-05", periods=4, freq="B"),
            "open": [100.0, 110.0, 120.0, 90.0],
            "high": [101.0, 111.0, 121.0, 91.0],
            "low": [99.0, 109.0, 119.0, 89.0],
            "close": [100.0, 110.0, 120.0, 90.0],
            "volume": [1000, 1000, 1000, 1000],
        })
        self.signals = pd.DataFrame({
            "date": self.prices["date"],
            "signal": ["BUY", "SELL", "HOLD", "HOLD"],
        })

    def test_signal_executes_at_next_open_and_account_is_tracked(self):
        report = run_backtest(
            self.prices,
            self.signals,
            symbol="TEST",
            config={"transaction_cost_pct": 0, "slippage_pct": 0},
            starting_cash=1000,
        )

        self.assertEqual(report["trades"][0]["signal_date"], "2026-01-05")
        self.assertEqual(report["trades"][0]["execution_date"], "2026-01-06")
        self.assertEqual(report["trades"][0]["side"], "BUY")
        self.assertEqual(report["trades"][1]["execution_date"], "2026-01-07")
        self.assertEqual(report["trades"][1]["side"], "SELL")
        self.assertEqual(report["equity"][0], 1000)
        self.assertAlmostEqual(report["summary"]["strategy"]["ending_equity"], 1000 * 120 / 110, places=2)

    def test_costs_and_slippage_are_reported(self):
        report = run_backtest(
            self.prices,
            self.signals,
            symbol="TEST",
            config={"transaction_cost_pct": 0.1, "slippage_pct": 0.1},
            starting_cash=1000,
        )
        no_costs = run_backtest(
            self.prices,
            self.signals,
            symbol="TEST",
            config={"transaction_cost_pct": 0, "slippage_pct": 0},
            starting_cash=1000,
        )

        self.assertGreater(report["trades"][0]["transaction_cost"], 0)
        self.assertGreater(report["trades"][0]["slippage_cost"], 0)
        self.assertEqual(report["assumptions"]["signals_are_past_only"], True)
        self.assertLess(
            report["summary"]["strategy"]["ending_equity"],
            no_costs["summary"]["strategy"]["ending_equity"],
        )


if __name__ == "__main__":
    unittest.main()

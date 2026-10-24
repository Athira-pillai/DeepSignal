import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from backend import scheduler


class MarketDataSchedulerTests(unittest.TestCase):
    def test_refresh_validates_features_and_persists_run_summary(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "open": [10.0, 11.0], "high": [11.0, 12.0],
            "low": [9.0, 10.0], "close": [10.5, 11.5], "volume": [100, 120],
        })
        featured = frame.assign(rsi=[50.0, 55.0], unused_nan=[float("nan"), float("nan")])
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"MARKET_DATA_SYMBOLS": "TEST", "SUPABASE_URL": "", "SUPABASE_SECRET_KEY": ""}
        ), mock.patch("backend.main.download_history", return_value=frame), mock.patch(
            "backend.main.indicators", return_value=featured
        ), mock.patch.object(scheduler.storage, "save_market_bars", return_value=2) as save_bars, mock.patch.object(
            scheduler.storage, "record_job_run"
        ) as record_run:
            result = scheduler.refresh_market_data_once()

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["rows_processed"], 2)
        self.assertEqual(result["details"]["symbols"], ["TEST"])
        records = save_bars.call_args.args[2]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["features"], {"rsi": 50.0})
        self.assertEqual(records[0]["volume"], 100)
        record_run.assert_called_once()

    def test_scheduler_symbol_list_uses_configured_tickers(self):
        with mock.patch.dict(os.environ, {"MARKET_DATA_SYMBOLS": " aapl, MSFT, aapl "}):
            self.assertEqual(scheduler._symbols(), ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()

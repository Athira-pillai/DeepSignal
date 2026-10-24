import os
import unittest
from unittest import mock

import pandas as pd

from backend import cache as cache_module
from backend import main


class TTLCacheTests(unittest.TestCase):
    def test_expiration_and_copy_protect_cached_values(self):
        cache = cache_module.TTLCache(max_entries=2)
        value = {"rows": [1, 2]}
        with mock.patch("backend.cache.time.monotonic", return_value=10.0):
            cache.set("key", value, 5)
        value["rows"].append(3)
        with mock.patch("backend.cache.time.monotonic", return_value=12.0):
            result = cache.get("key")
            result["rows"].append(4)
            second_read = cache.get("key")
        self.assertEqual(second_read, {"rows": [1, 2]})
        with mock.patch("backend.cache.time.monotonic", return_value=16.0):
            self.assertIs(cache.get("key"), cache_module.MISSING)
        self.assertEqual(cache.status()["expired"], 1)

    def test_repeated_price_history_request_avoids_provider_call(self):
        cache_module.cache.clear()
        provider_frame = pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [100]},
            index=pd.DatetimeIndex(["2026-01-02"], name="Date"),
        )
        ticker = mock.Mock()
        ticker.history.return_value = provider_frame
        with mock.patch.dict(os.environ, {"MARKET_DATA_CACHE_TTL_SECONDS": "900"}), mock.patch(
            "backend.main.yf.Ticker", return_value=ticker
        ) as ticker_factory:
            first = main.download_history("test", "1y")
            first.loc[0, "close"] = -1
            second = main.download_history("TEST", "1y")

        ticker_factory.assert_called_once_with("TEST")
        ticker.history.assert_called_once()
        self.assertEqual(second.loc[0, "close"], 10.5)


if __name__ == "__main__":
    unittest.main()

"""Create a deterministic, synthetic walk-forward report for smoke validation."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import evaluation, main


def main_run():
    seed = 1729
    rng = np.random.default_rng(seed)
    rows = 800
    returns = rng.normal(0.0003, 0.012, rows)
    close = 100 * np.cumprod(1 + returns)
    raw = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=rows, freq="B"),
        "open": close * (1 + rng.normal(0, 0.002, rows)),
        "high": close * 1.012,
        "low": close * 0.988,
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, rows),
    })
    config = {
        "lookback": 30,
        "lstm_units": 8,
        "epochs": 1,
        "rf_trees": 50,
        "transaction_cost_pct": 0.06,
        "slippage_pct": 0.04,
        "random_seed": seed,
        "include_xgboost": False,
        "data_source": "Deterministic synthetic smoke-test fixture; not market data",
    }
    output_path = ROOT / "research" / "walk_forward" / "synthetic_fixture_report.json"
    os.makedirs(output_path.parent, exist_ok=True)
    with mock.patch.object(
        evaluation, "_lstm_holdout_probabilities",
        side_effect=lambda frame, train_end, validation_end, test_start, test_end, config:
            (np.full(test_end - test_start, 0.5), "fixture_neutral_lstm"),
    ):
        report = evaluation.evaluate_direction("SYNTHETIC", raw, config, None, main.indicators)
    output_path.write_text(json.dumps([report], indent=2), encoding="utf-8")
    print({
        "status": report["status"],
        "data_source": report["data"]["source"],
        "data_sha256": report["data"]["sha256"],
        "walk_forward_folds": report["split"]["walk_forward_folds"],
        "final_holdout": [report["split"]["final_holdout_start"], report["split"]["final_holdout_end"]],
        "model_names": list(report["models"]),
            "report_path": str(output_path),
    })


if __name__ == "__main__":
    main_run()

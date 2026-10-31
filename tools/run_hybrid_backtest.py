"""Run and persist a reproducible real-data Phase 2 hybrid backtest report."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import main
from backend.hybrid_backtesting import run_hybrid_backtest


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--output", default=os.path.join(ROOT, "research", "real_data", "phase2_hybrid_backtest.json"))
    args = parser.parse_args()
    symbol = main.clean_symbol(args.symbol)
    raw = main.download_history(symbol, args.period, force_refresh=True)
    config = main.TUNING_CONFIG.copy()
    config["include_xgboost"] = False
    report = run_hybrid_backtest(symbol, raw, config, main.indicators)
    report["data"]["requested_period"] = args.period
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "status": report.get("status"), "symbol": symbol,
        "output": args.output, "window": report.get("data", {}).get("start"),
        "end": report.get("data", {}).get("end"), "rows": report.get("data", {}).get("rows"),
        "summary": report.get("summary"), "limitations": report.get("limitations"),
    }, indent=2))


if __name__ == "__main__":
    main_cli()

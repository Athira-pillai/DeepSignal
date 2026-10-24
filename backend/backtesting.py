"""Timestamp-aware portfolio simulation with explicit execution assumptions."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd


STRATEGY_VERSION = "technical-momentum-rsi-v1"
ENGINE_VERSION = "next-open-long-only-v1"


def _save_report(path, report):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        with open(path, encoding="utf-8") as handle:
            reports = json.load(handle)
        if not isinstance(reports, list):
            reports = []
    except (OSError, ValueError):
        reports = []
    reports.append(report)
    fd, temporary = tempfile.mkstemp(prefix="backtest-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metrics(equity, positions, trades, dates, starting_cash):
    values = np.asarray(equity, dtype=float)
    returns = pd.Series(values).pct_change().dropna().to_numpy()
    peak = np.maximum.accumulate(values)
    drawdown = values / peak - 1.0
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
    years = max((len(values) - 1) / 252.0, 1 / 252.0)
    ending = float(values[-1])
    cagr = ((ending / starting_cash) ** (1 / years) - 1) * 100 if ending > 0 else -100.0
    gross_traded = sum(float(trade["gross_value"]) for trade in trades)
    average_equity = float(np.mean(values)) or starting_cash
    return {
        "starting_cash": round(float(starting_cash), 2),
        "ending_equity": round(ending, 2),
        "total_return_pct": round((ending / starting_cash - 1) * 100, 4),
        "cagr_pct": round(float(cagr), 4),
        "max_drawdown_pct": round(float(np.min(drawdown) * 100), 4),
        "annualized_sharpe": round(sharpe, 4),
        "sessions": int(max(len(values) - 1, 0)),
        "exposure_pct": round(float(np.mean(np.asarray(positions) > 0) * 100), 4),
        "trade_count": int(len(trades)),
        "turnover": round(gross_traded / average_equity, 4),
        "window_start": dates[0],
        "window_end": dates[-1],
    }


def _simulate(prices, signal_map, starting_cash, position_fraction, cost_rate, slippage_rate, initial_buy=False):
    dates = prices["date"].dt.strftime("%Y-%m-%d").tolist()
    cash = float(starting_cash)
    shares = 0.0
    equity_curve = []
    cash_curve = []
    holdings_curve = []
    position_curve = []
    trades = []

    def transact(side, reference_price, execution_date, signal_date, reason):
        nonlocal cash, shares
        if side == "buy" and shares == 0:
            fill = reference_price * (1 + slippage_rate)
            equity_at_open = cash + shares * reference_price
            gross_budget = min(equity_at_open * position_fraction, cash / (1 + cost_rate))
            quantity = gross_budget / fill if fill > 0 else 0.0
            gross = quantity * fill
            fee = gross * cost_rate
            if quantity <= 0:
                return
            cash -= gross + fee
            shares += quantity
        elif side == "sell" and shares > 0:
            fill = reference_price * (1 - slippage_rate)
            quantity = shares
            gross = quantity * fill
            fee = gross * cost_rate
            cash += gross - fee
            shares = 0.0
        else:
            return
        trades.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "side": side.upper(),
            "reason": reason,
            "reference_price": round(float(reference_price), 6),
            "fill_price": round(float(fill), 6),
            "quantity": round(float(quantity), 8),
            "gross_value": round(float(gross), 4),
            "transaction_cost": round(float(fee), 4),
            "slippage_cost": round(float(abs(fill - reference_price) * quantity), 4),
        })

    for i, row in prices.iterrows():
        date = dates[i]
        if i == 0 and initial_buy:
            transact("buy", float(row["open"]), date, date, "buy_and_hold_entry")
        elif i > 0:
            signal_date = dates[i - 1]
            signal = signal_map.get(signal_date, "HOLD")
            if signal == "BUY":
                transact("buy", float(row["open"]), date, signal_date, "signal")
            elif signal == "SELL":
                transact("sell", float(row["open"]), date, signal_date, "signal")
        mark = cash + shares * float(row["close"])
        equity_curve.append(mark)
        cash_curve.append(cash)
        holdings_curve.append(shares * float(row["close"]))
        position_curve.append(1 if shares > 0 else 0)

    if shares > 0:
        last = prices.iloc[-1]
        transact("sell", float(last["close"]), dates[-1], dates[-1], "scheduled_end_liquidation")
        equity_curve[-1] = cash
        cash_curve[-1] = cash
        holdings_curve[-1] = 0.0
        position_curve[-1] = 0
    return {
        "equity": equity_curve,
        "cash": cash_curve,
        "holdings_value": holdings_curve,
        "positions": position_curve,
        "trades": trades,
        "metrics": _metrics(equity_curve, position_curve, trades, dates, starting_cash),
    }


def run_backtest(
    historical_prices,
    timestamped_signals,
    *,
    symbol,
    config,
    strategy_name="custom_timestamped_signals",
    strategy_version="custom-v1",
    strategy_parameters=None,
    starting_cash=100_000.0,
    position_fraction=1.0,
    persist_path=None,
):
    """Simulate signals created after close and executed at the next session open.

    `timestamped_signals` must contain `date` and `signal` columns with BUY,
    SELL or HOLD values. One long-only position is allowed at a time. The
    portfolio is valued at each close and liquidated at the final close.
    """
    required_prices = {"date", "open", "close", "volume", "high", "low"}
    missing = required_prices.difference(historical_prices.columns)
    if missing:
        raise ValueError(f"historical_prices missing columns: {sorted(missing)}")
    if not {"date", "signal"}.issubset(timestamped_signals.columns):
        raise ValueError("timestamped_signals must contain date and signal columns")
    if starting_cash <= 0 or not 0 < position_fraction <= 1:
        raise ValueError("starting_cash must be positive and position_fraction must be in (0, 1]")

    prices = historical_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices = prices.sort_values("date").drop_duplicates("date", keep="last").dropna(
        subset=["date", "open", "close"]
    ).reset_index(drop=True)
    if len(prices) < 2:
        raise ValueError("at least two valid daily price rows are required")
    signals = timestamped_signals[["date", "signal"]].copy()
    signals["date"] = pd.to_datetime(signals["date"], utc=True).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    signals["signal"] = signals["signal"].astype(str).str.upper()
    invalid = sorted(set(signals["signal"]) - {"BUY", "SELL", "HOLD"})
    if invalid:
        raise ValueError(f"unsupported signal values: {invalid}")
    signal_map = dict(zip(signals["date"], signals["signal"]))
    signal_rows = [
        {"date": row.date, "signal": row.signal}
        for row in signals.drop_duplicates("date", keep="last").sort_values("date").itertuples(index=False)
    ]
    signals_hash = hashlib.sha256(
        json.dumps(signal_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cost_rate = float(config["transaction_cost_pct"]) / 100
    slippage_rate = float(config["slippage_pct"]) / 100
    if not 0 <= cost_rate <= 1 or not 0 <= slippage_rate < 1:
        raise ValueError("transaction cost must be in [0, 100] percent and slippage in [0, 100) percent")

    strategy = _simulate(prices, signal_map, starting_cash, position_fraction, cost_rate, slippage_rate)
    benchmark = _simulate(prices, {}, starting_cash, 1.0, cost_rate, slippage_rate, initial_buy=True)
    no_trade_equity = [float(starting_cash)] * len(prices)
    no_trade = {
        "equity": no_trade_equity,
        "cash": no_trade_equity.copy(),
        "holdings_value": [0.0] * len(prices),
        "positions": [0] * len(prices),
        "trades": [],
        "metrics": _metrics(no_trade_equity, [0] * len(prices), [], prices["date"].dt.strftime("%Y-%m-%d").tolist(), starting_cash),
    }
    dates = prices["date"].dt.strftime("%Y-%m-%d").tolist()
    base = float(starting_cash)
    strategy_index = [round(x / base * 100, 4) for x in strategy["equity"]]
    benchmark_index = [round(x / base * 100, 4) for x in benchmark["equity"]]
    no_trade_index = [100.0] * len(prices)
    data_hash = hashlib.sha256(
        prices[["date", "open", "high", "low", "close", "volume"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "status": "ok",
        "symbol": symbol,
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "engine_version": ENGINE_VERSION,
        "model_version": None,
        "versions": {
            "trained_model_artifact": None,
            "strategy": strategy_version,
            "backtest_engine": ENGINE_VERSION,
        },
        "data": {
            "source": "Yahoo Finance adjusted daily OHLCV",
            "rows": int(len(prices)),
            "sha256": data_hash,
            "start": dates[0],
            "end": dates[-1],
            "signals_sha256": signals_hash,
        },
        "assumptions": {
            "signal_timestamp": "After the signal date session close.",
            "execution": "At the next available session open.",
            "slippage": "Adverse percentage adjustment to reference execution price.",
            "transaction_cost": "Percentage fee charged on each executed trade notional.",
            "position_sizing": "Long only; target fraction of account equity; fractional shares allowed.",
            "end_of_test": "Any open position is liquidated at the last close with costs.",
            "starting_cash": float(starting_cash),
            "position_fraction": float(position_fraction),
            "transaction_cost_pct": float(config["transaction_cost_pct"]),
            "slippage_pct": float(config["slippage_pct"]),
            "signals_are_past_only": True,
        },
        "configuration": {
            "strategy": strategy_name,
            "strategy_version": strategy_version,
            "strategy_parameters": strategy_parameters or {},
            "engine_version": ENGINE_VERSION,
        },
        "dates": dates,
        "signals": signal_rows,
        "equity": [round(float(x), 4) for x in strategy["equity"]],
        "cash": [round(float(x), 4) for x in strategy["cash"]],
        "holdings_value": [round(float(x), 4) for x in strategy["holdings_value"]],
        "position": strategy["positions"],
        "drawdown_pct": [round(float((x / peak - 1) * 100), 4) for x, peak in zip(strategy["equity"], np.maximum.accumulate(strategy["equity"]))],
        "trades": strategy["trades"],
        "summary": {
            "strategy": strategy["metrics"],
            "buy_hold": benchmark["metrics"],
            "no_trade": no_trade["metrics"],
            # Compatibility aliases for the existing frontend; the strategy is
            # not the full live hybrid and should be labelled by its name.
            "hybrid": strategy["metrics"],
            "naive": no_trade["metrics"],
        },
        "strategy_curve": strategy_index,
        "buy_hold": benchmark_index,
        "no_trade": no_trade_index,
        "naive": no_trade_index,
    }
    if persist_path:
        _save_report(persist_path, report)
    return report

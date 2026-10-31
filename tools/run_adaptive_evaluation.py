"""Reproduce fixed-versus-adaptive hybrid walk-forward trading evaluation."""

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import main
from backend.hybrid_backtesting import run_hybrid_backtest


def _adaptive_config(config):
    for env_name, key, cast, fallback in (
        ("ADAPTIVE_ENSEMBLE_MIN_WEIGHT", "adaptive_minimum_weight", float, .05),
        ("ADAPTIVE_ENSEMBLE_MAX_WEIGHT", "adaptive_maximum_weight", float, .80),
        ("ADAPTIVE_ENSEMBLE_TEMPERATURE", "adaptive_temperature", float, 8.0),
        ("ADAPTIVE_MIN_VALIDATION_SAMPLES", "adaptive_minimum_validation_samples", int, 20),
        ("ADAPTIVE_MIN_REGIME_SAMPLES", "adaptive_minimum_regime_samples", int, 20),
    ):
        try:
            config[key] = cast(os.getenv(env_name, str(fallback)))
        except ValueError:
            config[key] = fallback
    return config


def _write_markdown(report, path):
    fixed = report["summary"]["fixed_hybrid"]
    adaptive = report["summary"]["adaptive_hybrid"]
    lines = [
        "# Phase 4 — Fixed versus adaptive hybrid evaluation", "",
        f"**Ticker:** {report['symbol']}  ",
        f"**Window:** {report['data']['start']} through {report['data']['end']}  ",
        f"**Price rows:** {report['data']['rows']}  ",
        f"**Walk-forward folds:** {len(report['folds'])}; final holdout excluded: {report['data']['final_holdout_excluded']}", "",
        "## Results", "",
        "| Strategy | Ending equity | Return | Max drawdown | Sharpe | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("Fixed hybrid", fixed), ("Adaptive hybrid", adaptive),
        ("Buy and hold", report["summary"]["buy_hold"]),
        ("No-trade cash", report["summary"]["no_trade"]),
    ]
    for name, metrics in rows:
        lines.append(f"| {name} | ${metrics['ending_equity']:,.2f} | {metrics['total_return_pct']}% | {metrics['max_drawdown_pct']}% | {metrics['annualized_sharpe']} | {metrics['trade_count']} |")
    lines.extend(["", "## Method", "",
        "- Both hybrid strategies use identical walk-forward test dates, target, next-open execution, costs, slippage, sizing, and buy-and-hold/cash baselines.",
        "- For each test fold, component weights use only out-of-sample predictions and labels from that fold's immediately preceding validation window. Weights stay frozen over the following test fold.",
        "- Weights use Brier skill relative to a 0.50 direction probability plus renormalized fixed-component priors, projected onto configured bounds.",
        ("- Regime context is disabled for adaptive weight selection; scores use the overall prior validation window." if not report["configuration"].get("adaptive_regime_context_enabled", True) else "- Regime-specific validation scores are used when evidence is sufficient; otherwise overall validation evidence is used and logged."),
        "- HMM regime context is inferred causally through each test signal date. HMM fit/validation uses only earlier training and validation features; in the no-regime-weight control, those labels are not used to select weights.",
        "- Historical FinBERT news is unavailable. The adaptive strategy excludes sentiment and renormalizes available LSTM, Random Forest, and technical components; the fixed baseline retains its neutral sentiment contribution.",
        "- Both strategies use the same production probability-to-signal thresholds. The final test-date signal is audited and not executed without a next in-window open.",
        f"- Full-source OHLCV SHA-256: `{report['data']['full_source_sha256']}`; OOS prices SHA-256: `{report['data']['sha256']}`.", "",
        "## Fold audit", "",
        "| Fold | Test dates | Validation dates | HMM states | LSTM test status | Regime fallback |",
        "|---:|---|---|---:|---|---|"])
    for fold in report["folds"]:
        lines.append(f"| {fold['fold']} | {fold['test_start']} to {fold['test_end']} | {fold['validation_start']} to {fold['validation_end']} | {fold.get('regime_selected_states') or 'unavailable'} | {fold.get('lstm_status')} | {fold.get('regime_fallback_reason') or 'none'} |")
    audit = report["signal_audit"]
    lines.extend(["", "## Limitations", "",
        "- This is one ticker and one provider-dependent historical window. It does not establish future profitability or that adaptive weighting is better.",
        ("- This run intentionally uses overall validation scores only; it is the no-regime-weights control." if not report["configuration"].get("adaptive_regime_context_enabled", True) else "- Each test fold uses one preceding validation window; regime-specific validation evidence can be small and trigger the documented overall-window fallback."),
        "- Brier scores and regime labels are noisy on short windows. The weights are a transparent heuristic, not a trained or calibrated meta-model.",
        "- Historical point-in-time news is absent, so this experiment cannot measure FinBERT's historical incremental value.",
        "- Yahoo adjusted daily bars, fractional shares, constant fees/slippage, and next-open fills simplify spreads, market impact, partial fills, taxes, and funding costs.",
        "- The final chronological holdout is excluded. A separate later confirmation window and multi-ticker evaluation are needed before changing the fixed default.",
        "", f"Audit contains {len(audit)} test-date rows, including non-executable final-date signals. See the JSON for raw components, weights, validation metrics, availability, regimes, trades, and equity curves.", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--output", default=os.path.join(ROOT, "research", "real_data", "phase4_adaptive_evaluation.json"))
    parser.add_argument("--input-csv", help="Reuse a previously saved OHLCV frame for a paired/reproducible comparison.")
    parser.add_argument("--save-data", help="Save the downloaded OHLCV frame as CSV before evaluation.")
    parser.add_argument("--without-regime-weights", action="store_true",
                        help="Use overall validation weights instead of regime-specific validation weights (Phase 9 E4 ablation).")
    args = parser.parse_args()
    symbol = main.clean_symbol(args.symbol)
    if args.input_csv:
        prices = pd.read_csv(args.input_csv, parse_dates=["date"])
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(prices.columns)
        if missing:
            parser.error(f"input CSV is missing required OHLCV columns: {', '.join(sorted(missing))}")
        prices = prices[["date", "open", "high", "low", "close", "volume"]].copy()
    else:
        prices = main.download_history(symbol, args.period, force_refresh=True)
    if args.save_data:
        os.makedirs(os.path.dirname(args.save_data), exist_ok=True)
        prices.to_csv(args.save_data, index=False)
    config = _adaptive_config(main.TUNING_CONFIG.copy())
    config["include_xgboost"] = False
    config["adaptive_use_regime"] = not args.without_regime_weights
    report = run_hybrid_backtest(symbol, prices, config, main.indicators, selected_mode="fixed")
    report["data"]["requested_period"] = args.period
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown_path = os.path.splitext(args.output)[0] + ".md"
    _write_markdown(report, markdown_path)
    print(json.dumps({
        "status": report["status"], "symbol": symbol,
        "json": args.output, "markdown": markdown_path,
        "fixed": report["summary"]["fixed_hybrid"],
        "adaptive": report["summary"]["adaptive_hybrid"],
        "folds": report["folds"],
        "weight_sample": report["signal_audit"][0].get("adaptive_weights") if report["signal_audit"] else None,
    }, indent=2))


if __name__ == "__main__":
    main_cli()

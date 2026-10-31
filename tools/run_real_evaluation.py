"""Run and save a real-data Phase 1 walk-forward evaluation."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import evaluation, main  # noqa: E402


REPORT_DIR = ROOT / "research" / "real_data"
JSON_PATH = REPORT_DIR / "phase1_real_data_evaluation.json"
MARKDOWN_PATH = REPORT_DIR / "PHASE_1_REAL_DATA_EVALUATION.md"
TICKER_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials",
    "JNJ": "Healthcare", "XOM": "Energy",
}


def _metric(model, section, name):
    return model.get(section, {}).get(name)


def _format(value, suffix=""):
    return "n/a" if value is None else f"{float(value):.4f}{suffix}"


def _ticker_aggregate(payload):
    """Equal-weight ticker summary; descriptive, not a portfolio result."""
    model_names = sorted({
        name
        for item in payload["tickers"]
        for name in item.get("report", {}).get("walk_forward", {}).get("aggregate", {})
    })
    metric_names = (
        "classification.accuracy", "classification.balanced_accuracy",
        "classification.roc_auc", "classification.brier_score",
        "trading.cumulative_return_pct", "trading.max_drawdown_pct",
        "trading.annualized_sharpe",
    )
    output = {}
    for model_name in model_names:
        output[model_name] = {}
        for metric_name in metric_names:
            values = []
            for item in payload["tickers"]:
                entry = item.get("report", {}).get("walk_forward", {}).get("aggregate", {}).get(model_name, {}).get(metric_name)
                if entry and entry.get("mean") is not None:
                    values.append(float(entry["mean"]))
            if values:
                output[model_name][metric_name] = {
                    "equal_ticker_mean": round(float(np.mean(values)), 4),
                    "across_ticker_std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
                    "tickers": len(values),
                }
    return output


def _write_markdown(payload, markdown_path=MARKDOWN_PATH, title="Phase 1 real-data walk-forward evaluation"):
    lines = [
        f"# {title}",
        "",
        f"Run: `{payload['run_id']}`  ",
        f"Run started (UTC): `{payload['run_started_at_utc']}`  ",
        f"Run completed (UTC): `{payload['run_completed_at_utc']}`  ",
        f"Provider: `{payload['provider']}`  ",
        f"Status: **{payload['status']}**",
        "",
        "## Method and scope",
        "",
        "- Target: next session adjusted close greater than current session adjusted close.",
        "- Source: Yahoo Finance daily bars, `auto_adjust=True`, 3-year requested period.",
        "- Each successful ticker has expanding train/validation/test folds in its first 80% and a separate final chronological 20% holdout.",
        "- The final holdout is distinct from the walk-forward folds; no settings, weights, or thresholds were tuned on its results.",
        "- Fixed hybrid is retained at 40% LSTM, 40% Random Forest, 10% neutral historical sentiment, and 10% technical evidence.",
        "- XGBoost is included only if enabled in the run configuration.",
        "- Historical point-in-time news is unavailable, so sentiment is neutral for this evaluation.",
        "- The classifier trading figures are simplified long-or-cash close-to-close simulations, not the separate next-open technical backtest.",
        "- Raw data is not archived with this report; the SHA-256 identifies the evaluated OHLCV frame but does not make the provider's historical dataset immutable.",
        "",
        "## Configuration",
        "",
        f"- Features: `{', '.join(payload['configuration']['features'])}`.",
        f"- Random Forest: {payload['configuration']['rf_trees']} trees; max depth 7; minimum leaf size 4; balanced class weights; seed 42.",
        f"- LSTM: lookback {payload['configuration']['lookback']} rows; {payload['configuration']['lstm_units']} units; up to {payload['configuration']['epochs']} epochs; early stopping on validation data.",
        f"- Costs for simplified trading metrics: {payload['configuration']['transaction_cost_pct']}% fee and {payload['configuration']['slippage_pct']}% slippage on position changes.",
        "- Decision cutoff is after the current daily close; direction threshold is 0.50 for classifier metrics. The live hybrid's Buy/Sell thresholds are not applied to the evaluation classification table.",
        "- The final holdout model and fold models use fixed RF parameters. No model hyperparameter sweep, threshold search, calibration, adaptive weighting, or No Trade rule was run.",
        "",
        "## Data coverage and status",
        "",
        "| Ticker | Sector | Status | Raw rows | Labeled rows | Raw data dates | Labeled dates | SHA-256 | Folds | Final holdout |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in payload["tickers"]:
        report = item.get("report", {})
        data = report.get("data", {})
        split = report.get("split", {})
        acquisition = item.get("acquisition", {})
        if report.get("status") == "ok":
            rows = acquisition.get("raw_rows", "n/a")
            lines.append(
                f"| {item['symbol']} | {item.get('sector', 'unspecified')} | ok | {rows} | {data.get('rows_used', 'n/a')} | "
                f"{acquisition.get('first_date', 'n/a')}–{acquisition.get('last_date', 'n/a')} | "
                f"{data.get('first_date', 'n/a')}–{data.get('last_labeled_date', 'n/a')} | "
                f"`{data.get('sha256', 'n/a')}` | {split.get('walk_forward_folds', 'n/a')} | "
                f"{split.get('final_holdout_start', 'n/a')}–{split.get('final_holdout_end', 'n/a')} |"
            )
        else:
            lines.append(f"| {item['symbol']} | {item.get('sector', 'unspecified')} | {item.get('status', 'error')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend([
        "",
        "## Fold date boundaries",
        "",
        "| Ticker | Fold | Train end | Validation | Test | Train/validation/test rows |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ])
    for item in payload["tickers"]:
        report = item.get("report", {})
        for fold in report.get("walk_forward", {}).get("folds", []):
            lines.append(
                f"| {item['symbol']} | {fold['fold']} | {fold['train_start_date']}–{fold['train_end_date']} | "
                f"{fold['validation_start_date']}–{fold['validation_end_date']} | "
                f"{fold.get('test_start_date', 'n/a')}–{fold.get('test_end_date', 'n/a')} | "
                f"{fold['train_samples']}/{fold['validation_samples']}/{fold['test_samples']} |"
            )
    lines.extend(["", "## Package versions", ""])
    version_sets = {
        json.dumps(item.get("report", {}).get("versions", {}).get("python_packages", {}), sort_keys=True)
        for item in payload["tickers"] if item.get("report", {}).get("versions")
    }
    if version_sets:
        for version_set in sorted(version_sets):
            versions = json.loads(version_set)
            lines.append("| Package | Version |")
            lines.append("| --- | --- |")
            for package, version in versions.items():
                lines.append(f"| {package} | {version or 'not installed'} |")
            lines.append("")
    lines.extend(["", "## Walk-forward mean across folds, by ticker", ""])
    for item in payload["tickers"]:
        report = item.get("report", {})
        if report.get("status") != "ok":
            lines.append(f"### {item['symbol']} — unavailable")
            lines.append("")
            lines.append(f"Failure type: `{item.get('error_type', 'evaluation_failed')}`.")
            lines.append("")
            continue
        lines.extend([
            f"### {item['symbol']}",
            "",
            "| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        aggregate = report["walk_forward"]["aggregate"]
        for name in ("previous_direction", "random_forest", "xgboost", "lstm", "hybrid"):
            metrics = aggregate.get(name)
            if not metrics:
                continue
            def mean_sd(key):
                entry = metrics.get(key)
                return "n/a" if not entry else f"{entry['mean']:.4f} ± {entry['std']:.4f}"
            lines.append(
                f"| {name} | {mean_sd('classification.accuracy')} | "
                f"{mean_sd('classification.balanced_accuracy')} | {mean_sd('classification.roc_auc')} | "
                f"{mean_sd('classification.brier_score')} | {mean_sd('trading.cumulative_return_pct')}% |"
            )
        lines.append("")
    lines.extend([
        "## Equal-weight mean of ticker-level walk-forward means",
        "",
        "Each ticker is weighted equally after averaging its folds. This is descriptive and is not a combined portfolio.",
        "",
        "| Model | Accuracy mean +/- SD | Balanced accuracy mean +/- SD | ROC-AUC mean +/- SD | Brier mean +/- SD | Return mean +/- SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in ("previous_direction", "random_forest", "xgboost", "lstm", "hybrid"):
        metrics = payload.get("ticker_aggregate", {}).get(name, {})
        if not metrics:
            continue
        def ticker_mean_sd(key):
            entry = metrics.get(key)
            return "n/a" if not entry else f"{entry['equal_ticker_mean']:.4f} +/- {entry['across_ticker_std']:.4f}"
        lines.append(
            f"| {name} | {ticker_mean_sd('classification.accuracy')} | "
            f"{ticker_mean_sd('classification.balanced_accuracy')} | {ticker_mean_sd('classification.roc_auc')} | "
            f"{ticker_mean_sd('classification.brier_score')} | {ticker_mean_sd('trading.cumulative_return_pct')}% |"
        )
    lines.append("")
    lines.extend([
        "## Final chronological holdout, by ticker",
        "",
        "| Ticker | Model | Accuracy | Balanced accuracy | Precision up | Recall up | F1 up | ROC-AUC | Brier | Long-or-cash return | Max drawdown | Sharpe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["tickers"]:
        report = item.get("report", {})
        if report.get("status") != "ok":
            continue
        for name in ("previous_direction", "random_forest", "xgboost", "lstm", "hybrid", "buy_and_hold", "flat_cash"):
            model = report["models"].get(name)
            if not model:
                continue
            classification = model.get("classification", {})
            trading = model.get("trading", {})
            lines.append(
                f"| {item['symbol']} | {name} | {_format(classification.get('accuracy'))} | "
                f"{_format(classification.get('balanced_accuracy'))} | {_format(classification.get('precision_up'))} | "
                f"{_format(classification.get('recall_up'))} | {_format(classification.get('f1_up'))} | "
                f"{_format(classification.get('roc_auc'))} | {_format(classification.get('brier_score'))} | "
                f"{_format(trading.get('cumulative_return_pct'), '%')} | "
                f"{_format(trading.get('max_drawdown_pct'), '%')} | "
                f"{_format(trading.get('annualized_sharpe'))} |"
            )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Yahoo Finance is a provider-dependent source; values are adjusted daily bars and may be revised or unavailable. Run start/completion times and each data fingerprint are recorded; exact per-request provider response timestamps were not captured.",
        "- This is one 3-year historical sample per ticker. The five large-cap tickers (AAPL, MSFT, JPM, JNJ, XOM) span several sectors but are a small, current-company sample with selection and survivorship bias; they do not represent a broad universe or every market regime.",
        "- Yahoo Finance does not provide point-in-time historical news in this pipeline. FinBERT is neutral in every historical row; the live news contribution is not evaluated.",
        "- The LSTM uses fixed configured architecture/training settings and validation early stopping; there is no nested model search. The fixed hybrid weights are a baseline, not learned weights.",
        "- The final holdout is evaluated once in this report. Do not tune settings, weights, cutoffs, or claims against these final-holdout results.",
        "- The final holdout window was also scored in the superseded preliminary run that exposed the one-row fold. The fold splitter was corrected based on fold-size diagnostics, not holdout metrics; no model parameters, weights, or thresholds were changed. Treat the final holdout as exploratory rather than pristine confirmatory evidence.",
        "- Probability scores are raw model probabilities/proxies. Brier/reliability metrics assess observed quality; no calibration was fitted. Direction probability is not probability of profit.",
        "- The 0.50 classifier cutoff is fixed, not validation-selected. Reliability bins are descriptive only; fold standard deviations are not confidence intervals or significance tests.",
        "- Trading results are simplified close-to-close long-or-cash calculations. The configured 0.06% fee and 0.04% slippage are charged on position changes; there is no share/cash ledger or separate final liquidation charge. These are not the next-open execution backtest and omit taxes, market impact, liquidity constraints, and capital constraints.",
        "- TensorFlow ran on CPU on native Windows; GPU support is unavailable in this setup. Prediction-function retracing warnings increased runtime but training completed. Fixed seeds are set, but exact neural reproducibility across libraries/hardware is not guaranteed.",
        "- Per-fold aggregate outputs are saved, but per-date component predictions are not exported.",
        "- The target is one-session direction only. There is no separate 1W/1M/3M trained evaluation, no intraday data, and no fundamentals or point-in-time corporate/news features.",
        "- Synthetic fixture results are not included here and are not empirical market evidence.",
        "",
        "## Machine-readable details",
        "",
        f"The adjacent `{Path(markdown_path).with_suffix('.json').name}` contains per-fold date ranges, predictions' aggregate metrics, component statuses, run configuration, package versions, data hashes, and errors for failed ticker fetches/evaluations.",
        "",
    ])
    if payload.get("holdout_reused_from_prior_phase1_report"):
        limitations_index = lines.index("## Limitations")
        lines.insert(
            limitations_index + 2,
            "- This optional benchmark reused the previously evaluated final-holdout dates for the additional XGBoost model. No settings were tuned against those outcomes, but these holdout results are exploratory and are not a pristine confirmatory test.",
        )
    Path(markdown_path).write_text("\n".join(lines), encoding="utf-8")


def run(symbols, period, include_xgboost, output_json=JSON_PATH, output_markdown=None):
    config = main.TUNING_CONFIG.copy()
    config.update({
        "include_xgboost": include_xgboost,
        "data_source": "Yahoo Finance adjusted daily OHLCV",
    })
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    payload = {
        "run_id": run_id,
        "evaluation_name": "phase7_optional_xgboost_benchmark" if include_xgboost else "phase1_real_data_walk_forward",
        "holdout_reused_from_prior_phase1_report": bool(include_xgboost),
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance",
        "requested_period": period,
        "status": "pending",
        "configuration": {
            **{key: value for key, value in config.items()},
            "features": evaluation.FEATURE_COLUMNS,
            "target": "next close > current close",
            "hybrid_weights": evaluation.HYBRID_WEIGHTS,
            "hybrid_version": evaluation.HYBRID_VERSION,
        },
        "tickers": [],
    }
    output_json = Path(output_json)
    if output_markdown is None:
        output_markdown = MARKDOWN_PATH if output_json == JSON_PATH else output_json.with_suffix(".md")
    output_markdown = Path(output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        print(f"Fetching {symbol} ({period}) ...", flush=True)
        try:
            raw = main.download_history(symbol, period, force_refresh=True)
            acquisition = {
                "raw_rows": int(len(raw)),
                "first_date": raw.iloc[0]["date"].strftime("%Y-%m-%d"),
                "last_date": raw.iloc[-1]["date"].strftime("%Y-%m-%d"),
            }
            print(f"Evaluating {symbol}: {acquisition['raw_rows']} bars ...", flush=True)
            report = evaluation.evaluate_direction(symbol, raw, config, None, main.indicators)
            payload["tickers"].append({
                "symbol": symbol,
                "sector": TICKER_SECTORS.get(symbol, "unspecified"),
                "status": report.get("status", "unknown"),
                "acquisition": acquisition,
                "report": report,
            })
            print(
                f"Completed {symbol}: {report.get('status')} / "
                f"{report.get('split', {}).get('walk_forward_folds', 0)} folds",
                flush=True,
            )
        except Exception as exc:
            payload["tickers"].append({
                "symbol": symbol,
                "sector": TICKER_SECTORS.get(symbol, "unspecified"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"Failed {symbol}: {type(exc).__name__}: {exc}", flush=True)

    ok_count = sum(item.get("status") == "ok" for item in payload["tickers"])
    payload["status"] = "ok" if ok_count == len(symbols) else "partial" if ok_count else "failed"
    payload["run_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["ticker_aggregate"] = _ticker_aggregate(payload)
    output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(
        payload, output_markdown,
        title="Phase 7 optional XGBoost benchmark" if include_xgboost else "Phase 1 real-data walk-forward evaluation",
    )
    print(json.dumps({
        "status": payload["status"],
        "symbols": [{"symbol": item["symbol"], "status": item["status"]} for item in payload["tickers"]],
        "json_report": str(output_json),
        "markdown_report": str(output_markdown),
    }, indent=2), flush=True)


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="AAPL,MSFT,JPM,JNJ,XOM")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--output", default=str(JSON_PATH), help="JSON report path; Markdown is written beside it")
    parser.add_argument("--report-only", action="store_true", help="regenerate Markdown from saved JSON")
    args = parser.parse_args()
    if args.report_only:
        output_json = Path(args.output)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload.setdefault("evaluation_name", "phase7_optional_xgboost_benchmark" if payload.get("configuration", {}).get("include_xgboost") else "phase1_real_data_walk_forward")
        payload.setdefault("holdout_reused_from_prior_phase1_report", bool(payload.get("configuration", {}).get("include_xgboost")))
        output_markdown = MARKDOWN_PATH if output_json == JSON_PATH else output_json.with_suffix(".md")
        output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_markdown(payload, output_markdown,
                        title="Phase 7 optional XGBoost benchmark" if payload.get("configuration", {}).get("include_xgboost") else "Phase 1 real-data walk-forward evaluation")
        print(json.dumps({"status": "report_regenerated", "markdown_report": str(output_markdown)}))
        return
    symbols = list(dict.fromkeys(item.strip().upper() for item in args.symbols.split(",") if item.strip()))
    if not symbols or len(symbols) > 10:
        parser.error("provide 1 to 10 comma-separated ticker symbols")
    run(symbols, args.period, args.include_xgboost, output_json=args.output)


if __name__ == "__main__":
    main_cli()

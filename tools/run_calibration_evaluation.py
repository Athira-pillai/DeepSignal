"""Run Phase 5 calibrated-probability and abstention evaluation."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import main
from backend.hybrid_backtesting import run_hybrid_backtest


def _config():
    config = main.TUNING_CONFIG.copy()
    config["include_xgboost"] = False
    for env, key, cast, default in (
        ("ADAPTIVE_ENSEMBLE_MIN_WEIGHT", "adaptive_minimum_weight", float, 0.05),
        ("ADAPTIVE_ENSEMBLE_MAX_WEIGHT", "adaptive_maximum_weight", float, 0.80),
        ("ADAPTIVE_ENSEMBLE_TEMPERATURE", "adaptive_temperature", float, 8.0),
        ("ADAPTIVE_MIN_VALIDATION_SAMPLES", "adaptive_minimum_validation_samples", int, 20),
        ("ADAPTIVE_MIN_REGIME_SAMPLES", "adaptive_minimum_regime_samples", int, 20),
    ):
        try:
            config[key] = cast(os.getenv(env, str(default)))
        except ValueError:
            config[key] = default
    return config


def _markdown(report, path):
    evaluation = report["calibration_evaluation"]
    pooled = evaluation["pooled_test"]
    no_trade = evaluation["no_trade_backtest"]
    lines = [
        "# Phase 5 - Calibration and uncertainty-aware No Trade evaluation", "",
        f"**Ticker:** {report['symbol']}  ",
        f"**Window:** {report['data']['start']} to {report['data']['end']}  ",
        f"**OOS sessions:** {report['summary']['fixed_hybrid']['sessions']} over {len(report['folds'])} folds; final holdout excluded.", "",
        "## Untouched test-fold calibration metrics", "",
        f"**Calibrated event:** {evaluation.get('event', {}).get('meaning', 'Probability the next session closes above the current session close.')} Calibration estimates direction only, not trading profit.", "",
        "| Model score | Samples | Brier score | Reliability bins |",
        "|---|---:|---:|---:|",
    ]
    for key, title in (("raw_fixed", "Fixed raw"), ("calibrated_fixed", "Fixed calibrated"),
                       ("raw_adaptive", "Adaptive raw"), ("calibrated_adaptive", "Adaptive calibrated")):
        metric = pooled[key]
        lines.append(f"| {title} | {metric['samples']} | {metric['brier_score']} | {len(metric['reliability'])} |")
    lines.extend([
        "", "## No Trade backtest", "",
        f"No Trade decisions: {evaluation['no_trade_count']} of {len(report['signal_audit'])} audited OOS rows ({(1-evaluation['no_trade_coverage'])*100:.1f}% abstained).",
        f"Backtest return: {no_trade['total_return_pct']}%; max drawdown: {no_trade['max_drawdown_pct']}%; Sharpe: {no_trade['annualized_sharpe']}; trades: {no_trade['trade_count']}.",
        "No Trade retains an existing position and does not open a new position; the execution engine represents it as HOLD.", "",
        "## Fold selection audit", "",
        "| Fold | Calibration method | Validation fit rows | Selection Brier | No Trade margin | Dispersion limit | Test Brier (raw -> calibrated) |",
        "|---:|---|---:|---|---:|---:|---|",
    ])
    for fold in evaluation["folds"]:
        selection = fold["selection"]
        no_trade_thresholds = selection["no_trade_thresholds"]
        candidates = selection.get("candidate_statuses", {})
        candidate_text = ", ".join(
            f"{name}: {details['status']}" +
            (f" ({details['fallback_reason']})" if details.get("fallback_reason") else "")
            for name, details in candidates.items()
        ) or "none"
        brier_text = ", ".join(
            f"{name} {score:.4f}" for name, score in selection.get("candidate_brier", {}).items()
        ) or "unavailable"
        raw_brier = fold["test"]["raw_fixed"]["brier_score"]
        calibrated_brier = fold["test"]["calibrated_fixed"]["brier_score"]
        lines.append(
            f"| {fold['fold']} | {selection['method'] or 'unavailable'}; {candidate_text} | {selection['fit_samples']} | "
            f"{brier_text} | {no_trade_thresholds.get('neutral_margin')} | "
            f"{no_trade_thresholds.get('maximum_dispersion')} | {raw_brier:.4f} -> {calibrated_brier:.4f} |"
        )
    lines.extend([
        "", "## Method", "",
        "- Each fold generates component probabilities on a preceding chronological validation window. Adaptive validation probabilities use only earlier rows when choosing their weights.",
        "- Platt scaling and isotonic regression are candidates. Candidate calibrators fit on the first 60% of validation rows; validation Brier score on the later 40% selects a method; the selected calibrator is refit on the full past validation window before test inference.",
        "- No Trade neutral-band and maximum component-dispersion thresholds are selected using the later validation slice and frozen for the next test fold. Test labels are used only for evaluation.",
        "- Reliability uses fixed-width probability bins. The full fold-level audit, selected methods, cutoffs, raw/calibrated outputs, and data hashes are in the JSON file.",
        "- Uncertainty is the standard deviation among available LSTM, Random Forest, and technical component probabilities. It is a disagreement estimate, not a confidence interval or probability of profit.",
        "", "## Limitations", "",
        "This is one ticker/window with few folds. Isotonic regression needs a larger calibration prefix than many folds provide and will be reported unavailable when evidence is insufficient. Platt scaling and the validation-selected No Trade cutoffs can overfit small validation windows. The uncertainty proxy captures component disagreement only. Historical FinBERT news is neutral/unavailable; Yahoo daily data, transaction costs, slippage and next-open fills are simplified. Calibration metrics describe direction probability, never probability of profit. The live prediction endpoint explicitly reports calibration unavailable because it does not fit a point-in-time validation calibrator at request time.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--output", default=os.path.join(ROOT, "research", "real_data", "phase5_calibration_evaluation.json"))
    parser.add_argument("--report-only", action="store_true", help="regenerate Markdown from the saved JSON audit")
    args = parser.parse_args()
    if args.report_only:
        with open(args.output, encoding="utf-8") as handle:
            report = json.load(handle)
        report.setdefault("calibration_evaluation", {}).setdefault("event", {
            "name": "next_session_up_direction",
            "meaning": "Probability that the next session close is above the current session close.",
            "horizon_sessions": 1,
        })
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        _markdown(report, os.path.splitext(args.output)[0] + ".md")
        print(json.dumps({"status": "report_regenerated", "markdown": os.path.splitext(args.output)[0] + ".md"}))
        return
    symbol = main.clean_symbol(args.symbol)
    prices = main.download_history(symbol, args.period, force_refresh=True)
    report = run_hybrid_backtest(symbol, prices, _config(), main.indicators, selected_mode="fixed")
    if report.get("status") != "ok":
        raise RuntimeError(f"calibration evaluation failed: {report}")
    report["data"]["requested_period"] = args.period
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown_path = os.path.splitext(args.output)[0] + ".md"
    _markdown(report, markdown_path)
    print(json.dumps({
        "status": report["status"], "symbol": symbol,
        "json": args.output, "markdown": markdown_path,
        "calibration": report["calibration_evaluation"]["pooled_test"],
        "no_trade": report["calibration_evaluation"]["no_trade_backtest"],
        "folds": report["calibration_evaluation"]["folds"],
    }, indent=2))


if __name__ == "__main__":
    main_cli()

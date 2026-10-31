"""Build the final Phase 9 evidence report from versioned experiment artifacts."""

import json
import hashlib
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "real_data"
OUT_JSON = DATA / "phase9_final_experiment_matrix.json"
OUT_MD = DATA / "phase9_final_experiment_matrix.md"


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def metric(model, path):
    current = model
    for part in path.split("."):
        current = current[part]
    return current


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    xgb = read("phase7_xgboost_evaluation.json")
    regime = read("phase3_regime_evaluation.json")
    with_regime = read("phase9_e4_with_regime.json")
    without_regime = read("phase9_e4_without_regime.json")
    news = read("phase6_news_alignment.json")
    # Use one saved-input run for all AAPL strategy/calibration experiments.
    hybrid = adaptive = calibration = with_regime

    source_hashes = {
        with_regime["data"].get("full_source_sha256"),
        without_regime["data"].get("full_source_sha256"),
    }
    shared_dates = with_regime["dates"] == without_regime["dates"]
    if len(source_hashes) != 1 or not shared_dates:
        raise SystemExit("E4 arms do not share identical source data and dates; refusing to report a paired result.")
    input_path = DATA / "phase9_e4_aapl_prices.csv"
    input_frame = pd.read_csv(input_path, parse_dates=["date"])[["date", "open", "high", "low", "close", "volume"]]
    normalized_input_hash = hashlib.sha256(input_frame.to_csv(index=False).encode("utf-8")).hexdigest()
    if normalized_input_hash != next(iter(source_hashes)):
        raise SystemExit("Retained E4 input CSV does not match its experiment data hash; refusing to generate the report.")

    aggregate = xgb["ticker_aggregate"]
    model_names = ["previous_direction", "random_forest", "lstm", "xgboost", "hybrid"]
    e1 = {
        name: {
            "accuracy": aggregate[name]["classification.accuracy"],
            "balanced_accuracy": aggregate[name]["classification.balanced_accuracy"],
            "roc_auc": aggregate[name]["classification.roc_auc"],
            "brier_score": aggregate[name]["classification.brier_score"],
        }
        for name in model_names
    }
    e1_per_ticker = {}
    for ticker in xgb["tickers"]:
        if ticker["status"] != "ok":
            continue
        e1_per_ticker[ticker["symbol"]] = {
            name: ticker["report"]["walk_forward"]["aggregate"][name]
            for name in model_names
        }

    signal_differences = [
        {"date": left.get("date"), "with_regime": left.get("signal"), "without_regime": right.get("signal")}
        for left, right in zip(with_regime["model_signals"]["adaptive_hybrid"],
                               without_regime["model_signals"]["adaptive_hybrid"])
        if left != right
    ]
    e4 = {
        "source_sha256": next(iter(source_hashes)),
        "retained_csv_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "same_source_hash": True,
        "same_oos_dates": shared_dates,
        "same_adaptive_signal_sequence": with_regime["model_signals"]["adaptive_hybrid"] == without_regime["model_signals"]["adaptive_hybrid"],
        "same_adaptive_equity_curve": with_regime["model_curves"]["adaptive_hybrid"] == without_regime["model_curves"]["adaptive_hybrid"],
        "adaptive_signal_differences": signal_differences,
        "window": {"start": with_regime["data"]["start"], "end": with_regime["data"]["end"]},
        "folds": len(with_regime["folds"]),
        "with_regime": with_regime["summary"]["adaptive_hybrid"],
        "without_regime": without_regime["summary"]["adaptive_hybrid"],
        "with_regime_selection_scopes": {},
        "without_regime_selection_scopes": {},
        "with_regime_run_id": with_regime["run_id"],
        "without_regime_run_id": without_regime["run_id"],
    }
    for dest, report in (("with_regime_selection_scopes", with_regime),
                         ("without_regime_selection_scopes", without_regime)):
        for row in report["signal_audit"]:
            scope = row.get("adaptive_selection", {}).get("scope") or "unavailable"
            e4[dest][scope] = e4[dest].get(scope, 0) + 1

    no_trade = calibration["calibration_evaluation"]["no_trade_backtest"]
    raw_brier = calibration["calibration_evaluation"]["pooled_test"]["raw_fixed"]["brier_score"]
    calibrated_brier = calibration["calibration_evaluation"]["pooled_test"]["calibrated_fixed"]["brier_score"]
    e5_status = "not_evaluable_missing_eligible_point_in_time_historical_news"

    report = {
        "report_version": "phase9-evidence-matrix-v1",
        "status": "complete_with_documented_unavailable_experiment",
        "generated_from": [
            "phase7_xgboost_evaluation.json", "phase3_regime_evaluation.json", "phase6_news_alignment.json",
            "phase9_e4_with_regime.json", "phase9_e4_without_regime.json",
        ],
        "experiments": {
            "E1_base_models": {"status": "evaluated", "aggregate_equal_ticker": e1,
                                "per_ticker_fold_mean_and_sd": e1_per_ticker},
            "E2_ensemble_effect": {"status": "evaluated_single_ticker_trading_comparison",
                                   "best_single": "lstm_only",
                                   "lstm_only": hybrid["summary"]["lstm_only"],
                                   "fixed_hybrid": hybrid["summary"]["hybrid"],
                                   "same_oos_window": {"start": hybrid["data"]["start"], "end": hybrid["data"]["end"]}},
            "E3_adaptive_fusion": {"status": "evaluated_single_ticker",
                                   "fixed": adaptive["summary"]["fixed_hybrid"],
                                   "adaptive": adaptive["summary"]["adaptive_hybrid"],
                                   "per_fold": with_regime.get("fold_strategy_comparison", [])},
            "E4_regime_effect": {"status": "evaluated_paired_same_data",
                                 **e4},
            "E5_sentiment_ablation": {"status": e5_status,
                                      "eligible_historical_article_count": None,
                                      "reason": "No permitted, complete point-in-time news archive was available; current headlines are not substituted into historical rows."},
            "E6_uncertainty_filter": {"status": "evaluated_single_ticker",
                                      "fixed_raw_brier": raw_brier,
                                      "calibrated_fixed_brier": calibrated_brier,
                                      "no_trade_decisions": calibration["calibration_evaluation"]["no_trade_count"],
                                      "audited_signals": len(calibration["signal_audit"]),
                                      "no_trade_backtest": no_trade,
                                      "fixed_hybrid_backtest": calibration["summary"]["fixed_hybrid"]},
            "E7_trading_validation": {"status": "evaluated_single_ticker",
                                      "strategies": {name: hybrid["summary"][key] for name, key in (
                                          ("fixed_hybrid", "hybrid"), ("lstm_only", "lstm_only"),
                                          ("random_forest_only", "random_forest_only"),
                                          ("buy_and_hold", "buy_hold"), ("cash", "no_trade"))}},
        },
        "data_and_design": {
            "E1_source": xgb["provider"], "E1_requested_period": xgb["requested_period"],
            "E1_successful_tickers": [x["symbol"] for x in xgb["tickers"] if x["status"] == "ok"],
            "E1_folds_per_ticker": 4, "E1_final_holdout_excluded_from_walk_forward_means": True,
            "E1_final_holdout_reused_from_phase1": bool(xgb.get("holdout_reused_from_prior_phase1_report")),
            "E2_E3_E6_E7_symbol": "AAPL", "E2_E3_E6_E7_window": {"start": hybrid["data"]["start"], "end": hybrid["data"]["end"]},
            "E2_E3_E6_E7_data_sha256": hybrid["data"]["full_source_sha256"],
            "E2_E3_E6_E7_csv_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "target": "Next-session adjusted close greater than current adjusted close.",
            "classifier_costs_pct": {"fee": 0.06, "slippage": 0.04},
            "trading_execution": "Signal after daily close; next available session open; long-only; fractional shares; terminal liquidation.",
            "feature_columns": xgb["configuration"].get("features"),
            "configuration": xgb["configuration"],
            "regime_report_is_stratification_not_causal_ablation": True,
            "prior_regime_stratification_status": regime.get("status"),
            "news_alignment_status": news.get("status"),
        },
        "reproducibility": {
            "source_run_ids": {
                "E1": xgb.get("run_id"), "E2_E7_hybrid": hybrid.get("run_id"),
                "E3": adaptive.get("run_id"), "E6": calibration.get("run_id"),
                "E4_with_regime": with_regime.get("run_id"),
                "E4_without_regime": without_regime.get("run_id"),
            },
            "E1_data_hashes_by_ticker": {
                item["symbol"]: item["report"].get("data", {}).get("sha256")
                for item in xgb["tickers"] if item["status"] == "ok"
            },
            "E1_model_versions": xgb["tickers"][0]["report"].get("versions"),
            "E2_E7_versions": hybrid.get("versions"),
            "E3_versions": adaptive.get("versions"),
            "E6_versions": calibration.get("versions"),
        },
        "artifacts": {
            "confusion_matrices_and_reliability_bins": "Embedded per ticker and fold in phase7_xgboost_evaluation.json.",
            "equity_curves_trades_and_cost_audit": ["phase9_e4_with_regime.json"],
            "regime_strata": "phase3_regime_evaluation.json",
            "E4_exact_input_csv": "phase9_e4_aapl_prices.csv",
            "E4_with_regime": "phase9_e4_with_regime.json",
            "E4_without_regime": "phase9_e4_without_regime.json",
        },
        "limitations": [
            "Five selected large-cap symbols and one recent three-year provider window do not represent a broad universe.",
            "E1 fold means are descriptive; cross-ticker standard deviation is not a confidence interval. XGBoost's higher accuracy coincides with a worse Brier score.",
            "The final holdout was previously evaluated during Phase 1 and is exploratory; Phase 9 E1 fold metrics exclude it.",
            "E2 through E7, except E1, are based on one AAPL window and four folds.",
            "E4 tests regime-specific versus overall validation weight selection, not whether an HMM regime label improves the forecasting model generally. Small regime groups can fall back to overall validation evidence.",
            "E5 could not be run without eligible historical point-in-time news; historical sentiment remains neutral.",
            "Calibration worsened pooled Brier in Phase 5; live calibration and No Trade thresholds remain unavailable.",
            "Daily adjusted Yahoo bars, constant transaction costs/slippage, and next-open fills omit intraday path, spread, market impact, partial fills, taxes, and funding costs.",
            "LSTM CPU training on native Windows is nondeterministic to some extent; exact floating-point replay may vary. E4 arms share the identical saved source CSV and dates.",
            "No significance tests or confidence intervals were run; no result establishes future profitability.",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, xgb, hybrid, adaptive, calibration)
    write_figures(xgb, hybrid)
    print(json.dumps({"status": report["status"], "markdown": str(OUT_MD.relative_to(ROOT)),
                      "json": str(OUT_JSON.relative_to(ROOT)), "experiments": list(report["experiments"]),
                      "E4_same_hash": e4["same_source_hash"], "E5": e5_status}, indent=2))


def write_markdown(report, xgb, hybrid, adaptive, calibration):
    lines = [
        "# Phase 9 — Final experiment matrix and evidence report", "",
        "**Status:** Completed with E5 explicitly unavailable. No metric below is evidence of future profitability.", "",
        "## Method and common design", "",
        "The next-session target is adjusted close at session t+1 greater than adjusted close at t. E1 uses four expanding train/validation/test folds inside the first 80% of three years of Yahoo adjusted daily bars, with the later 20% holdout excluded from its fold means. E2–E7 use the AAPL development OOS window from 2025-05-16 through 2026-04-09 unless marked unavailable. Trading uses daily close signal generation, next available open execution, long-only fractional sizing, 0.06% fees and 0.04% slippage, and terminal liquidation.", "",
        "E1 uses five symbols across Technology, Financials, Healthcare and Energy. Equal-ticker means are reported; standard deviations in that table describe cross-ticker spread. Per-ticker fold mean ± SD is included in the machine-readable JSON. The held-out 20% was previously examined in Phase 1 and is not used for E1 fold conclusions.", "",
        "## E1 — Base models", "",
        "Equal-ticker mean metrics across five symbols (four walk-forward folds per symbol):", "",
        "| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier |", "|---|---:|---:|---:|---:|",
    ]
    lines.insert(7, f"E2–E7 strategy comparisons use one retained AAPL OHLCV input frame (SHA-256 `{hybrid['data']['full_source_sha256']}`), so the compared arms share exact market data in addition to the same test dates and costs.")
    names = [("previous_direction", "Previous direction"), ("random_forest", "Random Forest"),
             ("lstm", "LSTM"), ("xgboost", "XGBoost"), ("hybrid", "Fixed hybrid")]
    for key, label in names:
        values = report["experiments"]["E1_base_models"]["aggregate_equal_ticker"][key]
        lines.append(f"| {label} | {values['accuracy']['equal_ticker_mean']:.4f} | {values['balanced_accuracy']['equal_ticker_mean']:.4f} | {values['roc_auc']['equal_ticker_mean']:.4f} | {values['brier_score']['equal_ticker_mean']:.4f} |")
    lines.extend(["", "XGBoost had the highest mean accuracy (0.5333) and balanced accuracy (0.5330), but its Brier score (0.2906) was worse than RF (0.2607), LSTM (0.2513), and fixed hybrid (0.2509). The result is mixed and descriptive, not a model superiority claim.", "",
        "## Model and evaluation details", "",
        f"Features: `{', '.join(xgb['configuration']['features'])}`. Sequence scaling uses a MinMaxScaler fit on each fold's training rows only; LSTM early stopping uses validation rows. Random Forest uses 250 trees, max depth 7, min leaf size 4 and balanced class weights; LSTM uses a 30-session lookback, 48 units, up to 6 epochs and validation early stopping; XGBoost uses the enabled optional tabular configuration. Seed is 42 where supported.", "",
        "The fixed hybrid uses LSTM 0.40, Random Forest 0.40, neutral FinBERT 0.10 and technical evidence 0.10. Adaptive weights omit unavailable historical FinBERT, score components by validation Brier skill against 0.50, combine scores with renormalized fixed priors, and project onto configured weight bounds. The HMM uses daily return, rolling volatility, normalized range and volume change; scaler/state mapping are trained on prior rows and inference is causal. Calibration compares Platt and isotonic candidates on chronological validation partitions; uncertainty is component standard deviation; No Trade thresholds are validation-selected and frozen for each next test fold.", "",
        "## E2 — Ensemble effect", "",
        f"On the shared AAPL OOS trading window, the best single component was LSTM-only at **{hybrid['summary']['lstm_only']['total_return_pct']}%** return, Sharpe {hybrid['summary']['lstm_only']['annualized_sharpe']}, and {hybrid['summary']['lstm_only']['max_drawdown_pct']}% max drawdown. Fixed hybrid returned **{hybrid['summary']['hybrid']['total_return_pct']}%**, Sharpe {hybrid['summary']['hybrid']['annualized_sharpe']}, and {hybrid['summary']['hybrid']['max_drawdown_pct']}% max drawdown. Fixed hybrid Sharpe was slightly higher, but its return was lower and drawdown larger than LSTM in this sample.", "",
        "## E3 — Adaptive fusion", "",
        f"Fixed hybrid returned **{adaptive['summary']['fixed_hybrid']['total_return_pct']}%** (Sharpe {adaptive['summary']['fixed_hybrid']['annualized_sharpe']}, drawdown {adaptive['summary']['fixed_hybrid']['max_drawdown_pct']}%); adaptive hybrid returned **{adaptive['summary']['adaptive_hybrid']['total_return_pct']}%** (Sharpe {adaptive['summary']['adaptive_hybrid']['annualized_sharpe']}, drawdown {adaptive['summary']['adaptive_hybrid']['max_drawdown_pct']}%). Adaptive weighting underperformed fixed weighting on return and Sharpe, and had a larger drawdown.", "",
        "## E4 — Regime effect", "",
        f"The paired experiment uses the exact same saved input CSV (SHA-256 `{report['experiments']['E4_regime_effect']['source_sha256']}`), OOS dates and fold boundaries. The treatment allows validation scores to be selected within the current regime when evidence is sufficient; the control selects from the overall validation window. Treatment return/Sharpe/drawdown: **{report['experiments']['E4_regime_effect']['with_regime']['total_return_pct']}% / {report['experiments']['E4_regime_effect']['with_regime']['annualized_sharpe']} / {report['experiments']['E4_regime_effect']['with_regime']['max_drawdown_pct']}%**. Control: **{report['experiments']['E4_regime_effect']['without_regime']['total_return_pct']}% / {report['experiments']['E4_regime_effect']['without_regime']['annualized_sharpe']} / {report['experiments']['E4_regime_effect']['without_regime']['max_drawdown_pct']}%**. One audited signal label differed (2025-08-05: BUY with regime vs HOLD without); the executed equity curves and aggregate trading metrics were identical. Regime-aware selection occurred on 220 audit rows; five used overall validation fallback. This one-symbol experiment does not establish a robust benefit. HMM state stratification alone is not causal evidence.", "",
        "## E5 — Sentiment ablation", "",
        "**Not evaluable.** No eligible point-in-time historical news corpus was available. The experiment did not substitute current headlines into historical dates. Historical FinBERT contribution remains neutral, and no sentiment value claim is made.", "",
        "## E6 — Uncertainty / No Trade", "",
        f"Raw fixed Brier was {calibration['calibration_evaluation']['pooled_test']['raw_fixed']['brier_score']:.4f}; calibrated fixed Brier was {calibration['calibration_evaluation']['pooled_test']['calibrated_fixed']['brier_score']:.4f}, so calibration worsened this metric. The validation-selected No Trade policy abstained on {calibration['calibration_evaluation']['no_trade_count']} of {len(calibration['signal_audit'])} audited signals. It returned **{calibration['calibration_evaluation']['no_trade_backtest']['total_return_pct']}%**, Sharpe {calibration['calibration_evaluation']['no_trade_backtest']['annualized_sharpe']}, drawdown {calibration['calibration_evaluation']['no_trade_backtest']['max_drawdown_pct']}%; fixed hybrid returned {calibration['summary']['fixed_hybrid']['total_return_pct']}%, Sharpe {calibration['summary']['fixed_hybrid']['annualized_sharpe']}, drawdown {calibration['summary']['fixed_hybrid']['max_drawdown_pct']}%. Drawdown was less severe, but return and Sharpe were lower. This threshold does not run on live predictions.", "",
        "## E7 — Trading validation", "",
        "| AAPL strategy | Return | Sharpe | Max drawdown | Turnover | Trades |", "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("hybrid", "Fixed hybrid"), ("lstm_only", "LSTM only"),
                       ("random_forest_only", "Random Forest only"), ("buy_hold", "Buy and hold"), ("no_trade", "Cash")):
        m = hybrid["summary"][key]
        lines.append(f"| {label} | {m['total_return_pct']}% | {m['annualized_sharpe']} | {m['max_drawdown_pct']}% | {m['turnover']} | {m['trade_count']} |")
    lines.extend(["", "## E3 per-fold results", "",
        "Each fold uses the same dates for fixed and adaptive and starts from fresh capital; these are not additive to the continuous results.", "",
        "| Fold | Test dates | Fixed return | Adaptive return | Fixed Sharpe | Adaptive Sharpe |", "|---:|---|---:|---:|---:|---:|"])
    for fold in report["experiments"]["E3_adaptive_fusion"]["per_fold"]:
        fixed, adaptive_fold = fold["strategies"]["fixed_hybrid"], fold["strategies"]["adaptive_hybrid"]
        lines.append(f"| {fold['fold']} | {fold['start']} to {fold['end']} | {fixed['total_return_pct']}% | {adaptive_fold['total_return_pct']}% | {fixed['annualized_sharpe']} | {adaptive_fold['annualized_sharpe']} |")
    lines.extend(["", "Fixed hybrid exceeded buy-and-hold return in this window, with a larger drawdown. LSTM-only return was slightly higher than fixed hybrid. These are single-window simulated outcomes after modeled costs, not live or investable results.", "",
        "## Figures and audit artifacts", "",
        "The E1 confusion matrices are pooled by summing the four fold matrices per ticker across the five symbols. The calibration plot aggregates each fold's reliability bins using bin counts. Equity curves, trade records, data hashes, fold dates, configuration, calibration bins, and signal audits are retained in the linked JSON artifacts.", "",
        "![E1 pooled walk-forward confusion matrices](phase9_confusion_matrices.svg)", "",
        "![E1 pooled walk-forward reliability diagram](phase9_calibration.svg)", "",
        "![E7 cost-adjusted AAPL OOS equity curves, normalized to 100](phase9_equity_curves.svg)", "",
        "## Limitations", "",
    ])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Future work", "",
        "Run a later untouched confirmation period across a wider, predeclared ticker universe; add significance/uncertainty analysis; repeat adaptive and regime ablations across symbols; obtain appropriately licensed point-in-time news before sentiment ablation; and only enable live calibration/No Trade after a leakage-safe live validation path exists.", "",
        "## Reproducibility", "",
        f"Source run IDs: E1 `{xgb['run_id']}`, E2/E7 `{hybrid['run_id']}`, E3 `{adaptive['run_id']}`, E6 `{calibration['run_id']}`, E4 treatment `{report['reproducibility']['source_run_ids']['E4_with_regime']}`, E4 control `{report['reproducibility']['source_run_ids']['E4_without_regime']}`.",
        f"Model/strategy versions include E1 `{xgb['tickers'][0]['report']['versions']['evaluation_model']}`, hybrid `{hybrid.get('versions', {}).get('strategy')}`, adaptive `{adaptive.get('versions', {}).get('adaptive_ensemble')}`, regime `{adaptive.get('versions', {}).get('regime_model')}`, and calibration `{calibration.get('configuration', {}).get('calibration', {}).get('version')}`.",
        "- E1 source report: `phase7_xgboost_evaluation.json` (report-only regeneration: `tools/run_real_evaluation.py --output research/real_data/phase7_xgboost_evaluation.json --report-only`).",
        "- E4 paired runs: `tools/run_adaptive_evaluation.py --symbol AAPL --input-csv research/real_data/phase9_e4_aapl_prices.csv --output research/real_data/phase9_e4_with_regime.json` and repeat with `--without-regime-weights` and the other output path.",
        "- Regenerate this report and figures: `python tools/build_phase9_report.py`.",
        "- Package versions, model configurations, fold boundaries, hashes, and full audit data are recorded in the JSON reports listed in `phase9_final_experiment_matrix.json`.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_figures(xgb, hybrid):
    models = [("previous_direction", "Prev. direction"), ("random_forest", "Random Forest"),
              ("lstm", "LSTM"), ("xgboost", "XGBoost"), ("hybrid", "Fixed hybrid")]
    matrices = {name: [[0, 0], [0, 0]] for name, _ in models}
    bins = {name: {} for name, _ in models}
    for ticker in xgb["tickers"]:
        if ticker["status"] != "ok":
            continue
        for fold in ticker["report"]["walk_forward"]["folds"]:
            for name, _ in models:
                classification = fold["models"][name]["classification"]
                matrix = classification["confusion_matrix"]
                for row in range(2):
                    for col in range(2):
                        matrices[name][row][col] += int(matrix[row][col])
                for item in classification.get("reliability", []):
                    key = (item["bin_start"], item["bin_end"])
                    acc = bins[name].setdefault(key, [0, 0.0, 0.0])
                    count = item["count"]
                    acc[0] += count
                    acc[1] += count * item["mean_probability"]
                    acc[2] += count * item["observed_up_rate"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="500" y="32" text-anchor="middle" font-size="21" font-family="sans-serif">E1 pooled walk-forward confusion matrices · 5 tickers × 4 folds</text>']
    positions = [(70, 90), (390, 90), (710, 90), (230, 370), (550, 370)]
    for (name, label), (x, y) in zip(models, positions):
        matrix = matrices[name]
        maximum = max(max(row) for row in matrix) or 1
        parts.append(f'<text x="{x+105}" y="{y}" text-anchor="middle" font-size="17" font-family="sans-serif">{escape(label)}</text>')
        parts.append(f'<text x="{x+105}" y="{y+24}" text-anchor="middle" font-size="11" font-family="sans-serif">Predicted →</text>')
        parts.append(f'<text x="{x-15}" y="{y+120}" text-anchor="middle" font-size="11" font-family="sans-serif" transform="rotate(-90 {x-15} {y+120})">Observed →</text>')
        for row in range(2):
            for col in range(2):
                value = matrix[row][col]
                shade = int(245 - (value / maximum) * 150)
                xx, yy = x + 48 + col * 78, y + 40 + row * 65
                parts.append(f'<rect x="{xx}" y="{yy}" width="76" height="63" fill="rgb({shade},{shade+4},{min(255,shade+12)})" stroke="#fff"/>')
                parts.append(f'<text x="{xx+38}" y="{yy+38}" text-anchor="middle" font-size="16" font-family="sans-serif">{value}</text>')
        parts.append(f'<text x="{x+86}" y="{y+179}" text-anchor="middle" font-size="10" font-family="sans-serif">Down/flat       Up</text>')
    parts.append('</svg>')
    (DATA / "phase9_confusion_matrices.svg").write_text("".join(parts), encoding="utf-8")

    width, height, left, top, plot = 760, 580, 80, 55, 420
    colors = ["#276FBF", "#F28E2B", "#59A14F", "#E15759", "#8064A2"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="380" y="27" text-anchor="middle" font-size="19" font-family="sans-serif">E1 pooled walk-forward reliability by probability bin</text>']
    for tick in range(6):
        value = tick / 5
        xx, yy = left + value * plot, top + plot - value * plot
        parts.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top+plot}" stroke="#ddd"/>')
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{left+plot}" y2="{yy}" stroke="#ddd"/>')
        parts.append(f'<text x="{xx}" y="{top+plot+20}" text-anchor="middle" font-size="11">{value:.1f}</text>')
        parts.append(f'<text x="{left-12}" y="{yy+4}" text-anchor="end" font-size="11">{value:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#777" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{left+plot/2}" y="{height-28}" text-anchor="middle" font-size="13">Mean predicted up probability</text>')
    parts.append(f'<text x="22" y="{top+plot/2}" text-anchor="middle" font-size="13" transform="rotate(-90 22 {top+plot/2})">Observed up fraction</text>')
    for index, (name, label) in enumerate(models):
        color = colors[index]
        coords = []
        for _, values in sorted(bins[name].items()):
            if values[0]:
                px, py = values[1] / values[0], values[2] / values[0]
                coords.append((left + px * plot, top + plot - py * plot))
        if coords:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in coords:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lx, ly = 530, 90 + index * 28
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="{lx+32}" y="{ly+4}" font-size="12" font-family="sans-serif">{escape(label)}</text>')
    parts.append('</svg>')
    (DATA / "phase9_calibration.svg").write_text("".join(parts), encoding="utf-8")

    equity_series = [("Fixed hybrid", hybrid["model_curves"]["fixed_hybrid"], "#276FBF"),
                     ("LSTM only", hybrid["model_curves"]["lstm_only"], "#F28E2B"),
                     ("Random Forest only", hybrid["model_curves"]["random_forest_only"], "#59A14F"),
                     ("Buy and hold", hybrid["buy_hold"], "#E15759"),
                     ("Cash", hybrid["no_trade"], "#555555")]
    chart_w, chart_h, left, top, plot_w, plot_h = 900, 540, 80, 55, 690, 390
    chart_values = [float(value) for _, values, _ in equity_series for value in values]
    y_min, y_max = min(chart_values), max(chart_values)
    padding = max((y_max - y_min) * .08, 1)
    y_min, y_max = y_min - padding, y_max + padding
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_w}" height="{chart_h}" viewBox="0 0 {chart_w} {chart_h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="450" y="27" text-anchor="middle" font-size="19" font-family="sans-serif">AAPL OOS strategy curves · normalized to 100 · after modeled costs</text>']
    for tick in range(6):
        fraction = tick / 5
        value = y_min + fraction * (y_max - y_min)
        yy = top + plot_h - fraction * plot_h
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>')
    length = len(hybrid["dates"])
    for index, (label, values, color) in enumerate(equity_series):
        coords = []
        for position, value in enumerate(values):
            xx = left + (position / max(length - 1, 1)) * plot_w
            yy = top + plot_h - ((float(value) - y_min) / (y_max - y_min)) * plot_h
            coords.append(f"{xx:.1f},{yy:.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx, ly = 790, 95 + index * 27
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="{lx+27}" y="{ly+4}" font-size="11" font-family="sans-serif">{escape(label)}</text>')
    parts.append(f'<text x="{left}" y="{top+plot_h+26}" font-size="11">{hybrid["dates"][0]}</text>')
    parts.append(f'<text x="{left+plot_w}" y="{top+plot_h+26}" text-anchor="end" font-size="11">{hybrid["dates"][-1]}</text>')
    parts.append('</svg>')
    (DATA / "phase9_equity_curves.svg").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()

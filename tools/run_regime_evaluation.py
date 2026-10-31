"""Run a real-data walk-forward hybrid comparison stratified by HMM regime."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import main
from backend.regime_detection import evaluate_regime_walk_forward


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="AAPL")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--output", default=os.path.join(ROOT, "research", "real_data", "phase3_regime_evaluation.json"))
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown from an existing JSON report without fetching data or fitting models.")
    args = parser.parse_args()
    if args.report_only:
        with open(args.output, encoding="utf-8") as handle:
            output = json.load(handle)
        reports = output.get("reports", [])
    else:
        symbols = list(dict.fromkeys(main.clean_symbol(item) for item in args.symbols.split(",") if item.strip()))
        reports = []
        config = main.TUNING_CONFIG.copy()
        config["include_xgboost"] = False
        for symbol in symbols:
            try:
                prices = main.download_history(symbol, args.period, force_refresh=True)
                report = evaluate_regime_walk_forward(symbol, prices, config, main.indicators)
                report.setdefault("data", {})["requested_period"] = args.period
            except Exception as exc:
                report = {"status": "unavailable", "symbol": symbol, "error": type(exc).__name__}
            reports.append(report)
        output = {"status": "ok" if all(item.get("status") == "ok" for item in reports) else "partial",
                  "reports": reports}
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
    lines = [
        "# Phase 3 — Walk-forward HMM regime evaluation", "",
        "This report describes fixed-hybrid predictions pooled together and stratified by causal HMM regime. It does not test whether adding regime as a model feature improves predictions.", "",
        "## Method", "",
        "- Target: next-session adjusted close above the current adjusted close.",
        "- HMM inputs: daily return, 20-session rolling volatility, normalized high-low range, and volume change; calculated through each daily close.",
        "- A StandardScaler and diagonal Gaussian HMM are fit on each fold's training rows only. Two- and three-state candidates are selected with later validation mean log likelihood and a parameter penalty.",
        "- State descriptions are assigned from training-window average return and volatility; the mapping is fixed within each test fold. State IDs are fold-local.",
        "- Test posteriors use prefixes ending at each test date. The final chronological holdout is excluded.",
        "- HMM library: pomegranate 1.1.2; predictions use the same fixed-hybrid model outputs as the pooled baseline.", "",
    ]
    for report in reports:
        lines.extend([f"## {report.get('symbol', 'Unknown')} — {report.get('status')}", ""])
        if report.get("status") not in {"ok", "partial"} or "without_regime" not in report:
            lines.extend([f"Unavailable: {report.get('fallback_reason') or report.get('error')}", ""])
            continue
        metrics = report["without_regime"]["classification"]
        lines.extend([
            f"Pooled OOS rows: {report['without_regime']['samples']}; folds: {report['split']['walk_forward_folds']}; final holdout excluded: {report['split']['final_holdout_excluded']}.",
            f"Without regime stratification — accuracy {metrics['accuracy']}, balanced accuracy {metrics['balanced_accuracy']}, ROC-AUC {metrics['roc_auc']}, Brier {metrics['brier_score']}.",
            "", "| Inferred regime | Rows | Accuracy | Balanced accuracy | ROC-AUC | Brier |", "|---|---:|---:|---:|---:|---:|"])
        for label, group in report["with_regime_context"].items():
            m = group["classification"]
            lines.append(f"| {label} | {group['samples']} | {m['accuracy']} | {m['balanced_accuracy']} | {m['roc_auc']} | {m['brier_score']} |")
        lines.extend(["", f"Price data: {report['data']['first_date']} to {report['data']['last_date']}; SHA-256 `{report['data']['sha256']}`.", ""])
        lines.extend(["| Fold | Selected states | Test dates | Test regime counts |", "|---:|---:|---|---|"])
        for fold in report["folds"]:
            lines.append(f"| {fold['fold']} | {fold.get('selected_states', 'unavailable')} | {fold.get('test_start', '—')} to {fold.get('test_end', '—')} | `{json.dumps(fold.get('test_regime_counts', {}), sort_keys=True)}` |")
        lines.extend(["", "Full candidate scores, training/validation/test windows, per-fold metrics and state mappings are in the JSON report.", ""])
    lines.extend([
        "## Limitations", "",
        "- This is one ticker and one provider-dependent three-year sample. Regimes are latent model clusters, not known market regimes or ground truth.",
        "- The per-regime table is descriptive stratification of unchanged fixed-hybrid predictions. It is not a causal ablation, and it does not establish predictive benefit from HMM context.",
        "- State count and state descriptions can change by fold. Small groups can make the per-regime metrics unstable.",
        "- State probabilities are model state posteriors, not calibrated direction probabilities or probabilities of profit.",
        "- Yahoo adjusted daily bars are provider-dependent; no intraday path, spread, market impact, taxes or funding are modeled in this classification comparison.",
        "- This run excludes the final holdout and is exploratory; it does not constitute a later untouched confirmation experiment.",
    ])
    lines.append("")
    markdown_path = os.path.splitext(args.output)[0] + ".md"
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"status": output["status"], "json": args.output, "markdown": markdown_path,
                      "symbols": [{"symbol": item.get("symbol"), "status": item.get("status")} for item in reports]}, indent=2))


if __name__ == "__main__":
    main_cli()

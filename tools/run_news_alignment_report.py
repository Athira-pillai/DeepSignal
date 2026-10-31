"""Write the Phase 6 historical-news availability and limitations report."""

import argparse
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _configured_news_key():
    if os.getenv("NEWS_API_KEY", "").strip():
        return True
    path = os.path.join(ROOT, "backend", ".env")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            return any(line.partition("=")[0].strip() == "NEWS_API_KEY"
                       and bool(line.partition("=")[2].strip())
                       for line in handle if "=" in line and not line.lstrip().startswith("#"))
    except OSError:
        return False


def build_report():
    return {
        "phase": 6,
        "status": "historical_corpus_unavailable_no_ablation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_assessment": {
            "newsapi": {
                "credential_configured": _configured_news_key(),
                "account_plan_verified": False,
                "endpoint": "Everything /v2/everything",
                "published_time_field": "publishedAt (UTC)",
                "current_documented_history": "Developer: up to one month; paid Business/Advanced: up to five years.",
                "terms_limitations": "Terms prohibit reproducing/republishing copyrighted material and building a competing news database; Developer plan is development/testing only.",
                "historical_experiment_status": "not_queried; available plan and permitted retention/use for the intended research corpus were not verified",
                "documentation": "https://newsapi.org/docs/endpoints/everything",
                "pricing": "https://newsapi.org/pricing",
                "terms": "https://newsapi.org/terms",
            },
            "yahoo_finance_yfinance": {
                "current_behavior": "Yahoo quote search and ticker news results; no historical-news archive query is implemented.",
                "published_time_fields": ["providerPublishTime", "pubDate"],
                "historical_completeness_verified": False,
                "terms_limitations": "Yahoo terms restrict reuse to create a database/archive that competes with or materially substitutes for Yahoo services/data.",
                "terms": "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
            },
        },
        "alignment_implementation": {
            "version": "strict-published-at-point-in-time-v1",
            "cutoff": "News published at or before the explicit cutoff; historical session alignment uses 16:00 America/New_York converted to UTC.",
            "timestamp_policy": "Missing, date-only, invalid, or timezone-naive publication timestamps are rejected.",
            "relevance_policy": "Exact normalized ticker metadata, provider query match, or exact ticker token in headline is required.",
            "deduplication": "Canonical HTTP(S) article reference with tracking parameters removed; otherwise stable SHA-256 of source, timestamp, and normalized title.",
            "provenance": "Retains publication time, source, canonical article reference, ticker/query relevance, and acceptance status.",
        },
        "historical_coverage": {
            "licensed_point_in_time_archive_supplied": False,
            "sessions_evaluated": 0,
            "eligible_articles": 0,
            "coverage_fraction": None,
            "missing_news_dates": None,
            "meaning": "Coverage is unknown because no historical article corpus with verified access/retention rights was supplied. Zero here means no corpus was evaluated, not that the market had no news.",
        },
        "ablation": {
            "run": False,
            "reason": "No trustworthy, permitted, locally supplied point-in-time news archive was available for the historical walk-forward dates.",
            "finbert_historical_incremental_value": "not established",
            "today_headlines_substituted_for_history": False,
        },
    }


def write_markdown(report, path):
    provider = report["provider_assessment"]
    coverage = report["historical_coverage"]
    lines = [
        "# Phase 6 - Historical news alignment", "",
        f"**Status:** {report['status']}  ",
        f"**Generated:** {report['generated_at']}", "",
        "## Provider assessment", "",
        "| Provider | Timestamp support | Historical access | Result |",
        "|---|---|---|---|",
        f"| NewsAPI | `{provider['newsapi']['published_time_field']}` documented as UTC | {provider['newsapi']['current_documented_history']} Account plan verified: {provider['newsapi']['account_plan_verified']}. | Not queried for a historical corpus. |",
        f"| Yahoo Finance via yfinance | {', '.join(provider['yahoo_finance_yfinance']['published_time_fields'])} in current-search responses | No historical archive query in this application; completeness unverified. | Current search only. |",
        "", "NewsAPI documents the `/v2/everything` `publishedAt` timestamp and a five-year search on paid Business/Advanced plans. The Developer plan is limited to articles up to one month old and development/testing. The intended account plan was not verified, and the experiment's older dates fall outside the Developer window. NewsAPI's terms restrict republishing third-party content and building a competing news database. Yahoo results are current searches, and Yahoo's terms restrict creating a competing archive. Therefore a local historical corpus was not queried or retained.", "",
        "## Strict alignment added", "",
        "- Reject missing, invalid, date-only, and timezone-naive timestamps; require publication time at or before the prediction cutoff.",
        "- For historical sessions, use the NYSE 16:00 America/New_York close converted to UTC, including daylight-saving transitions.",
        "- Require ticker relevance through exact ticker metadata, the exact provider query used, or an exact ticker token in the title.",
        "- Deduplicate canonical URLs after removing tracking parameters; use a stable SHA-256 reference when a URL is absent.",
        "- Keep source, timestamp, reference, relevance match, and alignment status in the article provenance.",
        "- Live prediction now applies the same strict as-of filter and returns news-alignment counts. This does not make live headlines a historical archive.", "",
        "## Coverage and ablation", "",
        f"Historical sessions evaluated: {coverage['sessions_evaluated']}; eligible articles: {coverage['eligible_articles']}; coverage fraction: {coverage['coverage_fraction']}; missing-news dates: {coverage['missing_news_dates']}.",
        "Coverage is unknown because no licensed point-in-time archive was supplied. The zero eligible-article count means no corpus was evaluated; it does not mean there were no historical articles.",
        "No sentiment ablation was run, and no historical FinBERT benefit is claimed. Today's headlines were not substituted into past sessions.", "",
        "## Limitations", "",
        "A configured NewsAPI credential does not establish which plan is active, the plan's permitted historical window, or rights to persist a research archive. Provider timestamps indicate publisher-reported publication time, not necessarily the first moment the information became publicly available. Ticker keyword matches can still have imperfect issuer relevance. Coverage, missing dates, article count, and FinBERT sentiment quality therefore remain unknown for historical sessions.", "",
        "## Reproduction", "",
        "Run `python tools/run_news_alignment_report.py` to regenerate this report. The command does not make provider requests, expose keys, scrape articles, or claim a sentiment ablation.", "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "research", "real_data", "phase6_news_alignment.json"))
    args = parser.parse_args()
    report = build_report()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown = os.path.splitext(args.output)[0] + ".md"
    write_markdown(report, markdown)
    print(json.dumps({"status": report["status"], "json": args.output, "markdown": markdown,
                      "historical_coverage": report["historical_coverage"]}, indent=2))


if __name__ == "__main__":
    main()

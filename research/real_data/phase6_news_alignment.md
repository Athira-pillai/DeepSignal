# Phase 6 - Historical news alignment

**Status:** historical_corpus_unavailable_no_ablation  
**Generated:** 2026-10-31T14:09:59.516426+00:00

## Provider assessment

| Provider | Timestamp support | Historical access | Result |
|---|---|---|---|
| NewsAPI | `publishedAt (UTC)` documented as UTC | Developer: up to one month; paid Business/Advanced: up to five years. Account plan verified: False. | Not queried for a historical corpus. |
| Yahoo Finance via yfinance | providerPublishTime, pubDate in current-search responses | No historical archive query in this application; completeness unverified. | Current search only. |

NewsAPI documents the `/v2/everything` `publishedAt` timestamp and a five-year search on paid Business/Advanced plans. The Developer plan is limited to articles up to one month old and development/testing. The intended account plan was not verified, and the experiment's older dates fall outside the Developer window. NewsAPI's terms restrict republishing third-party content and building a competing news database. Yahoo results are current searches, and Yahoo's terms restrict creating a competing archive. Therefore a local historical corpus was not queried or retained.

## Strict alignment added

- Reject missing, invalid, date-only, and timezone-naive timestamps; require publication time at or before the prediction cutoff.
- For historical sessions, use the NYSE 16:00 America/New_York close converted to UTC, including daylight-saving transitions.
- Require ticker relevance through exact ticker metadata, the exact provider query used, or an exact ticker token in the title.
- Deduplicate canonical URLs after removing tracking parameters; use a stable SHA-256 reference when a URL is absent.
- Keep source, timestamp, reference, relevance match, and alignment status in the article provenance.
- Live prediction now applies the same strict as-of filter and returns news-alignment counts. This does not make live headlines a historical archive.

## Coverage and ablation

Historical sessions evaluated: 0; eligible articles: 0; coverage fraction: None; missing-news dates: None.
Coverage is unknown because no licensed point-in-time archive was supplied. The zero eligible-article count means no corpus was evaluated; it does not mean there were no historical articles.
No sentiment ablation was run, and no historical FinBERT benefit is claimed. Today's headlines were not substituted into past sessions.

## Limitations

A configured NewsAPI credential does not establish which plan is active, the plan's permitted historical window, or rights to persist a research archive. Provider timestamps indicate publisher-reported publication time, not necessarily the first moment the information became publicly available. Ticker keyword matches can still have imperfect issuer relevance. Coverage, missing dates, article count, and FinBERT sentiment quality therefore remain unknown for historical sessions.

## Reproduction

Run `python tools/run_news_alignment_report.py` to regenerate this report. The command does not make provider requests, expose keys, scrape articles, or claim a sentiment ablation.

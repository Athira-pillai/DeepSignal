# Phase 2 — Hybrid signal backtest

**Run date:** 2026-10-31  
**Status:** completed; historical result, not evidence of future profitability.  
**Ticker:** AAPL  
**Reproduce:** `\.venv\Scripts\python.exe tools\run_hybrid_backtest.py --symbol AAPL --period 3y`

## Result

The fixed hybrid ended at **$125,103.79** from $100,000 (**+25.1038%**) over 224 sessions. Its maximum drawdown was **−16.1417%**, annualized Sharpe **1.2878**, and it recorded 12 executed trades including the terminal liquidation. On the same dates, buy-and-hold ended at **$122,791.77** (**+22.7918%**), with a **−13.7985%** maximum drawdown and Sharpe **1.1746**. No-trade cash returned 0%.

| Strategy | Ending equity | Return | Max drawdown | Annualized Sharpe | Executed trades |
|---|---:|---:|---:|---:|---:|
| Fixed hybrid | $125,103.79 | 25.1038% | −16.1417% | 1.2878 | 12 |
| LSTM only | $125,419.95 | 25.4199% | −13.7985% | 1.2518 | 2 |
| Random Forest only | $119,377.17 | 19.3772% | −15.1112% | 1.1137 | 42 |
| Buy and hold | $122,791.77 | 22.7918% | −13.7985% | 1.1746 | 2 |
| No-trade cash | $100,000.00 | 0.0000% | 0.0000% | 0.0000 | 0 |

The hybrid exceeded buy-and-hold return in this sample but also had a larger maximum drawdown. The LSTM-only return was slightly higher; this single ticker and test window do not support a general model ranking.

## Method and provenance

- Yahoo Finance adjusted daily OHLCV; requested period 3 years. The labeled feature frame had 703 rows, spanning 2024-01-11 through 2026-10-30. The scored/executed walk-forward window was 2025-05-16 through 2026-04-09: 225 price rows and 224 measured sessions.
- Four expanding train/validation/test folds. The final chronological 20% holdout was excluded completely from the backtest. Each fold's Random Forest and LSTM were fit using data before that fold's test dates; LSTM early stopping used its validation window. LSTM trained in all four folds.
- Fixed production weights: LSTM 40%, Random Forest 40%, FinBERT 10%, technical score 10%. FinBERT sentiment was neutral (score 0) on every row because point-in-time historical news is unavailable. XGBoost was disabled.
- All three model strategies use the same out-of-sample dates and the production hybrid probability thresholds. Signals are formed after close and acted on at the next available open. The final test-date signal is retained in the audit but not executed because no following open exists inside the test window.
- Long-only, full-equity target sizing with fractional shares; no shorting. Starting cash $100,000. Transaction cost 0.06% and adverse slippage 0.04% per trade. Open positions are liquidated at the last close, with costs.
- Price window SHA-256: `9c6ecfcbc67337d8ce9a70301f1650f790c7e8a58bdec1b850e1899d3e0ec96f`. Full downloaded OHLCV SHA-256: `0dd4f287b30d1d646121b1bc227afd16750418a299f78a231a71ebf68a161daf`. Signal SHA-256: `8130a99228594827398798b651d974d478411651de327f67922b60096736e70f`.
- Machine-readable report and per-session component/signal audit: [phase2_hybrid_backtest.json](phase2_hybrid_backtest.json).

## Limitations

- This is one AAPL sample. It does not establish that the strategy will generalize to other securities or future periods.
- The walk-forward development folds overlap in training history, and there are only four test folds. Results depend on the chosen period and split configuration.
- Historical point-in-time news is unavailable, so the nominal 10% sentiment weight contributes a constant neutral value. This run does not test a functioning historical sentiment signal.
- The production probability thresholds and weights are fixed heuristics, not calibrated on this ticker's walk-forward predictions. The LSTM probability may not be calibrated; score bands therefore need not correspond to real confidence levels.
- Daily bars cannot reconstruct intraday price paths, bid/ask spreads, market impact, partial fills, or order queue behavior. Fees and slippage are constant assumptions; taxes and funding costs are omitted.
- Fractional shares and full-position transitions simplify real order sizing. Buy-and-hold uses modeled entry and terminal exit costs and is shown as a benchmark, not as an investable recommendation.
- Data are adjusted historical OHLCV from Yahoo Finance. Provider revisions, corporate-action handling, missing bars, and differences from executable prices can change results.
- This report deliberately excludes the final holdout. It is a development-stage OOS simulation and has not had a subsequent untouched confirmation period.


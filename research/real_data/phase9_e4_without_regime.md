# Phase 4 — Fixed versus adaptive hybrid evaluation

**Ticker:** AAPL  
**Window:** 2025-05-16 through 2026-04-09  
**Price rows:** 225  
**Walk-forward folds:** 4; final holdout excluded: True

## Results

| Strategy | Ending equity | Return | Max drawdown | Sharpe | Trades |
|---|---:|---:|---:|---:|---:|
| Fixed hybrid | $125,103.78 | 25.1038% | -16.1417% | 1.2878 | 12 |
| Adaptive hybrid | $114,280.24 | 14.2802% | -18.7356% | 0.8238 | 16 |
| Buy and hold | $122,791.79 | 22.7918% | -13.7985% | 1.1746 | 2 |
| No-trade cash | $100,000.00 | 0.0% | 0.0% | 0.0 | 0 |

## Method

- Both hybrid strategies use identical walk-forward test dates, target, next-open execution, costs, slippage, sizing, and buy-and-hold/cash baselines.
- For each test fold, component weights use only out-of-sample predictions and labels from that fold's immediately preceding validation window. Weights stay frozen over the following test fold.
- Weights use Brier skill relative to a 0.50 direction probability plus renormalized fixed-component priors, projected onto configured bounds.
- Regime context is disabled for adaptive weight selection; scores use the overall prior validation window.
- HMM regime context is inferred causally through each test signal date. HMM fit/validation uses only earlier training and validation features; in the no-regime-weight control, those labels are not used to select weights.
- Historical FinBERT news is unavailable. The adaptive strategy excludes sentiment and renormalizes available LSTM, Random Forest, and technical components; the fixed baseline retains its neutral sentiment contribution.
- Both strategies use the same production probability-to-signal thresholds. The final test-date signal is audited and not executed without a next in-window open.
- Full-source OHLCV SHA-256: `94bf5ba4720da5dbbf597317ee44ac0f143528f7fb0f9c13eaf79f19418c33e3`; OOS prices SHA-256: `ac6f357639306a93235895a5a9d4b9738a844edeaf46b089a04a749491b4e99b`.

## Fold audit

| Fold | Test dates | Validation dates | HMM states | LSTM test status | Regime fallback |
|---:|---|---|---:|---|---|
| 1 | 2025-05-16 to 2025-08-07 | 2025-02-26 to 2025-05-15 | 2 | trained | none |
| 2 | 2025-08-08 to 2025-10-27 | 2025-05-19 to 2025-08-07 | 3 | trained | none |
| 3 | 2025-10-28 to 2026-01-16 | 2025-08-08 to 2025-10-27 | 2 | trained | none |
| 4 | 2026-01-20 to 2026-04-09 | 2025-10-28 to 2026-01-16 | 2 | trained | none |

## Limitations

- This is one ticker and one provider-dependent historical window. It does not establish future profitability or that adaptive weighting is better.
- This run intentionally uses overall validation scores only; it is the no-regime-weights control.
- Brier scores and regime labels are noisy on short windows. The weights are a transparent heuristic, not a trained or calibrated meta-model.
- Historical point-in-time news is absent, so this experiment cannot measure FinBERT's historical incremental value.
- Yahoo adjusted daily bars, fractional shares, constant fees/slippage, and next-open fills simplify spreads, market impact, partial fills, taxes, and funding costs.
- The final chronological holdout is excluded. A separate later confirmation window and multi-ticker evaluation are needed before changing the fixed default.

Audit contains 225 test-date rows, including non-executable final-date signals. See the JSON for raw components, weights, validation metrics, availability, regimes, trades, and equity curves.

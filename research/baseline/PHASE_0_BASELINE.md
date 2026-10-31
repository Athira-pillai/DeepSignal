# DeepSignal Phase 0 baseline

## Scope and provenance

This is an inspection baseline, not a newly run model experiment. It summarizes the existing local artifacts without changing them. The stored records are dated 2026-10-24 and use AAPL. Market data is labeled Yahoo Finance adjusted daily OHLCV; the artifacts record hashes and date windows below. No new provider fetch or model fit was run during Phase 0.

| Artifact | Run ID | Window | Rows / sessions | Data SHA-256 |
| --- | --- | --- | ---: | --- |
| `backend/evaluation_runs.json` | `20261024T073936.351137Z` | data 2024-01-04–2026-10-22; test 2026-04-02–2026-10-22 | 703 / 141 test | `9e807960a723330c5b284fd4f1f9faf7f977ce20836abf443e2cd533b87289e4` |
| `backend/backtest_runs.json` (latest) | `20261024T075412.583516Z` | 2023-10-24–2026-10-23 | 753 | `9dd1dbc550265411b2c7dcb1fd3158572c8206aee591beaffbfc855c2ad2188` |
| `backend/prediction_runs.json` (latest saved sample) | saved 2026-10-24 | AAPL, market timestamp 2026-10-23; 753 rows, 2023-10-24–2026-10-23 | 753 | `1655ea19422b7705633f988857ae1599c323f353f09e33fc409d3e17dc7fa2b2` |

## Model and target behavior

- The live target is one-session direction: next session close greater than current session close, with the decision made after the current close.
- Live hybrid version `weighted-probability-v1` combines LSTM 0.40, Random Forest 0.40, mapped FinBERT sentiment 0.10, and technical probability 0.10. Sentiment is mapped from `[-1, 1]` to `[0, 1]`; the documentation and code call this heuristic, not calibrated probability. Signal cutoffs are 0.64/0.54 for strong-buy/buy and 0.36/0.46 for strong-sell/sell; values between are hold.
- The stored live response has top-level prediction fields including `stock_symbol`, `prediction_timestamp`, `market_data`, `as_of`, `latest_close`, `signal_type`, `confidence`, component probabilities, `signal_explanation`, `technical_indicators`, `news`, `model_comparison`, and `backtest`.
- The current evaluation code defines nine price/volume features: `return_1d`, `return_5d`, `sma20_gap`, `sma50_gap`, `rsi`, `macd`, `macd_signal`, `bb_position`, and `volume_change`. Its target is `(next close > current close)`. The current evaluator builds previous-direction, Random Forest, LSTM, and fixed-hybrid comparisons on a final 20% chronological test window; RF fits through the validation window, and LSTM scaler fits only on the first 60% rows. Historical sentiment is neutral because point-in-time news is unavailable.

## Stored evaluation metrics

The existing evaluation artifact contains only `previous_direction` and `random_forest` in its model table. It does **not** contain LSTM or hybrid evaluation results, despite those paths existing in current `backend/evaluation.py`. Do not treat this older artifact as evidence for the live hybrid.

| Model | Accuracy | Precision up | Recall up | F1 up | Test return | Max drawdown | Annualized Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous direction | 0.4113 | 0.4684 | 0.4744 | 0.4713 | -1.1371% | -6.0275% | -0.0682 |
| Random Forest | 0.5177 | 0.5625 | 0.5769 | 0.5696 | 18.3999% | -4.9667% | 1.9140 |
| Buy and hold | — | — | — | — | 35.9824% | -12.7062% | 2.2214 |
| Flat cash | — | — | — | — | 0.0000% | 0.0000% | 0.0000 |

Evaluation simulation reports 0.06% transaction cost and 0.04% slippage. The single stored holdout is not repeated walk-forward evidence.

## Separate technical backtest

The latest backtest is explicitly named `Five-day momentum and RSI; long-only` (`technical-momentum-rsi-v1`) and uses the next-open engine (`next-open-long-only-v1`). Signals are generated after close and executed at the next available open; the open position is liquidated at the final close. Assumptions are $100,000 starting cash, 100% target position, fractional shares, 0.06% fees and 0.04% slippage.

| Series | Total return | Max drawdown | Annualized Sharpe | Trades |
| --- | ---: | ---: | ---: | ---: |
| Technical momentum/RSI | 48.0371% | -22.1885% | 0.7933 | 92 |
| Buy and hold | 103.2528% | -33.3605% | 1.0181 | 2 |
| No trade | 0.0000% | 0.0000% | 0.0000 | 0 |

The report retains a compatibility summary field called `hybrid` that duplicates the technical strategy metrics. This is not a hybrid-model signal backtest; the frontend should rely on the explicit strategy label.

## Reproduction and limits

Existing test and frontend build commands are recorded in [CODEX_PROGRESS.md](../../CODEX_PROGRESS.md). The offline suite passed. The frontend production build passed. The running API returned `{"status":"ok"}` from `/api/health`; an additional server could not bind because port 8000 was already occupied. This baseline report uses preexisting local artifacts only and does not claim current live performance.

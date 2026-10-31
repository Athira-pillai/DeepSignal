# Phase 1 walk-forward pipeline validation

## Implementation

The existing `/api/evaluate/{symbol}` now evaluates an expanding sequence of chronological folds inside the earlier 80% development window, followed by one final 20% chronological holdout. Each fold reserves the most recent validation interval before its test interval. Folds expand by moving the test interval forward and incorporating only data whose target outcomes are known by that test cutoff. The final holdout is not included in any fold, validation, model choice, or threshold selection.

The endpoint preserves the existing `models` response field for final-holdout results and adds fold details plus mean and standard deviation of fold metrics. Walk-forward tabular models fit on training rows only. For the final holdout, Random Forest is refit on all pre-holdout development rows using its already fixed configuration; no hyperparameters or thresholds are selected on the final holdout. Metrics include accuracy, balanced accuracy, precision/recall/F1, confusion matrix, class counts, ROC-AUC when both classes occur, Brier score, and reliability bins. The unchanged fixed 40/40/10/10 hybrid remains the baseline. XGBoost is optional and off by default.

`GET /api/evaluate?symbols=AAPL,MSFT,NVDA` evaluates a configurable comma-separated universe. The default list is `EVALUATION_SYMBOLS` in `backend/.env`; each ticker has an independent status. XGBoost may be enabled with `EVALUATION_INCLUDE_XGBOOST=true`.

Preprocessing uses `MinMaxScaler` fitted only on training feature rows in each LSTM fold. The validation interval is reserved for LSTM early stopping and is not included in tabular model fits during walk-forward folds. Test labels are only used after all predictions are produced. Target alignment is next-session close greater than current-session close. Historical news remains unavailable and contributes neutral sentiment to the hybrid evaluation.

## Deterministic smoke artifact

The reproducible fixture is generated with:

```powershell
.\.venv\Scripts\python.exe tools\run_walk_forward_fixture.py
```

It creates [synthetic_fixture_report.json](synthetic_fixture_report.json) from seeded synthetic prices. The report contains four expanding test folds of 60 rows each, a 150-row reserved final holdout, and the data fingerprint `daa875fa32bed04039d147dfd66be993505e59a6e4abfa38abd67ecc4f124588`. LSTM outputs are deliberately neutral-stubbed in this fixture so this command tests fold orchestration and report serialization without a costly neural fit. These synthetic metrics are only a software smoke test; they are not stock-market results and do not support model-performance claims.

## Evidence status and remaining limitation

The offline backend suite passes, including tests for fold order, target alignment, train-only scaling, single-class metrics, optional XGBoost failure, and the multi-ticker API. A real Yahoo Finance report for AAPL, MSFT, JPM, JNJ, and XOM is now available at [PHASE_1_REAL_DATA_EVALUATION.md](../real_data/PHASE_1_REAL_DATA_EVALUATION.md), with its machine-readable companion JSON. Each ticker has 753 raw bars, 703 labeled rows, four balanced walk-forward folds, and a separate 141-row final holdout. The initial run exposed a one-row partial fold; the generator was corrected to distribute test rows evenly and all five evaluations were rerun. The final results are mixed and do not establish a consistent hybrid improvement. Keep the final holdout untouched for future model changes.

The real-data report documents provider, date coverage, fingerprints, configurations, versions, fold boundaries, metrics, and limits. The same final-holdout window was scored in the preliminary run and in the corrected run; the split fix was prompted by a one-row walk-forward fold, not by final-holdout metrics, and no model settings were changed. Treat these final-holdout metrics as exploratory rather than pristine confirmatory evidence. Results cover one three-year sample over five selected large-cap tickers; do not generalize them to a broader universe or treat simplified trading metrics as executable returns.

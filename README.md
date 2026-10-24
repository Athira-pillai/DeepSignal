# DeepSignal — Hybrid Stock Trend Prediction

A standalone version of the uploaded stock-prediction UI, extended to match the final-year-project proposal in the PDF.

## What is included
- Live/latest available OHLCV data from `yfinance`
- Technical indicators: SMA 20/50, RSI, MACD, Bollinger Bands
- LSTM price/trend forecasting (TensorFlow/Keras)
- Financial-news sentiment with FinBERT (Hugging Face)
- Random Forest signal classifier with optional XGBoost comparison
- Versioned hybrid that combines compatible next-session-up probabilities
- Buy / Sell / Hold signal generation
- Dedicated next-open backtesting engine with portfolio accounting, costs, and slippage
- Buy-and-hold and naive baselines
- Accuracy, cumulative return, Sharpe ratio and maximum drawdown
- SHAP feature importance when available
- Multi-stock/sector evaluation endpoint
- Reproducible chronological direction evaluation at `/api/evaluate/{symbol}`
- Streamlit-style research sections reproduced in the React UI
- Signal history stored locally in the browser

The proposal specifies yfinance/News API, LSTM, FinBERT, Random Forest/XGBoost/LightGBM, technical indicators, SHAP, walk-forward validation, transaction costs/slippage, baseline comparisons, and a Streamlit/Plotly dashboard. This implementation uses the same architecture but keeps the existing React UI style from the ZIP and exposes the ML system through a FastAPI backend. The proposal's pandas-only indicator fallback is implemented so TA-Lib is not required.

## 1. Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
python -m uvicorn main:app --reload --port 8000
```

Set `NEWS_API_KEY` if you want News API results. If it is absent, the backend falls back to Yahoo Finance news supplied through yfinance.

## 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`.

## Validation suite

From the project root, install the backend development dependencies and run the offline validation suite:

```powershell
.\.venv\Scripts\python -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python -m unittest discover -s backend/tests -v
```

The tests mock market/news providers and model dependencies, so they do not make network requests or download model weights.

## 3. First prediction

Open the Analysis page, enter a symbol such as `AAPL`, and click **Analyze**. The first request can take longer because TensorFlow and FinBERT may need to load/download models. Later requests are cached in the backend process.

## 4. Run a model evaluation

With the backend running, request a report for a ticker (PowerShell example):

```powershell
Invoke-RestMethod http://localhost:8000/api/evaluate/AAPL
```

The evaluator builds next-session direction labels from Yahoo Finance daily OHLCV, creates chronological train/validation/test windows, then compares a previous-direction baseline, Random Forest, LSTM, and hybrid on the same final test dates. It reports accuracy, precision, recall, F1, confusion matrices, and a simplified long-or-cash trading simulation against buy-and-hold and flat-cash baselines. Reports are saved to Supabase when configured, otherwise to `backend/evaluation_runs.json`, with the data hash, date windows, configuration, and package/model versions.

The live hybrid predicts whether the next session's close will be above the current session's close. It combines LSTM and Random Forest direction probabilities with a mapped FinBERT score and technical score using fixed, versioned weights (40%, 40%, 10%, 10%). The API response records the target, weights, component probabilities, contributions, confidence, and signal rule.

The evaluation compares the Random Forest, LSTM, hybrid, and previous-direction baseline on the same final test dates. Historical point-in-time news is unavailable, so neutral sentiment is used in the historical hybrid evaluation; this result does not validate the live news component. Run the offline tests with `python -m unittest discover -s backend/tests -v` from the project root.

## 5. Run a backtest

Use the Backtest page or call `http://localhost:8000/api/backtest/AAPL`. The current strategy creates technical BUY/SELL/HOLD signals after each close and executes them at the next session open. The engine applies the saved transaction-cost and slippage settings, tracks cash, shares, equity and drawdown, and compares the strategy with buy-and-hold and no-trade baselines. Each API run is saved to Supabase when configured, otherwise to `backend/backtest_runs.json`, with its input data hash, strategy/engine versions, and execution assumptions. This is a technical momentum strategy, not the live LSTM/FinBERT hybrid.

## 6. Use online persistent storage (no database software download)

The backend can save prediction, signal, market-data metadata, evaluation, and backtest records to a hosted Supabase Postgres project. It connects over HTTPS using the `requests` package already in the project; you manage the project from a website.

1. Create a Supabase project in the web dashboard.
2. In its **SQL Editor**, open `backend/sql/001_persistent_storage.sql`, paste the file contents, and run it once.
3. Add the two settings from `backend/.env.example` to `backend/.env` (keep any existing settings there). In Supabase, copy the **Project URL** and create/copy a **Secret key**. Put them in `SUPABASE_URL` and `SUPABASE_SECRET_KEY`. Keep the secret key on the backend; never put it in frontend settings or commit it. The backend sends it only in the `apikey` header.
4. Restart the backend and open `http://localhost:8000/api/storage/status`. Continue when the response says `connected`.
5. Run a prediction, evaluation, or backtest. Read saved predictions at `/api/history/predictions?symbol=AAPL` and saved runs at `/api/history/experiments`.
6. Restart the backend and read those history endpoints again. Supabase keeps the records outside the local app process.
7. To copy existing local JSON evaluation/backtest reports to Supabase, call `POST http://localhost:8000/api/storage/import-local-experiments` once. The import uses the report's run ID, so retries do not create duplicate experiment rows.

### Scheduled market-data refresh

When the backend starts, a lightweight background thread refreshes daily OHLCV bars and technical features for the tickers in `MARKET_DATA_SYMBOLS` (default `AAPL`). It runs once on startup and then every 6 hours by default. Set `MARKET_DATA_SYMBOLS=AAPL,MSFT,NVDA` and optionally `MARKET_DATA_REFRESH_MINUTES=360` in `backend/.env` to change the list and cadence; set `MARKET_DATA_REFRESH_ENABLED=false` to disable it. The provider controls when new daily bars become available, so the refresh does not guarantee real-time prices.

To store scheduled bars and job status in Supabase, run `backend/sql/002_market_data_scheduler.sql` once in the Supabase SQL Editor after the original storage setup. Without Supabase, the same data is saved locally in ignored JSON files. Check `http://localhost:8000/api/jobs/market-data` for whether the job is enabled, its last result, rows processed, and any per-symbol failures. Each daily bar is upserted by symbol, interval, and timestamp, so retries do not duplicate it.

### In-process cache

Repeated requests for the same Yahoo Finance daily history are cached for 15 minutes by default (`MARKET_DATA_CACHE_TTL_SECONDS`). Set it to `0` to disable OHLCV caching. Identical technical-indicator calculations are cached by a fingerprint of their input rows and the feature-code version. The scheduled market refresh forces a new provider fetch, so it does not reuse the HTTP cache. Inspect cache hits, misses, expirations, and evictions at `http://localhost:8000/api/cache/status`. This cache is bounded and local to one backend process; it clears when the backend restarts. Prediction responses are not cached because news and input data can change between calls.

### Saved model versions

The first prediction for a symbol, data fingerprint, model configuration, code version, and package set fits and saves a versioned LSTM/Random Forest artifact bundle under `backend/model_artifacts/`. It includes the fitted LSTM scaler, ordered feature lists, target and date window, configuration, seed, package versions, and Random Forest validation metrics. The next prediction with the exact same identity reloads those saved models; a changed data window or model identity gets a different version. Each prediction and its Supabase/local history record contains `model_version`. Browse metadata at `GET /api/models`, inspect one at `GET /api/models/{version}`, and load/validate it into memory at `POST /api/models/{version}/load`. That loaded model is only used for a prediction when its complete identity matches. Old artifact versions are kept for rollback and are ignored by Git. The pretrained FinBERT weights remain in the Transformers cache and their model name is recorded; they are not copied into each trained artifact. Backtests are rule-based rather than fitted models, so they record `model_version: null` and retain their strategy and engine versions.

Model files are local to the backend machine, while the database stores the version ID with each prediction. Back up `backend/model_artifacts/` if you need to reproduce predictions on another machine. Increment `PREDICTION_MODEL_CODE_VERSION` when changing live-model training or preprocessing code so the new run cannot accidentally reuse an older artifact.

No database server or new database package is installed on your computer. Without Supabase settings, the app keeps its local JSON fallback. Predictions previously stored only in browser local storage are not automatically copied; predictions made through the API after setup will be saved. Supabase Free projects can pause after a week of low activity and do not have the paid plans' automatic backups, so export important data periodically. See [project pausing](https://supabase.com/docs/guides/platform/free-project-pausing) and [backup guidance](https://supabase.com/docs/guides/platform/backups).

### Important
A prediction is a model output, not a guarantee of future returns or investment advice. The API reports the timestamp and latest available market bar used for the prediction so the UI is transparent about freshness.

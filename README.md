# DeepSignal — Hybrid Stock Trend Prediction

DeepSignal is a full-stack stock-direction research application. A React dashboard calls a FastAPI service that retrieves daily market data, builds model signals, runs chronological evaluations and backtests, and records reproducible run metadata. It is a research tool; it does not place trades.

## What is included
- Live/latest available OHLCV data from `yfinance`
- Technical indicators: SMA 20/50, RSI, MACD, Bollinger Bands
- LSTM price/trend forecasting (TensorFlow/Keras)
- Financial-news sentiment with FinBERT (Hugging Face)
- Random Forest signal classifier with optional XGBoost comparison
- Versioned fixed-weight hybrid combining next-session-up probabilities and technical evidence
- Gaussian HMM market-regime estimates (analysis context, not a live signal vote)
- Validation-selected adaptive weights, Platt/isotonic calibration, and No Trade analysis for historical evaluation
- Timestamp-aware live news filtering; historical point-in-time news is not available for sentiment ablations
- Buy / Sell / Hold signal generation
- Separate technical and hybrid walk-forward backtests with next-open execution, portfolio accounting, fees, and slippage
- Buy-and-hold and naive baselines
- Accuracy, balanced accuracy, ROC-AUC, Brier/reliability metrics, cumulative return, Sharpe ratio, and maximum drawdown
- SHAP feature importance when available
- Multi-stock/sector evaluation endpoint
- Versioned local model artifacts and optional Supabase persistence
- Research panels for regime, model contributions, adaptive weights, reliability, and experiment comparisons

The implemented stack is React/Vite for the browser UI and FastAPI for the Python service. The service uses Yahoo Finance market data, optional NewsAPI headlines, pandas indicators, and the model/evaluation components described below. Indicators use pandas; TA-Lib is not required.

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

For a production frontend bundle, run `npm run build` from `frontend/`.

## Validation suite

From the project root, install the backend development dependencies and run the offline validation suite:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\backend\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

The tests mock market/news providers and model dependencies, so they do not make network requests or download model weights.

The current full backend suite contains 62 tests. The latest verified production build transformed 2,211 modules.

## 3. First prediction

Open the Analysis page, enter a symbol such as `AAPL`, and click **Analyze**. The first request can take longer because TensorFlow and FinBERT may need to load/download models. Later requests are cached in the backend process.

## 4. Run a model evaluation

With the backend running, request a report for a ticker (PowerShell example):

```powershell
Invoke-RestMethod http://localhost:8000/api/evaluate/AAPL
```

The evaluator builds next-session direction labels from Yahoo Finance daily OHLCV, compares a previous-direction baseline, Random Forest, LSTM, and fixed hybrid on identical expanding-window folds, then reports a separate final chronological 20% holdout. The final holdout is not used to tune models or thresholds. Reports include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC when both classes occur, Brier score, reliability bins, confusion matrices, and simplified long-or-cash metrics. They are saved to Supabase when configured, otherwise to `backend/evaluation_runs.json`, with data fingerprints, fold dates, configuration, and package/model versions.

For a configurable multi-ticker evaluation, call `GET /api/evaluate?symbols=AAPL,MSFT,NVDA`. The default ticker list is controlled by `EVALUATION_SYMBOLS` in `backend/.env`. Each ticker receives a separate report and can fail independently. XGBoost comparison is optional and disabled by default; set `EVALUATION_INCLUDE_XGBOOST=true` to include it when the package is available.

To run a three-year real-data walk-forward report and save a standalone Markdown/JSON copy under `research/real_data/`, use:

```powershell
.\.venv\Scripts\python.exe tools\run_real_evaluation.py --symbols AAPL,MSFT,JPM,JNJ,XOM --period 3y
```

The optional XGBoost benchmark uses the same models, folds, features, and five-ticker universe. The current exploratory run gave XGBoost equal-ticker mean walk-forward accuracy 0.5333 and balanced accuracy 0.5330, but Brier score 0.2906; the fixed hybrid Brier score was 0.2509. The previously scored final holdout is reused, so this is not a pristine confirmatory result and supports no general model-superiority claim. See [XGBoost evaluation report](research/real_data/phase7_xgboost_evaluation.md) and [full audit](research/real_data/phase7_xgboost_evaluation.json). Reproduce with `--include-xgboost --output research/real_data/phase7_xgboost_evaluation.json`.

The dashboard's Market Regime panel and `GET /api/regime/AAPL` use a Gaussian HMM fitted on an earlier training window. The scaler and state-to-description mapping use training rows only; two- and three-state candidates are selected on a later validation window. Each displayed posterior is filtered through its stated daily cutoff. Regime descriptions are relative summaries, not ground truth or probabilities of return/profit. `pomegranate==1.1.2` is the only direct added package and reuses the existing PyTorch dependency; if it or market data is unavailable, the endpoint returns an explicit unavailable status.

To reproduce the descriptive walk-forward comparison between pooled hybrid metrics and metrics grouped by inferred HMM regime:

```powershell
.\.venv\Scripts\python.exe tools\run_regime_evaluation.py --symbols AAPL --period 3y
```

The current sample and full fold/state audit are in [research/real_data/phase3_regime_evaluation.md](research/real_data/phase3_regime_evaluation.md) and [research/real_data/phase3_regime_evaluation.json](research/real_data/phase3_regime_evaluation.json). Regenerate only the Markdown from an existing report with `--report-only`. The grouped metrics describe the fixed hybrid in different inferred states; they do not show that regime information improves predictions.

The live hybrid predicts whether the next session's close will be above the current session's close. It combines LSTM and Random Forest direction probabilities with a mapped FinBERT score and technical score using fixed, versioned weights (40%, 40%, 10%, 10%). The API response records the target, weights, component probabilities, contributions, confidence, and signal rule.

### Models and how the results are formed

- **LSTM:** learns from ordered sequences of daily price/volume indicators and estimates next-session direction. Its feature scaler is fit on the training portion of each evaluation fold.
- **Random Forest:** estimates next-session direction from tabular technical and volume features.
- **FinBERT:** scores eligible recent news text for sentiment. Articles with unknown timestamps or publication after the prediction cutoff are excluded. Historical evaluation uses neutral sentiment because a suitable point-in-time news archive is unavailable.
- **Technical evidence:** deterministic indicator/rule score used as one component of the fixed live hybrid; it is not a separately trained neural model.
- **Fixed hybrid:** the live default combines LSTM (40%), Random Forest (40%), mapped sentiment (10%), and technical evidence (10%). These fixed weights remain the default.
- **Gaussian HMM:** estimates latent market regimes from daily features. Its scaler and state descriptions are learned from earlier data; its filtered state probabilities describe model state, not the chance of a return. It does not vote in the live hybrid.
- **XGBoost:** optional evaluation comparison only; it is not part of the live hybrid.
- **Adaptive hybrid, calibration, and No Trade:** research paths. Adaptive weights and Platt/isotonic calibration are selected using chronological validation data and evaluated on later folds. No Trade thresholds are also validation-selected. These are not enabled as live calibrated controls.

Direction evaluation labels each session using whether the next session's adjusted close is above the current session's adjusted close. Expanding chronological train/validation/test folds keep later rows out of earlier fitting and selection. Backtests form signals after the daily close and execute at the next available open; reports include configured fees and slippage. Daily Yahoo bars are not tick-level data, and the simulations do not model market impact, partial fills, taxes, or funding costs.

### Current research findings and limitations

The consolidated research report covers seven experiments across multi-ticker model comparisons and a shared AAPL development window. Results are descriptive and do not establish future profitability:

- Across five tickers and four walk-forward folds, XGBoost had mean accuracy 0.5333 and balanced accuracy 0.5330, but Brier 0.2906. The fixed hybrid had mean accuracy 0.5165 and Brier 0.2509; the mixed metrics do not support a general superiority claim.
- On the shared AAPL window, fixed hybrid return was 25.1038%, versus 25.42% for LSTM-only. Adaptive hybrid returned 14.2802% with a larger drawdown than fixed hybrid.
- The paired regime-weight experiment had identical executed equity curves with and without regime-specific weight selection in this sample.
- Calibration worsened fixed-hybrid Brier (0.2676 versus raw 0.2532). The No Trade simulation reduced drawdown but also returned less than the fixed strategy.
- Historical sentiment ablation (E5) was not evaluable because eligible point-in-time news data were unavailable. The final holdout was previously examined during the initial walk-forward evaluation, so reused holdout results are exploratory rather than pristine confirmation.

See the [final experiment matrix](research/real_data/phase9_final_experiment_matrix.md) and [machine-readable results](research/real_data/phase9_final_experiment_matrix.json) for methods, fold details, data hashes, figures, and limitations. Supporting evaluation reports are in [`research/real_data/`](research/real_data/).

### API route overview

| Route | Purpose |
|---|---|
| `GET /api/health`, `/api/config`, `/api/models` | Service, tuning, and saved model metadata |
| `GET /api/predict/{symbol}` | Live prediction and component evidence |
| `GET /api/history/predictions`, `/api/history/experiments` | Saved prediction and experiment history |
| `GET /api/history/{symbol}`, `/api/company/{symbol}` | Daily price history and company lookup |
| `GET /api/regime/{symbol}` | Latest causal market-regime estimate |
| `GET /api/backtest/{symbol}` | Technical-strategy backtest |
| `GET /api/backtest/{symbol}/hybrid?mode=fixed\|adaptive` | Walk-forward hybrid backtest |
| `GET /api/evaluate/{symbol}`, `/api/evaluate?symbols=AAPL,MSFT` | Single- or multi-symbol evaluation |
| `GET /api/storage/status`, `/api/jobs/market-data`, `/api/cache/status` | Storage, refresh-job, and cache status |

The OpenAPI schema is available at `http://localhost:8000/docs` while the backend is running. The API retains compatibility fields and reports unavailable optional components explicitly.

The evaluation compares the Random Forest, LSTM, hybrid, and previous-direction baseline on the same final test dates. Historical point-in-time news is unavailable, so neutral sentiment is used in historical hybrid evaluation; this does not validate the live news component. Live articles with missing/ambiguous timestamps or publication times after the prediction cutoff are excluded and alignment counts are returned. The historical news assessment documents provider limits, archive rights, and why no historical sentiment ablation was run: [report](research/real_data/phase6_news_alignment.md), [audit](research/real_data/phase6_news_alignment.json).

## 5. Run a backtest

Use the Backtest page or call `http://localhost:8000/api/backtest/AAPL` for the technical momentum strategy. To simulate model signals, call `http://localhost:8000/api/backtest/AAPL/hybrid`. The hybrid route uses expanding walk-forward test predictions, excludes the final 20% holdout, and compares fixed hybrid, adaptive hybrid, LSTM-only and Random Forest-only signals on matching dates against buy-and-hold and no-trade. Fixed 40/40/10/10 weights remain the default (`HYBRID_BACKTEST_MODE=fixed`); select adaptive weights with `?mode=adaptive`. Adaptive weights use only the preceding validation fold, with configurable bounds and sample minimums in `backend/.env.example`. Both use after-close signals and next-open fills, with the saved transaction-cost and slippage settings. Reports are saved to Supabase when configured, otherwise to `backend/backtest_runs.json`.

The hybrid run uses neutral sentiment because historical point-in-time news is unavailable. It is long-only, uses fixed heuristic probability thresholds, and does not model market impact, intraday fills, taxes or funding costs. The AAPL fixed-versus-adaptive comparison found fixed hybrid return 25.1038% versus adaptive hybrid 14.2802% over the same 224 sessions. Platt-calibrated Brier score was worse than raw (0.2676 vs 0.2532) on the same 225 audited test rows, so calibration remains experimental and is not used by live predictions. The validation-selected No Trade backtest returned 10.1851% with 8 trades in this sample. These single-window results do not establish future profitability. See the [hybrid backtest report](research/real_data/PHASE_2_HYBRID_BACKTEST.md), [adaptive weighting report](research/real_data/phase4_adaptive_evaluation.md), and [calibration report](research/real_data/phase5_calibration_evaluation.md) with its [full audit](research/real_data/phase5_calibration_evaluation.json).

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

The local JSON fallback is selected when Supabase is not configured. If Supabase is configured but unreachable, the app reports the storage error; it does not silently mix local and hosted history. Check `/api/storage/status` and restore connectivity before relying on hosted persistence. Prediction and experiment responses include persistence status when saving fails; history reads can return HTTP 503 while the configured store is unavailable.

### Latest verification

- Backend: `python -m unittest discover -s backend/tests -v` — 62 tests passed.
- Frontend: `npm run build` — production build passed (2,211 modules transformed).
- Live integration: Yahoo Finance refresh, prediction, technical and hybrid backtests, evaluation, and local prediction/experiment-history reads returned successfully using real AAPL data. Hosted Supabase connectivity and history reads were also checked separately. Integration test records were written only through the local JSON fallback.
- `git diff --check` — passed.

These checks verify the current development environment; they do not guarantee future provider availability.

### Important
A prediction is a model output, not a guarantee of future returns or investment advice. The API reports the timestamp and latest available market bar used for the prediction so the UI is transparent about freshness.

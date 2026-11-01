# DeepSignal — Complete Project Guide

This guide describes the application as it exists in this repository: its architecture, pages, data flow, models, evaluation and backtesting, persistence, scheduled work, setup, APIs, and limitations. It is intended to be the main technical and user-facing reference for the project.

> **Purpose:** DeepSignal is an educational stock-analysis and machine-learning research prototype. It does not place trades, provide financial advice, or guarantee a prediction or return. Market data from Yahoo Finance can be delayed, incomplete, or unavailable.

## Contents

1. [At a glance](#1-at-a-glance)
2. [System architecture](#2-system-architecture)
3. [User interface and workflows](#3-user-interface-and-workflows)
4. [Market data, news, and features](#4-market-data-news-and-features)
5. [Prediction models and the hybrid signal](#5-prediction-models-and-the-hybrid-signal)
6. [Saved model artifacts and versioning](#6-saved-model-artifacts-and-versioning)
7. [Evaluation and backtesting](#7-evaluation-and-backtesting)
8. [Caching and scheduled market-data updates](#8-caching-and-scheduled-market-data-updates)
9. [Persistence and database design](#9-persistence-and-database-design)
10. [API reference](#10-api-reference)
11. [Project layout](#11-project-layout)
12. [Run the project](#12-run-the-project)
13. [Configuration](#13-configuration)
14. [Validation and troubleshooting](#14-validation-and-troubleshooting)
15. [Security, privacy, and boundaries](#15-security-privacy-and-boundaries)
16. [Current limitations and future work](#16-current-limitations-and-future-work)
17. [Glossary](#17-glossary)

## 1. At a glance

DeepSignal is a local full-stack web app. The browser provides the dashboard and analysis pages. A Python API fetches market data, engineers features, runs model components, combines their evidence, and returns a JSON result. The app can optionally use a hosted Supabase database for online history; without it, the backend writes supported records to local JSON files.

**Implementation status:** The planned implementation and research work is complete. The final report covers seven research experiments; the sentiment ablation was unavailable because no eligible point-in-time news archive was available. Current evidence, limitations, and report links are in [the final experiment report](../research/real_data/phase9_final_experiment_matrix.md) and [machine-readable report](../research/real_data/phase9_final_experiment_matrix.json). The implementation work log is [CODEX_PROGRESS.md](../CODEX_PROGRESS.md).

### Main capabilities

- Select a ticker from saved signal history or type a new symbol. The shared ticker control appears on Dashboard, Analysis, Backtest, and Models. It displays a company name from a small built-in list or looks one up through Yahoo Finance.
- Display daily OHLCV market history, a candlestick chart, technical indicators, recent headlines, sentiment, feature importance, and signal evidence.
- Estimate next-session up/down direction with an explicit weighted hybrid of LSTM, Random Forest, FinBERT sentiment, and technical rules.
- Show Buy, Sell, or Hold signals, including the stronger Buy/Sell variants in the backend response.
- Compare models on a chronological holdout and run a separate reproducible chronological evaluation.
- Run the original technical and production-hybrid walk-forward backtests with next-open execution, transaction fees, slippage, portfolio tracking, and baselines.
- Inspect causal market-regime estimates, validation-selected adaptive hybrid weights, calibration status, component disagreement, and No Trade decisions in historical evaluations.
- Compare fixed and adaptive strategies by fold on matching dates, alongside an aggregate continuous portfolio result.
- Save versioned fitted model artifacts and enough metadata to identify the data, preprocessing, configuration, software versions, and target used.
- Periodically refresh and persist daily bars and technical features in a lightweight background thread.
- Store cache, job, model, prediction, and experiment status for inspection through API routes.

### Main technologies

| Area | Technology |
|---|---|
| Browser UI | React, React Router, Vite, Recharts, Lucide icons, CSS |
| HTTP API | FastAPI, Uvicorn, Pydantic |
| Data handling | Python, pandas, NumPy, yfinance, requests |
| Models | TensorFlow/Keras LSTM, scikit-learn Random Forest, Hugging Face FinBERT |
| Optional comparison/explanation | XGBoost, SHAP |
| Hosted persistence | Supabase Postgres through its HTTPS REST API |

The project does not require installing a local database server. Supabase is optional. The backend's local JSON fallback works without it.

## 2. System architecture

The React application and FastAPI service run as separate processes. In local development, Vite serves the web app on port 5173 and forwards `/api` requests to FastAPI on port 8000. The browser never imports or runs the Python model code directly.

The diagram separates live prediction components from supporting analysis. LSTM, Random Forest, FinBERT, and deterministic technical rules feed the fixed live hybrid. XGBoost is a comparison model, while the Hidden Markov Model (HMM) provides a separate market-regime estimate; neither votes in the live hybrid. Adaptive weights, calibration, and No Trade are evaluation-only paths.

```mermaid
flowchart LR
    Person([User]) --> Browser

    subgraph UI[Browser application]
        Browser[React UI<br/>Dashboard · Analysis · Backtest · Models<br/>Tuning · Signals · Research · Settings]
        Local[(Browser localStorage<br/>selected ticker · theme · saved signals)]
        Browser <--> Local
    end

    Browser -->|HTTP / JSON: /api/...| Vite[Vite dev server<br/>port 5173 · /api proxy]
    Vite -->|HTTP| API[FastAPI service<br/>port 8000]

    subgraph DATA[Data retrieval and preparation]
        API --> History[yfinance<br/>daily OHLCV and company lookup]
        API --> News[NewsAPI if configured<br/>Yahoo Finance news fallback]
        History --> Features[pandas feature engineering<br/>clean, order, label, indicators]
        News --> HeadlineInput[Recent headline text]
    end

    subgraph MODELS[LIVE SIGNAL COMPONENTS]
        direction[Target: next session close<br/>above current close]
        Features --> LSTM[MODEL · LSTM<br/>time-ordered feature sequences<br/>up-direction probability]
        Features --> RF[MODEL · Random Forest<br/>tabular indicators and returns<br/>up-direction probability]
        HeadlineInput --> FinBERT[MODEL · FinBERT<br/>positive / negative / neutral<br/>headline sentiment score]
        Features --> Rules[TECHNICAL EVIDENCE<br/>moving-average alignment<br/>5-day momentum · RSI/MACD rules]
        Features --> XGB[OPTIONAL MODEL · XGBoost<br/>evaluation comparison only]
        LSTM --> Hybrid[VERSIONED HYBRID<br/>40% LSTM · 40% RF<br/>10% mapped sentiment · 10% technical]
        RF --> Hybrid
        FinBERT --> Hybrid
        Rules --> Hybrid
        direction -. defines .-> LSTM
        direction -. defines .-> RF
        direction -. defines .-> Hybrid
    end

    Hybrid --> Signal[Buy / Hold / Sell signal<br/>score · evidence · metadata]
    XGB -. separate comparison .-> Evaluation
    History --> HMM[ANALYSIS MODEL · Gaussian HMM<br/>latent daily market regime]
    HMM --> Regime[Regime label · state probabilities<br/>not a direction vote]
    subgraph RESEARCH[WALK-FORWARD EVALUATION ONLY]
        Evaluation[Chronological train / validation / test]
        Evaluation --> Adaptive[Adaptive weights<br/>validation Brier evidence]
        Evaluation --> Calibrate[Platt / isotonic calibration<br/>validation selection]
        Adaptive --> OOS[Out-of-sample signals]
        Calibrate --> NoTrade[No Trade threshold<br/>validation selected]
        NoTrade --> OOS
    end
    API --> Backtest[Backtest engine<br/>technical rules or hybrid OOS signals<br/>next-open fills · costs · portfolio metrics]
    Signal --> Browser
    Backtest --> Browser
    Regime --> Browser

    subgraph ASYNC[Background work]
        Scheduler[In-process scheduler<br/>startup refresh + configured interval]
        Scheduler --> History
        Scheduler --> Features
    end

    API --> Cache[(Bounded in-process TTL cache<br/>market data · features · company names)]
    Scheduler --> Storage
    API --> Storage[(Supabase REST / Postgres<br/>or local JSON fallback)]
    API --> Artifacts[(Local model_artifacts folder<br/>versioned fitted estimators and scalers)]
    Hybrid --> Artifacts
    Artifacts --> Hybrid

    classDef model fill:#173b67,stroke:#54a6ff,stroke-width:3px,color:#ffffff;
    classDef optional fill:#45315f,stroke:#c49aff,stroke-width:2px,color:#ffffff;
    classDef hybrid fill:#124d42,stroke:#39d6a0,stroke-width:4px,color:#ffffff;
    classDef service fill:#202a3b,stroke:#8292ad,stroke-width:1px,color:#f5f7fb;
    class LSTM,RF,FinBERT,HMM model;
    class XGB,Adaptive,Calibrate,NoTrade optional;
    class Hybrid hybrid;
    class API,Features,Rules,Signal,Backtest,Scheduler,History,News,HeadlineInput,Vite,Browser,Evaluation,Regime,OOS service;
```

### Main request flow: prediction

1. A user selects or types a ticker and chooses a forecast display horizon on Analysis.
2. React sends `GET /api/predict/{symbol}?timeframe=...` to FastAPI.
3. FastAPI normalizes the symbol and gets adjusted daily OHLCV history from Yahoo Finance through `yfinance`.
4. The service hashes the exact downloaded window and calculates indicators. It builds a next-session-up training target from adjacent closes.
5. It retrieves recent headlines from NewsAPI when configured and then Yahoo Finance sources as needed. FinBERT scores up to eight headlines; unavailable news/model output falls back to neutral sentiment.
6. LSTM and Random Forest independently produce probabilities for the same directional target. XGBoost may provide a separate holdout comparison. Technical rules produce a fourth evidence value.
7. `backend/hybrid.py` combines the four compatible inputs using the fixed versioned weights. Thresholds turn the combined score into a signal.
8. The backend creates the accompanying technical-strategy backtest payload, attaches data and model metadata, and persists the prediction when configured. This inline payload is not a hybrid-signal simulation.
9. React renders the result and adds the prediction to the browser's signal history.

```mermaid
BACKEND_PIPELINE_DIAGRAM
flowchart TB
    Request[React calls /api/predict/{symbol}]
    Request --> Market[Yahoo auto-adjusted daily OHLCV]
    Market --> Provenance[Normalize dates · last-bar as-of · retrieval time · SHA-256]
    Provenance --> Prepare[Indicators and adjacent-close labels<br/>latest row has no known label]
    Request --> News[NewsAPI if configured · Yahoo search / ticker-news fallback]
    News --> Gate[Timezone-aware cutoff · relevance · dedupe · up to 8 titles]
    Gate --> FinBERT[FinBERT positive minus negative<br/>mean title score]
    Prepare --> LSTM[LSTM sequence<br/>30 x 7 defaults · MinMaxScaler · P(up)]
    Prepare --> Forest[Random Forest tabular<br/>9 price / volume columns · P(up)]
    Prepare --> Rules[Deterministic technical score<br/>alignment · momentum · RSI / MACD]
    LSTM --> Hybrid[Fixed hybrid<br/>40% LSTM + 40% RF + 10% mapped news + 10% rules]
    Forest --> Hybrid
    FinBERT --> Hybrid
    Rules --> Hybrid
    Hybrid --> Signal[Thresholded signal + component evidence]
    Signal --> Response[JSON: target · data provenance · availability · model version]
    Response --> UI[Analysis / Models UI · browser signal history]
    Response --> Store[Supabase when configured · otherwise local JSON]
    LSTM -. matching estimator and scaler .-> Artifact[Local versioned artifact registry]
    Forest -. matching estimator .-> Artifact
    Market --> Regime[Separate causal Gaussian HMM<br/>return · volatility · range · volume change]
    Regime --> RegimeUI[Dashboard regime label / state probabilities]
    Prepare --> Eval[Offline expanding walk-forward evaluation<br/>final 20% holdout excluded from hybrid backtest]
    Eval --> XGB[XGBoost comparison only]
    Eval --> Adaptive[Adaptive weights from prior validation]
    Eval --> Calibration[Platt / isotonic calibration + No Trade]
    Eval --> Backtest[Next-open portfolio simulation<br/>fees · slippage · long-only · terminal liquidation]
    XGB --> Report[JSON / Markdown metrics, folds, audits, trades, hashes]
    Adaptive --> Report
    Calibration --> Report
    Backtest --> Report
    Report --> Store
```

The request pipeline and research path are intentionally shown separately. Live prediction uses the fixed hybrid and does not fit validation-selected calibration or No Trade thresholds. Walk-forward reports select adaptive weights/calibration from earlier validation rows, then score later test rows; the production hybrid backtest excludes the final chronological holdout. The Gaussian HMM supplies dashboard regime context, not a direction probability.

### Two distinct meanings of “saved”

- **Saved signals in the UI:** Stored in the current browser's `localStorage`. This is what the ticker dropdown and Signals page use. It does not require Supabase and is not synchronized between browsers.
- **Backend prediction/history records:** Saved through Supabase when configured, otherwise to backend JSON. These can be read from `/api/history/predictions`; the UI signal list does not currently import that backend history automatically.

## 3. User interface and workflows

The app uses React Router. All pages share a selected ticker state. The selected ticker is retained in browser local storage, so changing pages does not reset it.

### Shared ticker picker and company name

Dashboard, Analysis, Backtest, and Models share the same ticker control. It contains a text input with a datalist of Apple and tickers found in locally saved signals. The user can choose a suggestion or type another ticker. Typing is normalized to uppercase.

The company name appears beside the input, outside the ticker box. A short built-in map covers common symbols; other valid-looking symbols are looked up by `GET /api/company/{symbol}` after a short pause in typing. The backend caches successful names for one day and lookup misses for 30 minutes. If Yahoo cannot supply a name, the UI says the company name is unavailable. The lookup is a convenience and does not validate that a ticker has usable historical prices.

### Header and appearance

- The theme button switches between light and dark themes. The selected theme is saved in browser local storage.
- The small **Live** badge means the interface can refresh status/data periodically. It does **not** mean exchange-level real-time prices. Its tooltip explains that status refresh is periodic and prices use the latest available daily data.
- Layout, charts, cards, ticker fields, tables, and navigation are styled for desktop and smaller screens. The navigation collapses on mobile. The charts are lazy-loaded to reduce the initial UI bundle.

### Dashboard

Shows browser-saved signal counts, buy/sell counts, average saved confidence, and the data source. The market card lets the user pick a ticker, then loads approximately three months of daily history and displays a candlestick chart, latest available price, period change, indicators, and a directional price-momentum gauge.

Below the gauge, the **Latest Saved Signal** panel shows the most recently saved Buy/Hold/Sell-style result for that selected ticker, confidence when available, and its saved time. It does not infer a Buy or Sell from the gauge. If the browser has no saved signal for that ticker, it links to Analysis to generate one. This means a prediction saved on another browser or only in backend history will not appear in this panel.

Dashboard market history refreshes when the selected ticker changes and then periodically while the page is open. A separate status poll reads `/api/jobs/market-data`. The Market Regime panel shows the latest inferred label, filtered state probabilities, recent regime history, training window, inference cutoff, retrieval time, and explicit unavailable/error states. Those refreshes do not make the underlying daily price feed real-time.

### Analysis

Select a ticker, choose 1D, 1W, 1M, or 3M, and run Analyze. The result includes:

- signal, trend, latest close and market date;
- estimated direction score, confidence-style display, and a heuristic predicted-change figure;
- an evidence panel showing component probability, weight, and contribution;
- raw next-session direction score, calibration availability, component disagreement and the live No Trade control status/reason;
- the feature importance and available SHAP explanation;
- technical indicators and recent news/sentiment;
- an accompanying historical chart and risk/interpretation notes.

The underlying classifier target is always **next-session direction**. The selected horizon changes only the separate predicted-change heuristic; it does not train a 1-week or 1-month price model. Live calibrated probability and validation-selected No Trade thresholds are explicitly unavailable in the prediction response; the raw score remains an uncalibrated heuristic. The uncertainty value is component disagreement, not a confidence interval or chance of profit. Running a new analysis saves/replaces the same ticker+horizon item in browser history, retaining up to 100 signals.

### Backtest

The Backtest page can switch between the original technical momentum/RSI strategy and the production hybrid walk-forward backtest. The hybrid mode can select fixed or adaptive signals. It shows aggregate fixed/adaptive returns, calibration and No Trade summaries, audited adaptive weights, and per-fold fixed/adaptive returns and Sharpe on matching dates. Each per-fold comparison restarts capital and omits that fold's final unexecutable signal; those results are not additive to the continuous aggregate. The technical mode remains a separate rule-based baseline and is not a simulation of the live hybrid prediction signal. Both use explicit loading, unavailable, and error states.

### Models

Shows the model families and invokes a prediction to display model comparison data, chronological holdout dates, feature importance, and SHAP values if available. Random Forest and optional XGBoost holdout results are shown when returned. LSTM validation metadata is recorded in its saved artifact; the Models page's primary comparison report should not be mistaken for the dedicated `/api/evaluate` experiment.

### Tuning

Loads and saves backend settings for LSTM lookback, LSTM units, epochs, Random Forest tree count, transaction-cost percentage, and slippage percentage. The backend validates ranges and writes settings to `backend/tuning_config.json`. The prediction and backtest services read this shared process configuration.

### Signals

Lists prediction results saved by the current browser. The user can clear this browser's signal history. Clearing it does not delete backend/Supabase prediction records.

### Research and Settings

Research summarizes the workflow and methodology. Settings describes the local `/api` path and deployment CORS configuration; it does not expose or edit secrets.

## 4. Market data, news, and features

### Price history

`download_history` requests Yahoo Finance auto-adjusted daily OHLCV bars: open, high, low, close, volume. It normalizes dates, sorts/cleans data, and rejects empty or malformed results. Prediction normally uses a multi-year data window; dashboard history requests three months, while history/evaluation endpoints allow specified periods. Yahoo Finance is an external provider, not a guaranteed feed. U.S. symbols are usually plain tickers (for example `AAPL`); Yahoo uses suffix conventions for some markets (for example `RELIANCE.NS`). A symbol must be recognized by Yahoo and have usable data.

The prediction response distinguishes `market_timestamp`/`as_of` (date of last market bar) from `retrieved_at`/`prediction_timestamp` (when the backend fetched or produced the result). A recent retrieval time does not mean the underlying bar is from the current minute.

### Company-name lookup

`GET /api/company/{symbol}` calls yfinance ticker metadata and returns `symbol`, `company_name`, and `found`. It accepts a restricted ticker-character set and caches the result. Built-in frontend names can be displayed without calling the lookup endpoint. Company metadata is not a substitute for successful price-history validation.

### News collection and FinBERT input

If `NEWS_API_KEY` exists, the backend first tries the NewsAPI Everything endpoint. If fewer than eight headlines were collected or that provider fails, it can add Yahoo Finance search results and then yfinance ticker news. It filters empty/duplicate titles and returns at most eight headlines for sentiment scoring. Articles without an unambiguous timezone-aware publication timestamp, articles after the request cutoff, and results without a ticker/query relevance match are excluded. The response records publication timestamp, source reference, and alignment counts. Provider/network failures are tolerated; prediction continues with an empty or partial news list. This live search path is not a point-in-time historical archive.

The default sentiment model is `ProsusAI/finbert` from Hugging Face Transformers; `FINBERT_MODEL` can override it. The first use may download model weights into the local Transformers cache and consume disk, memory, and network bandwidth. FinBERT analyzes headline text and returns positive probability minus negative probability, producing a signed score approximately in [-1, +1]. No articles or model failure yields neutral sentiment (0). FinBERT is pretrained; it is not trained from this project's stock history.

### Technical features

Indicators are computed in pandas; TA-Lib is not required.

| Feature | Meaning |
|---|---|
| SMA 20 / SMA 50 | Rolling averages of adjusted daily closing prices. |
| RSI 14 | Rolling gain/loss momentum indicator. Extreme values are clues, not guaranteed reversals. |
| MACD / signal | EMA(12) minus EMA(26), with an EMA(9) signal line. |
| Bollinger bands | SMA(20) plus/minus two rolling standard deviations. |
| `return_1d`, `return_5d` | Change in close over one or five trading observations. |
| `sma20_gap`, `sma50_gap` | Close relative to the corresponding average. |
| `bb_position` | Close position relative to lower/upper Bollinger bands. |
| `volume_change` | Change in volume from the preceding observation. |
| `sentiment` | Current request's aggregate headline sentiment. |

Rolling indicators need a warm-up period. Rows with missing or infinite required feature values are excluded from model fitting.

### Backend data preparation, in execution order

The prediction path is orchestrated in `backend/main.py` by `build_prediction`. It normalizes the ticker, obtains daily bars, records retrieval time and the last bar date, hashes the raw input window, and computes or reuses technical features. It then gathers current headlines, scores them, creates next-session labels for historical rows, and invokes the LSTM and Random Forest paths. The latest feature row is inference input: its next-session target is unknown and is not used as a training label.

`download_history` asks yfinance for auto-adjusted daily bars and normalizes the date column to UTC before removing the timezone for internal calculations. The returned frame contains date, open, high, low, close, and volume. The input fingerprint is SHA-256 over those ordered bar values and is attached to the prediction so the exact source window can be identified later. `indicators` derives rolling and exponential calculations from close and volume. Early rows naturally lack some rolling values; the model paths remove rows missing required inputs. Infinite percentage changes are converted to missing values rather than passed to estimators.

The label is aligned to the row that makes the decision: `target[t] = 1` if `close[t+1] > close[t]`; otherwise it is 0 (down or unchanged). Thus the feature vector at row *t* contains information available through that close, while its label refers to the next trading observation. This is why the most recent row can be scored for inference but cannot contribute a known label. The live backend uses the current timestamp-filtered headline aggregate as a request-level sentiment value; it does not reconstruct historical news for every training row. See the historical news assessment for provider/terms assessment and unknown historical coverage.

There are two model input matrices because the model families consume different representations:

- **LSTM sequence input:** ordered `close`, `sma20`, `sma50`, `rsi`, `macd`, `bb_upper`, `bb_lower`; each sample is a sliding window of the most recent `lookback` rows. `MinMaxScaler` transforms those columns. The latest sequence produces one sigmoid value in [0, 1].
- **Random Forest tabular input:** `return_1d`, `return_5d`, `sma20_gap`, `sma50_gap`, `rsi`, `macd`, `macd_signal`, `bb_position`, and `volume_change`. It consumes one row at a time and does not scale the values. Although `sentiment` exists in the general feature frame, it is deliberately excluded from the forest input; sentiment has its own FinBERT component in the hybrid.

After these model outputs are ready, the backend computes explicit technical evidence from the latest row, maps it to a 0–1 value, and sends the four compatible values into `backend/hybrid.py`. The API response preserves the raw market dates, component values, weights, weighted contributions, signal rule, feature importance, optional SHAP values, model comparison, and model-version ID. React formats these fields; it does not train or alter the models.

## 5. Prediction models and the hybrid signal

### Target definition

The classifier target is 1 when the next trading session's adjusted close is above the current session's adjusted close, otherwise 0. The decision convention is after the current session close; the outcome is measured at the next session close. It is a **direction** target, not a future dollar-price prediction.

### Model roles

| Component | Role in the live hybrid | Inputs / output |
|---|---|---|
| **LSTM** | 40% of the hybrid | Sequence of recent price/indicator rows; probability of next-session up direction. |
| **Random Forest** | 40% of the hybrid | Tabular returns, gaps, RSI, MACD, Bollinger position, volume change; probability of up direction. |
| **FinBERT** | 10% of the hybrid after mapping | Recent headline text; positive-minus-negative sentiment, mapped to a 0–1 proxy. |
| **Technical rules** | 10% of the hybrid | Moving-average alignment, 5-day return threshold, and RSI/MACD conditions; mapped to a 0–1 score. |
| **XGBoost** | Not in the hybrid | Optional separate model comparison and probability; its result does not vote in the final signal. |

This table describes the **live fixed-weight signal**. Other model-like methods are used for regime description or offline evaluation; they do not silently become extra live predictors. The complete inventory and the provenance of each reported result are below.

### Complete model and method inventory

| Model or method | What it does and how it works | Where its output is used |
|---|---|---|
| LSTM direction classifier | TensorFlow/Keras sequence classifier over seven ordered price/indicator columns. The scaler transforms sliding lookback windows; a sigmoid head returns next-session `P(up)`. Live defaults are 30 rows, 48 recurrent units, dropout 0.2, a 16-unit dense layer, Adam/binary cross-entropy, and up to six epochs with early stopping. Walk-forward evaluation refits within each chronological fold and fits scaling on training rows. | 40% of the live fixed hybrid when trained; an explicit 0.5 neutral fallback is used and recorded if unavailable/insufficient. Also evaluated as a standalone strategy/model. |
| Random Forest direction classifier | scikit-learn `RandomForestClassifier` over nine tabular returns/indicator/volume features, with no scaling. Defaults: 250 trees, depth 7, minimum leaf size 4, balanced class weights, seed 42. Live fitting uses a chronological comparison split, then refits on labeled rows for inference. | 40% of the live fixed hybrid; standalone metrics/strategy; impurity feature importance. |
| FinBERT sentiment classifier | Pretrained `ProsusAI/finbert` (or `FINBERT_MODEL`) classifies up to eight currently retrieved, cutoff-filtered headlines. Per title the signed score is `positive - negative`; title scores are averaged. It is not trained on this repository's prices. | Live hybrid maps the signed score from approximately [-1, 1] to [0, 1] and gives it 10%. Missing news/model failure maps to neutral 0. Historical research cannot recreate this signal and uses neutral sentiment. |
| Deterministic technical evidence | Hand-authored moving-average alignment, five-session momentum, and RSI/MACD conditions add/subtract fixed points, clipped and mapped to [0, 1]. It is not a fitted machine-learning estimator. | 10% of the fixed live hybrid and returned as auditable evidence. The technical historical backtest uses its own separately documented signal rules. |
| Optional XGBoost classifier | `XGBClassifier` tabular direction comparison with 180 estimators, depth 4, learning rate 0.05, row/column subsampling 0.85, log-loss, and seed 42 in the live comparison path. Evaluation enables it only when requested/configured. Failures are tolerated. | Separate probability/holdout or walk-forward comparison. Never part of live hybrid weights. |
| Gaussian Hidden Markov Model | Diagonal Gaussian HMM over daily return, 20-session volatility, normalized high-low range, and volume change. `pomegranate==1.1.2` with PyTorch fits two- and three-state candidates; a training-only scaler and state descriptions are selected using a later validation window. Filtered-prefix inference ends at each displayed date. | `/api/regime` and dashboard Market Regime context; descriptive walk-forward stratification and optional adaptive-weight grouping. It estimates latent state, not up/down direction or profit probability. |
| Previous-direction baseline | Predicts the next direction from the prior observed direction, providing a simple persistence reference; no learned parameters. | Chronological direction evaluation only. |
| Buy-and-hold and cash baselines | Buy-and-hold invests under the same next-open/cost conventions; cash remains uninvested. These are portfolio references, not classifiers. | Direction-report trading summaries and portfolio backtests. |
| Adaptive ensemble weighting | A validation selection procedure, not a new trained model: scores available LSTM, forest, and technical components by Brier evidence, optionally within the current training-derived regime, adjusts available fixed priors, then projects weights onto configured lower/upper bounds. Historical FinBERT is unavailable, so its prior is omitted and remaining weights are renormalized. | Walk-forward experiment only. Each fold's chosen weights are frozen before its test fold; live prediction remains fixed-weight. |
| Platt and isotonic calibration | Two post-processing methods map raw direction scores to calibrated direction estimates. Candidate choice uses a chronological split of the preceding validation window and Brier score; the chosen calibrator is refit on past validation rows. Isotonic requires more samples/class diversity. | Walk-forward calibration metrics and calibrated No Trade experiment only; not applied by live `/api/predict`. Calibration concerns direction frequency, not chance of profit. |
| No Trade policy | Validation-selected calibrated-neutral-band and/or component-dispersion cutoffs; component dispersion is standard deviation across available component probabilities. A No Trade decision is implemented as HOLD. | Walk-forward experiment only. It retains an existing holding and does not enter a new position; live API returns unavailable until a validation-fitted threshold exists. |
| SHAP explanation | Optional model explanation applied to the latest Random Forest input when package/model support succeeds; it attributes the fitted model output to input features. | Analysis/Models display when available. It is not an additional predictor or causal explanation. |

### How project results are created

The application produces several different kinds of “result,” and they have different data paths:

1. **Live signal:** `GET /api/predict/{symbol}` downloads/caches adjusted daily bars, builds causal indicators and known adjacent-close labels, retrieves current headlines, runs the available live model components, applies the fixed weighted equation and signal thresholds, and returns the latest result with as-of/retrieval times, data hash, component contributions, and availability status. The latest bar can be scored but has no known next-close label. Backend persistence is attempted separately; the browser also saves its own signal-history copy.
2. **Regime display:** `GET /api/regime/{symbol}` builds causal regime features, fits/chooses an HMM using a training window plus later validation window, then filters the state posterior using data only through the displayed inference cutoff. It returns state labels/probabilities and provenance separately from the live direction signal.
3. **Direction metrics:** `GET /api/evaluate/{symbol}` or `tools/run_real_evaluation.py` partitions data chronologically, builds expanding development folds and a final 20% holdout, fits model transforms on permitted training data, predicts common dates, and only then compares predictions with labels to calculate accuracy, balanced accuracy, ROC-AUC, Brier, reliability, confusion counts, and fold summaries. Optional XGBoost is included only when enabled. Historical sentiment is neutral.
4. **Portfolio metrics:** the technical backtest creates rule signals after close; the hybrid walk-forward backtest creates model signals on development test folds. Both pass signals and raw bars to `backend/backtesting.py`, which executes at the next available open, applies transaction cost and adverse slippage, tracks long-only cash/shares/equity, and liquidates at the end. Buy-and-hold/cash baselines use the same interval/cost convention. Final-holdout rows are not traded by the hybrid report.
5. **Adaptive/calibration/No Trade experiments:** for each upcoming test fold, component weights and calibrator/threshold choices are derived only from earlier validation observations, recorded with their selection scope and fallback status, then frozen for test scoring. Metrics are calculated on those later test rows. These experiment outputs do not change live prediction behavior.
6. **Final summary and figures:** `tools/build_phase9_report.py` reads retained experiment JSON, validates that the paired E4 runs share the same input hash/dates and that the retained CSV matches its recorded fingerprint, then assembles E1–E7 tables, Markdown/JSON audit, confusion matrices, reliability plot, and cost-adjusted equity curves. E5 is reported as unavailable rather than fabricated because an eligible point-in-time news archive was absent.

Every metric is therefore conditional on its data window, ticker set, model availability, fold plan, and cost assumptions. The report files and machine-readable audits linked in this guide are the source for the displayed final report values; the dashboard does not generate those multi-fold research summary figures at runtime.

### LSTM

TensorFlow/Keras builds a binary sequence classifier. The live model uses the ordered columns `close`, `sma20`, `sma50`, `rsi`, `macd`, `bb_upper`, and `bb_lower`; a MinMaxScaler transforms them. Defaults are 30 observations of lookback, 48 LSTM units, 0.2 dropout, a 16-unit dense layer, sigmoid output, Adam, binary cross-entropy, batch size 32, and up to 6 epochs with early stopping. The random seed is 42.

The scaler is saved with the fitted LSTM artifact. In the live training path it is fit on the available feature window before the sequence validation split; therefore the reported validation behavior is not a fully untouched chronological test. If TensorFlow is missing/incompatible or there are too few sequences, LSTM probability falls back to 0.5 and metadata records why.

Each sample is one `lookback × 7` array and its label is the following close-direction value. The backend requires at least 80 sequences. It uses the earlier portion for fitting and a trailing validation slice for early stopping; the split point is `max(floor(85% of samples), sample_count - 40)`, so validation is at least roughly 15% until the 40-row cap applies. It restores the best validation weights. A saved matching LSTM and scaler bypass refitting and transform only the latest sequence for inference. A neutral 0.5 fallback is an explicit availability path, not a learned result.

### Random Forest

The forest is a scikit-learn `RandomForestClassifier` with a default of 250 trees, maximum depth 7, minimum leaf size 4, balanced class weights, and seed 42. It uses `return_1d`, `return_5d`, moving-average gaps, RSI, MACD and signal, Bollinger position, and volume change. No scaling is applied. With enough labeled data, a trailing chronological holdout is scored first; then a separate forest is refit on all available labeled rows for the current inference. With fewer than 100 usable rows, the component returns 0.5 and reports insufficient samples.

The live holdout is the latest 20% where possible, with at least 20 test rows and at least 80 earlier fitting rows. The holdout accuracy is descriptive metadata; the inference model is a newly fitted forest using all rows whose next-session label is known. The latest unlabeled row is held aside for the actual score. Feature importance is the forest's impurity-based feature importance, which describes split usage in this fitted forest and is not a causal explanation.

### Optional XGBoost and SHAP

When XGBoost imports and fits successfully, the backend scores it on the Random Forest comparison window, then refits it for an up probability. The comparison uses 180 estimators, depth 4, learning rate 0.05, row/column subsampling of 0.85, log-loss, and seed 42. XGBoost is guarded as optional; failures do not stop the main prediction. It is not included in hybrid weights. SHAP is also attempted for the latest Random Forest input; when unavailable or failing, its explanation is omitted/empty. Random Forest feature importance is separately returned.

FinBERT is different from the locally fitted estimators: it is a pretrained text classifier loaded lazily through Transformers. The backend asks it for positive, negative, and neutral label scores for each of up to eight titles; for each title it computes positive minus negative, then averages the title scores. Its neutral label is not added to the formula. No headlines or an import/inference error produces score 0 and allows price-based components to proceed. Headlines and sentiment accompany the API response; they are not persisted as a separate training corpus.

### Explicit hybrid rule

`backend/hybrid.py` owns the target definition, hybrid version (`weighted-probability-v1`), and fixed weights:

```text
hybrid up score =
    0.40 × LSTM probability-up
  + 0.40 × Random Forest probability-up
  + 0.10 × ((FinBERT score + 1) / 2)
  + 0.10 × technical probability-up
```

The sentiment mapping is a simple, documented proxy, not a calibrated probability. Technical rules add/subtract fixed points, clip the result to [-1, +1], then map it to [0, 1]. The score and its component inputs, weights, and contributions are returned for inspection.

In the technical component, bullish alignment (`close > SMA20 > SMA50`) adds 0.45; bearish alignment subtracts 0.45. A five-day return above +2% adds 0.25 and below -2% subtracts 0.25. RSI below 35 with MACD above its signal adds 0.30; RSI above 65 with MACD below its signal subtracts 0.30. The sum is clipped to [-1, 1], then converted to a probability-like value using `(score + 1) / 2`. These are hand-authored rules and fixed point sizes, not fitted coefficients.

| Combined score | Signal |
|---:|---|
| 0.64 or higher | Strong Buy |
| 0.54 to below 0.64 | Buy |
| Above 0.46 to below 0.54 | Hold |
| Above 0.36 through 0.46 | Sell |
| 0.36 or lower | Strong Sell |

The dashboard's saved-signal card presents the simpler BUY/HOLD/SELL wording and does not derive a signal from the price-momentum gauge. Backend prediction details may include Strong Buy or Strong Sell.

### Confidence and predicted change

The displayed confidence-like number is derived from distance of the combined score from 0.5. It is not a calibrated probability, confidence interval, or percentage chance of profit. The `predicted_change` figure is a separate heuristic based on the combined score, recent 5-day return, and square-root horizon scaling. It is not an output of a regression model and is not a price target. Selecting 1W/1M/3M does not change the model's next-session classification target.

### What the prediction output contains

The `GET /api/predict/{symbol}` response is a structured record rather than a single label. Its main fields have distinct purposes:

| Output | How the backend creates it | How the UI uses it |
|---|---|---|
| `stock_symbol`, `company_name` | Normalized ticker; `company_name` in this prediction payload is initially the ticker placeholder. The frontend resolves a display name separately through `/api/company/{symbol}`. | Labels the selected security, using the separate lookup when available. |
| `market_data`, `as_of`, `latest_close` | Source, interval, last market timestamp, retrieval time, date range, row count, and SHA-256 of the exact bar window. | Shows what data the result is based on and its freshness. |
| `lstm_probability_up`, `classifier_probability_up` | Latest sequence score and latest-row forest score. | Supports component comparison and evidence. |
| `sentiment_score`, `news` | FinBERT signed title scores and up to eight scored titles. | Displays current headline evidence and sentiment. |
| `signal_explanation` | Hybrid score, target, version, weights, each component's value and contribution, and technical factors. | Powers the evidence panel and makes the rule auditable. |
| `probability_quality` | Raw direction score, calibrated direction probability/status, uncertainty summary, raw trading decision/status, and No Trade availability/decision/reason. | Separates an uncalibrated live score from calibration and uncertainty; live calibration and abstention are explicitly unavailable until a leakage-safe validation threshold is fitted. |
| `news_alignment` | Request cutoff and counts for accepted, later-than-cutoff, and missing/ambiguous-timestamp articles. | Explains which current articles were included; it does not claim historical-news coverage. |
| `signal_type`, `predicted_trend`, `confidence` | Thresholded hybrid score and distance-from-neutral display. | Headline signal and confidence-style presentation. |
| `predicted_change` | Separate horizon-scaled heuristic, not a price-regression model. | Displays an estimated directional change figure with interpretation limits. |
| `technical_indicators`, `feature_importance`, `model_comparison` | Latest indicators, forest importances, holdout comparison and optional SHAP. | Feeds indicator cards and Models/Analysis explanation panels. |
| `model_version` | Deterministic identity for ticker, exact data window, code/configuration, features, and installed package set. | Identifies which saved fitted artifact matched this prediction. |
| `backtest`, `risk_factors`, `analysis_summary` | Separate technical strategy report and explanatory text. | Adds historical context and limitations; backtest is not the hybrid's simulated return. |

Once model construction and inference complete, the API tries to save the result. A storage write failure is returned as a `persistence.status=failed` field while the computed prediction is still returned. This lets the UI show a result while making its save status explicit.

## 6. Saved model artifacts and versioning

`backend/model_registry.py` writes fitted model artifacts under `backend/model_artifacts/` by default (or `DEEPSIGNAL_ARTIFACT_DIR` if configured). It stores a deterministic version identity based on ticker, exact market-data SHA-256 fingerprint and date window, target, ordered feature lists, preprocessing/feature/model-code versions, hybrid version, tuning configuration, sentiment model name, random seeds, and installed package versions.

Each artifact directory contains JSON metadata plus whichever fitted files were actually captured: a Random Forest estimator, optional XGBoost estimator, LSTM Keras model, and LSTM scaler. The metadata records data window/hash, target, feature order and input shape, hyperparameters, package versions, validation status/metrics, model statuses, and inference explanation metadata. The directory is written through a temporary path and atomically renamed to avoid a partially written version. Loading verifies the metadata identity, expected files, model/scaler dimensions, and Random Forest feature order before putting the bundle into the process cache. The version is an identity hash, not a claim that model quality improved.

The full identity includes the complete price-window hash and configuration. Since the hash changes when the downloaded window changes, matching is exact and a new version is usually created after new market bars arrive. FinBERT's pretrained weights are managed by the Hugging Face cache rather than this registry. The registry is for reproducibility and reuse on the same machine; metadata in Supabase does not upload the binary files.

When the same complete identity is requested again, the backend can load the saved model, scaler, and metadata instead of fitting again. If the market data window or identity changes, a different version is created. Saved LSTM, Random Forest, and available XGBoost estimators, the LSTM scaler, and optional Keras model are written as an artifact bundle with JSON metadata; the loader checks identifiers, expected files, input shape, and feature order. A bounded in-process loaded-model cache holds recent versions. Older artifacts are retained for rollback; they are local files, not automatically uploaded to Supabase.

Useful routes:

- `GET /api/models` — list model-version metadata.
- `GET /api/models/{version}` — inspect one version.
- `POST /api/models/{version}/load` — validate and load it into the current backend process.

Loading an artifact does not force its use for a different ticker/data/configuration identity. The pretrained FinBERT weights remain in the Hugging Face cache and are identified by model name; they are not copied into every trained-model artifact. The backtest is rule-based and carries strategy/engine versions instead of a fitted model version.

## 7. Evaluation and backtesting

### Dedicated direction evaluation

`GET /api/evaluate/{symbol}` downloads up to three years of daily history and builds a chronological train/validation/test report for next-session direction. It needs at least 120 usable samples and both classes in fitting data; otherwise it returns an explicit insufficient-data status.

- First 60%: initial training window; the LSTM scaler is fit only on these rows.
- Rows through the first 80% are available to fit/evaluate models before the final holdout. The Random Forest is fit on those first 80%; the LSTM uses an earlier fit portion and an internal validation portion for early stopping.
- Final 20%: shared holdout test window for all compared methods.
- Compares previous-session-direction baseline, Random Forest, LSTM, and weighted hybrid.
- Reports accuracy, precision/recall/F1 for up class, confusion matrix, and simplified trading metrics.
- Reports buy-and-hold and flat-cash baselines and hybrid deltas versus component models.
- Records the data hash, dates, parameters, model/package versions, and assumptions.

Evaluation fits the LSTM scaler using only the earlier 60% training rows. Historical point-in-time news is not available, so evaluation explicitly uses neutral sentiment on all historical hybrid test rows. This means it does **not** validate the live news/FinBERT contribution. The models share final test dates, but this is one final holdout, not repeated rolling retraining of every component.

The retained baseline snapshot in [the baseline report](../research/baseline/PHASE_0_BASELINE.md) is historical evidence from existing local artifacts dated 2026-10-24, not a rerun of today's evaluator. Its saved evaluation artifact contains only previous-direction and Random Forest results; it has no LSTM or hybrid metrics. Its saved technical-backtest record also has a legacy summary field named `hybrid` that duplicates the technical strategy. Use the explicitly named production-hybrid walk-forward endpoint and supporting evaluation reports when referring to actual hybrid results. No frontend lint or type-check command is configured; the documented build command is available for production compilation.

### Production-hybrid walk-forward evaluation

`GET /api/backtest/{symbol}/hybrid?mode=fixed` and `GET /api/backtest/{symbol}/hybrid?mode=adaptive` run the actual versioned hybrid and component signal paths on expanding walk-forward development folds. Training precedes validation, validation precedes each test fold, and the final chronological holdout is excluded. Signals are formed after a session close and executed at the next available session open. The engine charges the configured transaction fee and adverse slippage, tracks long-only positions, and liquidates remaining holdings at the end of the evaluated window.

The report compares fixed hybrid, adaptive hybrid, LSTM-only, Random-Forest-only, calibrated No Trade, buy-and-hold, and cash on matching out-of-sample dates. Each run includes a full source-data SHA-256, scored date range, model/strategy/engine versions, configuration, folds, assumptions, signals, weights, component availability, regime context, calibration details, and trade records. The fixed mode remains the default; the route accepts `mode=fixed` or `mode=adaptive` without changing the older technical endpoint.

The per-fold fixed/adaptive panel reruns each fold with fresh starting capital and excludes that fold's last signal because no following in-fold open exists. Aggregate metrics use the continuous OOS portfolio. Per-fold returns therefore describe fold behavior and cannot be summed to reconstruct aggregate returns.

### Adaptive weights, calibration, uncertainty, and No Trade

Adaptive weights are chosen using only the preceding fold's validation evidence. Available LSTM, Random Forest, and technical probabilities are scored by Brier skill relative to a 0.50 direction score; renormalized fixed priors are adjusted from the scores and projected onto the configured bounded simplex. Default min/max component weights are 0.05/0.80. With insufficient regime-specific evidence (default minimum 20 rows), the method records a fallback to the overall preceding validation window. Historical FinBERT is unavailable, so adaptive evaluation omits it and renormalizes the other components. The report logs selected scope, metrics, availability, fallback reason, and weights for each test signal.

Calibration compares Platt and isotonic candidates using a chronological split of the preceding validation window. It selects by validation Brier score and refits the selected method using that past validation window before evaluating the next fold. Isotonic is marked unavailable when sample/class-diversity requirements are not met. Calibration is for next-session direction, never profit probability. Component standard deviation is the uncertainty proxy; it measures disagreement and is not a confidence interval.

The experimental No Trade filter uses validation-selected neutral-band and component-dispersion thresholds, frozen before the next test fold. No Trade acts as HOLD: it does not open a new position or force an exit from an existing one. In the live prediction route no validation-selected calibrator or abstention threshold is fitted, so the API returns an explicit unavailable status and reason rather than implying that live No Trade was applied.

### Historical-news alignment gate

Live provider articles must have a timezone-aware publication timestamp at or before the request cutoff and a ticker/query relevance match. Duplicates are removed using canonical article references. Historical alignment helpers use the NYSE 16:00 `America/New_York` close with daylight-saving handling and report coverage only when an eligible corpus is supplied. NewsAPI account-plan/history rights were not verified, Yahoo search is not a historical archive, and no local licensed corpus was present. The implementation therefore applies strict live filtering and documented unknown historical coverage; it did not run a sentiment ablation or claim FinBERT's historical value. See [the historical news assessment](../research/real_data/phase6_news_alignment.md).

### Hidden Markov market regime

`GET /api/regime/{symbol}` fits a diagonal Gaussian HMM over daily return, 20-session rolling volatility, normalized high-low range, and volume change using `pomegranate==1.1.2` and the existing PyTorch dependency. Feature calculations are causal through each daily close. The scaler and state label mapping are fit from the training window only; two- and three-state models are fit on training rows and selected using a later validation window's penalized mean log likelihood. The latest posterior is filtered using prefixes ending at each displayed date, so it does not use later observations. State IDs are local to a fitted model and must not be compared across folds.

The response provides the descriptive regime label, state probabilities, training/validation windows, model version, selected state count and scores, data fingerprint, and recent history. Labels use training-window average return and volatility to describe higher-return, lower-return/high-volatility or lower-volatility, and remaining states; they are not ground truth, calibrated return probabilities, or profit probabilities. If data are short or invalid, fitting fails to converge, or `pomegranate` is unavailable, the API returns an explicit unavailable status and fallback reason. The dashboard distinguishes the inference cutoff from retrieval time and labels the data as daily.

`tools/run_regime_evaluation.py` reports the same fixed-hybrid walk-forward predictions pooled and grouped by their causal inferred regimes; the final holdout is excluded. This is descriptive stratification, not a causal ablation or evidence that regime information improves predictions. See [the regime evaluation report](../research/real_data/phase3_regime_evaluation.md) and its fold-level [JSON audit](../research/real_data/phase3_regime_evaluation.json).

### Rule-based historical backtest

The Backtest page and `GET /api/backtest/{symbol}` use a separate long-only technical strategy. Signals are created after each daily close using:

- BUY when 5-day return is above 0.6% and RSI is below 72;
- SELL when 5-day return is below -0.6% and RSI is above 28;
- otherwise HOLD.

Signals execute at the next available daily open, which prevents using a future price for the order. Position size targets the configured fraction of equity (default 100%); fractional shares are allowed. Buys/sells include adverse slippage and a percentage transaction cost. The engine tracks cash, holdings, equity, trades, drawdown, exposure, turnover, return, CAGR, and annualized Sharpe. It liquidates any remaining position at the final close with costs.

The report compares the strategy with buy-and-hold (including modeled costs) and no-trade cash. It contains the price-data hash, signal hash, date window, strategy and engine versions, parameters, and execution assumptions. It sets `model_version` to null because it is not a trained-model strategy. The live prediction endpoint also includes a rule-based backtest payload for presentation, but that payload is not a simulation of the prediction ensemble.

### Metrics: what they do and do not say

- **Accuracy:** fraction of direction labels classified correctly; can hide class imbalance.
- **Precision / recall / F1:** focus on the positive/up class and its false alarms/misses.
- **Confusion matrix:** counts actual versus predicted up/down labels.
- **Cumulative return / CAGR:** historical simulated account growth over a selected window.
- **Maximum drawdown:** largest peak-to-trough loss in the reported equity curve.
- **Sharpe:** annualized average return relative to variability under the report's assumptions.
- **Turnover / trade count / exposure:** describe trading activity and time invested.

Metrics depend on provider data, window, thresholds, costs, and implementation assumptions. Past simulation is not a promise of future results.

### Completed implementation and research evidence

The planned implementation and research work is complete, with one conditional exception: E5 was not run because an eligible point-in-time historical news archive was unavailable.

| Work completed | Main evidence |
|---|---|
| Repository inspection and code-path map (now incorporated in Sections 2 and 11), preserved baseline, safe branch checkpoint, and baseline suite/build. | [System architecture](#2-system-architecture), [project layout](#11-project-layout), [baseline report](../research/baseline/PHASE_0_BASELINE.md) |
| Shared chronological direction evaluation with previous-direction, RF, LSTM, hybrid, fold audits, and separate final holdout. | [Walk-forward report](../research/real_data/PHASE_1_REAL_DATA_EVALUATION.md) |
| Production-hybrid walk-forward signal backtest with next-open execution, costs, model components, buy-and-hold, and cash baselines. | [Hybrid backtest](../research/real_data/PHASE_2_HYBRID_BACKTEST.md) |
| Causal Gaussian-HMM regime fitting, train-window state mapping, and descriptive regime-stratified evaluation. | [Regime report](../research/real_data/phase3_regime_evaluation.md) |
| Fixed versus validation-selected adaptive hybrid weights on shared dates. | [Adaptive report](../research/real_data/phase4_adaptive_evaluation.md) |
| Validation-only Platt/isotonic calibration and No Trade threshold evaluation. | [Calibration report](../research/real_data/phase5_calibration_evaluation.md) |
| Strict live article cutoff/relevance/deduplication and a historical-news availability assessment; no sentiment ablation. | [News report](../research/real_data/phase6_news_alignment.md) |
| Optional XGBoost against baseline models across five symbols and four development folds. | [XGBoost report](../research/real_data/phase7_xgboost_evaluation.md) |
| Compatible API/persistence fields, explicit live calibration/No Trade availability, regime/reliability/contribution displays, adaptive audit, and per-fold comparisons. | [Implementation log](../CODEX_PROGRESS.md#phase-8-api-persistence-and-frontend-integration) |
| Consolidated E1–E7 matrix, exact-data regime ablation, figures, and limitations. | [Final Markdown report](../research/real_data/phase9_final_experiment_matrix.md), [JSON report](../research/real_data/phase9_final_experiment_matrix.json) |

The E1 walk-forward mean metrics below are equal-ticker means across AAPL, MSFT, JPM, JNJ, and XOM, each with four expanding folds. The later 20% holdout is excluded from these fold means. The final-holdout dates from the initial evaluation were reused in the XGBoost comparison, so holdout results are exploratory, not fresh confirmation.

| E1 model | Accuracy | Balanced accuracy | ROC-AUC | Brier score |
|---|---:|---:|---:|---:|
| Previous direction | 0.5229 | 0.5117 | 0.5117 | 0.4771 |
| Random Forest | 0.5067 | 0.5114 | 0.5176 | 0.2607 |
| LSTM | 0.5236 | 0.4902 | 0.4806 | 0.2513 |
| XGBoost | 0.5333 | 0.5330 | 0.5179 | 0.2906 |
| Fixed hybrid | 0.5165 | 0.4982 | 0.5092 | 0.2509 |

XGBoost led mean accuracy and balanced accuracy, while its Brier score was worse than the other learned models and hybrid. These metrics are descriptive and do not establish that XGBoost is generally better.

The runnable AAPL arms among E2–E7 use the same retained OHLCV frame (normalized frame SHA-256 `94bf5ba4720da5dbbf597317ee44ac0f143528f7fb0f9c13eaf79f19418c33e3`), 2025-05-16 through 2026-04-09 OOS window, four fold boundaries, target, and cost assumptions. E5 is unavailable and therefore has no comparison data. This remains a single ticker/window and is not a future-profitability estimate.

| Experiment | Comparison and observed result | Interpretation |
|---|---|---|
| E2 — Ensemble effect | LSTM-only returned 25.42%, Sharpe 1.2518, max drawdown −13.7985%; fixed hybrid returned 25.1038%, Sharpe 1.2878, drawdown −16.1417%. | Fixed hybrid had slightly higher Sharpe but lower return and larger drawdown than LSTM. |
| E3 — Adaptive fusion | Fixed hybrid: 25.1038% return, 1.2878 Sharpe, −16.1417% drawdown. Adaptive: 14.2802%, 0.8238, −18.7356%. | Adaptive weighting underperformed on return and Sharpe and had a larger drawdown in this sample. See the report for reset-capital per-fold results. |
| E4 — Regime effect | Regime-specific vs overall-validation adaptive weights: both returned 14.2802%, Sharpe 0.8238, drawdown −18.7356%, using identical data and dates. One audit label differed (2025-08-05 BUY vs HOLD); executed equity curves were identical. | No measurable portfolio effect in this one-symbol sample. Regime-stratified metrics alone are descriptive, not a causal test. |
| E5 — Sentiment ablation | Not run: no eligible point-in-time archive was available. Current headlines were not substituted into historical dates. | Historical FinBERT value remains unknown; the report makes no sentiment benefit claim. |
| E6 — Uncertainty / No Trade | Raw fixed Brier 0.2532; calibrated Brier 0.2676. No Trade abstained on 37/225 rows and returned 10.1851%, Sharpe 0.7945, drawdown −13.7985%. | Calibration worsened Brier. No Trade reduced drawdown but also reduced return and Sharpe. It is not enabled for live prediction. |
| E7 — Trading validation | Fixed hybrid: 25.1038% return; buy-and-hold: 22.7918%; cash: 0%. LSTM-only returned 25.42%. Fixed hybrid drawdown was −16.1417%, versus −13.7985% for buy-and-hold. | Historical simulated outcomes after modeled costs; hybrid return exceeded buy-and-hold here with a larger drawdown. |

The final report and linked machine-readable source artifacts provide pooled walk-forward confusion matrices, a reliability diagram, an AAPL cost-adjusted equity curve, per-ticker/fold metrics, version/run IDs, hashes, detailed fold audits, trades, assumptions, and future work:

- [Confusion matrices](../research/real_data/phase9_confusion_matrices.svg)
- [Reliability diagram](../research/real_data/phase9_calibration.svg)
- [AAPL OOS equity curves](../research/real_data/phase9_equity_curves.svg)
- [Retained E4 input CSV](../research/real_data/phase9_e4_aapl_prices.csv)

The comparison assumptions are 0.06% transaction fee, 0.04% adverse slippage, long-only fractional sizing, next-available-open execution, and terminal liquidation. The continuous aggregate portfolio starts with shared capital; per-fold comparisons reset capital and are not additive to that aggregate. Yahoo bars may be revised, and TensorFlow CPU runs can vary slightly between runs. The retained E4 CSV and hash preserve the exact paired input used for E2–E7.

### Regenerate research reports

From the repository root in PowerShell:

```powershell
# Re-run the five-ticker base-model benchmark (provider data may be revised).
.\.venv\Scripts\python.exe tools\run_real_evaluation.py --symbols AAPL,MSFT,JPM,JNJ,XOM --period 3y --include-xgboost --output research\real_data\phase7_xgboost_evaluation.json

# Re-run the E4 regime-enabled arm against the retained exact input.
.\.venv\Scripts\python.exe tools\run_adaptive_evaluation.py --symbol AAPL --input-csv research\real_data\phase9_e4_aapl_prices.csv --output research\real_data\phase9_e4_with_regime.json

# Run the paired overall-validation control against the same input.
.\.venv\Scripts\python.exe tools\run_adaptive_evaluation.py --symbol AAPL --input-csv research\real_data\phase9_e4_aapl_prices.csv --without-regime-weights --output research\real_data\phase9_e4_without_regime.json

# Rebuild the final matrix, JSON audit, and SVG figures from the saved reports.
.\.venv\Scripts\python.exe tools\build_phase9_report.py
```

`build_phase9_report.py` checks that the paired reports share data hashes and out-of-sample dates and that the retained CSV still matches the recorded normalized data hash before it writes the final report. Re-running model experiments can produce small floating-point changes; it does not freeze Yahoo's upstream data history.

## 8. Caching and scheduled market-data updates

### In-process cache

`backend/cache.py` provides a bounded, thread-safe TTL cache. It is local to one backend process and clears on restart. It tracks hits, misses, expirations, evictions, and provider call timing.

- Yahoo daily OHLCV requests: 15-minute default TTL (`MARKET_DATA_CACHE_TTL_SECONDS`); 0 disables this cache. Scheduled refresh explicitly bypasses it.
- Technical feature results: keyed by input-data fingerprint and feature-code version; cached for 24 hours.
- Company-name lookups: successful result cached for one day; unsuccessful lookup cached for 30 minutes.
- Prediction response: not cached because market data/news can change.

`GET /api/cache/status` reports cache and TTL details. This cache reduces repeated work but is not shared between backend workers and does not survive restart.

### Scheduler

`backend/scheduler.py` starts a daemon thread during backend startup unless `MARKET_DATA_REFRESH_ENABLED=false`. It immediately attempts a refresh, then waits the configured interval (six hours by default; minimum five minutes). `MARKET_DATA_SYMBOLS` is a comma-separated ticker list, default `AAPL`.

For each ticker it fetches about one year of daily bars, bypasses the request cache, validates missing/duplicate/out-of-order data, computes technical features, and stores bars and features. A ticker is retried up to three times with short exponential delays. The job records start/end times, success/partial/failure state, rows, tickers, and per-symbol failures. Its latest status is available at `GET /api/jobs/market-data`.

The scheduler handles market-data refresh and feature calculation. It does not perform scheduled model fitting/inference. Yahoo determines when daily data becomes available; scheduled polling cannot make that source real-time. A single in-process thread is suitable for this prototype, not a distributed job system.

## 9. Persistence and database design

### Storage choices

The backend talks to Supabase Postgres through HTTPS REST calls using `requests`; no local DB server or Postgres Python driver is required. Set both `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in `backend/.env` to use it. If both are absent, the backend uses local JSON fallbacks. If only one or invalid settings are supplied, storage reports an incomplete/invalid state rather than silently pretending it connected.

The backend persists:

- **Predictions:** prediction response, ticker, horizon, timestamp, signal, model version, input window, data fingerprint, and explanation.
- **Signals:** signal type, timestamp, strategy/hybrid version, reason, and metadata tied to the prediction.
- **Market-data runs:** metadata/fingerprint and reference to the data used by a prediction.
- **Experiments:** evaluation/backtest report, run ID, metric summaries, parameters, versions, date window, and assumptions.
- **Scheduled market bars:** daily OHLCV bars and computed feature JSON for configured tickers, after migration 002 is installed.
- **Background job runs:** scheduler status and details, after migration 002 is installed.

Each record exists to answer a different operational or research question:

| Stored record | Reason to retain it | Important boundary |
|---|---|---|
| Market-data run metadata | Identify the provider, date range, input row count, retrieval time, quality status, and hash that underpinned a prediction. | The live prediction path stores metadata; it does not upload every historical OHLCV row into this table. |
| Prediction | Reopen prior backend outputs and compare ticker, signal, horizon, model version, input window, score, and response. | Backend history is separate from the browser's local signal cards. |
| Signal | Maintain a queryable, prediction-linked signal record with its hybrid version and reason. | It is descriptive output storage; there is no order execution or risk-control engine. |
| Experiment | Preserve a backtest/evaluation run with parameters, metrics, dates, data hash, and assumptions for later inspection or comparison. | It stores the report and provenance, not a guarantee of future behavior. |
| Scheduled bars and features | Retain refreshable OHLCV and derived feature snapshots for configured symbols, keyed by symbol/interval/date. | Only the background refresh writes these rows; migration 002 is required. |
| Job run | Explain whether a scheduled refresh succeeded, partially failed, or failed, how many rows it processed, and per-ticker errors. | Job state is a persisted summary; the scheduler itself still runs in one backend process. |
| Model artifact files | Reuse a fitted estimator and scaler when the complete model identity matches; record enough metadata to inspect the build. | Files remain local unless manually backed up; Supabase stores identifiers/records, not these binaries. |
| Tuning configuration | Keep user-selected model and simulation parameters across backend restarts. | Stored in local `backend/tuning_config.json`, not in the Supabase schema. |
| Backend cache | Avoid repeating provider and feature work during a process lifetime. | Temporary and cleared when the backend restarts. |
| Browser `localStorage` | Keep selected ticker, theme, and up to 100 UI signal snapshots on that browser. | Device/browser-specific; not synchronized with backend prediction history. |

In Supabase, `market_data_runs` is the provenance record referenced by `predictions`; `predictions` holds the result; `signals` has a one-to-one foreign-key link to that result; and `experiments` holds evaluation/backtest report payloads. Migration 002 adds `market_data_bars` (unique by symbol, interval, timestamp) and `background_job_runs`. These are not interchangeable stores: model files are local, the TTL cache is ephemeral, and UI history lives in the browser.

Tables for predictions/signals/market-data runs/experiments are in `backend/sql/001_persistent_storage.sql`. Scheduler tables are in `backend/sql/002_market_data_scheduler.sql`. Both migrations are run in the Supabase Dashboard SQL Editor. They enable row-level security, revoke public/anonymous/authenticated access, and grant the backend service role the intended table access. The secret key belongs only in the backend environment; do not put it in frontend code or commit it.

Without Supabase, local fallbacks include `prediction_runs.json`, `evaluation_runs.json`, `backtest_runs.json`, `market_data_bars.json`, and `job_runs.json` under `backend/`. They are local and ignored by Git. Tuning config and model artifacts are also local. The frontend's browser signal history is separate from all these backend stores.

### Supabase setup

1. Create a Supabase project and copy its Project URL and Secret key.
2. Run `backend/sql/001_persistent_storage.sql` once in that project's SQL Editor.
3. If scheduled bar/job persistence is wanted, also run `backend/sql/002_market_data_scheduler.sql`.
4. Put `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in `backend/.env`, preserving other settings. Never paste the secret into frontend settings or a public place.
5. Restart the backend and open `http://localhost:8000/api/storage/status`. Confirm `status` is `connected`.
6. Run predictions/evaluations/backtests; inspect history via the API. Restarting the backend should not remove Supabase records.
7. Existing local experiment reports can be imported once through `POST /api/storage/import-local-experiments`; run IDs make retries idempotent.

The prediction/history API can save metadata and model outputs without uploading every historical price row. The scheduler separately stores daily bars when migration 002 is applied. Browser-only saved signals are not automatically migrated to Supabase.

## 10. API reference

Interactive API docs are available at `http://localhost:8000/docs` while the backend is running. API responses are JSON. Vite proxies browser `/api` calls to the backend during development.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Basic backend health and UTC time. |
| GET | `/api/company/{symbol}` | Resolve a ticker to a company name through Yahoo metadata. |
| GET | `/api/predict/{symbol}?timeframe=1D` | Run live model inference, return signal/evidence, include a rule-based backtest, and persist prediction. Horizons accepted: `1D`, `1W`, `1M`, `3M`. |
| GET | `/api/history/{symbol}?period=3mo` | Return daily OHLCV history. Periods accepted: `3mo`, `6mo`, `1y`, `2y`, `3y`. |
| GET | `/api/backtest/{symbol}` | Run and persist the rule-based next-open backtest. |
| GET | `/api/backtest/{symbol}/hybrid?mode=fixed` | Run and persist fixed/adaptive hybrid, validation-calibrated No Trade, LSTM-only and Random Forest-only signals on shared walk-forward OOS dates; excludes final holdout. Mode defaults to fixed; `mode=adaptive` selects adaptive signals for calibrated/No Trade fields. |
| GET | `/api/regime/{symbol}` | Estimate the latest causal HMM market regime and return state probabilities and training/validation provenance. |
| GET | `/api/evaluate/{symbol}` | Run and persist the chronological direction model evaluation. |
| GET | `/api/generalization` | Run the fixed multi-sector example backtest basket. |
| GET | `/api/config` | Read tuning configuration. |
| POST | `/api/config` | Validate and save tuning configuration JSON. |
| GET | `/api/models` | List saved model versions; accepts optional `limit`. |
| GET | `/api/models/{version}` | Return saved version metadata. |
| POST | `/api/models/{version}/load` | Validate/load a saved artifact into the backend process. Requires POST, not browser address-bar GET. |
| GET | `/api/history/predictions?symbol=AAPL` | Read backend-saved predictions; symbol filter is optional. |
| GET | `/api/history/experiments?kind=backtest` | Read evaluation/backtest runs; `kind` is optional. |
| GET | `/api/storage/status` | Check configured storage connectivity or local fallback mode. |
| POST | `/api/storage/import-local-experiments` | Import local evaluation/backtest JSON into configured Supabase. |
| GET | `/api/jobs/market-data` | Scheduler enablement, configured symbols, last run and latest successful refresh. |
| GET | `/api/cache/status` | In-process cache counts and TTL configuration. |

## 11. Project layout

```text
DeepSignal/
|-- backend/
|   |-- main.py                 FastAPI routes, prediction orchestration, features and model calls
|   |-- hybrid.py               Direction target, hybrid version, weights and probability fusion
|   |-- model_registry.py        Artifact identity, persistence, validation and loading
|   |-- evaluation.py            Chronological model evaluation report
|   |-- backtesting.py           Timestamp-aware portfolio simulation engine
|   |-- hybrid_backtesting.py    Production-signal walk-forward backtests and fold comparisons
|   |-- adaptive_ensemble.py     Validation-selected bounded adaptive weights
|   |-- calibration.py           Direction calibration and No Trade thresholds
|   |-- regime_detection.py      Causal Gaussian-HMM regime inference
|   |-- news_alignment.py        Strict as-of news filtering and historical alignment helpers
|   |-- scheduler.py             Background daily market-data refresh
|   |-- cache.py                 Bounded in-process TTL cache
|   |-- database.py              Supabase REST and local JSON persistence
|   |-- requirements.txt         Backend runtime dependencies
|   |-- requirements-dev.txt     Runtime plus test client dependency
|   |-- .env.example             Safe settings template
|   |-- sql/
|   |   |-- 001_persistent_storage.sql
|   |   `-- 002_market_data_scheduler.sql
|   `-- tests/                   Focused unittest suite
|-- frontend/
|   |-- package.json
|   |-- package-lock.json
|   |-- vite.config.js
|   |-- index.html
|   `-- src/
|       |-- main.jsx             Pages, shared ticker state, API calls, UI logic
|       |-- Charts.jsx           Reusable Recharts components
|       `-- styles.css           Themes, layouts, responsive UI and component styles
|-- docs/
|   `-- DeepSignal_Project_Guide.md  This complete project reference
`-- tools/
    |-- run_real_evaluation.py       Reproduce multi-symbol direction/model evaluations
    |-- run_hybrid_backtest.py       Reproduce production-hybrid walk-forward portfolios
    |-- run_regime_evaluation.py     Report predictions grouped by causal HMM regime
    |-- run_adaptive_evaluation.py   Reproduce fixed/adaptive/regime ablations
    |-- run_calibration_evaluation.py Evaluate calibration and No Trade thresholds
    |-- build_phase9_report.py       Rebuild the final experiment matrix and SVG figures
    `-- build_project_guide_pdf.py   Rebuild the guide PDF with Python only
```

Generated directories such as `backend/.venv`, `frontend/node_modules`, `frontend/dist`, `backend/model_artifacts`, local environment files, and local run histories are not source files and are ignored by Git.

### File-by-file responsibilities

The tree above is the quick map. This inventory describes the maintained source and configuration files individually; generated environments, downloaded model weights, caches, and local run-history JSON are runtime data and are intentionally not listed as source.

#### Repository-level files

| File | Responsibility |
|---|---|
| `README.md` | Short project overview, local setup, first prediction, evaluation, backtesting, and Supabase/scheduler guidance. The detailed behavior reference is this guide. |
| `.gitignore` | Excludes virtual environments, installed frontend packages, build output, secrets, local configuration, reports, and fitted artifacts from version control. |
| `.gitattributes` | Repository text-file and line-ending conventions. |
| `.vscode/settings.json` | Workspace editor preferences for contributors using Visual Studio Code. |
| `run_backend.bat` | Windows helper that checks Python, creates the backend virtual environment if needed, copies the safe environment template when absent, installs runtime requirements, and launches Uvicorn on port 8000. |
| `run_frontend.bat` | Windows helper that checks npm, installs locked frontend dependencies when needed, and starts the Vite development server. |

#### Backend Python modules

| File | Responsibility |
|---|---|
| `backend/main.py` | FastAPI application and route definitions; ticker normalization; Yahoo price and company lookups; indicator construction; news collection and FinBERT scoring; LSTM/forest inference and optional comparison; prediction orchestration; tuning configuration; scheduler lifecycle hooks. This is the service composition layer. |
| `backend/news_alignment.py` | Strict timezone-aware publication cutoff, ticker/query relevance, canonical article references, duplicate filtering, and NYSE session-close historical alignment/coverage helpers. |
| `backend/hybrid.py` | Defines the next-session-up target contract, hybrid version and component weights; validates probabilities; maps sentiment and produces the auditable weighted result and signal category. |
| `backend/adaptive_ensemble.py` | Selects bounded adaptive component weights from preceding validation Brier evidence, with regime-specific selection and explicit overall-validation fallback. |
| `backend/calibration.py` | Validation-only Platt/isotonic selection, reliability/Brier evaluation, component-disagreement summaries, and No Trade threshold selection. |
| `backend/model_registry.py` | Creates stable model identities and version IDs; writes and validates local artifact bundles; restores compatible estimators/scalers; maintains a bounded loaded-model cache. |
| `backend/evaluation.py` | Builds chronological train, validation, and final test windows; compares direction models and baselines on common dates; produces metrics, hashes, assumptions, and a persisted experiment report. |
| `backend/backtesting.py` | Implements timestamp-aware next-open simulation; accounts for costs/slippage, cash, shares, equity, baselines, trades, and performance metrics for both technical and model signal strategies. |
| `backend/hybrid_backtesting.py` | Builds out-of-sample production hybrid/LSTM/Random Forest signals from expanding folds and compares them on matching dates; final holdout is excluded. |
| `backend/regime_detection.py` | Builds causal regime features, fits/selects train-window Gaussian HMMs, maps states from training statistics, filters posteriors through each cutoff, and provides walk-forward regime-stratified evaluation. |
| `backend/database.py` | Storage adapter for optional Supabase Postgres over HTTPS and local JSON fallback; persists predictions, signals, experiments, market bars, and scheduler status. |
| `backend/cache.py` | Bounded, thread-safe in-process TTL cache and cache/provider timing counters. It does not share state between backend processes. |
| `backend/scheduler.py` | Starts/stops the background refresh thread; fetches configured symbols, validates and feature-engineers bars, retries failures, persists data, and reports job status. |

#### Backend configuration and database schema

| File | Responsibility |
|---|---|
| `backend/requirements.txt` | Runtime Python packages needed by the API and its supported model/data paths. |
| `backend/requirements-dev.txt` | Development/test dependency additions on top of runtime requirements. |
| `backend/.env.example` | Safe template for optional provider keys, Supabase settings, cache/scheduler controls, artifact path, and CORS origins. It contains no live secret. |
| `backend/sql/001_persistent_storage.sql` | Creates the core Supabase tables and access policies for prediction and experiment persistence. |
| `backend/sql/002_market_data_scheduler.sql` | Adds scheduled market-bar and refresh-job persistence used by the background scheduler. |

#### Backend test files

| File | Responsibility |
|---|---|
| `backend/tests/test_validation.py` | Checks health/prediction response shapes, signal boundaries, tuning validation/persistence, provider absence, and graceful optional-model failure behavior. |
| `backend/tests/test_news_alignment.py` | Verifies timezone-aware cutoff filtering, post-cutoff rejection, relevance and duplicate handling, canonical references, and New York session-close/DST alignment. |
| `backend/tests/test_routes.py` | Guards route declaration ordering so fixed history routes are not captured by the dynamic ticker route. |
| `backend/tests/test_hybrid.py` | Verifies weighted component arithmetic, audit fields, invalid-input rejection, and shared evaluation comparisons. |
| `backend/tests/test_adaptive_ensemble.py` | Checks bounded weights, validation-only selection, unavailable components, regime evidence thresholds, and safe fallbacks. |
| `backend/tests/test_calibration.py` | Checks validation-only calibrator selection, Brier/reliability metrics, minimum evidence, No Trade thresholds, and finite cutoffs. |
| `backend/tests/test_model_registry.py` | Verifies artifact identity changes and reloading compatible artifacts in a clean process. |
| `backend/tests/test_backtesting.py`, `backend/tests/test_hybrid_backtesting.py`, `backend/tests/test_regime_detection.py` | Verify next-open order timing, costs, walk-forward matching/holdout exclusion, HMM training boundaries, causal inference, state mapping, and failure fallbacks. |
| `backend/tests/test_database.py` | Checks local JSON persistence/readback and Supabase request/upsert behavior, including symbol filtering and backend-only secret headers. |
| `backend/tests/test_cache.py` | Checks TTL expiration, copy protection, and avoiding repeated provider calls for cached price history. |
| `backend/tests/test_scheduler.py` | Checks configured ticker selection and refresh validation/persistence summaries. |

The focused tests also exercise evaluation through its shared hybrid comparison logic; there is no separate `test_evaluation.py` file. The full suite is invoked from the repository root with `python -m unittest discover -s backend/tests -v`.

At completion of the final research report, the full backend suite reported **62 passing tests**. The frontend production build passed with 2,211 modules during API and frontend integration. Final report generation, JSON assertions, SVG XML parsing, and `git diff --check` passed. There is no configured lint or frontend type-check script. These are the latest recorded checks; they are not run automatically when the app starts.

#### Frontend files

| File | Responsibility |
|---|---|
| `frontend/package.json` | Declares React/Vite dependencies and frontend scripts. |
| `frontend/package-lock.json` | Locks the resolved npm dependency tree for reproducible installation. |
| `frontend/vite.config.js` | Configures the Vite development server and proxies `/api` requests to the backend. |
| `frontend/index.html` | Browser document shell and React mount point. |
| `frontend/src/main.jsx` | Application entry point: routing, shared ticker/theme state, API helpers, page components, prediction display, local signal history, dashboard and user interactions. |
| `frontend/src/Charts.jsx` | Lazy-loadable reusable Recharts panels used for historical and comparative visualizations. |
| `frontend/src/styles.css` | Theme tokens, page layout, cards, navigation, chart presentation, responsive behavior, and interaction styling. |

#### Documentation, research runners, and generated evidence

| File | Responsibility |
|---|---|
| `docs/DeepSignal_Project_Guide.md` | Canonical long-form reference for implemented behavior, architecture, APIs, setup, persistence, evaluation, dashboard, and limitations. |
| `tools/build_project_guide_pdf.py` | Dependency-free Markdown-to-PDF renderer used to create the styled, searchable guide at `docs/DeepSignal_Project_Guide.pdf`, including tables and updated vector architecture/request/evaluation diagrams. |
| `tools/run_real_evaluation.py` | Fetches requested Yahoo daily histories and creates per-symbol chronological direction evaluation reports; supports optional XGBoost and report-only rendering from saved JSON. |
| `tools/run_hybrid_backtest.py` | Runs the production hybrid walk-forward next-open portfolio simulation and writes a standalone JSON audit. |
| `tools/run_regime_evaluation.py` | Evaluates fixed-hybrid predictions grouped by causal HMM regime; can regenerate Markdown from saved JSON. |
| `tools/run_adaptive_evaluation.py` | Runs fixed/adaptive and paired regime-selection ablations; accepts a retained OHLCV CSV for exact paired input. |
| `tools/run_calibration_evaluation.py` | Runs validation-selected calibration and No Trade evaluation on walk-forward predictions. |
| `tools/run_news_alignment_report.py` | Regenerates the historical news provider-availability/terms report without fetching or retaining headlines. |
| `tools/build_phase9_report.py` | Validates paired E4 data/date identity and builds the final Markdown/JSON matrix and dependency-free SVG figures from retained report artifacts. |
| `research/real_data/phase9_final_experiment_matrix.md` | Final research summary covering E1–E7, measured findings, unavailable E5 sentiment ablation, limitations, and reproduction instructions. |
| `research/real_data/phase9_final_experiment_matrix.json` | Machine-readable final metrics, per-ticker/per-fold evidence, run IDs, hashes, versions, and artifact references. |
| `research/real_data/phase{1..7}_*.json` and matching Markdown reports | Supporting evaluation, hybrid backtest, regime/adaptive/calibration findings, news availability, and optional XGBoost results. |
| `research/real_data/phase9_*.svg`, `phase9_e4_aapl_prices.csv`, and paired E4 JSON/Markdown | Final report figures and retained paired input/audits used to reproduce and verify the regime-weight comparison. |
| `research/baseline/PHASE_0_BASELINE.md` | Preserved inspection of the original local evaluation, prediction, and technical-backtest artifacts; it records their dates, hashes, and important legacy-field limitations. |

> **Inventory note:** This guide intentionally does not enumerate ignored machine-local files such as `backend/.env`, tuning overrides, downloaded model weights, virtual environments, node_modules, or generated run-history JSON. They can contain secrets, large data, or machine-specific state and are not part of the shared source tree.

## 12. Run the project

### Windows: simplest start from scratch

Install Python and Node.js (which includes npm) first. In two PowerShell terminals opened at the project root, run:

```powershell
cd C:\Users\Athira\Downloads\DeepSignal
.\run_backend.bat
```

In the second terminal:

```powershell
cd C:\Users\Athira\Downloads\DeepSignal
.\run_frontend.bat
```

Keep both terminals open. Open the local URL printed by Vite, usually `http://localhost:5173`. Stop each server with Ctrl+C.

### Regenerate this guide as a PDF

With Python installed, run from the project root:

```powershell
python tools/build_project_guide_pdf.py
```

The script uses only Python's standard library. It writes `docs/DeepSignal_Project_Guide.pdf` and draws the architecture and request/evaluation pipeline diagrams from the documented model roles and result paths. The PDF renderer has explicit vector-diagram layouts, so update its corresponding drawing methods when changing a Mermaid diagram.

`run_backend.bat` creates `backend/.venv` if needed, creates `backend/.env` from the example only if no `.env` exists, installs/updates `backend/requirements.txt`, and starts Uvicorn on `127.0.0.1:8000` with auto-reload. It never replaces an existing `.env`.

`run_frontend.bat` checks npm, installs packages on first setup (`npm ci` when a lockfile is present; otherwise `npm install`), and runs Vite. If dependencies change after `node_modules` already exists, run `npm ci` manually inside `frontend` to synchronize with the lockfile.

### Manual Windows setup

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env  # only if .env does not already exist
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

### macOS / Linux manual setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env  # only if .env does not already exist
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

TensorFlow support depends on operating system and Python version. The app has a neutral LSTM fallback if TensorFlow cannot load, but the Random Forest/FinBERT dependency paths still need their packages and provider access. First FinBERT use may download model weights.

## 13. Configuration

Create `backend/.env` from `.env.example` and keep it private. Current settings:

| Setting | Default / purpose |
|---|---|
| `SUPABASE_URL` | Hosted project URL; blank means local JSON fallback. |
| `SUPABASE_SECRET_KEY` | Backend-only secret key; blank means local fallback. |
| `MARKET_DATA_REFRESH_ENABLED` | `true`; set `false` to disable the scheduler. |
| `MARKET_DATA_REFRESH_MINUTES` | `360`; job cadence, clamped to a minimum of five minutes. |
| `MARKET_DATA_SYMBOLS` | `AAPL`; comma-separated symbols for scheduled refresh. |
| `MARKET_DATA_CACHE_TTL_SECONDS` | `900`; daily history cache TTL; `0` disables it. |
| `NEWS_API_KEY` | Optional NewsAPI key; Yahoo news remains a fallback. |
| `FINBERT_MODEL` | Optional Transformers model name; default `ProsusAI/finbert`. |
| `DEEPSIGNAL_ARTIFACT_DIR` | Optional model artifact folder override. |
| `HYBRID_BACKTEST_MODE` | `fixed`; default mode for the hybrid backtest route (`fixed` or `adaptive`). |
| `ADAPTIVE_ENSEMBLE_MIN_WEIGHT` / `ADAPTIVE_ENSEMBLE_MAX_WEIGHT` | `0.05` / `0.80`; adaptive component weight bounds. |
| `ADAPTIVE_ENSEMBLE_TEMPERATURE` | `8.0`; sensitivity of bounded weights to validation Brier skill. |
| `ADAPTIVE_MIN_VALIDATION_SAMPLES` / `ADAPTIVE_MIN_REGIME_SAMPLES` | `20` / `20`; minimum evidence rows before using overall or regime-specific validation scores. |
| Calibration and No Trade | Experimental, walk-forward-only. Calibration method and neutral/disagreement cutoffs are selected from past validation rows; insufficient evidence leaves raw baseline decisions in place. |
| `CORS_ORIGINS` | Allowed origins for non-Vite deployment; default localhost Vite URL. |

Model and backtest values are managed through the Tuning page/API and saved in `backend/tuning_config.json`. Current defaults and accepted ranges:

| Setting | Default | Accepted range |
|---|---:|---:|
| LSTM lookback | 30 | 10–120 |
| LSTM units | 48 | 8–256 |
| LSTM epochs | 6 | 1–100 |
| Random Forest trees | 250 | 50–1,000 |
| Transaction cost | 0.06% | 0–2% |
| Slippage | 0.04% | 0–2% |

When live model training or preprocessing code changes, increment `PREDICTION_MODEL_CODE_VERSION` in `backend/main.py`; that makes the new identity avoid reusing artifacts created by older code. Keep the feature and hybrid versions accurate when their corresponding behavior changes.

## 14. Validation and troubleshooting

### Run the backend tests

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python -m pip install -r backend/requirements-dev.txt
python -m unittest discover -s backend/tests -v
```

The focused suite covers cache behavior, data/storage handling, model artifacts, hybrid math, backtesting, scheduler behavior, API routes, and input validation. Tests use mocks where needed so normal test runs do not need live market/news calls. Run the suite after code changes when you want verification.

### Useful checks

- Backend health: `http://localhost:8000/api/health`
- Interactive API docs: `http://localhost:8000/docs`
- Storage mode/connection: `http://localhost:8000/api/storage/status`
- Scheduler status: `http://localhost:8000/api/jobs/market-data`
- Cache status: `http://localhost:8000/api/cache/status`
- Saved model versions: `http://localhost:8000/api/models`
- Saved AAPL predictions: `http://localhost:8000/api/history/predictions?symbol=AAPL`
- Saved experiment reports: `http://localhost:8000/api/history/experiments`

### Common problems

| Symptom | Likely cause / next check |
|---|---|
| Frontend loads but API calls fail | Ensure the backend window is running on port 8000; restart the backend if its console shows an import or dependency error. |
| `No market data found` | Check ticker spelling/market suffix, Yahoo availability, network access, and provider throttling. |
| Supabase says `incomplete` or `invalid_url` | Check both settings in `backend/.env`, then restart. Ensure SQL migrations ran in the same Supabase project. |
| `PGRST205` table/schema cache error | Apply the referenced SQL migration to the correct project, then reload the Supabase schema cache/restart the backend. |
| Job status is running but last run failed | `running` means the scheduler thread is active; inspect `last_run.status`, `failures`, storage error, and backend logs. |
| Empty backend prediction/experiment history | Create a prediction/evaluation/backtest through the API after connecting storage. Browser-only signals do not populate these API endpoints. |
| Model artifact load returns method not allowed | The route requires POST. A browser address bar sends GET; use PowerShell `Invoke-RestMethod -Method Post` or the API docs. Substitute a real version ID, not the literal placeholder. |
| LSTM probability is neutral | TensorFlow could not import or there were too few sequences. Check artifact model status and Python/TensorFlow compatibility. |
| Company name does not appear | Wait briefly after entering the symbol; verify backend connectivity and Yahoo metadata for that symbol. The ticker may still be usable even if no display name is found. |
| First analysis is slow | Initial fitting and/or model loading, FinBERT download/load, and provider calls can take time. Later requests can reuse matching model artifacts and in-process caches. |

## 15. Security, privacy, and boundaries

- Keep `backend/.env` and Supabase secrets out of Git, screenshots, browser developer settings, and frontend bundles. `.gitignore` excludes the local backend `.env`.
- The backend uses the Supabase secret only in server-side REST headers. SQL enables RLS and restricts table permissions to the service role.
- This project has no user login, tenant boundary, brokerage connection, order execution, or public production hardening. Do not expose the backend or service secret as a public multi-user service without adding authentication, authorization, rate limiting, secret management, and operational controls.
- Browser `localStorage` holds theme, selected ticker, and locally saved signal copies. It is browser/device-specific and can be cleared by the user.
- Provider requests send ticker queries to Yahoo Finance and, when configured, NewsAPI. Avoid treating personal information as input.
- Supabase is optional; without it, local JSON reports, model artifacts, cache, and browser history have different backup and persistence scopes.

## 16. Current limitations and future work

These are material distinctions to state in demonstrations and reports:

1. **Daily, provider-dependent prices:** This is not an exchange-grade real-time feed. Yahoo may delay or revise data. The “Live” badge refers to periodic interface/status refresh, not live tick data.
2. **Forecast horizon:** LSTM/RF/hybrid direction target is one session. Longer horizon choices scale a heuristic change estimate; they do not create independently trained 1-week or 1-month models.
3. **Not a trading recommendation:** Signal thresholds, technical points, sentiment mapping, and confidence display are heuristic and not calibrated to realized profit probability.
4. **Historical news alignment:** Live headline sentiment is filtered at the request cutoff, but current provider searches do not constitute a verified historical archive. NewsAPI history depends on the active plan and its terms restrict archive use; Yahoo completeness/retention rights were not verified. The dedicated historical hybrid evaluation uses neutral news sentiment; see `research/real_data/phase6_news_alignment.md`.
5. **Research sample size:** E1 covers five selected symbols across four walk-forward folds. E2–E7 use one AAPL window and four folds. These samples do not represent a broad security universe or all market regimes; no confidence intervals or significance tests were run.
6. **Live LSTM validation:** The live scaler is fit on the feature window before its internal validation split. Dedicated evaluation fits the scaler on training rows only; prefer that report when discussing out-of-sample model assessment.
7. **Backtest assumptions and outcomes:** The technical endpoint tests its deterministic technical strategy; the hybrid endpoint evaluates production hybrid signals on development walk-forward folds and excludes the final holdout. Both simplify costs, fractional shares, fills, and long-only sizing. In the final report the adaptive hybrid underperformed the fixed hybrid in the AAPL sample, regime-specific weights made no aggregate portfolio difference, and No Trade reduced drawdown while lowering return and Sharpe. These are descriptive historical results, not strategy guarantees.
8. **Calibration and No Trade:** Calibration worsened pooled fixed-hybrid Brier from 0.2532 raw to 0.2676 calibrated. Calibration and No Trade remain evaluation-only; the live prediction route reports them unavailable because it does not fit a point-in-time validation calibrator or abstention threshold.
9. **News ablation:** E5 was not run because no eligible, appropriately permitted point-in-time historical archive was supplied. Historical FinBERT contribution remains neutral in research rows; its incremental value is unknown.
10. **Persistence boundaries:** Artifact binaries remain local. Database rows record model-version identifiers but do not automatically move artifact files to Supabase. Browser signals are not automatically imported to backend history.
11. **Prototype operations:** Scheduler/cache are in-process and local to one backend instance. There is no distributed worker, job queue, authentication, or deployment-grade monitoring.
12. **Optional dependencies/providers:** XGBoost/SHAP failures are tolerated, FinBERT can be unavailable, and news or company metadata can be missing without stopping all core features.

The planned implementation and research work is complete. Further research should be gated on (a) a permitted point-in-time news archive for the E5 ablation, (b) a predeclared, later untouched confirmation window, and (c) a wider predeclared ticker universe with repeated folds. Improve execution realism and deployment security before treating the prototype as a production service. More model complexity alone does not establish model quality.

## 17. Glossary

| Term | Simple meaning |
|---|---|
| OHLCV | Open, High, Low, Close, and Volume market bar fields. |
| Ticker / symbol | Short identifier for a listed security, such as `AAPL`. |
| Feature | A numeric input calculated from market data for a model. |
| Target / label | The outcome the model is trained to estimate; here, next-session up or down. |
| LSTM | A neural network architecture designed to process ordered sequences. |
| Random Forest | An ensemble of decision trees for classification. |
| FinBERT | A finance-oriented pretrained language model used here to score headlines. |
| Hybrid | A versioned rule that combines compatible component outputs. |
| Holdout | Later historical rows kept out of a model's fitting for evaluation. |
| Data fingerprint | A hash used to identify the exact input price window. |
| Artifact | Saved fitted model/preprocessor files plus the metadata needed to identify them. |
| Backtest | Historical simulation with defined signal timing, execution, and costs. |
| Slippage | Difference between a reference price and a less favorable assumed fill. |
| TTL cache | Temporary stored response that expires after a configured time. |
| Walk-forward | Testing in time order so decisions do not use future rows. |
| Adaptive ensemble | A weighted combination whose weights are selected from earlier validation evidence and bounded to sum to one. |
| Calibration | A validation-fitted mapping of model direction scores to empirical direction frequencies; it does not calibrate profit. |
| Component disagreement | Spread among available model component probabilities; this project uses standard deviation as a heuristic, not a confidence interval. |
| No Trade | Validation-selected abstention decision; in backtests it is represented as HOLD and retains an existing position. |
| Regime | A descriptive HMM-inferred cluster of daily market features; it is not known ground truth or a return forecast. |

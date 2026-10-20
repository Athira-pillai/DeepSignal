# DeepSignal — Hybrid Stock Trend Prediction

A standalone version of the uploaded stock-prediction UI, extended to match the final-year-project proposal in the PDF.

## What is included
- Live/latest available OHLCV data from `yfinance`
- Technical indicators: SMA 20/50, RSI, MACD, Bollinger Bands
- LSTM price/trend forecasting (TensorFlow/Keras)
- Financial-news sentiment with FinBERT (Hugging Face)
- Random Forest signal classifier with optional XGBoost comparison
- Buy / Sell / Hold signal generation
- Walk-forward backtesting with transaction costs and slippage
- Buy-and-hold and naive baselines
- Accuracy, cumulative return, Sharpe ratio and maximum drawdown
- SHAP feature importance when available
- Multi-stock/sector evaluation endpoint
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
uvicorn main:app --reload --port 8000
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

### Important
A prediction is a model output, not a guarantee of future returns or investment advice. The API reports the timestamp and latest available market bar used for the prediction so the UI is transparent about freshness.

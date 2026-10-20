import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

app = FastAPI(title="DeepSignal Hybrid Stock Prediction API", version="1.0.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

FEATURES = ["return_1d", "return_5d", "sma20_gap", "sma50_gap", "rsi", "macd", "macd_signal", "bb_position", "volume_change", "sentiment"]
TUNING_CONFIG = {
    "lookback": 30,
    "lstm_units": 48,
    "epochs": 6,
    "rf_trees": 250,
    "transaction_cost_pct": 0.06,
    "slippage_pct": 0.04,
}


class TuningConfig(BaseModel):
    lookback: int = Field(ge=10, le=120)
    lstm_units: int = Field(ge=8, le=256)
    epochs: int = Field(ge=1, le=100)
    rf_trees: int = Field(ge=50, le=1000)
    transaction_cost_pct: float = Field(ge=0, le=2)
    slippage_pct: float = Field(ge=0, le=2)


_tuning_config_path = os.path.join(os.path.dirname(__file__), "tuning_config.json")
try:
    with open(_tuning_config_path, encoding="utf-8") as config_file:
        TUNING_CONFIG.update(TuningConfig(**json.load(config_file)).model_dump())
except (OSError, ValueError):
    pass


def clean_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "")


def download_history(symbol: str, period: str = "3y") -> pd.DataFrame:
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval="1d", auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {symbol}.")
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        raise HTTPException(status_code=502, detail="Market provider returned no date column.")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna().copy()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    std20 = close.rolling(20).std()
    out["bb_upper"] = out["sma20"] + 2 * std20
    out["bb_lower"] = out["sma20"] - 2 * std20
    out["return_1d"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    out["sma20_gap"] = close / out["sma20"] - 1
    out["sma50_gap"] = close / out["sma50"] - 1
    out["bb_position"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["volume_change"] = out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


def get_news(symbol: str):
    articles = []

    def add_article(title, published_at=None, source=None, url=None):
        if not title:
            return
        title = str(title).strip()
        if not title or title.lower() in {"(no title)", "no title"}:
            return
        if any(item["title"] == title for item in articles):
            return
        articles.append({
            "title": title,
            "published_at": published_at,
            "source": source or "Financial news",
            "url": url,
        })

    # NewsAPI is optional and its free tier may not include the everything
    # endpoint. Keep it as the first source when configured, then fall back.
    key = os.getenv("NEWS_API_KEY", "").strip()
    if key:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": symbol, "language": "en", "sortBy": "publishedAt", "pageSize": 10, "apiKey": key},
                timeout=12,
            )
            r.raise_for_status()
            for a in r.json().get("articles", []):
                add_article(a.get("title"), a.get("publishedAt"), (a.get("source") or {}).get("name", "NewsAPI"), a.get("url"))
        except Exception:
            pass

    # Yahoo's search endpoint supplies recent news even when yfinance's
    # Ticker.news is empty or its response shape changes between releases.
    if len(articles) < 8:
        try:
            r = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": symbol, "newsCount": 10, "quotesCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            r.raise_for_status()
            for item in r.json().get("news", []):
                add_article(
                    item.get("title"),
                    item.get("providerPublishTime"),
                    item.get("publisher", "Yahoo Finance"),
                    item.get("link"),
                )
        except Exception:
            pass

    if len(articles) < 8:
        try:
            for item in (yf.Ticker(symbol).news or []):
                content = item.get("content") or {}
                provider = content.get("provider") or {}
                canonical = content.get("canonicalUrl") or {}
                add_article(
                    content.get("title") or item.get("title"),
                    content.get("pubDate") or item.get("providerPublishTime"),
                    provider.get("displayName") or item.get("publisher") or "Yahoo Finance",
                    canonical.get("url") or item.get("link"),
                )
                if len(articles) >= 10:
                    break
        except Exception:
            pass
    return articles


@lru_cache(maxsize=1)
def finbert_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model=os.getenv("FINBERT_MODEL", "ProsusAI/finbert"), tokenizer=os.getenv("FINBERT_MODEL", "ProsusAI/finbert"), top_k=None)


def sentiment_score(articles):
    if not articles:
        return 0.0, []
    try:
        pipe = finbert_pipeline()
        texts = [a["title"] for a in articles[:8]]
        outputs = pipe(texts, truncation=True, max_length=128)
        scores = []
        labeled = []
        for article, result in zip(articles[:8], outputs):
            # transformers versions return either list[dict] or list[list[dict]]
            rows = result if isinstance(result, list) else [result]
            mapping = {x["label"].lower(): float(x["score"]) for x in rows}
            score = mapping.get("positive", 0.0) - mapping.get("negative", 0.0)
            scores.append(score)
            labeled.append({**article, "sentiment": round(score, 4)})
        return float(np.mean(scores)), labeled
    except Exception:
        # Safe fallback: no generic sentiment is substituted; return neutral when FinBERT is unavailable.
        return 0.0, [{**a, "sentiment": None} for a in articles]


def make_training_frame(df: pd.DataFrame, sentiment: float):
    x = indicators(df)
    x["sentiment"] = sentiment
    next_close = x["close"].shift(-1)
    x["target"] = np.where(next_close.notna(), (next_close > x["close"]).astype(float), np.nan)
    # Keep the latest row as a prediction row; it has no known next-day label.
    x = x.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    return x


def lstm_direction(frame: pd.DataFrame, lookback=None):
    # Lazy import keeps non-model endpoints fast.
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except (ImportError, OSError):
        # TensorFlow may be unavailable on some local Python installations.
        # Keep the hybrid prediction endpoint usable with a neutral LSTM input.
        return 0.5

    lookback = lookback or TUNING_CONFIG["lookback"]
    cols = ["close", "sma20", "sma50", "rsi", "macd", "bb_upper", "bb_lower"]
    data = frame[cols].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback - 1, len(scaled) - 1):
        X.append(scaled[i-lookback+1:i+1])
        y.append(1 if frame.iloc[i+1]["close"] > frame.iloc[i]["close"] else 0)
    if len(X) < 80:
        return 0.5
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    split = max(int(len(X) * 0.85), len(X)-40)
    model = Sequential([LSTM(TUNING_CONFIG["lstm_units"], input_shape=(X.shape[1], X.shape[2])), Dropout(0.2), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]), epochs=TUNING_CONFIG["epochs"], batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
    latest = scaled[-lookback:][None, ...]
    return float(model.predict(latest, verbose=0)[0][0])


def classifier_prediction(frame: pd.DataFrame, lstm_prob: float):
    train = frame.copy()
    # Training proxy: rolling 5-day momentum is intentionally lagged and is not the target itself.
    train["lstm_prob"] = 1 / (1 + np.exp(-train["return_5d"].fillna(0) * 25))
    cols = FEATURES + ["lstm_prob"]
    latest = train[cols].iloc[-1].to_dict()
    train = train.dropna(subset=cols + ["target"])
    if len(train) < 100:
        return "hold", 0.5, {c: 0.0 for c in cols}, {
            "random_forest": {"accuracy": None, "balanced_accuracy": None},
            "xgboost": {"accuracy": None},
            "evaluation": {"method": "chronological_holdout", "status": "insufficient_samples", "train_samples": int(len(train)), "test_samples": 0},
            "shap": {},
        }

    # Keep the newest observations out of fitting when reporting quality.
    # The chronological split better reflects future use than scoring on rows
    # the model has already seen.
    test_size = max(20, int(np.ceil(len(train) * 0.20)))
    test_size = min(test_size, len(train) - 80)
    split_at = len(train) - test_size
    fit_rows, holdout_rows = train.iloc[:split_at], train.iloc[split_at:]
    rf_options = dict(n_estimators=TUNING_CONFIG["rf_trees"], max_depth=7, min_samples_leaf=4, random_state=42, class_weight="balanced")
    rf_eval = RandomForestClassifier(**rf_options)
    rf_eval.fit(fit_rows[cols], fit_rows["target"])
    rf_holdout = rf_eval.predict(holdout_rows[cols])
    rf_accuracy = round(float(accuracy_score(holdout_rows["target"], rf_holdout) * 100), 2)
    rf_balanced_accuracy = round(float(balanced_accuracy_score(holdout_rows["target"], rf_holdout) * 100), 2)

    # Refit all labeled rows after evaluation for the current live prediction.
    rf = RandomForestClassifier(**rf_options)
    rf.fit(train[cols], train["target"])
    latest["lstm_prob"] = lstm_prob
    prob_up = float(rf.predict_proba(pd.DataFrame([latest]))[0][1])
    importance = {c: round(float(v), 4) for c, v in zip(cols, rf.feature_importances_)}
    model_comparison = {
        "random_forest": {"accuracy": rf_accuracy, "balanced_accuracy": rf_balanced_accuracy},
        "evaluation": {
            "method": "chronological_holdout",
            "status": "ok",
            "train_samples": int(len(fit_rows)),
            "test_samples": int(len(holdout_rows)),
            "test_start": holdout_rows.iloc[0]["date"].strftime("%Y-%m-%d"),
            "test_end": holdout_rows.iloc[-1]["date"].strftime("%Y-%m-%d"),
        },
    }
    try:
        from xgboost import XGBClassifier
        xgb_options = dict(n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42)
        xgb_eval = XGBClassifier(**xgb_options)
        xgb_eval.fit(fit_rows[cols], fit_rows["target"])
        xgb_holdout = xgb_eval.predict(holdout_rows[cols])
        xgb_accuracy = round(float(accuracy_score(holdout_rows["target"], xgb_holdout) * 100), 2)
        xgb_balanced_accuracy = round(float(balanced_accuracy_score(holdout_rows["target"], xgb_holdout) * 100), 2)
        xgb = XGBClassifier(**xgb_options)
        xgb.fit(train[cols], train["target"])
        xgb_prob = float(xgb.predict_proba(pd.DataFrame([latest]))[0][1])
        model_comparison["xgboost"] = {"accuracy": xgb_accuracy, "balanced_accuracy": xgb_balanced_accuracy, "probability_up": round(xgb_prob, 4)}
    except Exception:
        pass
    # SHAP local explanation for the latest feature vector.
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        vals = explainer.shap_values(pd.DataFrame([latest]))
        if isinstance(vals, list):
            vals = vals[-1][0]
        else:
            vals = np.asarray(vals).reshape(-1)[:len(cols)]
        model_comparison["shap"] = {c: round(float(v), 5) for c, v in zip(cols, vals)}
    except Exception:
        model_comparison["shap"] = {}
    return ("buy" if prob_up >= 0.60 else "sell" if prob_up <= 0.40 else "hold"), prob_up, importance, model_comparison


def metrics(curve):
    values = np.asarray(curve, dtype=float)
    total = (values[-1] / values[0] - 1) * 100
    peak = np.maximum.accumulate(values)
    maxdd = np.min((values - peak) / peak) * 100
    returns = pd.Series(values).pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() or 1e-9)) * np.sqrt(252)
    return {"total_return": round(float(total), 2), "sharpe": round(float(sharpe), 2), "max_drawdown": round(float(maxdd), 2)}


def signal_from_score(score: float) -> str:
    score = float(np.clip(score, 0, 1)) if np.isfinite(score) else 0.5
    if score >= 0.64:
        return "strong_buy"
    if score >= 0.54:
        return "buy"
    if score <= 0.36:
        return "strong_sell"
    if score <= 0.46:
        return "sell"
    return "hold"


def walk_forward(df: pd.DataFrame):
    d = indicators(df).dropna().reset_index(drop=True)
    close = d["close"].values
    n = len(d)
    window = 180
    if n < window + 30:
        window = max(60, n // 2)
    strategy = [100.0]
    bh = [100.0]
    naive = [100.0]
    pos = 0
    prev_pos = 0
    for i in range(1, n):
        if i >= window:
            mom = close[i-5] / close[i-10] - 1 if i >= 10 else 0
            rsi = d.iloc[i]["rsi"]
            pos = 1 if mom > 0.006 and rsi < 72 else -1 if mom < -0.006 and rsi > 28 else 0
        ret = close[i] / close[i-1] - 1
        changed = pos != prev_pos
        cost = (TUNING_CONFIG["transaction_cost_pct"] + TUNING_CONFIG["slippage_pct"]) / 100 if changed else 0.0
        strategy.append(strategy[-1] * (1 + pos * ret - cost))
        bh.append(bh[-1] * (1 + ret))
        naive.append(naive[-1])
        prev_pos = pos
    return {
        "dates": [d.iloc[i]["date"].strftime("%Y-%m-%d") for i in range(n)],
        "hybrid": [round(x, 2) for x in strategy],
        "buy_hold": [round(x, 2) for x in bh],
        "naive": naive,
        "summary": {"hybrid": metrics(strategy), "buy_hold": metrics(bh), "naive": metrics(naive)},
    }


def build_prediction(symbol: str, timeframe: str):
    symbol = clean_symbol(symbol)
    raw = download_history(symbol)
    feat = indicators(raw)
    articles = get_news(symbol)
    sentiment, news = sentiment_score(articles)
    frame = make_training_frame(raw, sentiment)
    lstm_prob = lstm_direction(feat.dropna().reset_index(drop=True))
    signal, rf_prob, importance, model_comparison = classifier_prediction(frame, lstm_prob)
    lstm_prob = float(np.clip(lstm_prob, 0, 1)) if np.isfinite(lstm_prob) else 0.5
    rf_prob = float(np.clip(rf_prob, 0, 1)) if np.isfinite(rf_prob) else 0.5
    sentiment = float(np.clip(sentiment, -1, 1)) if np.isfinite(sentiment) else 0.0
    # Fuse model probabilities with sentiment/technical evidence.
    latest = feat.dropna().iloc[-1]
    # Add a small, explicit technical component. It previously was calculated
    # but never included in the ensemble, while neutral sentiment pulled scores
    # toward 0.5 and made the broad HOLD band dominate.
    technical_score = 0.0
    technical_factors = []
    if latest["close"] > latest["sma20"] > latest["sma50"]:
        technical_score += 0.45
        technical_factors.append("Bullish alignment: price is above the 20-day average, which is above the 50-day average")
    elif latest["close"] < latest["sma20"] < latest["sma50"]:
        technical_score -= 0.45
        technical_factors.append("Bearish alignment: price is below the 20-day average, which is below the 50-day average")
    else:
        technical_factors.append("Moving-average alignment is mixed")
    if latest["return_5d"] > 0.02:
        technical_score += 0.25
        technical_factors.append("Positive five-day momentum above 2%")
    elif latest["return_5d"] < -0.02:
        technical_score -= 0.25
        technical_factors.append("Negative five-day momentum below -2%")
    else:
        technical_factors.append("Five-day momentum is within ±2%")
    if latest["rsi"] < 35 and latest["macd"] > latest["macd_signal"]:
        technical_score += 0.30
        technical_factors.append("Oversold RSI with MACD above its signal")
    elif latest["rsi"] > 65 and latest["macd"] < latest["macd_signal"]:
        technical_score -= 0.30
        technical_factors.append("Overbought RSI with MACD below its signal")
    else:
        technical_factors.append("RSI/MACD reversal condition is not active")
    technical_probability = (float(np.clip(technical_score, -1, 1)) + 1) / 2
    sentiment_probability = (sentiment + 1) / 2
    components = {
        "lstm": {"probability_up": lstm_prob, "weight": 0.50, "weighted_contribution": 0.50 * lstm_prob},
        "random_forest": {"probability_up": rf_prob, "weight": 0.30, "weighted_contribution": 0.30 * rf_prob},
        "finbert": {"sentiment_score": sentiment, "probability_up": sentiment_probability, "weight": 0.10, "weighted_contribution": 0.10 * sentiment_probability},
        "technical": {"score": float(np.clip(technical_score, -1, 1)), "probability_up": technical_probability, "weight": 0.10, "weighted_contribution": 0.10 * technical_probability, "factors": technical_factors},
    }
    fused = float(np.clip(
        sum(component["weighted_contribution"] for component in components.values()),
        0, 1,
    ))
    signal = signal_from_score(fused)
    horizon_days = {"1D":1,"1W":5,"1M":21,"3M":63}.get(timeframe,1)
    daily_expected = (fused - 0.5) * 0.018 + latest["return_5d"] * 0.20
    predicted_change = float(daily_expected * np.sqrt(horizon_days) * 100)
    trend = "bullish" if fused >= 0.54 else "bearish" if fused <= 0.46 else "neutral"
    confidence = float(50 + abs(fused - 0.5) * 100)
    return {
        "stock_symbol": symbol,
        "company_name": symbol,
        "as_of": raw.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "latest_close": round(float(raw.iloc[-1]["close"]), 2),
        "signal_type": signal,
        "confidence": round(confidence, 1),
        "predicted_trend": trend,
        "predicted_change": round(predicted_change, 2),
        "sentiment_score": round(float(sentiment), 3),
        "lstm_probability_up": round(lstm_prob, 4),
        "classifier_probability_up": round(rf_prob, 4),
        "signal_explanation": {
            "ensemble_probability_up": round(fused, 4),
            "signal": signal,
            "components": {
                name: {
                    **{key: round(value, 4) for key, value in component.items() if isinstance(value, (int, float))},
                    **({"factors": component["factors"]} if "factors" in component else {}),
                }
                for name, component in components.items()
            },
        },
        "technical_indicators": {"rsi":round(float(latest["rsi"]),2),"macd":round(float(latest["macd"]),4),"macd_signal":round(float(latest["macd_signal"]),4),"sma_20":round(float(latest["sma20"]),2),"sma_50":round(float(latest["sma50"]),2),"bollinger_upper":round(float(latest["bb_upper"]),2),"bollinger_lower":round(float(latest["bb_lower"]),2)},
        "news": news[:8],
        "feature_importance": importance,
        "model_comparison": model_comparison,
        "analysis_summary": f"The hybrid model is {trend} for the {timeframe} horizon. It combines the LSTM trend probability, Random Forest classifier, finance-news sentiment and technical indicators using the latest available market bar.",
        "risk_factors": ["Market regime shifts can reduce out-of-sample performance.", "Financial data are non-stationary and black-swan events are difficult to model.", "News coverage may be incomplete or delayed.", "The predicted percentage is a model estimate, not a guaranteed return."],
        "model_recommendation": "Use the signal together with the backtest and baseline results rather than as a standalone trading decision.",
        "backtest": walk_forward(raw),
    }


@app.get("/api/health")
def health():
    return {"status":"ok", "time":datetime.now(timezone.utc).isoformat()}


@app.get("/api/config")
def get_tuning_config():
    return TUNING_CONFIG.copy()


@app.post("/api/config")
def save_tuning_config(config: TuningConfig):
    values = config.model_dump()
    with open(_tuning_config_path, "w", encoding="utf-8") as config_file:
        json.dump(values, config_file, indent=2)
    TUNING_CONFIG.update(values)
    return {"status": "saved", "config": TUNING_CONFIG.copy()}


@app.get("/api/predict/{symbol}")
def predict(symbol: str, timeframe: str = Query("1D", pattern="^(1D|1W|1M|3M)$")):
    try:
        return build_prediction(symbol, timeframe)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history/{symbol}")
def history(symbol: str, period: str = Query("1y", pattern="^(3mo|6mo|1y|2y|3y)$")):
    df = download_history(clean_symbol(symbol), period)
    return {
        "symbol": clean_symbol(symbol),
        "data": [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "open": round(float(r.open), 2),
                "high": round(float(r.high), 2),
                "low": round(float(r.low), 2),
                "close": round(float(r.close), 2),
                "volume": int(r.volume),
            }
            for r in df.itertuples()
        ],
    }


@app.get("/api/backtest/{symbol}")
def backtest(symbol: str):
    return walk_forward(download_history(clean_symbol(symbol)))


@app.get("/api/generalization")
def generalization():
    basket=[("AAPL","Technology"),("JNJ","Healthcare"),("JPM","Finance"),("XOM","Energy"),("PG","Consumer"),("GE","Industrial")]
    rows=[]
    for symbol,sector in basket:
        try:
            r=walk_forward(download_history(symbol, "2y"))
            h=r["summary"]["hybrid"]; b=r["summary"]["buy_hold"]
            rows.append({"ticker":symbol,"sector":sector,"return":h["total_return"],"sharpe":h["sharpe"],"max_drawdown":h["max_drawdown"],"edge":round(h["total_return"]-b["total_return"],2)})
        except Exception as e:
            rows.append({"ticker":symbol,"sector":sector,"error":str(e)})
    return rows

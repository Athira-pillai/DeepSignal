import json
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
try:  # Support both ``uvicorn main:app`` and ``from backend import main``.
    from .evaluation import evaluate_direction
    from .hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities
    from .backtesting import STRATEGY_VERSION, run_backtest
    from . import database as storage
    from . import scheduler
    from . import cache as response_cache
    from . import model_registry
except ImportError:
    from evaluation import evaluate_direction
    from hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities
    from backtesting import STRATEGY_VERSION, run_backtest
    import database as storage
    import scheduler
    import cache as response_cache
    import model_registry
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


@app.on_event("startup")
def start_background_jobs():
    scheduler.start_scheduler()


@app.on_event("shutdown")
def stop_background_jobs():
    scheduler.stop_scheduler()

FEATURES = ["return_1d", "return_5d", "sma20_gap", "sma50_gap", "rsi", "macd", "macd_signal", "bb_position", "volume_change", "sentiment"]
# Sentiment and LSTM probabilities are combined by the hybrid after their
# independent predictions; the forest is trained only on price/volume inputs.
CLASSIFIER_FEATURES = [feature for feature in FEATURES if feature != "sentiment"]
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


def _market_data_cache_ttl():
    try:
        return max(0, min(86400, int(os.getenv("MARKET_DATA_CACHE_TTL_SECONDS", "900"))))
    except ValueError:
        return 900


def download_history(symbol: str, period: str = "3y", *, force_refresh: bool = False) -> pd.DataFrame:
    symbol = clean_symbol(symbol)
    key = f"ohlcv:yahoo-auto-adjust-v1:{symbol}:1d:{period}"
    if not force_refresh:
        cached = response_cache.cache.get(key)
        if cached is not response_cache.MISSING:
            return cached
    t = yf.Ticker(symbol)
    provider_started = time.perf_counter()
    try:
        df = t.history(period=period, interval="1d", auto_adjust=True, actions=False)
    finally:
        response_cache.cache.record_provider_call(time.perf_counter() - provider_started)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No market data found for {symbol}.")
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        raise HTTPException(status_code=502, detail="Market provider returned no date column.")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    result = df[["date", "open", "high", "low", "close", "volume"]].dropna().copy()
    response_cache.cache.set(key, result, _market_data_cache_ttl())
    return result


FEATURE_CODE_VERSION = "technical-indicators-v1"
PREDICTION_MODEL_CODE_VERSION = "live-direction-models-v1"


def _feature_cache_key(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return f"features:{FEATURE_CODE_VERSION}:{digest.hexdigest()}"


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    key = _feature_cache_key(df)
    cached = response_cache.cache.get(key)
    if cached is not response_cache.MISSING:
        return cached
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
    response_cache.cache.set(key, out, 86400)
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


def lstm_direction(frame: pd.DataFrame, lookback=None, *, model_bundle=None, artifact_capture=None):
    # Lazy import keeps non-model endpoints fast.
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except (ImportError, OSError):
        # TensorFlow may be unavailable on some local Python installations.
        # Keep the hybrid prediction endpoint usable with a neutral LSTM input.
        if artifact_capture is not None:
            artifact_capture.setdefault("statuses", {})["lstm"] = "tensorflow_unavailable_neutral_fallback"
        return 0.5

    lookback = lookback or TUNING_CONFIG["lookback"]
    cols = ["close", "sma20", "sma50", "rsi", "macd", "bb_upper", "bb_lower"]
    data = frame[cols].dropna().values
    if model_bundle and "lstm" in model_bundle["models"]:
        model = model_bundle["models"]["lstm"]
        scaler = model_bundle["preprocessors"]["lstm_scaler"]
        scaled = scaler.transform(data).astype(np.float32)
        if len(scaled) < lookback:
            raise ValueError("Saved LSTM artifact needs more rows than the current input provides.")
        return float(model.predict(scaled[-lookback:][None, ...], verbose=0)[0][0])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback - 1, len(scaled) - 1):
        X.append(scaled[i-lookback+1:i+1])
        y.append(1 if frame.iloc[i+1]["close"] > frame.iloc[i]["close"] else 0)
    if len(X) < 80:
        if artifact_capture is not None:
            artifact_capture.setdefault("statuses", {})["lstm"] = "insufficient_training_sequences_neutral_fallback"
        return 0.5
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    split = max(int(len(X) * 0.85), len(X)-40)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = Sequential([LSTM(TUNING_CONFIG["lstm_units"], input_shape=(X.shape[1], X.shape[2])), Dropout(0.2), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]), epochs=TUNING_CONFIG["epochs"], batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
    latest = scaled[-lookback:][None, ...]
    if artifact_capture is not None:
        artifact_capture.setdefault("keras_models", {})["lstm"] = model
        artifact_capture.setdefault("preprocessors", {})["lstm_scaler"] = scaler
        artifact_capture.setdefault("statuses", {})["lstm"] = "trained"
        validation_accuracy = history.history.get("val_accuracy", [])
        artifact_capture.setdefault("validation_metrics", {})["lstm"] = {
            "best_validation_accuracy": float(max(validation_accuracy)) if validation_accuracy else None,
            "epochs_completed": len(history.history.get("loss", [])),
        }
    return float(model.predict(latest, verbose=0)[0][0])


def classifier_prediction(frame: pd.DataFrame, *, model_bundle=None, artifact_capture=None):
    train = frame.copy()
    cols = CLASSIFIER_FEATURES
    latest = train[cols].iloc[-1].to_dict()
    train = train.dropna(subset=cols + ["target"])
    if len(train) < 100:
        if artifact_capture is not None:
            artifact_capture.setdefault("statuses", {})["random_forest"] = "insufficient_samples_neutral_fallback"
        return 0.5, {c: 0.0 for c in cols}, {
            "random_forest": {"accuracy": None, "balanced_accuracy": None},
            "xgboost": {"accuracy": None},
            "evaluation": {"method": "chronological_holdout", "status": "insufficient_samples", "train_samples": int(len(train)), "test_samples": 0},
            "shap": {},
        }

    if model_bundle and "random_forest" in model_bundle["models"]:
        model = model_bundle["models"]["random_forest"]
        prob_up = float(model.predict_proba(pd.DataFrame([latest]))[0][list(model.classes_).index(1)])
        saved = model_bundle["metadata"].get("inference_metadata", {})
        return (
            prob_up,
            saved.get("feature_importance", {column: 0.0 for column in cols}),
            saved.get("model_comparison", {"random_forest": {"accuracy": None, "balanced_accuracy": None}}),
        )

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
    prob_up = float(rf.predict_proba(pd.DataFrame([latest]))[0][1])
    importance = {c: round(float(v), 4) for c, v in zip(cols, rf.feature_importances_)}
    if artifact_capture is not None:
        artifact_capture.setdefault("models", {})["random_forest"] = rf
        artifact_capture.setdefault("statuses", {})["random_forest"] = "trained"
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
        if artifact_capture is not None:
            artifact_capture.setdefault("models", {})["xgboost"] = xgb
            artifact_capture.setdefault("statuses", {})["xgboost"] = "trained"
        xgb_prob = float(xgb.predict_proba(pd.DataFrame([latest]))[0][1])
        model_comparison["xgboost"] = {"accuracy": xgb_accuracy, "balanced_accuracy": xgb_balanced_accuracy, "probability_up": round(xgb_prob, 4)}
    except Exception:
        if artifact_capture is not None:
            artifact_capture.setdefault("statuses", {})["xgboost"] = "unavailable_or_failed"
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
    return prob_up, importance, model_comparison


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


def walk_forward(df: pd.DataFrame, symbol="UNKNOWN"):
    """Generate after-close technical signals and delegate execution to the engine."""
    featured = indicators(df).reset_index(drop=True)
    signals = pd.DataFrame({"date": featured["date"], "signal": "HOLD"})
    buy = (featured["return_5d"] > 0.006) & (featured["rsi"] < 72)
    sell = (featured["return_5d"] < -0.006) & (featured["rsi"] > 28)
    signals.loc[buy, "signal"] = "BUY"
    signals.loc[sell, "signal"] = "SELL"
    return run_backtest(
        df,
        signals,
        symbol=symbol,
        config=TUNING_CONFIG.copy(),
        strategy_name="Five-day momentum and RSI; long-only",
        strategy_version=STRATEGY_VERSION,
        strategy_parameters={
            "buy_when": "return_5d > 0.006 and RSI < 72",
            "sell_when": "return_5d < -0.006 and RSI > 28",
            "otherwise": "HOLD",
        },
    )


def build_prediction(symbol: str, timeframe: str):
    symbol = clean_symbol(symbol)
    raw = download_history(symbol)
    data_retrieved_at = datetime.now(timezone.utc).isoformat()
    market_timestamp = pd.Timestamp(raw.iloc[-1]["date"]).tz_localize("UTC").isoformat()
    market_data_hash = hashlib.sha256(
        raw[["date", "open", "high", "low", "close", "volume"]]
        .to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S")
        .encode("utf-8")
    ).hexdigest()
    feat = indicators(raw)
    model_identity = {
        "symbol": symbol,
        "market_data_sha256": market_data_hash,
        "training_window": {
            "start": raw.iloc[0]["date"].strftime("%Y-%m-%d"),
            "end": raw.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "rows": int(len(raw)),
        },
        "target": "next session close > current session close; decision after current close",
        "target_horizon_sessions": 1,
        "features": {
            "lstm_ordered": ["close", "sma20", "sma50", "rsi", "macd", "bb_upper", "bb_lower"],
            "random_forest_ordered": CLASSIFIER_FEATURES,
        },
        "preprocessing_version": "lstm-minmax-fit-on-training-window-v1; random-forest-drop-missing-rows-v1",
        "feature_code_version": FEATURE_CODE_VERSION,
        "prediction_model_code_version": PREDICTION_MODEL_CODE_VERSION,
        "hybrid_version": HYBRID_VERSION,
        "sentiment_model": os.getenv("FINBERT_MODEL", "ProsusAI/finbert"),
        "configuration": TUNING_CONFIG.copy(),
        "random_seeds": {"lstm": 42, "random_forest": 42, "xgboost": 42},
        "package_versions": model_registry.package_versions(),
    }
    model_version = model_registry.version_for(model_identity)
    saved_bundle = model_registry.load_version(model_version) if model_registry.has_version(model_version) else None
    artifact_capture = {"models": {}, "preprocessors": {}, "keras_models": {}, "statuses": {}}
    articles = get_news(symbol)
    sentiment, news = sentiment_score(articles)
    frame = make_training_frame(raw, sentiment)
    lstm_prob = lstm_direction(
        feat.dropna().reset_index(drop=True), model_bundle=saved_bundle, artifact_capture=artifact_capture
    )
    rf_prob, importance, model_comparison = classifier_prediction(
        frame, model_bundle=saved_bundle, artifact_capture=artifact_capture
    )
    if saved_bundle is None:
        artifact_metadata = {
            "symbol": symbol,
            "data": {
                "source": "Yahoo Finance adjusted daily OHLCV",
                **model_identity["training_window"],
                "sha256": market_data_hash,
            },
            "target": {
                "name": "next_session_up_direction",
                "definition": model_identity["target"],
                "model_horizon_sessions": 1,
                "decision_timestamp_convention": "After the current exchange session close; all bars are UTC-normalized.",
            },
            "features": model_identity["features"],
            "input_shapes": {
                "lstm": [int(TUNING_CONFIG["lookback"]), 7],
                "random_forest": [len(CLASSIFIER_FEATURES)],
            },
            "preprocessing": {
                "lstm": "MinMaxScaler is fitted on the available ordered training window and saved as lstm_scaler.joblib.",
                "random_forest": "Rows with missing required features or target are dropped; no scaling is used.",
            },
            "hyperparameters": {
                "lstm": {
                    "lookback": int(TUNING_CONFIG["lookback"]),
                    "units": int(TUNING_CONFIG["lstm_units"]),
                    "epochs": int(TUNING_CONFIG["epochs"]),
                    "dropout": 0.2,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "loss": "binary_crossentropy",
                },
                "random_forest": {
                    "n_estimators": int(TUNING_CONFIG["rf_trees"]),
                    "max_depth": 7,
                    "min_samples_leaf": 4,
                    "class_weight": "balanced",
                },
            },
            "random_seeds": model_identity["random_seeds"],
            "package_versions": model_identity["package_versions"],
            "hybrid": {"version": HYBRID_VERSION, "weights": HYBRID_WEIGHTS.copy()},
            "sentiment_model": {
                "name": model_identity["sentiment_model"],
                "weights_location": "Pretrained model is loaded from the Transformers cache; it is not copied into this local trained-model artifact.",
            },
            "validation_metrics": {
                "random_forest": model_comparison.get("random_forest", {}),
                **artifact_capture.get("validation_metrics", {}),
            },
            "inference_metadata": {"feature_importance": importance, "model_comparison": model_comparison},
            "model_code_version": PREDICTION_MODEL_CODE_VERSION,
        }
        model_registry.save_version(model_version, model_identity, artifact_capture, artifact_metadata)
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
    hybrid = combine_probabilities(lstm_prob, rf_prob, sentiment, technical_probability)
    fused = hybrid["probability_up"]
    signal = hybrid["signal"]
    components = hybrid["components"]
    components["finbert"]["sentiment_score"] = sentiment
    components["technical"]["score"] = float(np.clip(technical_score, -1, 1))
    components["technical"]["factors"] = technical_factors
    horizon_days = {"1D":1,"1W":5,"1M":21,"3M":63}.get(timeframe,1)
    daily_expected = (fused - 0.5) * 0.018 + latest["return_5d"] * 0.20
    predicted_change = float(daily_expected * np.sqrt(horizon_days) * 100)
    trend = "bullish" if fused >= 0.54 else "bearish" if fused <= 0.46 else "neutral"
    confidence = float(50 + (hybrid["confidence"] - 0.5) * 100)
    prediction_timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "model_version": model_version,
        "stock_symbol": symbol,
        "company_name": symbol,
        "prediction_timestamp": prediction_timestamp,
        "market_data": {
            "source": "Yahoo Finance",
            "interval": "1d",
            "market_timestamp": market_timestamp,
            "retrieved_at": data_retrieved_at,
            "window_start": raw.iloc[0]["date"].strftime("%Y-%m-%d"),
            "window_end": raw.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "row_count": int(len(raw)),
            "sha256": market_data_hash,
            "quality_status": "ok",
        },
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
            "hybrid_version": HYBRID_VERSION,
            "model_version": model_version,
            "method": hybrid["method"],
            "target": hybrid["target"],
            "weights": hybrid["weights"],
            "sentiment_mapping": hybrid["sentiment_mapping"],
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
        "analysis_summary": f"The hybrid estimates next-session direction as {trend}. The {timeframe} predicted-change figure is a separate heuristic extrapolation. The direction score combines LSTM and Random Forest probabilities with current news sentiment and technical evidence.",
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


@app.get("/api/models")
def list_saved_model_versions(limit: int = Query(50, ge=1, le=200)):
    return model_registry.list_versions(limit)


@app.get("/api/models/{model_version}")
def saved_model_metadata(model_version: str):
    try:
        return model_registry.read_metadata(model_version)
    except model_registry.ModelArtifactError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/models/{model_version}/load")
def load_saved_model_version(model_version: str):
    """Validate and warm a saved version for requests with its exact data identity."""
    try:
        bundle = model_registry.load_version(model_version)
    except model_registry.ModelArtifactError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    metadata = bundle["metadata"]
    return {
        "status": "loaded",
        "model_version": model_version,
        "symbol": metadata.get("symbol"),
        "model_statuses": metadata.get("model_statuses", {}),
        "loaded_components": sorted(bundle["models"]),
        "artifact_location": metadata.get("artifact_location"),
        "note": "Predictions reuse this model only when symbol, market-data fingerprint, configuration, code version, and package versions match.",
    }


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
        result = build_prediction(symbol, timeframe)
        fallback_path = os.path.join(os.path.dirname(__file__), "prediction_runs.json")
        try:
            result["persistence"] = storage.save_prediction_bundle(result, fallback_path)
        except storage.StorageError as exc:
            result["persistence"] = {"status": "failed", "mode": "configured_storage", "message": str(exc)}
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history/predictions")
def prediction_history(
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    fallback_path = os.path.join(os.path.dirname(__file__), "prediction_runs.json")
    try:
        return storage.list_predictions(symbol, limit, fallback_path)
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/history/experiments")
def experiment_history(
    kind: Optional[str] = Query(None, pattern="^(evaluation|backtest)$"),
    limit: int = Query(50, ge=1, le=200),
):
    try:
        return storage.list_experiments(kind, limit, os.path.dirname(__file__))
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


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


@app.get("/api/company/{symbol}")
def company_name(symbol: str):
    ticker = clean_symbol(symbol)
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,16}", ticker):
        raise HTTPException(status_code=422, detail="Enter a valid stock ticker.")

    cache_key = f"company-name:yahoo-v1:{ticker}"
    cached = response_cache.cache.get(cache_key)
    if cached is not response_cache.MISSING:
        return cached

    provider_started = time.perf_counter()
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    finally:
        response_cache.cache.record_provider_call(time.perf_counter() - provider_started)

    name = None
    if isinstance(info, dict):
        name = info.get("longName") or info.get("shortName") or info.get("displayName")
    result = {
        "symbol": ticker,
        "company_name": str(name).strip() if name else None,
        "found": bool(name),
    }
    response_cache.cache.set(cache_key, result, 86400 if name else 1800)
    return result


@app.get("/api/backtest/{symbol}")
def backtest(symbol: str):
    clean = clean_symbol(symbol)
    raw = download_history(clean)
    report_path = os.path.join(os.path.dirname(__file__), "backtest_runs.json")
    report = walk_forward(raw, symbol=clean)
    try:
        report["persistence"] = storage.save_experiment(report, "backtest", report_path)
    except storage.StorageError as exc:
        report["persistence"] = {"status": "failed", "mode": "configured_storage", "message": str(exc)}
    return report


@app.get("/api/evaluate/{symbol}")
def evaluate(symbol: str):
    """Create and persist a repeatable next-session direction evaluation report."""
    clean = clean_symbol(symbol)
    raw = download_history(clean, "3y")
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_runs.json")
    report = evaluate_direction(clean, raw, TUNING_CONFIG.copy(), None, indicators)
    try:
        report["persistence"] = storage.save_experiment(report, "evaluation", report_path)
    except storage.StorageError as exc:
        report["persistence"] = {"status": "failed", "mode": "configured_storage", "message": str(exc)}
    return report


@app.get("/api/storage/status")
def storage_status():
    try:
        status = storage.check_connection()
        if status["status"] == "not_configured":
            status["mode"] = "local_json_fallback"
        return status
    except storage.StorageError as exc:
        return {"status": "error", "persistent_storage": False, "message": str(exc)}


@app.get("/api/jobs/market-data")
def market_data_job_status():
    return scheduler.scheduler_status()


@app.get("/api/cache/status")
def cache_status():
    return {
        "status": "in_process",
        "market_data_ttl_seconds": _market_data_cache_ttl(),
        "feature_ttl_seconds": 86400,
        "feature_code_version": FEATURE_CODE_VERSION,
        "cache": response_cache.cache.status(),
        "note": "Cache is local to this backend process and clears on restart. Predictions are not cached because news and input data can change.",
    }


@app.post("/api/storage/import-local-experiments")
def import_local_experiments():
    if storage.configuration_status() != "configured":
        raise HTTPException(status_code=400, detail="Configure Supabase before importing local experiment reports.")
    return storage.migrate_local_experiments(os.path.dirname(__file__))


@app.get("/api/generalization")
def generalization():
    basket=[("AAPL","Technology"),("JNJ","Healthcare"),("JPM","Finance"),("XOM","Energy"),("PG","Consumer"),("GE","Industrial")]
    rows=[]
    for symbol,sector in basket:
        try:
            r=walk_forward(download_history(symbol, "2y"))
            h=r["summary"]["strategy"]; b=r["summary"]["buy_hold"]
            rows.append({"ticker":symbol,"sector":sector,"return":h["total_return_pct"],"sharpe":h["annualized_sharpe"],"max_drawdown":h["max_drawdown_pct"],"edge":round(h["total_return_pct"]-b["total_return_pct"],2)})
        except Exception as e:
            rows.append({"ticker":symbol,"sector":sector,"error":str(e)})
    return rows

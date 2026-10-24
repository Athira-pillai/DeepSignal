"""Repeatable, chronological evaluation for next-session stock direction."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from .hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities
except ImportError:
    from hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "sma20_gap", "sma50_gap", "rsi",
    "macd", "macd_signal", "bb_position", "volume_change",
]
MODEL_VERSION = "direction-rf-v1"
LSTM_VERSION = "direction-lstm-v1"


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_up": round(float(precision), 4),
        "recall_up": round(float(recall), 4),
        "f1_up": round(float(f1), 4),
        "confusion_matrix_labels": ["down_or_flat", "up"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _trading_metrics(predictions, forward_returns, dates, cost_pct, slippage_pct):
    # Decide after today's close, then hold cash or go long for the next session.
    position = np.asarray(predictions, dtype=float)
    returns = np.asarray(forward_returns, dtype=float)
    changes = np.abs(np.diff(np.r_[0.0, position]))
    cost = (cost_pct + slippage_pct) / 100.0
    strategy_returns = position * returns - changes * cost
    equity = np.cumprod(1.0 + strategy_returns)
    curve = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(curve)
    drawdown = (curve / peaks) - 1.0
    std = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = (float(np.mean(strategy_returns)) / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "start": dates[0],
        "end": dates[-1],
        "sessions": int(len(returns)),
        "cumulative_return_pct": round(float((curve[-1] - 1) * 100), 4),
        "max_drawdown_pct": round(float(np.min(drawdown) * 100), 4),
        "annualized_sharpe": round(float(sharpe), 4),
        "turnover": round(float(np.sum(changes)), 4),
        "transaction_cost_pct": cost_pct,
        "slippage_pct": slippage_pct,
    }


def _save_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, encoding="utf-8") as handle:
            runs = json.load(handle)
        if not isinstance(runs, list):
            runs = []
    except (OSError, ValueError):
        runs = []
    runs.append(report)
    fd, temp_path = tempfile.mkstemp(prefix="evaluation-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(runs, handle, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _lstm_holdout_probabilities(frame, train_end, fit_end, test_start, test_end, config):
    """Fit sequence model using only pre-test labels; scale from train rows only."""
    columns = ["close", "sma20", "sma50", "rsi", "macd", "bb_upper", "bb_lower"]
    lookback = int(config["lookback"])
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM

        if fit_end <= lookback + 20:
            return np.full(test_end - test_start, 0.5), "insufficient_training_sequences"
        scaler = MinMaxScaler().fit(frame.iloc[:train_end][columns])
        scaled = scaler.transform(frame[columns]).astype(np.float32)
        X, y, ends = [], [], []
        for i in range(lookback - 1, fit_end):
            X.append(scaled[i - lookback + 1:i + 1])
            y.append(int(frame.iloc[i]["target"]))
            ends.append(i)
        X, y, ends = np.asarray(X), np.asarray(y, dtype=np.float32), np.asarray(ends)
        train_mask = ends < int(fit_end * 0.8)
        val_mask = ~train_mask
        if train_mask.sum() < 20 or val_mask.sum() < 5 or np.unique(y[train_mask]).size < 2:
            return np.full(test_end - test_start, 0.5), "insufficient_training_sequences"
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        model = Sequential([
            LSTM(int(config["lstm_units"]), input_shape=(lookback, len(columns))),
            Dropout(0.2), Dense(16, activation="relu"), Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(
            X[train_mask], y[train_mask],
            validation_data=(X[val_mask], y[val_mask]),
            epochs=int(config["epochs"]), batch_size=32, verbose=0,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        )
        test_X = np.asarray([
            scaled[i - lookback + 1:i + 1] for i in range(test_start, test_end)
        ], dtype=np.float32)
        probabilities = model.predict(test_X, verbose=0).reshape(-1)
        return probabilities.astype(float), "trained"
    except (ImportError, OSError):
        return np.full(test_end - test_start, 0.5), "tensorflow_unavailable_neutral_fallback"


def _technical_probabilities(frame):
    scores = np.zeros(len(frame), dtype=float)
    bullish = (frame["close"] > frame["sma20"]) & (frame["sma20"] > frame["sma50"])
    bearish = (frame["close"] < frame["sma20"]) & (frame["sma20"] < frame["sma50"])
    scores += np.where(bullish, 0.45, np.where(bearish, -0.45, 0.0))
    scores += np.where(frame["return_5d"] > 0.02, 0.25, np.where(frame["return_5d"] < -0.02, -0.25, 0.0))
    scores += np.where(
        (frame["rsi"] < 35) & (frame["macd"] > frame["macd_signal"]), 0.30,
        np.where((frame["rsi"] > 65) & (frame["macd"] < frame["macd_signal"]), -0.30, 0.0),
    )
    return (np.clip(scores, -1, 1) + 1) / 2


def evaluate_direction(symbol, raw, config, report_path=None, indicator_builder=None):
    """Fit on earlier rows, compare fixed baselines/models on the same final holdout."""
    if indicator_builder is None:
        raise ValueError("indicator_builder is required")
    frame = indicator_builder(raw)
    frame["target"] = (frame["close"].shift(-1) > frame["close"]).astype(float)
    frame["forward_return"] = frame["close"].shift(-1) / frame["close"] - 1
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["target", "forward_return"]
    ).reset_index(drop=True)
    n = len(frame)
    if n < 120:
        return {
            "status": "insufficient_data", "symbol": symbol,
            "required_samples": 120, "available_samples": n,
        }

    # The last 20% is held out. The first 60% is the fitting window and the
    # intervening 20% is a validation window reserved for future model choice.
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    fit = frame.iloc[:validation_end]
    test = frame.iloc[validation_end:]
    y_train = fit["target"].astype(int)
    y_test = test["target"].astype(int).to_numpy()
    if y_train.nunique() < 2:
        return {"status": "insufficient_class_diversity", "symbol": symbol, "available_samples": n}

    # Previous-session direction is a transparent, no-fit baseline.
    baseline_pred = (test["return_1d"].fillna(0).to_numpy() > 0).astype(int)
    rf = RandomForestClassifier(
        n_estimators=int(config["rf_trees"]), max_depth=7, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf.fit(fit[FEATURE_COLUMNS], y_train)
    rf_pred = rf.predict(test[FEATURE_COLUMNS]).astype(int)
    rf_prob = rf.predict_proba(test[FEATURE_COLUMNS])[:, list(rf.classes_).index(1)]

    lstm_prob, lstm_status = _lstm_holdout_probabilities(
        frame, train_end, validation_end, validation_end, n, config
    )
    technical_prob = np.asarray(_technical_probabilities(test), dtype=float)
    # Historical point-in-time news is unavailable. Neutral sentiment is used
    # consistently for every historical hybrid row and disclosed in the report.
    hybrid_prob = np.asarray([
        combine_probabilities(lp, rp, 0.0, tp)["probability_up"]
        for lp, rp, tp in zip(lstm_prob, rf_prob, technical_prob)
    ])

    dates = [d.strftime("%Y-%m-%d") for d in test["date"]]
    forward_returns = test["forward_return"].to_numpy(dtype=float)
    results = {}
    model_predictions = (
        ("previous_direction", baseline_pred),
        ("random_forest", rf_pred),
        ("lstm", (lstm_prob >= 0.5).astype(int)),
        ("hybrid", (hybrid_prob >= 0.5).astype(int)),
    )
    for name, predictions in model_predictions:
        results[name] = {
            "classification": _metrics(y_test, predictions),
            "trading": _trading_metrics(
                predictions, forward_returns, dates,
                float(config["transaction_cost_pct"]), float(config["slippage_pct"]),
            ),
        }
    hybrid_metrics = results["hybrid"]
    comparison_to_components = {}
    for name in ("previous_direction", "random_forest", "lstm"):
        comparison_to_components[name] = {
            "accuracy_delta": round(
                hybrid_metrics["classification"]["accuracy"] - results[name]["classification"]["accuracy"], 4
            ),
            "cumulative_return_delta_pct": round(
                hybrid_metrics["trading"]["cumulative_return_pct"] - results[name]["trading"]["cumulative_return_pct"], 4
            ),
        }
    buy_hold = _trading_metrics(
        np.ones(len(test), dtype=int), forward_returns, dates,
        float(config["transaction_cost_pct"]), float(config["slippage_pct"]),
    )
    flat = _trading_metrics(
        np.zeros(len(test), dtype=int), forward_returns, dates,
        float(config["transaction_cost_pct"]), float(config["slippage_pct"]),
    )
    data_hash = hashlib.sha256(
        raw[["date", "open", "high", "low", "close", "volume"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "status": "ok",
        "symbol": symbol,
        "task": "next_session_direction",
        "data": {
            "source": "Yahoo Finance adjusted daily OHLCV",
            "rows_used": n,
            "sha256": data_hash,
            "first_date": frame.iloc[0]["date"].strftime("%Y-%m-%d"),
            "last_labeled_date": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        },
        "split": {
            "method": "chronological_train_validation_test",
            "train_samples": train_end,
            "validation_samples": validation_end - train_end,
            "fit_samples": len(fit),
            "test_samples": len(test),
            "test_start": dates[0],
            "test_end": dates[-1],
            "models_share_test_window": True,
        },
        "models": results,
        "hybrid_comparison_deltas": comparison_to_components,
        "probability_diagnostics": {
            "lstm_status": lstm_status,
            "lstm_mean_probability_up": round(float(np.mean(lstm_prob)), 4),
            "random_forest_mean_probability_up": round(float(np.mean(rf_prob)), 4),
            "hybrid_mean_probability_up": round(float(np.mean(hybrid_prob)), 4),
        },
        "trading_baselines": {"buy_and_hold": buy_hold, "flat_cash": flat},
        "configuration": {
            "random_forest_trees": int(config["rf_trees"]),
            "lstm_lookback": int(config["lookback"]),
            "lstm_units": int(config["lstm_units"]),
            "lstm_epochs": int(config["epochs"]),
            "hybrid_weights": HYBRID_WEIGHTS,
            "hybrid_version": HYBRID_VERSION,
            "historical_sentiment_score": 0.0,
            "max_depth": 7,
            "min_samples_leaf": 4,
            "random_state": 42,
            "transaction_cost_pct": float(config["transaction_cost_pct"]),
            "slippage_pct": float(config["slippage_pct"]),
            "features": FEATURE_COLUMNS,
            "target": "next close > current close",
        },
        "versions": {
            "application": "1.0.0",
            "evaluation_model": MODEL_VERSION,
            "lstm_model": LSTM_VERSION,
            "hybrid_model": HYBRID_VERSION,
            "python_packages": {
                name: _package_version(name)
                for name in ("numpy", "pandas", "scikit-learn", "yfinance", "tensorflow", "xgboost", "transformers")
            },
        },
        "limitations": [
            "Historical point-in-time news sentiment is unavailable; neutral sentiment is used in every hybrid test row.",
            "The LSTM is trained once before the final test window; its scaler is fit only on the earlier 60% training rows.",
            "The fixed Random Forest configuration is fit on the earlier 80% train plus validation rows.",
            "Trading assumes close-time decisions execute for the next session and uses simplified proportional costs.",
        ],
    }
    if report_path:
        _save_report(report_path, report)
    return report

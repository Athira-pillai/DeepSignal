"""Leakage-aware walk-forward evaluation for next-session stock direction."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

try:
    from .hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities
except ImportError:
    from hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, combine_probabilities


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "sma20_gap", "sma50_gap", "rsi",
    "macd", "macd_signal", "bb_position", "volume_change",
]
SEQUENCE_COLUMNS = ["close", "sma20", "sma50", "rsi", "macd", "bb_upper", "bb_lower"]
MODEL_VERSION = "direction-rf-walk-forward-v2"
LSTM_VERSION = "direction-lstm-walk-forward-v2"
MINIMUM_SAMPLES = 200


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def generate_walk_forward_folds(
    n_samples,
    final_holdout_fraction=0.20,
    minimum_training_fraction=0.50,
    validation_fraction=0.10,
    test_fraction=0.10,
):
    """Return expanding-window folds and a final holdout boundary.

    The final holdout is excluded from every fold. For each development fold,
    training precedes a validation window, which immediately precedes test.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        return {"status": "insufficient_data", "folds": [], "final_holdout_start": 0}
    if not 0 < final_holdout_fraction < 1:
        raise ValueError("final_holdout_fraction must be between 0 and 1")
    if not 0 < minimum_training_fraction < 1:
        raise ValueError("minimum_training_fraction must be between 0 and 1")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive")

    final_holdout_start = n_samples - int(np.ceil(n_samples * final_holdout_fraction))
    development_size = final_holdout_start
    minimum_training = int(development_size * minimum_training_fraction)
    validation_size = max(1, int(development_size * validation_fraction))
    test_size = max(1, int(development_size * test_fraction))
    first_test_start = minimum_training + validation_size
    folds = []
    test_start = first_test_start
    fold_number = 1
    available_test_rows = development_size - first_test_start
    fold_count = max(1, int(round(available_test_rows / test_size))) if available_test_rows > 0 else 0
    base_test_size = available_test_rows // fold_count if fold_count else 0
    extra_rows = available_test_rows % fold_count if fold_count else 0
    for fold_index in range(fold_count):
        validation_start = test_start - validation_size
        current_test_size = base_test_size + (1 if fold_index < extra_rows else 0)
        test_end = test_start + current_test_size
        if validation_start <= 0 or test_end <= test_start:
            break
        folds.append({
            "fold": fold_number,
            "train_start": 0,
            "train_end": validation_start,
            "validation_start": validation_start,
            "validation_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_number += 1
        test_start = test_end

    return {
        "status": "ok" if len(folds) >= 2 else "insufficient_folds",
        "folds": folds,
        "final_holdout_start": final_holdout_start,
        "development_end": development_size,
        "minimum_training_samples": minimum_training,
        "validation_samples": validation_size,
        "nominal_test_samples": test_size,
        "test_fold_sizes": [fold["test_end"] - fold["test_start"] for fold in folds],
    }


def _probability_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return {"brier_score": None, "roc_auc": None, "reliability": []}
    auc = None
    if np.unique(y_true).size == 2:
        auc = round(float(roc_auc_score(y_true, probabilities)), 4)
    reliability = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        mask = (probabilities >= left) & (
            probabilities <= right if index == len(edges) - 2 else probabilities < right
        )
        if mask.any():
            reliability.append({
                "bin_start": round(float(left), 1),
                "bin_end": round(float(right), 1),
                "count": int(mask.sum()),
                "mean_probability": round(float(probabilities[mask].mean()), 4),
                "observed_up_rate": round(float(y_true[mask].mean()), 4),
            })
    return {
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "roc_auc": auc,
        "reliability": reliability,
    }


def _metrics(y_true, y_pred, y_probability=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_probability is None:
        y_probability = y_pred.astype(float)
    y_probability = np.asarray(y_probability, dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", zero_division=0
    )
    classes, counts = np.unique(y_true, return_counts=True)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": (
            round(float(balanced_accuracy_score(y_true, y_pred)), 4)
            if len(classes) == 2 else None
        ),
        "precision_up": round(float(precision), 4),
        "recall_up": round(float(recall), 4),
        "f1_up": round(float(f1), 4),
        "class_counts": {"down_or_flat": int(counts[list(classes).index(0)]) if 0 in classes else 0,
                         "up": int(counts[list(classes).index(1)]) if 1 in classes else 0},
        "confusion_matrix_labels": ["down_or_flat", "up"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        **_probability_metrics(y_true, y_probability),
    }


def _trading_metrics(predictions, forward_returns, dates, cost_pct, slippage_pct):
    """Simplified close-decision long-or-cash comparison on each test window."""
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
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        with open(path, encoding="utf-8") as handle:
            runs = json.load(handle)
        if not isinstance(runs, list):
            runs = []
    except (OSError, ValueError):
        runs = []
    runs.append(report)
    fd, temp_path = tempfile.mkstemp(prefix="evaluation-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(runs, handle, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _fit_scale_sequence_features(frame, train_end):
    """Fit the sequence scaler on train feature rows, then transform rows."""
    scaler = MinMaxScaler().fit(frame.iloc[:train_end][SEQUENCE_COLUMNS])
    return scaler, scaler.transform(frame[SEQUENCE_COLUMNS]).astype(np.float32)


def _lstm_holdout_probabilities(frame, train_end, validation_end, test_start, test_end, config):
    """Train on the pre-validation window and early-stop on validation only."""
    lookback = int(config["lookback"])
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

        if train_end <= lookback + 20:
            return np.full(test_end - test_start, 0.5), "insufficient_training_sequences"
        _, scaled = _fit_scale_sequence_features(frame, train_end)
        X, y, ends = [], [], []
        for index in range(lookback - 1, validation_end):
            X.append(scaled[index - lookback + 1:index + 1])
            y.append(int(frame.iloc[index]["target"]))
            ends.append(index)
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float32)
        ends = np.asarray(ends)
        train_mask = ends < train_end
        validation_mask = (ends >= train_end) & (ends < validation_end)
        if (
            train_mask.sum() < 20 or validation_mask.sum() < 5
            or np.unique(y[train_mask]).size < 2
        ):
            return np.full(test_end - test_start, 0.5), "insufficient_training_sequences"
        import tensorflow as tf
        tf.keras.utils.set_random_seed(int(config.get("random_seed", 42)))
        model = Sequential([
            Input(shape=(lookback, len(SEQUENCE_COLUMNS))),
            LSTM(int(config["lstm_units"])),
            Dropout(0.2), Dense(16, activation="relu"), Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(
            X[train_mask], y[train_mask],
            validation_data=(X[validation_mask], y[validation_mask]),
            epochs=int(config["epochs"]), batch_size=32, verbose=0,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        )
        test_X = np.asarray([
            scaled[index - lookback + 1:index + 1] for index in range(test_start, test_end)
        ], dtype=np.float32)
        probabilities = model.predict(test_X, verbose=0).reshape(-1)
        return probabilities.astype(float), "trained"
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return np.full(test_end - test_start, 0.5), f"lstm_fallback_{type(exc).__name__}"


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


def _add_next_session_target(frame):
    """Attach close-to-close next-session labels and forward returns."""
    result = frame.copy()
    result["target"] = (result["close"].shift(-1) > result["close"]).astype(float)
    result["forward_return"] = result["close"].shift(-1) / result["close"] - 1
    return result


def _positive_class_probability(model, features):
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(features), dtype=float)


def _evaluate_window(frame, fit_end, train_end, validation_end, test_start, test_end, config):
    fit = frame.iloc[:fit_end]
    test = frame.iloc[test_start:test_end]
    y_fit = fit["target"].astype(int)
    y_test = test["target"].astype(int).to_numpy()
    if y_fit.nunique() < 2:
        return {"status": "insufficient_class_diversity", "models": {}}

    baseline_probability = (test["return_1d"].fillna(0).to_numpy() > 0).astype(float)
    rf = RandomForestClassifier(
        n_estimators=int(config["rf_trees"]), max_depth=7, min_samples_leaf=4,
        random_state=int(config.get("random_seed", 42)), class_weight="balanced",
    )
    rf.fit(fit[FEATURE_COLUMNS], y_fit)
    rf_probability = _positive_class_probability(rf, test[FEATURE_COLUMNS])
    lstm_probability, lstm_status = _lstm_holdout_probabilities(
        frame, train_end, validation_end, test_start, test_end, config
    )
    technical_probability = np.asarray(_technical_probabilities(test), dtype=float)
    hybrid_probability = np.asarray([
        combine_probabilities(lstm, forest, 0.0, technical)["probability_up"]
        for lstm, forest, technical in zip(lstm_probability, rf_probability, technical_probability)
    ])

    probabilities = {
        "previous_direction": baseline_probability,
        "random_forest": rf_probability,
        "lstm": lstm_probability,
        "hybrid": hybrid_probability,
    }
    statuses = {"lstm": lstm_status}
    if config.get("include_xgboost", False):
        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                n_estimators=int(config.get("xgboost_trees", 200)), max_depth=4,
                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=int(config.get("random_seed", 42)), n_jobs=1,
                eval_metric="logloss", verbosity=0,
            )
            xgb.fit(fit[FEATURE_COLUMNS], y_fit)
            probabilities["xgboost"] = _positive_class_probability(xgb, test[FEATURE_COLUMNS])
            statuses["xgboost"] = "trained"
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            statuses["xgboost"] = f"unavailable_{type(exc).__name__}"
    else:
        statuses["xgboost"] = "disabled_by_configuration"

    y_test_probabilities = {}
    for name, values in probabilities.items():
        y_test_probabilities[name] = {
            "predictions": (values >= 0.5).astype(int),
            "probabilities": values,
            "classification": _metrics(y_test, (values >= 0.5).astype(int), values),
        }
    dates = [date.strftime("%Y-%m-%d") for date in test["date"]]
    forward_returns = test["forward_return"].to_numpy(dtype=float)
    cost_pct = float(config["transaction_cost_pct"])
    slippage_pct = float(config["slippage_pct"])
    for result in y_test_probabilities.values():
        result["trading"] = _trading_metrics(
            result["predictions"], forward_returns, dates, cost_pct, slippage_pct
        )
    y_test_probabilities["buy_and_hold"] = {
        "trading": _trading_metrics(np.ones(len(test)), forward_returns, dates, cost_pct, slippage_pct)
    }
    y_test_probabilities["flat_cash"] = {
        "trading": _trading_metrics(np.zeros(len(test)), forward_returns, dates, cost_pct, slippage_pct)
    }
    return {
        "status": "ok",
        "models": y_test_probabilities,
        "component_status": statuses,
        "target_class_counts": {
            "down_or_flat": int(np.sum(y_test == 0)), "up": int(np.sum(y_test == 1)),
        },
        "test_start": dates[0],
        "test_end": dates[-1],
    }


def _aggregate_folds(folds):
    aggregate = {}
    model_names = sorted({
        name for fold in folds if fold.get("status") == "ok"
        for name in fold.get("models", {})
    })
    metric_paths = (
        ("classification", "accuracy"), ("classification", "balanced_accuracy"),
        ("classification", "precision_up"), ("classification", "recall_up"),
        ("classification", "f1_up"), ("classification", "roc_auc"),
        ("classification", "brier_score"), ("trading", "cumulative_return_pct"),
        ("trading", "max_drawdown_pct"), ("trading", "annualized_sharpe"),
    )
    for name in model_names:
        per_metric = {}
        for section, metric in metric_paths:
            values = [
                fold["models"][name].get(section, {}).get(metric)
                for fold in folds
                if fold.get("status") == "ok" and name in fold.get("models", {})
            ]
            values = [float(value) for value in values if value is not None and np.isfinite(value)]
            if values:
                per_metric[f"{section}.{metric}"] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
                    "folds": len(values),
                }
        aggregate[name] = per_metric
    return aggregate


def _compact_window_result(result):
    """Remove in-memory prediction arrays before a result enters a report."""
    if result.get("status") != "ok":
        return {"status": result.get("status", "failed"), "models": {}}
    return {
        "status": "ok",
        "models": {
            name: {
                key: value
                for key, value in model.items()
                if key in {"classification", "trading"}
            }
            for name, model in result["models"].items()
        },
        "component_status": result.get("component_status", {}),
        "target_class_counts": result.get("target_class_counts", {}),
        "test_start_date": result.get("test_start"),
        "test_end_date": result.get("test_end"),
    }


def evaluate_direction(symbol, raw, config, report_path=None, indicator_builder=None):
    """Compare models on expanding folds and one untouched final holdout."""
    if indicator_builder is None:
        raise ValueError("indicator_builder is required")
    frame = _add_next_session_target(indicator_builder(raw))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + SEQUENCE_COLUMNS + ["target", "forward_return"]
    ).reset_index(drop=True)
    n = len(frame)
    split_plan = generate_walk_forward_folds(
        n,
        final_holdout_fraction=float(config.get("final_holdout_fraction", 0.20)),
        minimum_training_fraction=float(config.get("minimum_training_fraction", 0.50)),
        validation_fraction=float(config.get("validation_fraction", 0.10)),
        test_fraction=float(config.get("test_fraction", 0.10)),
    )
    if n < MINIMUM_SAMPLES or split_plan["status"] != "ok":
        return {
            "status": "insufficient_data" if n < MINIMUM_SAMPLES else "insufficient_walk_forward_history",
            "symbol": symbol,
            "required_samples": MINIMUM_SAMPLES,
            "available_samples": n,
            "walk_forward_status": split_plan["status"],
            "required_folds": 2,
            "available_folds": len(split_plan["folds"]),
        }

    folds = []
    for boundaries in split_plan["folds"]:
        result = _evaluate_window(
            frame,
            fit_end=boundaries["train_end"],
            train_end=boundaries["train_end"],
            validation_end=boundaries["validation_end"],
            test_start=boundaries["test_start"],
            test_end=boundaries["test_end"],
            config=config,
        )
        fold_report = {
            **boundaries,
            "train_start_date": frame.iloc[boundaries["train_start"]]["date"].strftime("%Y-%m-%d"),
            "train_end_date": frame.iloc[boundaries["train_end"] - 1]["date"].strftime("%Y-%m-%d"),
            "validation_start_date": frame.iloc[boundaries["validation_start"]]["date"].strftime("%Y-%m-%d"),
            "validation_end_date": frame.iloc[boundaries["validation_end"] - 1]["date"].strftime("%Y-%m-%d"),
            "train_samples": boundaries["train_end"] - boundaries["train_start"],
            "validation_samples": boundaries["validation_end"] - boundaries["validation_start"],
            "test_samples": boundaries["test_end"] - boundaries["test_start"],
            "model_fit_end": boundaries["train_end"],
            "model_fit_end_date": frame.iloc[boundaries["train_end"] - 1]["date"].strftime("%Y-%m-%d"),
            "model_fit_samples": boundaries["train_end"],
            **_compact_window_result(result),
        }
        folds.append(fold_report)

    final_start = split_plan["final_holdout_start"]
    development_end = final_start
    final_train_end = int(development_end * 0.8)
    final_result = _evaluate_window(
        frame,
        fit_end=development_end,
        train_end=final_train_end,
        validation_end=development_end,
        test_start=final_start,
        test_end=n,
        config=config,
    )
    if final_result["status"] != "ok":
        return {
            "status": final_result["status"], "symbol": symbol,
            "available_samples": n, "final_holdout_start": final_start,
        }

    raw_columns = ["date", "open", "high", "low", "close", "volume"]
    data_hash = hashlib.sha256(
        raw[raw_columns].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    final_model_outputs = final_result["models"]
    model_results = _compact_window_result(final_result)["models"]
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "status": "ok",
        "symbol": symbol,
        "task": "next_session_direction",
        "evaluation_version": "walk-forward-v2",
        "data": {
            "source": config.get("data_source", "Yahoo Finance adjusted daily OHLCV"),
            "rows_used": n,
            "sha256": data_hash,
            "first_date": frame.iloc[0]["date"].strftime("%Y-%m-%d"),
            "last_labeled_date": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
        },
        "split": {
            "method": "expanding_train_validation_test_with_final_holdout",
            "walk_forward_folds": len(folds),
            "final_holdout_fraction": float(config.get("final_holdout_fraction", 0.20)),
            "final_holdout_start_index": final_start,
            "final_holdout_samples": n - final_start,
            "final_training_samples": final_train_end,
            "final_validation_samples": development_end - final_train_end,
            "final_fit_samples": development_end,
            "final_holdout_start": frame.iloc[final_start]["date"].strftime("%Y-%m-%d"),
            "final_holdout_end": frame.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "final_holdout_untouched_for_selection": True,
            "models_share_test_dates": True,
        },
        "walk_forward": {
            "status": "ok" if all(fold["status"] == "ok" for fold in folds) else "partial",
            "folds": folds,
            "aggregate": _aggregate_folds(folds),
        },
        # Retain the original models field as the one-time final-holdout result.
        "models": model_results,
        "component_status": final_result["component_status"],
        "probability_diagnostics": {
            name: {
                "mean_probability_up": round(float(np.mean(result["probabilities"])), 4),
                "status": final_result["component_status"].get(name, "available"),
            }
            for name, result in final_model_outputs.items() if "probabilities" in result
        },
        "configuration": {
            "random_forest_trees": int(config["rf_trees"]),
            "lstm_lookback": int(config["lookback"]),
            "lstm_units": int(config["lstm_units"]),
            "lstm_epochs": int(config["epochs"]),
            "hybrid_weights": HYBRID_WEIGHTS,
            "hybrid_version": HYBRID_VERSION,
            "historical_sentiment_score": 0.0,
            "historical_sentiment_status": "unavailable_neutral_fallback",
            "xgboost_enabled": bool(config.get("include_xgboost", False)),
            "xgboost_trees": int(config.get("xgboost_trees", 200)),
            "random_seed": int(config.get("random_seed", 42)),
            "preprocessing": {
                "sequence_scaler": "MinMaxScaler fit on training feature rows per fold",
                "walk_forward_validation": "LSTM early stopping; tabular fold models fit training rows only",
                "final_holdout_refit": "Random Forest uses all pre-holdout development rows with fixed parameters",
                "tabular_scaler": "none",
            },
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
            "Historical point-in-time news sentiment is unavailable; neutral sentiment is used in every hybrid row.",
            "The LSTM scaler is fit on training feature rows only; validation is used only for early stopping.",
            "The final chronological holdout is evaluated once and is not used to choose models, thresholds or weights.",
            "Trading metrics are simplified long-or-cash close-to-close simulations with proportional costs.",
        ],
    }
    if report_path:
        _save_report(report_path, report)
    return report

"""Train-window Gaussian HMMs and causal market-regime inference."""

import hashlib
import importlib.metadata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _as_sequence(values, torch):
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32).unsqueeze(0)


def _viterbi_states(model, values, torch):
    return model.predict(_as_sequence(values, torch))[0].detach().cpu().numpy().astype(int)


REGIME_MODEL_VERSION = "gaussian-hmm-market-regimes-v1"
REGIME_FEATURES = ["daily_return", "volatility_20d", "normalized_range", "volume_change"]
REGIME_LABELS = ("bearish/high-risk", "bearish/low-volatility", "sideways/low-trend", "bullish/trending")
MIN_TRAINING_ROWS = 100


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_regime_features(prices):
    """Build causal daily features; each row uses data through that session close."""
    required = {"date", "high", "low", "close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")
    frame = prices[["date", "high", "low", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["daily_return"] = frame["close"].pct_change()
    frame["volatility_20d"] = frame["daily_return"].rolling(20, min_periods=10).std()
    frame["normalized_range"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["volume_change"] = frame["volume"].pct_change()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=REGIME_FEATURES).reset_index(drop=True)


def _unavailable(status, reason, *, features=None):
    return {
        "status": status,
        "regime": None,
        "state_id": None,
        "state_probabilities": {},
        "model_version": REGIME_MODEL_VERSION,
        "features": list(features or REGIME_FEATURES),
        "fallback_reason": reason,
    }


def _state_mapping(model, scaler, training_rows):
    scaled = scaler.transform(training_rows[REGIME_FEATURES].to_numpy(dtype=float))
    import torch
    state_ids = _viterbi_states(model, scaled, torch)
    statistics = {}
    for state_id in range(len(model.distributions)):
        subset = training_rows.iloc[state_ids == state_id]
        statistics[state_id] = {
            "mean_daily_return": float(subset["daily_return"].mean()) if len(subset) else 0.0,
            "mean_volatility_20d": float(subset["volatility_20d"].mean()) if len(subset) else 0.0,
            "training_rows": int(len(subset)),
        }
    ranked_by_return = sorted(
        statistics,
        key=lambda state: (statistics[state]["mean_daily_return"], -state),
    )
    labels = {ranked_by_return[-1]: REGIME_LABELS[3]}
    low_return_state = ranked_by_return[0]
    median_volatility = float(np.median([
        item["mean_volatility_20d"] for item in statistics.values()
    ]))
    lower_state_volatility = statistics[low_return_state]["mean_volatility_20d"]
    labels[low_return_state] = (
        REGIME_LABELS[0] if lower_state_volatility >= median_volatility else REGIME_LABELS[1]
    )
    remaining = [state for state in ranked_by_return[1:-1]]
    for rank, state in enumerate(remaining, start=1):
        labels[state] = REGIME_LABELS[2] if rank == 1 else f"sideways/low-trend ({rank + 1})"
    return labels, statistics


def fit_regime_model(training_rows, validation_rows, *, random_seed=42, candidate_states=(2, 3)):
    """Select a compact HMM using validation likelihood; fit transforms/model on train only."""
    if not isinstance(training_rows, pd.DataFrame) or not isinstance(validation_rows, pd.DataFrame):
        return _unavailable("invalid_input", "training and validation rows must be dataframes")
    needed = {"date", *REGIME_FEATURES}
    if not needed.issubset(training_rows.columns) or not needed.issubset(validation_rows.columns):
        return _unavailable("invalid_input", "regime feature columns are missing")
    train = training_rows.dropna(subset=REGIME_FEATURES).reset_index(drop=True)
    validation = validation_rows.dropna(subset=REGIME_FEATURES).reset_index(drop=True)
    if len(train) < MIN_TRAINING_ROWS or len(validation) < 10:
        return _unavailable(
            "insufficient_history",
            f"need at least {MIN_TRAINING_ROWS} training and 10 validation rows; got {len(train)} and {len(validation)}",
        )
    try:
        import torch
        from pomegranate.distributions import Normal
        from pomegranate.hmm import DenseHMM
    except (ImportError, OSError) as exc:
        return _unavailable("unavailable_dependency", f"pomegranate unavailable: {type(exc).__name__}")

    scaler = StandardScaler().fit(train[REGIME_FEATURES].to_numpy(dtype=float))
    train_scaled = scaler.transform(train[REGIME_FEATURES].to_numpy(dtype=float))
    validation_scaled = scaler.transform(validation[REGIME_FEATURES].to_numpy(dtype=float))
    candidates = []
    failures = []
    for state_count in candidate_states:
        state_count = int(state_count)
        if state_count < 2 or len(train) < state_count * 30:
            failures.append({"states": state_count, "status": "insufficient_training_rows"})
            continue
        try:
            distributions = [Normal(covariance_type="diag", min_cov=1e-4) for _ in range(state_count)]
            model = DenseHMM(
                distributions=distributions,
                max_iter=1,
                tol=1e-3,
                random_state=int(random_seed),
                verbose=False,
            )
            train_sequence = _as_sequence(train_scaled, torch)
            previous_log_likelihood = None
            train_converged = False
            iterations = 0
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_seed))
                for iterations in range(1, 251):
                    model.fit(train_sequence)
                    current_log_likelihood = float(model.log_probability(train_sequence).sum().item())
                    if not np.isfinite(current_log_likelihood):
                        break
                    if previous_log_likelihood is not None:
                        improvement = current_log_likelihood - previous_log_likelihood
                        tolerance = max(1e-3, abs(previous_log_likelihood) * 1e-5)
                        if improvement < -tolerance:
                            break
                        if improvement <= tolerance:
                            train_converged = True
                            break
                    previous_log_likelihood = current_log_likelihood
            if not train_converged:
                failures.append({"states": state_count, "status": "non_converged"})
                continue
            validation_sequence = _as_sequence(validation_scaled, torch)
            validation_log_likelihood = float(model.log_probability(validation_sequence).sum().item()) / len(validation_scaled)
            if not np.isfinite(validation_log_likelihood):
                raise ValueError("non-finite validation log likelihood")
            parameter_count = (
                state_count * (state_count - 1) + state_count - 1
                + 2 * state_count * len(REGIME_FEATURES)
            )
            penalty = 0.5 * parameter_count * np.log(len(train)) / len(train)
            candidates.append({
                "model": model,
                "states": state_count,
                "validation_log_likelihood_per_row": validation_log_likelihood,
                "selection_score": validation_log_likelihood - float(penalty),
                "train_converged": train_converged,
                "iterations": int(iterations),
            })
        except Exception as exc:
            failures.append({"states": state_count, "status": f"fit_failed_{type(exc).__name__}"})
    if not candidates:
        return _unavailable("fit_failed", "no candidate HMM completed successfully") | {"candidate_failures": failures}
    selected = max(candidates, key=lambda item: (item["selection_score"], -item["states"]))
    model = selected["model"]
    labels, training_statistics = _state_mapping(model, scaler, train)
    return {
        "status": "ok",
        "model": model,
        "scaler": scaler,
        "state_labels": labels,
        "training_statistics": training_statistics,
        "selected_states": int(selected["states"]),
        "selection_method": "validation_mean_log_likelihood_with_parameter_penalty",
        "candidate_scores": [
            {key: item[key] for key in (
                "states", "validation_log_likelihood_per_row", "selection_score", "train_converged", "iterations"
            )}
            for item in candidates
        ],
        "candidate_failures": failures,
        "training_rows": int(len(train)),
        "training_start": train.iloc[0]["date"].strftime("%Y-%m-%d"),
        "training_end": train.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "validation_start": validation.iloc[0]["date"].strftime("%Y-%m-%d"),
        "validation_end": validation.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "features": list(REGIME_FEATURES),
        "model_version": REGIME_MODEL_VERSION,
        "scaler_fit_rows": int(len(train)),
    }


def infer_regime_sequence(fitted, observed_rows):
    """Return one causal filtered posterior per row; no row sees a later observation."""
    if fitted.get("status") != "ok":
        return []
    observed = observed_rows.dropna(subset=REGIME_FEATURES).reset_index(drop=True)
    if observed.empty:
        return []
    values = fitted["scaler"].transform(observed[REGIME_FEATURES].to_numpy(dtype=float))
    import torch
    output = []
    for end in range(1, len(values) + 1):
        # predict_proba is smoothed over its input. Restrict each call to the
        # prefix ending at this row so there are no observations after cutoff.
        if end == 1:
            sequence = _as_sequence(values[:1], torch)[0, 0].reshape(1, -1)
            emission_log_probabilities = torch.stack([
                distribution.log_probability(sequence).reshape(-1)[0]
                for distribution in fitted["model"].distributions
            ])
            posterior = torch.softmax(fitted["model"].starts + emission_log_probabilities, dim=0).detach().cpu().numpy()
        else:
            posterior = fitted["model"].predict_proba(_as_sequence(values[:end], torch))[0, -1].detach().cpu().numpy()
        state_id = int(np.argmax(posterior))
        label = fitted["state_labels"].get(state_id, "unknown")
        output.append({
            "date": observed.iloc[end - 1]["date"].strftime("%Y-%m-%d"),
            "state_id": state_id,
            "regime": label,
            "state_probabilities": {
                fitted["state_labels"].get(index, f"state_{index}"): float(probability)
                for index, probability in enumerate(posterior)
            },
        })
    return output


def estimate_current_regime(prices, *, random_seed=42):
    """Train on an initial history, select state count on later validation, infer latest cutoff."""
    try:
        features = build_regime_features(prices)
    except (KeyError, TypeError, ValueError) as exc:
        return _unavailable("invalid_data", str(exc))
    n = len(features)
    if n < MIN_TRAINING_ROWS + 30:
        return _unavailable("insufficient_history", f"only {n} usable regime feature rows") | {
            "available_rows": n,
        }
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    fitted = fit_regime_model(
        features.iloc[:train_end], features.iloc[train_end:validation_end], random_seed=random_seed
    )
    if fitted.get("status") != "ok":
        return {**fitted, "available_rows": n}
    try:
        history = infer_regime_sequence(fitted, features.iloc[:])
    except Exception as exc:
        return _unavailable("inference_failed", f"causal posterior failed: {type(exc).__name__}")
    if not history:
        return _unavailable("inference_failed", "no causal posterior was produced")
    latest = history[-1]
    price_columns = ["date", "high", "low", "close", "volume"]
    source_hash = hashlib.sha256(prices[price_columns].to_csv(index=False).encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "regime": latest["regime"],
        "state_id": latest["state_id"],
        "state_probabilities": latest["state_probabilities"],
        "inference_cutoff": latest["date"],
        "state_history": history[-60:],
        "model_version": REGIME_MODEL_VERSION,
        "method": "GaussianHMM_filtered_posterior",
        "features": list(REGIME_FEATURES),
        "training_window": {
            "start": fitted["training_start"], "end": fitted["training_end"],
            "rows": fitted["training_rows"], "scaler_fit_rows": fitted["scaler_fit_rows"],
        },
        "validation_window": {
            "start": fitted["validation_start"], "end": fitted["validation_end"],
            "state_selection": fitted["selection_method"],
            "candidate_scores": fitted["candidate_scores"],
        },
        "selected_states": fitted["selected_states"],
        "state_mapping": {
            str(state): {"regime": fitted["state_labels"][state], **fitted["training_statistics"][state]}
            for state in fitted["state_labels"]
        },
        "data": {
            "source": "Yahoo Finance adjusted daily OHLCV",
            "rows": int(len(prices)),
            "features_rows": int(n),
            "sha256": source_hash,
            "first_date": features.iloc[0]["date"].strftime("%Y-%m-%d"),
            "last_date": latest["date"],
        },
        "model_metadata": {
            "family": "Gaussian HMM",
            "library": "pomegranate",
            "library_version": _package_version("pomegranate"),
            "torch_version": _package_version("torch"),
            "covariance_type": "diag",
            "random_seed": int(random_seed),
            "n_iter": 250,
            "validation_candidate_scores": fitted["candidate_scores"],
            "candidate_failures": fitted["candidate_failures"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Regime names are descriptive and mapped from training-window return statistics; they are not ground truth.",
            "Filtered state probabilities are model-based state probabilities, not calibrated probabilities of returns or profit.",
            "The model is refit per request and may select different state IDs/mappings across training windows.",
            "Daily provider data is not tick-level or guaranteed exchange real-time data.",
        ],
    }


def evaluate_regime_walk_forward(symbol, prices, config, indicator_builder):
    """Compare pooled OOS hybrid metrics with descriptive metrics by causal regime."""
    try:
        from . import evaluation
    except ImportError:
        import evaluation
    if indicator_builder is None:
        raise ValueError("indicator_builder is required")
    frame = evaluation._add_next_session_target(indicator_builder(prices))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=evaluation.FEATURE_COLUMNS + evaluation.SEQUENCE_COLUMNS + ["target", "forward_return"]
    ).reset_index(drop=True)
    regime_frame = build_regime_features(prices)
    # `indicators` already provides the evaluation feature `volume_change`;
    # avoid a duplicate merge suffix for the identically computed HMM input.
    regime_columns = ["date", "daily_return", "volatility_20d", "normalized_range"]
    frame = frame.merge(regime_frame[regime_columns], on="date", how="inner").reset_index(drop=True)
    plan = evaluation.generate_walk_forward_folds(
        len(frame),
        final_holdout_fraction=float(config.get("final_holdout_fraction", 0.20)),
        minimum_training_fraction=float(config.get("minimum_training_fraction", 0.50)),
        validation_fraction=float(config.get("validation_fraction", 0.10)),
        test_fraction=float(config.get("test_fraction", 0.10)),
    )
    if len(frame) < evaluation.MINIMUM_SAMPLES or plan["status"] != "ok":
        return _unavailable("insufficient_walk_forward_history", "not enough rows/folds for a walk-forward regime comparison")
    fold_reports = []
    all_truth, all_probabilities, all_predictions = [], [], []
    by_regime = {}
    for boundaries in plan["folds"]:
        test = evaluation._evaluate_window(
            frame, fit_end=boundaries["train_end"], train_end=boundaries["train_end"],
            validation_end=boundaries["validation_end"], test_start=boundaries["test_start"],
            test_end=boundaries["test_end"], config=config,
        )
        if test.get("status") != "ok":
            fold_reports.append({"fold": boundaries["fold"], "status": test.get("status")})
            continue
        train = frame.iloc[boundaries["train_start"]:boundaries["train_end"]]
        validation = frame.iloc[boundaries["validation_start"]:boundaries["validation_end"]]
        fitted = fit_regime_model(train, validation, random_seed=int(config.get("random_seed", 42)))
        if fitted.get("status") != "ok":
            fold_reports.append({"fold": boundaries["fold"], "status": fitted["status"], "fallback_reason": fitted.get("fallback_reason")})
            continue
        context = frame.iloc[boundaries["train_start"]:boundaries["test_end"]]
        inferred = infer_regime_sequence(fitted, context)
        test_regimes = inferred[-(boundaries["test_end"] - boundaries["test_start"]):]
        labels = [item["regime"] for item in test_regimes]
        test_rows = frame.iloc[boundaries["test_start"]:boundaries["test_end"]]
        hybrid = test["models"]["hybrid"]
        truth = test_rows["target"].astype(int).to_numpy()
        probabilities = np.asarray(hybrid["probabilities"], dtype=float)
        predictions = np.asarray(hybrid["predictions"], dtype=int)
        all_truth.extend(truth.tolist())
        all_probabilities.extend(probabilities.tolist())
        all_predictions.extend(predictions.tolist())
        for regime in sorted(set(labels)):
            mask = np.asarray(labels) == regime
            group = by_regime.setdefault(regime, {"truth": [], "probabilities": [], "predictions": [], "folds": set()})
            group["truth"].extend(truth[mask].tolist())
            group["probabilities"].extend(probabilities[mask].tolist())
            group["predictions"].extend(predictions[mask].tolist())
            group["folds"].add(int(boundaries["fold"]))
        fold_reports.append({
            "fold": int(boundaries["fold"]), "status": "ok",
            "train_start": train.iloc[0]["date"].strftime("%Y-%m-%d"),
            "train_end": train.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "validation_start": validation.iloc[0]["date"].strftime("%Y-%m-%d"),
            "validation_end": validation.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "test_start": test_rows.iloc[0]["date"].strftime("%Y-%m-%d"),
            "test_end": test_rows.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "selected_states": fitted["selected_states"],
            "state_mapping": {str(key): value for key, value in fitted["state_labels"].items()},
            "state_selection_scores": fitted["candidate_scores"],
            "test_regime_counts": {label: labels.count(label) for label in sorted(set(labels))},
            "without_regime_hybrid_metrics": hybrid["classification"],
        })
    if not all_truth:
        return _unavailable("fit_failed", "no walk-forward fold produced regime predictions")
    successful_folds = sum(fold.get("status") == "ok" for fold in fold_reports)
    aggregate = evaluation._metrics(all_truth, all_predictions, all_probabilities)
    regime_metrics = {}
    for label, values in by_regime.items():
        regime_metrics[label] = {
            "samples": len(values["truth"]), "folds": sorted(values["folds"]),
            "classification": evaluation._metrics(values["truth"], values["predictions"], values["probabilities"]),
        }
    return {
        "status": "ok" if successful_folds == len(plan["folds"]) else "partial",
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "symbol": symbol,
        "comparison_method": "same fixed-hybrid predictions scored pooled and stratified by causal HMM regime",
        "without_regime": {"samples": len(all_truth), "classification": aggregate},
        "with_regime_context": regime_metrics,
        "folds": fold_reports,
        "successful_folds": successful_folds,
        "split": {
            "walk_forward_folds": len(plan["folds"]),
            "final_holdout_excluded": True,
            "final_holdout_start": frame.iloc[plan["final_holdout_start"]]["date"].strftime("%Y-%m-%d"),
        },
        "regime_model_version": REGIME_MODEL_VERSION,
        "versions": {
            "regime_model": REGIME_MODEL_VERSION,
            "pomegranate": _package_version("pomegranate"),
            "torch": _package_version("torch"),
        },
        "features": list(REGIME_FEATURES),
        "target": "next session adjusted close > current session adjusted close",
        "data": {
            "source": config.get("data_source", "Yahoo Finance adjusted daily OHLCV"),
            "rows": int(len(prices)),
            "sha256": hashlib.sha256(
                prices[["date", "open", "high", "low", "close", "volume"]].to_csv(index=False).encode("utf-8")
            ).hexdigest(),
            "first_date": pd.to_datetime(prices["date"], utc=True).min().strftime("%Y-%m-%d"),
            "last_date": pd.to_datetime(prices["date"], utc=True).max().strftime("%Y-%m-%d"),
        },
        "configuration": {
            "random_seed": int(config.get("random_seed", 42)),
            "random_forest_trees": int(config["rf_trees"]),
            "lstm_lookback": int(config["lookback"]),
            "lstm_units": int(config["lstm_units"]),
            "lstm_epochs": int(config["epochs"]),
            "hmm_candidates": [2, 3],
            "hmm_covariance_type": "diag",
            "validation_selection": "validation mean log likelihood with parameter penalty",
            "preprocessing": "StandardScaler fitted per fold on HMM training rows only",
        },
        "limitations": [
            "By-regime metrics are descriptive strata of the same fixed-hybrid predictions, not a causal estimate of adding regime features.",
            "Regime state IDs and training-statistic label mappings are refit per fold and must not be compared as identical IDs across folds.",
            "The final chronological holdout is excluded; only development walk-forward folds are summarized.",
        ],
    }

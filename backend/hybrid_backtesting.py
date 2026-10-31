"""Walk-forward out-of-sample backtest of production hybrid signals."""

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from .adaptive_ensemble import ADAPTIVE_COMPONENTS, ADAPTIVE_ENSEMBLE_VERSION, select_adaptive_weights
    from .calibration import (CALIBRATION_VERSION, NO_TRADE_VERSION, apply_calibrator,
                              calibration_metrics, select_no_trade_thresholds,
                              select_validation_calibrator)
    from . import evaluation
    from .backtesting import ENGINE_VERSION, run_backtest
    from .hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, TARGET_SPECIFICATION, combine_probabilities, signal_from_probability
    from .regime_detection import REGIME_MODEL_VERSION, build_regime_features, fit_regime_model, infer_regime_sequence
except ImportError:
    from adaptive_ensemble import ADAPTIVE_COMPONENTS, ADAPTIVE_ENSEMBLE_VERSION, select_adaptive_weights
    from calibration import (CALIBRATION_VERSION, NO_TRADE_VERSION, apply_calibrator,
                             calibration_metrics, select_no_trade_thresholds,
                             select_validation_calibrator)
    import evaluation
    from backtesting import ENGINE_VERSION, run_backtest
    from hybrid import HYBRID_VERSION, HYBRID_WEIGHTS, TARGET_SPECIFICATION, combine_probabilities, signal_from_probability
    from regime_detection import REGIME_MODEL_VERSION, build_regime_features, fit_regime_model, infer_regime_sequence


BACKTEST_VERSION = "walk-forward-hybrid-next-open-v1"


def _adaptive_settings(config):
    return {
        "minimum_weight": float(config.get("adaptive_minimum_weight", 0.05)),
        "maximum_weight": float(config.get("adaptive_maximum_weight", 0.80)),
        "temperature": float(config.get("adaptive_temperature", 8.0)),
        "minimum_validation_samples": int(config.get("adaptive_minimum_validation_samples", 20)),
        "minimum_regime_samples": int(config.get("adaptive_minimum_regime_samples", 20)),
    }


def _adaptive_regime(regime, config):
    return regime if config.get("adaptive_use_regime", True) else None


def _signal_action(signal):
    return "BUY" if signal in {"buy", "strong_buy"} else "SELL" if signal in {"sell", "strong_sell"} else "HOLD"


def run_hybrid_backtest(symbol, raw, config, indicator_builder, *, selected_mode="fixed"):
    """Compare fixed/adaptive hybrid and component signals on matching OOS dates.

    The final holdout is kept untouched and excluded. Signals from the last
    development test date are audited but cannot be executed in-window.
    Adaptive weights for a test fold use only OOS predictions and labels from
    that fold's earlier validation window. Regime inference is causal by date.
    """
    if indicator_builder is None:
        raise ValueError("indicator_builder is required")
    if selected_mode not in {"fixed", "adaptive"}:
        raise ValueError("selected_mode must be fixed or adaptive")
    frame = evaluation._add_next_session_target(indicator_builder(raw))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=evaluation.FEATURE_COLUMNS + evaluation.SEQUENCE_COLUMNS + ["target", "forward_return"]
    ).reset_index(drop=True)
    regime_frame = build_regime_features(raw)
    regime_extra = ["date", "daily_return", "volatility_20d", "normalized_range"]
    frame = frame.merge(regime_frame[regime_extra], on="date", how="inner").reset_index(drop=True)
    plan = evaluation.generate_walk_forward_folds(
        len(frame),
        final_holdout_fraction=float(config.get("final_holdout_fraction", 0.20)),
        minimum_training_fraction=float(config.get("minimum_training_fraction", 0.50)),
        validation_fraction=float(config.get("validation_fraction", 0.10)),
        test_fraction=float(config.get("test_fraction", 0.10)),
    )
    if len(frame) < evaluation.MINIMUM_SAMPLES or plan["status"] != "ok":
        return {
            "status": "insufficient_data" if len(frame) < evaluation.MINIMUM_SAMPLES else "insufficient_walk_forward_history",
            "symbol": symbol,
            "required_samples": evaluation.MINIMUM_SAMPLES,
            "available_samples": len(frame),
            "walk_forward_status": plan["status"],
            "available_folds": len(plan["folds"]),
        }

    audit = []
    fold_metadata = []
    fold_calibration_reports = []
    for boundaries in plan["folds"]:
        result = evaluation._evaluate_window(
            frame, fit_end=boundaries["train_end"], train_end=boundaries["train_end"],
            validation_end=boundaries["validation_end"], test_start=boundaries["test_start"],
            test_end=boundaries["test_end"], config=config,
        )
        if result.get("status") != "ok":
            raise ValueError(f"fold {boundaries['fold']} failed: {result.get('status')}")

        validation_start = boundaries["validation_start"]
        validation_end = boundaries["validation_end"]
        lookback = int(config.get("lookback", 30))
        validation_train_end = min(validation_start - 1, max(lookback + 20, int(validation_start * 0.80)))
        validation_result = {"status": "insufficient_validation_history", "models": {}, "component_status": {}}
        if validation_train_end > lookback + 20 and validation_end > validation_start:
            validation_result = evaluation._evaluate_window(
                frame,
                fit_end=validation_start,
                train_end=validation_train_end,
                validation_end=validation_start,
                test_start=validation_start,
                test_end=validation_end,
                config=config,
            )

        regime_fit = fit_regime_model(
            frame.iloc[boundaries["train_start"]:boundaries["train_end"]],
            frame.iloc[validation_start:validation_end],
            random_seed=int(config.get("random_seed", 42)),
        )
        regime_path = []
        regime_fallback = None
        if regime_fit.get("status") == "ok":
            try:
                regime_path = infer_regime_sequence(regime_fit, frame.iloc[:boundaries["test_end"]])
            except Exception as exc:
                regime_fallback = f"regime_inference_failed_{type(exc).__name__}"
        else:
            regime_fallback = regime_fit.get("fallback_reason") or regime_fit.get("status")

        validation_rows = []
        if validation_result.get("status") == "ok":
            val_models = validation_result["models"]
            val_lstm_available = validation_result.get("component_status", {}).get("lstm") == "trained"
            val_lstm = val_models.get("lstm", {}).get("probabilities", np.full(validation_end - validation_start, 0.5))
            val_forest = val_models.get("random_forest", {}).get("probabilities", np.full(validation_end - validation_start, 0.5))
            val_technical = evaluation._technical_probabilities(frame.iloc[validation_start:validation_end])
            val_regimes = regime_path[validation_start:validation_end] if regime_path else []
            for offset, index in enumerate(range(validation_start, validation_end)):
                validation_rows.append({
                    "date": frame.iloc[index]["date"].strftime("%Y-%m-%d"),
                    "target": int(frame.iloc[index]["target"]),
                    "regime": val_regimes[offset]["regime"] if len(val_regimes) == validation_end - validation_start else None,
                    "components": {
                        "lstm": float(val_lstm[offset]),
                        "random_forest": float(val_forest[offset]),
                        "technical": float(val_technical[offset]),
                    },
                    "availability": {
                        "lstm": val_lstm_available,
                        "random_forest": "random_forest" in val_models,
                        "technical": True,
                        "finbert": False,
                    },
                })

        fixed_validation_probabilities = [
            combine_probabilities(
                row["components"]["lstm"], row["components"]["random_forest"], 0.0,
                row["components"]["technical"],
            )["probability_up"] for row in validation_rows
        ]
        adaptive_validation_probabilities = []
        validation_dispersions = []
        for validation_index, validation_row in enumerate(validation_rows):
            available = [name for name in ADAPTIVE_COMPONENTS
                         if validation_row["availability"].get(name)]
            selected = select_adaptive_weights(
                validation_rows[:validation_index], current_regime=_adaptive_regime(validation_row.get("regime"), config),
                available_components=available, **_adaptive_settings(config),
            )
            weights = selected.get("weights")
            adaptive_validation_probabilities.append(
                sum(weights[name] * validation_row["components"][name] for name in weights)
                if weights else combine_probabilities(
                    validation_row["components"]["lstm"],
                    validation_row["components"]["random_forest"], 0.0,
                    validation_row["components"]["technical"],
                )["probability_up"]
            )
            values = [validation_row["components"][name] for name in available]
            validation_dispersions.append(float(np.std(values)) if values else 0.0)

        validation_labels = [row["target"] for row in validation_rows]

        def validation_policy(probabilities):
            calibrator = select_validation_calibrator(probabilities, validation_labels)
            selection_probabilities = calibrator.get("selection_probabilities", [])
            selection_count = len(selection_probabilities)
            selection_dispersions = validation_dispersions[-selection_count:] if selection_count else []
            thresholds = select_no_trade_thresholds(
                selection_probabilities, selection_dispersions,
                calibrator.get("selection_labels", []),
            ) if selection_count else {
                "status": "unavailable", "neutral_margin": 0.0,
                "maximum_dispersion": 1.0, "samples": 0,
                "fallback_reason": calibrator.get("fallback_reason") or "calibration_unavailable",
            }
            return calibrator, thresholds

        fixed_calibrator, fixed_thresholds = validation_policy(fixed_validation_probabilities)
        adaptive_calibrator, adaptive_thresholds = validation_policy(adaptive_validation_probabilities)
        selected_calibrator, selected_thresholds = (
            (adaptive_calibrator, adaptive_thresholds) if selected_mode == "adaptive"
            else (fixed_calibrator, fixed_thresholds)
        )
        fold_calibration_audit = []

        indices = range(boundaries["test_start"], boundaries["test_end"])
        models = result["models"]
        for offset, index in enumerate(indices):
            row = frame.iloc[index]
            lstm = float(models["lstm"]["probabilities"][offset])
            forest = float(models["random_forest"]["probabilities"][offset])
            technical = float(evaluation._technical_probabilities(frame.iloc[[index]])[0])
            hybrid = combine_probabilities(lstm, forest, 0.0, technical)
            regime = regime_path[index]["regime"] if len(regime_path) > index else None
            lstm_available = result.get("component_status", {}).get("lstm") == "trained"
            available_components = ["random_forest", "technical"] + (["lstm"] if lstm_available else [])
            weights_result = select_adaptive_weights(
                validation_rows,
                current_regime=_adaptive_regime(regime, config),
                available_components=available_components,
                **_adaptive_settings(config),
            )
            fallback_reasons = [reason for reason in (
                weights_result.get("fallback_reason"),
                f"regime_context_unavailable:{regime_fallback}" if regime_fallback else None,
                "lstm_unavailable_at_test" if not lstm_available else None,
            ) if reason]
            if weights_result.get("weights"):
                adaptive_weights = weights_result["weights"]
                adaptive_probability = sum(
                    adaptive_weights[name] * {"lstm": lstm, "random_forest": forest, "technical": technical}[name]
                    for name in adaptive_weights
                )
            else:
                adaptive_weights = dict(HYBRID_WEIGHTS)
                adaptive_probability = hybrid["probability_up"]
                fallback_reasons.append("fixed_hybrid_fallback_no_adaptive_validation_evidence")
            adaptive_signal = signal_from_probability(adaptive_probability)
            fixed_calibrated = float(apply_calibrator(fixed_calibrator, [hybrid["probability_up"]])[0])
            adaptive_calibrated = float(apply_calibrator(adaptive_calibrator, [adaptive_probability])[0])
            dispersion_components = [lstm, forest, technical] if lstm_available else [forest, technical]
            component_dispersion = float(np.std(dispersion_components))
            selected_calibrated = adaptive_calibrated if selected_mode == "adaptive" else fixed_calibrated
            neutral_margin = float(selected_thresholds.get("neutral_margin", 0.0))
            maximum_dispersion = float(selected_thresholds.get("maximum_dispersion", 1.0))
            no_trade = (
                selected_calibrator.get("status") == "ok"
                and selected_thresholds.get("status") == "ok"
                and (abs(selected_calibrated - 0.5) < neutral_margin
                     or component_dispersion > maximum_dispersion)
            )
            calibrated_signal = signal_from_probability(selected_calibrated)
            if no_trade and abs(selected_calibrated - 0.5) < neutral_margin:
                no_trade_reason = "calibrated_probability_near_neutral"
            elif no_trade:
                no_trade_reason = "component_disagreement_above_validation_limit"
            else:
                no_trade_reason = None
            fold_calibration_audit.append({
                "target": int(row["target"]),
                "raw_fixed": float(hybrid["probability_up"]),
                "calibrated_fixed": fixed_calibrated,
                "raw_adaptive": float(adaptive_probability),
                "calibrated_adaptive": adaptive_calibrated,
            })
            audit.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "fold": int(boundaries["fold"]),
                "target": int(row["target"]),
                "lstm_probability_up": lstm,
                "random_forest_probability_up": forest,
                "technical_probability_up": technical,
                "sentiment_score": 0.0,
                "hybrid_probability_up": hybrid["probability_up"],
                "raw_ensemble_probability_up": float(hybrid["probability_up"]),
                "calibrated_probability_up": selected_calibrated,
                "calibrated_fixed_probability_up": fixed_calibrated,
                "calibration_status": selected_calibrator.get("status"),
                "calibration_method": selected_calibrator.get("selected_method"),
                "calibration_fallback_reason": selected_calibrator.get("fallback_reason"),
                "uncertainty": {
                    "type": "component_probability_standard_deviation",
                    "value": component_dispersion,
                    "interpretation": "Component disagreement dispersion; not a confidence interval or probability of profit.",
                },
                "calibrated_signal": calibrated_signal,
                "trading_decision": "no_trade" if no_trade else calibrated_signal,
                "no_trade_reason": no_trade_reason,
                "hybrid_signal": hybrid["signal"],
                "lstm_signal": signal_from_probability(lstm),
                "random_forest_signal": signal_from_probability(forest),
                "regime": regime,
                "adaptive_probability_up": float(adaptive_probability),
                "raw_adaptive_probability_up": float(adaptive_probability),
                "calibrated_adaptive_probability_up": adaptive_calibrated,
                "adaptive_signal": adaptive_signal,
                "adaptive_weights": adaptive_weights,
                "adaptive_method": ADAPTIVE_ENSEMBLE_VERSION if weights_result.get("weights") else HYBRID_VERSION,
                "adaptive_selection": {
                    "status": weights_result["status"],
                    "scope": weights_result.get("selected_scope"),
                    "validation_metrics": weights_result.get("validation_metrics", {}),
                    "component_availability": weights_result.get("component_availability", {}),
                    "fallback_reason": ";".join(fallback_reasons) or None,
                },
                "validation_period": {
                    "start": validation_rows[0]["date"] if validation_rows else None,
                    "end": validation_rows[-1]["date"] if validation_rows else None,
                    "samples": len(validation_rows),
                },
                "component_availability": {
                    "lstm": lstm_available, "random_forest": True,
                    "technical": True, "finbert": False,
                },
                "executable": index < plan["development_end"] - 1,
            })
        fold_calibration_audit = {
            **{"fold": int(boundaries["fold"])},
            "selection": {
                "method": selected_calibrator.get("selected_method"),
                "candidate_statuses": selected_calibrator.get("candidate_statuses", {}),
                "candidate_brier": selected_calibrator.get("selection_brier", {}),
                "selection_samples": selected_calibrator.get("selection_samples", 0),
                "fit_samples": selected_calibrator.get("samples", 0),
                "status": selected_calibrator.get("status"),
                "fallback_reason": selected_calibrator.get("fallback_reason"),
                "no_trade_thresholds": selected_thresholds,
            },
            "test": {
                "raw_fixed": calibration_metrics([r["target"] for r in fold_calibration_audit], [r["raw_fixed"] for r in fold_calibration_audit]),
                "calibrated_fixed": calibration_metrics([r["target"] for r in fold_calibration_audit], [r["calibrated_fixed"] for r in fold_calibration_audit]),
                "raw_adaptive": calibration_metrics([r["target"] for r in fold_calibration_audit], [r["raw_adaptive"] for r in fold_calibration_audit]),
                "calibrated_adaptive": calibration_metrics([r["target"] for r in fold_calibration_audit], [r["calibrated_adaptive"] for r in fold_calibration_audit]),
                "samples": len(fold_calibration_audit),
                "no_trade_count": sum(item.get("trading_decision") == "no_trade" for item in audit if item["fold"] == boundaries["fold"]),
            },
        }
        fold_calibration_reports.append(fold_calibration_audit)
        fold_metadata.append({
            "fold": int(boundaries["fold"]),
            "train_start": frame.iloc[boundaries["train_start"]]["date"].strftime("%Y-%m-%d"),
            "train_end": frame.iloc[boundaries["train_end"] - 1]["date"].strftime("%Y-%m-%d"),
            "validation_start": frame.iloc[boundaries["validation_start"]]["date"].strftime("%Y-%m-%d"),
            "validation_end": frame.iloc[boundaries["validation_end"] - 1]["date"].strftime("%Y-%m-%d"),
            "test_start": result["test_start"], "test_end": result["test_end"],
            "lstm_status": result["component_status"].get("lstm"),
            "validation_prediction_status": validation_result.get("status"),
            "validation_lstm_status": validation_result.get("component_status", {}).get("lstm"),
            "regime_model_status": regime_fit.get("status"),
            "regime_selected_states": regime_fit.get("selected_states"),
            "regime_state_mapping": {str(key): value for key, value in regime_fit.get("state_labels", {}).items()},
            "regime_fallback_reason": regime_fallback,
        })

    oos_start = plan["folds"][0]["test_start"]
    oos_end = plan["development_end"]
    prices = raw.copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices = prices.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    date_by_text = {date.strftime("%Y-%m-%d"): date for date in prices["date"]}
    first_date = frame.iloc[oos_start]["date"].strftime("%Y-%m-%d")
    last_date = frame.iloc[oos_end - 1]["date"].strftime("%Y-%m-%d")
    prices = prices[(prices["date"] >= date_by_text[first_date]) & (prices["date"] <= date_by_text[last_date])].reset_index(drop=True)
    executable_dates = {item["date"] for item in audit if item["executable"]}
    strategy_signals = {
        "fixed_hybrid": "hybrid_signal", "lstm_only": "lstm_signal",
        "random_forest_only": "random_forest_signal", "adaptive_hybrid": "adaptive_signal",
        "calibrated_no_trade": "trading_decision",
    }
    reports = {}
    for name, field in strategy_signals.items():
        signals = pd.DataFrame([
            {"date": item["date"], "signal": _signal_action(item[field])}
            for item in audit if item["date"] in executable_dates
        ])
        reports[name] = run_backtest(
            prices, signals, symbol=symbol, config=config,
            strategy_name=name, strategy_version=BACKTEST_VERSION,
            strategy_parameters={"hybrid_version": HYBRID_VERSION,
                                 "adaptive_method": ADAPTIVE_ENSEMBLE_VERSION if name == "adaptive_hybrid" else None,
                                 "calibration_method": CALIBRATION_VERSION if name == "calibrated_no_trade" else None,
                                 "no_trade_method": NO_TRADE_VERSION if name == "calibrated_no_trade" else None,
                                 "selected_mode": selected_mode if name == "calibrated_no_trade" else None,
                                 "weights": HYBRID_WEIGHTS if name == "fixed_hybrid" else None},
        )

    # Fold-reset results make the fixed/adaptive comparison auditable per
    # validation regime. Each fold starts with fresh cash and omits its final
    # signal because no next-session open is inside that fold.
    fold_strategy_comparison = []
    for fold in fold_metadata:
        fold_audit = [item for item in audit if item["fold"] == fold["fold"]]
        fold_start = pd.Timestamp(fold["test_start"])
        fold_end = pd.Timestamp(fold["test_end"])
        fold_prices = prices[(prices["date"] >= fold_start) & (prices["date"] <= fold_end)].reset_index(drop=True)
        fold_executable = fold_audit[:-1]
        entry = {"fold": fold["fold"], "start": fold["test_start"], "end": fold["test_end"],
                 "signal_dates": len(fold_audit), "executable_signals": len(fold_executable),
                 "strategies": {}}
        for name in ("fixed_hybrid", "adaptive_hybrid"):
            field = strategy_signals[name]
            fold_signals = pd.DataFrame([
                {"date": item["date"], "signal": _signal_action(item[field])}
                for item in fold_executable
            ])
            fold_report = run_backtest(
                fold_prices, fold_signals, symbol=symbol, config=config,
                strategy_name=name, strategy_version=BACKTEST_VERSION,
                strategy_parameters={"hybrid_version": HYBRID_VERSION,
                                     "adaptive_method": ADAPTIVE_ENSEMBLE_VERSION if name == "adaptive_hybrid" else None,
                                     "weights": HYBRID_WEIGHTS if name == "fixed_hybrid" else None},
            )
            entry["strategies"][name] = fold_report["summary"]["strategy"]
        fold_strategy_comparison.append(entry)

    base = reports["fixed_hybrid"]
    selected_report = reports["adaptive_hybrid" if selected_mode == "adaptive" else "fixed_hybrid"]
    raw_columns = ["date", "open", "high", "low", "close", "volume"]
    source_hash = hashlib.sha256(raw[raw_columns].to_csv(index=False).encode("utf-8")).hexdigest()
    base.update({
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "strategy": "Fixed-weight hybrid walk-forward OOS signals",
        "strategy_version": BACKTEST_VERSION,
        "evaluation_version": "walk-forward-v2",
        "model_version": "direction-rf-walk-forward-v2 + direction-lstm-walk-forward-v2",
        "data": {**base["data"], "full_source_sha256": source_hash,
                 "labeled_rows": int(len(frame)), "final_holdout_excluded": True},
        "summary": {
            "strategy": selected_report["summary"]["strategy"],
            "hybrid": reports["fixed_hybrid"]["summary"]["strategy"],
            "fixed_hybrid": reports["fixed_hybrid"]["summary"]["strategy"],
            "adaptive_hybrid": reports["adaptive_hybrid"]["summary"]["strategy"],
            "calibrated_no_trade": reports["calibrated_no_trade"]["summary"]["strategy"],
            "lstm_only": reports["lstm_only"]["summary"]["strategy"],
            "random_forest_only": reports["random_forest_only"]["summary"]["strategy"],
            "buy_hold": reports["fixed_hybrid"]["summary"]["buy_hold"],
            "no_trade": reports["fixed_hybrid"]["summary"]["no_trade"],
        },
        "model_comparison": {name: report["summary"]["strategy"] for name, report in reports.items()},
        "model_curves": {name: report["strategy_curve"] for name, report in reports.items()},
        "model_trades": {name: report["trades"] for name, report in reports.items()},
        "model_signals": {name: report["signals"] for name, report in reports.items()},
        "model_signal_hashes": {name: report["data"]["signals_sha256"] for name, report in reports.items()},
        "folds": fold_metadata,
        "fold_strategy_comparison": fold_strategy_comparison,
        "calibration_version": CALIBRATION_VERSION,
        "no_trade_version": NO_TRADE_VERSION,
        "calibration_evaluation": {
            "event": TARGET_SPECIFICATION.copy(),
            "folds": fold_calibration_reports,
            "pooled_test": {
                name: calibration_metrics([item["target"] for item in audit], [item[key] for item in audit])
                for name, key in (
                    ("raw_fixed", "raw_ensemble_probability_up"),
                    ("calibrated_fixed", "calibrated_fixed_probability_up"),
                    ("raw_adaptive", "raw_adaptive_probability_up"),
                    ("calibrated_adaptive", "calibrated_adaptive_probability_up"),
                )
            },
            "no_trade_count": sum(item["trading_decision"] == "no_trade" for item in audit),
            "no_trade_coverage": round(1 - sum(item["trading_decision"] == "no_trade" for item in audit) / max(len(audit), 1), 4),
            "no_trade_backtest": reports["calibrated_no_trade"]["summary"]["strategy"],
        },
        "signal_audit": audit,
        "selected_mode": selected_mode,
        "adaptive_method": ADAPTIVE_ENSEMBLE_VERSION,
        "regime_model_version": REGIME_MODEL_VERSION,
        "configuration": {
            **base["configuration"], "hybrid_version": HYBRID_VERSION,
            "hybrid_weights": HYBRID_WEIGHTS, "sentiment_score": 0.0,
            "xgboost_enabled": bool(config.get("include_xgboost", False)),
            "selected_mode": selected_mode,
            "adaptive_ensemble": {"version": ADAPTIVE_ENSEMBLE_VERSION, **_adaptive_settings(config)},
            "adaptive_regime_context_enabled": bool(config.get("adaptive_use_regime", True)),
            "calibration": {"version": CALIBRATION_VERSION, "methods": ["platt", "isotonic"],
                            "validation_selection_fraction": 0.40},
            "no_trade": {"version": NO_TRADE_VERSION,
                         "selection_metric": "validation directional accuracy at minimum coverage"},
        },
        "assumptions": {
            **base["assumptions"],
            "signal_timestamp": "After each signal date close; NYSE session date, 16:00 ET convention.",
            "execution": "At the next available session open; NYSE session date, 09:30 ET convention.",
            "scope": "Expanding walk-forward development folds only; final chronological 20% holdout excluded.",
            "position_rules": "Long only; buy enters while flat, sell exits while invested, hold retains; no shorting.",
            "last_test_signal": "Included in audit; excluded from execution because no next open remains in-window.",
            "adaptive_weight_timing": "Weights are frozen from the preceding fold validation window for the upcoming test fold; test labels never select their own weights.",
            "calibration_timing": "Calibrators are selected using an early/late chronological split of each preceding validation window and refit on that validation window; current test labels are excluded.",
            "uncertainty": "Standard deviation of available LSTM, Random Forest and technical probabilities; disagreement heuristic, not a confidence interval.",
            "no_trade": "Validation-selected calibrated-neutral-band or component-dispersion cutoff; No Trade retains an existing position and does not open a new one (implemented as HOLD).",
        },
        "limitations": [
            "Historical point-in-time news is unavailable; the sentiment component is neutral in every row.",
            "The LSTM may use the evaluator's documented neutral fallback when TensorFlow training is unavailable or insufficient.",
            "Only expanding walk-forward development test rows are executed; the final 20% holdout is excluded.",
            "Daily Yahoo adjusted OHLCV cannot model intraday path, spreads, market impact, partial fills, taxes, or borrow costs.",
            "Fractional shares, fixed proportional transaction costs, and fixed slippage are simplified assumptions.",
            "The comparison covers the requested symbol and historical window; results do not establish future profitability.",
            "Adaptive weights use one preceding validation window per fold; regime-specific validation samples may be too small and trigger an overall-validation fallback.",
            "Historical FinBERT news is unavailable, so sentiment receives no adaptive historical weight; the fixed control retains its documented neutral contribution.",
        ],
    })
    base["strategy"] = "Adaptive hybrid walk-forward OOS signals" if selected_mode == "adaptive" else "Fixed-weight hybrid walk-forward OOS signals"
    base["selected_mode"] = selected_mode
    base["configuration"]["strategy"] = selected_report["strategy"]
    base["configuration"]["strategy_parameters"] = selected_report["configuration"]["strategy_parameters"]
    base["configuration"]["selected_mode"] = selected_mode
    base["versions"]["adaptive_ensemble"] = ADAPTIVE_ENSEMBLE_VERSION
    base["versions"]["regime_model"] = REGIME_MODEL_VERSION
    for key in ("equity", "cash", "holdings_value", "position", "drawdown_pct", "trades", "strategy_curve", "signals"):
        base[key] = selected_report[key]
    base["data"]["signals_sha256"] = selected_report["data"]["signals_sha256"]
    base["strategy_parameters"] = selected_report["configuration"]["strategy_parameters"]
    return base

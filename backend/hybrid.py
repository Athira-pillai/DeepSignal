"""Explicit probability-level hybrid for one-session direction predictions."""

import math


TARGET_SPECIFICATION = {
    "name": "next_session_up_direction",
    "meaning": "Probability that the next session close is above the current session close.",
    "horizon_sessions": 1,
    "decision_time": "After the current session close.",
    "outcome_time": "At the next session close.",
}

HYBRID_VERSION = "weighted-probability-v1"
HYBRID_WEIGHTS = {"lstm": 0.40, "random_forest": 0.40, "finbert": 0.10, "technical": 0.10}


def _probability(name, value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability between 0 and 1")
    return value


def combine_probabilities(lstm_probability, forest_probability, sentiment_score, technical_probability):
    """Blend compatible next-session-up probabilities and return an audit trail.

    FinBERT's [-1, 1] sentiment score is mapped linearly to [0, 1]. This is a
    transparent heuristic probability proxy, not a calibrated news forecast.
    The fixed weights are versioned and are not fitted on the final test set.
    """
    sentiment_score = float(sentiment_score)
    if not math.isfinite(sentiment_score) or not -1 <= sentiment_score <= 1:
        raise ValueError("sentiment_score must be finite and between -1 and 1")
    components = {
        "lstm": _probability("lstm_probability", lstm_probability),
        "random_forest": _probability("forest_probability", forest_probability),
        "finbert": (sentiment_score + 1.0) / 2.0,
        "technical": _probability("technical_probability", technical_probability),
    }
    probability_up = sum(HYBRID_WEIGHTS[key] * value for key, value in components.items())
    signal = (
        "strong_buy" if probability_up >= 0.64 else
        "buy" if probability_up >= 0.54 else
        "strong_sell" if probability_up <= 0.36 else
        "sell" if probability_up <= 0.46 else "hold"
    )
    return {
        "target": TARGET_SPECIFICATION.copy(),
        "hybrid_version": HYBRID_VERSION,
        "method": "fixed_weighted_average_of_direction_probabilities",
        "probability_up": float(probability_up),
        "confidence": float(max(probability_up, 1.0 - probability_up)),
        "signal": signal,
        "weights": HYBRID_WEIGHTS.copy(),
        "components": {
            key: {
                "probability_up": value,
                "weight": HYBRID_WEIGHTS[key],
                "weighted_contribution": value * HYBRID_WEIGHTS[key],
            }
            for key, value in components.items()
        },
        "sentiment_mapping": "(FinBERT positive-minus-negative score + 1) / 2; heuristic, not calibrated",
    }

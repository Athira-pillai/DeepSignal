"""Transparent adaptive weights from past out-of-sample validation evidence."""

import math


ADAPTIVE_ENSEMBLE_VERSION = "validation-brier-regime-weights-v1"
ADAPTIVE_COMPONENTS = ("lstm", "random_forest", "technical")
UNAVAILABLE_COMPONENTS = ("finbert",)
FIXED_AVAILABLE_PRIORS = {"lstm": 0.40, "random_forest": 0.40, "technical": 0.10}


def _project_bounded_simplex(values, lower, upper):
    """Euclidean projection onto a box-constrained probability simplex."""
    if not values:
        return {}
    count = len(values)
    if lower < 0 or upper <= 0 or count * lower > 1 + 1e-12 or count * upper < 1 - 1e-12:
        raise ValueError("weight bounds cannot contain a simplex for the available components")
    names = list(values)
    low = min(values.values()) - upper
    high = max(values.values()) - lower
    for _ in range(100):
        midpoint = (low + high) / 2
        total = sum(min(upper, max(lower, values[name] - midpoint)) for name in names)
        if total > 1:
            low = midpoint
        else:
            high = midpoint
    projected = {name: min(upper, max(lower, values[name] - high)) for name in names}
    residual = 1.0 - sum(projected.values())
    if abs(residual) > 1e-10:
        for name in names:
            capacity = upper - projected[name] if residual > 0 else projected[name] - lower
            adjustment = math.copysign(min(abs(residual), capacity), residual)
            projected[name] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-10:
                break
    return projected


def _eligible_rows(validation_rows, component, regime=None):
    rows = []
    for row in validation_rows:
        if regime is not None and row.get("regime") != regime:
            continue
        components = row.get("components", {})
        probability = components.get(component)
        if not row.get("availability", {}).get(component, False):
            continue
        try:
            probability = float(probability)
            target = int(row["target"])
        except (TypeError, ValueError, KeyError):
            continue
        if target not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            continue
        rows.append((target, probability))
    return rows


def _score_components(validation_rows, components, regime=None):
    scores = {}
    for component in components:
        rows = _eligible_rows(validation_rows, component, regime)
        if rows:
            brier = sum((probability - target) ** 2 for target, probability in rows) / len(rows)
            scores[component] = {"samples": len(rows), "brier_score": float(brier)}
    return scores


def select_adaptive_weights(
    validation_rows,
    *,
    current_regime=None,
    available_components=ADAPTIVE_COMPONENTS,
    minimum_weight=0.05,
    maximum_weight=0.80,
    temperature=8.0,
    minimum_validation_samples=20,
    minimum_regime_samples=20,
):
    """Use only prior OOS validation Brier scores to assign bounded weights.

    When there is not enough evidence for the current regime, the method uses
    the full preceding validation window. Missing components are omitted and
    priors are renormalized deterministically; unavailable sentiment is never
    assigned historical weight.
    """
    available = [name for name in ADAPTIVE_COMPONENTS if name in set(available_components)]
    config = {
        "minimum_weight": float(minimum_weight), "maximum_weight": float(maximum_weight),
        "temperature": float(temperature), "minimum_validation_samples": int(minimum_validation_samples),
        "minimum_regime_samples": int(minimum_regime_samples),
    }
    invalid_settings = (
        not 0 <= config["minimum_weight"] <= config["maximum_weight"] <= 1
        or config["temperature"] < 0
        or config["minimum_validation_samples"] < 1
        or config["minimum_regime_samples"] < 1
    )
    if invalid_settings:
        return {
            "status": "fixed_prior_fallback", "weights": {},
            "method": ADAPTIVE_ENSEMBLE_VERSION,
            "fallback_reason": "invalid_adaptive_configuration",
            "component_availability": {},
            "unavailable_components": list(UNAVAILABLE_COMPONENTS), "configuration": config,
        }
    if not available:
        return {
            "status": "unavailable", "weights": {}, "method": ADAPTIVE_ENSEMBLE_VERSION,
            "fallback_reason": "no_available_prediction_components", "component_availability": {},
            "unavailable_components": list(UNAVAILABLE_COMPONENTS), "configuration": config,
        }

    regime_scores = _score_components(validation_rows, available, current_regime) if current_regime else {}
    regime_has_evidence = bool(regime_scores) and all(
        score["samples"] >= config["minimum_regime_samples"] for score in regime_scores.values()
    )
    selected_scope = "regime_validation" if regime_has_evidence else "overall_validation"
    selected_scores = regime_scores if regime_has_evidence else _score_components(validation_rows, available)
    fallback_reasons = []
    if current_regime and not regime_has_evidence:
        fallback_reasons.append("insufficient_validation_rows_for_current_regime")

    evidence_components = [
        component for component in available
        if selected_scores.get(component, {}).get("samples", 0) >= config["minimum_validation_samples"]
    ]
    component_availability = {
        component: {"available_for_test": component in available,
                    "validation_evidence_available": component in evidence_components}
        for component in (*ADAPTIVE_COMPONENTS, *UNAVAILABLE_COMPONENTS)
    }
    if not evidence_components:
        fallback_reasons.append("insufficient_component_validation_evidence")
        return {
            "status": "fixed_prior_fallback", "weights": {},
            "method": ADAPTIVE_ENSEMBLE_VERSION,
            "fallback_reason": ";".join(fallback_reasons),
            "selected_scope": selected_scope, "validation_metrics": selected_scores,
            "component_availability": component_availability,
            "unavailable_components": list(UNAVAILABLE_COMPONENTS), "configuration": config,
        }

    priors = {name: FIXED_AVAILABLE_PRIORS[name] for name in evidence_components}
    prior_total = sum(priors.values())
    priors = {name: value / prior_total for name, value in priors.items()}
    qualities = {
        name: max(-0.25, min(0.25, 0.25 - selected_scores[name]["brier_score"]))
        for name in evidence_components
    }
    logits = {name: math.log(priors[name]) + config["temperature"] * qualities[name]
              for name in evidence_components}
    maximum_logit = max(logits.values())
    exponentials = {name: math.exp(value - maximum_logit) for name, value in logits.items()}
    denominator = sum(exponentials.values())
    unconstrained = {name: value / denominator for name, value in exponentials.items()}

    effective_minimum = min(config["minimum_weight"], 1.0 / len(evidence_components))
    effective_maximum = max(config["maximum_weight"], 1.0 / len(evidence_components))
    if effective_minimum != config["minimum_weight"] or effective_maximum != config["maximum_weight"]:
        fallback_reasons.append("effective_bounds_adjusted_for_available_component_count")
    weights = _project_bounded_simplex(unconstrained, effective_minimum, effective_maximum)
    return {
        "status": "ok",
        "weights": weights,
        "method": ADAPTIVE_ENSEMBLE_VERSION,
        "selected_scope": selected_scope,
        "current_regime": current_regime,
        "validation_metrics": selected_scores,
        "component_quality": qualities,
        "component_availability": component_availability,
        "unavailable_components": list(UNAVAILABLE_COMPONENTS),
        "fallback_reason": ";".join(fallback_reasons) or None,
        "configuration": {
            **config, "effective_minimum_weight": effective_minimum,
            "effective_maximum_weight": effective_maximum,
        },
        "interpretation": "Weights combine fixed component priors with validation Brier skill; validation rows precede the test interval.",
    }

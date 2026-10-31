"""Leakage-aware probability calibration and abstention helpers."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CALIBRATION_VERSION = "validation-selected-platt-isotonic-v1"
NO_TRADE_VERSION = "validation-selected-neutral-dispersion-v1"
METHODS = ("platt", "isotonic")


def _arrays(probabilities, labels):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(p) != len(y):
        raise ValueError("probabilities and labels must have equal length")
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    return np.clip(p[valid], 1e-6, 1 - 1e-6), y[valid]


def fit_calibrator(method, probabilities, labels):
    """Fit on caller-supplied past validation observations only."""
    p, y = _arrays(probabilities, labels)
    minimum = 20 if method == "platt" else 40
    if method not in METHODS:
        raise ValueError(f"unknown calibration method: {method}")
    if len(p) < minimum or np.unique(y).size < 2:
        return {"status": "unavailable", "method": method,
                "fallback_reason": "insufficient_validation_samples_or_class_diversity",
                "samples": int(len(p)), "estimator": None}
    if method == "platt":
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        estimator = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
        estimator.fit(logits, y)
    else:
        estimator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        estimator.fit(p, y)
    return {"status": "ok", "method": method, "samples": int(len(p)),
            "fallback_reason": None, "estimator": estimator}


def apply_calibrator(calibrator, probabilities):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if calibrator.get("status") != "ok":
        return np.clip(p, 0, 1)
    if calibrator["method"] == "platt":
        logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return calibrator["estimator"].predict_proba(logits)[:, 1]
    return np.asarray(calibrator["estimator"].predict(np.clip(p, 0, 1)), dtype=float)


def select_validation_calibrator(probabilities, labels, *, minimum_selection_samples=10):
    """Choose a method on a later validation slice, then refit on all validation rows.

    The chronological prefix fits candidates; the suffix selects by Brier score.
    The returned estimator is refit on all prior validation rows for test use.
    """
    p, y = _arrays(probabilities, labels)
    split = int(np.floor(len(p) * 0.60))
    prefix_p, prefix_y = p[:split], y[:split]
    suffix_p, suffix_y = p[split:], y[split:]
    if len(suffix_p) < minimum_selection_samples or np.unique(suffix_y).size < 2:
        return {"status": "unavailable", "method": None, "estimator": None,
                "selection_brier": {}, "candidate_statuses": {}, "selection_samples": int(len(suffix_p)),
                "fallback_reason": "insufficient_chronological_selection_validation"}
    scores = {}
    selection_predictions = {}
    candidate_statuses = {}
    for method in METHODS:
        candidate = fit_calibrator(method, prefix_p, prefix_y)
        if candidate["status"] != "ok":
            candidate_statuses[method] = {
                "status": candidate.get("status"),
                "fallback_reason": candidate.get("fallback_reason"),
                "samples": candidate.get("samples"),
            }
            continue
        candidate_statuses[method] = {"status": "ok", "samples": candidate["samples"]}
        calibrated = apply_calibrator(candidate, suffix_p)
        scores[method] = float(np.mean((calibrated - suffix_y) ** 2))
        selection_predictions[method] = calibrated
    if not scores:
        return {"status": "unavailable", "method": None, "estimator": None,
                "selection_brier": scores, "selection_samples": int(len(suffix_p)),
                "candidate_statuses": candidate_statuses,
                "fallback_reason": "no_calibrator_has_sufficient_prefix_evidence"}
    selected = min(scores, key=lambda name: (scores[name], METHODS.index(name)))
    final = fit_calibrator(selected, p, y)
    return {**final, "status": final["status"], "selection_brier": scores,
            "selection_samples": int(len(suffix_p)), "selected_method": selected,
            "candidate_statuses": candidate_statuses,
            "selection_probabilities": selection_predictions[selected].tolist(),
            "selection_labels": suffix_y.tolist(),
            "fallback_reason": final.get("fallback_reason")}


def select_no_trade_thresholds(probabilities, dispersions, labels, *, minimum_coverage=0.35):
    """Select neutral-band and dispersion cutoffs using validation rows only."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    d = np.asarray(dispersions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if not (len(p) == len(d) == len(y)):
        raise ValueError("No Trade inputs must have equal length")
    valid = np.isfinite(p) & np.isfinite(d) & np.isin(y, [0, 1])
    p, d, y = np.clip(p[valid], 0, 1), np.clip(d[valid], 0, 1), y[valid]
    if len(y) < 20 or np.unique(y).size < 2:
        return {"status": "unavailable", "neutral_margin": 0.0,
                "maximum_dispersion": 1.0, "samples": int(len(y)),
                "fallback_reason": "insufficient_threshold_selection_validation"}
    margins = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
    dispersion_limits = (1.0, 0.20, 0.15, 0.10, 0.075, 0.05)
    choices = []
    for margin in margins:
        for dispersion_limit in dispersion_limits:
            keep = (np.abs(p - 0.5) >= margin) & (d <= dispersion_limit)
            coverage = float(np.mean(keep))
            if keep.sum() < max(10, int(np.ceil(len(y) * minimum_coverage))):
                continue
            predictions = p[keep] >= 0.5
            accuracy = float(np.mean(predictions == y[keep]))
            choices.append((accuracy, coverage, -margin, dispersion_limit, margin))
    if not choices:
        return {"status": "unavailable", "neutral_margin": 0.0,
                "maximum_dispersion": 1.0, "samples": int(len(y)),
                "fallback_reason": "no_threshold_meets_minimum_coverage"}
    best = max(choices)
    return {"status": "ok", "neutral_margin": float(best[4]),
            "maximum_dispersion": float(best[3]), "validation_accuracy_when_active": float(best[0]),
            "validation_coverage": float(best[1]), "samples": int(len(y)),
            "fallback_reason": None,
            "meaning": "No Trade if calibrated direction probability is inside the neutral band or component disagreement exceeds the selected limit."}


def calibration_metrics(labels, probabilities):
    """Brier score and ten fixed-bin reliability points."""
    p, y = _arrays(probabilities, labels)
    if len(p) == 0:
        return {"status": "unavailable", "samples": 0, "brier_score": None, "reliability": []}
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    assignments = np.digitize(p, edges[1:-1], right=False)
    for index in range(10):
        left, right = edges[index], edges[index + 1]
        mask = assignments == index
        if mask.any():
            bins.append({"bin_start": round(float(left), 1), "bin_end": round(float(right), 1),
                         "count": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                         "observed_up_rate": float(y[mask].mean())})
    return {"status": "ok", "samples": int(len(p)),
            "brier_score": float(np.mean((p - y) ** 2)), "reliability": bins}

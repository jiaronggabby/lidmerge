"""Patient-level metrics and selection helpers used by the LidMerge protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    specificity: float
    sensitivity: float
    false_negatives: int
    false_positives: int


def _as_binary(values: Iterable[float | int]) -> np.ndarray:
    array = np.asarray(list(values), dtype=int)
    if array.ndim != 1 or not np.isin(array, (0, 1)).all():
        raise ValueError("labels must be one-dimensional binary values")
    return array


def _as_probability(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("probabilities must be one-dimensional finite values")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return array


def _as_decision(values: Iterable[bool | int]) -> np.ndarray:
    array = np.asarray(list(values), dtype=bool)
    if array.ndim != 1:
        raise ValueError("decisions must be one-dimensional values")
    return array


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(labels))
    error = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & ((probabilities < upper) if index < bins - 1 else (probabilities <= upper))
        if mask.any():
            error += float(mask.sum()) / total * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def _calibration_regression(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Regularized logistic calibration fit; used descriptively, not for model selection."""

    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def select_threshold_at_specificity(
    labels: Iterable[int], probabilities: Iterable[float], minimum_specificity: float = 0.90
) -> ThresholdResult:
    """Select a validation-only threshold without looking at any outer-test label."""

    y = _as_binary(labels)
    p = _as_probability(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if not 0.0 < float(minimum_specificity) <= 1.0:
        raise ValueError("minimum_specificity must be in (0, 1]")
    # The value just above the largest validation probability represents the
    # legitimate all-negative rule. It guarantees that a specificity-constrained
    # threshold exists even for a degenerate validation predictor that emits 1
    # for every patient.
    candidates = np.unique(np.concatenate(([0.0], p, [np.nextafter(float(p.max()), np.inf)])))
    rows: list[ThresholdResult] = []
    for threshold in candidates:
        predicted = p >= threshold
        tn = int(((y == 0) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        tp = int(((y == 1) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        specificity = tn / max(tn + fp, 1)
        sensitivity = tp / max(tp + fn, 1)
        rows.append(ThresholdResult(float(threshold), float(specificity), float(sensitivity), fn, fp))
    eligible = [row for row in rows if row.specificity >= float(minimum_specificity)]
    return max(eligible or rows, key=lambda row: (row.sensitivity, row.specificity, -row.threshold))


def patient_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float | None = None,
    decisions: Iterable[bool | int] | None = None,
    include_extended: bool = True,
) -> dict[str, float | int | None]:
    """Compute only metrics that are valid for patient-level binary predictions."""

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = _as_binary(labels)
    p = _as_probability(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if len(np.unique(y)) < 2:
        raise ValueError("both classes are required for patient-level metrics")
    if decisions is not None and threshold is not None:
        raise ValueError("provide either a threshold or precomputed patient-level decisions, not both")
    chosen: float | None
    if decisions is None:
        chosen = float(threshold) if threshold is not None else select_threshold_at_specificity(y, p).threshold
        predicted = p >= chosen
    else:
        predicted = _as_decision(decisions)
        if len(predicted) != len(y):
            raise ValueError("decisions must have the same length as labels")
        chosen = None
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    values: dict[str, float | int | None] = {
        "n_patients": int(len(y)),
        "n_malignant": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, p)),
        # Standardized pAUC over the prespecified high-specificity region.
        # This is a threshold-free summary and is therefore kept in the base
        # metric set used by point estimates, bootstrap draws, and paired
        # comparisons.
        "standardized_pauc_at_fpr_0_10": float(roc_auc_score(y, p, max_fpr=0.10)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": chosen,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "fnr": fn / max(tp + fn, 1),
        "ece_10": _expected_calibration_error(y, p, bins=10),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
    }
    if include_extended:
        calibration_intercept, calibration_slope = _calibration_regression(y, p)
        values.update({
            "standardized_pauc_at_fpr_0_20": float(roc_auc_score(y, p, max_fpr=0.20)),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
        })
    return values


def stratified_patient_bootstrap(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float | None,
    n_bootstrap: int,
    seed: int,
    decisions: Iterable[bool | int] | None = None,
) -> dict[str, list[float]]:
    """Return patient-resampled metric draws; callers decide which CIs to report."""

    y = _as_binary(labels)
    p = _as_probability(probabilities)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal length")
    d = _as_decision(decisions) if decisions is not None else None
    if d is not None and len(d) != len(y):
        raise ValueError("decisions must have the same length as labels")
    if d is None and threshold is None:
        raise ValueError("a threshold is required when decisions are not supplied")
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("bootstrap requires at least two patients in each class")
    draws: dict[str, list[float]] = {
        "auroc": [],
        "standardized_pauc_at_fpr_0_10": [],
        "auprc": [],
        "brier": [],
        "ece_10": [],
        "sensitivity": [],
        "specificity": [],
        "fnr": [],
    }
    for _ in range(int(n_bootstrap)):
        index = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        values = patient_metrics(
            y[index], p[index], threshold=threshold, decisions=d[index] if d is not None else None, include_extended=False
        )
        for key in draws:
            draws[key].append(float(values[key]))
    return draws

"""
Hybrid Random Forest + SVM probability fusion.

P_hybrid = w_RF * P_RF(Fall) + w_SVM * P_SVM(Fall)
Both classifiers always contribute: weights are non-negative, sum to 1, and
are constrained so neither model is a silent fallback (min weight 0.2).
Weights and the decision threshold are selected on validation data only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


MIN_WEIGHT = 0.20
WEIGHT_STEP = 0.05
THRESHOLD_GRID = np.round(np.arange(0.30, 0.76, 0.05), 2)


class HybridFusionError(RuntimeError):
    """Raised when fusion cannot be computed because a required model output is missing."""


def fuse_fall_probability(
    p_rf_fall: float,
    p_svm_fall: float,
    weight_rf: float,
    weight_svm: float,
    threshold: float,
) -> Dict[str, Any]:
    """Combine two Fall probabilities into a single Fall / Non-Fall decision."""
    if p_rf_fall is None or p_svm_fall is None:
        raise HybridFusionError("Both Random Forest and SVM Fall probabilities are required.")
    if weight_rf < 0 or weight_svm < 0:
        raise HybridFusionError("Fusion weights must be non-negative.")
    total = weight_rf + weight_svm
    if abs(total - 1.0) > 1e-6:
        raise HybridFusionError("Fusion weights must sum to 1.0.")

    p_hybrid = float(weight_rf * p_rf_fall + weight_svm * p_svm_fall)
    is_fall = bool(p_hybrid >= threshold)
    return {
        "hybrid_fall_probability": round(p_hybrid, 6),
        "is_fall": is_fall,
        "activity": "FALL" if is_fall else "NON_FALL",
        "weight_rf": weight_rf,
        "weight_svm": weight_svm,
        "decision_threshold": threshold,
        "rf_fall_probability": float(p_rf_fall),
        "svm_fall_probability": float(p_svm_fall),
    }


def select_fusion_parameters(
    p_rf_val: np.ndarray,
    p_svm_val: np.ndarray,
    y_fall_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Grid-search fusion weights and threshold on validation Fall labels only.
    Never call this with test-set probabilities.
    """
    y_true = np.asarray(y_fall_val).astype(int)
    p_rf = np.asarray(p_rf_val, dtype=float)
    p_svm = np.asarray(p_svm_val, dtype=float)
    if len(y_true) == 0:
        raise HybridFusionError("Validation set is empty; cannot select fusion parameters.")

    best: Optional[Dict[str, Any]] = None
    weights = np.round(np.arange(MIN_WEIGHT, 1.0 - MIN_WEIGHT + 1e-9, WEIGHT_STEP), 2)

    for w_rf in weights:
        w_svm = round(1.0 - float(w_rf), 2)
        if w_svm < MIN_WEIGHT:
            continue
        p_h = w_rf * p_rf + w_svm * p_svm
        for thr in THRESHOLD_GRID:
            y_pred = (p_h >= thr).astype(int)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            # Prefer F1, then recall (safety), then precision
            score = (f1, rec, prec)
            if best is None or score > best["_score"]:
                best = {
                    "weight_rf": float(w_rf),
                    "weight_svm": float(w_svm),
                    "threshold": float(thr),
                    "validation_fall_f1": f1,
                    "validation_fall_recall": rec,
                    "validation_fall_precision": prec,
                    "_score": score,
                }

    if best is None:
        raise HybridFusionError("Unable to select fusion parameters from validation data.")

    p_h_best = best["weight_rf"] * p_rf + best["weight_svm"] * p_svm
    if len(np.unique(y_true)) > 1:
        best["validation_fall_roc_auc"] = float(roc_auc_score(y_true, p_h_best))
    else:
        best["validation_fall_roc_auc"] = None
    best.pop("_score")
    return best


def extract_fall_probability(
    probabilities: Sequence[float],
    classes: Sequence[str],
    fall_class_name: str = "FALL",
) -> float:
    """Sum P(FALL) from a multi-class probability vector, or 0 if FALL is absent."""
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx = class_to_idx.get(fall_class_name)
    if idx is None or idx >= len(probabilities):
        return 0.0
    return float(probabilities[idx])

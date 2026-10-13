"""
FallGuard AI - Unit Tests for Inference & Two-Stage Fall Detection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from ml.inference.predictor import FallDetector

BASE_DIR = Path(__file__).resolve().parent.parent

def test_inference_missing_model_unavailable():
    """Verify missing model artifacts result in an explicit unavailable state."""
    detector = FallDetector(model_path=BASE_DIR / "nonexistent" / "model.joblib")
    assert detector.available is False
    assert detector.unavailable_reason is not None

    ax = np.zeros(128)
    ay = np.zeros(128)
    az = np.ones(128)
    gx = np.zeros(128)
    gy = np.zeros(128)
    gz = np.zeros(128)

    pred = detector.predict_window(ax, ay, az, gx, gy, gz)
    assert pred["model_available"] is False
    assert pred["activity"] == "UNAVAILABLE"
    assert pred["risk_level"] == "UNKNOWN"
    assert pred["is_fall"] is False
    assert pred["confidence"] == 0.0
    assert pred["unavailable_reason"] is not None


def test_hybrid_fusion_calculation():
    """Verify hybrid fusion probability math: P = w_rf*p_rf + w_svm*p_svm."""
    from ml.inference.hybrid_fusion import fuse_fall_probability

    # Normal activity below threshold
    res_normal = fuse_fall_probability(
        p_rf_fall=0.10,
        p_svm_fall=0.20,
        weight_rf=0.50,
        weight_svm=0.50,
        threshold=0.50,
    )
    assert res_normal["is_fall"] is False
    assert res_normal["activity"] == "NON_FALL"
    assert res_normal["hybrid_fall_probability"] == 0.15

    # Severe fall above threshold
    res_fall = fuse_fall_probability(
        p_rf_fall=0.90,
        p_svm_fall=0.80,
        weight_rf=0.60,
        weight_svm=0.40,
        threshold=0.50,
    )
    assert res_fall["is_fall"] is True
    assert res_fall["activity"] == "FALL"
    assert res_fall["hybrid_fall_probability"] == 0.86


def test_synthetic_hybrid_inference_with_mock_bundle(tmp_path):
    """Verify FallDetector executes hybrid RF + SVM inference correctly when bundle is present."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from ml.features.feature_engineering import extract_features_from_window

    # Create synthetic feature vector
    ax = np.zeros(128)
    ay = np.zeros(128)
    az = np.ones(128)
    gx = np.zeros(128)
    gy = np.zeros(128)
    gz = np.zeros(128)

    feats = extract_features_from_window(ax, ay, az, gx, gy, gz, sampling_rate=50.0)
    feat_names = sorted(list(feats.keys()))

    X = np.random.randn(20, len(feat_names))
    y = np.array(["WALKING", "FALL"] * 10)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    svm = SVC(probability=True, random_state=42).fit(X_scaled, y)

    bundle_path = tmp_path / "test_hybrid.joblib"
    joblib.dump({
        "rf_model": rf,
        "svm_model": svm,
        "scaler": scaler,
        "feature_names": feat_names,
        "classes": list(rf.classes_),
        "fall_class_idx": list(rf.classes_).index("FALL"),
        "model_name": "Hybrid Random Forest + SVM",
        "model_version": "2.0.0-test",
        "fusion": {
            "method": "probability_weighted_sum",
            "weight_rf": 0.50,
            "weight_svm": 0.50,
            "threshold": 0.50,
        }
    }, bundle_path)

    detector = FallDetector(model_path=bundle_path)
    assert detector.available is True

    pred = detector.predict_window(ax, ay, az, gx, gy, gz)
    assert pred["model_available"] is True
    assert pred["activity"] in ("FALL", "WALKING", "NON_FALL")
    assert "hybrid_fall_probability" in pred
    assert pred["rf_fall_probability"] is not None
    assert pred["svm_fall_probability"] is not None

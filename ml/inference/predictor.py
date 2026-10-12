"""
FallGuard AI - Hybrid Random Forest + SVM inference.

Loads both trained classifiers and the fitted preprocessing pipeline.
If either model is missing or fails to load, inference is unavailable —
there is no heuristic, untrained, or third-model fallback.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ml.features.feature_engineering import extract_features_from_window
from ml.inference.hybrid_fusion import fuse_fall_probability

logger = logging.getLogger("FallPredictor")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "fall_detection_model.joblib"


class ModelUnavailableError(RuntimeError):
    """Raised when the hybrid RF+SVM artifacts cannot be used for inference."""


class FallDetector:
    """Hybrid RF + SVM fall detector. Both models are required."""

    def __init__(self, model_path: Optional[Path] = None, window_size: int = 128, sampling_rate: float = 50.0):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.window_size = window_size
        self.sampling_rate = sampling_rate

        self.model_bundle: Optional[Dict[str, Any]] = None
        self.rf_model = None
        self.svm_model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.classes: List[str] = []
        self.fall_class_idx: int = -1
        self.model_name = "Hybrid Random Forest + SVM (not loaded)"
        self.model_version = "unavailable"
        self.fusion_config: Dict[str, Any] = {}
        self.available = False
        self.unavailable_reason: Optional[str] = "Models have not been loaded."
        self.buffer = deque(maxlen=self.window_size)
        self.load_model()

    @property
    def model(self):
        """Backward-compatible alias for primary classifier."""
        return self.rf_model

    def load_model(self) -> bool:
        """Load the hybrid RF+SVM bundle. Returns False if unavailable (never falls back)."""
        self.available = False
        self.rf_model = None
        self.svm_model = None
        if not self.model_path.exists():
            self.unavailable_reason = (
                f"Hybrid model file not found at {self.model_path}. "
                "Train with `python scripts/train_model.py` before enabling detection."
            )
            logger.error(self.unavailable_reason)
            return False

        try:
            bundle = joblib.load(self.model_path)
            rf = bundle.get("rf_model")
            svm = bundle.get("svm_model")
            # Reject legacy single-model bundles
            if rf is None or svm is None:
                self.unavailable_reason = (
                    "Model artifact is not a hybrid RF+SVM bundle (rf_model/svm_model missing). "
                    "Retrain with `python scripts/train_model.py`."
                )
                logger.error(self.unavailable_reason)
                return False
            if not hasattr(rf, "predict_proba") or not hasattr(svm, "predict_proba"):
                self.unavailable_reason = "RF and SVM must both expose predict_proba for probability fusion."
                logger.error(self.unavailable_reason)
                return False

            fusion = bundle.get("fusion") or {}
            w_rf = float(fusion.get("weight_rf", 0.5))
            w_svm = float(fusion.get("weight_svm", 0.5))
            if w_rf < 0 or w_svm < 0 or abs(w_rf + w_svm - 1.0) > 1e-6:
                self.unavailable_reason = "Invalid fusion weights in model artifact."
                logger.error(self.unavailable_reason)
                return False

            self.model_bundle = bundle
            self.rf_model = rf
            self.svm_model = svm
            self.scaler = bundle.get("scaler")
            self.feature_names = list(bundle.get("feature_names") or [])
            self.classes = list(bundle.get("classes") or [])
            self.fall_class_idx = int(bundle.get("fall_class_idx", -1))
            self.model_name = bundle.get("model_name", "Hybrid Random Forest + SVM")
            self.model_version = bundle.get("model_version", "2.0.0-hybrid-rf-svm")
            self.fusion_config = fusion
            self.available = True
            self.unavailable_reason = None
            logger.info(
                f"Loaded {self.model_name} ({len(self.feature_names)} features, "
                f"w_RF={w_rf}, w_SVM={w_svm}, threshold={fusion.get('threshold')})"
            )
            return True
        except Exception as exc:
            self.unavailable_reason = f"Failed to load hybrid models from {self.model_path}: {exc}"
            logger.error(self.unavailable_reason)
            return False

    def _class_proba(self, model, feature_vector: np.ndarray) -> Dict[str, float]:
        probs = model.predict_proba(feature_vector)[0]
        out = {}
        for idx, cls_name in enumerate(self.classes):
            if idx < len(probs):
                out[cls_name] = float(round(probs[idx], 6))
        return out

    def _fall_p(self, prob_dict: Dict[str, float]) -> float:
        return float(prob_dict.get("FALL", 0.0))

    def predict_window(
        self,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
    ) -> Dict[str, Any]:
        now_ts = datetime.now(timezone.utc).isoformat()

        if not self.available or self.rf_model is None or self.svm_model is None or not self.feature_names:
            return {
                "activity": "UNAVAILABLE",
                "is_fall": False,
                "confidence": 0.0,
                "risk_level": "UNKNOWN",
                "model_available": False,
                "unavailable_reason": self.unavailable_reason or "Hybrid models are not loaded.",
                "rf_fall_probability": None,
                "svm_fall_probability": None,
                "hybrid_fall_probability": None,
                "rf_activity": None,
                "svm_activity": None,
                "fusion": None,
                "probabilities": {},
                "model_name": self.model_name,
                "model_version": self.model_version,
                "timestamp": now_ts,
                "features_summary": {},
            }

        feats = extract_features_from_window(
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, sampling_rate=self.sampling_rate
        )
        feature_vector = np.array([[feats.get(name, 0.0) for name in self.feature_names]], dtype=float)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        rf_probs = self._class_proba(self.rf_model, feature_vector)
        if self.scaler is None:
            raise ModelUnavailableError("Fitted StandardScaler is missing from the hybrid bundle.")
        svm_input = self.scaler.transform(feature_vector)
        svm_probs = self._class_proba(self.svm_model, svm_input)

        p_rf = self._fall_p(rf_probs)
        p_svm = self._fall_p(svm_probs)
        fused = fuse_fall_probability(
            p_rf_fall=p_rf,
            p_svm_fall=p_svm,
            weight_rf=float(self.fusion_config["weight_rf"]),
            weight_svm=float(self.fusion_config["weight_svm"]),
            threshold=float(self.fusion_config["threshold"]),
        )

        rf_activity = max(rf_probs, key=rf_probs.get) if rf_probs else "UNKNOWN"
        svm_activity = max(svm_probs, key=svm_probs.get) if svm_probs else "UNKNOWN"
        is_fall = fused["is_fall"]
        p_hybrid = fused["hybrid_fall_probability"]
        activity = "FALL" if is_fall else (
            rf_activity if rf_activity != "FALL" else "NON_FALL"
        )
        if is_fall:
            risk_level = "HIGH"
        elif p_hybrid >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "activity": activity,
            "is_fall": is_fall,
            "confidence": round(float(p_hybrid if is_fall else (1.0 - p_hybrid)), 4),
            "risk_level": risk_level,
            "model_available": True,
            "unavailable_reason": None,
            "rf_fall_probability": round(p_rf, 4),
            "svm_fall_probability": round(p_svm, 4),
            "hybrid_fall_probability": round(p_hybrid, 4),
            "rf_activity": rf_activity,
            "svm_activity": svm_activity,
            "ml_activity": rf_activity,
            "ml_fall_probability": round(p_hybrid, 4),
            "safety_override": False,
            "fusion": {
                "method": self.fusion_config.get("method", "probability_weighted_sum"),
                "weight_rf": self.fusion_config.get("weight_rf"),
                "weight_svm": self.fusion_config.get("weight_svm"),
                "threshold": self.fusion_config.get("threshold"),
            },
            "probabilities": {
                "FALL": round(p_hybrid, 4),
                "NON_FALL": round(1.0 - p_hybrid, 4),
                "rf": rf_probs,
                "svm": svm_probs,
            },
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": now_ts,
            "features_summary": {
                "acc_mag_max": round(float(feats.get("acc_mag_max", 0.0)), 3),
                "gyro_mag_max": round(float(feats.get("gyro_mag_max", 0.0)), 3),
                "svmag_mean": round(float(feats.get("svmag_mean", feats.get("acc_mag_mean", 0.0))), 3),
                "sma": round(float(feats.get("sma", feats.get("acc_sma", 0.0))), 3),
                "jerk_max": round(float(feats.get("jerk_max", 0.0)), 3),
            },
        }

    def predict_from_array(self, window_data: np.ndarray) -> Dict[str, Any]:
        if window_data.shape[1] < 6:
            raise ValueError(f"Expected at least 6 sensor columns, got {window_data.shape[1]}")
        return self.predict_window(
            acc_x=window_data[:, 0],
            acc_y=window_data[:, 1],
            acc_z=window_data[:, 2],
            gyro_x=window_data[:, 3],
            gyro_y=window_data[:, 4],
            gyro_z=window_data[:, 5],
        )

    def add_sensor_reading(
        self,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> Optional[Dict[str, Any]]:
        self.buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        if len(self.buffer) == self.window_size:
            return self.predict_from_array(np.array(self.buffer))
        return None

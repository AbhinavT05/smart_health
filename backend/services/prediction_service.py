"""
FallGuard AI - Central Prediction Service (hybrid RF + SVM only)
"""

from datetime import datetime, timezone
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import numpy as np

from ml.inference.predictor import FallDetector
from backend.models.db_models import ActivityPrediction
from backend.services.fall_confirmation_service import fall_confirmation_service
from backend.services.alert_service import alert_service
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("PredictionService")


class PredictionService:
    def __init__(self):
        self.detector = FallDetector(
            model_path=settings.MODEL_PATH,
            window_size=settings.WINDOW_SAMPLES,
            sampling_rate=settings.SAMPLING_RATE,
        )

    def reload_model(self):
        return self.detector.load_model()

    async def predict_single_window(
        self,
        db: Session,
        acc_x: List[float],
        acc_y: List[float],
        acc_z: List[float],
        gyro_x: List[float],
        gyro_y: List[float],
        gyro_z: List[float],
        device_id: str = "WEARABLE_DEV_01",
        user_id: Optional[int] = None,
        location: Optional[Dict[str, float]] = None,
        require_confirmation: bool = True,
    ) -> Dict[str, Any]:
        pred = self.detector.predict_window(
            acc_x=np.array(acc_x),
            acc_y=np.array(acc_y),
            acc_z=np.array(acc_z),
            gyro_x=np.array(gyro_x),
            gyro_y=np.array(gyro_y),
            gyro_z=np.array(gyro_z),
        )

        if not pred.get("model_available"):
            await ws_manager.broadcast_to_user(user_id, {
                "type": "MODEL_UNAVAILABLE",
                "user_id": user_id,
                "data": {
                    "device_id": device_id,
                    "message": pred.get("unavailable_reason"),
                    "timestamp": pred["timestamp"],
                },
            })
            return pred

        act_record = ActivityPrediction(
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            activity=pred["activity"],
            confidence=pred["confidence"],
            risk_level=pred["risk_level"],
            is_fall=pred["is_fall"],
            device_id=device_id,
            model_version=pred["model_version"],
            rf_fall_probability=pred.get("rf_fall_probability"),
            svm_fall_probability=pred.get("svm_fall_probability"),
            hybrid_fall_probability=pred.get("hybrid_fall_probability"),
            model_available=True,
        )
        db.add(act_record)
        db.commit()

        if pred["is_fall"]:
            acc_peak = pred["features_summary"].get("acc_mag_max") if pred.get("features_summary") else None
            gyro_peak = pred["features_summary"].get("gyro_mag_max") if pred.get("features_summary") else None
            if require_confirmation:
                await fall_confirmation_service.initiate_fall_confirmation(
                    db=db,
                    confidence=pred["confidence"],
                    risk_level=pred["risk_level"],
                    device_id=device_id,
                    user_id=user_id,
                    ml_activity=pred.get("rf_activity") or pred.get("activity"),
                    ml_fall_probability=pred.get("hybrid_fall_probability") or 0.0,
                    safety_override=False,
                    acc_peak=acc_peak,
                    gyro_peak=gyro_peak,
                    location=location,
                    rf_fall_probability=pred.get("rf_fall_probability"),
                    svm_fall_probability=pred.get("svm_fall_probability"),
                    hybrid_fall_probability=pred.get("hybrid_fall_probability"),
                )
            else:
                await alert_service.trigger_fall_alert(
                    db=db,
                    confidence=pred["confidence"],
                    risk_level=pred["risk_level"],
                    device_id=device_id,
                    acc_peak=acc_peak,
                    gyro_peak=gyro_peak,
                    user_id=user_id,
                    ml_activity=pred.get("rf_activity"),
                    fall_probability=pred.get("hybrid_fall_probability"),
                    safety_override=False,
                    rf_fall_probability=pred.get("rf_fall_probability"),
                    svm_fall_probability=pred.get("svm_fall_probability"),
                    hybrid_fall_probability=pred.get("hybrid_fall_probability"),
                )

        ws_payload = {
            "type": "ACTIVITY_UPDATE",
            "user_id": user_id,
            "data": {
                "timestamp": pred["timestamp"],
                "activity": pred["activity"],
                "rf_activity": pred.get("rf_activity"),
                "svm_activity": pred.get("svm_activity"),
                "rf_fall_probability": pred.get("rf_fall_probability"),
                "svm_fall_probability": pred.get("svm_fall_probability"),
                "hybrid_fall_probability": pred.get("hybrid_fall_probability"),
                "ml_activity": pred.get("ml_activity", pred["activity"]),
                "ml_fall_probability": pred.get("hybrid_fall_probability", 0.0),
                "safety_override": False,
                "confidence": pred["confidence"],
                "risk_level": pred["risk_level"],
                "is_fall": pred["is_fall"],
                "device_id": device_id,
                "user_id": user_id,
                "fusion": pred.get("fusion"),
                "features_summary": pred.get("features_summary"),
            },
        }
        await ws_manager.broadcast_to_user(user_id, ws_payload)
        return pred


prediction_service = PredictionService()

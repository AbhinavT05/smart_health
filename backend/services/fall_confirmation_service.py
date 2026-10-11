"""
FallGuard AI - Fall Confirmation & False Alarm Mitigation Service
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session

from backend.database.session import SessionLocal
from backend.models.db_models import FallEvent, User
from backend.services.alert_service import alert_service
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("FallConfirmationService")


class PendingConfirmation:
    def __init__(self, fall_event_id: int, user_id: Optional[int], device_id: str, task: asyncio.Task):
        self.fall_event_id = fall_event_id
        self.user_id = user_id
        self.device_id = device_id
        self.task = task
        self.created_at = datetime.now(timezone.utc)


class FallConfirmationService:
    def __init__(self):
        self.pending_confirmations: Dict[int, PendingConfirmation] = {}
        self.last_incident_by_user: Dict[int, datetime] = {}

    def _dedupe_allows(self, user_id: Optional[int]) -> bool:
        if user_id is None:
            return True
        now = datetime.now(timezone.utc)
        last = self.last_incident_by_user.get(user_id)
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed < settings.ALERT_COOLDOWN_SECONDS:
                logger.info(
                    "Skipping duplicate fall incident for user %s (cooldown %ss, elapsed %.1fs). "
                    "This delays additional alerts by up to %ss.",
                    user_id,
                    settings.ALERT_COOLDOWN_SECONDS,
                    elapsed,
                    settings.ALERT_COOLDOWN_SECONDS,
                )
                return False
        self.last_incident_by_user[user_id] = now
        return True

    async def initiate_fall_confirmation(
        self,
        db: Session,
        confidence: float,
        risk_level: str = "HIGH",
        device_id: str = "phone_001",
        user_id: Optional[int] = None,
        ml_activity: str = "WALKING",
        ml_fall_probability: float = 0.0,
        safety_override: bool = False,
        acc_peak: Optional[float] = None,
        gyro_peak: Optional[float] = None,
        location: Optional[Dict[str, float]] = None,
        timeout_seconds: Optional[float] = None,
        rf_fall_probability: Optional[float] = None,
        svm_fall_probability: Optional[float] = None,
        hybrid_fall_probability: Optional[float] = None,
    ) -> Optional[FallEvent]:
        if not self._dedupe_allows(user_id):
            return None

        timeout = timeout_seconds or settings.FALL_CONFIRMATION_TIMEOUT_SECONDS
        now = datetime.now(timezone.utc)

        fall_event = FallEvent(
            user_id=user_id,
            device_id=device_id,
            timestamp=now,
            activity="FALL",
            ml_activity=ml_activity,
            confidence=round(confidence, 4),
            fall_probability=round(ml_fall_probability, 4),
            rf_fall_probability=rf_fall_probability,
            svm_fall_probability=svm_fall_probability,
            hybrid_fall_probability=hybrid_fall_probability,
            safety_override=False,
            risk_level=risk_level,
            event_kind="MODEL_PREDICTION",
            status="PENDING_CONFIRMATION",
            confirmed=False,
            alert_sent=False,
            acknowledged=False,
            acc_peak=acc_peak,
            gyro_peak=gyro_peak,
            location_lat=location.get("lat") if location else None,
            location_lon=location.get("lon") if location else None,
            notes=f"Hybrid model predicted a fall. Awaiting user confirmation ({timeout}s timeout).",
        )
        db.add(fall_event)
        db.commit()
        db.refresh(fall_event)

        ws_msg = {
            "type": "FALL_PENDING_CONFIRMATION",
            "user_id": user_id,
            "event": {
                "id": fall_event.id,
                "event_id": fall_event.id,
                "device_id": device_id,
                "user_id": user_id,
                "timestamp": now.isoformat(),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "timeout_seconds": timeout,
                "safety_override": False,
                "ml_activity": ml_activity,
                "ml_fall_probability": round(ml_fall_probability, 4),
                "rf_fall_probability": rf_fall_probability,
                "svm_fall_probability": svm_fall_probability,
                "hybrid_fall_probability": hybrid_fall_probability,
                "event_kind": "MODEL_PREDICTION",
                "message": "Possible fall detected by hybrid RF+SVM. Are you okay?",
            },
        }
        await ws_manager.broadcast_to_user(user_id, ws_msg)

        task = asyncio.create_task(self._timeout_handler(fall_event.id, timeout))
        self.pending_confirmations[fall_event.id] = PendingConfirmation(
            fall_event_id=fall_event.id,
            user_id=user_id,
            device_id=device_id,
            task=task,
        )
        return fall_event

    async def _timeout_handler(self, fall_event_id: int, timeout_seconds: float):
        try:
            await asyncio.sleep(timeout_seconds)
            await self.confirm_and_escalate_fall(fall_event_id, reason="TIMEOUT")
        except asyncio.CancelledError:
            pass

    async def handle_user_response(
        self,
        db: Session,
        fall_event_id: int,
        action: str,
        location: Optional[Dict[str, float]] = None,
        acting_user_id: Optional[int] = None,
    ) -> Optional[FallEvent]:
        action_norm = action.upper().replace(" ", "_").replace("'", "")
        pending = self.pending_confirmations.pop(fall_event_id, None)
        if pending and not pending.task.done():
            pending.task.cancel()

        query = db.query(FallEvent).filter(FallEvent.id == fall_event_id)
        if acting_user_id is not None:
            query = query.filter(FallEvent.user_id == acting_user_id)
        fall = query.first()
        if not fall:
            logger.warning("FallEvent #%s not found or not owned by user.", fall_event_id)
            return None

        if location:
            fall.location_lat = location.get("lat", fall.location_lat)
            fall.location_lon = location.get("lon", fall.location_lon)

        now = datetime.now(timezone.utc)
        if action_norm in ("I_AM_OK", "IM_OK", "OK", "SAFE", "CANCEL"):
            fall.status = "CANCELLED_BY_USER"
            fall.confirmed = False
            fall.acknowledged = True
            fall.acknowledged_at = now
            fall.acknowledged_by = "User (Smartphone)"
            fall.notes = (fall.notes or "") + "\nUser verified safe: Emergency escalation cancelled."
            db.commit()
            db.refresh(fall)
            await ws_manager.broadcast_to_user(fall.user_id, {
                "type": "FALL_CONFIRMATION_RESOLVED",
                "user_id": fall.user_id,
                "data": {
                    "event_id": fall.id,
                    "status": "CANCELLED_BY_USER",
                    "event_kind": fall.event_kind,
                    "device_id": fall.device_id,
                    "message": "User confirmed safe. Alert cancelled.",
                },
            })
            return fall

        if action_norm in ("NEED_HELP", "HELP", "CONFIRM", "FALL"):
            return await self.confirm_and_escalate_fall(fall_event_id, reason="MANUAL_HELP_REQUEST", db=db)

        return fall

    async def confirm_and_escalate_fall(
        self,
        fall_event_id: int,
        reason: str = "TIMEOUT",
        db: Optional[Session] = None,
    ) -> Optional[FallEvent]:
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            self.pending_confirmations.pop(fall_event_id, None)
            fall = db.query(FallEvent).filter(FallEvent.id == fall_event_id).first()
            if not fall:
                return None
            if fall.status in ("CANCELLED_BY_USER", "CONFIRMED"):
                return fall

            fall.status = "CONFIRMED"
            fall.confirmed = True
            fall.event_kind = "CONFIRMED_FALL"

            user = db.query(User).filter(User.id == fall.user_id).first() if fall.user_id else None
            location_dict = None
            if fall.location_lat is not None and fall.location_lon is not None:
                location_dict = {"lat": fall.location_lat, "lon": fall.location_lon}

            if user:
                await alert_service.dispatch_emergency_alert(
                    db=db,
                    fall_event=fall,
                    user=user,
                    location=location_dict,
                )
            else:
                logger.info("Confirmed fall #%s has no owning user; contacts were not notified.", fall.id)

            await ws_manager.broadcast_to_user(fall.user_id, {
                "type": "FALL_CONFIRMATION_RESOLVED",
                "user_id": fall.user_id,
                "data": {
                    "event_id": fall.id,
                    "status": "CONFIRMED",
                    "event_kind": "CONFIRMED_FALL",
                    "device_id": fall.device_id,
                    "reason": reason,
                    "message": f"Fall confirmed ({reason}). Emergency workflow executed.",
                    "simulated": alert_service.provider.simulated,
                },
            })
            return fall
        finally:
            if should_close:
                db.close()


fall_confirmation_service = FallConfirmationService()

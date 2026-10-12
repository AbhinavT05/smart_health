"""
FallGuard AI - Alert Management & Emergency Dispatch Service
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from backend.models.db_models import FallEvent, Alert, SystemLog, User, EmergencyContact
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("AlertService")


def _mask_phone(phone: str) -> str:
    if not phone or len(phone) < 4:
        return "***"
    return f"***{phone[-4:]}"


class ProviderResult:
    def __init__(
        self,
        status: str,
        simulated: bool,
        provider: str,
        provider_message_id: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        self.status = status
        self.simulated = simulated
        self.provider = provider
        self.provider_message_id = provider_message_id
        self.detail = detail
        self.ok = status in ("SIMULATED", "ACCEPTED", "DELIVERED", "UNCONFIRMED")


class BaseEmergencyProvider(ABC):
    @abstractmethod
    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any],
        channel: str = "sms",
    ) -> ProviderResult:
        pass

    @property
    def name(self) -> str:
        return "base"

    @property
    def simulated(self) -> bool:
        return True


class SimulationEmergencyProvider(BaseEmergencyProvider):
    """Explicit development simulation. Never reports that a real person was contacted."""

    def __init__(self):
        self.dispatched_alerts: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "simulation"

    @property
    def simulated(self) -> bool:
        return True

    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any],
        channel: str = "sms",
    ) -> ProviderResult:
        incident = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_name": user_name,
            "contact_name": contact_name,
            "contact_phone_masked": _mask_phone(contact_phone),
            "contact_phone": contact_phone,
            "channel": channel,
            "event_id": event.get("fall_id"),
            "event": event,
        }
        self.dispatched_alerts.append(incident)
        logger.info(
            "[SIMULATION] Emergency message composed for contact %s (%s) user=%s event=%s. "
            "No real message was sent.",
            contact_name,
            _mask_phone(contact_phone),
            user_name,
            event.get("fall_id"),
        )
        return ProviderResult(
            status="SIMULATED",
            simulated=True,
            provider="simulation",
            detail="Simulation mode: no real SMS or email was sent.",
        )


# Backward-compatible alias used by existing tests
MockEmergencyProvider = SimulationEmergencyProvider


class TwilioEmergencyProvider(BaseEmergencyProvider):
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.dispatched_alerts: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "twilio"

    @property
    def simulated(self) -> bool:
        return False

    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any],
        channel: str = "sms",
    ) -> ProviderResult:
        if channel != "sms":
            return ProviderResult(
                status="FAILED",
                simulated=False,
                provider="twilio",
                detail=f"Twilio provider does not send channel '{channel}'.",
            )
        if not (self.account_sid and self.auth_token and self.from_number):
            return ProviderResult(
                status="FAILED",
                simulated=False,
                provider="twilio",
                detail="Twilio credentials are incomplete.",
            )

        message_body = (
            f"EMERGENCY: {user_name} needs help via FallGuard AI. "
            f"Time (UTC): {event.get('timestamp')}. "
            "This is not a guarantee of emergency-service dispatch."
        )
        loc = event.get("location")
        if loc and loc.get("lat") is not None and loc.get("lon") is not None:
            message_body += f" Location: https://maps.google.com/?q={loc['lat']},{loc['lon']}"

        try:
            import httpx

            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    data={"From": self.from_number, "To": contact_phone, "Body": message_body},
                    auth=(self.account_sid, self.auth_token),
                    timeout=10.0,
                )
            if resp.status_code in (200, 201):
                sid = None
                try:
                    sid = resp.json().get("sid")
                    twilio_status = resp.json().get("status", "queued")
                except Exception:
                    twilio_status = "accepted"
                logger.info("Twilio accepted SMS for event %s (sid present: %s).", event.get("fall_id"), bool(sid))
                return ProviderResult(
                    status="ACCEPTED",
                    simulated=False,
                    provider="twilio",
                    provider_message_id=sid,
                    detail=f"Twilio accepted the message (provider status: {twilio_status}). Delivery is not confirmed.",
                )
            logger.error("Twilio SMS failed with HTTP %s", resp.status_code)
            return ProviderResult(
                status="FAILED",
                simulated=False,
                provider="twilio",
                detail=f"Twilio rejected the request (HTTP {resp.status_code}).",
            )
        except Exception as exc:
            logger.error("Twilio dispatch error: %s", exc)
            return ProviderResult(
                status="FAILED",
                simulated=False,
                provider="twilio",
                detail="Network or provider timeout while sending SMS.",
            )


def get_emergency_provider() -> BaseEmergencyProvider:
    if settings.emergency_mode() == "twilio":
        return TwilioEmergencyProvider(
            account_sid=settings.TWILIO_ACCOUNT_SID,
            auth_token=settings.TWILIO_AUTH_TOKEN,
            from_number=settings.TWILIO_FROM_NUMBER,
        )
    return SimulationEmergencyProvider()


class AlertService:
    def __init__(self):
        self.last_alert_time_by_user: Dict[int, datetime] = {}
        self.provider: BaseEmergencyProvider = get_emergency_provider()

    def set_provider(self, provider: BaseEmergencyProvider):
        self.provider = provider

    def _cooldown_allows(self, user_id: Optional[int]) -> bool:
        key = user_id if user_id is not None else -1
        now = datetime.now(timezone.utc)
        last = self.last_alert_time_by_user.get(key)
        if last and now - last < timedelta(seconds=settings.ALERT_COOLDOWN_SECONDS):
            return False
        self.last_alert_time_by_user[key] = now
        return True

    def selected_contacts(self, db: Session, user_id: int) -> List[EmergencyContact]:
        contacts = (
            db.query(EmergencyContact)
            .filter(EmergencyContact.user_id == user_id, EmergencyContact.selected == True)  # noqa: E712
            .order_by(EmergencyContact.id.asc())
            .all()
        )
        if not contacts:
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.emergency_contact_name and user.emergency_contact_phone:
                legacy = EmergencyContact(
                    user_id=user.id,
                    name=user.emergency_contact_name,
                    phone_number=user.emergency_contact_phone,
                    relationship_label="Primary Contact",
                    channel="sms",
                    selected=True,
                    verified=True,
                )
                db.add(legacy)
                db.commit()
                db.refresh(legacy)
                contacts = [legacy]
        return contacts

    def _compose_message(self, user: User, event_kind: str, timestamp: datetime, location: Optional[Dict[str, float]]) -> str:
        kind = "manual help request" if event_kind == "MANUAL_HELP" else "confirmed fall"
        msg = f"EMERGENCY: {user.name} triggered a {kind} via FallGuard AI at {timestamp.isoformat()} UTC."
        if user.share_location and location and location.get("lat") is not None:
            msg += f" Location: https://maps.google.com/?q={location['lat']},{location['lon']}"
        msg += " Delivery is not guaranteed. This does not dispatch public emergency services."
        return msg

    async def dispatch_to_contacts(
        self,
        db: Session,
        fall_event: FallEvent,
        user: User,
        location: Optional[Dict[str, float]] = None,
        alert_type: str = "CRITICAL_FALL",
    ) -> List[Dict[str, Any]]:
        contacts = self.selected_contacts(db, user.id)
        if not contacts:
            logger.info("No selected emergency contacts for user %s; not dispatching.", user.id)
            fall_event.alert_sent = False
            fall_event.notes = (fall_event.notes or "") + "\nNo selected emergency contacts configured."
            db.commit()
            return []

        loc = location
        if not user.share_location:
            loc = None

        body = self._compose_message(user, fall_event.event_kind or "CONFIRMED_FALL", fall_event.timestamp, loc)
        event_data = {
            "fall_id": fall_event.id,
            "timestamp": fall_event.timestamp.isoformat(),
            "risk_level": fall_event.risk_level,
            "confidence": fall_event.confidence,
            "device_id": fall_event.device_id,
            "location": loc,
            "message": body,
        }

        deliveries = []
        any_attempt = False
        for contact in contacts:
            any_attempt = True
            result = await self.provider.send_emergency_alert(
                user_name=user.name,
                contact_name=contact.name,
                contact_phone=contact.phone_number,
                event=event_data,
                channel=contact.channel or "sms",
            )
            alert = Alert(
                user_id=user.id,
                fall_event_id=fall_event.id,
                contact_id=contact.id,
                alert_type=alert_type,
                channel=contact.channel or "sms",
                message=body,
                delivery_status=result.status,
                provider=result.provider,
                provider_message_id=result.provider_message_id,
                created_at=datetime.now(timezone.utc),
            )
            db.add(alert)
            deliveries.append({
                "contact_id": contact.id,
                "contact_name": contact.name,
                "channel": contact.channel or "sms",
                "delivery_status": result.status,
                "provider": result.provider,
                "simulated": result.simulated,
                "provider_message_id": result.provider_message_id,
                "detail": result.detail,
            })

        fall_event.alert_sent = any_attempt
        fall_event.emergency_contact = f"{len(contacts)} selected contact(s)"
        db.commit()
        return deliveries

    async def dispatch_emergency_alert(
        self,
        db: Session,
        fall_event: FallEvent,
        user: Optional[User] = None,
        location: Optional[Dict[str, float]] = None,
    ) -> bool:
        if not user:
            logger.info("Cannot dispatch emergency alert without an authenticated user.")
            return False
        deliveries = await self.dispatch_to_contacts(db, fall_event, user, location)
        return len(deliveries) > 0

    async def trigger_fall_alert(
        self,
        db: Session,
        confidence: float,
        risk_level: str = "HIGH",
        device_id: str = "WEARABLE_DEV_01",
        acc_peak: Optional[float] = None,
        gyro_peak: Optional[float] = None,
        notes: Optional[str] = None,
        user_id: Optional[int] = None,
        ml_activity: Optional[str] = None,
        fall_probability: Optional[float] = None,
        safety_override: bool = False,
        confirmed: bool = True,
        event_kind: str = "CONFIRMED_FALL",
        rf_fall_probability: Optional[float] = None,
        svm_fall_probability: Optional[float] = None,
        hybrid_fall_probability: Optional[float] = None,
    ) -> Optional[FallEvent]:
        if not self._cooldown_allows(user_id):
            logger.info("Fall alert ignored due to per-user cooldown (%ss).", settings.ALERT_COOLDOWN_SECONDS)
            return None

        now = datetime.now(timezone.utc)
        fall_event = FallEvent(
            user_id=user_id,
            device_id=device_id,
            timestamp=now,
            activity="FALL",
            ml_activity=ml_activity or "FALL",
            confidence=round(confidence, 4),
            fall_probability=fall_probability or confidence,
            rf_fall_probability=rf_fall_probability,
            svm_fall_probability=svm_fall_probability,
            hybrid_fall_probability=hybrid_fall_probability,
            safety_override=False,
            risk_level=risk_level,
            event_kind=event_kind,
            status="UNACKNOWLEDGED" if confirmed else "PENDING_CONFIRMATION",
            confirmed=confirmed,
            acknowledged=False,
            acc_peak=acc_peak,
            gyro_peak=gyro_peak,
            notes=notes or f"Fall signature ({confidence*100:.1f}%).",
        )
        db.add(fall_event)
        db.commit()
        db.refresh(fall_event)

        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        if confirmed and user:
            await self.dispatch_to_contacts(db, fall_event, user)

        ws_payload = {
            "type": "FALL_ALERT",
            "user_id": user_id,
            "event": {
                "id": fall_event.id,
                "timestamp": now.isoformat(),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "status": fall_event.status,
                "event_kind": fall_event.event_kind,
                "device_id": device_id,
                "acc_peak": acc_peak,
                "gyro_peak": gyro_peak,
            },
        }
        await ws_manager.broadcast_to_user(user_id, ws_payload) if user_id else await ws_manager.broadcast(ws_payload)
        return fall_event

    def acknowledge_fall(
        self,
        db: Session,
        fall_id: int,
        acknowledged_by: str = "Operator",
        notes: Optional[str] = None,
        owner_user_id: Optional[int] = None,
    ) -> Optional[FallEvent]:
        query = db.query(FallEvent).filter(FallEvent.id == fall_id)
        if owner_user_id is not None:
            query = query.filter(FallEvent.user_id == owner_user_id)
        fall = query.first()
        if not fall:
            return None

        now = datetime.now(timezone.utc)
        fall.acknowledged = True
        fall.acknowledged_at = now
        fall.acknowledged_by = acknowledged_by
        fall.status = "ACKNOWLEDGED"
        if notes:
            fall.notes = (fall.notes + f"\n[Caregiver Note]: {notes}") if fall.notes else notes

        alert = db.query(Alert).filter(Alert.fall_event_id == fall_id).first()
        if alert:
            alert.acknowledged_at = now
        db.commit()
        db.refresh(fall)
        return fall

    def add_fall_notes(self, db: Session, fall_id: int, notes: str, owner_user_id: Optional[int] = None) -> Optional[FallEvent]:
        query = db.query(FallEvent).filter(FallEvent.id == fall_id)
        if owner_user_id is not None:
            query = query.filter(FallEvent.user_id == owner_user_id)
        fall = query.first()
        if not fall:
            return None
        now = datetime.now(timezone.utc)
        fall.notes = (fall.notes + f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]: {notes}") if fall.notes else notes
        db.commit()
        db.refresh(fall)
        return fall


alert_service = AlertService()

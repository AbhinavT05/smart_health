"""
FallGuard AI - Emergency Contacts & Manual Help API Endpoints
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.models.db_models import User, EmergencyContact, FallEvent, Alert
from backend.schemas.schemas import (
    EmergencyContactCreate,
    EmergencyContactUpdate,
    EmergencyContactResponse,
    SendHelpRequest,
    SendHelpResponse,
    DeliveryResult,
)
from backend.api.dependencies import get_current_user
from backend.services.alert_service import alert_service
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("ContactsRoute")

router = APIRouter(tags=["Emergency Contacts & Help"])


# -------------------------------------------------------------
# Emergency Contacts CRUD & Selection
# -------------------------------------------------------------

@router.get("/contacts", response_model=List[EmergencyContactResponse])
@router.get("/emergency-contacts", response_model=List[EmergencyContactResponse])
def get_emergency_contacts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all emergency contacts belonging to the authenticated user."""
    contacts = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.user_id == current_user.id)
        .order_by(EmergencyContact.id.asc())
        .all()
    )
    # Auto-seed legacy contact if no records exist yet
    if not contacts and current_user.emergency_contact_name and current_user.emergency_contact_phone:
        contacts = alert_service.selected_contacts(db, current_user.id)
    return contacts


@router.post("/contacts", response_model=EmergencyContactResponse, status_code=status.HTTP_201_CREATED)
@router.post("/emergency-contacts", response_model=EmergencyContactResponse, status_code=status.HTTP_201_CREATED)
def create_emergency_contact(
    payload: EmergencyContactCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new emergency contact scoped strictly to the authenticated user."""
    contact = EmergencyContact(
        user_id=current_user.id,
        name=payload.name.strip(),
        phone_number=payload.phone_number.strip(),
        relationship_label=payload.relationship_label.strip() if payload.relationship_label else None,
        channel=payload.channel.lower().strip(),
        selected=payload.selected,
        verified=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return contact


@router.get("/contacts/{contact_id}", response_model=EmergencyContactResponse)
@router.get("/emergency-contacts/{contact_id}", response_model=EmergencyContactResponse)
def get_emergency_contact(
    contact_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve specific emergency contact with strict ownership verification."""
    contact = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.id == contact_id, EmergencyContact.user_id == current_user.id)
        .first()
    )
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emergency contact #{contact_id} not found or not owned by user."
        )
    return contact


@router.put("/contacts/{contact_id}", response_model=EmergencyContactResponse)
@router.put("/emergency-contacts/{contact_id}", response_model=EmergencyContactResponse)
def update_emergency_contact(
    contact_id: int,
    payload: EmergencyContactUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update contact attributes with strict ownership verification."""
    contact = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.id == contact_id, EmergencyContact.user_id == current_user.id)
        .first()
    )
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emergency contact #{contact_id} not found or not owned by user."
        )

    if payload.name is not None:
        contact.name = payload.name.strip()
    if payload.phone_number is not None:
        contact.phone_number = payload.phone_number.strip()
    if payload.relationship_label is not None:
        contact.relationship_label = payload.relationship_label.strip()
    if payload.channel is not None:
        contact.channel = payload.channel.lower().strip()
    if payload.selected is not None:
        contact.selected = payload.selected

    contact.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(contact)
    return contact


@router.delete("/contacts/{contact_id}", status_code=status.HTTP_200_OK)
@router.delete("/emergency-contacts/{contact_id}", status_code=status.HTTP_200_OK)
def delete_emergency_contact(
    contact_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete contact with strict ownership verification."""
    contact = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.id == contact_id, EmergencyContact.user_id == current_user.id)
        .first()
    )
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emergency contact #{contact_id} not found or not owned by user."
        )
    db.delete(contact)
    db.commit()
    return {"status": "deleted", "contact_id": contact_id}


@router.put("/contacts/{contact_id}/select", response_model=EmergencyContactResponse)
@router.put("/emergency-contacts/{contact_id}/select", response_model=EmergencyContactResponse)
def toggle_contact_selection(
    contact_id: int,
    selected: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Select or deselect an emergency contact for fall escalation dispatch."""
    contact = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.id == contact_id, EmergencyContact.user_id == current_user.id)
        .first()
    )
    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emergency contact #{contact_id} not found or not owned by user."
        )
    contact.selected = selected
    contact.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(contact)
    return contact


# -------------------------------------------------------------
# Manual Send Help Action
# -------------------------------------------------------------

@router.post("/help", response_model=SendHelpResponse)
@router.post("/send-help", response_model=SendHelpResponse)
@router.post("/emergency/help", response_model=SendHelpResponse)
async def manual_send_help(
    payload: SendHelpRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Trigger manual emergency assistance request to authenticated user's selected contacts.
    Works even if ML model is unavailable. Enforces confirmation and idempotency.
    """
    if not payload.confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit in-app confirmation is required before dispatching emergency help."
        )

    now = datetime.now(timezone.utc)

    # 1. Idempotency Check & Duplicate Protection
    if payload.idempotency_key:
        existing = (
            db.query(FallEvent)
            .filter(
                FallEvent.user_id == current_user.id,
                FallEvent.idempotency_key == payload.idempotency_key
            )
            .first()
        )
        if existing:
            alerts = (
                db.query(Alert)
                .filter(Alert.fall_event_id == existing.id)
                .order_by(Alert.created_at.asc())
                .all()
            )
            deliveries = [
                DeliveryResult(
                    contact_id=a.contact_id or 0,
                    contact_name=(a.contact.name if a.contact else "Emergency Contact"),
                    channel=a.channel,
                    delivery_status=a.delivery_status,
                    provider=a.provider or alert_service.provider.name,
                    simulated=(a.delivery_status == "SIMULATED" or alert_service.provider.simulated),
                    provider_message_id=a.provider_message_id,
                    detail=f"Reused previous dispatch ({a.delivery_status})."
                )
                for a in alerts
            ]
            return SendHelpResponse(
                event_id=existing.id,
                event_kind=existing.event_kind,
                status=existing.status,
                simulated=alert_service.provider.simulated,
                provider=alert_service.provider.name,
                message="Duplicate request recognized; reused existing emergency dispatch.",
                deliveries=deliveries,
                reused_existing=True
            )

    # 2. Cooldown check
    if not alert_service._cooldown_allows(current_user.id):
        # Look for the most recent incident to reuse or notify
        recent = (
            db.query(FallEvent)
            .filter(FallEvent.user_id == current_user.id)
            .order_by(FallEvent.timestamp.desc())
            .first()
        )
        if recent:
            alerts = db.query(Alert).filter(Alert.fall_event_id == recent.id).all()
            deliveries = [
                DeliveryResult(
                    contact_id=a.contact_id or 0,
                    contact_name=(a.contact.name if a.contact else "Emergency Contact"),
                    channel=a.channel,
                    delivery_status=a.delivery_status,
                    provider=a.provider or alert_service.provider.name,
                    simulated=(a.delivery_status == "SIMULATED" or alert_service.provider.simulated),
                    provider_message_id=a.provider_message_id,
                    detail="Alert cooldown active: Suppressed duplicate dispatch storm."
                )
                for a in alerts
            ]
            return SendHelpResponse(
                event_id=recent.id,
                event_kind=recent.event_kind,
                status=recent.status,
                simulated=alert_service.provider.simulated,
                provider=alert_service.provider.name,
                message=f"Alert cooldown active ({settings.ALERT_COOLDOWN_SECONDS}s). Duplicate dispatch suppressed.",
                deliveries=deliveries,
                reused_existing=True
            )

    # 3. Create FallEvent record for manual assistance
    if payload.share_location is not None:
        current_user.share_location = payload.share_location
        db.commit()

    loc_lat = payload.location.get("lat") if payload.location else None
    loc_lon = payload.location.get("lon") if payload.location else None

    fall_event = FallEvent(
        user_id=current_user.id,
        device_id=payload.device_id or "MANUAL_HELP_WEB",
        timestamp=now,
        activity="MANUAL_HELP",
        ml_activity="MANUAL",
        confidence=1.0,
        risk_level="HIGH",
        event_kind="MANUAL_HELP",
        status="UNACKNOWLEDGED",
        confirmed=True,
        alert_sent=False,
        acknowledged=False,
        location_lat=loc_lat if current_user.share_location else None,
        location_lon=loc_lon if current_user.share_location else None,
        idempotency_key=payload.idempotency_key,
        notes="Manual Send Help action triggered by user.",
    )
    db.add(fall_event)
    db.commit()
    db.refresh(fall_event)

    # 4. Dispatch alert strictly to user's selected contacts
    delivery_dicts = await alert_service.dispatch_to_contacts(
        db=db,
        fall_event=fall_event,
        user=current_user,
        location=payload.location if current_user.share_location else None,
        alert_type="MANUAL_HELP"
    )

    deliveries = [DeliveryResult(**d) for d in delivery_dicts]

    # 5. Push targeted notification to user's connected monitors and devices
    ws_msg = {
        "type": "HELP_RESULT",
        "user_id": current_user.id,
        "data": {
            "event_id": fall_event.id,
            "event_kind": "MANUAL_HELP",
            "status": fall_event.status,
            "simulated": alert_service.provider.simulated,
            "deliveries": delivery_dicts,
            "message": "Manual emergency assistance request dispatched to selected contacts."
        }
    }
    await ws_manager.broadcast_to_user(current_user.id, ws_msg)

    return SendHelpResponse(
        event_id=fall_event.id,
        event_kind=fall_event.event_kind,
        status=fall_event.status,
        simulated=alert_service.provider.simulated,
        provider=alert_service.provider.name,
        message=f"Manual emergency alert dispatched to {len(deliveries)} contact(s).",
        deliveries=deliveries,
        reused_existing=False
    )

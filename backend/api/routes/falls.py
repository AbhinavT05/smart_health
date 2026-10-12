"""
FallGuard AI - Fall Incident & Caregiver Management API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.session import get_db
from backend.models.db_models import FallEvent, Alert, User
from backend.schemas.schemas import FallEventResponse, FallAcknowledgeRequest, FallNoteRequest
from backend.services.alert_service import alert_service
from backend.api.dependencies import get_current_user

router = APIRouter(prefix="/falls", tags=["Fall Events"])

@router.get("", response_model=List[FallEventResponse])
def get_fall_events(
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List historical fall incidents sorted newest first, scoped to authenticated user."""
    query = db.query(FallEvent)
    if current_user.role != "ADMIN":
        query = query.filter(FallEvent.user_id == current_user.id)
    if status:
        query = query.filter(FallEvent.status == status.upper())
    falls = query.order_by(FallEvent.timestamp.desc()).offset(offset).limit(limit).all()
    return falls

@router.get("/{fall_id}", response_model=FallEventResponse)
def get_fall_event(
    fall_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve detailed information for a specific fall incident with ownership verification."""
    query = db.query(FallEvent).filter(FallEvent.id == fall_id)
    if current_user.role != "ADMIN":
        query = query.filter(FallEvent.user_id == current_user.id)
    fall = query.first()
    if not fall:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found or access denied.")
    return fall

@router.post("/{fall_id}/acknowledge", response_model=FallEventResponse)
def acknowledge_fall_event(
    fall_id: int,
    payload: FallAcknowledgeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a fall event as acknowledged by clinical staff / caregiver with ownership check."""
    owner_id = None if current_user.role == "ADMIN" else current_user.id
    updated = alert_service.acknowledge_fall(
        db=db,
        fall_id=fall_id,
        acknowledged_by=payload.acknowledged_by,
        notes=payload.notes,
        owner_user_id=owner_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found or access denied.")
    return updated

@router.post("/{fall_id}/notes", response_model=FallEventResponse)
def add_fall_investigation_note(
    fall_id: int,
    payload: FallNoteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add clinical notes to an ongoing or past fall incident with ownership check."""
    owner_id = None if current_user.role == "ADMIN" else current_user.id
    updated = alert_service.add_fall_notes(
        db=db,
        fall_id=fall_id,
        notes=payload.notes,
        owner_user_id=owner_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found or access denied.")
    return updated

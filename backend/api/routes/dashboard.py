"""
FallGuard AI - Central Dashboard Summary API
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, date, timezone
from typing import Dict, Any

from backend.database.session import get_db
from backend.models.db_models import ActivityPrediction, FallEvent, Alert, EmergencyContact, User
from backend.schemas.schemas import DashboardSummary, AlertResponse, ActivityHistoryItem, FallEventResponse
from backend.api.dependencies import get_current_user
from backend.services.auth_service import build_user_response
from backend.services.simulation_service import simulation_service
from backend.services.prediction_service import prediction_service
from backend.services.sensor_stream_manager import sensor_stream_manager
from backend.core.config import settings

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve comprehensive summary statistics for the monitoring dashboard scoped to the authenticated user."""
    # Current active state for this user
    latest_pred = (
        db.query(ActivityPrediction)
        .filter(ActivityPrediction.user_id == current_user.id)
        .order_by(ActivityPrediction.timestamp.desc())
        .first()
    )

    detector = prediction_service.detector
    model_available = detector.available
    model_name = detector.model_name

    if latest_pred:
        current_activity = latest_pred.activity
        current_confidence = round(latest_pred.confidence, 4)
        current_risk = latest_pred.risk_level
        hybrid_p = latest_pred.hybrid_fall_probability
        rf_p = latest_pred.rf_fall_probability
        svm_p = latest_pred.svm_fall_probability
    else:
        current_activity = "NO_DATA"
        current_confidence = 0.0
        current_risk = "UNKNOWN"
        hybrid_p = None
        rf_p = None
        svm_p = None

    total_acts = db.query(ActivityPrediction).filter(ActivityPrediction.user_id == current_user.id).count()
    total_falls = db.query(FallEvent).filter(FallEvent.user_id == current_user.id).count()

    today_start = datetime.combine(date.today(), datetime.min.time())
    falls_today = (
        db.query(FallEvent)
        .filter(FallEvent.user_id == current_user.id, FallEvent.timestamp >= today_start)
        .count()
    )
    unack_falls = (
        db.query(FallEvent)
        .filter(FallEvent.user_id == current_user.id, FallEvent.acknowledged == False)  # noqa: E712
        .count()
    )

    last_fall = (
        db.query(FallEvent)
        .filter(FallEvent.user_id == current_user.id)
        .order_by(FallEvent.timestamp.desc())
        .first()
    )

    # Active unacknowledged alerts for this user
    alerts = (
        db.query(Alert)
        .filter(Alert.user_id == current_user.id, Alert.acknowledged_at == None)  # noqa: E712
        .order_by(Alert.created_at.desc())
        .limit(5)
        .all()
    )
    recent_acts = (
        db.query(ActivityPrediction)
        .filter(ActivityPrediction.user_id == current_user.id)
        .order_by(ActivityPrediction.timestamp.desc())
        .limit(15)
        .all()
    )
    recent_falls = (
        db.query(FallEvent)
        .filter(FallEvent.user_id == current_user.id)
        .order_by(FallEvent.timestamp.desc())
        .limit(10)
        .all()
    )

    user_phone_connected = any(
        buf.connected for buf in sensor_stream_manager.devices.values() if buf.user_id == current_user.id
    )
    is_connected = user_phone_connected

    if not model_available:
        system_status = "MODEL_UNAVAILABLE"
    elif is_connected:
        system_status = "ONLINE"
    else:
        system_status = "READY"

    selected_contacts_count = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.user_id == current_user.id, EmergencyContact.selected == True)  # noqa: E712
        .count()
    )

    return DashboardSummary(
        user=build_user_response(db, current_user),
        current_activity=current_activity,
        current_confidence=current_confidence,
        current_risk=current_risk,
        hybrid_fall_probability=hybrid_p,
        rf_fall_probability=rf_p,
        svm_fall_probability=svm_p,
        model_available=model_available,
        model_name=model_name,
        system_status=system_status,
        sensor_connected=is_connected,
        total_activities_count=total_acts,
        total_falls_count=total_falls,
        falls_today_count=falls_today,
        unacknowledged_falls_count=unack_falls,
        last_fall_timestamp=last_fall.timestamp if last_fall else None,
        selected_contact_count=selected_contacts_count,
        has_emergency_contact=selected_contacts_count > 0,
        emergency_provider_mode=settings.emergency_mode(),
        location_sharing_enabled=bool(current_user.share_location) if current_user.share_location is not None else True,
        active_alerts=[AlertResponse.model_validate(a) for a in alerts],
        recent_activities=[ActivityHistoryItem.model_validate(a) for a in recent_acts],
        recent_events=[FallEventResponse.model_validate(f) for f in recent_falls]
    )

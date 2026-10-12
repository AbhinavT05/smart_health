"""
FallGuard AI - Activity Telemetry & History API Endpoints
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database.session import get_db
from backend.models.db_models import ActivityPrediction, User
from backend.schemas.schemas import ActivityHistoryItem
from backend.api.dependencies import get_current_user

router = APIRouter(prefix="/activities", tags=["Activities"])

@router.get("/history", response_model=List[ActivityHistoryItem])
def get_activity_history(
    activity: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve chronologically ordered time-series predictions scoped to authenticated user."""
    query = db.query(ActivityPrediction)
    if current_user.role != "ADMIN":
        query = query.filter(ActivityPrediction.user_id == current_user.id)
    if activity:
        query = query.filter(ActivityPrediction.activity == activity.upper())
    if risk_level:
        query = query.filter(ActivityPrediction.risk_level == risk_level.upper())
        
    records = query.order_by(ActivityPrediction.timestamp.desc()).offset(offset).limit(limit).all()
    return records

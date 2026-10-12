"""
FallGuard AI - Authentication API Endpoints
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.schemas.schemas import (
    UserLogin,
    UserRegister,
    TokenResponse,
    UserResponse,
    LegacyEmergencyContactUpdate,
    DeviceStatus,
    DeviceResponse,
    DeviceRegisterRequest,
)
from backend.services.auth_service import auth_service
from backend.services.sensor_stream_manager import sensor_stream_manager
from backend.core.security import hash_password
from backend.api.dependencies import get_current_user
from backend.models.db_models import User, Device, EmergencyContact

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user with email and password."""
    token_resp = auth_service.authenticate_user(db, login_data)
    if not token_resp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    return token_resp

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user account. Role is strictly assigned server-side as PATIENT to prevent privilege escalation."""
    clean_email = user_data.email.strip().lower()
    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered."
        )
    # Server-side safe role assignment: public users can NEVER register as ADMIN or any privileged role
    safe_role = "PATIENT"
    try:
        user = User(
            name=user_data.name.strip(),
            email=clean_email,
            hashed_password=hash_password(user_data.password),
            role=safe_role,
            phone_number=user_data.phone_number,
            share_location=True,
            emergency_contact_name=user_data.emergency_contact_name,
            emergency_contact_phone=user_data.emergency_contact_phone,
            emergency_contact_email=user_data.emergency_contact_email,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        if user_data.emergency_contact_name and user_data.emergency_contact_phone:
            contact = EmergencyContact(
                user_id=user.id,
                name=user_data.emergency_contact_name.strip(),
                phone_number=user_data.emergency_contact_phone.strip(),
                relationship_label="Primary Contact",
                channel="sms",
                selected=True,
                verified=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(contact)
            db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during user registration: {str(exc)}"
        )

    login_data = UserLogin(email=clean_email, password=user_data.password)
    token_resp = auth_service.authenticate_user(db, login_data)
    if not token_resp:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration completed, but token generation failed."
        )
    return token_resp

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve current authenticated user/patient/caregiver profile."""
    return current_user

@router.put("/emergency-contact", response_model=UserResponse)
def update_emergency_contact(
    payload: LegacyEmergencyContactUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure designated emergency contact details for fall escalation (legacy compatibility)."""
    current_user.emergency_contact_name = payload.emergency_contact_name
    current_user.emergency_contact_phone = payload.emergency_contact_phone
    if payload.emergency_contact_email is not None:
        current_user.emergency_contact_email = payload.emergency_contact_email
    if payload.phone_number is not None:
        current_user.phone_number = payload.phone_number

    contact = db.query(EmergencyContact).filter(EmergencyContact.user_id == current_user.id).first()
    if not contact:
        contact = EmergencyContact(
            user_id=current_user.id,
            name=payload.emergency_contact_name,
            phone_number=payload.emergency_contact_phone,
            relationship_label="Primary Contact",
            channel="sms",
            selected=True,
            verified=True,
        )
        db.add(contact)
    else:
        contact.name = payload.emergency_contact_name
        contact.phone_number = payload.emergency_contact_phone
        contact.selected = True

    db.commit()
    db.refresh(current_user)
    return current_user

@router.get("/devices", response_model=List[Dict[str, Any]])
def get_connected_devices(current_user: User = Depends(get_current_user)):
    """List connected smartphone devices and in-memory buffer statistics scoped to authenticated user."""
    statuses = sensor_stream_manager.get_all_device_statuses()
    if current_user.role == "ADMIN":
        return statuses
    # Strictly return devices belonging to authenticated user; unowned devices are never exposed
    return [s for s in statuses if s.get("user_id") == current_user.id]

@router.post("/devices", response_model=DeviceResponse)
def register_device(
    payload: DeviceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Register smartphone device ID to authenticated user account."""
    device = (
        db.query(Device)
        .filter(Device.user_id == current_user.id, Device.device_id == payload.device_id)
        .first()
    )
    if not device:
        device = Device(
            user_id=current_user.id,
            device_id=payload.device_id,
            display_name=payload.display_name or payload.device_id,
            last_seen=datetime.now(timezone.utc),
        )
        db.add(device)
    else:
        if payload.display_name:
            device.display_name = payload.display_name
        device.last_seen = datetime.now(timezone.utc)
    db.commit()
    db.refresh(device)
    return device

@router.delete("/devices/{device_id}")
def unregister_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unregister device ID ensuring strict ownership verification."""
    device = (
        db.query(Device)
        .filter(Device.user_id == current_user.id, Device.device_id == device_id)
        .first()
    )
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found or not owned by user."
        )
    db.delete(device)
    db.commit()
    return {"status": "deleted", "device_id": device_id}

"""
FallGuard AI - Authentication Service
"""

from typing import Optional
from sqlalchemy.orm import Session
from backend.models.db_models import User, EmergencyContact
from backend.core.security import verify_password, create_access_token
from backend.core.config import settings
from backend.schemas.schemas import UserLogin, TokenResponse, UserResponse


def build_user_response(db: Session, user: User) -> UserResponse:
    selected = (
        db.query(EmergencyContact)
        .filter(EmergencyContact.user_id == user.id, EmergencyContact.selected == True)  # noqa: E712
        .count()
    )
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        phone_number=user.phone_number,
        share_location=bool(user.share_location) if user.share_location is not None else True,
        created_at=user.created_at,
        selected_contact_count=selected,
    )


class AuthService:
    def authenticate_user(self, db: Session, login_data: UserLogin) -> Optional[TokenResponse]:
        user = db.query(User).filter(User.email == login_data.email.strip().lower()).first()
        if not user:
            return None
        if not verify_password(login_data.password, user.hashed_password):
            return None

        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "name": user.name,
        }
        token = create_access_token(data=token_data)
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            user=build_user_response(db, user),
        )


auth_service = AuthService()

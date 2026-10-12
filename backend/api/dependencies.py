"""
FallGuard AI - FastAPI Route Dependencies
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional

from backend.database.session import get_db
from backend.core.security import decode_access_token
from backend.models.db_models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get authenticated user if token present, otherwise None."""
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    subject = payload.get("sub")
    if subject is None:
        return None
    # Login issues the database ID as a string subject. Treat malformed or
    # stale subjects as unauthenticated instead of leaking a 500 response.
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        return None
    if user_id <= 0:
        return None
    return db.query(User).filter(User.id == user_id).first()

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Enforce authentication."""
    user = get_current_user_optional(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

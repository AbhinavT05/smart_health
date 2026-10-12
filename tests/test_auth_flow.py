"""
FallGuard AI - End-to-End User Registration & Authentication Flow Tests
Validates the complete signup-to-login lifecycle according to project requirements:
1. Register a new user with valid details.
2. Confirm user is saved in the database with secure hashed password and safe PATIENT role.
3. Reject duplicate email registration with HTTP 409 Conflict and clear message.
4. Reject invalid or missing registration fields (missing name, bad email, short password, mismatched confirmation).
5. Log in with newly registered credentials and receive valid JWT token.
6. Reject incorrect passwords and unknown accounts with HTTP 401 Unauthorized.
7. Verify protected dashboard and fall-event APIs require valid authentication and enforce data isolation.
8. Verify existing system functionality remains intact.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.session import SessionLocal
from backend.models.db_models import User, EmergencyContact, FallEvent
from backend.core.security import verify_password

client = TestClient(app)


def test_1_and_2_register_valid_user_and_verify_persistence():
    """1. Register a new user with valid details. 2. Confirm user is saved in DB."""
    email = "newpatient@testcare.org"
    password = "SecurePassword123!"

    payload = {
        "name": "Alice Cooper",
        "email": email,
        "password": password,
        "confirm_password": password,
        "phone_number": "+15559876543",
        "emergency_contact_name": "Bob Cooper",
        "emergency_contact_phone": "+15559870000"
    }

    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201, f"Expected 201 Created, got {response.status_code}: {response.text}"
    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == email
    assert data["user"]["name"] == "Alice Cooper"
    assert data["user"]["role"] == "PATIENT"

    # Confirm user is actually persisted in database
    db = SessionLocal()
    try:
        user_in_db = db.query(User).filter(User.email == email).first()
        assert user_in_db is not None, "User must be persisted in database"
        assert user_in_db.name == "Alice Cooper"
        assert user_in_db.role == "PATIENT"

        # Password must be hashed with bcrypt, NEVER stored in plain-text
        assert user_in_db.hashed_password != password
        assert user_in_db.hashed_password.startswith("$2b$") or user_in_db.hashed_password.startswith("$2a$")
        assert verify_password(password, user_in_db.hashed_password) is True

        # Emergency contact must be created and linked
        contact = db.query(EmergencyContact).filter(EmergencyContact.user_id == user_in_db.id).first()
        assert contact is not None
        assert contact.name == "Bob Cooper"
        assert contact.phone_number == "+15559870000"
    finally:
        db.close()


def test_3_reject_duplicate_email_registration():
    """3. Register again with the same email and confirm a clear duplicate-account error (409 Conflict)."""
    email = "newpatient@testcare.org"
    payload = {
        "name": "Alice Duplicate",
        "email": email,
        "password": "AnotherPassword456!",
        "confirm_password": "AnotherPassword456!"
    }

    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 409, f"Expected 409 Conflict for duplicate email, got {response.status_code}"
    data = response.json()
    assert "already registered" in data.get("detail", "").lower()


def test_4_reject_invalid_or_missing_fields():
    """4. Attempt registration with invalid or missing fields."""
    # 4a. Missing name
    res1 = client.post("/api/auth/register", json={
        "email": "noname@test.org",
        "password": "ValidPassword123!",
        "confirm_password": "ValidPassword123!"
    })
    assert res1.status_code == 422

    # 4b. Invalid email format
    res2 = client.post("/api/auth/register", json={
        "name": "Invalid Email",
        "email": "not-an-email",
        "password": "ValidPassword123!",
        "confirm_password": "ValidPassword123!"
    })
    assert res2.status_code == 422

    # 4c. Password too short (< 8 chars)
    res3 = client.post("/api/auth/register", json={
        "name": "Short Pass",
        "email": "shortpass@test.org",
        "password": "short",
        "confirm_password": "short"
    })
    assert res3.status_code == 422

    # 4d. Password confirmation mismatch
    res4 = client.post("/api/auth/register", json={
        "name": "Mismatch Pass",
        "email": "mismatch@test.org",
        "password": "ValidPassword123!",
        "confirm_password": "DifferentPassword999!"
    })
    assert res4.status_code == 422

    # 4e. Attempt privilege escalation: public signup trying role="ADMIN" must stay PATIENT
    res5 = client.post("/api/auth/register", json={
        "name": "Hacker Attempt",
        "email": "hacker@test.org",
        "password": "ValidPassword123!",
        "confirm_password": "ValidPassword123!",
        "role": "ADMIN"
    })
    assert res5.status_code == 201
    assert res5.json()["user"]["role"] == "PATIENT", "Public signup must NEVER grant ADMIN privileges"


def test_5_login_with_registered_credentials():
    """5. Log in with the newly registered credentials."""
    login_payload = {
        "email": "newpatient@testcare.org",
        "password": "SecurePassword123!"
    }
    response = client.post("/api/auth/login", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "newpatient@testcare.org"
    assert data["user"]["role"] == "PATIENT"


def test_6_reject_incorrect_passwords():
    """6. Verify incorrect passwords and non-existent accounts are rejected."""
    # Incorrect password
    res1 = client.post("/api/auth/login", json={
        "email": "newpatient@testcare.org",
        "password": "WrongPassword999!"
    })
    assert res1.status_code == 401
    assert "incorrect" in res1.json().get("detail", "").lower()

    # Non-existent user
    res2 = client.post("/api/auth/login", json={
        "email": "nobody@nowhere.com",
        "password": "SomePassword123!"
    })
    assert res2.status_code == 401


def test_7_protected_apis_require_authentication_and_prevent_cross_user_access():
    """7. Verify protected dashboard and fall-event APIs require valid authentication and enforce data isolation."""
    # 7a. Unauthenticated requests rejected
    assert client.get("/api/dashboard/summary").status_code == 401
    assert client.get("/api/falls").status_code == 401
    assert client.get("/api/auth/me").status_code == 401

    # 7b. Invalid bearer token rejected
    assert client.get("/api/dashboard/summary", headers={"Authorization": "Bearer invalid_token"}).status_code == 401

    # 7c. User Alice logs in
    alice_login = client.post("/api/auth/login", json={
        "email": "newpatient@testcare.org",
        "password": "SecurePassword123!"
    }).json()
    alice_token = alice_login["access_token"]
    alice_headers = {"Authorization": f"Bearer {alice_token}"}

    # Alice can view her own dashboard and profile
    alice_me = client.get("/api/auth/me", headers=alice_headers)
    assert alice_me.status_code == 200
    assert alice_me.json()["email"] == "newpatient@testcare.org"

    alice_dash = client.get("/api/dashboard/summary", headers=alice_headers)
    assert alice_dash.status_code == 200
    assert alice_dash.json()["user"]["email"] == "newpatient@testcare.org"

    # Alice cannot access another user's fall event
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.role == "ADMIN").first()
        other_fall = FallEvent(
            user_id=admin_user.id,
            device_id="OTHER_DEVICE",
            confidence=0.95,
            risk_level="HIGH",
            status="UNACKNOWLEDGED"
        )
        db.add(other_fall)
        db.commit()
        db.refresh(other_fall)
        other_fall_id = other_fall.id
    finally:
        db.close()

    # Alice attempts to access admin's fall incident #other_fall_id
    forbidden_res = client.get(f"/api/falls/{other_fall_id}", headers=alice_headers)
    assert forbidden_res.status_code == 404, "Patient cannot access other user's fall incidents"


def test_8_existing_functionality_remains_intact():
    """8. Verify existing seeded accounts, health check, and model info remain functional."""
    # Default admin login works
    admin_res = client.post("/api/auth/login", json={
        "email": "admin@fallguard.ai",
        "password": "admin123"
    })
    assert admin_res.status_code == 200
    assert admin_res.json()["user"]["role"] == "ADMIN"

    # Default caregiver login works
    cg_res = client.post("/api/auth/login", json={
        "email": "caregiver@fallguard.ai",
        "password": "caregiver123"
    })
    assert cg_res.status_code == 200
    assert cg_res.json()["user"]["role"] == "CAREGIVER"

    # Health check works
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    # Model info works
    model_res = client.get("/api/model/info")
    assert model_res.status_code == 200
    assert "model_name" in model_res.json()

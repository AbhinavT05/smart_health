"""
FallGuard AI - Integration Tests for FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.init_db import init_db
from backend.database.session import SessionLocal
from backend.models.db_models import User
from backend.core.security import create_access_token

# Initialize database
init_db()
client = TestClient(app)

def get_auth_headers():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.role == "ADMIN").first()
        if not user:
            user = db.query(User).first()
        token = create_access_token({"sub": str(user.id), "email": user.email, "role": user.role})
        return {"Authorization": f"Bearer {token}"}
    finally:
        db.close()

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "FallGuard AI"

def test_dashboard_summary():
    # 1. Unauthenticated request must be rejected with 401
    unauth_res = client.get("/api/dashboard/summary")
    assert unauth_res.status_code == 401

    # 2. Authenticated request must succeed with 200
    headers = get_auth_headers()
    response = client.get("/api/dashboard/summary", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "current_activity" in data
    assert "total_falls_count" in data
    assert "system_status" in data

def test_predict_endpoint():
    # 128 samples dummy window
    samples = 128
    payload = {
        "acc_x": [0.1] * samples,
        "acc_y": [0.2] * samples,
        "acc_z": [0.98] * samples,
        "gyro_x": [0.0] * samples,
        "gyro_y": [0.0] * samples,
        "gyro_z": [0.0] * samples,
        "device_id": "TEST_DEVICE_01"
    }
    # 1. Unauthenticated request rejected with 401
    unauth_res = client.post("/api/predict", json=payload)
    assert unauth_res.status_code == 401

    # 2. Authenticated request succeeds with 200
    headers = get_auth_headers()
    response = client.post("/api/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "activity" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert "is_fall" in data

def test_fall_events_and_acknowledgment():
    # 1. Unauthenticated request rejected with 401
    unauth_res = client.get("/api/falls")
    assert unauth_res.status_code == 401

    # 2. Authenticated request succeeds with 200
    headers = get_auth_headers()
    response = client.get("/api/falls", headers=headers)
    assert response.status_code == 200
    falls = response.json()
    assert isinstance(falls, list)
    
    if len(falls) > 0:
        fall_id = falls[0]["id"]
        # Acknowledge fall
        ack_payload = {
            "acknowledged_by": "Test Staff",
            "notes": "Patient inspected. Normal condition."
        }
        ack_res = client.post(f"/api/falls/{fall_id}/acknowledge", json=ack_payload, headers=headers)
        assert ack_res.status_code == 200
        ack_data = ack_res.json()
        assert ack_data["acknowledged"] is True
        assert ack_data["status"] == "ACKNOWLEDGED"

def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data

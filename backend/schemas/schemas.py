"""
FallGuard AI - Pydantic Request & Response Schemas
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

E164_PHONE = re.compile(r"^\+[1-9]\d{7,14}$")
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserLogin(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean


class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: Optional[str] = None
    role: Optional[str] = "PATIENT"
    phone_number: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_email: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Name must be at least 2 characters.")
        return cleaned

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

    @model_validator(mode="after")
    def check_passwords_match(self):
        if self.confirm_password is not None and self.password != self.confirm_password:
            raise ValueError("Password confirmation does not match password.")
        return self


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    phone_number: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_email: Optional[str] = None
    share_location: bool = True
    created_at: datetime
    selected_contact_count: int = 0

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = 1440
    user: UserResponse


class EmergencyContactCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone_number: str
    relationship_label: Optional[str] = Field(default=None, max_length=80)
    channel: str = Field(default="sms")
    selected: bool = True

    @field_validator("phone_number")
    @classmethod
    def e164(cls, v: str) -> str:
        phone = v.strip().replace(" ", "")
        if not E164_PHONE.match(phone):
            raise ValueError("Phone number must be in international E.164 format, e.g. +15551234567.")
        return phone

    @field_validator("channel")
    @classmethod
    def channel_ok(cls, v: str) -> str:
        allowed = {"sms", "email"}
        val = v.lower().strip()
        if val not in allowed:
            raise ValueError("Channel must be 'sms' or 'email'.")
        return val


class EmergencyContactUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    phone_number: Optional[str] = None
    relationship_label: Optional[str] = Field(default=None, max_length=80)
    channel: Optional[str] = None
    selected: Optional[bool] = None

    @field_validator("phone_number")
    @classmethod
    def e164(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        phone = v.strip().replace(" ", "")
        if not E164_PHONE.match(phone):
            raise ValueError("Phone number must be in international E.164 format, e.g. +15551234567.")
        return phone


class EmergencyContactResponse(BaseModel):
    id: int
    name: str
    phone_number: str
    relationship_label: Optional[str] = None
    channel: str
    selected: bool
    verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DeviceResponse(BaseModel):
    id: int
    device_id: str
    display_name: Optional[str] = None
    last_seen: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., min_length=4, max_length=120)
    display_name: Optional[str] = None


class SensorAxisData(BaseModel):
    x: float
    y: float
    z: float


class SmartphoneSensorMessage(BaseModel):
    type: str = "sensor_data"
    version: int = 1
    device_id: str = "phone_001"
    timestamp: float = Field(..., description="Timestamp in epoch seconds or milliseconds")
    unit: str = Field(default="m/s2", description="Accelerometer units: 'm/s2' (Android default) or 'g'")
    accelerometer: SensorAxisData
    gyroscope: SensorAxisData
    location: Optional[Dict[str, float]] = None


class FallConfirmationPayload(BaseModel):
    type: str = "fall_confirmation"
    event_id: int
    action: str = Field(..., description="'I_AM_OK' or 'NEED_HELP'")
    location: Optional[Dict[str, float]] = None


class DeviceStatus(BaseModel):
    device_id: str
    user_id: Optional[int] = None
    connected: bool = False
    last_seen: Optional[datetime] = None
    samples_received: int = 0
    buffer_samples: int = 0
    buffer_seconds: float = 0.0


class SensorReadingCreate(BaseModel):
    device_id: str = "WEARABLE_DEV_01"
    timestamp: Optional[datetime] = None
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    session_id: Optional[str] = None


class SensorStreamBatch(BaseModel):
    device_id: str = "WEARABLE_DEV_01"
    readings: List[SensorReadingCreate]


class SingleWindowPredictionRequest(BaseModel):
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]
    gyro_x: List[float]
    gyro_y: List[float]
    gyro_z: List[float]
    device_id: Optional[str] = "WEARABLE_DEV_01"
    sampling_rate: Optional[float] = 50.0


class PredictionResponse(BaseModel):
    activity: str
    is_fall: bool
    confidence: float
    risk_level: str
    model_available: bool = True
    unavailable_reason: Optional[str] = None
    rf_fall_probability: Optional[float] = None
    svm_fall_probability: Optional[float] = None
    hybrid_fall_probability: Optional[float] = None
    rf_activity: Optional[str] = None
    svm_activity: Optional[str] = None
    ml_activity: str = "UNKNOWN"
    ml_fall_probability: float = 0.0
    safety_override: bool = False
    fusion: Optional[Dict[str, Any]] = None
    probabilities: Dict[str, Any] = {}
    model_name: str
    model_version: str
    timestamp: str
    features_summary: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int


class FallEventResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    device_id: str
    timestamp: datetime
    activity: str = "FALL"
    ml_activity: Optional[str] = None
    confidence: float
    fall_probability: Optional[float] = None
    rf_fall_probability: Optional[float] = None
    svm_fall_probability: Optional[float] = None
    hybrid_fall_probability: Optional[float] = None
    safety_override: bool = False
    risk_level: str
    event_kind: str = "MODEL_PREDICTION"
    status: str
    confirmed: bool = False
    alert_sent: bool = False
    emergency_contact: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notes: Optional[str] = None
    acc_peak: Optional[float] = None
    gyro_peak: Optional[float] = None
    model_version: Optional[str] = "2.0.0"

    model_config = {"from_attributes": True}


class FallAcknowledgeRequest(BaseModel):
    acknowledged_by: str = "Operator"
    notes: Optional[str] = None


class FallNoteRequest(BaseModel):
    notes: str


class AlertResponse(BaseModel):
    id: int
    fall_event_id: Optional[int]
    alert_type: str
    channel: Optional[str] = "sms"
    message: str
    delivery_status: Optional[str] = None
    provider: Optional[str] = None
    created_at: datetime
    acknowledged_at: Optional[datetime]

    model_config = {"from_attributes": True}


class SendHelpRequest(BaseModel):
    device_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    share_location: Optional[bool] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=80)
    confirmed: bool = Field(..., description="Client must set true after an in-app confirmation dialog.")


class DeliveryResult(BaseModel):
    contact_id: int
    contact_name: str
    channel: str
    delivery_status: str
    provider: str
    simulated: bool
    provider_message_id: Optional[str] = None
    detail: Optional[str] = None


class SendHelpResponse(BaseModel):
    event_id: int
    event_kind: str
    status: str
    simulated: bool
    provider: str
    message: str
    deliveries: List[DeliveryResult]
    reused_existing: bool = False


class ActivityHistoryItem(BaseModel):
    id: int
    timestamp: datetime
    activity: str
    confidence: float
    risk_level: str
    is_fall: bool
    device_id: str
    rf_fall_probability: Optional[float] = None
    svm_fall_probability: Optional[float] = None
    hybrid_fall_probability: Optional[float] = None
    model_available: Optional[bool] = True

    model_config = {"from_attributes": True}


class DashboardSummary(BaseModel):
    user: UserResponse
    current_activity: str
    current_confidence: float
    current_risk: str
    hybrid_fall_probability: Optional[float] = None
    rf_fall_probability: Optional[float] = None
    svm_fall_probability: Optional[float] = None
    model_available: bool = True
    model_name: str = "Hybrid Random Forest + SVM"
    system_status: str
    sensor_connected: bool
    total_activities_count: int
    total_falls_count: int
    falls_today_count: int
    unacknowledged_falls_count: int
    last_fall_timestamp: Optional[datetime]
    selected_contact_count: int
    has_emergency_contact: bool
    emergency_provider_mode: str
    location_sharing_enabled: bool
    active_alerts: List[AlertResponse]
    recent_activities: List[ActivityHistoryItem]
    recent_events: List[FallEventResponse]


class SimulationControlRequest(BaseModel):
    action: str = Field(..., description="'start', 'stop', 'pause', or 'resume'")
    playback_speed: Optional[float] = 1.0
    scenario: Optional[str] = "dataset_replay"
    subject: Optional[str] = None
    activity_code: Optional[str] = None
    trial: Optional[str] = None


class SimulationStatusResponse(BaseModel):
    is_running: bool
    is_paused: bool
    current_scenario: str
    playback_speed: float
    samples_replayed: int
    total_samples: int
    current_activity: str
    current_risk: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_available: bool
    unavailable_reason: Optional[str] = None
    trained_at: Optional[str]
    features_count: int
    classes: List[str]
    window_size_seconds: float
    sampling_rate: float
    fusion: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    feature_importance_top10: List[Dict[str, Any]]
    alert_cooldown_seconds: int
    confirmation_timeout_seconds: float


# Backward-compatible alias used by older auth route
class LegacyEmergencyContactUpdate(BaseModel):
    emergency_contact_name: str
    emergency_contact_phone: str
    emergency_contact_email: Optional[str] = None
    phone_number: Optional[str] = None

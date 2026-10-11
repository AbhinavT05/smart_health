"""
FallGuard AI - SQLAlchemy Database Models
Users, devices, emergency contacts, predictions, fall events, and alert deliveries.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="PATIENT", nullable=False)
    phone_number = Column(String(50), nullable=True)
    share_location = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Legacy single-contact columns retained for migration compatibility
    emergency_contact_name = Column(String(100), nullable=True)
    emergency_contact_phone = Column(String(50), nullable=True)
    emergency_contact_email = Column(String(150), nullable=True)

    falls = relationship("FallEvent", back_populates="user")
    contacts = relationship("EmergencyContact", back_populates="user", cascade="all, delete-orphan")
    devices = relationship("Device", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("ActivityPrediction", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (UniqueConstraint("user_id", "device_id", name="uq_user_device"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    device_id = Column(String(120), nullable=False, index=True)
    display_name = Column(String(120), nullable=True)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="devices")


class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    phone_number = Column(String(30), nullable=False)
    relationship_label = Column(String(80), nullable=True)
    channel = Column(String(30), default="sms", nullable=False)
    selected = Column(Boolean, default=True, nullable=False)
    verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="contacts")
    deliveries = relationship("Alert", back_populates="contact")


class SensorReading(Base):
    """
    DEPRECATED TABLE:
    Raw sensor streams stay in the 5-second in-memory ring buffer.
    """
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)
    gyro_x = Column(Float, nullable=False)
    gyro_y = Column(Float, nullable=False)
    gyro_z = Column(Float, nullable=False)
    acc_magnitude = Column(Float, nullable=True)
    session_id = Column(String(100), nullable=True)


class ActivityPrediction(Base):
    __tablename__ = "activity_predictions"
    __table_args__ = (Index("ix_pred_user_ts", "user_id", "timestamp"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    activity = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(50), default="LOW", nullable=False)
    is_fall = Column(Boolean, default=False, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01")
    model_version = Column(String(50), default="2.0.0")
    rf_fall_probability = Column(Float, nullable=True)
    svm_fall_probability = Column(Float, nullable=True)
    hybrid_fall_probability = Column(Float, nullable=True)
    model_available = Column(Boolean, default=True, nullable=False)

    user = relationship("User", back_populates="predictions")


class FallEvent(Base):
    __tablename__ = "fall_events"
    __table_args__ = (Index("ix_fall_user_ts", "user_id", "timestamp"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    activity = Column(String(100), default="FALL", nullable=False)
    ml_activity = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    fall_probability = Column(Float, nullable=True)
    safety_override = Column(Boolean, default=False, nullable=False)
    risk_level = Column(String(50), default="HIGH", nullable=False)

    # MODEL_PREDICTION | CONFIRMED_FALL | MANUAL_HELP
    event_kind = Column(String(40), default="MODEL_PREDICTION", nullable=False)
    # PENDING_CONFIRMATION -> CONFIRMED / CANCELLED_BY_USER -> ACKNOWLEDGED / RESOLVED
    status = Column(String(50), default="UNACKNOWLEDGED", nullable=False)
    confirmed = Column(Boolean, default=False, nullable=False)
    alert_sent = Column(Boolean, default=False, nullable=False)
    emergency_contact = Column(String(200), nullable=True)
    idempotency_key = Column(String(80), nullable=True, index=True)

    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)

    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    acc_peak = Column(Float, nullable=True)
    gyro_peak = Column(Float, nullable=True)
    model_version = Column(String(50), default="2.0.0")
    rf_fall_probability = Column(Float, nullable=True)
    svm_fall_probability = Column(Float, nullable=True)
    hybrid_fall_probability = Column(Float, nullable=True)

    user = relationship("User", back_populates="falls")
    alerts = relationship("Alert", back_populates="fall_event", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    fall_event_id = Column(Integer, ForeignKey("fall_events.id", ondelete="CASCADE"), nullable=True)
    contact_id = Column(Integer, ForeignKey("emergency_contacts.id", ondelete="SET NULL"), nullable=True)
    alert_type = Column(String(100), default="CRITICAL_FALL", nullable=False)
    channel = Column(String(30), default="sms", nullable=False)
    message = Column(String(512), nullable=False)
    # SIMULATED | ACCEPTED | DELIVERED | FAILED | UNCONFIRMED
    delivery_status = Column(String(40), default="UNCONFIRMED", nullable=False)
    provider = Column(String(40), nullable=True)
    provider_message_id = Column(String(80), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime, nullable=True)

    fall_event = relationship("FallEvent", back_populates="alerts")
    user = relationship("User", back_populates="alerts")
    contact = relationship("EmergencyContact", back_populates="deliveries")


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(50), default="INFO")
    event = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)

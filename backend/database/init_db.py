"""
FallGuard AI - Database Initialization & Seed Script
Creates all tables and populates default users, initial alerts, and activity telemetry.
"""

from datetime import datetime, timedelta
import logging
from backend.database.session import engine, SessionLocal
from backend.models.db_models import Base, User, ActivityPrediction, FallEvent, Alert, SystemLog, SensorReading
from backend.core.security import hash_password

from sqlalchemy import inspect, text

logger = logging.getLogger("InitDB")

def migrate_sqlite_columns(bind_engine=None):
    """Ensure newly added columns exist in existing SQLite databases."""
    active_engine = bind_engine or engine
    try:
        insp = inspect(active_engine)
        tables = insp.get_table_names()
        with active_engine.connect() as conn:
            if "users" in tables:
                user_cols = [c["name"] for c in insp.get_columns("users")]
                if "phone_number" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN phone_number VARCHAR(50)"))
                if "share_location" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN share_location BOOLEAN DEFAULT 1"))
                    conn.execute(text("UPDATE users SET share_location = 1 WHERE share_location IS NULL"))
                if "emergency_contact_name" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN emergency_contact_name VARCHAR(100)"))
                if "emergency_contact_phone" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN emergency_contact_phone VARCHAR(50)"))
                if "emergency_contact_email" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN emergency_contact_email VARCHAR(150)"))

            if "fall_events" in tables:
                fall_cols = [c["name"] for c in insp.get_columns("fall_events")]
                if "user_id" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN user_id INTEGER"))
                if "activity" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN activity VARCHAR(100) DEFAULT 'FALL'"))
                if "ml_activity" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN ml_activity VARCHAR(100)"))
                if "fall_probability" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN fall_probability FLOAT"))
                if "safety_override" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN safety_override BOOLEAN DEFAULT 0"))
                if "event_kind" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN event_kind VARCHAR(40) DEFAULT 'MODEL_PREDICTION'"))
                    conn.execute(text("UPDATE fall_events SET event_kind = 'MODEL_PREDICTION' WHERE event_kind IS NULL"))
                if "confirmed" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN confirmed BOOLEAN DEFAULT 0"))
                if "alert_sent" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN alert_sent BOOLEAN DEFAULT 0"))
                if "emergency_contact" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN emergency_contact VARCHAR(200)"))
                if "idempotency_key" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN idempotency_key VARCHAR(80)"))
                if "location_lat" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN location_lat FLOAT"))
                if "location_lon" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN location_lon FLOAT"))
                if "model_version" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN model_version VARCHAR(50) DEFAULT '1.0.0'"))
                if "rf_fall_probability" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN rf_fall_probability FLOAT"))
                if "svm_fall_probability" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN svm_fall_probability FLOAT"))
                if "hybrid_fall_probability" not in fall_cols:
                    conn.execute(text("ALTER TABLE fall_events ADD COLUMN hybrid_fall_probability FLOAT"))

            if "activity_predictions" in tables:
                act_cols = [c["name"] for c in insp.get_columns("activity_predictions")]
                if "user_id" not in act_cols:
                    conn.execute(text("ALTER TABLE activity_predictions ADD COLUMN user_id INTEGER"))
                if "rf_fall_probability" not in act_cols:
                    conn.execute(text("ALTER TABLE activity_predictions ADD COLUMN rf_fall_probability FLOAT"))
                if "svm_fall_probability" not in act_cols:
                    conn.execute(text("ALTER TABLE activity_predictions ADD COLUMN svm_fall_probability FLOAT"))
                if "hybrid_fall_probability" not in act_cols:
                    conn.execute(text("ALTER TABLE activity_predictions ADD COLUMN hybrid_fall_probability FLOAT"))
                if "model_available" not in act_cols:
                    conn.execute(text("ALTER TABLE activity_predictions ADD COLUMN model_available BOOLEAN DEFAULT 1"))

            if "alerts" in tables:
                alert_cols = [c["name"] for c in insp.get_columns("alerts")]
                if "user_id" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN user_id INTEGER"))
                if "contact_id" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN contact_id INTEGER"))
                if "channel" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN channel VARCHAR(30) DEFAULT 'sms'"))
                if "delivery_status" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN delivery_status VARCHAR(40) DEFAULT 'UNCONFIRMED'"))
                if "provider" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN provider VARCHAR(40)"))
                if "provider_message_id" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN provider_message_id VARCHAR(80)"))

            conn.commit()
    except Exception as e:
        logger.warning(f"Note on DB schema migration: {e}")

def init_db(bind_engine=None):
    """Create all database tables and seed default users."""
    active_engine = bind_engine or engine
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=active_engine)
    migrate_sqlite_columns(bind_engine=active_engine)
    
    from sqlalchemy.orm import sessionmaker
    maker = sessionmaker(autocommit=False, autoflush=False, bind=active_engine) if bind_engine else SessionLocal
    db = maker()
    try:
        # Check if users already exist
        admin_user = db.query(User).filter(User.email == "admin@fallguard.ai").first()
        if not admin_user:
            logger.info("Seeding default ADMIN and CAREGIVER accounts...")
            admin = User(
                name="Clinical Administrator",
                email="admin@fallguard.ai",
                hashed_password=hash_password("admin123"),
                role="ADMIN",
                phone_number="+15550100",
                emergency_contact_name="Emergency Response Desk",
                emergency_contact_phone="+15550199",
                emergency_contact_email="emergency@fallguard.ai",
                created_at=datetime.utcnow()
            )
            caregiver = User(
                name="Primary Caregiver",
                email="caregiver@fallguard.ai",
                hashed_password=hash_password("caregiver123"),
                role="CAREGIVER",
                phone_number="+15550101",
                emergency_contact_name="Dr. Smith (Primary Physician)",
                emergency_contact_phone="+15550198",
                emergency_contact_email="caregiver.desk@fallguard.ai",
                created_at=datetime.utcnow()
            )
            db.add_all([admin, caregiver])
            db.commit()
            
        # Seed initial system log if empty
        if db.query(SystemLog).count() == 0:
            log1 = SystemLog(
                timestamp=datetime.utcnow() - timedelta(hours=2),
                level="INFO",
                event="SYSTEM_STARTUP",
                message="FallGuard AI Central Monitoring Server initialized successfully."
            )
            log2 = SystemLog(
                timestamp=datetime.utcnow() - timedelta(hours=1),
                level="INFO",
                event="MODEL_LOADED",
                message="Wearable Sensor Inference Engine loaded with 50Hz sliding window protocol."
            )
            db.add_all([log1, log2])
            db.commit()
            
        # Seed initial fall event for demo if empty
        if db.query(FallEvent).count() == 0:
            past_fall_time = datetime.utcnow() - timedelta(minutes=45)
            fall_event = FallEvent(
                timestamp=past_fall_time,
                confidence=0.96,
                risk_level="HIGH",
                status="UNACKNOWLEDGED",
                acknowledged=False,
                device_id="PATIENT_ROOM_104",
                acc_peak=3.85,
                gyro_peak=4.12,
                notes="Automated alert triggered: Sudden high-impact deceleration detected from wearable sensor."
            )
            db.add(fall_event)
            db.commit()
            db.refresh(fall_event)
            
            alert = Alert(
                fall_event_id=fall_event.id,
                alert_type="CRITICAL_FALL",
                message="CRITICAL: Severe Fall Detected for Patient in Room 104 (Confidence 96%). Immediate assistance required.",
                created_at=past_fall_time
            )
            db.add(alert)
            db.commit()
            
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()

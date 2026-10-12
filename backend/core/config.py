"""
FallGuard AI - Core Application Settings & Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _split_origins(raw: str) -> list:
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "FallGuard AI")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

    CORS_ALLOW_ORIGINS: list = _split_origins(
        os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    )

    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'fall_guard.db'}")

    MODEL_PATH: Path = BASE_DIR / os.getenv("MODEL_PATH", "data/models/fall_detection_model.joblib")
    METRICS_PATH: Path = BASE_DIR / os.getenv("METRICS_PATH", "reports/model_metrics.json")
    SAMPLING_RATE: float = float(os.getenv("SAMPLING_RATE", 50.0))
    WINDOW_SIZE_SECONDS: float = float(os.getenv("WINDOW_SIZE_SECONDS", 2.56))
    WINDOW_SAMPLES: int = int(SAMPLING_RATE * WINDOW_SIZE_SECONDS)

    FALL_CONFIDENCE_HIGH: float = float(os.getenv("FALL_CONFIDENCE_HIGH", 0.85))
    FALL_CONFIDENCE_MEDIUM: float = float(os.getenv("FALL_CONFIDENCE_MEDIUM", 0.60))
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", 45))

    BUFFER_DURATION_SECONDS: float = float(os.getenv("BUFFER_DURATION_SECONDS", 5.0))
    PHONE_SAMPLING_RATE: float = float(os.getenv("PHONE_SAMPLING_RATE", 50.0))
    MAX_BUFFER_SAMPLES: int = int(BUFFER_DURATION_SECONDS * PHONE_SAMPLING_RATE)
    INFERENCE_STRIDE_SAMPLES: int = int(os.getenv("INFERENCE_STRIDE_SAMPLES", 25))

    FALL_CONFIRMATION_TIMEOUT_SECONDS: float = float(os.getenv("FALL_CONFIRMATION_TIMEOUT_SECONDS", 15.0))

    # mock = labeled simulation; twilio = real SMS when credentials are present
    EMERGENCY_PROVIDER: str = os.getenv("EMERGENCY_PROVIDER", "mock").lower()
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")

    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")

    STATIC_DIR: Path = BASE_DIR / "frontend"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"
    DATA_DIR: Path = BASE_DIR / "data"

    def uses_default_secret(self) -> bool:
        return not self.SECRET_KEY or self.SECRET_KEY in (
            "fallguard-super-secret-key-change-in-production-2026",
            "dev-only-insecure-secret-set-SECRET_KEY-in-dotenv",
            "secret",
            "changeme",
        )

    @property
    def resolved_secret(self) -> str:
        if self.ENVIRONMENT == "production":
            if self.uses_default_secret() or len(self.SECRET_KEY) < 32:
                raise RuntimeError(
                    "Production startup rejected: Insecure or missing SECRET_KEY. "
                    "A strong SECRET_KEY (at least 32 characters) must be configured in production."
                )
            return self.SECRET_KEY
        if self.SECRET_KEY:
            return self.SECRET_KEY
        return "dev-only-insecure-secret-set-SECRET_KEY-in-dotenv"

    def emergency_mode(self) -> str:
        if self.EMERGENCY_PROVIDER == "twilio" and self.TWILIO_ACCOUNT_SID and self.TWILIO_AUTH_TOKEN and self.TWILIO_FROM_NUMBER:
            return "twilio"
        return "simulation"


settings = Settings()

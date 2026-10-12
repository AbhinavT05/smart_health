"""
Pytest configuration and test database isolation fixture.
Ensures automated tests run against an isolated SQLite test database,
preserving the existing fall_guard.db database untouched.
"""

import os
import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DB_PATH = BASE_DIR / "test_fall_guard.db"

# Set DATABASE_URL to isolated test database before importing backend modules
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

from backend.database.init_db import init_db
from backend.database.session import engine

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initialize clean isolated test database before test session and clean up after."""
    init_db()
    yield
    engine.dispose()
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass

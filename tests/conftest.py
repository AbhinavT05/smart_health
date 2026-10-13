"""
Pytest configuration and test database isolation fixture.
Ensures automated tests run against an isolated SQLite test database,
preserving the existing fall_guard.db database untouched.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="fallguard-tests-", dir=BASE_DIR / "tests"))
TEST_DB_PATH = TEST_DB_DIR / "test_fall_guard.db"

# Set DATABASE_URL to isolated test database before importing backend modules
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

from backend.database.init_db import init_db
from backend.database.session import engine

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initialize clean isolated test database before test session and clean up after."""
    try:
        init_db()
    except Exception:
        engine.dispose()
        shutil.rmtree(TEST_DB_DIR, ignore_errors=True)
        raise
    yield
    engine.dispose()
    # This fixture only removes its own unique temporary directory. Existing
    # project databases, including a developer's test_fall_guard.db, are kept.
    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)

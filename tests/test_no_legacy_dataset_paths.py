"""Guard against restoring legacy mixed/synthetic dataset readers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("unified_sensor_stream.csv", "generate_validated_fall_dataset")
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "data", "reports"}


def test_non_test_sources_do_not_reference_removed_dataset_paths():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if "test" in path.name.lower() or path.suffix.lower() in {".joblib", ".png", ".csv", ".json", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{path.relative_to(ROOT)}: {needle}")
    assert not hits, "Legacy dataset path references found:\n" + "\n".join(hits)

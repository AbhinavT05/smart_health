"""Fail-closed MobiFall data availability check; this project does not download datasets."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MOBIFALL_DIR = BASE_DIR / "data" / "raw" / "MobiFall"


def build_dataset_pipeline():
    if not MOBIFALL_DIR.is_dir() or not any(MOBIFALL_DIR.rglob("*_acc_*.txt")):
        raise FileNotFoundError(f"Real MobiFall v2.0 trial files are required under {MOBIFALL_DIR}")
    print(f"Verified real MobiFall input: {MOBIFALL_DIR}")
    return MOBIFALL_DIR

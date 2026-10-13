"""Compatibility entry point backed exclusively by the real MobiFall pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
MOBIFALL_DIR = BASE_DIR / "data" / "raw" / "MobiFall"


def train_and_evaluate_all_models(dataset_csv_path=None, sampling_rate: float = 50.0) -> dict[str, Any]:
    """Run grouped MobiFall evaluation and fit the backend bundle; no CSV fallback."""
    if not MOBIFALL_DIR.is_dir():
        raise FileNotFoundError(f"Real MobiFall v2.0 data is required at {MOBIFALL_DIR}; no CSV fallback is supported.")
    if dataset_csv_path is not None:
        raise ValueError("dataset_csv_path is unsupported; training reads MobiFall through the native loader.")
    from scripts.run_mobifall_experiments import main as run_experiment
    run_experiment()
    from scripts.save_mobifall_bundle import main as save_bundle
    save_bundle()
    import joblib
    return joblib.load(BASE_DIR / "data" / "models" / "fall_detection_model.joblib")

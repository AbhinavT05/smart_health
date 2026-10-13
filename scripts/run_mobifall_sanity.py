"""Recompute MobiFall sanity checks from the reproducible feature cache."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.run_mobifall_experiments import _feature_sets, _sanity_checks
from ml.training.mobifall_dataset import _feature_hash


def main() -> None:
    cache = BASE_DIR / "data" / "processed" / f"mobifall_features_{_feature_hash()}.csv"
    if not cache.exists():
        raise FileNotFoundError(f"MobiFall feature cache not found: {cache}; run scripts/build_mobifall_manifest.py first")
    features = __import__("pandas").read_csv(cache)
    checks = _sanity_checks(features, _feature_sets(features)["all features"])
    reports = BASE_DIR / "reports"
    manifest = json.loads((reports / "dataset_manifest.json").read_text(encoding="utf-8"))
    checks["cohort_composition"] = manifest["cohort_relationship"]
    path = reports / "sanity.json"
    path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

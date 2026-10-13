"""Build the real MobiFall window manifest and compare observed cohorts."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.training.mobifall_dataset import build_mobifall_windows, write_manifest


EXPECTED_FALL = {f"MOB_{i:02d}" for i in (*range(1, 22), 29, 30, 31)}
EXPECTED_ADL = {f"MOB_{i:02d}" for i in (2, 3, 4, 5, 7, 8, 9, 10, 11)}


def main() -> None:
    features, cap = build_mobifall_windows(verbose=True)
    path = write_manifest(features, cap)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fall_subjects = set(manifest["classes"]["fall"]["subject_ids"])
    adl_subjects = set(manifest["classes"]["non_fall"]["subject_ids"])
    print(f"Manifest written: {path}")
    print(json.dumps(manifest, indent=2))
    differences = []
    if fall_subjects != EXPECTED_FALL:
        differences.append(f"fall subjects missing={sorted(EXPECTED_FALL - fall_subjects)} extra={sorted(fall_subjects - EXPECTED_FALL)}")
    if adl_subjects != EXPECTED_ADL:
        differences.append(f"ADL subjects missing={sorted(EXPECTED_ADL - adl_subjects)} extra={sorted(adl_subjects - EXPECTED_ADL)}")
    if manifest["classes"]["fall"]["trials"] != 288:
        differences.append(f"fall trials observed={manifest['classes']['fall']['trials']} expected=288")
    if manifest["classes"]["non_fall"]["trials"] != 342:
        differences.append(f"ADL trials observed={manifest['classes']['non_fall']['trials']} expected=342")
    print("Cohort check: " + ("PASS" if not differences else "DIFFERENCES FOUND"))
    for difference in differences:
        print(f"- {difference}")


if __name__ == "__main__":
    main()

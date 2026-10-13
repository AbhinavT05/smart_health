"""Inspect real MobiFall trial spans and choose a reproducible window cap."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "MobiFall"
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.loaders.mobifall import _parse_trial_files, _read_sensor_file


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(f"MobiFall dataset directory not found: {ROOT}")
    groups = _parse_trial_files(ROOT)
    durations = defaultdict(list)
    missing = []
    for key, files in sorted(groups.items()):
        if set(files) != {"acc", "gyro"}:
            missing.append(key)
            continue
        acc_t, _ = _read_sensor_file(files["acc"], "acc")
        gyro_t, _ = _read_sensor_file(files["gyro"], "gyro")
        overlap_ns = min(acc_t[-1], gyro_t[-1]) - max(acc_t[0], gyro_t[0])
        if overlap_ns <= 0:
            raise ValueError(f"No overlapping acc/gyro span for trial {key}")
        durations[key[2]].append(overlap_ns / 1e9)
    print(f"paired_trials={sum(map(len, durations.values()))}; incomplete_trials={len(missing)}")
    for code, values in sorted(durations.items()):
        print(
            f"{code}: n={len(values)} duration_s min={min(values):.3f} "
            f"median={np.median(values):.3f} max={max(values):.3f}"
        )
    all_durations = np.asarray([v for values in durations.values() for v in values])
    median_samples = int(np.floor(np.median(all_durations) * 50)) + 1
    median_windows = max(0, (median_samples - 128) // 64 + 1)
    print(f"global_median_trial_duration_s={np.median(all_durations):.3f}")
    print(f"global_median_trial_window_count={median_windows}")
    print(f"chosen_per_trial_window_cap={median_windows}")
    if missing:
        print(f"incomplete_trial_keys={missing}")


if __name__ == "__main__":
    main()

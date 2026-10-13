"""MobiFall v2.0 mirror loader.

Expected layout: ``sub<N>/(ADL|FALLS)/<CODE>/<CODE>_(acc|gyro)_<N>_<trial>.txt``.
Acceleration and gyroscope are independently low-pass filtered and interpolated
to a common 50 Hz grid over their overlapping timestamp span.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)
G0 = 9.80665
TARGET_HZ = 50.0
ADL_CODES = {"STD", "WAL", "JOG", "JUM", "STU", "STN", "SCH", "CSI", "CSO"}
FALL_CODES = {"FOL", "FKL", "SDL", "BSC"}
FILE_RE = re.compile(r"^(?P<code>[A-Z]+)_(?P<sensor>acc|gyro)_(?P<subject>\d+)_(?P<trial>\d+)\.txt$", re.I)
SENSOR_COLUMNS = {
    "acc": ("acc_x", "acc_y", "acc_z"),
    "gyro": ("gyro_x", "gyro_y", "gyro_z"),
}
OUTPUT_COLUMNS = [
    "dataset", "subject_id", "trial_id", "activity_code", "is_fall", "timestamp_s",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
]


def _read_sensor_file(path: Path, sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """Read numeric CSV records following the ``@DATA`` marker."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if line.strip().upper() == "@DATA")
    except StopIteration as exc:
        raise ValueError(f"Missing @DATA marker in {path}") from exc
    rows = []
    for line_number, line in enumerate(lines[marker + 1 :], start=marker + 2):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Expected timestamp and three axes at {path}:{line_number}; got {len(fields)} fields")
        try:
            rows.append([int(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric sensor record at {path}:{line_number}") from exc
    if len(rows) < 2:
        raise ValueError(f"Fewer than two samples after @DATA in {path}")
    data = np.asarray(rows, dtype=np.float64)
    # Average rows with identical clocks before interpolation. One actual mirror
    # trial contains a repeated gyro timestamp; reversed clocks remain invalid.
    if np.any(np.diff(data[:, 0]) < 0):
        raise ValueError(f"Timestamps must be nondecreasing in {path}")
    if np.any(np.diff(data[:, 0]) == 0):
        unique_t, inverse = np.unique(data[:, 0], return_inverse=True)
        collapsed = np.zeros((len(unique_t), 4), dtype=np.float64)
        collapsed[:, 0] = unique_t
        counts = np.bincount(inverse)
        for column in range(1, 4):
            collapsed[:, column] = np.bincount(inverse, weights=data[:, column]) / counts
        logger.warning("Collapsed %d repeated timestamps in %s by averaging sensor values", len(data) - len(unique_t), path)
        data = collapsed
    values = data[:, 1:] / G0 if sensor.lower() == "acc" else data[:, 1:]
    return data[:, 0], values


def _lowpass(timestamps_ns: np.ndarray, values: np.ndarray, target_hz: float) -> np.ndarray:
    """Apply a zero-phase anti-aliasing low-pass filter using measured rate."""
    sample_hz = 1e9 / float(np.median(np.diff(timestamps_ns)))
    cutoff_hz = min(20.0, 0.4 * sample_hz, 0.9 * target_hz / 2.0)
    if sample_hz <= 2.0 * cutoff_hz or len(values) < 16:
        logger.warning("Skipping low-pass filter: insufficient samples or sampling rate %.3f Hz", sample_hz)
        return values
    sos = butter(4, cutoff_hz, btype="lowpass", fs=sample_hz, output="sos")
    try:
        return sosfiltfilt(sos, values, axis=0)
    except ValueError as exc:
        logger.warning("Skipping low-pass filter for short trial (%s)", exc)
        return values


def _parse_trial_files(root: Path) -> dict[tuple[str, str, str, str], dict[str, Path]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Path]] = {}
    for path in root.rglob("*.txt"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        sensor = match.group("sensor").lower()
        code = match.group("code").upper()
        subject_number = int(match.group("subject"))
        trial = match.group("trial")
        subject_dir = path.parent.parent.parent.name
        category = path.parent.parent.name.upper()
        if not re.fullmatch(r"sub\d+", subject_dir, re.I):
            raise ValueError(f"Unexpected subject directory for MobiFall file: {path}")
        if category not in {"ADL", "FALLS"}:
            raise ValueError(f"Unexpected category directory for MobiFall file: {path}")
        if subject_number != int(subject_dir[3:]):
            raise ValueError(f"Subject in filename does not match directory: {path}")
        key = (subject_dir.upper(), category, code, trial)
        if sensor in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate {sensor} file for trial {key}: {path}")
        grouped[key][sensor] = path
    return grouped


def load_mobifall(root: str | Path | None = None, *, target_hz: float = TARGET_HZ) -> pd.DataFrame:
    """Load real MobiFall trials as a unified, aligned sensor DataFrame."""
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "raw" / "MobiFall"
    if not root.is_dir():
        raise FileNotFoundError(f"MobiFall dataset directory not found: {root}")
    grouped = _parse_trial_files(root)
    if not grouped:
        raise FileNotFoundError(f"No MobiFall acc/gyro files matching the documented layout under {root}")

    records: list[pd.DataFrame] = []
    half_period_ns = 0.5e9 / target_hz
    for (subject_dir, category, code, trial), files in sorted(grouped.items()):
        if set(files) != {"acc", "gyro"}:
            raise ValueError(f"Trial {subject_dir}/{category}/{code}/{trial} must have acc and gyro files; found {sorted(files)}")
        if category == "ADL" and code not in ADL_CODES:
            raise ValueError(f"Unrecognized MobiFall ADL activity code {code!r}")
        if category == "FALLS" and code not in FALL_CODES:
            raise ValueError(f"Unrecognized MobiFall fall code {code!r}")

        acc_t, acc = _read_sensor_file(files["acc"], "acc")
        gyro_t, gyro = _read_sensor_file(files["gyro"], "gyro")
        overlap_start = max(int(acc_t[0]), int(gyro_t[0]))
        overlap_end = min(int(acc_t[-1]), int(gyro_t[-1]))
        overlap_ns = overlap_end - overlap_start
        shorter_span_ns = min(int(acc_t[-1] - acc_t[0]), int(gyro_t[-1] - gyro_t[0]))
        if overlap_ns <= 0:
            raise ValueError(f"Acceleration and gyro have no overlapping span for {subject_dir}/{category}/{code}/{trial}")
        if overlap_ns < 0.5 * shorter_span_ns:
            logger.warning(
                "Barely overlapping sensor clocks (%s/%s/%s/%s): overlap %.3fs of shorter %.3fs",
                subject_dir, category, code, trial, overlap_ns / 1e9, shorter_span_ns / 1e9,
            )
        # Keep grid points at least half a period inside both clocks, avoiding extrapolation.
        grid_start = int(np.ceil((overlap_start + half_period_ns) / (1e9 / target_hz)))
        grid_end = int(np.floor((overlap_end - half_period_ns) / (1e9 / target_hz)))
        grid_ns = np.arange(grid_start, grid_end + 1, dtype=np.int64) * int(1e9 / target_hz)
        if len(grid_ns) < 128:
            logger.warning("Trial shorter than a 128-sample window: %s/%s/%s/%s", subject_dir, category, code, trial)
        if not len(grid_ns):
            raise ValueError(f"No 50 Hz grid points in overlapping span for {subject_dir}/{category}/{code}/{trial}")

        acc_f = _lowpass(acc_t, acc, target_hz)
        gyro_f = _lowpass(gyro_t, gyro, target_hz)
        # Shift epoch-sized nanosecond timestamps before float conversion for precision.
        acc_seconds = (acc_t - overlap_start).astype(np.float64) / 1e9
        gyro_seconds = (gyro_t - overlap_start).astype(np.float64) / 1e9
        grid_seconds = (grid_ns - overlap_start).astype(np.float64) / 1e9
        aligned_acc = np.column_stack([np.interp(grid_seconds, acc_seconds, acc_f[:, axis]) for axis in range(3)])
        aligned_gyro = np.column_stack([np.interp(grid_seconds, gyro_seconds, gyro_f[:, axis]) for axis in range(3)])
        trial_id = f"MOB_{subject_dir[3:]}_{category}_{code}_{trial}"
        frame = pd.DataFrame(np.column_stack([aligned_acc, aligned_gyro]), columns=SENSOR_COLUMNS["acc"] + SENSOR_COLUMNS["gyro"])
        frame.insert(0, "timestamp_s", np.arange(len(grid_ns), dtype=np.float64) / target_hz)
        frame.insert(0, "is_fall", int(category == "FALLS"))
        frame.insert(0, "activity_code", code)
        frame.insert(0, "trial_id", trial_id)
        frame.insert(0, "subject_id", f"MOB_{int(subject_dir[3:]):02d}")
        frame.insert(0, "dataset", "MobiFall")
        records.append(frame[OUTPUT_COLUMNS])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=OUTPUT_COLUMNS)


def main() -> None:
    """Print one actual source-file head and a loaded stream summary."""
    root = Path(__file__).resolve().parents[1] / "raw" / "MobiFall"
    if not root.is_dir():
        raise FileNotFoundError(f"MobiFall dataset directory not found: {root}")
    files = sorted(path for path in root.rglob("*.txt") if FILE_RE.match(path.name))
    if not files:
        raise FileNotFoundError(f"No MobiFall sensor files found under {root}")
    sample = files[0]
    lines = sample.read_text(encoding="utf-8-sig").splitlines()
    print(f"Sample file: {sample}")
    print("\n".join(lines[: min(len(lines), 12)]))
    data = load_mobifall(root)
    print(f"Loaded {len(data)} samples from {data.trial_id.nunique()} trials and {data.subject_id.nunique()} subjects")


if __name__ == "__main__":
    main()

"""Replay selected real MobiFall v2.0 trials at the loader's 50 Hz grid."""
from __future__ import annotations

from typing import Any, Iterator

from data.loaders.mobifall import load_mobifall


def generate_sensor_stream_generator(
    scenario: str = "dataset_replay", sampling_rate: float = 50.0,
    subject: str | None = None, activity_code: str | None = None, trial: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield actual MobiFall readings; `simulated` marks replay transport provenance."""
    if scenario != "dataset_replay":
        raise ValueError("Only scenario='dataset_replay' is supported; generated waveforms were removed.")
    if sampling_rate != 50.0:
        raise ValueError("MobiFall replay is fixed to the loader's 50 Hz resampling grid.")
    frame = load_mobifall()
    if subject:
        frame = frame[frame.subject_id == subject]
    if activity_code:
        frame = frame[frame.activity_code.str.upper() == activity_code.upper()]
    if trial:
        frame = frame[frame.trial_id.str.endswith("_" + str(trial))]
    if frame.empty:
        raise ValueError(f"No real MobiFall trial matches subject={subject!r}, activity_code={activity_code!r}, trial={trial!r}")
    for _, row in frame.iterrows():
        yield {
            "acc_x": float(row.acc_x), "acc_y": float(row.acc_y), "acc_z": float(row.acc_z),
            "gyro_x": float(row.gyro_x), "gyro_y": float(row.gyro_y), "gyro_z": float(row.gyro_z),
            "timestamp_s": float(row.timestamp_s), "subject_id": row.subject_id,
            "trial_id": row.trial_id, "ground_truth": "FALL" if row.is_fall else row.activity_code,
            "is_fall": int(row.is_fall), "simulated": True,
        }

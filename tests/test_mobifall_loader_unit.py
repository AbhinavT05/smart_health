"""Unit tests for MobiFall parsing using explicitly synthetic fixture files."""

import numpy as np

from data.loaders.mobifall import load_mobifall


def _write_sensor(path, timestamps, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["MobiFall unit-test fixture", "@DATA"]
    rows.extend(f"{int(t)}, {x}, {y}, {z}" for t, (x, y, z) in zip(timestamps, values))
    path.write_bytes(("\r\n".join(rows) + "\r\n").encode("utf-8"))


def test_synthetic_mobifall_unit_fixture_aligns_and_converts(tmp_path):
    """Synthetic unit fixture only: independent clocks, units, and labels."""
    base = tmp_path / "sub2" / "ADL" / "STD"
    t_acc = np.arange(0, 4_000_000_000, 10_000_000, dtype=np.int64)
    t_gyro = np.arange(5_000_000, 4_000_000_000, 5_000_000, dtype=np.int64)
    acc_values = np.tile([0.0, 0.0, 9.80665], (len(t_acc), 1))
    gyro_values = np.tile([1.0, 2.0, 3.0], (len(t_gyro), 1))
    _write_sensor(base / "STD_acc_2_1.txt", t_acc, acc_values)
    _write_sensor(base / "STD_gyro_2_1.txt", t_gyro, gyro_values)

    stream = load_mobifall(tmp_path)
    assert len(stream) >= 128
    assert set(stream.dataset) == {"MobiFall"}
    assert set(stream.subject_id) == {"MOB_02"}
    assert set(stream.is_fall) == {0}
    assert np.allclose(stream[["acc_x", "acc_y", "acc_z"]].mean().to_numpy(), [0, 0, 1], atol=1e-5)
    assert np.allclose(stream[["gyro_x", "gyro_y", "gyro_z"]].mean().to_numpy(), [1, 2, 3], atol=1e-5)
    assert np.allclose(np.diff(stream.timestamp_s), 0.02)


def test_synthetic_mobifall_unit_fixture_labels_fall(tmp_path):
    """Synthetic unit fixture only: verify category and code map to fall label."""
    base = tmp_path / "sub7" / "FALLS" / "FOL"
    timestamps = np.arange(0, 3_000_000_000, 10_000_000, dtype=np.int64)
    values = np.tile([0.0, 0.0, 9.80665], (len(timestamps), 1))
    gyros = np.zeros_like(values)
    _write_sensor(base / "FOL_acc_7_1.txt", timestamps, values)
    _write_sensor(base / "FOL_gyro_7_1.txt", timestamps, gyros)

    stream = load_mobifall(tmp_path)
    assert set(stream.is_fall) == {1}
    assert set(stream.activity_code) == {"FOL"}

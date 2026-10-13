"""Real MobiFall replay tests; samples are loaded from the supplied raw files."""
from simulation.realtime_simulator import generate_sensor_stream_generator


def test_real_mobifall_replay_marks_provenance():
    sample = next(generate_sensor_stream_generator(subject="MOB_02", activity_code="WAL", trial="1"))
    assert sample["simulated"] is True
    assert sample["subject_id"] == "MOB_02"
    assert sample["ground_truth"] == "WAL"

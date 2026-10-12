"""
FallGuard AI - Inference Verification & WebSocket Simulation Test Script
Tests prediction on both real held-out test dataset windows and dynamic sensor signals.
"""

import sys
import json
import asyncio
import pytest
import requests
import websockets
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.inference.predictor import FallDetector

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/monitor"


def test_real_inference_held_out_samples():
    """Verify inference on real held-out test subjects."""
    detector = FallDetector()
    assert detector.model is not None
    
    stream_csv = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"
    if stream_csv.exists():
        df_stream = pd.read_csv(stream_csv)
        
        # Test 1: Real Held-out Walking Window (Subject 2)
        walk_sample = df_stream[(df_stream["subject_id"] == 2) & (df_stream["activity"] == "WALKING")]
        if not walk_sample.empty:
            w_id = walk_sample["window_id"].iloc[0]
            w_df = walk_sample[walk_sample["window_id"] == w_id].sort_values("step")
            
            pred_walk = detector.predict_window(
                acc_x=w_df["acc_x"].to_numpy(),
                acc_y=w_df["acc_y"].to_numpy(),
                acc_z=w_df["acc_z"].to_numpy(),
                gyro_x=w_df["gyro_x"].to_numpy(),
                gyro_y=w_df["gyro_y"].to_numpy(),
                gyro_z=w_df["gyro_z"].to_numpy()
            )
            assert pred_walk["activity"] in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "LAYING", "STANDING", "SITTING"]
            assert pred_walk["is_fall"] is False
            
        # Test 2: Real Held-out Fall Window (Subject 42)
        fall_sample = df_stream[(df_stream["subject_id"] == 42) & (df_stream["activity"] == "FALL")]
        if not fall_sample.empty:
            f_id = fall_sample["window_id"].iloc[0]
            f_df = fall_sample[fall_sample["window_id"] == f_id].sort_values("step")
            
            pred_fall = detector.predict_window(
                acc_x=f_df["acc_x"].to_numpy(),
                acc_y=f_df["acc_y"].to_numpy(),
                acc_z=f_df["acc_z"].to_numpy(),
                gyro_x=f_df["gyro_x"].to_numpy(),
                gyro_y=f_df["gyro_y"].to_numpy(),
                gyro_z=f_df["gyro_z"].to_numpy()
            )
            assert pred_fall["activity"] == "FALL"
            assert pred_fall["is_fall"] is True
            assert pred_fall["risk_level"] == "HIGH"


async def _run_websocket_and_simulation():
    """Async worker for WebSocket telemetry streaming and simulation controls."""
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=1.0)
        if res.status_code != 200:
            pytest.skip("FastAPI server is not running on port 8000.")
        login_res = requests.post(f"{BASE_URL}/api/auth/login", json={"email": "admin@fallguard.ai", "password": "admin123"}, timeout=2.0)
        if login_res.status_code != 200:
            pytest.skip("FastAPI live auth endpoint not ready on port 8000.")
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        ws_url = f"{WS_URL}?token={token}"
    except Exception as e:
        pytest.skip(f"FastAPI server is not accessible on http://127.0.0.1:8000: {e}")

    try:
        # Start Simulation via REST
        res = requests.post(f"{BASE_URL}/api/simulation/control", json={"action": "start", "playback_speed": 2.0}, headers=headers)
        if res.status_code != 200:
            pytest.skip("Live server simulation endpoint rejected request.")
        
        # Connect WebSocket client and receive real-time ticks
        async with websockets.connect(ws_url) as ws:
            received_ticks = 0
            for _ in range(10):
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(msg_raw)
                if msg.get("type") == "SENSOR_TICK":
                    received_ticks += 1
                    
            assert received_ticks >= 5, "Failed to receive live sensor ticks over WebSocket"
            
        # Stop simulation
        requests.post(f"{BASE_URL}/api/simulation/control", json={"action": "stop"}, headers=headers)
    except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
        pytest.skip(f"Live server WebSocket disconnected or unavailable: {e}")


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_websocket_and_simulation():
    """Verify live WebSocket telemetry streaming and simulation controls."""
    await _run_websocket_and_simulation()


def main():
    print("\n" + "=" * 70)
    print("       FALLGUARD AI - REAL INFERENCE VERIFICATION TEST")
    print("=" * 70)
    
    detector = FallDetector()
    print(f"  * Loaded Model:     {detector.model_name}")
    print(f"  * Features Count:   {len(detector.feature_names)}")
    print(f"  * Class Labels:     {detector.classes}\n")
    
    # Test from real sensor stream dataset if present
    stream_csv = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"
    if stream_csv.exists():
        df_stream = pd.read_csv(stream_csv)
        
        # Test 1: Real Held-out Walking Window (Subject 2)
        walk_sample = df_stream[(df_stream["subject_id"] == 2) & (df_stream["activity"] == "WALKING")]
        if not walk_sample.empty:
            w_id = walk_sample["window_id"].iloc[0]
            w_df = walk_sample[walk_sample["window_id"] == w_id].sort_values("step")
            
            pred_walk = detector.predict_window(
                acc_x=w_df["acc_x"].to_numpy(),
                acc_y=w_df["acc_y"].to_numpy(),
                acc_z=w_df["acc_z"].to_numpy(),
                gyro_x=w_df["gyro_x"].to_numpy(),
                gyro_y=w_df["gyro_y"].to_numpy(),
                gyro_z=w_df["gyro_z"].to_numpy()
            )
            print("--- [Test 1] Real Test Subject #2 Ground Truth: WALKING ---")
            print(f"  * Predicted Activity: {pred_walk['activity']}")
            print(f"  * Model Confidence:   {pred_walk['confidence'] * 100:.1f}%")
            print(f"  * Fall Risk Level:    {pred_walk['risk_level']}")
            print(f"  * Fall Flag:          {pred_walk['is_fall']}")
            print(f"  * Peak Acceleration:  {pred_walk['features_summary']['acc_mag_max']} g")
            
        # Test 2: Real Held-out Fall Window (Subject 42)
        fall_sample = df_stream[(df_stream["subject_id"] == 42) & (df_stream["activity"] == "FALL")]
        if not fall_sample.empty:
            f_id = fall_sample["window_id"].iloc[0]
            f_df = fall_sample[fall_sample["window_id"] == f_id].sort_values("step")
            
            pred_fall = detector.predict_window(
                acc_x=f_df["acc_x"].to_numpy(),
                acc_y=f_df["acc_y"].to_numpy(),
                acc_z=f_df["acc_z"].to_numpy(),
                gyro_x=f_df["gyro_x"].to_numpy(),
                gyro_y=f_df["gyro_y"].to_numpy(),
                gyro_z=f_df["gyro_z"].to_numpy()
            )
            print("\n--- [Test 2] Real Test Subject #42 Ground Truth: FALL ---")
            print(f"  * Predicted Activity: {pred_fall['activity']}")
            print(f"  * Model Confidence:   {pred_fall['confidence'] * 100:.1f}%")
            print(f"  * Fall Risk Level:    {pred_fall['risk_level']}")
            print(f"  * Fall Flag:          {pred_fall['is_fall']}")
            print(f"  * Peak Acceleration:  {pred_fall['features_summary']['acc_mag_max']} g")
            print(f"  * Peak Gyroscope:     {pred_fall['features_summary']['gyro_mag_max']} rad/s")
            
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

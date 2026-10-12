"""
FallGuard AI - WebSocket Telemetry & Smartphone Sensor Streaming Routes
Provides:
1. /ws/monitor: Real-time telemetry broadcast for dashboard UI.
2. /ws/sensor: Authenticated, persistent 50Hz sensor stream ingestion from user smartphones.
"""

import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status

from backend.core.security import decode_access_token
from backend.database.session import SessionLocal
from backend.models.db_models import User
from backend.services.websocket_service import ws_manager
from backend.services.sensor_stream_manager import sensor_stream_manager
from backend.services.prediction_service import prediction_service
from backend.services.fall_confirmation_service import fall_confirmation_service
from backend.core.config import settings

router = APIRouter(tags=["WebSockets"])
logger = logging.getLogger("WebSocketRoute")

@router.websocket("/ws/monitor")
async def websocket_monitor_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Real-time streaming telemetry and instant fall alert push endpoint for dashboard UI.
    Requires validated JWT token via query parameter or immediate auth handshake frame.
    """
    await websocket.accept()
    jwt_token = token
    if not jwt_token:
        # Check for initial auth handshake frame within 4 seconds
        try:
            auth_text = await asyncio.wait_for(websocket.receive_text(), timeout=4.0)
            auth_packet = json.loads(auth_text)
            if auth_packet.get("type") == "auth":
                jwt_token = auth_packet.get("token")
        except Exception:
            pass

    if not jwt_token:
        await websocket.send_text(json.dumps({
            "type": "error",
            "code": "AUTH_REQUIRED",
            "message": "Authentication token missing. Provide token query parameter or auth handshake."
        }))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    payload = decode_access_token(jwt_token)
    if not payload or not payload.get("sub"):
        await websocket.send_text(json.dumps({
            "type": "error",
            "code": "AUTH_FAILED",
            "message": "Invalid or expired JWT token."
        }))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_id = int(payload["sub"])
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "USER_NOT_FOUND",
                "message": f"User ID {user_id} not found."
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        user_name = user.name
    finally:
        db.close()

    await ws_manager.connect(websocket, user_id=user_id)
    await websocket.send_text(json.dumps({
        "type": "auth_success",
        "user_id": user_id,
        "user_name": user_name
    }))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"Dashboard WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)


@router.websocket("/ws/sensor")
async def websocket_smartphone_sensor_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    device_id: Optional[str] = Query("phone_001")
):
    """
    Persistent, authenticated live 50Hz sensor stream connection for user smartphones.
    Maintains a bounded 5-second in-memory ring buffer and executes sliding 128-sample inference.
    """
    await websocket.accept()
    db = SessionLocal()
    user: Optional[User] = None
    dev_id = device_id or "phone_001"

    try:
        # Step 1: Authentication Verification
        jwt_token = token
        if not jwt_token:
            # Check for initial auth handshake frame within 5 seconds
            try:
                auth_text = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                auth_packet = json.loads(auth_text)
                if auth_packet.get("type") == "auth":
                    jwt_token = auth_packet.get("token")
                    dev_id = auth_packet.get("device_id", dev_id)
            except Exception as e:
                logger.warning(f"Handshake auth timeout/error for device {dev_id}: {e}")

        if not jwt_token:
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "AUTH_REQUIRED",
                "message": "Authentication token missing. Provide token query parameter or auth handshake."
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        payload = decode_access_token(jwt_token)
        if not payload or not payload.get("sub"):
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "AUTH_FAILED",
                "message": "Invalid or expired JWT token."
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        user_id = int(payload["sub"])
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "USER_NOT_FOUND",
                "message": f"User ID {user_id} not found."
            }))
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Step 2: Register Device & Initialize Isolated Bounded Ring Buffer
        ws_manager.register_device(dev_id, websocket, user_id=user.id)
        buffer = sensor_stream_manager.get_or_create_buffer(device_id=dev_id, user_id=user.id, source_type="REAL_PHONE")
        buffer.connected = True

        logger.info(f"phone connected: device_id='{dev_id}' user='{user.name}' (id={user.id})")
        logger.info(f"sensor stream started: bounded buffer initialized (max {settings.MAX_BUFFER_SAMPLES} samples)")

        # Send authentication & initialization acknowledgement
        await websocket.send_text(json.dumps({
            "type": "auth_success",
            "device_id": dev_id,
            "user_id": user.id,
            "user_name": user.name,
            "expected_sampling_rate": settings.PHONE_SAMPLING_RATE,
            "window_samples": settings.WINDOW_SAMPLES,
            "max_buffer_samples": settings.MAX_BUFFER_SAMPLES,
            "buffer_duration_seconds": settings.BUFFER_DURATION_SECONDS,
            "confirmation_timeout_seconds": settings.FALL_CONFIRMATION_TIMEOUT_SECONDS
        }))

        # Notify user dashboard of device connection
        await ws_manager.broadcast_to_user(user.id, {
            "type": "DEVICE_STATUS_UPDATE",
            "data": buffer.get_status()
        })

        # Step 3: Real-Time Ingestion Loop
        while True:
            data_text = await websocket.receive_text()
            try:
                packet = json.loads(data_text)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Malformed JSON packet."}))
                continue

            msg_type = packet.get("type", "sensor_data")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                continue

            elif msg_type == "fall_confirmation":
                # User responded to pending confirmation on their phone
                event_id = packet.get("event_id")
                action = packet.get("action")
                location = packet.get("location")
                if event_id and action:
                    await fall_confirmation_service.handle_user_response(
                        db=db,
                        fall_event_id=int(event_id),
                        action=str(action),
                        location=location
                    )
                continue

            elif msg_type == "sensor_data":
                acc = packet.get("accelerometer")
                gyro = packet.get("gyroscope")
                if not acc or not gyro:
                    continue

                ts = packet.get("timestamp", time.time())
                unit = packet.get("unit", "m/s2")
                loc = packet.get("location")

                try:
                    ax = float(acc.get("x", 0.0))
                    ay = float(acc.get("y", 0.0))
                    az = float(acc.get("z", 0.0))
                    gx = float(gyro.get("x", 0.0))
                    gy = float(gyro.get("y", 0.0))
                    gz = float(gyro.get("z", 0.0))
                except (ValueError, TypeError):
                    continue

                # Add reading into bounded in-memory buffer (maxlen = 250; drops oldest sample if full)
                norm_reading = buffer.add_reading(
                    timestamp=ts,
                    ax=ax, ay=ay, az=az,
                    gx=gx, gy=gy, gz=gz,
                    unit=unit,
                    location=loc
                )

                # Broadcast live sensor waveform point to authenticated user monitor
                if buffer.total_samples_received % 2 == 0:
                    acc_mag = np.sqrt(norm_reading[1]**2 + norm_reading[2]**2 + norm_reading[3]**2)
                    await ws_manager.broadcast_to_user(user.id, {
                        "type": "SENSOR_TICK",
                        "data": {
                            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                            "acc_x": round(norm_reading[1], 4),
                            "acc_y": round(norm_reading[2], 4),
                            "acc_z": round(norm_reading[3], 4),
                            "acc_magnitude": round(float(acc_mag), 3),
                            "gyro_x": round(norm_reading[4], 4),
                            "gyro_y": round(norm_reading[5], 4),
                            "gyro_z": round(norm_reading[6], 4),
                            "ground_truth": "REAL_PHONE",
                            "device_id": dev_id
                        }
                    })

                # Check sliding window stride: run inference if stride reached & buffer >= 64 samples
                if buffer.should_run_inference(stride=settings.INFERENCE_STRIDE_SAMPLES, min_samples=64):
                    window_arr = buffer.extract_inference_window(target_samples=settings.WINDOW_SAMPLES)
                    logger.info(f"inference executed for device='{dev_id}' (window shape: {window_arr.shape})")
                    
                    await prediction_service.predict_single_window(
                        db=db,
                        acc_x=window_arr[:, 0].tolist(),
                        acc_y=window_arr[:, 1].tolist(),
                        acc_z=window_arr[:, 2].tolist(),
                        gyro_x=window_arr[:, 3].tolist(),
                        gyro_y=window_arr[:, 4].tolist(),
                        gyro_z=window_arr[:, 5].tolist(),
                        device_id=dev_id,
                        user_id=user.id,
                        location=buffer.last_location,
                        require_confirmation=True
                    )

    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected: device_id='{dev_id}'")
    except Exception as e:
        logger.error(f"Error in sensor stream loop for {dev_id}: {e}")
    finally:
        ws_manager.unregister_device(dev_id)
        sensor_stream_manager.set_device_connection(dev_id, user.id if user else None, False)
        # Broadcast status update strictly to device owner
        if user:
            await ws_manager.broadcast_to_user(user.id, {
                "type": "DEVICE_STATUS_UPDATE",
                "data": {
                    "device_id": dev_id,
                    "connected": False,
                    "last_seen": datetime.now(timezone.utc).isoformat()
                }
            })
        db.close()

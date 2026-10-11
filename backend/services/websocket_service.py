"""
FallGuard AI - WebSocket Connection & Telemetry Broadcast Manager
"""

import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

logger = logging.getLogger("WebSocketManager")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitor_users: Dict[WebSocket, Optional[int]] = {}
        self.device_connections: Dict[str, WebSocket] = {}
        self.device_users: Dict[str, int] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[int] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.monitor_users[websocket] = user_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.monitor_users.pop(websocket, None)

    def register_device(self, device_id: str, websocket: WebSocket, user_id: Optional[int] = None):
        self.device_connections[device_id] = websocket
        if user_id is not None:
            self.device_users[device_id] = user_id

    def unregister_device(self, device_id: str):
        self.device_connections.pop(device_id, None)
        self.device_users.pop(device_id, None)

    async def send_to_device(self, device_id: str, message: Dict[str, Any]) -> bool:
        ws = self.device_connections.get(device_id)
        if not ws:
            return False
        try:
            await ws.send_text(json.dumps(message, default=str))
            return True
        except Exception as e:
            logger.warning(f"Failed to send targeted message to device {device_id}: {e}")
            self.unregister_device(device_id)
            return False

    async def broadcast_to_user(self, user_id: Optional[int], message: Dict[str, Any]):
        if user_id is None:
            await self.broadcast(message)
            return
        payload = dict(message)
        payload["user_id"] = user_id
        json_data = json.dumps(payload, default=str)

        dead_monitors = []
        for connection in self.active_connections:
            owner = self.monitor_users.get(connection)
            if owner is not None and owner != user_id:
                continue
            try:
                await connection.send_text(json_data)
            except Exception:
                dead_monitors.append(connection)
        for dead in dead_monitors:
            self.disconnect(dead)

        msg_type = message.get("type")
        if msg_type in ("FALL_PENDING_CONFIRMATION", "FALL_CONFIRMATION_RESOLVED", "FALL_ALERT", "MODEL_UNAVAILABLE", "HELP_RESULT"):
            dead_devices = []
            for dev_id, dev_ws in list(self.device_connections.items()):
                if self.device_users.get(dev_id) not in (user_id, None):
                    continue
                try:
                    await dev_ws.send_text(json_data)
                except Exception:
                    dead_devices.append(dev_id)
            for dead_dev in dead_devices:
                self.unregister_device(dead_dev)

    async def broadcast(self, message: Dict[str, Any]):
        json_data = json.dumps(message, default=str)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json_data)
            except Exception:
                disconnected.append(connection)
        for dead_conn in disconnected:
            self.disconnect(dead_conn)

        if message.get("type") in ("FALL_PENDING_CONFIRMATION", "FALL_CONFIRMATION_RESOLVED", "FALL_ALERT"):
            target_user = message.get("user_id") or (message.get("event") or {}).get("user_id")
            dead_devices = []
            for dev_id, dev_ws in list(self.device_connections.items()):
                if target_user is not None and self.device_users.get(dev_id) not in (target_user, None):
                    continue
                try:
                    await dev_ws.send_text(json_data)
                except Exception:
                    dead_devices.append(dev_id)
            for dead_dev in dead_devices:
                self.unregister_device(dead_dev)


ws_manager = ConnectionManager()

"""
FallGuard AI - Async Background Simulation Service
Runs live simulated wearable sensor stream and streams telemetry through WebSockets.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from collections import deque
import numpy as np

from simulation.realtime_simulator import generate_sensor_stream_generator
from backend.services.prediction_service import prediction_service
from backend.services.websocket_service import ws_manager
from backend.database.session import SessionLocal
from backend.models.db_models import SensorReading

logger = logging.getLogger("SimulationService")

class SimulationService:
    def __init__(self):
        self.is_running: bool = False
        self.is_paused: bool = False
        self.playback_speed: float = 1.0
        self.scenario: str = "mixed_activities_with_fall"
        self.samples_replayed: int = 0
        self.total_samples: int = 2000
        self.current_activity: str = "IDLE"
        self.current_risk: str = "LOW"
        self._task: Optional[asyncio.Task] = None
        self._sliding_buffer = deque(maxlen=128)
        self._manual_fall_injected: bool = False
        self.user_id: Optional[int] = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_scenario": self.scenario,
            "playback_speed": self.playback_speed,
            "samples_replayed": self.samples_replayed,
            "total_samples": self.total_samples,
            "current_activity": self.current_activity,
            "current_risk": self.current_risk
        }

    async def start(self, scenario: str = "mixed_activities_with_fall", playback_speed: float = 1.0, user_id: Optional[int] = None):
        if self._task and not self._task.done():
            self.stop()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
        self.scenario = scenario
        self.user_id = user_id
        self.playback_speed = max(0.2, min(playback_speed, 10.0))
        self.is_running = True
        self.is_paused = False
        self.samples_replayed = 0
        self._sliding_buffer.clear()
        
        self._task = asyncio.create_task(self._simulation_loop(user_id))
        logger.info(f"Started simulation scenario: '{scenario}' at {playback_speed}x speed.")

    def stop(self):
        self.is_running = False
        self.is_paused = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Stopped simulation.")

    def pause(self):
        self.is_paused = True
        logger.info("Paused simulation.")

    def resume(self):
        self.is_paused = False
        logger.info("Resumed simulation.")

    def trigger_manual_fall(self):
        """Inject a high-impact fall signature on demand."""
        self._manual_fall_injected = True
        logger.info("Manual high-impact fall signature injected into stream.")

    async def _simulation_loop(self, user_id: Optional[int]):
        gen = generate_sensor_stream_generator(scenario=self.scenario)
        db = SessionLocal()
        
        # Base tick interval at 50Hz = 0.02s
        base_interval = 0.020
        step_counter = 0
        
        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(0.2)
                    continue
                    
                reading = next(gen)
                self.samples_replayed += 1
                step_counter += 1
                
                # Check manual fall override
                if self._manual_fall_injected:
                    reading["acc_x"] = 3.8
                    reading["acc_y"] = 2.9
                    reading["acc_z"] = -4.2
                    reading["gyro_x"] = 4.5
                    reading["gyro_y"] = 3.9
                    reading["gyro_z"] = 2.8
                    reading["ground_truth"] = "FALL"
                    reading["is_fall"] = 1
                    self._manual_fall_injected = False
                    
                ax = reading["acc_x"]
                ay = reading["acc_y"]
                az = reading["acc_z"]
                gx = reading["gyro_x"]
                gy = reading["gyro_y"]
                gz = reading["gyro_z"]
                
                acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
                
                # Add to rolling buffer
                self._sliding_buffer.append([ax, ay, az, gx, gy, gz])
                
                # Broadcast raw sensor point over WebSocket every 2 samples (25Hz update rate for smooth UI rendering)
                if step_counter % 2 == 0:
                    ws_point = {
                        "type": "SENSOR_TICK",
                        "data": {
                            "timestamp": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
                            "acc_x": ax,
                            "acc_y": ay,
                            "acc_z": az,
                            "acc_magnitude": round(float(acc_mag), 3),
                            "gyro_x": gx,
                            "gyro_y": gy,
                            "gyro_z": gz,
                            "ground_truth": reading.get("ground_truth", "NORMAL")
                        }
                    }
                    await ws_manager.broadcast_to_user(user_id, ws_point)
                    
                # Run sliding window prediction every 10 samples (every 0.2s) once buffer has minimum 64 samples
                if len(self._sliding_buffer) >= 64 and step_counter % 10 == 0:
                    arr = np.array(self._sliding_buffer)
                    # Pad to 128 if needed
                    if len(arr) < 128:
                        pad = np.repeat(arr[-1:], 128 - len(arr), axis=0)
                        arr_padded = np.vstack([arr, pad])
                    else:
                        arr_padded = arr
                        
                    pred = await prediction_service.predict_single_window(
                        db=db,
                        acc_x=arr_padded[:, 0].tolist(),
                        acc_y=arr_padded[:, 1].tolist(),
                        acc_z=arr_padded[:, 2].tolist(),
                        gyro_x=arr_padded[:, 3].tolist(),
                        gyro_y=arr_padded[:, 4].tolist(),
                        gyro_z=arr_padded[:, 5].tolist(),
                        user_id=user_id,
                    )
                    self.current_activity = pred["activity"]
                    self.current_risk = pred["risk_level"]
                    
                # Delay based on playback speed
                delay = base_interval / self.playback_speed
                await asyncio.sleep(delay)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
        finally:
            db.close()
            self.is_running = False

simulation_service = SimulationService()

"""
FallGuard AI - Simulation Control API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from backend.api.dependencies import get_current_user
from backend.models.db_models import User
from backend.schemas.schemas import SimulationControlRequest, SimulationStatusResponse
from backend.services.simulation_service import simulation_service

router = APIRouter(prefix="/simulation", tags=["Simulation"])

@router.get("/status", response_model=SimulationStatusResponse)
def get_simulation_status(current_user: User = Depends(get_current_user)):
    """Get current state of the real MobiFall trial replay."""
    if simulation_service.is_running and simulation_service.user_id != current_user.id:
        raise HTTPException(status_code=409, detail="The simulator is active for another account.")
    return simulation_service.get_status()

@router.post("/control", response_model=SimulationStatusResponse)
async def control_simulation(payload: SimulationControlRequest, current_user: User = Depends(get_current_user)):
    """Control a real MobiFall trial replay (start, stop, pause, resume)."""
    action = payload.action.lower()
    if simulation_service.is_running and simulation_service.user_id != current_user.id:
        raise HTTPException(status_code=409, detail="The simulator is active for another account.")
    
    if action == "start":
        await simulation_service.start(
            scenario=payload.scenario or "dataset_replay",
            playback_speed=payload.playback_speed or 1.0,
            user_id=current_user.id,
            subject=payload.subject,
            activity_code=payload.activity_code,
            trial=payload.trial,
        )
    elif action == "stop":
        simulation_service.stop()
    elif action == "pause":
        simulation_service.pause()
    elif action == "resume":
        simulation_service.resume()
    else:
        raise HTTPException(status_code=400, detail=f"Invalid simulation action '{action}'.")
        
    return simulation_service.get_status()

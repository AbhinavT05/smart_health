"""
FallGuard AI - Model Metadata & Explainability API Endpoints
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, List

from backend.core.config import settings
from backend.services.prediction_service import prediction_service
from backend.schemas.schemas import ModelInfoResponse

router = APIRouter(prefix="/model", tags=["Model"])

@router.get("/info", response_model=ModelInfoResponse)
def get_model_info():
    """Retrieve metadata, features count, metrics, and top influential features."""
    detector = prediction_service.detector
    
    # Load metrics from file or bundle
    metrics = {}
    if settings.METRICS_PATH.exists():
        try:
            with open(settings.METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = detector.model_bundle.get("metrics", {}) if detector.model_bundle else {}
    elif detector.model_bundle:
        metrics = detector.model_bundle.get("metrics", {})

    top10_feats = []
    if detector.model_bundle and "feature_importance" in detector.model_bundle:
        feat_imp = detector.model_bundle["feature_importance"]
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        top10_feats = [{"feature": k, "importance": round(v, 4)} for k, v in sorted_feats]

    return ModelInfoResponse(
        model_name=detector.model_name,
        model_version=detector.model_bundle.get("model_version", detector.model_version) if detector.model_bundle else detector.model_version,
        model_available=detector.available,
        unavailable_reason=detector.unavailable_reason,
        trained_at=detector.model_bundle.get("trained_at", None) if detector.model_bundle else None,
        features_count=len(detector.feature_names),
        classes=detector.classes if detector.classes else ["FALL", "NON_FALL"],
        window_size_seconds=settings.WINDOW_SIZE_SECONDS,
        sampling_rate=settings.SAMPLING_RATE,
        fusion=detector.fusion_config if detector.available else None,
        metrics=metrics,
        feature_importance_top10=top10_feats,
        alert_cooldown_seconds=settings.ALERT_COOLDOWN_SECONDS,
        confirmation_timeout_seconds=settings.FALL_CONFIRMATION_TIMEOUT_SECONDS,
    )

@router.get("/comparisons")
def get_model_comparisons():
    """Retrieve full 5-model benchmark comparison table."""
    comp_file = settings.REPORTS_DIR / "model_comparison_report.json"
    if comp_file.exists():
        with open(comp_file, "r") as f:
            return json.load(f)
    return []

@router.get("/figures/{figure_name}")
def get_model_figure(figure_name: str):
    """Retrieve generated high-resolution evaluation figures."""
    fig_path = settings.FIGURES_DIR / figure_name
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{figure_name}' not found.")
    return FileResponse(fig_path)

"""ML endpoints, served by the SAME app as the agents (src/api/main.py includes this router automatically).

    GET  /api/v1/anomaly/health    is the model file loadable?            (no key)
    POST /api/v1/anomaly/predict   ONE raw sensor reading -> incident JSON
    POST /api/v1/anomaly/window    a window of readings   -> incident JSON (the worst reading decides)

Both POST endpoints return the incident JSON the agent layer expects, so you can forward it unchanged to
POST /api/v1/incident/analyze (see docs/CONTRACTS.md section 1). They need the same X-API-Key header as the agents.

The model bundle (models/best_model.joblib) is loaded on the first call, not at start-up, so the agent layer
still starts on a machine without scikit-learn. If the bundle or a library is missing the ML endpoints answer 503
with the reason and everything else keeps working.
"""
from __future__ import annotations

import hmac
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from src.agents.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
# The saved model refers to classes defined in src/data/pipelines.py, so that file must be importable as `pipelines`.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = get_settings().agent_api_key.encode("utf-8")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), expected):    # bytes: no 500 on odd characters
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


router = APIRouter(prefix="/api/v1/anomaly", tags=["ml"])


class SingleReadingRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-001"])
    asset_id: str = Field("esp32-conveyor-01", examples=["esp32-conveyor-01"])
    timestamp: Optional[datetime] = Field(None, description="Reading time. UTC now when omitted.")
    sensor_data: Dict[str, Any] = Field(..., description="One raw sensor reading (the CSV columns).")


class SensorSample(BaseModel):
    timestamp: datetime
    sensor_data: Dict[str, Any]


class WindowRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-001"])
    asset_id: str = Field("esp32-conveyor-01", examples=["esp32-conveyor-01"])
    samples: List[SensorSample] = Field(..., min_length=1, description="The readings of one window.")


_bundle: Any = None
_predict: Any = None


def _load():
    """Load the model once. Raises HTTPException(503) with the reason when it cannot be used."""
    global _bundle, _predict
    if _bundle is None:
        if not BUNDLE_PATH.exists():
            raise HTTPException(503, f"ML model file not found: {BUNDLE_PATH}")
        try:
            import joblib
            from pipelines import predict_raw_with_top_features
            _bundle = joblib.load(BUNDLE_PATH)
            _predict = predict_raw_with_top_features
        except Exception as e:     # missing scikit-learn / joblib, or a scikit-learn version the model was not saved with
            raise HTTPException(503, f"ML model cannot be loaded ({type(e).__name__}: {e}). "
                                     f"Install the versions in requirements.txt (the model was saved with scikit-learn 1.7.2).")
    return _bundle, _predict


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_anomaly_score(raw_score: float, threshold: float) -> float:
    """Map the model's raw score to 0..1 (0.5 = the model threshold)."""
    if threshold <= 0:
        return 1.0 if raw_score > threshold else 0.0
    return float(max(0.0, min(1.0, raw_score / (raw_score + threshold))))


def _model_info(bundle) -> dict:
    return {"name": bundle["key"], "version": "0.1", "feature_version": "f1"}


@router.get("/health")
def anomaly_health():
    """Can the model be loaded? (loads it if needed, so the first call may take a moment)."""
    try:
        bundle, _ = _load()
    except HTTPException as e:
        return {"status": "unavailable", "model_loaded": False, "reason": e.detail}
    return {"status": "ok", "model_loaded": True, "model": _model_info(bundle), "threshold": float(bundle["threshold"])}


@router.post("/predict", dependencies=[Depends(require_key)])
def predict_single(request: SingleReadingRequest):
    """Score ONE raw reading. The result is the incident JSON for POST /api/v1/incident/analyze."""
    bundle, predict = _load()
    try:
        scores, flags, top = predict(bundle, pd.DataFrame([request.sensor_data]), top_n=5)
        raw, threshold = float(scores[0]), float(bundle["threshold"])
        stamp = iso_utc(request.timestamp or datetime.now(timezone.utc))
        return {"incident_id": request.incident_id, "asset_id": request.asset_id, "timestamp": stamp,
                "window": {"start": stamp, "end": stamp, "n_samples": 1},
                "is_anomaly": bool(flags[0]), "anomaly_score": normalized_anomaly_score(raw, threshold),
                "raw_score": raw, "threshold": threshold, "model": _model_info(bundle), "top_features": top[0]}
    except (ValueError, TypeError, KeyError, NotImplementedError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e


@router.post("/window", dependencies=[Depends(require_key)])
def predict_window(request: WindowRequest):
    """Score a window of readings. Anomalous if any reading is; the worst reading gives the score and top features."""
    bundle, predict = _load()
    try:
        ordered = sorted(request.samples, key=lambda s: s.timestamp)
        scores, flags, top = predict(bundle, pd.DataFrame([s.sensor_data for s in ordered]), top_n=5)
        scores, flags = [float(x) for x in scores], [bool(x) for x in flags]
        threshold = float(bundle["threshold"])
        worst = max(range(len(scores)), key=lambda i: scores[i])
        start, end = ordered[0].timestamp, ordered[-1].timestamp
        return {"incident_id": request.incident_id, "asset_id": request.asset_id, "timestamp": iso_utc(end),
                "window": {"start": iso_utc(start), "end": iso_utc(end), "n_samples": len(ordered),
                           "duration_seconds": (end - start).total_seconds()},
                "is_anomaly": any(flags), "anomaly_score": normalized_anomaly_score(scores[worst], threshold),
                "raw_score": scores[worst], "threshold": threshold, "model": _model_info(bundle),
                "top_features": top[worst]}
    except (ValueError, TypeError, KeyError, NotImplementedError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Window inference failed: {e}") from e

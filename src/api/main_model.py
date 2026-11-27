# To run this FastAPI app, use the following command: uvicorn api.main_model:app --reload

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "script"
sys.path.insert(0, str(SCRIPT_DIR))

import sys
DATA_DIR = PROJECT_ROOT / "src" / "data"
sys.path.insert(0, str(DATA_DIR))
from pipelines import predict_raw_with_top_features

BUNDLE_PATH = PROJECT_ROOT / "data" / "results" / "best_model.joblib"

app = FastAPI(
    title="AegisOT Anomaly Detection API",
    version="1.2.0",
    description="Anomaly detection API for single readings and window analysis.",
)


class SingleReadingRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-001"])
    asset_id: str = Field("esp32-conveyor-01", examples=["esp32-conveyor-01"])
    timestamp: Optional[datetime] = Field(
        None,
        description="Sensor reading timestamp. UTC is used when omitted."
    )
    sensor_data: Dict[str, Any] = Field(
        ...,
        description="One raw sensor reading."
    )


class SensorSample(BaseModel):
    timestamp: datetime
    sensor_data: Dict[str, Any]


class WindowAnalyzeRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-001"])
    asset_id: str = Field("esp32-conveyor-01", examples=["esp32-conveyor-01"])
    samples: List[SensorSample] = Field(
        ...,
        min_length=1,
        description="Sensor readings belonging to one window."
    )


_bundle = None


def get_bundle():
    global _bundle

    if _bundle is None:
        if not BUNDLE_PATH.exists():
            raise RuntimeError(f"Model bundle not found: {BUNDLE_PATH}")

        _bundle = joblib.load(BUNDLE_PATH)

    return _bundle


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_anomaly_score(raw_score: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0 if raw_score > threshold else 0.0

    return float(
        max(0.0, min(1.0, raw_score / (raw_score + threshold)))
    )


def get_model_metadata(bundle):
    return {
        "name": bundle["key"],
        "version": "0.1",
        "feature_version": "f1",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": BUNDLE_PATH.exists()
    }


@app.post("/api/v1/incident/analyze")
def analyze_single_reading(request: SingleReadingRequest):
    try:
        bundle = get_bundle()
        raw_frame = pd.DataFrame([request.sensor_data])
        scores, flags, top_features = predict_raw_with_top_features(
            bundle,
            raw_frame,
            top_n=5
        )

        raw_score = float(scores[0])
        threshold = float(bundle["threshold"])
        reading_time = request.timestamp or datetime.now(timezone.utc)
        timestamp = iso_utc(reading_time)
        return {
            "incident_id": request.incident_id,
            "asset_id": request.asset_id,
            "timestamp": timestamp,
            "window": {
                "start": timestamp,
                "end": timestamp,
                "n_samples": 1,
            },
            "is_anomaly": bool(flags[0]),
            "anomaly_score": normalized_anomaly_score(raw_score, threshold),
            "raw_score": raw_score,
            "threshold": threshold,
            "model": get_model_metadata(bundle),
            "top_features": top_features[0],
        }

    except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}"
        ) from exc


@app.post("/api/v1/window/analyze")
def analyze_window(request: WindowAnalyzeRequest):
    try:
        bundle = get_bundle()

        ordered_samples = sorted(
            request.samples,
            key=lambda sample: sample.timestamp
        )

        raw_frame = pd.DataFrame([
            sample.sensor_data
            for sample in ordered_samples
        ])

        scores, flags, top_features = predict_raw_with_top_features(
            bundle,
            raw_frame,
            top_n=5
        )

        scores = [float(score) for score in scores]
        flags = [bool(flag) for flag in flags]

        threshold = float(bundle["threshold"])

        start_time = ordered_samples[0].timestamp
        end_time = ordered_samples[-1].timestamp
        n_samples = len(ordered_samples)

        is_anomaly = any(flags)

        worst_index = max(
            range(len(scores)),
            key=lambda index: scores[index]
        )

        window_raw_score = scores[worst_index]
        window_anomaly_score = normalized_anomaly_score(
            window_raw_score,
            threshold
        )

        duration_seconds = (end_time - start_time).total_seconds()

        return {
            "incident_id": request.incident_id,
            "asset_id": request.asset_id,
            "timestamp": iso_utc(end_time),
            "window": {
                "start": iso_utc(start_time),
                "end": iso_utc(end_time),
                "n_samples": n_samples,
                "duration_seconds": duration_seconds,
            },
            "is_anomaly": is_anomaly,
            "anomaly_score": window_anomaly_score,
            "raw_score": window_raw_score,
            "threshold": threshold,
            "model": get_model_metadata(bundle),
            "top_features": top_features[worst_index],
        }

    except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Window inference failed: {exc}"
        ) from exc

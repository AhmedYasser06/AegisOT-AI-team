"""Small helpers shared by the agent tests (kept here so tests/ stays a flat folder)."""
from __future__ import annotations

from .schemas import AlertIn


def make_alert(asset_id: str = "esp32-conveyor-01", score: float = 0.93, incident_id: str = "inc-1", **kw) -> AlertIn:
    return AlertIn(incident_id=incident_id, alert_id=f"{asset_id}:1", asset_id=asset_id,
                   anomaly_score=score, model_version="iforest-0.1", **kw)

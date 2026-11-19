"""Deterministic mock upstreams so the agent runs and is testable without other teams.

Try these asset_ids:
  esp32-conveyor-01   -> physical fault (vibration + current up, network healthy)
  esp32-conveyor-net  -> network issue  (telemetry gaps + packet loss)
  esp32-rogue-01      -> security       (unclaimed device, broker denials, ACL hits)
  esp32-sensor-01     -> sensor fault   (temperature stuck at -127)
  anything else       -> healthy data, weak anomaly -> inconclusive
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .base import ClientError

BASE = {"temperature": 40.0, "vibration_rms": 0.8, "current": 1.2, "speed": 1500.0}
INTERVAL_S = 1.0


def _samples(asset_id: str, minutes: int, *, fault_scale: dict | None = None, keep_ratio: float = 1.0,
             stuck: dict | None = None) -> list[dict]:
    rng = random.Random(asset_id)
    n = int(minutes * 60 / INTERVAL_S)
    now = datetime.now(timezone.utc)
    out = []
    for i in range(n):
        if i / n > keep_ratio:            # data stops arriving (network gap)
            break
        row = {"ts": (now - timedelta(seconds=(n - i) * INTERVAL_S)).isoformat()}
        for k, v in BASE.items():
            val = v * (1 + rng.uniform(-0.03, 0.03))
            if fault_scale and i / n > 0.5 and k in fault_scale:     # fault starts mid-window
                val *= fault_scale[k]
            if stuck and k in stuck:
                val = stuck[k]
            row[k] = round(val, 3)
        out.append(row)
    return out


SCENARIOS: dict[str, dict[str, Any]] = {
    "esp32-conveyor-01": dict(
        telemetry=dict(fault_scale={"vibration_rms": 2.6, "current": 1.7}),
        network=dict(reachable=True, latency_ms=12, packet_loss_pct=0.0, link_status="up"),
        events=[], enrollment="operational",
        maintenance=dict(health_score=0.55, days_since_maintenance=120, open_work_items=0),
        anomaly=dict(score=0.93, top_features=[{"name": "vibration_rms", "contribution": 0.71},
                                              {"name": "current", "contribution": 0.22}])),
    "esp32-conveyor-net": dict(
        telemetry=dict(keep_ratio=0.35),
        network=dict(reachable=True, latency_ms=480, packet_loss_pct=38.0, link_status="degraded"),
        events=[], enrollment="operational",
        maintenance=dict(health_score=0.9, days_since_maintenance=20, open_work_items=0),
        anomaly=dict(score=0.7, top_features=[{"name": "missing_samples", "contribution": 0.8}])),
    "esp32-rogue-01": dict(
        telemetry=dict(), network=dict(reachable=True, latency_ms=15, packet_loss_pct=0.0,
                                       link_status="up", acl_violations=3),
        events=[{"ts": "now", "type": "broker_denied", "detail": "publish to aegisot/devices/esp32-conveyor-01/commands"}] * 4
               + [{"ts": "now", "type": "enroll_rejected", "detail": "unknown identifier"}],
        enrollment="unclaimed",
        maintenance=dict(health_score=1.0, days_since_maintenance=0, open_work_items=0),
        anomaly=dict(score=0.88, top_features=[{"name": "msg_rate", "contribution": 0.9}])),
    "esp32-sensor-01": dict(
        telemetry=dict(stuck={"temperature": -127.0}),
        network=dict(reachable=True, latency_ms=10, packet_loss_pct=0.0, link_status="up"),
        events=[], enrollment="operational",
        maintenance=dict(health_score=0.85, days_since_maintenance=30, open_work_items=0),
        anomaly=dict(score=0.8, top_features=[{"name": "temperature", "contribution": 0.95}])),
    "esp32-hot-01": dict(      # temperature really climbs (40C -> ~88C) while the other channels stay normal
        telemetry=dict(fault_scale={"temperature": 2.2}),
        network=dict(reachable=True, latency_ms=11, packet_loss_pct=0.0, link_status="up"),
        events=[], enrollment="operational",
        maintenance=dict(health_score=0.8, days_since_maintenance=40, open_work_items=0),
        anomaly=dict(score=0.91, top_features=[{"name": "temperature", "contribution": 0.9}])),
    "esp32-hot-sensor-01": dict(   # temperature reads an impossible 250C constantly -> broken sensor
        telemetry=dict(stuck={"temperature": 250.0}),
        network=dict(reachable=True, latency_ms=10, packet_loss_pct=0.0, link_status="up"),
        events=[], enrollment="operational",
        maintenance=dict(health_score=0.85, days_since_maintenance=30, open_work_items=0),
        anomaly=dict(score=0.85, top_features=[{"name": "temperature", "contribution": 0.95}])),
}
DEFAULT = dict(
    telemetry=dict(), network=dict(reachable=True, latency_ms=10, packet_loss_pct=0.0, link_status="up"),
    events=[], enrollment="operational",
    maintenance=dict(health_score=0.9, days_since_maintenance=15, open_work_items=0),
    anomaly=dict(score=0.3, top_features=[]))


def _sc(asset_id: str) -> dict[str, Any]:
    return SCENARIOS.get(asset_id, DEFAULT)


class MockBackendClient:
    def __init__(self):
        self.posted: list[dict] = []

    def get_asset(self, asset_id):
        return {"asset_id": asset_id, "station": "B", "type": "conveyor_drive",
                "enrollment_state": _sc(asset_id)["enrollment"], "firmware": "0.3.1",
                "profile": "conveyor-drive-v1", "baseline": BASE}

    def get_telemetry(self, asset_id, minutes, end=None):
        return {"asset_id": asset_id, "sampling_interval_s": INTERVAL_S, "window_minutes": minutes,
                "baseline": BASE, "samples": _samples(asset_id, minutes, **_sc(asset_id)["telemetry"])}

    def get_events(self, asset_id, kind, minutes):
        return {"asset_id": asset_id, "events": _sc(asset_id)["events"] if kind == "security" else []}

    def get_maintenance(self, asset_id):
        return {"asset_id": asset_id, **_sc(asset_id)["maintenance"]}

    def post_assessment(self, assessment):
        self.posted.append(assessment)
        return {"stored": True, "approval_request_id": f"apr-{assessment['incident_id']}"}


class MockAnomalyClient:
    def get_alert(self, alert_id):
        # alert ids from the mock usually look like "<asset_id>:<n>"; fall back to default
        asset = alert_id.split(":")[0]
        a = _sc(asset)["anomaly"]
        return {"alert_id": alert_id, "score": a["score"], "threshold": 0.6,
                "model_version": "iforest-0.1", "top_features": a["top_features"]}


class MockNetworkClient:
    def get_health(self, asset_id):
        n = dict(_sc(asset_id)["network"])
        return {"asset_id": asset_id, "zone": "ot-segment", "interface": {"name": "Gi0/1", "status": n["link_status"]},
                "acl_violations": 0, **n}


class FailingClient:
    """Test helper: every call raises ClientError (simulates a team's service being down)."""
    def __getattr__(self, name):
        def _boom(*a, **k):
            raise ClientError(f"{name}: upstream unavailable")
        return _boom
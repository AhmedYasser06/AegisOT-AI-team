"""Real HTTP adapters. Paths are the contract to agree with each team (see README §6)."""
from __future__ import annotations

from typing import Any

import httpx

from .base import ClientError


class _Http:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 5.0,
                 transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._c = httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, **kw) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(2):  # one retry for transient failures
            try:
                r = self._c.request(method, path, **kw)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise ClientError(f"{path}: expected JSON object")
                return data
            except ClientError:
                raise
            except (httpx.HTTPError, ValueError) as e:
                last = e
        raise ClientError(f"{method} {path} failed: {last}")


class HttpBackendClient(_Http):
    def get_asset(self, asset_id):        return self._request("GET", f"/api/devices/{asset_id}")
    def get_telemetry(self, asset_id, minutes):
        return self._request("GET", f"/api/devices/{asset_id}/telemetry", params={"minutes": minutes})
    def get_events(self, asset_id, kind, minutes):
        return self._request("GET", f"/api/devices/{asset_id}/events", params={"kind": kind, "minutes": minutes})
    def get_maintenance(self, asset_id):  return self._request("GET", f"/api/devices/{asset_id}/maintenance")
    def post_assessment(self, assessment): return self._request("POST", "/api/agent/assessments", json=assessment)


class HttpAnomalyClient(_Http):
    def get_alert(self, alert_id):        return self._request("GET", f"/v1/alerts/{alert_id}")


class HttpNetworkClient(_Http):
    def get_health(self, asset_id):       return self._request("GET", f"/network/devices/{asset_id}/health")


class PythonAnomalyClient:
    """In-process adapter: call the AI team's own code instead of HTTP.

    target = "package.module:function" (or a callable). The function receives `alert_id`
    and must return a dict: {score, threshold?, model_version?, top_features: [{name, contribution}]}.
    """
    def __init__(self, target):
        self._target, self._fn = target, (target if callable(target) else None)

    def _resolve(self):
        if self._fn is None:
            import importlib
            mod, _, name = str(self._target).partition(":")
            if not mod or not name:
                raise ClientError(f"ANOMALY_FUNCTION must look like 'package.module:function', got {self._target!r}")
            try:
                self._fn = getattr(importlib.import_module(mod), name)
            except (ImportError, AttributeError) as e:
                raise ClientError(f"cannot import {self._target}: {e}")
        return self._fn

    def get_alert(self, alert_id):
        try:
            out = self._resolve()(alert_id)
        except ClientError:
            raise
        except Exception as e:
            raise ClientError(f"anomaly function failed: {e}")
        if not isinstance(out, dict):
            raise ClientError("anomaly function must return a dict")
        return out

"""Integration interfaces. Each team implements the HTTP side; we provide mocks."""
from __future__ import annotations

from typing import Any, Protocol


class ClientError(Exception):
    """Any upstream failure (timeout, 5xx, bad JSON). Agents degrade gracefully on this."""


class BackendClient(Protocol):          # Laravel team
    def get_asset(self, asset_id: str) -> dict[str, Any]: ...
    def get_telemetry(self, asset_id: str, minutes: int) -> dict[str, Any]: ...
    def get_events(self, asset_id: str, kind: str, minutes: int) -> dict[str, Any]: ...
    def get_maintenance(self, asset_id: str) -> dict[str, Any]: ...
    def post_assessment(self, assessment: dict[str, Any]) -> dict[str, Any]: ...


class AnomalyClient(Protocol):          # ML team (anomaly service)
    def get_alert(self, alert_id: str) -> dict[str, Any]: ...


class NetworkClient(Protocol):          # Network / GNS3 team
    def get_health(self, asset_id: str) -> dict[str, Any]: ...

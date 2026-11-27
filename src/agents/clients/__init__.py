"""Integration adapters. Each upstream (backend / anomaly / network) switches independently via *_MODE."""
from __future__ import annotations

from dataclasses import dataclass

from ..settings import Settings
from .base import AnomalyClient, BackendClient, ClientError, NetworkClient
from .http import HttpAnomalyClient, HttpBackendClient, HttpNetworkClient, PythonAnomalyClient
from .mock import MockAnomalyClient, MockBackendClient, MockNetworkClient

__all__ = ["Clients", "build_clients", "ClientError"]


@dataclass
class Clients:
    backend: BackendClient
    anomaly: AnomalyClient
    network: NetworkClient


class _NoAnomalyClient:
    """anomaly_mode=inline: the ML result arrives inside the request, nothing to fetch."""
    def get_alert(self, alert_id):
        raise ClientError("anomaly_mode=inline: no external anomaly service configured")


def _anomaly(s: Settings):
    if s.anomaly_mode == "http":
        return HttpAnomalyClient(s.anomaly_url, s.anomaly_token)
    if s.anomaly_mode == "python":
        if not s.anomaly_function:
            raise ValueError("ANOMALY_MODE=python requires ANOMALY_FUNCTION=package.module:function")
        return PythonAnomalyClient(s.anomaly_function)
    if s.anomaly_mode == "mock":
        return MockAnomalyClient()
    return _NoAnomalyClient()


def _backend(s: Settings):
    if s.backend_mode == "http":
        return HttpBackendClient(s.backend_url, s.backend_token)
    if s.backend_mode == "csv":
        from .csv_backend import CsvBackendClient          # lazy: needs pandas, only when this mode is used
        return CsvBackendClient(s.backend_csv)
    return MockBackendClient()


def build_clients(s: Settings) -> Clients:
    return Clients(
        backend=_backend(s),
        anomaly=_anomaly(s),
        network=HttpNetworkClient(s.network_url, s.network_token) if s.network_mode == "http" else MockNetworkClient(),
    )

import httpx

from src.agents.clients import ClientError, Clients
from src.agents.clients.http import HttpNetworkClient, HttpBackendClient
from src.agents.clients.mock import FailingClient, MockAnomalyClient, MockBackendClient
from src.agents.settings import Settings
from src.agents.graph import build_graph, run_incident
from src.agents.testing import make_alert

import pytest


def _client(cls, handler):
    return cls("http://svc", "tok", transport=httpx.MockTransport(handler))


def test_http_network_client_sends_bearer_and_parses_json():
    seen = {}
    def handler(req):
        seen["auth"], seen["path"] = req.headers["authorization"], req.url.path
        return httpx.Response(200, json={"reachable": True, "latency_ms": 5, "packet_loss_pct": 0})
    out = _client(HttpNetworkClient, handler).get_health("esp32-1")
    assert out["reachable"] and seen == {"auth": "Bearer tok", "path": "/network/devices/esp32-1/health"}


def test_http_client_retries_once_then_raises_client_error():
    calls = []
    def handler(req):
        calls.append(1)
        return httpx.Response(503)
    with pytest.raises(ClientError):
        _client(HttpBackendClient, handler).get_asset("x")
    assert len(calls) == 2


def test_http_client_rejects_non_object_json():
    with pytest.raises(ClientError):
        _client(HttpNetworkClient, lambda r: httpx.Response(200, json=[1, 2])).get_health("x")


def test_network_service_down_still_returns_assessment_with_penalty(agent_settings):
    c = Clients(MockBackendClient(), MockAnomalyClient(), FailingClient())
    a = run_incident(build_graph(agent_settings, c), make_alert())
    assert a.likely_cause.value == "physical_fault"
    assert any("unavailable" in n and "network" in n for n in a.notes)
    assert a.confidence < 0.98


def test_everything_down_is_inconclusive_not_a_crash(agent_settings):
    c = Clients(FailingClient(), FailingClient(), FailingClient())
    a = run_incident(build_graph(agent_settings, c), make_alert())
    assert a.status == "inconclusive" and a.recommended_action.value == "escalate_to_human"


def test_backend_push_posts_assessment_when_enabled(agent_settings, agent_clients, agent_client):
    body = {"incident_id": "p1", "alert_id": "esp32-conveyor-01:1", "asset_id": "esp32-conveyor-01",
            "anomaly_score": 0.93}
    r = agent_client.post("/api/v1/incident/analyze?push=true", headers={"X-API-Key": "test-key"}, json=body)
    assert r.status_code == 200
    assert len(agent_clients.backend.posted) == 1 and agent_clients.backend.posted[0]["incident_id"] == "p1"
    r2 = agent_client.post("/api/v1/incident/analyze", headers={"X-API-Key": "test-key"}, json={**body, "incident_id": "p2"})
    assert len(agent_clients.backend.posted) == 1          # default: no push


def test_backend_push_failure_is_noted_not_fatal(agent_settings):
    from src.api.main import create_app
    from fastapi.testclient import TestClient
    bad = Clients(FailingClient(), MockAnomalyClient(), FailingClient())
    c = TestClient(create_app(agent_settings, bad))
    r = c.post("/api/v1/incident/analyze?push=true", headers={"X-API-Key": "test-key"},
               json={"incident_id": "p3", "alert_id": "a", "asset_id": "x", "anomaly_score": 0.5})
    assert r.status_code == 200 and any("backend push failed" in n for n in r.json()["notes"])


import sys
import types

import pytest

from src.agents.clients import ClientError, build_clients
from src.agents.clients.http import PythonAnomalyClient
from src.agents.settings import Settings


def test_python_mode_calls_function_by_path(monkeypatch):
    mod = types.ModuleType("fake_ai_team_mod")
    mod.get_alert = lambda alert_id: {"score": 0.91, "top_features": [{"name": "vibration_rms", "contribution": 0.8}]}
    monkeypatch.setitem(sys.modules, "fake_ai_team_mod", mod)
    s = Settings(_env_file=None, anomaly_mode="python", anomaly_function="fake_ai_team_mod:get_alert")
    out = build_clients(s).anomaly.get_alert("a1")
    assert out["score"] == 0.91


def test_python_mode_errors_become_client_errors(monkeypatch):
    with pytest.raises(ClientError):
        PythonAnomalyClient("no_such_module_xyz:f").get_alert("a")
    with pytest.raises(ClientError):
        PythonAnomalyClient("bad-format").get_alert("a")
    with pytest.raises(ClientError):
        PythonAnomalyClient(lambda a: [1]).get_alert("a")          # not a dict
    def boom(a): raise RuntimeError("model crashed")
    with pytest.raises(ClientError):
        PythonAnomalyClient(boom).get_alert("a")


def test_python_mode_requires_function():
    with pytest.raises(ValueError):
        build_clients(Settings(_env_file=None, anomaly_mode="python"))

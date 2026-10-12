"""Repo-specific behaviour: team JSON formats, config.yaml, LLM_PROVIDER, ML hook, CLI."""
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from src.agents.clients import Clients, build_clients
from src.agents.clients.mock import MockBackendClient, MockNetworkClient
from src.agents.graph import build_graph, run_incident
from src.agents.llm import normalize_model, resolve_llm
from src.agents.schemas import AlertIn
from src.agents.settings import Settings
from src.api.main import create_app

H = {"X-API-Key": "test-key"}
REPO = Path(__file__).resolve().parents[1]
# The exact "AI Result Example" from the team README (+ nothing else)
TEAM_PAYLOAD = {"asset_id": "esp32-conveyor-01", "incident_id": "INC-001", "is_anomaly": True,
                "anomaly_score": 0.91, "model": "isolation_forest", "timestamp": "2026-10-01T12:30:00Z"}


# ───────── input/output formats ─────────
def test_team_ai_result_format_is_accepted_as_is(agent_client):
    r = agent_client.post("/api/v1/incident/analyze", json=TEAM_PAYLOAD, headers=H)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["incident_id"] == "INC-001" and d["alert_id"] == "INC-001"        # alert_id defaults to incident_id
    assert d["likely_cause"] == "physical_fault" and d["approval_required"] is True
    for k in ("diagnosis", "evidence", "recommended_action", "recommended_action_text", "confidence"):
        assert d[k]


def test_alert_aliases_and_defaults():
    a = AlertIn(**TEAM_PAYLOAD)
    assert a.model_version == "isolation_forest" and a.detected_at.year == 2026 and a.alert_id == "INC-001"


def test_diagnosis_endpoint_matches_team_example_shape(agent_client):
    d = agent_client.post("/api/v1/diagnosis/analyze", json=TEAM_PAYLOAD, headers=H).json()
    assert set(d) == {"incident_id", "asset_id", "diagnosis", "confidence", "evidence", "recommended_action",
                      "action_id", "approval_required", "status"}
    assert isinstance(d["evidence"], list) and d["evidence"] and d["action_id"] == "stop_equipment"
    assert "stop" in d["recommended_action"].lower()


def test_not_an_anomaly_is_rejected(agent_client):
    r = agent_client.post("/api/v1/incident/analyze", json={**TEAM_PAYLOAD, "is_anomaly": False}, headers=H)
    assert r.status_code == 422


# ───────── inline ML result (anomaly_mode=inline, the default) ─────────
def test_inline_anomaly_is_used_without_any_anomaly_service():
    s = Settings(_env_file=None, anomaly_mode="inline", agent_api_key="test-key")
    c = Clients(MockBackendClient(), build_clients(s).anomaly, MockNetworkClient())
    a = run_incident(build_graph(s, c), AlertIn(**TEAM_PAYLOAD, top_features=[{"name": "vibration_rms", "contribution": 0.8}]))
    assert "anomaly:INC-001" in a.evidence_refs
    assert any("vibration_rms" in e for e in a.evidence)
    assert not any("unavailable" in n for n in a.notes)


# ───────── configs/config.yaml ─────────
def test_repo_config_yaml_is_valid_and_safe_by_default(monkeypatch):
    monkeypatch.setenv("AGENT_CONFIG_FILE", str(REPO / "configs" / "config.yaml"))
    for k in ("LLM_ENABLED", "LLM_PROVIDER", "ANOMALY_MODE", "API_PORT", "BACKEND_MODE"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_enabled is False and s.llm_provider == "gemini"
    assert s.anomaly_mode == "inline" and s.backend_mode == "mock" and s.api_port == 8001


def test_env_overrides_yaml(tmp_path, monkeypatch):
    f = tmp_path / "c.yaml"
    f.write_text("agents:\n  llm_enabled: true\n  min_confidence: 0.7\napi:\n  port: 9999\n")
    monkeypatch.setenv("AGENT_CONFIG_FILE", str(f))
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("MIN_CONFIDENCE", raising=False)
    monkeypatch.setenv("LLM_ENABLED", "false")                      # env wins over yaml
    s = Settings(_env_file=None)
    assert s.llm_enabled is False and s.min_confidence == 0.7 and s.api_port == 9999


# ───────── LLM_PROVIDER / LLM_MODEL (team .env style) ─────────
@pytest.mark.parametrize("provider,model,expected", [
    ("gemini", None, "gemini/gemini-flash-latest"),
    ("openai", None, "openai/gpt-4o-mini"),
    ("anthropic", None, "anthropic/claude-sonnet-4-6"),
    ("ollama", "qwen2.5", "ollama/qwen2.5"),
    ("gemini", "gemini-2.5-pro", "gemini/gemini-2.5-pro"),
    ("gemini", "openai/gpt-4o", "openai/gpt-4o"),            # explicit provider/model wins
    (None, None, "gemini/gemini-flash-latest"),                  # team default provider
    (None, "gpt-4o-mini", "gpt-4o-mini"),
])
def test_provider_and_model_resolution(provider, model, expected):
    s = Settings(_env_file=None, llm_provider=provider, llm_model=model)
    assert normalize_model(s, s.llm_model) == expected
    assert resolve_llm(s, "network").model == expected


def test_per_agent_override_with_provider():
    s = Settings(_env_file=None, llm_provider="gemini", llm_model_security="anthropic/claude-sonnet-4-6")
    assert resolve_llm(s, "security").model == "anthropic/claude-sonnet-4-6"
    assert resolve_llm(s, "diagnosis").model == "gemini/gemini-flash-latest"


# ───────── ML team hook in src/api/main.py ─────────
def test_ml_router_is_included_when_present(agent_settings, agent_clients, monkeypatch):
    router = APIRouter()
    router.add_api_route("/api/v1/anomaly/predict", lambda: {"is_anomaly": False}, methods=["POST"])
    fake = types.ModuleType("src.ml.router")
    fake.router = router
    monkeypatch.setitem(sys.modules, "src.ml.router", fake)
    c = TestClient(create_app(agent_settings, agent_clients))
    assert c.post("/api/v1/anomaly/predict").json() == {"is_anomaly": False}


def test_app_works_without_ml_module(agent_client):
    assert agent_client.get("/health").status_code == 200


# ───────── README commands ─────────
def test_cli_demo_runs_without_llm_or_keys():
    out = subprocess.run([sys.executable, "-m", "src.agents.graph"], cwd=REPO, capture_output=True, text=True,
                         timeout=90, env={**__import__("os").environ, "AGENT_CONFIG_FILE": "", "LLM_ENABLED": "false"})
    assert out.returncode == 0, out.stderr[-500:]
    for cause in ("physical_fault", "network_issue", "security_incident", "sensor_fault", "INCONCLUSIVE"):
        assert cause in out.stdout

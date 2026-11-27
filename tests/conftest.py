"""Fixtures for the agent tests (prefixed `agent_` to avoid clashing with other teams' fixtures)."""
import os

# hermetic: ignore configs/config.yaml while testing (tests build Settings explicitly)
os.environ["AGENT_CONFIG_FILE"] = ""
# a developer's .env / shell must not change test results (it made the key, LLM and backend differ per machine)
for _k, _v in {"AGENT_API_KEY": "test-key", "LLM_ENABLED": "false", "BACKEND_MODE": "mock", "NETWORK_MODE": "mock",
               "ANOMALY_MODE": "inline", "ML_VALIDATED": "false", "SPECIALIST_ENGINE": "direct"}.items():
    os.environ[_k] = _v
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pytest
from fastapi.testclient import TestClient

from src.agents.clients import build_clients
from src.agents.graph import build_graph
from src.agents.settings import Settings
from src.api.main import create_app


@pytest.fixture
def agent_settings():
    return Settings(_env_file=None, agent_api_key="test-key", llm_enabled=False, anomaly_mode="mock",
                    backend_mode="mock", network_mode="mock", ml_validated=True)


@pytest.fixture
def agent_clients(agent_settings):
    return build_clients(agent_settings)


@pytest.fixture
def agent_graph(agent_settings, agent_clients):
    return build_graph(agent_settings, agent_clients)


@pytest.fixture
def agent_client(agent_settings, agent_clients):
    return TestClient(create_app(agent_settings, agent_clients))

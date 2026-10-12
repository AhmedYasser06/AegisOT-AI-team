"""Fixtures for the agent tests (prefixed `agent_` to avoid clashing with other teams' fixtures)."""
import os

# hermetic: ignore configs/config.yaml while testing (tests build Settings explicitly)
os.environ["AGENT_CONFIG_FILE"] = ""
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
                    backend_mode="mock", network_mode="mock")


@pytest.fixture
def agent_clients(agent_settings):
    return build_clients(agent_settings)


@pytest.fixture
def agent_graph(agent_settings, agent_clients):
    return build_graph(agent_settings, agent_clients)


@pytest.fixture
def agent_client(agent_settings, agent_clients):
    return TestClient(create_app(agent_settings, agent_clients))

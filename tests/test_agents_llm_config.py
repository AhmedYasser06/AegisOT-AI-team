import pytest

from src.agents.settings import Settings
from src.agents.llm import ROLES, resolve_llm


def test_default_model_used_for_all_roles():
    s = Settings(_env_file=None, llm_model="openai/gpt-4o-mini")
    assert {resolve_llm(s, r).model for r in ROLES} == {"openai/gpt-4o-mini"}


def test_per_role_override_mixes_providers():
    s = Settings(_env_file=None, llm_model="ollama/llama3.1", llm_model_diagnosis="anthropic/claude-sonnet-4-6",
                 llm_model_supervisor="openai/gpt-4o")
    assert resolve_llm(s, "diagnosis").model == "anthropic/claude-sonnet-4-6"
    assert resolve_llm(s, "network").model == "ollama/llama3.1"
    assert resolve_llm(s, "supervisor").model == "openai/gpt-4o"


def test_env_vars_drive_config(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("LLM_API_BASE", "http://gateway:4000")
    s = Settings(_env_file=None)
    spec = resolve_llm(s, "security")
    assert spec.model == "gemini/gemini-2.0-flash" and spec.api_base == "http://gateway:4000"


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        resolve_llm(Settings(_env_file=None), "hacker")


def test_public_view_hides_key():
    s = Settings(_env_file=None, llm_api_key="sk-secret")
    pub = resolve_llm(s, "network").public()
    assert pub["api_key_set"] is True and "sk-secret" not in str(pub)

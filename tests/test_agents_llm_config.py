import pytest

from src.agents.settings import Settings
from src.agents.llm import resolve_llm
from src.agents.roles import LLM_ROLES


def test_default_model_used_for_all_roles():
    s = Settings(_env_file=None, llm_model="openai/gpt-4o-mini")
    assert {resolve_llm(s, r).model for r in LLM_ROLES} == {"openai/gpt-4o-mini"}


def test_per_role_override_mixes_providers():
    s = Settings(_env_file=None, llm_model="ollama/llama3.1", llm_model_diagnosis="anthropic/claude-sonnet-4-6",
                 llm_model_supervisor="openai/gpt-4o")
    assert resolve_llm(s, "diagnosis").model == "anthropic/claude-sonnet-4-6"
    assert resolve_llm(s, "supervisor").model == "openai/gpt-4o"
    assert resolve_llm(Settings(_env_file=None, llm_model="ollama/llama3.1"), "diagnosis").model == "ollama/llama3.1"


def test_env_vars_drive_config(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("LLM_API_BASE", "http://gateway:4000")
    s = Settings(_env_file=None)
    spec = resolve_llm(s, "diagnosis")
    assert spec.model == "gemini/gemini-2.0-flash" and spec.api_base == "http://gateway:4000"


@pytest.mark.parametrize("role", ["hacker", "network", "security", "maintenance"])
def test_unknown_or_rule_only_role_rejected(role):
    with pytest.raises(ValueError):
        resolve_llm(Settings(_env_file=None), role)


def test_public_view_hides_key():
    s = Settings(_env_file=None, llm_api_key="sk-secret")
    pub = resolve_llm(s, "supervisor").public()
    assert pub["api_key_set"] is True and "sk-secret" not in str(pub)

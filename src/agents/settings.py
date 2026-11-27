from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, InitSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]          # src/agents/settings.py -> repo root
ENV_FILES = (".env", str(REPO_ROOT / ".env"))

# Provider keys (GEMINI_API_KEY, OPENAI_API_KEY, ...) are read by LiteLLM from os.environ,
# not from this class, so export the .env file into the process environment first.
for _f in ENV_FILES:
    load_dotenv(_f, override=False)

Mode = Literal["mock", "http"]
BackendMode = Literal["mock", "http", "csv"]       # csv = replay a recorded CSV (clients/csv_backend.py)
AnomalyMode = Literal["inline", "mock", "http", "python"]


def _yaml_defaults() -> dict[str, Any]:
    """Flat `agents:` section of configs/config.yaml (keys = setting names). Set AGENT_CONFIG_FILE="" to disable."""
    path = os.environ.get("AGENT_CONFIG_FILE", str(REPO_ROOT / "configs" / "config.yaml"))
    if not path or not Path(path).is_file():
        return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, Any] = {}
    api = data.get("api") or {}                       # the team's existing `api:` block (host/port)
    if isinstance(api, dict):
        out.update({f"api_{k}": v for k, v in api.items() if k in ("host", "port")})
    section = data.get("agents") or {}
    if isinstance(section, dict):
        out.update(section)
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        return (init_settings, env_settings, dotenv_settings,
                InitSettingsSource(settings_cls, _yaml_defaults()), file_secret_settings)

    # ── server (same names as the team's .env.example) ──
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    default_asset_id: str = "esp32-conveyor-01"   # used when the ML JSON has no asset_id (single-ESP setup)
    agent_api_key: str = "dev-key"          # callers send it as header X-API-Key
    push_result_to_backend: bool = False
    evidence_window_minutes: int = 5
    min_confidence: float = 0.55            # below this -> "inconclusive"
    min_margin: float = 0.15                # winner must beat runner-up by this much
    # Safety gate: until the ML model is proven on NEW recordings, "stop_equipment" is downgraded to
    # the specific check for the fault (check_motor_temperature, ...). Set ML_VALIDATED=true only after that test passes.
    ml_validated: bool = False
    # Only these telemetry channels are judged by the rules (empty = all). Derived channels
    # (accel_mag, gyro_mag) can be listed too. Example for the real dataset: temperature_c,gyro_mag
    telemetry_channels: str = ""

    # ── LLM: provider-agnostic. Nothing hardcoded; see llm.py ──
    llm_enabled: bool = False               # false = deterministic rules only (no key needed)
    llm_provider: Optional[str] = None      # gemini | openai | anthropic | ollama | azure | ...
    llm_model: Optional[str] = None         # "gemini/<model>" or just "<model>" (+ llm_provider)
    llm_api_base: Optional[str] = None      # self-hosted / gateway only
    llm_api_key: Optional[str] = None       # optional; otherwise provider env var (GEMINI_API_KEY, ...)
    llm_temperature: float = 0.0
    llm_timeout_s: int = 30                 # per LLM call; on timeout the agent falls back to rules
    llm_retries: int = 2                    # extra attempts on 503 / 429 / timeout before the rules fallback
    llm_retry_backoff_s: float = 1.5        # pause before retry n = this * n
    llm_fallback_model: Optional[str] = None   # tried once if the main model keeps failing (e.g. gemini/gemini-flash-lite-latest)
    # Only "diagnosis" and "supervisor" can use an LLM. network / security / maintenance are rule-only by design.
    llm_agents: str = "diagnosis,supervisor"
    # Hard cap (seconds, TOTAL) for the streamed explanation. After it the template text is used.
    narrative_timeout_s: float = 20.0
    # When the diagnosis LLM and the rules disagree about the cause, merge both votes: the supervisor then sees
    # a small margin and escalates to a human (or the rules win if the LLM simply had no answer).
    llm_crosscheck: bool = True
    # "direct"  = ONE structured LLM call per agent (fast, default)
    # "crewai"  = crewAI Agent with a tool loop (slower: several LLM rounds)
    specialist_engine: Literal["direct", "crewai"] = "direct"
    llm_max_concurrent: int = 4             # how many agents may call the LLM at once (1 = one at a time)
    llm_model_diagnosis: Optional[str] = None
    llm_model_supervisor: Optional[str] = None

    # ── integrations (each: mock | http) ──
    backend_mode: BackendMode = "mock"
    backend_csv: str = "data/processed/final.csv"           # used when BACKEND_MODE=csv
    backend_url: str = "http://localhost:8000"        # same name as the team's .env.example
    backend_token: Optional[str] = None
    anomaly_mode: AnomalyMode = "inline"              # inline = ML result arrives inside the request
    anomaly_url: str = "http://localhost:8200"
    anomaly_token: Optional[str] = None
    anomaly_function: Optional[str] = None            # "package.module:function" when anomaly_mode=python
    network_mode: Mode = "mock"
    network_url: str = "http://localhost:8300"
    network_token: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

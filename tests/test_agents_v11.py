"""Tests for the v1.1 changes: ML JSON contract, evidence window, rule-only roles, one-call LLM, SSE streaming,
ML-validation gate, and the real-data replay."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents import agents as agents_mod
from src.agents import service as service_mod
from src.agents.clients import Clients
from src.agents.clients.mock import MockAnomalyClient, MockBackendClient, MockNetworkClient
from src.agents.graph import build_graph, run_incident
from src.agents.rules import rules_diagnosis
from src.agents.schemas import AlertIn, Cause
from src.agents.settings import Settings
from src.agents.testing import make_alert
from src.agents.tools import EvidencePack, parse_channels, summarize_telemetry
from src.api.main import create_app

H = {"X-API-Key": "test-key"}

# the exact JSON the ML team was asked to send
ML_JSON = {
    "incident_id": "INC-001", "asset_id": "esp32-conveyor-01", "timestamp": "2026-10-01T12:30:00Z",
    "window": {"start": "2026-10-01T12:29:50Z", "end": "2026-10-01T12:30:00Z", "n_samples": 50},
    "is_anomaly": True, "anomaly_score": 0.93, "raw_score": -0.21, "threshold": 0.5,
    "model": {"name": "isolation_forest", "version": "0.1", "feature_version": "f1"},
    "top_features": [{"name": "vibration_rms", "contribution": 0.42, "value": 2.6, "baseline": 1.0, "direction": "high"}],
}


# ───────── input contract ─────────
def test_ml_team_json_is_accepted_as_is():
    a = AlertIn(**ML_JSON)
    assert a.model_name == "isolation_forest" and a.model_version == "0.1" and a.feature_version == "f1"
    assert a.raw_score == -0.21 and a.window.n_samples == 50
    assert a.top_features[0].direction == "high" and a.top_features[0].baseline == 1.0
    assert a.evidence_end.isoformat().startswith("2026-10-01T12:30:00")


def test_asset_id_is_optional_and_falls_back_to_the_default():
    a = AlertIn(incident_id="x", anomaly_score=0.5)
    assert a.asset_id == "esp32-conveyor-01"
    assert AlertIn(incident_id="x", anomaly_score=0.5, asset_id="esp32-02").asset_id == "esp32-02"


def test_model_may_still_be_a_plain_string_and_unknown_fields_are_ignored():
    a = AlertIn(incident_id="x", anomaly_score=0.5, model="isolation_forest", brand_new_field=1)
    assert a.model_version == "isolation_forest"


def test_api_accepts_ml_json_without_asset_id(agent_client):
    body = {k: v for k, v in ML_JSON.items() if k != "asset_id"}
    r = agent_client.post("/api/v1/incident/analyze?view=full", json=body, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert d["asset_id"] == "esp32-conveyor-01" and d["ml_model_version"] == "0.1" and d["schema_version"] == "1.2"
    assert d["latency_ms"] is not None and set(d["timings_ms"]) >= {"collect_evidence", "diagnosis", "supervisor"}
    assert {v["role"] for v in d["votes"]} == {"diagnosis", "network", "security", "maintenance"}


# ───────── evidence window ends at the ML timestamp ─────────
class RecordingBackend(MockBackendClient):
    def __init__(self):
        super().__init__()
        self.ends = []

    def get_telemetry(self, asset_id, minutes, end=None):
        self.ends.append(end)
        return super().get_telemetry(asset_id, minutes, end)


def _run_with(alert, agent_settings):
    b = RecordingBackend()
    run_incident(build_graph(agent_settings, Clients(b, MockAnomalyClient(), MockNetworkClient())), alert)
    return b.ends


def test_backend_is_asked_for_the_window_ending_at_the_ml_timestamp(agent_settings):
    a = AlertIn(incident_id="i", anomaly_score=0.9, timestamp="2026-10-01T12:30:00Z")
    assert _run_with(a, agent_settings)[0].startswith("2026-10-01T12:30:00")


def test_window_end_wins_over_timestamp(agent_settings):
    body = {**ML_JSON, "timestamp": "2026-10-01T12:31:00Z"}
    assert _run_with(AlertIn(**body), agent_settings)[0].startswith("2026-10-01T12:30:00")


# ───────── telemetry summary on real-world data ─────────
def test_nan_and_null_values_are_ignored_not_poisoning_the_stats():
    samples = [{"ts": i, "temperature_c": (float("nan") if i % 10 == 0 else 26.5 + i * 0.01)} for i in range(60)]
    ch = summarize_telemetry({"sampling_interval_s": 1, "window_minutes": 1, "samples": samples})["channels"]["temperature_c"]
    assert ch["invalid_ratio"] == 0.1 and 26 < ch["mean"] < 28


def test_mostly_null_channel_is_reported_as_failing_sensor():
    samples = [{"ts": i, "temperature_c": None if i < 40 else 26.5} for i in range(60)]
    s = summarize_telemetry({"sampling_interval_s": 1, "window_minutes": 1, "samples": samples})
    assert "temperature_c" in s["implausible"]


def test_gyro_and_accel_magnitudes_are_derived_and_allowlist_filters_channels():
    samples = [{"ts": i, "gyro_x_rad_s": 3.0, "gyro_y_rad_s": 4.0, "gyro_z_rad_s": 0.0, "current_ma": 200.0} for i in range(20)]
    s = summarize_telemetry({"sampling_interval_s": 0.2, "window_minutes": 1 / 15, "samples": samples}, ("gyro_mag",))
    assert list(s["channels"]) == ["gyro_mag"] and s["channels"]["gyro_mag"]["mean"] == 5.0
    assert parse_channels(" temperature_c , gyro_mag ") == ("temperature_c", "gyro_mag") and parse_channels("") == ()


def test_short_window_does_not_call_a_steady_reading_a_dead_sensor():
    samples = [{"ts": i, "temperature_c": 26.5} for i in range(50)]            # 10 s at 5 Hz
    assert summarize_telemetry({"sampling_interval_s": 0.2, "window_minutes": 1 / 6, "samples": samples})["flatlined"] == []
    long = [{"ts": i, "temperature_c": 26.5} for i in range(60)]               # 60 s: now it IS suspicious
    assert summarize_telemetry({"sampling_interval_s": 1, "window_minutes": 1, "samples": long})["flatlined"] == ["temperature_c"]


def test_temperature_is_judged_by_degrees_not_ratio():
    """26.5 -> 33.2 C is only 1.25x, but +6.7 C is overheating."""
    samples = [{"ts": i, "temperature_c": 26.5 if i < 30 else 33.2} for i in range(60)]
    raw = {"sampling_interval_s": 1, "window_minutes": 1, "baseline": {"temperature_c": 26.5}, "samples": samples}
    pack = EvidencePack(asset_id="a", telemetry=summarize_telemetry(raw), refs={"telemetry:a:last1m": "t"})
    rep = rules_diagnosis(make_alert("a", 0.9), pack)
    assert rep.scores[Cause.physical_fault] >= 0.9 and any("temperature_c" in f and "above" in f for f in rep.findings)


# ───────── real recordings (sample of the team's CSV) ─────────
SAMPLE = Path(__file__).parent / "data" / "aegisot_sample.csv"


def test_real_recording_replay_normal_is_quiet_and_faults_are_caught():
    from src.agents.scripts.replay_csv import replay
    r = replay(pd.read_csv(SAMPLE), parse_channels("temperature_c,gyro_mag"), window=50, stride=10)
    assert r["normal"]["physical"] == 0                                         # no false alarms on normal data
    assert r["overheated"]["physical"] / r["overheated"]["windows"] >= 0.8
    assert r["vibration_issue"]["physical"] / r["vibration_issue"]["windows"] >= 0.8
    assert all(v["sensor"] == 0 for v in r.values())                            # steady temperature is not a dead sensor


# ───────── rule-only roles ─────────
def test_network_security_maintenance_cannot_use_the_llm():
    from src.agents.roles import LLM_ROLES, SPECIALISTS
    assert LLM_ROLES == ("diagnosis", "supervisor")
    assert [n for n, r in SPECIALISTS.items() if r.uses_llm] == ["diagnosis"]


# ───────── ML validation gate ─────────
def test_stop_equipment_is_downgraded_until_ml_is_validated(agent_clients):
    s = Settings(_env_file=None, anomaly_mode="mock", ml_validated=False)
    a = run_incident(build_graph(s, agent_clients), make_alert("esp32-conveyor-01", 0.93))
    assert a.likely_cause == Cause.physical_fault and a.recommended_action.value == "check_motor_vibration"
    assert not a.approval_required and a.ml_validated is False and any("ML_VALIDATED" in n for n in a.notes)
    s2 = Settings(_env_file=None, anomaly_mode="mock", ml_validated=True)
    assert run_incident(build_graph(s2, agent_clients), make_alert("esp32-conveyor-01", 0.93)).recommended_action.value == "stop_equipment"


# ───────── ONE-call LLM engine ─────────
GOOD = {"scores": [{"cause": "physical_fault", "score": 0.9}, {"cause": "network_issue", "score": 0.99}],
        "findings": ["vibration high"], "evidence_refs": ["telemetry:esp32-conveyor-01:last5m", "bogus:ref"],
        "checks_performed": ["telemetry_summary"]}


def _direct_settings(**kw):
    return Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True, **kw)


def test_direct_runner_makes_exactly_one_call_and_still_sanitizes(agent_clients, monkeypatch):
    calls = []
    monkeypatch.setattr(agents_mod, "complete_text",
                        lambda spec, system, user: calls.append(system) or "```json\n" + json.dumps(GOOD) + "\n```")
    s = _direct_settings(llm_agents="diagnosis")
    a = run_incident(build_graph(s, agent_clients), make_alert())
    assert len(calls) == 1 and "Machine Diagnosis Engineer" in calls[0]
    assert a.likely_cause == Cause.physical_fault and "bogus:ref" not in a.evidence_refs
    assert a.engine == "mixed" and "openai/fake" in a.llm_models_used
    assert any("dropped unknown evidence refs" in n for n in a.notes) and any("out-of-scope" in n for n in a.notes)


def test_direct_runner_bad_json_or_timeout_falls_back_to_rules(agent_clients, monkeypatch):
    for answer in ("sorry, I cannot do that", TimeoutError("timed out")):
        def fake(spec, system, user, answer=answer):
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(agents_mod, "complete_text", fake)
        a = run_incident(build_graph(_direct_settings(llm_agents="diagnosis"), agent_clients), make_alert())
        assert a.likely_cause == Cause.physical_fault and a.engine == "rules"
        assert any("LLM path failed" in n for n in a.notes)


# ───────── SSE streaming ─────────
def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_event_order_without_llm(agent_client):
    r = agent_client.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[:3] == ["accepted", "evidence_collected", "preliminary"]
    assert sorted(d["role"] for e, d in events if e == "vote") == ["diagnosis", "maintenance", "network", "security"]
    assert names[-2:] == ["decision", "done"] and "explanation_token" not in names
    pre = dict(events)["preliminary"]
    assert pre["final"] is False
    done = dict(events)["done"]["assessment"]
    assert done["likely_cause"] == "physical_fault" and done["recommended_action"] == "stop_equipment"
    assert agent_client.get("/api/v1/incident/INC-001", headers=H).status_code == 200      # stored like a normal run


def test_stream_tokens_only_rewrite_the_explanation(agent_settings, agent_clients, monkeypatch):
    s = agent_settings.model_copy(update={"llm_enabled": True, "llm_agents": "diagnosis,supervisor"})
    monkeypatch.setattr(service_mod, "stream_text", lambda spec, system, user: iter(["The ", "motor ", "is vibrating."]))

    def runner(role, alert, pack):                       # diagnosis "LLM" = the rules, labelled as llm
        from src.agents.schemas import SpecialistReport
        rep = agents_mod.SPECIALISTS[role].rules(alert, pack)
        return SpecialistReport(**{**rep.model_dump(), "engine": "llm", "model": "fake/m"}), []

    c = TestClient(create_app(s, agent_clients, runner))
    events = parse_sse(c.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H).text)
    names = [e for e, _ in events]
    assert names.index("decision") < names.index("explanation_token") < names.index("done")
    assert "".join(d["t"] for e, d in events if e == "explanation_token") == "The motor is vibrating."
    decision, done = dict(events)["decision"], dict(events)["done"]["assessment"]
    assert decision["explanation_pending"] is True
    for k in ("likely_cause", "recommended_action", "approval_required", "confidence"):      # decision never changes
        assert decision["assessment"][k] == done[k]
    assert done["explanation"] == "The motor is vibrating."


def test_stream_narrative_failure_keeps_the_template_and_finishes(agent_settings, agent_clients, monkeypatch):
    s = agent_settings.model_copy(update={"llm_enabled": True, "llm_agents": "supervisor"})

    def boom(spec, system, user):
        raise RuntimeError("llm down")
        yield
    monkeypatch.setattr(service_mod, "stream_text", boom)
    c = TestClient(create_app(s, agent_clients, runner=lambda *a: None))
    events = parse_sse(c.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H).text)
    assert events[-1][0] == "done" and "narrative failed" in " ".join(events[-1][1]["assessment"]["notes"])
    assert any(e == "note" for e, _ in events)


def test_stream_is_idempotent_and_validates_before_streaming(agent_client):
    agent_client.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H)
    again = parse_sse(agent_client.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H).text)
    assert [e for e, _ in again] == ["accepted", "done"] and again[-1][1]["cached"] is True
    assert agent_client.post("/api/v1/incident/analyze/stream", json={**ML_JSON, "is_anomaly": False}, headers=H).status_code == 422
    assert agent_client.post("/api/v1/incident/analyze/stream", json=ML_JSON).status_code == 401

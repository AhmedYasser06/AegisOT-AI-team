"""Schema 1.2: specific fault types, specific actions, no canned text, data-driven confidence, compact view,
LLM retry/fallback and the honest `llm` status."""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from src.agents import agents as agents_mod
from src.agents.clients import Clients
from src.agents.clients.mock import MockAnomalyClient, MockBackendClient, MockNetworkClient, _samples
from src.agents.enums import ActionId, Cause, FaultType
from src.agents.graph import build_graph, run_incident
from src.agents.llm import LLMSpec, complete_text, is_transient
from src.agents.policy import ALLOWLIST, choose_action, inspection_action, ml_gate
from src.agents.rules import rules_diagnosis
from src.agents.settings import Settings
from src.agents.signals import build_signals, corroborated, evidence_confidence
from src.agents.testing import make_alert
from src.agents.tools import EvidencePack, summarize_telemetry

H = {"X-API-Key": "test-key"}


def _graph(agent_settings, backend_cls, **settings_kw):
    s = agent_settings if not settings_kw else Settings(_env_file=None, anomaly_mode="mock", ml_validated=True, **settings_kw)
    return build_graph(s, Clients(backend_cls(), MockAnomalyClient(), MockNetworkClient()))


def _backend(scale: dict):
    class B(MockBackendClient):
        def get_telemetry(self, asset_id, minutes, end=None):
            d = super().get_telemetry(asset_id, minutes, end)
            d["samples"] = _samples(asset_id, minutes, fault_scale=scale)
            return d
    return B


# ───────── the fault is NAMED, and so is the check ─────────
@pytest.mark.parametrize("scale,fault,action,channel", [
    ({"temperature": 1.5}, "overheating", "check_motor_temperature", "temperature"),
    ({"vibration_rms": 3.0}, "abnormal_vibration", "check_motor_vibration", "vibration_rms"),
    ({"current": 1.8}, "electrical_fault", "check_power_supply", "current"),
    ({"speed": 0.4}, "speed_deviation", "check_motor_speed", "speed"),
])
def test_each_machine_fault_gets_its_own_name_and_check(agent_settings, scale, fault, action, channel):
    a = run_incident(_graph(agent_settings, _backend(scale)), make_alert("esp32-x", 0.7))
    assert a.likely_cause == Cause.physical_fault and a.fault_type.value == fault
    assert a.recommended_action.value == action and not a.approval_required
    assert a.affected_channels == [channel]
    assert any(s["channel"] == channel for s in a.signals)


def test_heat_wins_over_the_other_channels_it_drags_along(agent_settings):
    """Real overheating also drops rpm/voltage. The operator must read 'overheating', not 'speed deviation'."""
    a = run_incident(_graph(agent_settings, _backend({"temperature": 1.5, "speed": 0.3, "current": 1.8})),
                     make_alert("esp32-x", 0.7))
    assert a.fault_type == FaultType.overheating and a.recommended_action == ActionId.check_motor_temperature
    assert {"temperature", "speed", "current"} <= set(a.affected_channels)


def test_dead_temperature_sensor_gets_a_temperature_sensor_check(agent_graph):
    a = run_incident(agent_graph, make_alert("esp32-sensor-01", 0.8))
    assert a.fault_type == FaultType.sensor_fault and a.recommended_action == ActionId.recheck_temperature_sensor
    assert a.action_params == {"channels": ["temperature"]}


def test_other_causes_keep_their_names(agent_graph):
    assert run_incident(agent_graph, make_alert("esp32-conveyor-net", 0.7, incident_id="n")).fault_type == FaultType.network_issue
    assert run_incident(agent_graph, make_alert("esp32-rogue-01", 0.88, incident_id="r")).fault_type == FaultType.security_incident
    u = run_incident(agent_graph, make_alert("esp32-unknown", 0.3, incident_id="u"))
    assert u.fault_type == FaultType.unknown and u.status == "inconclusive"


def test_policy_every_action_listed_and_stop_downgrades_to_the_specific_check():
    assert set(ActionId) == set(ALLOWLIST)
    act, params, note = ml_gate(ActionId.stop_equipment, {}, False, FaultType.overheating, ["temperature_c"])
    assert act == ActionId.check_motor_temperature and params == {"channels": ["temperature_c"]} and "check_motor_temperature" in note
    assert inspection_action(FaultType.machine_fault)[0] == ActionId.inspect_equipment
    assert inspection_action(FaultType.sensor_fault, ["gyro_mag"])[0] == ActionId.request_sensor_recheck
    assert choose_action(Cause.physical_fault, 0.9, 0.9, FaultType.overheating)[0] == ActionId.stop_equipment


# ───────── no canned text ─────────
def test_policy_has_no_text_tables():
    import src.agents.policy as policy
    assert not hasattr(policy, "ACTION_TEXT") and not hasattr(policy, "DIAGNOSIS_TEXT")


def test_headline_and_steps_are_composed_from_the_measurements(agent_settings):
    a = run_incident(_graph(agent_settings, _backend({"temperature": 1.5, "current": 1.8})), make_alert("esp32-x", 0.8))
    assert a.llm.status == "disabled"
    assert a.headline.startswith("Overheating") and "temperature is" in a.headline
    assert a.diagnosis == a.headline                                   # old field = the same dynamic text
    assert any("motor temperature" in s and "temperature is" in s for s in a.recommended_solution)
    assert a.recommended_action_text == a.recommended_solution[0]
    different = run_incident(_graph(agent_settings, _backend({"speed": 0.4})), make_alert("esp32-x", 0.8, incident_id="i2"))
    assert different.headline != a.headline and different.recommended_solution != a.recommended_solution


def test_high_impact_action_is_the_first_step(agent_graph):
    a = run_incident(agent_graph, make_alert("esp32-conveyor-01", 0.93))
    assert a.recommended_action == ActionId.stop_equipment and a.approval_required
    assert "stop_equipment" in a.recommended_solution[0] and "approval" in a.recommended_solution[0]


# ───────── confidence follows the data, not the LLM's habit ─────────
def test_rules_confidence_moves_with_the_evidence():
    base = {"channels": {"temperature_c": {"recent_mean": 33.0, "baseline": 26.0, "delta_to_baseline": 7.0, "ratio_to_baseline": 1.27}}}
    sig = build_signals(base)
    assert evidence_confidence(0.6, sig) < evidence_confidence(0.9, sig)
    more = build_signals({"channels": {**base["channels"], "current_ma": {"recent_mean": 300.0, "baseline": 190.0,
                                                                         "delta_to_baseline": 110.0, "ratio_to_baseline": 1.58}}})
    assert corroborated(more) and not corroborated(sig)
    assert evidence_confidence(0.8, more) > evidence_confidence(0.8, sig)


def test_a_constant_llm_answer_no_longer_gives_a_constant_confidence(agent_clients):
    """The LLM answered physical 0.85 / sensor 0.15 for every incident. The final confidence must still differ."""
    def runner(role, alert, pack):
        rep = agents_mod.sanitize(role, agents_mod.SpecialistLLMOutput(
            scores=[{"cause": "physical_fault", "score": 0.85}, {"cause": "sensor_fault", "score": 0.15}],
            evidence_refs=list(pack.refs)), pack, "fake/m")
        return rep
    s = Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True,
                 llm_agents="diagnosis")
    g = build_graph(s, agent_clients, runner)
    low = run_incident(g, make_alert("esp32-conveyor-01", 0.55, incident_id="lo"))
    high = run_incident(g, make_alert("esp32-conveyor-01", 0.95, incident_id="hi"))
    assert low.llm.status == high.llm.status == "ok"
    assert low.confidence != high.confidence and low.confidence < high.confidence


# ───────── LLM wording is used, but validated ─────────
def _direct(monkeypatch, answer):
    monkeypatch.setattr(agents_mod, "complete_text", lambda spec, system, user: answer if isinstance(answer, str) else json.dumps(answer))
    return Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True,
                    llm_agents="diagnosis")


def test_llm_headline_and_steps_reach_the_result(agent_clients, monkeypatch):
    s = _direct(monkeypatch, {
        "scores": [{"cause": "physical_fault", "score": 0.9}, {"cause": "sensor_fault", "score": 0.1}],
        "findings": ["vibration_rms is 2.6x baseline"], "evidence_refs": ["telemetry:esp32-conveyor-01:last5m"],
        "fault_type": "abnormal_vibration", "headline": "Motor vibration 2.6x normal with current +70%",
        "solution_steps": ["Check the motor mounting bolts", "Inspect the bearings", "  ", "Verify belt alignment"]})
    a = run_incident(build_graph(s, agent_clients), make_alert())
    assert a.headline == "Motor vibration 2.6x normal with current +70%"
    assert a.recommended_solution[-3:] == ["Check the motor mounting bolts", "Inspect the bearings", "Verify belt alignment"]
    assert a.llm.status == "ok" and a.llm.model == "openai/fake" and a.llm.error is None


def test_llm_cannot_choose_the_fault_type_or_smuggle_junk(agent_clients, monkeypatch):
    """The rules say vibration. An LLM that says 'overheating' (or invents a type) changes nothing."""
    s = _direct(monkeypatch, {"scores": [{"cause": "physical_fault", "score": 0.9}],
                              "evidence_refs": ["telemetry:esp32-conveyor-01:last5m"], "fault_type": "teleportation",
                              "solution_steps": ["x" * 500] * 9})
    a = run_incident(build_graph(s, agent_clients), make_alert())
    assert a.fault_type == FaultType.abnormal_vibration
    assert any("unknown fault_type" in n for n in a.notes)
    steps = [x for x in a.recommended_solution if x.startswith("x")]
    assert len(steps) <= 5 and all(len(x) <= 240 for x in steps)
    s2 = _direct(monkeypatch, {"scores": [{"cause": "physical_fault", "score": 0.9}], "fault_type": "network_issue",
                               "evidence_refs": ["telemetry:esp32-conveyor-01:last5m"]})
    b = run_incident(build_graph(s2, agent_clients), make_alert(incident_id="b"))
    assert b.fault_type == FaultType.abnormal_vibration and any("out-of-scope fault_type" in n for n in b.notes)


def test_failed_llm_is_reported_and_the_answer_is_still_complete(agent_clients, monkeypatch):
    def boom(spec, system, user):
        raise RuntimeError('503 UNAVAILABLE. {"message": "This model is currently experiencing high demand"}')
    monkeypatch.setattr(agents_mod, "complete_text", boom)
    s = Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True,
                 llm_agents="diagnosis")
    a = run_incident(build_graph(s, agent_clients), make_alert())
    assert a.engine == "rules" and a.llm.status == "fallback_rules" and "high demand" in a.llm.error
    assert a.llm.model is None and a.headline and a.recommended_solution and a.fault_type == FaultType.abnormal_vibration


def test_diagnosis_prompt_carries_the_measured_deviations_and_the_rules(agent_clients, monkeypatch):
    seen = []
    monkeypatch.setattr(agents_mod, "complete_text", lambda spec, system, user: seen.append(user) or "{}")
    s = Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", llm_agents="diagnosis")
    run_incident(build_graph(s, agent_clients), make_alert())
    p = seen[0]
    assert "MEASURED DEVIATIONS" in p and "vibration_rms is" in p and "fault_type" in p and "headline" in p
    assert "never reuse example numbers" in p and "sensor_fault must be low" in p


# ───────── LLM retry / fallback model ─────────
def _fake_litellm(monkeypatch, outcomes):
    calls = []

    def completion(**kw):
        calls.append(kw["model"])
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=o))])
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setattr("src.agents.llm.time.sleep", lambda s: None)
    return calls


def _spec(**kw):
    return LLMSpec(model="gemini/a", api_base=None, api_key=None, temperature=0.0, timeout_s=5, **kw)


def test_temporary_errors_are_retried_then_succeed(monkeypatch):
    calls = _fake_litellm(monkeypatch, [RuntimeError("503 high demand"), RuntimeError("429 RESOURCE_EXHAUSTED"), "ok!"])
    assert complete_text(_spec(retries=2), "s", "u") == "ok!" and len(calls) == 3


def test_permanent_errors_are_not_retried(monkeypatch):
    calls = _fake_litellm(monkeypatch, [RuntimeError("401 invalid API key"), "never"])
    with pytest.raises(RuntimeError, match="invalid API key"):
        complete_text(_spec(retries=3), "s", "u")
    assert len(calls) == 1 and not is_transient(RuntimeError("404 model not found"))


def test_fallback_model_is_tried_once_and_the_original_error_is_kept(monkeypatch):
    calls = _fake_litellm(monkeypatch, [RuntimeError("503 a"), RuntimeError("503 a"), "from fallback"])
    assert complete_text(_spec(retries=1, fallback_model="gemini/b"), "s", "u") == "from fallback" and calls == ["gemini/a", "gemini/a", "gemini/b"]
    calls = _fake_litellm(monkeypatch, [RuntimeError("503 first"), RuntimeError("503 fb")])
    with pytest.raises(RuntimeError, match="503 first"):
        complete_text(_spec(retries=0, fallback_model="gemini/b"), "s", "u")


# ───────── the HTTP views ─────────
BODY = {"incident_id": "c-1", "asset_id": "esp32-conveyor-01", "anomaly_score": 0.93, "model": "iforest-0.1"}
DUPLICATES = ("findings", "votes", "evidence_refs", "timings_ms", "checks_performed", "alert_id", "schema_version")


def test_default_answer_is_the_compact_one(agent_client):
    d = agent_client.post("/api/v1/incident/analyze", json=BODY, headers=H).json()
    assert not any(k in d for k in DUPLICATES)
    for k in ("fault_type", "headline", "confidence", "summary", "evidence", "recommended_solution", "action", "severity", "llm"):
        assert k in d
    assert d["fault_type"] == "abnormal_vibration" and d["action"]["id"] == "stop_equipment" and d["action"]["approval_required"]
    assert d["evidence"][0]["channel"] == "vibration_rms" and {"value", "baseline", "change", "direction"} <= set(d["evidence"][0])
    assert any(c.startswith("network:") for c in d["other_checks"])
    assert len(json.dumps(d)) < len(json.dumps(agent_client.post("/api/v1/incident/analyze?view=full", json=BODY, headers=H).json())) / 2


def test_full_view_keeps_everything_for_the_backend(agent_client):
    d = agent_client.post("/api/v1/incident/analyze?view=full", json=BODY, headers=H).json()
    for k in (*DUPLICATES, "fault_type", "recommended_solution", "signals", "llm", "reason", "severity"):
        assert k in d
    assert agent_client.post("/api/v1/incident/analyze?view=banana", json=BODY, headers=H).status_code == 422


def test_get_and_list_follow_the_view(agent_client):
    agent_client.post("/api/v1/incident/analyze", json=BODY, headers=H)
    assert "action" in agent_client.get("/api/v1/incident/c-1", headers=H).json()
    assert "recommended_action" in agent_client.get("/api/v1/incident/c-1?view=full", headers=H).json()
    assert "action" in agent_client.get("/api/v1/incidents", headers=H).json()[0]
    assert "recommended_action" in agent_client.get("/api/v1/incidents?view=full", headers=H).json()[0]


def test_inconclusive_answer_says_why(agent_client):
    d = agent_client.post("/api/v1/incident/analyze", json={**BODY, "incident_id": "u-1", "asset_id": "esp32-unknown", "anomaly_score": 0.3}, headers=H).json()
    assert d["status"] == "inconclusive" and d["reason"] and d["fault_type"] == "unknown"
    assert d["action"]["id"] == "escalate_to_human" and d["recommended_solution"]


def test_old_diagnosis_endpoint_still_works(agent_client):
    d = agent_client.post("/api/v1/diagnosis/analyze", json={**BODY, "incident_id": "d-1"}, headers=H).json()
    assert d["action_id"] == "stop_equipment" and d["diagnosis"] and d["recommended_action"]


# ───────── real recordings ─────────
SAMPLE = Path(__file__).parent / "data" / "aegisot_sample.csv"


def test_real_recording_faults_get_the_right_name():
    from src.agents.scripts.replay_csv import baseline_from, to_samples
    df = pd.read_csv(SAMPLE)
    base = baseline_from(df[df.condition == "normal"])
    names = {}
    for cond, d in df.groupby("condition", sort=False):
        d = d.reset_index(drop=True)
        found = []
        for st in range(0, len(d) - 50 + 1, 10):
            raw = {"sampling_interval_s": 0.2, "window_minutes": 50 / 5 / 60, "baseline": base, "samples": to_samples(d.iloc[st:st + 50])}
            pack = EvidencePack(asset_id="r", telemetry=summarize_telemetry(raw), refs={"telemetry:r:last1m": "w"})
            found.append(rules_diagnosis(make_alert("r", 0.9), pack).fault_type)
        names[cond] = found
    assert all(f is None for f in names["normal"])
    assert names["overheated"].count(FaultType.overheating) / len(names["overheated"]) >= 0.8
    assert names["vibration_issue"].count(FaultType.abnormal_vibration) / len(names["vibration_issue"]) >= 0.8


# ───────── severity follows the measured deviation; steps follow the main fault ─────────
def test_severity_follows_how_far_the_data_moved():
    from src.agents.supervisor import severity_of
    small = [{"group": "temperature", "strength": 2.2}]                       # +6.6 C
    assert severity_of(0.77, small, False) == "medium"
    big_vib = [{"group": "vibration", "strength": 9.4}]                       # +470 %
    assert severity_of(0.96, big_vib, False) == "critical"
    assert severity_of(0.95, [], False) == "medium"                           # ML alone, nothing measured
    assert severity_of(0.3, small, True) == "high"                            # a proposed stop is never low


def test_llm_steps_are_ordered_by_the_main_fault():
    from src.agents.enums import FaultType
    from src.agents.supervisor import order_steps
    sig = [{"group": "temperature", "strength": 2.2}, {"group": "speed", "strength": 1.6}]
    steps = ["Inspect the motor for binding causing the speed drop to 47 RPM.",
             "Check the power supply wiring.",
             "Inspect motor cooling and ventilation for the 6.6 C temperature rise."]
    out = order_steps(steps, FaultType.overheating, sig)
    assert "cooling" in out[0] and "binding" in out[1] and "power supply" in out[2]


def test_prompt_limits_the_llm_to_the_measured_deviations():
    from src.agents.agents import DIAGNOSIS_BRIEF
    assert "ONLY the channels listed under" in DIAGNOSIS_BRIEF and "do NOT write a step about it" in DIAGNOSIS_BRIEF


def test_template_summary_does_not_repeat_the_headline():
    from src.agents.enums import ActionId, Cause
    from src.agents.supervisor import default_explanation
    from tests.test_agents import make_alert
    out = default_explanation(make_alert(), Cause.physical_fault, 0.92, ActionId.check_motor_temperature, False, False,
                              [], "Motor overheating at 46 C", "Inspect motor cooling.")
    assert "Motor overheating" not in out and "Do first: Inspect motor cooling." in out


def test_narrative_prompt_does_not_get_the_headline():
    from src.agents.graph import NARRATIVE_FACTS, NARRATIVE_SYSTEM
    assert "headline" not in NARRATIVE_FACTS and "do NOT restate it" in NARRATIVE_SYSTEM


def test_prompts_ask_for_plain_names_and_one_number_per_measurement():
    from src.agents.agents import DIAGNOSIS_BRIEF
    from src.agents.graph import NARRATIVE_SYSTEM
    assert "Never write raw field names" in DIAGNOSIS_BRIEF and "state each measurement ONCE" in DIAGNOSIS_BRIEF
    assert "Do not state the confidence" in NARRATIVE_SYSTEM

"""LLM-vs-rules cross-check and the hard deadline for the streamed explanation."""
import time

import pytest
from fastapi.testclient import TestClient

from src.agents import service as service_mod
from src.agents.clients import Clients
from src.agents.crosscheck import crosscheck
from src.agents.graph import build_graph, run_incident
from src.agents.llm import iter_with_deadline
from src.agents.schemas import Cause, SpecialistReport
from src.agents.settings import Settings
from src.agents.testing import make_alert
from src.api.main import create_app
from tests.test_agents_v11 import H, ML_JSON, parse_sse


def _report(engine, **scores):
    return SpecialistReport(role="diagnosis", engine=engine, scores={Cause(k): v for k, v in scores.items()},
                            findings=[f"{engine} finding"], evidence_refs=[], checks_performed=[],
                            model="fake/m" if engine == "llm" else None)


# ───────── cross-check unit ─────────
def test_crosscheck_agreement_changes_nothing():
    llm, rules = _report("llm", physical_fault=0.85, sensor_fault=0.1), _report("rules", physical_fault=0.95, sensor_fault=0.05)
    out, note = crosscheck(llm, rules)
    # agreement: the score is a 70/30 blend of rules and LLM, so it follows the data, not the LLM's habit
    assert note is None and out.scores[Cause.physical_fault] == round(0.7 * 0.95 + 0.3 * 0.85, 3)


def test_crosscheck_rules_see_nothing_so_the_llm_is_accepted():
    llm, rules = _report("llm", physical_fault=0.1, sensor_fault=0.9), _report("rules", physical_fault=0.05, sensor_fault=0.05)
    out, note = crosscheck(llm, rules)
    assert out.scores == llm.scores and note is None


def test_crosscheck_caps_llm_claim_when_telemetry_is_all_normal():
    llm = _report("llm", physical_fault=0.9, sensor_fault=0.1)
    rules = _report("rules", physical_fault=0.05, sensor_fault=0.05).model_copy(
        update={"checks_performed": ["telemetry_all_within_baseline"]})
    out, note = crosscheck(llm, rules)
    assert out.scores[Cause.physical_fault] == 0.3 and "capped" in note


def test_crosscheck_confident_disagreement_merges_the_votes():
    llm, rules = _report("llm", physical_fault=0.05, sensor_fault=0.95), _report("rules", physical_fault=0.955, sensor_fault=0.05)
    out, note = crosscheck(llm, rules)
    assert out.scores[Cause.physical_fault] == 0.955 and out.scores[Cause.sensor_fault] == 0.95
    assert "LLM says sensor_fault" in note and any("[rules check]" in f for f in out.findings)


# ───────── cross-check through the whole graph ─────────
def _runner(**scores):
    def run(role, alert, pack):
        return _report("llm", **scores), []
    return run


def _settings(**kw):
    return Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True,
                    llm_agents="diagnosis", **kw)


def test_llm_disagreeing_with_rules_ends_in_escalation_not_a_silent_downgrade(agent_clients):
    """esp32-hot-01: ONLY the temperature moved (+48 C, everything else normal): a sensor fault is plausible, so the
    LLM's sensor vote is not capped. Rules say physical, LLM says sensor -> escalate."""
    a = run_incident(build_graph(_settings(), agent_clients, _runner(physical_fault=0.05, sensor_fault=0.95)),
                     make_alert("esp32-hot-01", 0.91))
    assert a.status == "inconclusive" and a.recommended_action.value == "escalate_to_human"
    assert any("LLM says sensor_fault" in n for n in a.notes)
    assert a.reason and "too close" in a.reason          # the operator is told WHY it is inconclusive


def test_sensor_vote_is_capped_when_other_channels_move_with_it(agent_clients):
    """conveyor-01: vibration AND current moved together. A broken sensor changes one channel, so an LLM
    'sensor_fault 0.95' (the flip that made a real overheating inconclusive) is capped and the fault stands."""
    a = run_incident(build_graph(_settings(), agent_clients, _runner(physical_fault=0.05, sensor_fault=0.95)),
                     make_alert())
    assert a.status == "complete" and a.likely_cause == Cause.physical_fault
    assert any("capped" in n for n in a.notes)


def test_llm_that_is_unsure_does_not_hide_a_rules_fault(agent_clients):
    a = run_incident(build_graph(_settings(), agent_clients, _runner(physical_fault=0.05, sensor_fault=0.1)), make_alert())
    assert a.likely_cause == Cause.physical_fault and any("rules vote was kept" in n for n in a.notes)


def test_crosscheck_can_be_switched_off(agent_clients):
    s = _settings(llm_crosscheck=False)
    a = run_incident(build_graph(s, agent_clients, _runner(physical_fault=0.05, sensor_fault=0.95)), make_alert())
    assert a.likely_cause == Cause.sensor_fault and not any("LLM says" in n for n in a.notes)


# ───────── deadline ─────────
def test_iter_with_deadline_passes_fast_streams_and_errors_through():
    assert list(iter_with_deadline(iter(["a", "b"]), 2)) == ["a", "b"]

    def bad():
        yield "a"
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        list(iter_with_deadline(bad(), 2))


def test_iter_with_deadline_gives_up_on_a_stalled_stream():
    def stalled():
        yield "a "
        time.sleep(3)
        yield "b"
    got, t0 = [], time.perf_counter()
    with pytest.raises(TimeoutError):
        for piece in iter_with_deadline(stalled(), 0.3):
            got.append(piece)
    assert got == ["a "] and time.perf_counter() - t0 < 2


def test_stream_stalled_explanation_falls_back_to_the_template(agent_settings, agent_clients, monkeypatch):
    s = agent_settings.model_copy(update={"llm_enabled": True, "llm_agents": "supervisor", "narrative_timeout_s": 0.3})

    def stalled(spec, system, user):
        yield "The "
        time.sleep(3)
        yield "never shown"
    monkeypatch.setattr(service_mod, "stream_text", stalled)
    c = TestClient(create_app(s, agent_clients, runner=lambda *a: None))
    t0 = time.perf_counter()
    events = parse_sse(c.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H).text)
    assert time.perf_counter() - t0 < 2.5
    done = events[-1]
    assert done[0] == "done" and "deadline" in " ".join(done[1]["assessment"]["notes"])
    assert "never shown" not in done[1]["assessment"]["explanation"] and done[1]["assessment"]["likely_cause"] == "physical_fault"

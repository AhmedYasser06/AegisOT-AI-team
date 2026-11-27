"""LLM path guardrails, tested with a fake runner (no API key needed)."""
from src.agents.agents import sanitize
from src.agents.settings import Settings
from src.agents.tools import collect_evidence
from src.agents.graph import build_graph, run_incident
from src.agents.schemas import Cause, CauseScore, SpecialistLLMOutput, SpecialistReport
from src.agents.testing import make_alert


def _pack(agent_clients):
    return collect_evidence(agent_clients, "esp32-conveyor-01", "esp32-conveyor-01:1")


def test_sanitize_drops_unknown_refs_and_out_of_scope_causes(agent_clients):
    pack = _pack(agent_clients)
    out = SpecialistLLMOutput(
        scores=[CauseScore(cause=Cause.network_issue, score=0.99),       # network agent only
                CauseScore(cause=Cause.physical_fault, score=0.9)],      # diagnosis may score this
        evidence_refs=["telemetry:esp32-conveyor-01:last5m", "made:up:ref"])
    rep, notes = sanitize("diagnosis", out, pack, "m")
    assert Cause.network_issue not in rep.scores
    assert rep.evidence_refs == ["telemetry:esp32-conveyor-01:last5m"]
    assert any("made:up:ref" in n for n in notes) and any("out-of-scope" in n for n in notes)


def test_confident_score_without_evidence_is_capped(agent_clients):
    out = SpecialistLLMOutput(scores=[CauseScore(cause=Cause.physical_fault, score=0.95)], evidence_refs=["fake"])
    rep, notes = sanitize("diagnosis", out, _pack(agent_clients), None)
    assert rep.scores[Cause.physical_fault] == 0.3 and any("capped" in n for n in notes)


def test_llm_failure_falls_back_to_rules(agent_settings, agent_clients):
    def boom(role, alert, pack):
        raise RuntimeError("llm down")
    a = run_incident(build_graph(agent_settings, agent_clients, runner=boom), make_alert())
    assert a.likely_cause == Cause.physical_fault and a.engine == "rules"
    assert sum("LLM path failed" in n for n in a.notes) == 1      # only diagnosis can use the LLM


def test_llm_path_cannot_invent_actions_or_skip_approval(agent_settings, agent_clients):
    """Even a runner that screams 'stop everything' only produces scores; action comes from policy."""
    def runner(role, alert, pack):
        assert role == "diagnosis"                                   # the other specialists never reach the LLM
        rep = SpecialistReport(role=role, engine="llm",
                               scores={Cause.physical_fault: 0.95, Cause.network_issue: 0.99},   # 2nd = out of scope
                               findings=["x"], evidence_refs=list(pack.refs)[:1], checks_performed=["c"], model="fake/model")
        return rep, []
    a = run_incident(build_graph(agent_settings, agent_clients, runner=runner), make_alert("esp32-hot-01", 0.9))
    assert a.likely_cause == Cause.physical_fault                    # supervisor ignored the out-of-scope network vote
    assert a.recommended_action.value == "stop_equipment" and a.approval_required and not a.auto_executable
    assert a.engine == "mixed" and "fake/model" in a.llm_models_used


def test_supervisor_narrative_failure_is_non_fatal(agent_clients):
    s = Settings(_env_file=None, agent_api_key="k", llm_enabled=True, llm_model="bogus/none", anomaly_mode="mock")
    def runner(role, alert, pack):
        from src.agents.roles import SPECIALISTS
        return SPECIALISTS[role].rules(alert, pack), []
    a = run_incident(build_graph(s, agent_clients, runner=runner), make_alert())
    assert a.likely_cause == Cause.physical_fault
    assert any("narrative failed" in n for n in a.notes)


# ───────── failure notes show the reason but never leak secrets ─────────
def test_failure_notes_include_reason_and_redact_secrets(agent_settings, agent_clients):
    from src.agents.llm import short_error

    err = RuntimeError("404 models/x not found: https://generativelanguage.googleapis.com/v1?key=AIzaSyA1234567890abcdefghijklmnop&alt=sse sk-abcdefghijklmnopqrstuvwx")
    text = short_error(err)
    assert "RuntimeError" in text and "not found" in text
    assert "AIza" not in text and "sk-abc" not in text and "key=***" in text

    def boom(role, alert, pack):
        raise ImportError("Google Gen AI native provider not available")
    a = run_incident(build_graph(agent_settings, agent_clients, runner=boom), make_alert())
    assert any("ImportError: Google Gen AI native provider" in n for n in a.notes)


# ───────── free-tier controls ─────────
def _llm_runner(calls, active=None):
    import threading, time
    from src.agents.roles import SPECIALISTS
    from src.agents.schemas import SpecialistReport

    def runner(role, alert, pack):
        calls.append(role)
        if active is not None:
            with active["lock"]:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.15)
            with active["lock"]:
                active["now"] -= 1
        rep = SPECIALISTS[role].rules(alert, pack)
        return SpecialistReport(**{**rep.model_dump(), "engine": "llm", "model": "fake/m"}), []
    return runner


def test_llm_agents_limits_which_roles_use_the_llm(agent_clients):
    s = Settings(_env_file=None, anomaly_mode="mock", llm_agents="diagnosis")
    calls = []
    a = run_incident(build_graph(s, agent_clients, runner=_llm_runner(calls)), make_alert())
    assert calls == ["diagnosis"]                    # network/security/maintenance + supervisor narrative: rules only
    assert a.engine == "mixed" and a.likely_cause == Cause.physical_fault
    assert not any("narrative" in n for n in a.notes)


def test_llm_agents_rejects_unknown_and_rule_only_roles(agent_clients):
    import pytest
    for bad in ("diagnosis,wizard", "network", "security", "maintenance"):     # rule-only by design
        with pytest.raises(ValueError):
            build_graph(Settings(_env_file=None, anomaly_mode="mock", llm_agents=bad), agent_clients)


def test_rule_only_roles_never_call_the_runner(agent_clients):
    calls = []
    s = Settings(_env_file=None, anomaly_mode="mock", llm_agents="diagnosis,supervisor")
    run_incident(build_graph(s, agent_clients, runner=_llm_runner(calls)), make_alert("esp32-rogue-01", 0.9))
    assert calls == ["diagnosis"]


def test_rate_limit_note_is_short_and_actionable():
    from src.agents.llm import short_error
    msg = "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota ' }}" * 5
    text = short_error(RuntimeError(msg))
    assert "rate limit" in text and "LLM_AGENTS" in text and len(text) < 330

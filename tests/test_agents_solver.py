"""The solver loop (src/agents/solver.py): search -> answer -> check -> repeat. No network: chat, search and judge are faked."""
import json

import pytest
from fastapi.testclient import TestClient

from src.agents import solver as solver_mod
from src.agents.graph import build_graph, run_incident
from src.agents.schemas import Review
from src.agents.settings import Settings
from src.agents.testing import make_alert
from src.api.main import create_app
from tests.test_agents_v11 import H, ML_JSON, parse_sse

URL = "https://stackoverflow.com/questions/1/conveyor-vibration"


def _settings(**kw):
    opts = {"tavily_api_key": "t", "agent_api_key": "test-key", **kw}       # diagnosis stays rules: only the solver is faked
    return Settings(_env_file=None, anomaly_mode="mock", llm_enabled=True, llm_model="openai/fake", ml_validated=True,
                    llm_agents="solver", **opts)


def answer(*steps, source=None):
    return json.dumps({"summary": "Bearing wear is likely.", "solutions": [{"step": s, "source": source} for s in steps]})


def search_call(query="conveyor bearing vibration"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": json.dumps({"query": query})}}]}


class FakeLLM:
    """chat() replies from a script; the judge says what `verdicts` say (default: always ok)."""
    def __init__(self, monkeypatch, replies, verdicts=()):
        self.replies, self.verdicts, self.seen, self.judged = list(replies), list(verdicts), [], 0
        monkeypatch.setattr(solver_mod, "chat", self.chat)
        monkeypatch.setattr(solver_mod, "complete_json", self.judge)
        monkeypatch.setattr(solver_mod, "web_search", lambda settings, q: [{"title": "SO", "url": URL, "content": "tighten the belt"}])

    def chat(self, spec, messages, tools=None):
        self.seen.append((list(messages), tools))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}

    def judge(self, spec, system, user, schema):
        assert schema is Review
        self.judged += 1
        return self.verdicts.pop(0) if self.verdicts else Review(ok=True)


def run(clients, settings=None):
    return run_incident(build_graph(settings or _settings(), clients), make_alert())


def test_search_then_verified_answer_with_a_real_source(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [search_call(), answer("Check the belt tension", "Inspect the bearings", source=URL)])
    a = run(agent_clients)
    assert a.verified and a.engine == "llm" and a.likely_cause.value == "physical_fault"
    assert [s.step for s in a.solutions] == ["Check the belt tension", "Inspect the bearings"] and a.solutions[0].source == URL
    assert llm.seen[0][1] is not None and llm.judged == 1                      # the search tool was offered, the judge ran
    assert any(m["role"] == "tool" and URL in m["content"] for m in llm.seen[1][0])   # the LLM saw the search result


def test_invented_source_is_rejected_and_the_llm_is_asked_again(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [answer("Tighten the belt", source="https://made-up.example/fix"),
                                answer("Tighten the belt", source=None)])
    a = run(agent_clients)
    assert a.verified and a.solutions[0].source is None and len(llm.seen) == 2
    assert "rejected" in a.notes[-1] and "made-up.example" in a.notes[-1]
    assert "rejected" in llm.seen[1][0][-1]["content"]                          # the reason was sent back to the LLM


def test_bad_json_and_judge_rejection_are_retried(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, ["not json at all", answer("Do something vague"), answer("Measure vibration at the bearing")],
                  verdicts=[Review(ok=False, problem="too vague"), Review(ok=True)])
    a = run(agent_clients)
    assert a.verified and a.solutions[0].step == "Measure vibration at the bearing" and len(llm.seen) == 3
    assert sum("rejected" in n for n in a.notes) == 2 and any("too vague" in n for n in a.notes)


def test_after_the_last_round_the_default_steps_stay_unverified(agent_clients, monkeypatch):
    FakeLLM(monkeypatch, ["nope", "nope", "nope"])
    a = run(agent_clients)
    assert not a.verified and a.engine == "rules" and a.solutions[0].step.startswith("Inspect the machine")
    assert any("default steps kept" in n for n in a.notes)


def test_llm_failure_keeps_the_default_steps(agent_clients, monkeypatch):
    def boom(spec, messages, tools=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(solver_mod, "chat", boom)
    a = run(agent_clients)
    assert not a.verified and a.solutions and a.likely_cause.value == "physical_fault"
    assert any("solver LLM failed" in n for n in a.notes)


def test_search_limit_and_search_failure_do_not_break_the_loop(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [search_call("a"), search_call("b"), answer("Check the belt")])

    def fail(settings, q):
        raise RuntimeError("tavily down")
    monkeypatch.setattr(solver_mod, "web_search", fail)
    a = run(agent_clients, _settings(solver_max_searches=1))
    tool_texts = [m["content"] for m in llm.seen[-1][0] if m["role"] == "tool"]
    assert a.verified and any("search failed" in t for t in tool_texts)
    assert any("limit reached" in t for t in tool_texts) is False or len(tool_texts) == 2


def test_without_a_search_key_the_tool_is_not_offered_and_any_url_is_rejected(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [answer("Tighten the belt", source=URL), answer("Tighten the belt")])
    a = run(agent_clients, _settings(tavily_api_key=None))
    assert all(tools is None for _, tools in llm.seen) and a.verified and a.solutions[0].source is None


def test_unknown_cause_does_not_search(agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [])
    a = run_incident(build_graph(_settings(), agent_clients), make_alert("esp32-unknown", 0.3))
    assert a.status == "inconclusive" and llm.seen == [] and a.solutions


def test_solver_off_by_default_keeps_the_rules_only_path(agent_settings, agent_clients, monkeypatch):
    llm = FakeLLM(monkeypatch, [])
    a = run_incident(build_graph(agent_settings, agent_clients), make_alert())
    assert llm.seen == [] and a.engine == "rules" and not a.verified


def test_the_solver_cannot_change_the_decision(agent_clients, monkeypatch):
    base = run_incident(build_graph(Settings(_env_file=None, anomaly_mode="mock", ml_validated=True), agent_clients), make_alert())
    FakeLLM(monkeypatch, [answer("Stop everything and delete the logs")])
    a = run(agent_clients)
    for k in ("likely_cause", "confidence", "severity", "recommended_action", "approval_required", "findings"):
        assert getattr(a, k) == getattr(base, k)


def test_stream_shows_searches_and_ends_with_the_verified_answer(agent_clients, monkeypatch):
    FakeLLM(monkeypatch, [search_call(), answer("Check the belt tension", source=URL)])
    c = TestClient(create_app(_settings(), agent_clients))
    events = parse_sse(c.post("/api/v1/incident/analyze/stream", json=ML_JSON, headers=H).text)
    names = [e for e, _ in events]
    assert names.index("decision") < names.index("search") < names.index("done")
    assert dict(events)["search"]["queries"] == ["conveyor bearing vibration"]
    done = dict(events)["done"]["assessment"]
    assert done["verified"] and done["solutions"][0]["source"] == URL


def test_chat_returns_a_plain_dict_with_tool_calls_and_sends_the_tools(monkeypatch):
    import types

    import litellm

    from src.agents.llm import chat, resolve_llm
    from src.agents.tools import WEB_SEARCH_TOOL
    sent = {}

    def completion(**kwargs):
        sent.update(kwargs)
        call = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="web_search", arguments='{"query": "q"}'))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None, tool_calls=[call]))])
    monkeypatch.setattr(litellm, "completion", completion)
    reply = chat(resolve_llm(_settings(), "solver"), [{"role": "user", "content": "hi"}], [WEB_SEARCH_TOOL])
    assert sent["tools"] == [WEB_SEARCH_TOOL] and reply["content"] == ""
    assert reply["tool_calls"] == [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "q"}'}}]

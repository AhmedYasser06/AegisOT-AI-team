"""Runs the REAL crewAI + LangGraph code against a local fake OpenAI-compatible server.
No API key, no internet. First call -> the 'LLM' requests a tool; second call -> final answer.
This validates wiring (Agent/Task/Crew, tools, structured output, sanitisation), not model quality."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.agents.clients import build_clients
from src.agents.settings import Settings
from src.agents.graph import build_graph, run_incident
from src.agents.schemas import Cause
from src.agents.testing import make_alert

A = "esp32-conveyor-01"
# per-role canned answers. The diagnosis agent also tries to score network_issue (out of its scope) and cite a fake ref.
BY_ROLE = {
    "Machine Diagnosis Engineer": {"scores": [{"cause": "physical_fault", "score": 0.9}, {"cause": "network_issue", "score": 0.99}],
                                   "findings": ["vibration high"], "evidence_refs": [f"telemetry:{A}:last5m", "bogus:ref"],
                                   "checks_performed": ["telemetry_summary"]},
    "OT Network Analyst": {"scores": [{"cause": "network_issue", "score": 0.05}], "findings": ["network healthy"],
                           "evidence_refs": [f"network:{A}:health"], "checks_performed": ["network_health"]},
    "OT Security Analyst": {"scores": [{"cause": "security_incident", "score": 0.05}], "findings": ["clean"],
                           "evidence_refs": [f"security:{A}:events60m"], "checks_performed": ["security_events"]},
    "Maintenance Planner": {"scores": [{"cause": "physical_fault", "score": 0.6}], "findings": ["health 0.55"],
                            "evidence_refs": [f"maintenance:{A}"], "checks_performed": ["maintenance_record"]},
}
CALLS = {"tool_calls": 0, "final": 0}


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
        if body.get("tools") and not has_tool_result and "response_format" not in body:
            CALLS["tool_calls"] += 1
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": body["tools"][0]["function"]["name"], "arguments": "{}"}}]}
            finish = "tool_calls"
        else:
            CALLS["final"] += 1
            system = " ".join(str(m.get("content")) for m in body["messages"] if m.get("role") == "system")
            final = next((v for k, v in BY_ROLE.items() if k in system), BY_ROLE["Machine Diagnosis Engineer"])
            msg, finish = {"role": "assistant", "content": json.dumps(final)}, "stop"
        out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                          "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def fake_llm():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.mark.parametrize("engine", ["crewai", "direct"])
def test_full_workflow_with_real_llm_code(fake_llm, monkeypatch, engine):
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    CALLS["tool_calls"] = 0
    s = Settings(_env_file=None, llm_enabled=True, llm_model="openai/fake-model", specialist_engine=engine, ml_validated=True,
                 anomaly_mode="mock", llm_api_base=fake_llm, llm_api_key="x", llm_model_supervisor="openai/fake-supervisor")
    a = run_incident(build_graph(s, build_clients(s)), make_alert())
    # guardrails held even though the fake LLM claimed network_issue=0.99 for the diagnosis agent
    assert a.likely_cause == Cause.physical_fault and a.status == "complete"
    assert a.engine in ("llm", "mixed") and "openai/fake-model" in a.llm_models_used
    assert "bogus:ref" not in a.evidence_refs
    assert a.recommended_action.value == "stop_equipment" and a.approval_required
    if engine == "crewai":
        assert CALLS["tool_calls"] >= 1, "crewAI never called a tool"
    else:
        assert CALLS["tool_calls"] == 0, "the direct engine must answer in ONE call (no tool loop)"

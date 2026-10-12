"""LangGraph workflow (the fixed, supervised pipeline).

    collect_evidence ─┬─> diagnosis ──┐
                      ├─> network ────┤
                      ├─> security ───┼─> supervisor ─> END
                      └─> maintenance ┘

* collect_evidence : deterministic fetch (backend / anomaly / network) -> EvidencePack with evidence refs
* specialists      : crewAI crew if LLM_ENABLED, else rules; any failure -> rules fallback
* supervisor       : merge + policy gate + allowlist + inconclusive logic + optional LLM narrative

Run the demo:   python -m src.agents.graph            (5 mock scenarios, no LLM / key needed)
                python -m src.agents.graph --asset esp32-conveyor-01 --score 0.93 --json
"""
from __future__ import annotations

import argparse
import json
import operator
import re
import textwrap
import threading
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import CrewRunner, Runner
from .clients import Clients, build_clients
from .llm import complete_text, resolve_llm
from .rules import RULES
from .schemas import AlertIn, IncidentAssessment, SpecialistReport
from .settings import Settings, get_settings
from .supervisor import build_assessment
from .tools import EvidencePack, collect_evidence

ROLES = ("diagnosis", "network", "security", "maintenance")

_SECRET = re.compile(r"(key=)[^&\s'\"]+|AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{16,}", re.I)


def why(e: Exception, limit: int = 180) -> str:
    """Short, secret-free description of an exception for the `notes` field."""
    msg = " ".join(str(e).split())
    if re.search(r"429|RESOURCE_EXHAUSTED|RateLimit|quota", msg, re.I):
        return (f"{type(e).__name__}: rate limit / quota exceeded (HTTP 429). Free tiers allow only a few requests per "
                f"minute per model: use different models per agent, LLM_AGENTS, LLM_MAX_CONCURRENT=1, or a paid key")
    msg = _SECRET.sub(lambda m: f"{m.group(1)}***" if m.group(1) else "***", msg)
    return f"{type(e).__name__}: {msg[:limit]}"


NARRATIVE_SYSTEM = (
    "You write a 2-3 sentence incident explanation for a plant operator. Use ONLY the JSON facts provided. "
    "Do not add causes, actions or numbers that are not in the JSON. Plain text, no markdown.")


class State(TypedDict, total=False):
    alert: AlertIn
    pack: EvidencePack
    reports: Annotated[list[SpecialistReport], operator.add]
    notes: Annotated[list[str], operator.add]
    assessment: IncidentAssessment


def build_graph(settings: Settings, clients: Clients, runner: Runner | None = None):
    if runner is None and settings.llm_enabled:
        runner = CrewRunner(settings, clients)
    llm_roles = {r.strip() for r in settings.llm_agents.split(",") if r.strip()}
    unknown = llm_roles - set(ROLES) - {"supervisor"}
    if unknown:
        raise ValueError(f"LLM_AGENTS has unknown role(s) {sorted(unknown)}; valid: {[*ROLES, 'supervisor']}")
    gate = threading.Semaphore(max(1, settings.llm_max_concurrent))

    def collect(state: State):
        a = state["alert"]
        return {"pack": collect_evidence(clients, a.asset_id, a.alert_id, settings.evidence_window_minutes,
                                         a.inline_anomaly(), settings.anomaly_mode == "inline")}

    def make_specialist(role: str):
        def node(state: State):
            alert, pack = state["alert"], state["pack"]
            note = None
            if runner is not None and role in llm_roles:
                try:
                    with gate:                      # limit simultaneous LLM calls (rate limits)
                        report, notes = runner(role, alert, pack)
                    return {"reports": [report], "notes": notes}
                except Exception as e:  # LLM down, invalid output, tool loop failure, ...
                    note = f"{role}: LLM path failed ({why(e)}); deterministic rules used."
            return {"reports": [RULES[role](alert, pack)], "notes": [note] if note else []}
        node.__name__ = f"{role}_node"
        return node

    def supervisor(state: State):
        alert, pack, reports, notes = state["alert"], state["pack"], state["reports"], list(state.get("notes", []))
        explanation, extra_models = None, []
        if runner is not None and "supervisor" in llm_roles:
            # build the validated facts first, then let the LLM only rephrase them
            draft = build_assessment(alert, pack, reports, settings, notes)
            try:
                spec = resolve_llm(settings, "supervisor")
                facts = draft.model_dump(include={"asset_id", "status", "likely_cause", "confidence", "alternatives",
                                                  "recommended_action", "approval_required", "findings"}, mode="json")
                explanation = complete_text(spec, NARRATIVE_SYSTEM, json.dumps(facts)) or None
                extra_models = [spec.model] if explanation else []
            except Exception as e:
                notes.append(f"supervisor narrative failed ({why(e)}); template used.")
        return {"assessment": build_assessment(alert, pack, reports, settings, notes, explanation, extra_models)}

    g = StateGraph(State)
    g.add_node("collect_evidence", collect)
    for r in ROLES:
        g.add_node(r, make_specialist(r))
        g.add_edge("collect_evidence", r)
        g.add_edge(r, "supervisor")
    g.add_node("supervisor", supervisor)
    g.add_edge(START, "collect_evidence")
    g.add_edge("supervisor", END)
    return g.compile()


def run_incident(graph, alert: AlertIn) -> IncidentAssessment:
    return graph.invoke({"alert": alert, "reports": [], "notes": []})["assessment"]


# ───────── CLI demo: python -m src.agents.graph ─────────
DEMO_CASES = [("esp32-conveyor-01", 0.93), ("esp32-conveyor-net", 0.70), ("esp32-rogue-01", 0.88),
              ("esp32-sensor-01", 0.80), ("esp32-unknown-99", 0.30)]


def _print(a: IncidentAssessment) -> None:
    print(f"\n=== {a.asset_id}  [{a.status.upper()}]  engine={a.engine}")
    print(f"cause      : {a.likely_cause.value} ({a.confidence:.0%})   alternatives: "
          f"{[(x.cause.value, x.confidence) for x in a.alternatives]}")
    print(f"action     : {a.recommended_action.value}  -> {a.recommended_action_text}")
    print(f"approval   : {'REQUIRED' if a.approval_required else 'not required'}   risk={a.risk_level.value}")
    print("explain    : " + textwrap.fill(a.explanation, 100, subsequent_indent=" " * 13))
    print(f"evidence   : {', '.join(a.evidence_refs)}")
    for n in a.notes:
        print(f"note       : {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the AegisOT agent workflow on demo or custom incidents.")
    ap.add_argument("--asset", help="asset_id to analyse (default: run the 5 demo scenarios)")
    ap.add_argument("--score", type=float, default=0.9, help="anomaly score 0-1 for --asset")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of the readable summary")
    args = ap.parse_args()

    settings = get_settings()
    graph = build_graph(settings, build_clients(settings))
    cases = [(args.asset, args.score)] if args.asset else DEMO_CASES
    print(f"LLM enabled: {settings.llm_enabled} | backend={settings.backend_mode} "
          f"anomaly={settings.anomaly_mode} network={settings.network_mode}")
    for i, (asset, score) in enumerate(cases, 1):
        alert = AlertIn(incident_id=f"DEMO-{i:03d}", asset_id=asset, anomaly_score=score, model_version="demo")
        result = run_incident(graph, alert)
        print(result.model_dump_json(indent=2)) if args.json else _print(result)


if __name__ == "__main__":
    main()

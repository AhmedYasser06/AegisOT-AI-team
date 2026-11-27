from __future__ import annotations

import json
import operator
import threading
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import Runner, make_runner
from .clients import Clients
from .crosscheck import crosscheck
from .llm import complete_text, resolve_llm, short_error
from .roles import LLM_ROLES, SPECIALISTS
from .schemas import AlertIn, IncidentAssessment, SpecialistReport
from .settings import Settings
from .supervisor import build_assessment
from .tools import EvidencePack, InlineAnomaly, collect_evidence, parse_channels

NARRATIVE_SYSTEM = (
    "You write a 2-3 sentence incident explanation for a plant operator. Use ONLY the JSON facts provided. "
    "The reader already sees a headline with the main measurement, so do NOT restate it. Say what it means for the "
    "machine, what to check first and why, and how sure the diagnosis is. Add a number only if it is new. Do not state the confidence percentage (it is shown separately). "
    "Do not add causes, actions or numbers that are not in the JSON. Plain text, no markdown.")
NARRATIVE_FACTS = {"asset_id", "status", "likely_cause", "fault_type", "confidence", "alternatives",
                   "recommended_action", "recommended_solution", "approval_required", "reason", "findings"}


def narrative_facts(assessment: IncidentAssessment) -> str:
    """The ONLY thing the narrative LLM sees: validated facts as JSON (it cannot change them)."""
    return json.dumps(assessment.model_dump(include=NARRATIVE_FACTS, mode="json"))


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class State(TypedDict, total=False):
    alert: AlertIn
    pack: EvidencePack
    reports: Annotated[list[SpecialistReport], operator.add]    # each specialist appends its own report
    notes: Annotated[list[str], operator.add]
    assessment: IncidentAssessment
    defer_narrative: bool                                       # True = streaming endpoint writes the explanation itself
    timings_ms: Annotated[dict[str, int], _merge_dicts]    # how long each node took (find the slow one)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def llm_roles_of(settings: Settings) -> set[str]:
    """Roles named in LLM_AGENTS (validated); the others always use deterministic rules."""
    roles = {r.strip() for r in settings.llm_agents.split(",") if r.strip()}
    unknown = roles - set(LLM_ROLES)
    if unknown:
        raise ValueError(f"LLM_AGENTS has unknown role(s) {sorted(unknown)}; valid: {list(LLM_ROLES)}")
    return roles


def build_graph(settings: Settings, clients: Clients, runner: Runner | None = None):
    if runner is None and settings.llm_enabled:
        runner = make_runner(settings, clients)        # DirectRunner (default) or CrewRunner
    llm_roles = llm_roles_of(settings)
    gate = threading.Semaphore(max(1, settings.llm_max_concurrent))     # limit simultaneous LLM calls (rate limits)

    def collect(state: State):
        a = state["alert"]
        t0 = time.perf_counter()
        pack = collect_evidence(clients, a.asset_id, a.alert_id, settings.evidence_window_minutes,
                                InlineAnomaly.from_alert(a, settings), end=a.evidence_end.isoformat(),
                                channels=parse_channels(settings.telemetry_channels))
        return {"pack": pack, "timings_ms": {"collect_evidence": _ms(t0)}}

    def make_specialist(role: str):
        def node(state: State):
            alert, pack = state["alert"], state["pack"]
            t0 = time.perf_counter()
            if runner is not None and role in llm_roles and SPECIALISTS[role].uses_llm:
                try:
                    with gate:
                        report, notes = runner(role, alert, pack)
                    if settings.llm_crosscheck and report.engine == "llm":
                        report, why_note = crosscheck(report, SPECIALISTS[role].rules(alert, pack))
                        notes = [*notes, why_note] if why_note else notes
                    return {"reports": [report], "notes": notes, "timings_ms": {role: _ms(t0)}}
                except Exception as e:  # LLM down, invalid output, timeout, tool loop failure, ...
                    err = short_error(e)
                    note = f"{role}: LLM path failed ({err}); deterministic rules used."
                    fallback = SPECIALISTS[role].rules(alert, pack).model_copy(update={"llm_error": err})
                    return {"reports": [fallback], "notes": [note], "timings_ms": {role: _ms(t0)}}
            return {"reports": [SPECIALISTS[role].rules(alert, pack)], "notes": [],
                    "timings_ms": {role: _ms(t0)}}
        node.__name__ = f"{role}_node"
        return node

    def supervisor(state: State):
        alert, pack, reports = state["alert"], state["pack"], state["reports"]
        notes = list(state.get("notes", []))
        explanation, extra_models = None, []
        if runner is not None and "supervisor" in llm_roles and not state.get("defer_narrative"):
            # build the validated facts first, then let the LLM only rephrase them
            draft = build_assessment(alert, pack, reports, settings, notes)
            try:
                spec = resolve_llm(settings, "supervisor")
                explanation = complete_text(spec, NARRATIVE_SYSTEM, narrative_facts(draft)) or None
                extra_models = [spec.model] if explanation else []
            except Exception as e:
                notes.append(f"supervisor narrative failed ({short_error(e)}); template used.")
        t0 = time.perf_counter()
        result = build_assessment(alert, pack, reports, settings, notes, explanation, extra_models)
        result.timings_ms = {**state.get("timings_ms", {}), "supervisor": _ms(t0)}
        return {"assessment": result}

    g = StateGraph(State)
    g.add_node("collect_evidence", collect)
    g.add_node("supervisor", supervisor)
    for role in SPECIALISTS:                      # fan-out: collect_evidence -> each specialist -> supervisor
        g.add_node(role, make_specialist(role))
        g.add_edge("collect_evidence", role)
        g.add_edge(role, "supervisor")
    g.add_edge(START, "collect_evidence")
    g.add_edge("supervisor", END)
    return g.compile()


def initial_state(alert: AlertIn, defer_narrative: bool = False) -> dict:
    return {"alert": alert, "reports": [], "notes": [], "timings_ms": {}, "defer_narrative": defer_narrative}


def run_incident(graph, alert: AlertIn) -> IncidentAssessment:
    return graph.invoke(initial_state(alert))["assessment"]

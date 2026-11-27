from __future__ import annotations

import time
from typing import Iterator

from .agents import Runner
from .clients import ClientError, Clients, build_clients
from .graph import (NARRATIVE_SYSTEM, build_graph, initial_state, llm_roles_of, narrative_facts, run_incident)
from .llm import iter_with_deadline, resolve_llm, short_error, stream_text
from .roles import SPECIALISTS
from .schemas import AlertIn, IncidentAssessment
from .settings import Settings, get_settings
from .store import IncidentStore
from .supervisor import build_assessment


class NotAnAnomaly(ValueError):
    """The alert says is_anomaly=false: only anomalies create incidents."""


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


class IncidentService:
    def __init__(self, settings: Settings | None = None, clients: Clients | None = None,
                 runner: Runner | None = None):
        self.settings = settings or get_settings()
        self.clients = clients or build_clients(self.settings)
        self.graph = build_graph(self.settings, self.clients, runner)
        self.store = IncidentStore()
        # same rule as the graph: the supervisor narrative needs an LLM to be on AND the role to be enabled
        self.narrative_on = (runner is not None or self.settings.llm_enabled) \
            and "supervisor" in llm_roles_of(self.settings)

    # ───────── shared by both ways ─────────
    def _check(self, alert: AlertIn) -> None:
        if not alert.is_anomaly:
            raise NotAnAnomaly("is_anomaly=false: nothing to analyse (only anomalies create incidents)")

    def _finish(self, assessment: IncidentAssessment, push: bool | None, t0: float) -> IncidentAssessment:
        assessment.latency_ms = _ms(t0)
        if self.settings.push_result_to_backend if push is None else push:
            try:
                self.clients.backend.post_assessment(assessment.model_dump(mode="json"))
            except ClientError as e:
                assessment.notes.append(f"backend push failed: {e}")
        self.store.put(assessment)
        return assessment

    # ───────── normal (wait for the end) ─────────
    def analyze(self, alert: AlertIn, push: bool | None = None, force: bool = False) -> IncidentAssessment:
        self._check(alert)
        existing = self.store.get(alert.incident_id)
        if existing and not force:                         # idempotent for retries from ML/backend
            return existing
        t0 = time.perf_counter()
        return self._finish(run_incident(self.graph, alert), push, t0)

    # ───────── streaming ─────────
    def analyze_stream(self, alert: AlertIn, push: bool | None = None, force: bool = False
                       ) -> Iterator[tuple[str, dict]]:
        self._check(alert)
        t0 = time.perf_counter()
        yield "accepted", {"incident_id": alert.incident_id, "asset_id": alert.asset_id}

        existing = self.store.get(alert.incident_id)
        if existing and not force:
            yield "done", {"cached": True, "assessment": existing.model_dump(mode="json")}
            return

        final, stored = None, False
        try:
            # defer_narrative=True: the supervisor decides first; the explanation is streamed afterwards
            for update in self.graph.stream(initial_state(alert, defer_narrative=True), stream_mode="updates"):
                for node, out in update.items():
                    if node == "collect_evidence":
                        yield from self._evidence_events(alert, out["pack"], t0)
                    elif node in SPECIALISTS:
                        yield from self._vote_events(node, out, t0)
                    elif node == "supervisor":
                        final = out["assessment"]
                        yield "decision", {"t_ms": _ms(t0), "explanation_pending": self.narrative_on,
                                           "assessment": final.model_dump(mode="json")}
            if final is None:
                raise RuntimeError("workflow ended without a decision")
            yield from self._narrative_events(final)
            final = self._finish(final, push, t0)
            stored = True
            yield "done", {"cached": False, "latency_ms": final.latency_ms,
                           "assessment": final.model_dump(mode="json")}
        except Exception as e:                       # never leave the client hanging
            yield "error", {"message": short_error(e)}
        finally:
            # The client may disconnect in the middle of the explanation (GeneratorExit). The decision is already
            # valid, so still store it (and push it to the backend) instead of losing the incident.
            if final is not None and not stored:
                try:
                    self._finish(final, push, t0)
                except Exception:
                    pass

    def _evidence_events(self, alert, pack, t0):
        yield "evidence_collected", {"t_ms": _ms(t0), "refs": list(pack.refs), "unavailable": pack.unavailable}
        # Rules-only answer in milliseconds, so the operator sees something immediately.
        # It is PRELIMINARY: the UI must not show Approve/Reject until `decision` arrives.
        reports = [r.rules(alert, pack) for r in SPECIALISTS.values()]
        pre = build_assessment(alert, pack, reports, self.settings, [])
        yield "preliminary", {"t_ms": _ms(t0), "final": False, "status": pre.status,
                              "likely_cause": pre.likely_cause.value, "fault_type": pre.fault_type.value,
                              "headline": pre.headline, "confidence": pre.confidence,
                              "recommended_action": pre.recommended_action.value, "engine": "rules"}

    def _vote_events(self, role, out, t0):
        report = out["reports"][0]
        yield "vote", {"t_ms": _ms(t0), "role": role, "engine": report.engine, "model": report.model,
                       "scores": {c.value: s for c, s in report.scores.items()},
                       "findings": report.findings, "notes": out.get("notes", [])}

    def _narrative_events(self, final: IncidentAssessment):
        """Stream the explanation token by token. It only rephrases validated facts; it cannot change the decision."""
        if not self.narrative_on:
            return
        try:
            spec = resolve_llm(self.settings, "supervisor")
            pieces = []
            for piece in iter_with_deadline(stream_text(spec, NARRATIVE_SYSTEM, narrative_facts(final)),
                                            self.settings.narrative_timeout_s):
                pieces.append(piece)
                yield "explanation_token", {"t": piece}
            text = "".join(pieces).strip()
            if text:
                final.explanation = text
                final.llm_models_used = sorted({*final.llm_models_used, spec.model})
                final.llm = final.llm.model_copy(update={"narrative_model": spec.model})
        except Exception as e:
            final.notes.append(f"supervisor narrative failed ({short_error(e)}); template used.")
            yield "note", {"message": final.notes[-1]}

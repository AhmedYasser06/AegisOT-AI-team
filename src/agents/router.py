from __future__ import annotations

import hmac
import json
from typing import Union

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from .agents import Runner
from .clients import ClientError, Clients
from .llm import resolve_llm
from .policy import ALLOWLIST
from .roles import LLM_ROLES
from .schemas import AlertIn, DiagnosisOut, IncidentAssessment, IncidentCompact
from .service import IncidentService, NotAnAnomaly
from .views import to_compact
from .settings import Settings, get_settings
from .tools import EvidencePack, collect_evidence, get_network_status, parse_channels


def build_agent_router(settings: Settings | None = None, clients: Clients | None = None,
                       runner: Runner | None = None) -> APIRouter:
    settings = settings or get_settings()
    service = IncidentService(settings, clients, runner)

    def auth(x_api_key: str | None = Header(default=None)):
        if not x_api_key or not hmac.compare_digest(x_api_key, settings.agent_api_key):
            raise HTTPException(401, "invalid or missing X-API-Key")

    r = APIRouter(dependencies=[Depends(auth)], tags=["agents"])

    View = Query(default="compact", pattern="^(compact|full)$",
                 description="compact = short answer for the dashboard/mobile app (default); full = every field")

    def shaped(a: IncidentAssessment, view: str):
        return a if view == "full" else to_compact(a)

    def analyze(alert: AlertIn, push: bool | None, force: bool) -> IncidentAssessment:
        try:
            return service.analyze(alert, push, force)
        except NotAnAnomaly as e:
            raise HTTPException(422, str(e))

    # ───────── main endpoints (names from the team README) ─────────
    @r.post("/api/v1/incident/analyze", response_model=Union[IncidentCompact, IncidentAssessment])
    def incident_analyze(alert: AlertIn,
                         push: bool | None = Query(default=None, description="override PUSH_RESULT_TO_BACKEND"),
                         force: bool = Query(default=False, description="re-run even if incident_id was analysed"),
                         view: str = View):
        """Assessment: fault type, headline, confidence, evidence numbers, solution steps, action, approval flags.
        `?view=full` returns the complete record (votes, refs, timings) used by the backend database."""
        return shaped(analyze(alert, push, force), view)

    @r.post("/api/v1/incident/analyze/stream")
    def incident_analyze_stream(alert: AlertIn, push: bool | None = Query(default=None),
                                force: bool = Query(default=False)):
        """Server-Sent Events: accepted, evidence_collected, preliminary, vote, decision, explanation_token, done.
        Browsers must NOT call this directly (no X-API-Key header, key must stay secret): the Laravel backend
        proxies the stream to the dashboard."""
        if not alert.is_anomaly:                       # check BEFORE the stream starts so we can still return HTTP 422
            raise HTTPException(422, "is_anomaly=false: nothing to analyse (only anomalies create incidents)")

        def sse():
            for event, data in service.analyze_stream(alert, push, force):
                yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @r.post("/api/v1/diagnosis/analyze", response_model=DiagnosisOut)
    def diagnosis_analyze(alert: AlertIn, push: bool | None = None, force: bool = False):
        """Older compact format from the team README (diagnosis, confidence, evidence, recommended_action...).
        New code should use /api/v1/incident/analyze (compact view), which also has the fault type and solution steps."""
        a = analyze(alert, push, force)
        return DiagnosisOut(incident_id=a.incident_id, asset_id=a.asset_id, diagnosis=a.diagnosis,
                            confidence=a.confidence, evidence=a.evidence, recommended_action=a.recommended_action_text,
                            action_id=a.recommended_action, approval_required=a.approval_required, status=a.status)

    @r.get("/api/v1/incident/{incident_id}", response_model=Union[IncidentCompact, IncidentAssessment])
    def get_incident(incident_id: str, view: str = View):
        a = service.store.get(incident_id)
        if not a:
            raise HTTPException(404, "incident not found")
        return shaped(a, view)

    @r.get("/api/v1/incidents", response_model=list[Union[IncidentCompact, IncidentAssessment]])
    def list_incidents(limit: int = Query(default=20, ge=1, le=200), view: str = View):
        return [shaped(a, view) for a in service.store.latest(limit)]

    # ───────── integration debugging / introspection ─────────
    @r.get("/api/v1/agents/evidence/{asset_id}", response_model=EvidencePack)
    def evidence(asset_id: str, alert_id: str | None = None):
        """Everything the agents would see for this asset. Use it to verify each team's integration."""
        return collect_evidence(service.clients, asset_id, alert_id, settings.evidence_window_minutes,
                                channels=parse_channels(settings.telemetry_channels))

    @r.get("/api/v1/agents/network-health/{asset_id}")
    def network_health(asset_id: str):
        try:
            data, ref, _ = get_network_status(service.clients, asset_id)
        except ClientError as e:
            raise HTTPException(502, f"network service error: {e}")
        return {"ref": ref, "data": data}

    @r.get("/api/v1/agents/llm")
    def llm_config():
        """Which model each agent would use. Secrets are never returned."""
        return {"enabled": settings.llm_enabled, "roles": {x: resolve_llm(settings, x).public() for x in LLM_ROLES}}

    @r.get("/api/v1/agents/actions")
    def actions():
        return {a.value: {"risk": p.risk.value, "requires_approval": p.requires_approval,
                          "auto_allowed": p.auto_allowed} for a, p in ALLOWLIST.items()}

    return r

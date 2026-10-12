"""FastAPI router for the agent layer. Included by src/api/main.py (the shared AI API)."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .agents import Runner
from .clients import Clients, ClientError, build_clients
from .graph import build_graph, run_incident
from .llm import ROLES, resolve_llm
from .policy import ALLOWLIST
from .schemas import AlertIn, DiagnosisOut, IncidentAssessment
from .settings import Settings, get_settings
from .store import IncidentStore
from .tools import EvidencePack, collect_evidence, get_network_status


def build_agent_router(settings: Settings | None = None, clients: Clients | None = None,
                       runner: Runner | None = None) -> APIRouter:
    settings = settings or get_settings()
    clients = clients or build_clients(settings)
    graph = build_graph(settings, clients, runner)
    store = IncidentStore()

    def auth(x_api_key: str | None = Header(default=None)):
        if not x_api_key or not hmac.compare_digest(x_api_key, settings.agent_api_key):
            raise HTTPException(401, "invalid or missing X-API-Key")

    r = APIRouter(dependencies=[Depends(auth)], tags=["agents"])

    def analyze(alert: AlertIn, push: bool | None, force: bool) -> IncidentAssessment:
        if not alert.is_anomaly:
            raise HTTPException(422, "is_anomaly=false: nothing to analyse (only anomalies create incidents)")
        existing = store.get(alert.incident_id)
        if existing and not force:                         # idempotent for retries from ML/backend
            return existing
        assessment = run_incident(graph, alert)
        if settings.push_result_to_backend if push is None else push:
            try:
                clients.backend.post_assessment(assessment.model_dump(mode="json"))
            except ClientError as e:
                assessment.notes.append(f"backend push failed: {e}")
        store.put(assessment)
        return assessment

    # ───────── main endpoints (names from the team README) ─────────
    @r.post("/api/v1/incident/analyze", response_model=IncidentAssessment)
    def incident_analyze(alert: AlertIn,
                         push: bool | None = Query(default=None, description="override PUSH_RESULT_TO_BACKEND"),
                         force: bool = Query(default=False, description="re-run even if incident_id was analysed")):
        """Full assessment: cause, alternatives, evidence refs, action id, risk, approval flags, notes."""
        return analyze(alert, push, force)

    @r.post("/api/v1/diagnosis/analyze", response_model=DiagnosisOut)
    def diagnosis_analyze(alert: AlertIn, push: bool | None = None, force: bool = False):
        """Compact diagnosis in the team's example format (diagnosis, confidence, evidence, recommended_action...)."""
        a = analyze(alert, push, force)
        return DiagnosisOut(incident_id=a.incident_id, asset_id=a.asset_id, diagnosis=a.diagnosis,
                            confidence=a.confidence, evidence=a.evidence, recommended_action=a.recommended_action_text,
                            action_id=a.recommended_action, approval_required=a.approval_required, status=a.status)

    @r.get("/api/v1/incident/{incident_id}", response_model=IncidentAssessment)
    def get_incident(incident_id: str):
        a = store.get(incident_id)
        if not a:
            raise HTTPException(404, "incident not found")
        return a

    @r.get("/api/v1/incidents", response_model=list[IncidentAssessment])
    def list_incidents(limit: int = Query(default=20, ge=1, le=200)):
        return store.latest(limit)

    # ───────── integration debugging / introspection ─────────
    @r.get("/api/v1/agents/evidence/{asset_id}", response_model=EvidencePack)
    def evidence(asset_id: str, alert_id: str | None = None):
        """Everything the agents would see for this asset. Use it to verify each team's integration."""
        return collect_evidence(clients, asset_id, alert_id, settings.evidence_window_minutes)

    @r.get("/api/v1/agents/network-health/{asset_id}")
    def network_health(asset_id: str):
        try:
            data, ref, _ = get_network_status(clients, asset_id)
        except ClientError as e:
            raise HTTPException(502, f"network service error: {e}")
        return {"ref": ref, "data": data}

    @r.get("/api/v1/agents/llm")
    def llm_config():
        """Which model each agent would use. Secrets are never returned."""
        return {"enabled": settings.llm_enabled, "roles": {x: resolve_llm(settings, x).public() for x in ROLES}}

    @r.get("/api/v1/agents/actions")
    def actions():
        return {a.value: {"risk": p.risk.value, "requires_approval": p.requires_approval,
                          "auto_allowed": p.auto_allowed} for a, p in ALLOWLIST.items()}

    return r

"""Two views of the same assessment.

* full    = IncidentAssessment.
* compact = IncidentCompact.
"""
from __future__ import annotations

from .schemas import (CompactAction, CompactEvidence, IncidentAssessment, IncidentCompact)


def to_compact(a: IncidentAssessment) -> IncidentCompact:
    # results of the rule-based agents (maintenance / network / security): "network: Network healthy ..."
    other = [f"{role}: {text}" for f in a.findings if not f.startswith("[diagnosis]")
             for role, _, text in [f[1:].partition("] ")]]
    return IncidentCompact(
        incident_id=a.incident_id, asset_id=a.asset_id, status=a.status, severity=a.severity,
        likely_cause=a.likely_cause, fault_type=a.fault_type, headline=a.headline, confidence=a.confidence,
        summary=a.explanation, affected_channels=a.affected_channels,
        evidence=[CompactEvidence(channel=s["channel"], value=s["value"], baseline=s["baseline"],
                                  change=s["change"], direction=s["direction"]) for s in a.signals],
        recommended_solution=a.recommended_solution,
        action=CompactAction(id=a.recommended_action, params=a.action_params, risk=a.risk_level,
                             approval_required=a.approval_required, auto_executable=a.auto_executable),
        alternatives=a.alternatives, other_checks=other, reason=a.reason, llm=a.llm, notes=a.notes,
        anomaly_score=a.anomaly_score, latency_ms=a.latency_ms, created_at=a.created_at)

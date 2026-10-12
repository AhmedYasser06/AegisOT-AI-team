"""Supervisor logic: merge specialist reports -> validated IncidentAssessment.

Pure functions (no LLM calls) so behaviour is testable and deterministic.
The optional LLM narrative is added in graph.py and can only change the
`explanation` text, never the cause, confidence, action or approval flags.
"""
from __future__ import annotations

from .settings import Settings
from .tools import EvidencePack
from .policy import (ACTION_TEXT, ALLOWED_CAUSES, ALLOWLIST, DIAGNOSIS_TEXT, choose_action,
                     coerce_action)
from .schemas import (ActionId, AlertIn, Alternative, Cause, IncidentAssessment,
                      SpecialistReport)

def merge_reports(reports: list[SpecialistReport], pack: EvidencePack) -> tuple[dict[Cause, float], list[str]]:
    notes: list[str] = []
    best: dict[Cause, float] = {c: 0.0 for c in Cause if c != Cause.unknown}
    by_role = {r.role: r for r in reports}
    for r in reports:
        if r.role == "maintenance":
            continue
        for cause, sc in r.scores.items():
            if cause in ALLOWED_CAUSES.get(r.role, set()):
                best[cause] = max(best[cause], sc)
    maint = by_role.get("maintenance")
    if maint and maint.scores.get(Cause.physical_fault, 0) > 0.5 and best[Cause.physical_fault] >= 0.4:
        best[Cause.physical_fault] = min(0.98, best[Cause.physical_fault] + 0.1)
        notes.append("Maintenance history supports a wear-related fault (+0.1).")
    missing = (pack.telemetry or {}).get("missing_ratio", 0.0)
    if missing >= 0.3 and best[Cause.network_issue] >= 0.6:
        best[Cause.physical_fault] *= 0.5
        best[Cause.sensor_fault] *= 0.5
        notes.append("Telemetry gaps + network degradation: machine/sensor causes down-weighted (proposal 6.3).")
    return best, notes


def build_assessment(alert: AlertIn, pack: EvidencePack, reports: list[SpecialistReport],
                     settings: Settings, notes: list[str], explanation: str | None = None,
                     extra_models: list[str] | None = None) -> IncidentAssessment:
    best, merge_notes = merge_reports(reports, pack)
    notes = [*notes, *merge_notes]
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    (top_cause, top_score), (_, second) = ranked[0], ranked[1]

    penalty = min(0.3, 0.1 * len(pack.unavailable))
    if pack.unavailable:
        notes.append(f"Evidence sources unavailable: {pack.unavailable} (confidence -{penalty:.1f}).")
    confidence = round(max(0.0, top_score - penalty), 3)

    if pack.anomaly and abs(pack.anomaly.get("score", alert.anomaly_score) - alert.anomaly_score) > 0.2:
        notes.append("Alert anomaly_score differs from anomaly service score by >0.2 - check contract/versions.")

    refs = list(dict.fromkeys(r for rep in reports for r in rep.evidence_refs if r in pack.refs))
    findings = [f"[{rep.role}] {f}" for rep in reports for f in rep.findings]
    evidence = list(dict.fromkeys(f for rep in reports for f in rep.findings))
    checks = list(dict.fromkeys(c for rep in reports for c in rep.checks_performed))

    inconclusive = (confidence < settings.min_confidence or (top_score - second) < settings.min_margin or not refs)
    if not refs:
        notes.append("No valid evidence references - result forced to inconclusive.")
    cause = Cause.unknown if inconclusive else top_cause

    action, params = choose_action(cause, confidence, alert.anomaly_score)
    action = coerce_action(action)                      # allowlist enforcement
    pol = ALLOWLIST[action]

    engines = {r.engine for r in reports}
    engine = "llm" if engines == {"llm"} else "rules" if engines == {"rules"} else "mixed"
    models = sorted({r.model for r in reports if r.model} | set(extra_models or []))

    alternatives = [Alternative(cause=c, confidence=round(s, 3)) for c, s in ranked
                    if s >= 0.2 and (inconclusive or c != top_cause)][:3]

    if explanation is None:
        explanation = default_explanation(alert, cause, confidence, action, pol.requires_approval, inconclusive, evidence)

    return IncidentAssessment(
        incident_id=alert.incident_id, alert_id=alert.alert_id, asset_id=alert.asset_id,
        status="inconclusive" if inconclusive else "complete", likely_cause=cause, confidence=confidence,
        alternatives=alternatives, evidence_refs=refs, findings=findings, recommended_action=action,
        action_params=params, risk_level=pol.risk, approval_required=pol.requires_approval,
        auto_executable=pol.auto_allowed and not pol.requires_approval, checks_performed=checks,
        explanation=explanation, diagnosis=DIAGNOSIS_TEXT[cause], evidence=evidence,
        recommended_action_text=ACTION_TEXT[action], engine=engine, llm_models_used=models, notes=notes)


def default_explanation(alert, cause, confidence, action, needs_approval, inconclusive, findings) -> str:
    head = f"Anomaly on {alert.asset_id} (score {alert.anomaly_score:.2f}). "
    if inconclusive:
        return head + ("Evidence is weak or contradictory, so no cause is asserted; escalating to a human for inspection. "
                       + (findings[0] if findings else ""))
    key = " ".join(findings[:2])
    tail = f" Proposed action: {action.value}" + (" (requires operator approval)." if needs_approval else ".")
    return f"{head}Most likely cause: {cause.value} ({confidence:.0%}). {key}{tail}"

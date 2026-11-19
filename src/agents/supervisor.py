"""Supervisor logic: merge specialist reports -> validated IncidentAssessment.

Pure functions (no LLM calls) so behaviour is testable and deterministic.
The optional LLM narrative is added in graph.py and can only change the
`explanation` text, never the cause, confidence, action or approval flags.

Pipeline:  merge_reports -> rank causes -> confidence (minus missing-evidence penalty)
           -> inconclusive? -> policy picks the action -> allowlist enforced -> IncidentAssessment
"""
from __future__ import annotations

from .enums import ActionId, Cause
from .policy import ACTION_TEXT, ALLOWLIST, DIAGNOSIS_TEXT, choose_action, coerce_action, ml_gate
from .roles import SPECIALISTS
from .schemas import AlertIn, Alternative, IncidentAssessment, SpecialistReport, Vote
from .settings import Settings
from .tools import EvidencePack

MAX_UNAVAILABLE_PENALTY = 0.3      # each missing evidence source costs 0.1 confidence, up to this
MIN_ALTERNATIVE_SCORE = 0.2        # weaker causes are not reported as alternatives


def merge_reports(reports: list[SpecialistReport], pack: EvidencePack) -> tuple[dict[Cause, float], list[str]]:
    """Best score per cause (each specialist may only vote inside its own domain), plus cross-checks."""
    notes: list[str] = []
    best: dict[Cause, float] = {c: 0.0 for c in Cause if c != Cause.unknown}
    for r in reports:
        if r.role == "maintenance":                      # maintenance never votes, it only supports (below)
            continue
        allowed = SPECIALISTS[r.role].causes if r.role in SPECIALISTS else frozenset()
        for cause, sc in r.scores.items():
            if cause in allowed:
                best[cause] = max(best[cause], sc)

    maint = next((r for r in reports if r.role == "maintenance"), None)
    if maint and maint.scores.get(Cause.physical_fault, 0) > 0.5 and best[Cause.physical_fault] >= 0.4:
        best[Cause.physical_fault] = min(0.98, best[Cause.physical_fault] + 0.1)
        notes.append("Maintenance history supports a wear-related fault (+0.1).")

    missing = (pack.telemetry or {}).get("missing_ratio", 0.0)
    if missing >= 0.3 and best[Cause.network_issue] >= 0.6:
        best[Cause.physical_fault] *= 0.5
        best[Cause.sensor_fault] *= 0.5
        notes.append("Telemetry gaps + network degradation: machine/sensor causes down-weighted (proposal 6.3).")
    return best, notes


def _joined(reports: list[SpecialistReport], field: str, valid: dict | None = None) -> list[str]:
    """Flatten one list field across reports, de-duplicated, keeping order."""
    items = (x for rep in reports for x in getattr(rep, field))
    return list(dict.fromkeys(x for x in items if valid is None or x in valid))


def _engine(reports: list[SpecialistReport]) -> str:
    engines = {r.engine for r in reports}
    return engines.pop() if len(engines) == 1 else "mixed"


def build_assessment(alert: AlertIn, pack: EvidencePack, reports: list[SpecialistReport],
                     settings: Settings, notes: list[str], explanation: str | None = None,
                     extra_models: list[str] | None = None) -> IncidentAssessment:
    best, merge_notes = merge_reports(reports, pack)
    notes = [*notes, *merge_notes]
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    (top_cause, top_score), (_, runner_up) = ranked[0], ranked[1]

    # confidence: the winner's score, reduced when evidence sources were unavailable
    penalty = min(MAX_UNAVAILABLE_PENALTY, 0.1 * len(pack.unavailable))
    if pack.unavailable:
        notes.append(f"Evidence sources unavailable: {pack.unavailable} (confidence -{penalty:.1f}).")
    confidence = round(max(0.0, top_score - penalty), 3)

    if pack.anomaly and abs(pack.anomaly.get("score", alert.anomaly_score) - alert.anomaly_score) > 0.2:
        notes.append("Alert anomaly_score differs from anomaly service score by >0.2 - check contract/versions.")

    refs = _joined(reports, "evidence_refs", valid=pack.refs)
    findings = [f"[{rep.role}] {f}" for rep in reports for f in rep.findings]
    evidence = _joined(reports, "findings")
    checks = _joined(reports, "checks_performed")

    # inconclusive: weak, too close to the runner-up, or nothing verifiable behind it
    inconclusive = (confidence < settings.min_confidence or (top_score - runner_up) < settings.min_margin or not refs)
    if not refs:
        notes.append("No valid evidence references - result forced to inconclusive.")
    cause = Cause.unknown if inconclusive else top_cause

    action, params = choose_action(cause, confidence, alert.anomaly_score)
    action = coerce_action(action)                      # allowlist enforcement
    action, params, gate_note = ml_gate(action, params, settings.ml_validated)    # ML not proven yet -> no stop
    if gate_note:
        notes.append(gate_note)
    pol = ALLOWLIST[action]

    alternatives = [Alternative(cause=c, confidence=round(s, 3)) for c, s in ranked
                    if s >= MIN_ALTERNATIVE_SCORE and (inconclusive or c != top_cause)][:3]
    if explanation is None:
        explanation = default_explanation(alert, cause, confidence, action, pol.requires_approval, inconclusive, evidence)

    return IncidentAssessment(
        incident_id=alert.incident_id, alert_id=alert.alert_id, asset_id=alert.asset_id,
        status="inconclusive" if inconclusive else "complete", likely_cause=cause, confidence=confidence,
        alternatives=alternatives, evidence_refs=refs, findings=findings, recommended_action=action,
        action_params=params, risk_level=pol.risk, approval_required=pol.requires_approval,
        auto_executable=pol.auto_allowed and not pol.requires_approval, checks_performed=checks,
        explanation=explanation, diagnosis=DIAGNOSIS_TEXT[cause], evidence=evidence,
        recommended_action_text=ACTION_TEXT[action], engine=_engine(reports),
        llm_models_used=sorted({r.model for r in reports if r.model} | set(extra_models or [])), notes=notes,
        anomaly_score=alert.anomaly_score, ml_model_version=alert.model_version, ml_validated=settings.ml_validated,
        votes=[Vote(role=r.role, engine=r.engine, scores=r.scores, findings=r.findings) for r in reports])


def default_explanation(alert: AlertIn, cause: Cause, confidence: float, action: ActionId,
                        needs_approval: bool, inconclusive: bool, findings: list[str]) -> str:
    head = f"Anomaly on {alert.asset_id} (score {alert.anomaly_score:.2f}). "
    if inconclusive:
        return head + ("Evidence is weak or contradictory, so no cause is asserted; escalating to a human for inspection. "
                       + (findings[0] if findings else ""))
    key = " ".join(findings[:2])
    tail = f" Proposed action: {action.value}" + (" (requires operator approval)." if needs_approval else ".")
    return f"{head}Most likely cause: {cause.value} ({confidence:.0%}). {key}{tail}"

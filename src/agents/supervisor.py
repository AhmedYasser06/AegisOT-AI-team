from __future__ import annotations

import re

from .enums import ActionId, Cause, FaultType
from .policy import ALLOWLIST, choose_action, coerce_action, ml_gate
from .roles import SPECIALISTS
from .schemas import AlertIn, Alternative, IncidentAssessment, LlmInfo, SpecialistReport, Vote
from .settings import Settings
from .signals import TARGET_OF_GROUP, describe
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


MACHINE_FAULTS = {FaultType.overheating, FaultType.abnormal_vibration, FaultType.electrical_fault,
                  FaultType.speed_deviation, FaultType.machine_fault}
SEVERITIES = ("low", "medium", "high", "critical")


def _diagnosis(reports: list[SpecialistReport]) -> SpecialistReport | None:
    return next((r for r in reports if r.role == "diagnosis"), None)


def resolve_fault(cause: Cause, diag: SpecialistReport | None) -> tuple[FaultType, list[str], list[dict]]:
    """(specific fault type, affected channels, measured signals) for the final cause."""
    signals = diag.signals if diag else []
    channels = list(diag.affected_channels) if diag else []
    if cause == Cause.physical_fault:
        ft = diag.fault_type if diag and diag.fault_type in MACHINE_FAULTS else FaultType.machine_fault
        return ft, (channels if ft != FaultType.machine_fault or channels else []), signals
    if cause == Cause.sensor_fault:
        return FaultType.sensor_fault, channels if diag and diag.fault_type == FaultType.sensor_fault else [], signals
    mapping = {Cause.network_issue: FaultType.network_issue, Cause.security_incident: FaultType.security_incident}
    return mapping.get(cause, FaultType.unknown), [], []


def _lead_finding(reports: list[SpecialistReport], cause: Cause) -> str:
    """First finding of the specialist that owns this cause (network -> network report, ...)."""
    for r in reports:
        if cause in r.scores and r.role in SPECIALISTS and cause in SPECIALISTS[r.role].causes and r.findings:
            return r.findings[0]
    return ""


def _per_group(signals: list[dict]) -> list[dict]:
    """The strongest signal of each kind (rpm AND speed_percent are one 'speed' problem, not two)."""
    seen, out = set(), []
    for s in signals:
        if s["group"] not in seen:
            seen.add(s["group"])
            out.append(s)
    return out


def compose_headline(ft: FaultType, signals: list[dict], reports: list[SpecialistReport], cause: Cause,
                     channels: list[str], reason: str | None) -> str:
    label = ft.value.replace("_", " ").capitalize()
    if ft == FaultType.unknown:
        return f"Inconclusive: {reason}" if reason else "Inconclusive"
    if ft == FaultType.sensor_fault and channels:
        return f"{label}: {', '.join(channels)} flat, invalid or out of range"
    if signals:
        return f"{label}: " + "; ".join(describe(s) for s in _per_group(signals)[:2])
    lead = _lead_finding(reports, cause)
    return f"{label}: {lead.rstrip('.')}" if lead else label


def compose_steps(ft: FaultType, signals: list[dict], channels: list[str], action: ActionId, params: dict,
                  approval_required: bool, reports: list[SpecialistReport], cause: Cause,
                  reason: str | None) -> list[str]:
    """What to do, composed from the measurements (used only when the LLM did not write the steps)."""
    steps: list[str] = []
    if ft in MACHINE_FAULTS:
        for s in _per_group(signals)[:3]:
            steps.append(f"Check {TARGET_OF_GROUP.get(s['group'], s['channel'])}: {describe(s)}.")
    elif ft == FaultType.sensor_fault:
        steps += [f"Re-check the {c} sensor: its readings are flat, invalid or out of range." for c in channels[:3]]
    elif ft == FaultType.unknown:
        steps.append(f"Escalate to an engineer: {reason}." if reason else "Escalate to an engineer.")
    else:
        lead = _lead_finding(reports, cause)
        if lead:
            steps.append(f"Verify: {lead}")
    return steps


def severity_of(anomaly_score: float, signals: list[dict], high_risk_action: bool) -> str:
    """Driven by HOW FAR the measurements moved (signal strength: 1 = just past its limit), not by the ML score alone.
    No measured deviation -> at most 'medium'. A proposed high-impact action never goes below 'high'."""
    if signals:
        strength = max(s["strength"] for s in signals)
        level = 0 if strength < 1.5 else 1 if strength < 3.0 else 2 if strength < 6.0 else 3
        if len({s["group"] for s in signals}) >= 2:         # several kinds of measurement moved together
            level = min(2, level + 1) if level < 3 else level
        if anomaly_score >= 0.75:
            level = max(level, 1)
    else:
        level = 0 if anomaly_score < 0.75 else 1
    if high_risk_action:
        level = max(level, 2)
    return SEVERITIES[min(3, level)]


_STEP_WORDS = {"temperature": r"temperature|cooling|ventilat|thermal|heat|fan",
               "vibration": r"vibrat|bearing|mounting|gyro|loose",
               "electrical": r"power|voltage|current|wiring|electrical|supply",
               "speed": r"speed|rpm|load|binding|regulation|belt"}


def order_steps(steps: list[str], fault_type: FaultType | None, signals: list[dict]) -> list[str]:
    """Most important first: the steps about the main fault's measurement, then the others by signal strength."""
    from .signals import FAULT_OF_GROUP
    groups = [g for g, f in FAULT_OF_GROUP.items() if f == fault_type] + [s["group"] for s in _per_group(signals)]
    rank = {g: i for i, g in enumerate(dict.fromkeys(groups))}

    def key(item):
        i, text = item
        hits = [(m.start(), rank.get(g, len(rank))) for g, w in _STEP_WORDS.items()
                if (m := re.search(w, text, re.I))]
        return (min(hits)[1] if hits else len(rank), i) if len(hits) < 2 else (
            min(hits, key=lambda h: h[0])[1], i)         # several topics in one step: the one named first wins
    return [t for _, t in sorted(enumerate(steps), key=key)]


def llm_info(reports: list[SpecialistReport], narrative_models: list[str] | None) -> LlmInfo:
    """Did the LLM really answer? (`engine=mixed` only says the diagnosis role was allowed to use it.)"""
    answered = next((r for r in reports if r.engine == "llm"), None)
    error = next((r.llm_error for r in reports if r.llm_error), None)
    status = "ok" if answered else "fallback_rules" if error else "disabled"
    return LlmInfo(status=status, model=answered.model if answered else None,
                   narrative_model=(narrative_models or [None])[0], error=error)


def build_assessment(alert: AlertIn, pack: EvidencePack, reports: list[SpecialistReport],
                     settings: Settings, notes: list[str], explanation: str | None = None,
                     extra_models: list[str] | None = None) -> IncidentAssessment:
    best, merge_notes = merge_reports(reports, pack)
    notes = [*notes, *merge_notes]
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    (top_cause, top_score), (runner_cause, runner_up) = ranked[0], ranked[1]

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
    why: list[str] = []
    if top_score < MIN_ALTERNATIVE_SCORE:
        why.append(f"no cause is supported by the evidence (best score {top_score:.2f})")
    elif confidence < settings.min_confidence:
        why.append(f"confidence {confidence:.2f} is below the {settings.min_confidence:.2f} minimum")
    elif (top_score - runner_up) < settings.min_margin:
        why.append(f"{top_cause.value} ({top_score:.2f}) and {runner_cause.value} ({runner_up:.2f}) are too close to separate")
    if not refs:
        why.append("no valid evidence references")
        notes.append("No valid evidence references - result forced to inconclusive.")
    inconclusive = bool(why)
    reason = "; ".join(why) if inconclusive else None
    cause = Cause.unknown if inconclusive else top_cause

    diag = _diagnosis(reports)
    fault_type, channels, signals = resolve_fault(cause, diag)
    action, params = choose_action(cause, confidence, alert.anomaly_score, fault_type, channels)
    action = coerce_action(action)                      # allowlist enforcement
    action, params, gate_note = ml_gate(action, params, settings.ml_validated, fault_type, channels)   # ML not proven -> no stop
    if gate_note:
        notes.append(gate_note)
    pol = ALLOWLIST[action]

    alternatives = [Alternative(cause=c, confidence=round(s, 3)) for c, s in ranked
                    if s >= MIN_ALTERNATIVE_SCORE and (inconclusive or c != top_cause)][:3]

    # prose: the diagnosis LLM wrote it, or it is composed from the numbers
    llm_wrote = diag is not None and diag.engine == "llm" and cause in (Cause.physical_fault, Cause.sensor_fault)
    headline = (diag.headline if llm_wrote else None) or compose_headline(fault_type, signals, reports, cause, channels, reason)
    steps = order_steps(diag.solution_steps, fault_type, signals) if llm_wrote and diag.solution_steps else compose_steps(
        fault_type, signals, channels, action, params, pol.requires_approval, reports, cause, reason)
    if pol.requires_approval:                       # the proposed high-impact action comes first
        steps = [f"{action.value} is proposed and needs operator approval before it runs.", *steps]
    if explanation is None:
        explanation = default_explanation(alert, cause, confidence, action, pol.requires_approval, inconclusive,
                                          evidence, headline, steps[0] if steps else "")

    return IncidentAssessment(
        incident_id=alert.incident_id, alert_id=alert.alert_id, asset_id=alert.asset_id,
        status="inconclusive" if inconclusive else "complete", likely_cause=cause, confidence=confidence,
        alternatives=alternatives, evidence_refs=refs, findings=findings, recommended_action=action,
        action_params=params, risk_level=pol.risk, approval_required=pol.requires_approval,
        auto_executable=pol.auto_allowed and not pol.requires_approval, checks_performed=checks,
        explanation=explanation, diagnosis=headline, evidence=evidence,
        recommended_action_text=steps[0] if steps else "", engine=_engine(reports),
        llm_models_used=sorted({r.model for r in reports if r.model} | set(extra_models or [])), notes=notes,
        anomaly_score=alert.anomaly_score, ml_model_version=alert.model_version, ml_validated=settings.ml_validated,
        votes=[Vote(role=r.role, engine=r.engine, scores=r.scores, findings=r.findings) for r in reports],
        fault_type=fault_type, severity=severity_of(alert.anomaly_score, signals, pol.risk.value == "high"),
        headline=headline, affected_channels=channels, signals=signals, recommended_solution=steps,
        reason=reason, llm=llm_info(reports, extra_models))


def default_explanation(alert: AlertIn, cause: Cause, confidence: float, action: ActionId,
                        needs_approval: bool, inconclusive: bool, findings: list[str], headline: str = "",
                        first_step: str = "") -> str:
    head = f"Anomaly on {alert.asset_id} (score {alert.anomaly_score:.2f}). "
    if inconclusive:
        return head + ("Evidence is weak or contradictory, so no cause is asserted; escalating to a human for inspection. "
                       + (findings[0] if findings else ""))
    if first_step and headline:       # the headline is shown separately: do not repeat it here
        return f"{head}Confidence {confidence:.0%}. Do first: {first_step}"
    what = headline or f"Most likely cause: {cause.value}"
    tail = f" Proposed action: {action.value}" + (" (requires operator approval)." if needs_approval else ".")
    return f"{head}{what} (confidence {confidence:.0%}).{tail}"

"""Deterministic specialists. They are:
  * the fallback when the LLM is disabled, down, or returns invalid output 
  * the baseline the agent workflow is compared against in the evaluation 
"""
from __future__ import annotations

from .enums import Cause, FaultType
from .schemas import AlertIn, SpecialistReport
from .signals import build_signals, corroborated, describe, evidence_confidence, primary_fault
from .tools import EvidencePack

MISSING_RATIO = 0.3       # >=30% of expected samples missing = incomplete telemetry


def _report(role, scores, findings, refs, checks) -> SpecialistReport:
    return SpecialistReport(role=role, engine="rules", scores=scores, findings=findings,
                            evidence_refs=refs, checks_performed=checks)


def _num(value, default: float = 0.0) -> float:
    """Backends may send null / strings for a field: treat anything unusable as `default` instead of crashing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rules_diagnosis(alert: AlertIn, pack: EvidencePack) -> SpecialistReport:
    """Machine vs sensor, decided from MEASURED deviations (signals.py). Also names the specific fault type
    (overheating / abnormal_vibration / electrical_fault / speed_deviation / sensor_fault) and the channels involved."""
    scores = {Cause.physical_fault: 0.05, Cause.sensor_fault: 0.05}
    findings, refs, checks = [], [], ["telemetry_summary", "anomaly_details"]
    t, signals, bad = pack.telemetry, [], set()
    if t is None:
        findings.append("Telemetry unavailable - cannot confirm a machine or sensor fault.")
        signals = build_signals(None, pack.anomaly)         # the ML's own value/baseline pairs, if it sent them
    else:
        refs += [r for r in pack.refs if r.startswith("telemetry:")]
        bad = set(t["flatlined"]) | set(t["implausible"])
        if bad:
            scores[Cause.sensor_fault] = 0.85
            findings.append(f"Channel(s) {sorted(bad)} flatlined or outside plausible range - likely sensor fault.")
        signals = build_signals(t, pack.anomaly, skip=bad)
        if t["missing_ratio"] >= MISSING_RATIO:
            scores[Cause.physical_fault] = 0.2
            findings.append(f"{t['missing_ratio']:.0%} of expected samples missing - machine fault cannot be confirmed.")
            signals = []
        elif signals:
            scores[Cause.physical_fault] = evidence_confidence(alert.anomaly_score, signals)
            findings += [describe(s) + "." for s in signals]
        elif not bad:
            findings.append("All channels are close to baseline.")
            checks.append("telemetry_all_within_baseline")
    if pack.anomaly:
        refs += [r for r in pack.refs if r.startswith("anomaly:")]
        feats = [f["name"] for f in pack.anomaly.get("top_features", [])[:3]]
        if feats:
            findings.append(f"Anomaly model top features: {', '.join(feats)}.")
    machine_fault, channels = primary_fault(signals)
    if scores[Cause.sensor_fault] >= scores[Cause.physical_fault] and bad:
        fault, channels = FaultType.sensor_fault, sorted(bad)
    else:
        fault = machine_fault if scores[Cause.physical_fault] >= 0.5 else None
        channels = channels if fault else []
    rep = _report("diagnosis", scores, findings, refs, checks)
    return rep.model_copy(update={"fault_type": fault, "affected_channels": channels, "signals": signals,
                                  "corroborated": corroborated(signals)})


def rules_network(alert: AlertIn, pack: EvidencePack) -> SpecialistReport:
    n, findings, refs = pack.network, [], []
    checks = ["network_health"]
    score = 0.1
    if n is None:
        findings.append("Network health unavailable - network cause not verified.")
    else:
        refs = [r for r in pack.refs if r.startswith("network:")]
        loss, lat = _num(n.get("packet_loss_pct")), _num(n.get("latency_ms"))
        link = str((n.get("interface") or {}).get("status", n.get("link_status", "up")))
        score = 0.05
        if not n.get("reachable", True) or link == "down":
            score = 0.95
            findings.append("Asset unreachable or link down.")
        elif loss >= 20 or lat >= 300:
            score = 0.9
            findings.append(f"Severe degradation: packet loss {loss}%, latency {lat} ms.")
        elif loss >= 5 or lat >= 100 or link == "degraded":
            score = 0.6
            findings.append(f"Degraded path: packet loss {loss}%, latency {lat} ms, link {link}.")
        else:
            findings.append(f"Network healthy (loss {loss}%, latency {lat} ms).")
        if pack.telemetry and pack.telemetry["missing_ratio"] >= MISSING_RATIO and score >= 0.6:
            score = min(0.97, score + 0.05)
            findings.append("Telemetry gaps coincide with network degradation.")
    return _report("network", {Cause.network_issue: score}, findings, refs, checks)


def rules_security(alert: AlertIn, pack: EvidencePack) -> SpecialistReport:
    s, findings, refs = pack.security, [], []
    checks = ["enrollment_state", "security_events", "acl_violations"]
    score = 0.05
    if s is None:
        findings.append("Security events unavailable - trust not verified.")
        score = 0.1
    else:
        refs = [r for r in pack.refs if r.startswith(("security:", "asset:"))]
        state = s.get("enrollment_state")
        counts = s.get("event_counts_60m", {})
        denials = sum(counts.get(k, 0) for k in ("broker_denied", "enroll_rejected", "auth_failed"))
        acl = int(_num((pack.network or {}).get("acl_violations")))
        if state in ("unclaimed", "revoked"):
            score = 0.95
            findings.append(f"Enrollment state is '{state}' - device must not be trusted.")
        elif denials >= 3 or acl >= 1:
            score = 0.85
            findings.append(f"{denials} broker/enrollment denials and {acl} ACL violations in the last hour.")
        elif denials > 0:
            score = 0.4
            findings.append(f"{denials} denial(s) in the last hour - below alert threshold.")
        else:
            findings.append("Enrollment operational and no denials or ACL violations.")
    return _report("security", {Cause.security_incident: score}, findings, refs, checks)


def rules_maintenance(alert: AlertIn, pack: EvidencePack) -> SpecialistReport:
    m, findings, refs = pack.maintenance, [], []
    score = 0.1
    if m is None:
        findings.append("Maintenance record unavailable.")
    else:
        refs = [r for r in pack.refs if r.startswith("maintenance:")]
        health, days = _num(m.get("health_score"), 1.0), int(_num(m.get("days_since_maintenance")))
        if health < 0.6:
            score = 0.6
            findings.append(f"Health score {health:.2f} is low - wear-related fault plausible.")
        elif days > 180:
            score = 0.4
            findings.append(f"{days} days since last maintenance.")
        else:
            findings.append(f"Health score {health:.2f}, {days} days since maintenance - no wear indication.")
    return _report("maintenance", {Cause.physical_fault: score}, findings, refs, ["maintenance_record"])


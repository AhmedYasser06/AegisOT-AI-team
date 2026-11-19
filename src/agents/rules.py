"""Deterministic specialists (no LLM). They are:
  * the fallback when the LLM is disabled, down, or returns invalid output (proposal §10.2)
  * the baseline the agent workflow is compared against in the evaluation (proposal §10.3)
Thresholds are starting points - tune them with your real sensor data.
"""
from __future__ import annotations

from .enums import Cause
from .schemas import AlertIn, SpecialistReport
from .tools import EvidencePack

ELEVATED_RATIO = 1.5      # channel >= 1.5x baseline counts as elevated
LOW_RATIO = 0.6           # channel <= 0.6x baseline counts as dropped (e.g. RPM falling)
MISSING_RATIO = 0.3       # >=30% of expected samples missing = incomplete telemetry
# Celsius is not a ratio scale (26.5 -> 33 C is only 1.25x but it IS overheating), so temperature
# channels are judged by how many degrees they rose above baseline instead of a ratio.
RISE_DEGREES = {"temperature": 5.0, "temperature_c": 5.0}


def _report(role, scores, findings, refs, checks) -> SpecialistReport:
    return SpecialistReport(role=role, engine="rules", scores=scores, findings=findings,
                            evidence_refs=refs, checks_performed=checks)


def _num(value, default: float = 0.0) -> float:
    """Backends may send null / strings for a field: treat anything unusable as `default` instead of crashing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _off_baseline(telemetry: dict, skip: set[str]) -> list[str]:
    """Findings for channels far from their baseline: temperature by degrees, the rest by ratio
    (>= 1.5x, or <= 0.6x including 0 = stalled)."""
    out = []
    for name, ch in telemetry["channels"].items():
        if name in skip:
            continue
        if name in RISE_DEGREES:
            delta = ch.get("delta_to_baseline")
            if delta is not None and delta >= RISE_DEGREES[name]:
                out.append(f"{name} is {delta:+.1f} above its baseline over the recent window.")
            continue
        ratio = ch["ratio_to_baseline"]
        if ratio is None:                       # None = no baseline; 0.0 is a real value (channel dropped to zero)
            continue
        if ratio >= ELEVATED_RATIO or 0 <= ratio <= LOW_RATIO:
            out.append(f"{name} is {ratio}x its baseline over the recent window.")
    return out


def rules_diagnosis(alert: AlertIn, pack: EvidencePack) -> SpecialistReport:
    scores = {Cause.physical_fault: 0.05, Cause.sensor_fault: 0.05}
    findings, refs, checks = [], [], ["telemetry_summary", "anomaly_details"]
    t = pack.telemetry
    if t is None:
        findings.append("Telemetry unavailable - cannot confirm a machine or sensor fault.")
    else:
        refs += [r for r in pack.refs if r.startswith("telemetry:")]
        bad = set(t["flatlined"]) | set(t["implausible"])
        if bad:
            scores[Cause.sensor_fault] = 0.85
            findings.append(f"Channel(s) {sorted(bad)} flatlined or outside plausible range - likely sensor fault.")
        elevated = _off_baseline(t, skip=bad)
        if t["missing_ratio"] >= MISSING_RATIO:
            scores[Cause.physical_fault] = 0.2
            findings.append(f"{t['missing_ratio']:.0%} of expected samples missing - machine fault cannot be confirmed.")
        elif elevated:
            scores[Cause.physical_fault] = round(min(0.97, 0.5 + 0.5 * alert.anomaly_score), 3)
            findings += elevated
        elif not bad:
            findings.append("All channels are close to baseline.")
    if pack.anomaly:
        refs += [r for r in pack.refs if r.startswith("anomaly:")]
        feats = [f["name"] for f in pack.anomaly.get("top_features", [])[:3]]
        if feats:
            findings.append(f"Anomaly model top features: {', '.join(feats)}.")
    return _report("diagnosis", scores, findings, refs, checks)


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


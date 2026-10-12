"""Safe action policy (proposal §10.2 / §11.2).

The LLM never picks or executes an action. The supervisor picks from this
allowlist; anything else is replaced by `escalate_to_human`.
This service only PROPOSES. The backend executes after human approval.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import ActionId, Cause, Risk


@dataclass(frozen=True)
class ActionPolicy:
    risk: Risk
    requires_approval: bool
    auto_allowed: bool


ALLOWLIST: dict[ActionId, ActionPolicy] = {
    ActionId.notify_operator:         ActionPolicy(Risk.low, False, True),
    ActionId.run_network_diagnostics: ActionPolicy(Risk.low, False, True),
    ActionId.request_sensor_recheck:  ActionPolicy(Risk.low, False, True),
    ActionId.inspect_equipment:       ActionPolicy(Risk.low, False, True),
    ActionId.escalate_to_human:       ActionPolicy(Risk.low, False, True),
    ActionId.stop_equipment:          ActionPolicy(Risk.high, True, False),
    ActionId.isolate_device:          ActionPolicy(Risk.high, True, False),
    ActionId.apply_network_template:  ActionPolicy(Risk.high, True, False),
}


def coerce_action(raw: str | ActionId) -> ActionId:
    """Anything outside the allowlist becomes escalate_to_human."""
    try:
        action = ActionId(raw)
    except ValueError:
        return ActionId.escalate_to_human
    return action if action in ALLOWLIST else ActionId.escalate_to_human


def choose_action(cause: Cause, confidence: float, anomaly_score: float) -> tuple[ActionId, dict]:
    if cause == Cause.physical_fault:
        if anomaly_score >= 0.85 and confidence >= 0.75:
            return ActionId.stop_equipment, {"mode": "controlled_stop"}
        return ActionId.inspect_equipment, {}
    if cause == Cause.network_issue:
        if confidence >= 0.8:
            return ActionId.apply_network_template, {"template": "restore_path"}
        return ActionId.run_network_diagnostics, {}
    if cause == Cause.security_incident:
        return ActionId.isolate_device, {"mode": "quarantine"}
    if cause == Cause.sensor_fault:
        return ActionId.request_sensor_recheck, {}
    return ActionId.escalate_to_human, {}


# Which causes each specialist is allowed to score (guardrail: an agent cannot vote outside its domain).
ALLOWED_CAUSES: dict[str, set[Cause]] = {
    "diagnosis": {Cause.physical_fault, Cause.sensor_fault},
    "network": {Cause.network_issue},
    "security": {Cause.security_incident},
    "maintenance": {Cause.physical_fault},
}

ACTION_TEXT: dict[ActionId, str] = {
    ActionId.notify_operator: "Notify the operator",
    ActionId.run_network_diagnostics: "Run read-only network diagnostics",
    ActionId.request_sensor_recheck: "Re-check or recalibrate the sensor",
    ActionId.inspect_equipment: "Inspect the machine (load, bearings, alignment)",
    ActionId.escalate_to_human: "Escalate to an engineer for manual inspection",
    ActionId.stop_equipment: "Perform a controlled stop of the equipment",
    ActionId.isolate_device: "Isolate the device in the quarantine zone",
    ActionId.apply_network_template: "Apply the approved network recovery template",
}

DIAGNOSIS_TEXT: dict[Cause, str] = {
    Cause.physical_fault: "Possible machine fault (wear, overload or mechanical problem)",
    Cause.sensor_fault: "Possible sensor fault (reading not trustworthy)",
    Cause.network_issue: "Possible network / communication problem",
    Cause.security_incident: "Possible security incident (untrusted device or policy violation)",
    Cause.unknown: "Inconclusive: evidence is weak or contradictory",
}

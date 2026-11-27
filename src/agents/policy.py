"""Safe action policy.

The LLM never picks or executes an action. The supervisor picks from this
allowlist; anything else is replaced by `escalate_to_human`.
This service only PROPOSES. The backend executes after human approval.

There is no canned text here: an action is an id (`check_motor_temperature`, `check_motor_speed`, ...) plus
`params` (the channels involved). The sentences that explain it are written by the LLM or composed from the data.
"""
from __future__ import annotations

from dataclasses import dataclass

from .enums import ActionId, Cause, FaultType, Risk


@dataclass(frozen=True)
class ActionPolicy:
    risk: Risk
    requires_approval: bool
    auto_allowed: bool


_LOW = ActionPolicy(Risk.low, False, True)
_HIGH = ActionPolicy(Risk.high, True, False)

ALLOWLIST: dict[ActionId, ActionPolicy] = {
    ActionId.notify_operator:             _LOW,
    ActionId.run_network_diagnostics:     _LOW,
    ActionId.check_motor_temperature:     _LOW,
    ActionId.check_motor_vibration:       _LOW,
    ActionId.check_power_supply:          _LOW,
    ActionId.check_motor_speed:           _LOW,
    ActionId.recheck_temperature_sensor:  _LOW,
    ActionId.request_sensor_recheck:      _LOW,
    ActionId.inspect_equipment:           _LOW,
    ActionId.escalate_to_human:           _LOW,
    ActionId.stop_equipment:              _HIGH,
    ActionId.isolate_device:              _HIGH,
    ActionId.apply_network_template:      _HIGH,
}

# the specific check for each specific machine fault
CHECK_FOR_FAULT: dict[FaultType, ActionId] = {
    FaultType.overheating: ActionId.check_motor_temperature,
    FaultType.abnormal_vibration: ActionId.check_motor_vibration,
    FaultType.electrical_fault: ActionId.check_power_supply,
    FaultType.speed_deviation: ActionId.check_motor_speed,
}
TEMPERATURE_CHANNELS = {"temperature", "temperature_c"}


def coerce_action(raw: str | ActionId) -> ActionId:
    """Anything outside the allowlist becomes escalate_to_human."""
    try:
        action = ActionId(raw)
    except ValueError:
        return ActionId.escalate_to_human
    return action if action in ALLOWLIST else ActionId.escalate_to_human


# Thresholds that escalate a proposal to a high-impact action (which always needs human approval).
STOP_MIN_ANOMALY, STOP_MIN_CONFIDENCE = 0.85, 0.75
NETWORK_TEMPLATE_MIN_CONFIDENCE = 0.8


def inspection_action(fault_type: FaultType | None, channels: list[str] | tuple[str, ...] = ()) -> tuple[ActionId, dict]:
    """The low-impact check that matches the fault ("check the motor temperature", "re-check the temperature
    sensor", ...). Falls back to the generic inspect_equipment only when the fault is not specific."""
    params = {"channels": list(channels)} if channels else {}
    if fault_type == FaultType.sensor_fault:
        if channels and all(c in TEMPERATURE_CHANNELS for c in channels):
            return ActionId.recheck_temperature_sensor, params
        return ActionId.request_sensor_recheck, params
    return CHECK_FOR_FAULT.get(fault_type, ActionId.inspect_equipment), params


def choose_action(cause: Cause, confidence: float, anomaly_score: float, fault_type: FaultType | None = None,
                  channels: list[str] | tuple[str, ...] = ()) -> tuple[ActionId, dict]:
    if cause == Cause.physical_fault:
        if anomaly_score >= STOP_MIN_ANOMALY and confidence >= STOP_MIN_CONFIDENCE:
            return ActionId.stop_equipment, {"mode": "controlled_stop", **({"channels": list(channels)} if channels else {})}
        return inspection_action(fault_type, channels)
    if cause == Cause.network_issue:
        if confidence >= NETWORK_TEMPLATE_MIN_CONFIDENCE:
            return ActionId.apply_network_template, {"template": "restore_path"}
        return ActionId.run_network_diagnostics, {}
    if cause == Cause.security_incident:
        return ActionId.isolate_device, {"mode": "quarantine"}
    if cause == Cause.sensor_fault:
        return inspection_action(FaultType.sensor_fault, channels)
    return ActionId.escalate_to_human, {}


def ml_gate(action: ActionId, params: dict, ml_validated: bool, fault_type: FaultType | None = None,
            channels: list[str] | tuple[str, ...] = ()) -> tuple[ActionId, dict, str | None]:
    """Until the ML model is proven on NEW recordings, never propose stopping the machine: run the specific check."""
    if action == ActionId.stop_equipment and not ml_validated:
        new, new_params = inspection_action(fault_type, channels)
        return (new, new_params,
                f"ML model not yet validated on new recordings (ML_VALIDATED=false): stop_equipment "
                f"downgraded to {new.value}.")
    return action, params, None

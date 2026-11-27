"""Closed vocabularies shared by every layer (like `models/enums` in mini-rag)."""
from __future__ import annotations

from enum import Enum


class Cause(str, Enum):
    """The BROAD category (who must act). The specific fault is `FaultType`."""
    physical_fault = "physical_fault"
    sensor_fault = "sensor_fault"
    network_issue = "network_issue"
    security_incident = "security_incident"
    unknown = "unknown"


class FaultType(str, Enum):
    """The SPECIFIC fault, named after what is wrong (shown to the operator)."""
    overheating = "overheating"                  # temperature above its baseline
    abnormal_vibration = "abnormal_vibration"    # gyro / accel magnitude (or vibration_rms) above its baseline
    electrical_fault = "electrical_fault"        # current / bus voltage / shunt voltage / power off baseline
    speed_deviation = "speed_deviation"          # rpm / speed_percent off baseline
    machine_fault = "machine_fault"              # physical, but no known channel group explains it
    sensor_fault = "sensor_fault"                # a channel is flat, invalid or implausible
    network_issue = "network_issue"
    security_incident = "security_incident"
    unknown = "unknown"


# Fault types the diagnosis LLM may choose from (it only sees the machine/sensor domain).
LLM_FAULT_TYPES = (FaultType.overheating, FaultType.abnormal_vibration, FaultType.electrical_fault,
                   FaultType.speed_deviation, FaultType.sensor_fault, FaultType.machine_fault)


class ActionId(str, Enum):
    # read-only / low impact (may be automated by policy). Each one names WHAT to check.
    notify_operator = "notify_operator"
    run_network_diagnostics = "run_network_diagnostics"
    check_motor_temperature = "check_motor_temperature"        # overheating
    check_motor_vibration = "check_motor_vibration"            # abnormal_vibration
    check_power_supply = "check_power_supply"                  # electrical_fault
    check_motor_speed = "check_motor_speed"                    # speed_deviation
    recheck_temperature_sensor = "recheck_temperature_sensor"  # sensor_fault on a temperature channel
    request_sensor_recheck = "request_sensor_recheck"          # sensor_fault on any other channel
    inspect_equipment = "inspect_equipment"                    # physical fault with no known channel group
    escalate_to_human = "escalate_to_human"
    # high impact (ALWAYS needs human approval)
    stop_equipment = "stop_equipment"
    isolate_device = "isolate_device"
    apply_network_template = "apply_network_template"


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

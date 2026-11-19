"""Closed vocabularies shared by every layer (like `models/enums` in mini-rag)."""
from __future__ import annotations

from enum import Enum


class Cause(str, Enum):
    physical_fault = "physical_fault"
    sensor_fault = "sensor_fault"
    network_issue = "network_issue"
    security_incident = "security_incident"
    unknown = "unknown"


class ActionId(str, Enum):
    # read-only / low impact (may be automated by policy)
    notify_operator = "notify_operator"
    run_network_diagnostics = "run_network_diagnostics"
    request_sensor_recheck = "request_sensor_recheck"
    inspect_equipment = "inspect_equipment"
    escalate_to_human = "escalate_to_human"
    # high impact (ALWAYS needs human approval)
    stop_equipment = "stop_equipment"
    isolate_device = "isolate_device"
    apply_network_template = "apply_network_template"


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

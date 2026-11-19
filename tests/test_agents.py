from src.agents.policy import ALLOWLIST, choose_action, coerce_action
from src.agents.schemas import ActionId, Cause
"""End-to-end agent_graph runs on the mock upstreams (LLM disabled -> pure rules)."""
import pytest

from src.agents.graph import run_incident
from src.agents.testing import make_alert

CASES = [
    # asset, score, cause, action, needs_approval
    ("esp32-conveyor-01", 0.93, "physical_fault", "stop_equipment", True),
    ("esp32-conveyor-net", 0.70, "network_issue", "apply_network_template", True),
    ("esp32-rogue-01", 0.88, "security_incident", "isolate_device", True),
    ("esp32-sensor-01", 0.80, "sensor_fault", "request_sensor_recheck", False),
]


@pytest.mark.parametrize("asset,score,cause,action,approval", CASES)
def test_scenarios(agent_graph, asset, score, cause, action, approval):
    a = run_incident(agent_graph, make_alert(asset, score, incident_id=f"i-{asset}"))
    assert a.status == "complete"
    assert a.likely_cause.value == cause
    assert a.recommended_action.value == action
    assert a.approval_required is approval
    assert a.auto_executable is (not approval)
    assert a.evidence_refs and a.checks_performed and a.engine == "rules"


def test_unknown_asset_is_inconclusive_and_escalated(agent_graph):
    a = run_incident(agent_graph, make_alert("esp32-unknown", 0.3))
    assert a.status == "inconclusive" and a.likely_cause.value == "unknown"
    assert a.recommended_action.value == "escalate_to_human"


def test_network_gap_does_not_blame_machine(agent_graph):
    a = run_incident(agent_graph, make_alert("esp32-conveyor-net", 0.99))
    assert a.likely_cause.value == "network_issue"          # proposal 6.3
    assert any("down-weighted" in n for n in a.notes)


def test_low_anomaly_physical_fault_only_inspects(agent_graph):
    a = run_incident(agent_graph, make_alert("esp32-conveyor-01", 0.5))
    assert a.recommended_action.value == "inspect_equipment" and not a.approval_required




def test_high_impact_actions_always_need_approval():
    for a in (ActionId.stop_equipment, ActionId.isolate_device, ActionId.apply_network_template):
        assert ALLOWLIST[a].requires_approval and not ALLOWLIST[a].auto_allowed


def test_every_action_is_in_allowlist():
    assert set(ActionId) == set(ALLOWLIST)


def test_unknown_action_becomes_escalation():
    assert coerce_action("rm -rf /") == ActionId.escalate_to_human
    assert coerce_action("shutdown_plant") == ActionId.escalate_to_human
    assert coerce_action("stop_equipment") == ActionId.stop_equipment


def test_choose_action_mapping():
    assert choose_action(Cause.physical_fault, 0.9, 0.9)[0] == ActionId.stop_equipment
    assert choose_action(Cause.physical_fault, 0.9, 0.6)[0] == ActionId.inspect_equipment
    assert choose_action(Cause.network_issue, 0.6, 0.9)[0] == ActionId.run_network_diagnostics
    assert choose_action(Cause.network_issue, 0.9, 0.9)[0] == ActionId.apply_network_template
    assert choose_action(Cause.security_incident, 0.9, 0.9)[0] == ActionId.isolate_device
    assert choose_action(Cause.unknown, 0.1, 0.9)[0] == ActionId.escalate_to_human


# ───────── README scenario: "RPM anomaly" (speed DROPS below baseline) ─────────
def test_rpm_drop_is_detected_as_physical_fault(agent_settings):
    from src.agents.clients import Clients
    from src.agents.clients.mock import MockAnomalyClient, MockBackendClient, MockNetworkClient, _samples

    class SlowMotorBackend(MockBackendClient):
        def get_telemetry(self, asset_id, minutes, end=None):
            d = super().get_telemetry(asset_id, minutes, end)
            d["samples"] = _samples(asset_id, minutes, fault_scale={"speed": 0.4})
            return d

    from src.agents.graph import build_graph
    g = build_graph(agent_settings, Clients(SlowMotorBackend(), MockAnomalyClient(), MockNetworkClient()))
    a = run_incident(g, make_alert("esp32-unknown-motor", 0.9))
    assert a.likely_cause.value == "physical_fault"
    assert any("speed" in e and "baseline" in e for e in a.evidence)

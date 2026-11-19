from src.agents.clients.mock import FailingClient
from src.agents.clients import Clients
from src.agents.crew_tools import build_crewai_tools
from src.agents.roles import SPECIALISTS
from src.agents.tools import ToolBox, collect_evidence, summarize_telemetry


def test_summary_detects_flatline_implausible_and_missing():
    raw = {"sampling_interval_s": 1, "window_minutes": 1, "baseline": {"t": 40},
           "samples": [{"ts": i, "temperature": -127.0, "v": 1.0 + i * 0.01} for i in range(30)]}
    s = summarize_telemetry(raw)
    assert "temperature" in s["flatlined"] and "temperature" in s["implausible"]
    assert s["n_samples"] == 30 and s["missing_ratio"] == 0.5


def test_collect_evidence_records_refs_and_unavailable(agent_clients):
    pack = collect_evidence(agent_clients, "esp32-conveyor-01", "esp32-conveyor-01:1")
    assert not pack.unavailable
    assert "network:esp32-conveyor-01:health" in pack.refs and "anomaly:esp32-conveyor-01:1" in pack.refs
    broken = Clients(backend=agent_clients.backend, anomaly=agent_clients.anomaly, network=FailingClient())
    pack2 = collect_evidence(broken, "esp32-conveyor-01", "esp32-conveyor-01:1")
    assert pack2.unavailable == ["network"] and pack2.network is None


def test_toolbox_registers_refs_clamps_window_and_returns_errors_as_data(agent_clients):
    refs = {}
    box = ToolBox(agent_clients, "esp32-conveyor-01", "esp32-conveyor-01:1", refs)
    out = box.run("telemetry", minutes=999)                   # clamped to 30
    assert out["ref"].endswith("last30m") and out["ref"] in refs
    bad = ToolBox(Clients(FailingClient(), FailingClient(), FailingClient()), "x", "y", {})
    assert "error" in bad.run("network")


def test_role_tools_are_least_privilege_and_buildable(agent_clients):
    box = ToolBox(agent_clients, "esp32-conveyor-01", "esp32-conveyor-01:1", {})
    for role, spec in SPECIALISTS.items():
        tools = build_crewai_tools(box, role)
        assert len(tools) == len(spec.sources)
    assert "network" not in SPECIALISTS["maintenance"].sources and "security" not in SPECIALISTS["diagnosis"].sources


def test_missing_baseline_falls_back_to_window_start():
    raw = {"sampling_interval_s": 1, "window_minutes": 1,
           "samples": [{"ts": i, "vibration_rms": 1.0 if i < 30 else 3.0} for i in range(60)]}
    ch = summarize_telemetry(raw)["channels"]["vibration_rms"]
    assert ch["baseline_source"] == "window_start" and ch["ratio_to_baseline"] >= 2.5


# ───────── rules: regressions found by probing edge cases ─────────
def _diag(samples, baseline=None):
    from src.agents.rules import rules_diagnosis
    from src.agents.schemas import Cause
    from src.agents.testing import make_alert
    from src.agents.tools import EvidencePack
    raw = {"sampling_interval_s": 1, "window_minutes": 1, "baseline": baseline or {"speed": 1500.0}, "samples": samples}
    pack = EvidencePack(asset_id="a", telemetry=summarize_telemetry(raw), refs={"telemetry:a:last1m": "t"})
    return rules_diagnosis(make_alert(), pack).scores[Cause.physical_fault]


def test_channel_dropping_to_zero_is_a_machine_fault_not_ignored():
    stalled = [{"ts": i, "speed": 1500.0 if i < 30 else 0.0} for i in range(60)]
    assert _diag(stalled) >= 0.9                       # was 0.05: ratio 0.0 was treated as "no baseline"
    healthy = [{"ts": i, "speed": 1500.0 + (i % 3)} for i in range(60)]
    assert _diag(healthy) < 0.5


def test_null_fields_from_a_backend_do_not_crash_the_rules():
    from src.agents.rules import rules_maintenance, rules_network, rules_security
    from src.agents.testing import make_alert
    from src.agents.tools import EvidencePack
    pack = EvidencePack(asset_id="a", refs={"network:a:health": "n", "maintenance:a": "m", "security:a:events60m": "s"},
                        network={"latency_ms": None, "packet_loss_pct": None, "acl_violations": None, "reachable": True},
                        maintenance={"health_score": None, "days_since_maintenance": None},
                        security={"enrollment_state": "operational", "event_counts_60m": {}})
    for fn in (rules_network, rules_security, rules_maintenance):
        fn(make_alert(), pack)

from src.agents.clients.mock import FailingClient
from src.agents.clients import Clients
from src.agents.tools import collect_evidence, summarize_telemetry
from src.agents.tools import ROLE_TOOLS, ToolBox, build_crewai_tools


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
    out = box.telemetry(minutes=999)                   # clamped to 30
    assert out["ref"].endswith("last30m") and out["ref"] in refs
    bad = ToolBox(Clients(FailingClient(), FailingClient(), FailingClient()), "x", "y", {})
    assert "error" in bad.network()


def test_role_tools_are_least_privilege_and_buildable(agent_clients):
    box = ToolBox(agent_clients, "esp32-conveyor-01", "esp32-conveyor-01:1", {})
    for role, names in ROLE_TOOLS.items():
        tools = build_crewai_tools(box, role)
        assert len(tools) == len(names)
    assert "network" not in ROLE_TOOLS["maintenance"] and "security" not in ROLE_TOOLS["diagnosis"]


def test_missing_baseline_falls_back_to_window_start():
    raw = {"sampling_interval_s": 1, "window_minutes": 1,
           "samples": [{"ts": i, "vibration_rms": 1.0 if i < 30 else 3.0} for i in range(60)]}
    ch = summarize_telemetry(raw)["channels"]["vibration_rms"]
    assert ch["baseline_source"] == "window_start" and ch["ratio_to_baseline"] >= 2.5

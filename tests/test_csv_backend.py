"""CSV replay backend (BACKEND_MODE=csv): the agents read the same recorded rows the ML model scored."""
import numpy as np
import pandas as pd
import pytest

from src.agents.clients import ClientError, Clients, build_clients
from src.agents.clients.csv_backend import ROOT, CsvBackendClient
from src.agents.clients.mock import MockAnomalyClient, MockNetworkClient
from src.agents.graph import build_graph, run_incident
from src.agents.schemas import AlertIn
from src.agents.settings import Settings

CSV = ROOT / "data" / "processed" / "final.csv"
needs_csv = pytest.mark.skipif(not CSV.exists(), reason="data/processed/final.csv is missing")


def _tiny_csv(tmp_path, rows):
    p = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _row(i, temp=26.0, valid=1, cond="normal", gyro=0.03, gap=0.0):
    t = pd.Timestamp("2026-11-20T10:00:00Z") + pd.Timedelta(seconds=0.2 * i + gap)
    return {"host_received_timestamp_utc": t.isoformat(), "condition": cond, "temperature_c": temp,
            "temperature_valid": valid, "gyro_x_rad_s": gyro, "gyro_y_rad_s": 0.0, "gyro_z_rad_s": 0.0, "motor_running": 1}


def test_labels_never_reach_the_agents_and_invalid_temperature_is_missing(tmp_path):
    rows = [_row(i, valid=0 if i == 3 else 1) for i in range(10)]
    c = CsvBackendClient(_tiny_csv(tmp_path, rows))
    t = c.get_telemetry("a", 1)
    assert len(t["samples"]) == 10
    assert all("condition" not in s and "host_received_timestamp_utc" not in s for s in t["samples"])
    assert t["samples"][3]["temperature_c"] is None and t["samples"][2]["temperature_c"] == 26.0
    assert "gyro_mag" in t["baseline"] and abs(t["sampling_interval_s"] - 0.2) < 1e-6


def test_window_ends_at_the_ml_timestamp_and_never_crosses_a_pause(tmp_path):
    rows = [_row(i) for i in range(5)] + [_row(i, gap=60) for i in range(5, 10)]        # a 60 s pause after row 4
    c = CsvBackendClient(_tiny_csv(tmp_path, rows))
    end_first_part = rows[4]["host_received_timestamp_utc"]
    assert len(c.get_telemetry("a", 5, end_first_part)["samples"]) == 5                   # stops at `end`
    assert len(c.get_telemetry("a", 5, rows[9]["host_received_timestamp_utc"])["samples"]) == 5   # not across the pause
    assert c.get_telemetry("a", 5, "2020-01-01T00:00:00Z")["samples"] == []              # before the recording
    # window_minutes = what was really returned, so a short segment does not look like "missing samples"
    t = c.get_telemetry("a", 5, rows[4]["host_received_timestamp_utc"])
    assert abs(t["window_minutes"] - 5 * 0.2 / 60) < 1e-6


def test_missing_file_is_a_clear_error_and_build_clients_picks_the_csv_mode(tmp_path):
    with pytest.raises(ClientError):
        CsvBackendClient(str(tmp_path / "nope.csv"))
    s = Settings(_env_file=None, backend_mode="csv", backend_csv=_tiny_csv(tmp_path, [_row(i) for i in range(5)]))
    assert isinstance(build_clients(s).backend, CsvBackendClient)


@needs_csv
def test_agents_diagnose_from_the_recorded_rows_not_from_mock_data():
    """Hot rows in the CSV -> physical_fault from temperature_c; a cool normal row -> not a fault."""
    df = pd.read_csv(CSV)
    hot = df[df.condition == "overheated"].sort_values("temperature_c").iloc[-1]
    cool = df[df.condition == "normal"].sort_values("temperature_c").iloc[0]
    s = Settings(_env_file=None, anomaly_mode="inline", llm_enabled=False, ml_validated=True,
                 telemetry_channels="temperature_c,gyro_mag")
    graph = build_graph(s, Clients(CsvBackendClient(str(CSV)), MockAnomalyClient(), MockNetworkClient()))

    a = run_incident(graph, AlertIn(incident_id="hot", anomaly_score=0.9, timestamp=hot.host_received_timestamp_utc))
    assert a.likely_cause.value == "physical_fault" and any("temperature_c" in f for f in a.evidence)

    b = run_incident(graph, AlertIn(incident_id="cool", anomaly_score=0.6, timestamp=cool.host_received_timestamp_utc))
    assert b.likely_cause.value != "physical_fault" and b.recommended_action.value == "escalate_to_human"

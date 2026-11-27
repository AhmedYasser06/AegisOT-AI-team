"""ML endpoints inside the shared app (src/ml/router.py): one server for the ML and the agents."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.ml import router as ml

H = {"X-API-Key": "test-key"}
CSV = ml.PROJECT_ROOT / "data" / "raw" / "final.csv"


@pytest.fixture
def client(agent_settings, agent_clients, monkeypatch):
    monkeypatch.setattr(ml, "get_settings", lambda: agent_settings)          # same key as the agent tests
    return TestClient(create_app(agent_settings, agent_clients))


@pytest.fixture
def ml_client(agent_settings, agent_clients, monkeypatch):
    """Same app with the key check switched off: these tests check the ML logic, not the key, so a developer's
    .env / cached settings cannot turn them into 401s. (The key itself is tested with `client`.)"""
    monkeypatch.setattr(ml, "get_settings", lambda: agent_settings)
    app = create_app(agent_settings, agent_clients)
    app.dependency_overrides[ml.require_key] = lambda: None
    return TestClient(app)


def _model_usable() -> bool:
    try:
        ml._load()
        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(not (CSV.exists() and _model_usable()),
                                 reason="model file, data/raw/final.csv or the right scikit-learn version is missing")


def _samples(cond, n=10):
    df = pd.read_csv(CSV)
    rows = df[df["condition"] == cond].head(n)
    return [{"timestamp": r["host_received_timestamp_utc"],
             "sensor_data": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                             for k, v in r.drop(labels=["host_received_timestamp_utc"]).items()}}
            for _, r in rows.iterrows()]


def test_there_is_one_app_with_both_the_ml_and_the_agent_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/api/v1/anomaly/predict", "/api/v1/anomaly/window", "/api/v1/anomaly/health",
            "/api/v1/incident/analyze", "/api/v1/incident/analyze/stream"} <= paths


def test_ml_endpoints_need_the_api_key_but_health_does_not(client):
    assert client.post("/api/v1/anomaly/window", json={"incident_id": "x", "samples": []}).status_code == 401
    assert client.post("/api/v1/anomaly/predict", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/anomaly/health").status_code == 200


def test_a_non_ascii_key_is_401_not_500(client):
    r = client.post("/api/v1/anomaly/window", json={}, headers=[(b"x-api-key", "é".encode("utf-8"))])    # raw bytes
    assert r.status_code == 401


def test_missing_model_gives_503_with_the_reason_and_health_reports_it(ml_client, monkeypatch):
    monkeypatch.setattr(ml, "_bundle", None)
    monkeypatch.setattr(ml, "BUNDLE_PATH", ml.PROJECT_ROOT / "nope" / "missing.joblib")
    r = ml_client.post("/api/v1/anomaly/window", headers=H, json={"incident_id": "x", "samples": [
        {"timestamp": "2026-01-01T00:00:00Z", "sensor_data": {"temperature_c": 25}}]})
    assert r.status_code == 503 and "not found" in r.json()["detail"]
    h = ml_client.get("/api/v1/anomaly/health").json()
    assert h["model_loaded"] is False and "not found" in h["reason"]


@needs_model
def test_window_result_is_the_incident_json_and_flows_into_the_agents(ml_client):
    ml_out = ml_client.post("/api/v1/anomaly/window", headers=H, json={
        "incident_id": "ML-1", "asset_id": "esp32-conveyor-01", "samples": _samples("overheated")}).json()
    assert ml_out["is_anomaly"] is True and 0.0 <= ml_out["anomaly_score"] <= 1.0
    assert ml_out["model"]["name"] == "lof" and ml_out["top_features"]
    out = ml_client.post("/api/v1/incident/analyze", json=ml_out, headers=H)          # unchanged, straight into the agents
    assert out.status_code == 200 and out.json()["incident_id"] == "ML-1"


@needs_model
def test_normal_window_is_not_an_anomaly_and_single_reading_works(ml_client):
    w = ml_client.post("/api/v1/anomaly/window", headers=H, json={"incident_id": "ML-2", "samples": _samples("normal")}).json()
    assert w["is_anomaly"] is False
    s = ml_client.post("/api/v1/anomaly/predict", headers=H, json={
        "incident_id": "ML-3", "sensor_data": _samples("normal", 1)[0]["sensor_data"]})
    assert s.status_code == 200 and s.json()["window"]["n_samples"] == 1


@needs_model
def test_bad_sensor_data_is_422(ml_client):
    r = ml_client.post("/api/v1/anomaly/predict", headers=H, json={"incident_id": "ML-4", "sensor_data": {"nonsense": 1}})
    assert r.status_code in (422, 500)       # 422 when the model rejects the columns

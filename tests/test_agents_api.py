H = {"X-API-Key": "test-key"}
BODY = {"incident_id": "inc-api-1", "alert_id": "esp32-conveyor-01:1", "asset_id": "esp32-conveyor-01",
        "anomaly_score": 0.93, "model_version": "iforest-0.1"}


def test_health_is_open(agent_client):
    r = agent_client.get("/health")
    assert r.status_code == 200 and r.json()["integrations"]["backend"] == "mock"


def test_auth_required(agent_client):
    assert agent_client.post("/api/v1/incident/analyze", json=BODY).status_code == 401
    assert agent_client.post("/api/v1/incident/analyze", json=BODY, headers={"X-API-Key": "nope"}).status_code == 401


def test_analyze_returns_contract(agent_client):
    r = agent_client.post("/api/v1/incident/analyze?view=full", json=BODY, headers=H)
    assert r.status_code == 200
    d = r.json()
    for k in ("incident_id", "asset_id", "evidence_refs", "likely_cause", "alternatives", "confidence",
              "recommended_action", "risk_level", "approval_required", "checks_performed", "status"):
        assert k in d
    assert d["likely_cause"] == "physical_fault" and d["approval_required"] is True


def test_validation_errors(agent_client):
    bad = {**BODY, "anomaly_score": 1.7}
    assert agent_client.post("/api/v1/incident/analyze", json=bad, headers=H).status_code == 422
    assert agent_client.post("/api/v1/incident/analyze", json={"asset_id": "x"}, headers=H).status_code == 422


def test_idempotent_retry_and_force(agent_client):
    first = agent_client.post("/api/v1/incident/analyze", json=BODY, headers=H).json()
    again = agent_client.post("/api/v1/incident/analyze", json=BODY, headers=H).json()
    assert first["created_at"] == again["created_at"]
    forced = agent_client.post("/api/v1/incident/analyze?force=true", json=BODY, headers=H).json()
    assert forced["created_at"] != first["created_at"]


def test_get_and_list_incidents(agent_client):
    agent_client.post("/api/v1/incident/analyze", json=BODY, headers=H)
    assert agent_client.get("/api/v1/incident/inc-api-1", headers=H).status_code == 200
    assert agent_client.get("/api/v1/incident/nope", headers=H).status_code == 404
    assert len(agent_client.get("/api/v1/incidents", headers=H).json()) == 1


def test_debug_endpoints(agent_client):
    ev = agent_client.get("/api/v1/agents/evidence/esp32-conveyor-net?alert_id=esp32-conveyor-net:1", headers=H).json()
    assert ev["network"]["packet_loss_pct"] == 38.0 and ev["telemetry"]["missing_ratio"] > 0.3
    nh = agent_client.get("/api/v1/agents/network-health/esp32-conveyor-net", headers=H).json()
    assert nh["ref"] == "network:esp32-conveyor-net:health"


def test_actions_and_llm_endpoints(agent_client):
    acts = agent_client.get("/api/v1/agents/actions", headers=H).json()
    assert acts["stop_equipment"]["requires_approval"] is True
    llm = agent_client.get("/api/v1/agents/llm", headers=H).json()
    assert llm["enabled"] is False and set(llm["roles"]) == {"diagnosis", "supervisor"}

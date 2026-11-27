#!/usr/bin/env bash
# Full check of the agent layer. Run from the repo root with the .venv active.
set -u
PORT=8001; PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

[ -f pytest.ini ] && [ -d src/agents ] || { echo "Run this from the repo root (needs pytest.ini and src/agents)"; exit 2; }
KEY=$(grep -s '^AGENT_API_KEY=' .env | cut -d= -f2-); KEY=${KEY:-dev-key}

echo "1) automated tests"
pytest > /tmp/pytest_full.txt 2>&1; RC=$?
SUMMARY=$(grep -E "passed|failed|error" /tmp/pytest_full.txt | tail -1)
[ $RC -eq 0 ] && ok "pytest: ${SUMMARY:-all tests passed}" || bad "pytest exit code $RC: ${SUMMARY} (run: pytest -x)"

echo "2) demo workflow (python -m src.agents.cli)"
OUT=$(LLM_ENABLED=false ML_VALIDATED=true python -m src.agents.cli 2>&1)
for c in "physical_fault" "network_issue" "security_incident" "sensor_fault" "INCONCLUSIVE"; do
  echo "$OUT" | grep -q "$c" && ok "demo shows $c" || bad "demo missing $c"; done

echo "3) API on port $PORT (rules only, no LLM)"
LLM_ENABLED=false ML_VALIDATED=true uvicorn src.api.main:app --port $PORT > /tmp/agents_api.log 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 40); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
curl -s localhost:$PORT/health | grep -q '"status":"ok"' && ok "/health" || bad "/health (log: /tmp/agents_api.log)"
post() { curl -s -X POST "localhost:$PORT$1" -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d "$2"; }
exp() { # asset expected_action expected_approval
  R=$(post /api/v1/diagnosis/analyze "{\"asset_id\":\"$1\",\"incident_id\":\"CHK-$1\",\"anomaly_score\":0.9,\"model\":\"isolation_forest\"}")
  echo "$R" | python -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if (d.get('action_id'),d.get('approval_required'))==('$2',$3) else 1)" 2>/dev/null \
    && ok "$1 -> $2 (approval=$3)" || bad "$1 expected $2: $R"; }
exp esp32-conveyor-01 stop_equipment True
exp esp32-conveyor-net apply_network_template True
exp esp32-rogue-01 isolate_device True
exp esp32-sensor-01 recheck_temperature_sensor False
exp esp32-unknown-99 escalate_to_human False
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$PORT/api/v1/incident/analyze -H 'Content-Type: application/json' -d '{}')" = 401 ] && ok "no key -> 401" || bad "no key should be 401"
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$PORT/api/v1/incident/analyze -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{"asset_id":"x","incident_id":"y","anomaly_score":5}')" = 422 ] && ok "score 5 -> 422" || bad "bad score should be 422"
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:$PORT/api/v1/incident/analyze -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d '{"asset_id":"x","incident_id":"z","is_anomaly":false,"anomaly_score":0.9}')" = 422 ] && ok "is_anomaly=false -> 422" || bad "is_anomaly=false should be 422"

# streaming endpoint: events must arrive in order and end with `done`
STREAM=$(curl -s -N -X POST "localhost:$PORT/api/v1/incident/analyze/stream" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"incident_id":"CHK-STREAM","anomaly_score":0.9,"model":{"name":"isolation_forest","version":"0.1"}}')
echo "$STREAM" | grep '^event:' | tr '\n' ' ' | grep -q "accepted.*evidence_collected.*preliminary.*vote.*decision.*done" \
  && ok "SSE stream: accepted > evidence > preliminary > votes > decision > done (asset_id omitted)" || bad "SSE stream order: $STREAM"

echo "4) repo hygiene"
if git rev-parse --git-dir >/dev/null 2>&1; then
  git ls-files --error-unmatch .env >/dev/null 2>&1 && bad ".env is tracked by git (remove it: git rm --cached .env)" || ok ".env not tracked"
  git check-ignore -q .env && ok ".env is git-ignored" || bad ".env is NOT in .gitignore"
else echo "  (not a git repo here, skipped)"; fi


echo; echo "RESULT: $PASS passed, $FAIL failed"; [ $FAIL -eq 0 ]

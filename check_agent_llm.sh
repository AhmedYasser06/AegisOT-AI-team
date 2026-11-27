#!/usr/bin/env bash
# Full check of the agent layer. Run from the repo root with the venv/conda env active.
#
# Usage:
#   bash check_agents_llm.sh              # rules-only checks (same as before)
#   bash check_agents_llm.sh --llm        # rules-only checks + real LLM test (step 5)
#   bash check_agents_llm.sh --only-llm   # LLM test only (fast, uses 1 request)
set -u
PORT=8001; LLM_PORT=8002; PASS=0; FAIL=0
RUN_RULES=1; RUN_LLM=0
case "${1:-}" in
  --llm)      RUN_LLM=1 ;;
  --only-llm) RUN_LLM=1; RUN_RULES=0 ;;
  "")         ;;
  *) echo "Unknown option: $1 (use --llm or --only-llm)"; exit 2 ;;
esac

ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

[ -f pytest.ini ] && [ -d src/agents ] || { echo "Run this from the repo root (needs pytest.ini and src/agents)"; exit 2; }

# read a value from .env: strips inline comments, spaces and quotes
envval() { grep -s "^$1=" .env | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//'; }
KEY=$(envval AGENT_API_KEY); KEY=${KEY:-dev-key}

PIDS=""
trap 'for p in $PIDS; do kill $p 2>/dev/null; done' EXIT

if [ $RUN_RULES -eq 1 ]; then
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
PIDS="$PIDS $!"
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

echo "4) repo hygiene"
if git rev-parse --git-dir >/dev/null 2>&1; then
  git ls-files --error-unmatch .env >/dev/null 2>&1 && bad ".env is tracked by git (remove it: git rm --cached .env)" || ok ".env not tracked"
  git check-ignore -q .env && ok ".env is git-ignored" || bad ".env is NOT in .gitignore"
else echo "  (not a git repo here, skipped)"; fi
[ ! -f src/agents/empty_file.txt ] && ok "placeholder removed" || echo "  NOTE  delete src/agents/empty_file.txt (git rm)"
fi

if [ $RUN_LLM -eq 1 ]; then
echo "5) real LLM test on port $LLM_PORT (uses a few requests of your API quota)"
PROVIDER=$(envval LLM_PROVIDER); PROVIDER=${PROVIDER:-gemini}
case "$PROVIDER" in
  gemini)    KEYVAR=GEMINI_API_KEY ;;
  openai)    KEYVAR=OPENAI_API_KEY ;;
  anthropic) KEYVAR=ANTHROPIC_API_KEY ;;
  ollama)    KEYVAR="" ;;
  *)         KEYVAR="" ;;
esac
if [ -n "$KEYVAR" ] && [ -z "$(envval $KEYVAR)" ] && [ -z "${!KEYVAR:-}" ]; then
  bad "$KEYVAR is empty in .env (provider=$PROVIDER). Put a valid key there first."
else
  # free-tier friendly: only diagnosis + supervisor use the LLM, one call at a time
  LLM_ENABLED=true LLM_AGENTS="${LLM_AGENTS:-diagnosis,supervisor}" LLM_MAX_CONCURRENT=1 \
    uvicorn src.api.main:app --port $LLM_PORT > /tmp/agents_api_llm.log 2>&1 &
  PIDS="$PIDS $!"
  for i in $(seq 1 60); do curl -s localhost:$LLM_PORT/health >/dev/null 2>&1 && break; sleep 0.5; done
  H=$(curl -s localhost:$LLM_PORT/health)
  echo "$H" | grep -q '"llm_enabled": *true' && ok "/health reports llm_enabled=true" || bad "llm_enabled is not true: $H (log: /tmp/agents_api_llm.log)"

  echo "  ... calling /incident/analyze with the LLM (can take up to ~2 min)"
  R=$(curl -s -m 180 -X POST localhost:$LLM_PORT/api/v1/incident/analyze \
        -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
        -d '{"asset_id":"esp32-conveyor-01","incident_id":"LLM-CHECK","anomaly_score":0.9,"model":"isolation_forest"}')
  RES=$(echo "$R" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print('BADJSON'); sys.exit(0)
llm = d.get('llm') or {}
print('OK' if (d.get('status') and llm.get('status') == 'ok') else 'RULES_ONLY', '|', llm.get('status'), '|', llm.get('model'), '|', d.get('fault_type'), '|', (d.get('action') or {}).get('id'), '|', llm.get('error'))
" 2>/dev/null)
  case "$RES" in
    OK*)       ok "LLM answered -> $RES" ;;
    BADJSON*)  bad "no valid JSON from the API: $R (log: /tmp/agents_api_llm.log)" ;;
    *)         bad "LLM was NOT used (fell back to rules): $RES  -> check /tmp/agents_api_llm.log for 429 / model-not-found / invalid key" ;;
  esac
fi
fi

echo; echo "RESULT: $PASS passed, $FAIL failed"; [ $FAIL -eq 0 ]
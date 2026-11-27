# Testing the ML model, the agents and the full system

Run everything from the repo root with the virtual environment active.

    pip install -r requirements.txt scikit-learn==1.7.2 joblib      # the model was saved with scikit-learn 1.7.2
    cp -n .env.example .env

`.env` (minimum):

    AGENT_API_KEY=dev-key
    LLM_ENABLED=true
    LLM_MODEL=gemini/gemini-flash-lite-latest
    GEMINI_API_KEY=<your key>
    LLM_RETRIES=2                                  # retry Gemini 503 "high demand" before the rules fallback
    # LLM_FALLBACK_MODEL=gemini-flash-lite-latest

Needed files: `models/best_model.joblib` (the model) and `data/processed/final.csv` (the recording).

## THE ONE RULE that avoids confusing results
The agents read their telemetry from `BACKEND_MODE`. For ML -> agents tests it MUST be `csv`, so the agents diagnose the
SAME rows the model scored. With the default `mock` they diagnose made-up data (a vibrating conveyor) whatever the ML
said, and an "overheated" window comes back as `abnormal_vibration`. Check `GET /health`: `integrations.backend` must be `csv`.
(`scripts/full_pipeline_check.py --start` and `scripts/test_full_system.py --start` set it for you.)

---------------------------------------------------------------------------------------------------------------------
## 1. Unit tests (no server, no LLM, no model)
    pytest -q                       # ~140 tests
    bash check_agents.sh            # tests + demo + API on port 8001 (rules only)

## 2. The ML model ALONE

### 2a. Terminal, no server
    python scripts/evaluate_model.py

### 2b. Terminal, through the API
Start the server (one terminal):

    BACKEND_MODE=csv BACKEND_CSV=data/processed/final.csv uvicorn src.api.main:app --port 8001

Another terminal:

    curl http://localhost:8001/api/v1/anomaly/health
    curl -s -X POST http://localhost:8001/api/v1/anomaly/predict -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/1_ml_predict_overheated.json
    curl -s -X POST http://localhost:8001/api/v1/anomaly/window  -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/2_ml_window_overheated.json
    curl -s -X POST http://localhost:8001/api/v1/anomaly/window  -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/2_ml_window_vibration.json
    curl -s -X POST http://localhost:8001/api/v1/anomaly/window  -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/2_ml_window_normal.json

Expected: `normal` -> `"is_anomaly": false`; `overheated` / `vibration` -> `"is_anomaly": true` with `top_features`
(temperature_c ... / gyro_* ...). The answer IS the input of the agents (section 3).

### 2c. FastAPI docs  ->  http://localhost:8001/docs
1. Open `POST /api/v1/anomaly/window` -> Try it out.
2. Fill the `x-api-key` field with `dev-key`.
3. Paste the content of `examples/2_ml_window_overheated.json` as the body -> Execute.
(`/predict` takes one reading: `examples/1_ml_predict_*.json`.)

## 3. The AGENTS alone (the input is an ML answer)
Terminal:

    curl -s -X POST http://localhost:8001/api/v1/incident/analyze -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/3_agents_input_overheated.json
    curl -s -X POST http://localhost:8001/api/v1/incident/analyze -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/3_agents_input_vibration.json
    curl -s -X POST http://localhost:8001/api/v1/incident/analyze -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/3_agents_input_normal.json      # 422: not an anomaly
    curl -s -X POST "http://localhost:8001/api/v1/incident/analyze?view=full&force=true" -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/3_agents_input_overheated.json   # the complete record
    curl -N -X POST http://localhost:8001/api/v1/incident/analyze/stream -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/3_agents_input_overheated.json                      # live events (SSE)

Docs: `POST /api/v1/incident/analyze` -> `x-api-key=dev-key`, `view=compact` (short answer, default) or `full`,
`force=true` to re-run an incident_id that was already analysed -> paste the JSON -> Execute.
Without a server, quick demo on mock data: `python -m src.agents.cli --asset esp32-hot-01 --score 0.91`.

What to look at in the answer: `fault_type`, `headline`, `recommended_solution`, `action.id`, `confidence`, `reason`
(only when inconclusive) and **`llm.status`**: `ok` = the Gemini diagnosis answered, `fallback_rules` = it failed (see
`llm.error`), `disabled` = LLM off.

## 4. FULL SYSTEM: sensor window -> ML -> agents -> JSON

### 4a. Terminal, one command (CSV -> windows -> ML -> agents -> table)
    python scripts/test_full_system.py --start                                  # rules only, fast
    python scripts/test_full_system.py --start --llm --limit 3 --show 3         # with Gemini on 3 anomalies per condition
    python scripts/test_full_system.py --start --llm --limit 3 --save out.json  # keep every answer

### 4b. Terminal, the self-checking script (fails with exit code 1 if anything is wrong)
    python scripts/full_pipeline_check.py --key dev-key --start --expect-llm

It checks: server, model loaded, ML verdict per condition, 401/422, the answer's fields, that `overheated` gives
`overheating` and `vibration_issue` gives `abnormal_vibration`, idempotency, the compact view, the live stream and
that the LLM really answered.

### 4c. Terminal, by hand (two calls: the ML answer is pasted unchanged into the agents)
    curl -s -X POST http://localhost:8001/api/v1/anomaly/window -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @examples/2_ml_window_overheated.json > ml.json
    curl -s -X POST http://localhost:8001/api/v1/incident/analyze -H "X-API-Key: dev-key" -H "Content-Type: application/json" -d @ml.json

(`jq .is_anomaly ml.json` first: the agents answer 422 when it is false.) The server must have been started with
`BACKEND_MODE=csv`, and the window must come from the recording (the files in `examples/` do).

### 4d. FastAPI docs  ->  http://localhost:8001/docs
1. `POST /api/v1/anomaly/window` with `examples/2_ml_window_overheated.json` -> Execute -> copy the whole response body.
2. `POST /api/v1/incident/analyze` with that response as the body -> Execute.
3. Read `fault_type`, `headline`, `recommended_solution`, `llm.status`.
4. Repeat with `2_ml_window_vibration.json`. Use `force=true` to re-run the same `incident_id`.

## 4e. FastAPI docs, step by step (everything from the browser)
Start: `BACKEND_MODE=csv BACKEND_CSV=data/processed/final.csv uvicorn src.api.main:app --port 8001`, open http://localhost:8001/docs
1. Click **Authorize** (top right) is NOT needed: the key is a header field in each endpoint. Use `dev-key` (your AGENT_API_KEY).
2. `GET /health` -> Execute: `integrations.backend` must be `csv`, `llm_enabled` true.
3. `GET /api/v1/anomaly/health` -> `model_loaded: true`.
4. `POST /api/v1/anomaly/window` -> x-api-key=`dev-key`, body = `examples/2_ml_window_overheated.json` -> Execute.
   Select the WHOLE response body and copy it.
5. `POST /api/v1/incident/analyze` -> x-api-key=`dev-key`, view=`compact` (or `full`), force=`true`, body = the copied
   response -> Execute. Read `fault_type`, `severity`, `headline`, `recommended_solution`, `llm.status`.
6. Repeat 4-5 with `2_ml_window_vibration.json`; with `2_ml_window_normal.json` step 5 answers 422 (not an anomaly).
7. `GET /api/v1/incidents` (list) and `GET /api/v1/incident/{incident_id}` (one) read what was stored.
8. Wrong/missing key -> 401 on every endpoint except /health.

## 4f. Demo UI (Gradio) - for presenting
    pip install gradio
    python scripts/demo_ui.py --key dev-key          # needs the server of section 2b running; open http://localhost:7860
Tabs: Full system (ML + agents) / ML model / What-if. Notes for the video: `docs/PROGRESS_VIDEO.md`.

## 5. Production-like check of the LLM
    bash check_agent_llm.sh         # 'OK' = Gemini answered, 'RULES_ONLY' = it did not (reason printed)

## Troubleshooting
| You see | Meaning |
|---|---|
| `integrations.backend: mock` in /health | the agents read fake telemetry; restart with `BACKEND_MODE=csv` |
| `llm.status: fallback_rules` + `503 ... high demand` | Gemini was overloaded; retried `LLM_RETRIES` times, then the rules answered. Set `LLM_FALLBACK_MODEL` or run again |
| `422 is_anomaly=false` | the ML said "normal"; only anomalies create incidents |
| `status: inconclusive` + `reason` | evidence too weak or contradictory; the action is `escalate_to_human` |
| `LiteLLM ... temperature ... Gemini 3+` warning | harmless deprecation notice, nothing to fix |

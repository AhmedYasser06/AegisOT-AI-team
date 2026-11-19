> **Note:** the module layout changed, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current map (new: `roles.py`, `service.py`, `cli.py`, `crew_tools.py`, `enums.py`).

# `src/agents/`: Supervised Agentic AI (LangGraph + crewAI)

The **Agentic AI diagnosis layer** of AegisOT (proposal §10). It receives an **anomaly result from the ML pipeline**,
collects evidence (telemetry, network, security, maintenance), lets specialist agents reason over it, and returns a
**validated, evidence-cited diagnosis** with a **recommended action**. It never executes anything: authorization and
execution belong to the AegisOT backend after a human approves (this is the README's "AI Safety Rules").

| Piece | Role |
|---|---|
| **LangGraph** (`graph.py`) | The fixed, auditable workflow: collect evidence → 4 specialists in parallel → supervisor |
| **crewAI** (`agents.py`) | The specialist agents (diagnosis, network, security, maintenance), each with read-only tools |
| **LiteLLM** (under both) | Provider-agnostic LLM layer. **No model is hardcoded**: Gemini by default, switch in `.env` |
| **Rules** (`rules.py`) | Every specialist also exists as deterministic code: the fallback when the LLM is off/down/invalid, **and the baseline** to compare the agents against |
| **FastAPI** (`router.py` → `src/api/main.py`) | The endpoints the Backend/Laravel team calls |

**Status:** 65 automated tests pass (Python 3.12, crewai 1.15, langgraph 1.2, litellm 1.104, fastapi 0.142), including a
test that runs the **real crewAI + LangGraph** code against a local fake OpenAI-compatible server. **Not verified:**
reasoning quality of a real Gemini/GPT/Claude/Llama on your data. You must measure that (§11).

---

## 0. Decisions you should review (read first)

1. **Folder is `src/agents/`** (plural), as in the repo's structure, not `src/agent/`. The README commands depend on it (`python -m src.agents.graph`).
2. **The graph runs the specialists in parallel, not in a chain.** The repo README sketches Monitoring → Network → Security → Diagnosis → Maintenance sequentially. I made them independent so no agent is biased by another's conclusion and latency is one LLM round instead of four. The **Diagnosis step ("combine evidence")** is the deterministic **supervisor**, deliberately *not* an LLM, so no model can talk its way into `stop_equipment` without approval. The **Monitoring agent** = the `collect_evidence` node. Challenge this if your supervisor/Team Lead wants a literal chain; the change is local to `graph.py`.
3. **ML result goes in the request** (`anomaly_mode=inline`). The README's "AI Result Example" JSON is accepted exactly as is; nothing else has to be built on the ML side to start (§7.1).
4. **`recommended_action` is a stable machine id** (`stop_equipment`); the human text is `recommended_action_text` (and `recommended_action` in the compact `/diagnosis/analyze` format). Agree with the Backend team which one they use.
5. **Case warning:** the zip I received has `src/ML/` but the README says `src/ml/`. Linux/CI is case-sensitive, Windows is not. Pick one (`git mv src/ML src/ml_tmp && git mv src/ml_tmp src/ml`) before the ML team writes code, or imports like `src.ml.router` will fail on Linux.
6. **Shared files touched** (merge carefully with other PRs): `src/api/main.py`, `requirements.txt`, `.env.example`, `.gitignore`, `configs/config.yaml`, `pytest.ini`, `tests/conftest.py`. All were empty in the zip I received.

---

## 1. Where it fits

```
 Sensors → Preprocessing → Features → ML detection ──(is_anomaly=true)──┐
                                                                         ▼
                                          POST /api/v1/incident/analyze  (this layer, port 8001)
 Backend (Laravel) ◄── GET evidence ──┐                                  │
 Network service  ◄── GET health ─────┤                                  ▼
                                      └───────────  collect_evidence → 4 specialists → supervisor
                                                                         │
            assessment JSON (response + optional POST /api/agent/assessments to Laravel)
                                                                         ▼
                      Laravel: approval request → operator approves → Laravel executes (bounded)
```

```
START → collect_evidence ─┬→ diagnosis ───┐   crewAI Agent + read-only tools
 (Monitoring)             ├→ network ─────┤   (or rules fallback)
                          ├→ security ────┼→ supervisor → END     merge + policy gate + allowlist
                          └→ maintenance ─┘                        + "inconclusive" logic
```

| Node | LLM? | What it does |
|---|---|---|
| `collect_evidence` | no | Calls backend / network clients, takes the ML result from the request, builds an `EvidencePack`; every source gets an **evidence ref** |
| `diagnosis` | crewAI | Physical fault vs sensor fault from telemetry. Tools: machine status, sensor data, anomaly details |
| `network` | crewAI | Is it a communication problem? Tools: network status, sensor data |
| `security` | crewAI | Untrusted device / policy violation? Tools: security events, machine status, network status |
| `maintenance` | crewAI | Does health/maintenance history support a physical fault? Tools: machine history, machine status |
| `supervisor` | optional | **Deterministic** merge → cause, confidence, action, approval. The LLM may only reword `explanation` |

---

## 2. Files

```
src/agents/
├── README.md        this file
├── __init__.py
├── agents.py        crewAI agents/tasks/crews + the output sanitizer (the guardrails)
├── graph.py         LangGraph workflow  +  `python -m src.agents.graph` demo CLI
├── tools.py         tools named as in the repo README: get_sensor_data, get_machine_status,
│                    get_network_status, get_security_events, get_machine_history (+ get_anomaly_details),
│                    EvidencePack, telemetry summary, per-agent crewAI tool wrappers
├── rules.py         deterministic specialists (fallback AND evaluation baseline)
├── supervisor.py    merge specialist reports → IncidentAssessment (pure functions, no LLM)
├── policy.py        action ALLOWLIST (risk, approval), allowed causes per agent, human texts
├── schemas.py       versioned contracts: AlertIn (team "AI Result" format), IncidentAssessment, DiagnosisOut
├── settings.py      config: env > .env > configs/config.yaml > defaults
├── llm.py           provider/model resolution (global + per-agent override), LiteLLM completion
├── router.py        FastAPI router (the agent endpoints)
├── store.py         in-memory incident store (bounded)
├── testing.py       tiny helper used by the tests
└── clients/         one adapter per upstream: base.py (interfaces), http.py, mock.py, __init__.py (mode switch)

src/api/main.py      shared AI API: /health + agent router (+ auto-includes src/ml/router.py when the ML team adds it)
configs/config.yaml  non-secret config (`agents:` section) .env.example  requirements.txt  .gitignore  pytest.ini
tests/               conftest.py, test_agents.py, test_agents_{guardrails,tools,integrations,llm_config,api,
                     repo_compat,crewai_e2e}.py
```
(Delete the placeholder `src/agents/empty_file.txt`; it is no longer needed.)

---

## 3. Add it to the repo, install, test, run (step by step)

Follow the repo's Git workflow: `main` is protected, work in a branch, open a Pull Request, the Team Lead merges.
Commands are for Windows PowerShell where they differ; Linux/macOS/WSL equivalents in comments.

**Step 1: latest `main` and your branch**
```bash
git checkout main
git pull origin main
git checkout -b feature/langgraph-agents
```

**Step 2: copy the files in.** Unzip `agents-changes.zip` **at the repository root** (paths inside match the repo) and
overwrite when asked. Then:
```bash
git rm src/agents/empty_file.txt           # placeholder no longer needed
git status                                  # see the list below
```
Expected new/changed paths: `.env.example`, `.gitignore`, `pytest.ini`, `requirements.txt`, `configs/config.yaml`,
`src/agents/*`, `src/api/main.py`, `tests/conftest.py`, `tests/test_agents*.py`. If a teammate already changed one of
the shared files on `main`, **merge by hand instead of overwriting** (§0 item 6).

**Step 3: environment + dependencies** (first install takes a few minutes: crewAI is large)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1                  # Linux/macOS/WSL: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Python 3.10–3.13 (crewAI requirement).

**Step 4: configuration**
```powershell
copy .env.example .env                      # Linux/macOS/WSL: cp .env.example .env
```
Edit `.env`: set `AGENT_API_KEY` to your own value. Leave `LLM_ENABLED=false` for now (rules only, no API key needed).
`.env` is git-ignored; never commit it.

**Step 5: run the tests: everything must be green**
```bash
pytest
```
Expected: **`65 passed`** in about 10 s, offline, no API key. (`pytest.ini` puts the repo root on the Python path.)

**Step 6: run the agent workflow demo** (the command from the repo README)
```bash
python -m src.agents.graph
```
Expected: five incidents, each with cause, action, approval, evidence and a short explanation:

| Mock `asset_id` | Scenario | Expected result |
|---|---|---|
| `esp32-conveyor-01` | vibration 2.6× and current 1.7× baseline, network fine | `physical_fault` (98%) → `stop_equipment`, approval **required** |
| `esp32-conveyor-net` | 65% telemetry missing + 38% packet loss | `network_issue` (95%) → `apply_network_template`, approval required; machine **not** blamed |
| `esp32-rogue-01` | enrollment `unclaimed`, broker denials, ACL hits | `security_incident` (95%) → `isolate_device`, approval required |
| `esp32-sensor-01` | temperature stuck at −127 | `sensor_fault` (85%) → `request_sensor_recheck`, no approval |
| anything else (`esp32-unknown-99`) | healthy data, weak anomaly | `INCONCLUSIVE` → `escalate_to_human` |

One asset: `python -m src.agents.graph --asset esp32-conveyor-01 --score 0.93 --json`.

**Step 7: run the API** (port 8001, as in the repo README)
```bash
uvicorn src.api.main:app --reload --port 8001        # or:  python -m src.api.main
```
Open http://localhost:8001/docs (Try it out; fill the `x-api-key` header field). From a terminal:
```powershell
# PowerShell
$body = '{"asset_id":"esp32-conveyor-01","incident_id":"INC-001","is_anomaly":true,"anomaly_score":0.91,"model":"isolation_forest","timestamp":"2026-10-01T12:30:00Z"}'
Invoke-RestMethod -Uri http://localhost:8001/api/v1/incident/analyze -Method Post `
  -Headers @{"X-API-Key"="<your AGENT_API_KEY>"} -ContentType "application/json" -Body $body
```
```bash
# Linux / macOS / WSL / Git Bash
curl -s -X POST localhost:8001/api/v1/incident/analyze -H "X-API-Key: <your AGENT_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"asset_id":"esp32-conveyor-01","incident_id":"INC-001","is_anomaly":true,"anomaly_score":0.91,"model":"isolation_forest","timestamp":"2026-10-01T12:30:00Z"}'
```
Expected: `"likely_cause":"physical_fault"`, `"recommended_action":"stop_equipment"`, `"approval_required":true`,
`"engine":"rules"`. `GET /health` needs no key.

**Step 8: commit and open the Pull Request** (convention from the repo README)
```bash
git add .
git commit -m "feat: add supervised agent workflow (LangGraph + crewAI) and agent API endpoints"
git push -u origin feature/langgraph-agents
```
On GitHub: *Compare & pull request* → base `main`. Do not merge it yourself; the Team Lead does.
Optional follow-up PR: add one line to the root README pointing to `src/agents/README.md`.

### "Everything is good" checklist
- [ ] `pytest` → 65 passed
- [ ] `python -m src.agents.graph` → the 5 results in the table above
- [ ] `GET /health` → `status: ok`; `POST /api/v1/incident/analyze` → `physical_fault` for `esp32-conveyor-01`
- [ ] `GET /api/v1/agents/llm` shows the models you intend (§4)
- [ ] after §7: `GET /api/v1/agents/evidence/<real_asset_id>` shows real data and an empty `unavailable`
- [ ] `git status` shows **no** `.env`

---

## 4. Using any LLM (nothing hardcoded)

The LLM is **off by default** (`LLM_ENABLED=false` → rules only). Turn it on in `.env` (or `configs/config.yaml`):

```env
LLM_ENABLED=true
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
```
Restart, then `GET /api/v1/agents/llm` shows which model each agent will use (never secrets). Run Step 6/7 again and
check `"engine"` is `llm` or `mixed`, `llm_models_used` is filled, and `notes` has no "LLM path failed".

| Provider | `.env` |
|---|---|
| Gemini (team default) | `LLM_PROVIDER=gemini` · `GEMINI_API_KEY=…` (default `gemini/gemini-flash-latest`, an alias; see the model-name warning below) |
| OpenAI | `LLM_PROVIDER=openai` · `OPENAI_API_KEY=…` |
| Anthropic | `LLM_PROVIDER=anthropic` · `ANTHROPIC_API_KEY=…` |
| Local Ollama (free, offline) | `LLM_PROVIDER=ollama` · `LLM_MODEL=llama3.1` · `LLM_API_BASE=http://localhost:11434` |
| LM Studio / vLLM / any OpenAI-compatible server | `LLM_MODEL=openai/<served-name>` · `LLM_API_BASE=http://host:port/v1` · `LLM_API_KEY=anything` |
| Azure OpenAI | `LLM_MODEL=azure/<deployment>` · `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |

- **Model names get retired.** `gemini-2.0-flash` was shut down in 2026 and `gemini-2.5-flash` is reported to be scheduled for retirement in October 2026 (check Google's deprecations page). The default is therefore the `gemini-flash-latest` alias. To list the models your key can call and pin one for reproducible results:
  ```bash
  KEY=$(grep '^GEMINI_API_KEY=' .env | cut -d= -f2-)
  curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$KEY" | python -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin).get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]"
  ```
  Then set `LLM_MODEL=gemini/<name-without-the-models/-prefix>` in `.env` and restart.
- `LLM_MODEL` may be `provider/model` (exact) or just a model name (the provider is prepended from `LLM_PROVIDER`).
- **Different model per agent:** `LLM_MODEL_DIAGNOSIS`, `LLM_MODEL_NETWORK`, `LLM_MODEL_SECURITY`, `LLM_MODEL_MAINTENANCE`, `LLM_MODEL_SUPERVISOR` (e.g. a strong model for diagnosis, a cheap one for the rest). Put each provider's key in its own variable.
- Keys in `.env` are exported to the process automatically (LiteLLM reads them from the environment).
- **Free-tier rate limits (Gemini free tier: about 5 requests per minute *per model*).** One incident makes roughly 15 LLM calls (4 agents × tool call + answer + structured output, plus the supervisor), so a free key hits HTTP 429. Failed agents fall back to rules automatically, so you still get a result, but with fewer LLM-written parts (`engine=mixed`). Options, cheapest first:
  1. **Spread agents over different models.** The quota is counted per model, so each model has its own bucket (*not verified: free quota differs per model, so test*):
     ```env
     LLM_MODEL=gemini/gemini-3.5-flash
     LLM_MODEL_NETWORK=gemini/gemini-3.1-flash-lite
     LLM_MODEL_SECURITY=gemini/gemini-2.5-flash-lite
     LLM_MODEL_MAINTENANCE=gemini/gemini-3.5-flash-lite
     ```
  2. **Use the LLM only where it adds value:** `LLM_AGENTS=diagnosis,supervisor` (the other agents use the rules).
  3. **One agent at a time:** `LLM_MAX_CONCURRENT=1` (slower, fewer simultaneous requests; crewAI also waits and retries when Google says "retry in N s").
  4. **Wait about a minute between runs**, or use a billing-enabled key for real evaluations.
- Agents need **tool calling + structured output**. Small local models (≤8B) often fail; that specialist then falls back to rules and `notes` says `"<role>: LLM path failed …"`.
- Four crews run in parallel, so a free-tier Gemini key may hit **rate limits**; the same fallback applies. Expect **10–40 s** per incident with LLM on (~100 ms with rules only): give callers a timeout ≥ 90 s.

---

## 5. Endpoints (all except `/health` need header `X-API-Key: <AGENT_API_KEY>`)

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness, LLM on/off, mock/http/inline per integration |
| `POST /api/v1/incident/analyze` | **Main.** ML result in → full `IncidentAssessment` out. Query `push=true\|false` (post result to Laravel), `force=true` (re-run). **Idempotent** on `incident_id` |
| `POST /api/v1/diagnosis/analyze` | Same analysis, compact format from the repo README (`diagnosis`, `confidence`, `evidence`, `recommended_action`, `approval_required`) |
| `GET /api/v1/incident/{incident_id}` · `GET /api/v1/incidents?limit=20` | Stored results (404 if unknown) |
| `GET /api/v1/agents/evidence/{asset_id}?alert_id=` | Everything the agents would see. **Use it to verify each team's integration** |
| `GET /api/v1/agents/network-health/{asset_id}` | Proxy to the network service (502 on failure) |
| `GET /api/v1/agents/llm` · `GET /api/v1/agents/actions` | Model per agent (no secrets) · action allowlist with risk/approval rules |

`POST /api/v1/anomaly/predict` belongs to the **ML team**: add `src/ml/router.py` with `router = APIRouter()`; `src/api/main.py` includes it automatically (a missing module is ignored; any other import error is raised).

**Input** (the repo README's "AI Result Example" works as is):
```json
{"asset_id": "motor_01", "incident_id": "INC-001", "is_anomaly": true, "anomaly_score": 0.91,
 "model": "isolation_forest", "timestamp": "2026-10-01T12:30:00Z"}
```
Required: `incident_id`, `asset_id`, `anomaly_score` (0–1). Optional: `alert_id` (default = `incident_id`), `threshold`,
`station`, `summary`, `features{}`, and **`top_features: [{"name":"vibration_rms","contribution":0.7}]`** (strongly
recommended: it is the "why" the model fired). `model`/`timestamp` are aliases of `model_version`/`detected_at`.
`is_anomaly:false` → **422** (only anomalies create incidents).

**Output** (`/api/v1/incident/analyze`): `status` (`complete`|`inconclusive`), `likely_cause`, `confidence`,
`alternatives[]`, `diagnosis`, `evidence[]` (sentences), `evidence_refs[]` (ids), `findings[]`, `recommended_action`
(id), `recommended_action_text`, `action_params`, `risk_level`, `approval_required`, `auto_executable`,
`checks_performed[]`, `explanation`, `engine`, `llm_models_used[]`, `notes[]`.

Compact (`/api/v1/diagnosis/analyze`): `{incident_id, asset_id, diagnosis, confidence, evidence[], recommended_action (text), action_id, approval_required, status}`.

---

## 6. Configuration: where things go

| Where | What |
|---|---|
| `.env` (git-ignored) | **Secrets and local choices:** API keys, `AGENT_API_KEY`, `LLM_*`, `*_MODE`, URLs/tokens |
| `configs/config.yaml` → `agents:` | **Shared non-secret defaults** (keys = setting names, lowercase). The team's `api: host/port` block is honoured too |
| real environment variables | Override everything (CI, Docker) |

Priority: environment > `.env` > `config.yaml` > code defaults. Test the effective values with `GET /health` and `GET /api/v1/agents/llm`.

---

## 7. Connecting the other teams (one at a time)

Each integration has its own switch. After each one, call
`GET /api/v1/agents/evidence/<real_asset_id>?alert_id=<real_incident_id>` and confirm that source has real data and is
**not** in `unavailable`. If a source is down the agents still answer: `notes` lists it and confidence drops by 0.1 per
missing source. Upstream auth: `Authorization: Bearer <BACKEND_TOKEN|NETWORK_TOKEN|ANOMALY_TOKEN>` when set.
**The paths/fields below are my proposal. Agree them with each team, or change `src/agents/clients/http.py`.**

### 7.1 ML team (same repo): anomaly detection → agents
**Option A (recommended, already works): send the ML result in the request.** In `src/ml/predict.py` (or the FastAPI
route that wraps it), when `is_anomaly` is true, create an `incident_id` and post:
```python
import os, requests
requests.post("http://localhost:8001/api/v1/incident/analyze", timeout=120,
              headers={"X-API-Key": os.environ["AGENT_API_KEY"]},
              json={"asset_id": asset_id, "incident_id": incident_id, "is_anomaly": True,
                    "anomaly_score": score, "model": "isolation_forest", "timestamp": ts,
                    "top_features": [{"name": "vibration_rms", "contribution": 0.71}]})
```
**Option B: call in-process** (no HTTP; build the graph **once** at startup, reuse it):
```python
from src.agents.settings import get_settings
from src.agents.clients import build_clients
from src.agents.graph import build_graph, run_incident
from src.agents.schemas import AlertIn
graph = build_graph(get_settings(), build_clients(get_settings()))
assessment = run_incident(graph, AlertIn(asset_id=..., incident_id=..., anomaly_score=..., model="isolation_forest"))
```
**Option C: the agents pull details** from the ML side: `ANOMALY_MODE=python` +
`ANOMALY_FUNCTION=src.ml.predict:get_alert_details` (function takes `alert_id`, returns
`{score, threshold, model_version, top_features:[{name,contribution}]}`), or `ANOMALY_MODE=http` + `ANOMALY_URL` serving `GET /v1/alerts/{alert_id}`.

Requirements on the ML side:
- `anomaly_score` **normalized 0–1** (higher = more abnormal). The "stop equipment" rule (`policy.choose_action`) assumes ≥ 0.85 is severe.
- Give the backend a per-asset **`baseline`** (normal mean per channel from your training data). It turns "vibration 2.6× baseline" into evidence. Without it the agents fall back to the first third of the window (weaker: misses faults that began before the window).
- Your `features.py` channel names (`vibration_rms`, `current_mean`, …) should match the telemetry channel names the backend returns, or the diagnosis can't relate them.

### 7.2 Backend team (Laravel)
```env
BACKEND_MODE=http
BACKEND_URL=http://<laravel-host>:8000
BACKEND_TOKEN=<api token>
PUSH_RESULT_TO_BACKEND=true            # or per call: ?push=true
```
Laravel must **provide**:

| Endpoint | Response (JSON object) |
|---|---|
| `GET /api/devices/{asset_id}` | `{asset_id, station, type, enrollment_state: operational\|claimed\|unclaimed\|degraded\|revoked, firmware, profile, baseline?}` |
| `GET /api/devices/{asset_id}/telemetry?minutes=5` | `{asset_id, sampling_interval_s, window_minutes, baseline?: {channel: normal_value}, samples: [{ts, <channel>: number, …}]}` |
| `GET /api/devices/{asset_id}/events?kind=security&minutes=60` | `{events: [{ts, type: broker_denied\|enroll_rejected\|auth_failed\|…, detail}]}` |
| `GET /api/devices/{asset_id}/maintenance` | `{health_score: 0–1, days_since_maintenance, open_work_items}` |
| `POST /api/agent/assessments` | Receives the full `IncidentAssessment`; create the incident and, if `approval_required`, an approval request |

Laravel can also call this service directly instead of the ML code:
```php
$res = Http::withHeaders(['X-API-Key' => config('services.aegis_ai.key')])->timeout(120)
    ->post(config('services.aegis_ai.url').'/api/v1/incident/analyze', [
        'incident_id' => $incident->uuid, 'asset_id' => $device->asset_id,
        'anomaly_score' => $alert->score, 'model' => $alert->model]);
```
Backend rules: execute `recommended_action` **only after human approval** (or when `auto_executable` is true), re-check
it against the backend's own allowlist, then verify recovery with new telemetry. Channel names are free (any numeric key); `ts` is ISO-8601.

### 7.3 Network team
```env
NETWORK_MODE=http
NETWORK_URL=http://<net-host>:8300
```
They must **provide** `GET /network/devices/{asset_id}/health` (same asset id as the device):
```json
{"asset_id": "esp32-conveyor-01", "zone": "ot-segment", "reachable": true, "latency_ms": 12,
 "packet_loss_pct": 0.0, "link_status": "up", "interface": {"name": "Gi0/1", "status": "up"}, "acl_violations": 0}
```
`link_status` / `interface.status` ∈ `up|degraded|down`. Thresholds (edit in `rules.py`): loss ≥ 5% or latency ≥ 100 ms or
`degraded` → 0.6; loss ≥ 20% or latency ≥ 300 ms → 0.9; unreachable/down → 0.95. Test it alone with
`GET /api/v1/agents/network-health/<asset_id>`. The proposed action `apply_network_template {"template":"restore_path"}`
is **only a request**: agree the template names with them; the backend triggers their automation after approval.

### 7.4 Hardware / sensors
Agents never touch ESP32 or MQTT; data arrives through the backend's telemetry endpoint. Ask the hardware team for stable
channel names, units, the sampling interval (→ `sampling_interval_s`) and each sensor's "error value" (e.g. DS18B20
reads −127: add valid ranges to `PLAUSIBLE` in `tools.py`).

### 7.5 Frontend / mobile
Show `diagnosis`, `explanation`, `confidence`, `evidence`. Show **Approve/Reject only if `approval_required`**. If
`status` is `inconclusive`, show "needs human inspection".

---

## 8. Guardrails (repo "AI Safety Rules" / proposal §10.2)

| Rule | Where |
|---|---|
| No shell / SQL / router / device access exists in the tools; read-only, typed, allowlisted | `tools.py` |
| Each tool is bound to one asset (the LLM cannot pass another asset id) | `tools.ToolBox` |
| Least privilege per agent (maintenance agent cannot read network or security data) | `tools.ROLE_TOOLS` |
| An agent can only score its own causes | `policy.ALLOWED_CAUSES`, `agents.sanitize` |
| No evidence → no confidence: unknown refs dropped; a confident score without valid refs is capped at 0.3 | `agents.sanitize` |
| The LLM never picks the action; off-list → `escalate_to_human` | `policy.py`, `supervisor.py` |
| Stop / isolate / network change always `approval_required=true`, `auto_executable=false` | `policy.ALLOWLIST` |
| Inconclusive when confidence < 0.55, or winner beats runner-up by < 0.15, or no evidence | `supervisor.py`, `settings.py` |
| Any LLM/tool/validation error → deterministic rules for that agent | `graph.py`, `rules.py` |
| Device text is truncated and marked as data in prompts (prompt-injection hygiene) | `tools.py`, `agents.py` |
| Telemetry gaps + network degradation ⇒ don't blame the machine (proposal §6.3) | `supervisor.merge_reports` |

---

## 9. Tests

`pytest` runs 65 offline tests (no API key, no internet):

| File | Covers |
|---|---|
| `tests/test_agents.py` | The 5 mock scenarios through the whole graph, falling RPM, action policy |
| `tests/test_agents_guardrails.py` | Sanitizer, LLM failure → rules, LLM cannot force an action, narrative failure non-fatal |
| `tests/test_agents_tools.py` | Telemetry summary (flatline/implausible/missing/baseline fallback), evidence refs, tool clamping, least-privilege tools |
| `tests/test_agents_integrations.py` | HTTP client auth/retry/bad JSON, network down, everything down, backend push, `ANOMALY_MODE=python` |
| `tests/test_agents_llm_config.py` | Model resolution, per-agent override, env switching, secrets hidden |
| `tests/test_agents_api.py` | Auth, contract, 422 validation, idempotency, 404, debug endpoints |
| `tests/test_agents_repo_compat.py` | Team JSON formats, `is_anomaly:false`, inline ML result, `config.yaml`, `LLM_PROVIDER`, ML-router hook, the demo CLI |
| `tests/test_agents_crewai_e2e.py` | **Real crewAI + LangGraph** vs a local fake OpenAI-compatible server (tool call → answer → sanitize) |

---

## 10. Things you should change / tune

1. **`AGENT_API_KEY`**: strong value; share it with the Backend and ML code; keep the service on the internal network.
2. **Thresholds**: `rules.py` (`ELEVATED_RATIO`, `LOW_RATIO`, `MISSING_RATIO`, network limits), `min_confidence` / `min_margin` in `config.yaml`. Reasonable guesses: **tune them on real sensor data**.
3. **`PLAUSIBLE` ranges** in `tools.py` for every real sensor.
4. **Stop rule** in `policy.choose_action` (`anomaly_score ≥ 0.85` and `confidence ≥ 0.75`): match your model's calibration.
5. **Action / template names** in `policy.py`: must match what the Backend and Network teams can actually execute.
6. **`clients/mock.py`**: replace the channel names with the real ones for demos; switch to `http` modes for real runs.
7. **`store.py` is in-memory**: a restart loses history (the backend keeps it with `PUSH_RESULT_TO_BACKEND=true`).
8. **The endpoint is synchronous** (10–40 s with LLM). If callers must not block, add a queue or `202 + polling` (not implemented).
9. **Adding a cause** (e.g. `overload`): extend `Cause`, `ALLOWED_CAUSES`, `choose_action`, the rules and the tests together.
10. **Bump `SCHEMA_VERSION`** in `schemas.py` for any breaking change and tell the other teams.

---

## 11. Evaluation (the README's "Testing Scenarios")

| README scenario | How it is covered |
|---|---|
| 1 Normal operation | Not an agent case: ML says normal, nothing is sent. Weak/odd anomalies end `inconclusive` |
| 2 High temperature, 3 vibration, 4 current, 5 RPM | Any channel ≥1.5× or ≤0.6× its baseline → `physical_fault` (RPM-drop test included). Add per-channel mock scenarios in `clients/mock.py` |
| 6 Sensor failure | `esp32-sensor-01` (flatlined / implausible reading) → `sensor_fault` |
| 7 Network failure | `esp32-conveyor-net` → `network_issue`; the machine is not blamed |
| 8 Multiple simultaneous problems | Conflicting strong causes within 0.15 → `inconclusive` + escalation. **Add real cases and decide the expected behaviour with your supervisor** |

For the agents-vs-rules comparison promised in the proposal (§10.3):
1. Build 15–20 fixed incident cases with ground-truth labels, **including ambiguous ones** (high vibration *and* packet loss, borderline loss, a sensor spike during maintenance).
2. Run each with `LLM_ENABLED=false` (rules baseline) and `true` (agents), using `?force=true`. Compare `likely_cause` vs the label, evidence completeness, latency and whether approval rules held.
3. Report honestly: on easy cases the rules match the agents. The agents earn their place only on ambiguous multi-source cases. If the rules win, say so.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root (where `src/` and `pytest.ini` are) |
| `401 invalid or missing X-API-Key` | Send `X-API-Key` equal to `AGENT_API_KEY` in your `.env` (default `dev-key`) |
| `engine: rules` + notes "LLM path failed (…)" | The note now shows the reason. `ImportError … native provider` → `pip install -r requirements.txt` again (needs `google-genai` / `anthropic`). `NotFoundError` → model name retired or not available to your key (list models, section 4). `AuthenticationError` → wrong key. `RateLimitError` / 429 → free-tier limit (see the rate-limits bullet in section 4: spread models, `LLM_AGENTS`, `LLM_MAX_CONCURRENT=1`). Also check `GET /api/v1/agents/llm` |
| `unavailable: ["network"]` in evidence | Network service unreachable or not returning a JSON object; curl its URL; check `NETWORK_URL` / token |
| `ANOMALY_MODE=python` → `unavailable: ["anomaly"]` | `ANOMALY_FUNCTION` not importable from the repo root, or it didn't return a dict |
| Physical fault never detected on real data | No `baseline` and the fault began before the window, or channel names differ; inspect `/api/v1/agents/evidence/<id>` |
| Always `inconclusive` | Weak/contradictory evidence or thresholds too strict for your data; read `notes` and `evidence` |
| `src.ml...` imports fail on Linux/CI | Folder case: `src/ML` vs `src/ml` (§0 item 5) |
| `pip install` conflicts | Fresh `.venv`, Python 3.10–3.13 |

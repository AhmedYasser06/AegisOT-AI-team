# Agents layer: architecture

Same behaviour as before (verified on 21 mock scenarios, all API endpoints and the crewAI tool definitions);
only the structure changed. The layering follows the mini-rag project.

```
HTTP / CLI            router.py   cli.py                 thin: validate input, call the service, map errors to HTTP
Business logic        service.py                          IncidentService.analyze(): idempotency, run, push, store
Workflow              graph.py                            LangGraph: collect_evidence -> 4 specialists -> supervisor
Decisions             supervisor.py  policy.py            merge scores, inconclusive rule, allowlisted action
Specialists           roles.py  rules.py  agents.py       who they are | no-LLM logic | crewAI + sanitize guardrails
Evidence / tools      tools.py  crew_tools.py            read-only fetchers, one table feeds rules, graph and crewAI
Integrations          clients/  (base, mock, http)        Protocol + mock/http adapters + build_clients() factory
Contracts             enums.py  schemas.py                Cause/ActionId/Risk | AlertIn, reports, IncidentAssessment
Config                settings.py  llm.py                 .env + config.yaml  |  LiteLLM model resolution per role
Support               store.py  testing.py                in-memory incident store | test helper
```

| mini-rag | here |
|---|---|
| `routes/` | `router.py` |
| `controllers/` | `service.py` (`IncidentService`) |
| `stores/llm` interface + `LLMProviderFactory` | `clients/base.py` Protocols + `build_clients()`; LLM vendors are handled by LiteLLM, selected in `llm.py` |
| `models/enums` | `enums.py` |
| `models/db_schemes` / schemas | `schemas.py` |
| `helpers/config.py` (`Settings`, `get_settings`) | `settings.py` |

## Request flow

`POST /api/v1/incident/analyze` -> `router` -> `IncidentService.analyze`
-> `graph`: `collect_evidence` (tools.py) -> `diagnosis | network | security | maintenance` in parallel
(crewAI if `LLM_ENABLED` and the role is in `LLM_AGENTS`, otherwise `roles.SPECIALISTS[role].rules`; any LLM failure
falls back to rules) -> `supervisor` (`build_assessment`, optional LLM narrative that only rewrites the explanation)
-> `IncidentAssessment`.

## Where do I change X?

| I want to... | Edit |
|---|---|
| add / change a specialist (prompt, allowed causes, tools it may call) | one entry in `roles.py` + its function in `rules.py` |
| add a new evidence source or tool | a fetcher + one line in `ToolBox._fetch` (`tools.py`), a doc line in `crew_tools.py`, a field in `EvidencePack` |
| change thresholds for the rule-based specialists | constants at the top of `rules.py` |
| change which action is proposed / what needs approval | `policy.py` (`choose_action`, `ALLOWLIST`) |
| change how scores are merged or when a result is inconclusive | `supervisor.py` |
| plug in a real backend / anomaly / network service | `clients/http.py`, then `BACKEND_MODE=http` etc. |
| change provider or model | `.env` / `configs/config.yaml` (no code) |
| try it without the server | `python -m src.agents.cli` |

## What changed vs. the previous layout

* Role facts were scattered over 6 dicts (`PROFILES`, `SLICES`, `ROLE_TOOLS`, `ALLOWED_CAUSES`, `RULES`, `ROLES`); now `roles.py`.
* `tools.py` had every source written 3 times (collect lambdas, `ToolBox` methods, crewAI closures); now one table.
* `inline_anomaly` + `inline_first` were threaded through 5 signatures; now one `InlineAnomaly` object.
* `graph.py` no longer contains the CLI demo or the error formatter (`cli.py`, `llm.short_error`).
* `router.py` no longer contains the business logic (`service.py`); enums moved to `enums.py`.
* Rules fixes (behaviour change): a channel that dropped to 0 (stalled motor, zero current) is now detected; null fields from a backend no longer crash the rules.
* `supervisor.build_assessment` split into small helpers; magic numbers in `policy.py` got names.

Renamed for the tests: `graph.why` -> `llm.short_error`, `python -m src.agents.graph` -> `python -m src.agents.cli`,
`ToolBox.telemetry()/network()...` -> `ToolBox.run("telemetry")`, `tools.ROLE_TOOLS` -> `roles.SPECIALISTS[role].sources`.


## v1.1 (this change set)

* `asset_id` optional (`DEFAULT_ASSET_ID`); ML `model` may be an object; `window` / `raw_score` accepted; richer `top_features`.
* Evidence window ends at the ML `timestamp` (`end=` query on the backend telemetry call).
* network / security / maintenance are rule-only (`roles.py: uses_llm=False`); only diagnosis + supervisor narrative may use an LLM.
* `SPECIALIST_ENGINE=direct` (default): one LLM call per agent, same sanitizer; `crewai` keeps the tool loop.
* `POST /api/v1/incident/analyze/stream` (SSE): see docs/CONTRACTS.md section 6. Streaming lives in `service.analyze_stream`.
* `ML_VALIDATED=false` downgrades `stop_equipment` to `inspect_equipment` until the new-recording test passes.
* Telemetry: NaN/null safe, derived `gyro_mag`/`accel_mag`, `TELEMETRY_CHANNELS` allowlist, temperature judged in degrees,
  flatline rule needs a >= 30 s window. `python -m src.agents.scripts.replay_csv <csv>` replays a recording through the rules.

# What changed (v1.1)

| File | Change |
|---|---|
| `schemas.py` | `asset_id` optional; `model` string or object; `window`, `raw_score`; richer `TopFeature`; output: `votes`, `latency_ms`, `timings_ms`, `anomaly_score`, `ml_model_version`, `ml_validated`; schema 1.1 (additive) |
| `settings.py` | `default_asset_id`, `ml_validated`, `telemetry_channels`, `specialist_engine`; `llm_agents` default `diagnosis,supervisor`; timeout 30 s |
| `roles.py` | `uses_llm` flag; `LLM_ROLES = (diagnosis, supervisor)` |
| `agents.py` | `DirectRunner` (one call, no tool loop), `make_runner()`; crewAI path kept |
| `graph.py` | evidence window end, channel allowlist, per-node timings, `defer_narrative` for streaming |
| `service.py` | `analyze_stream()` generator + shared `_finish()`; `latency_ms` |
| `router.py` | `POST /api/v1/incident/analyze/stream` (SSE) |
| `tools.py` | NaN/null safe stats, derived `gyro_mag`/`accel_mag`, allowlist, `delta_to_baseline`, flatline needs >= 30 s |
| `rules.py` | temperature judged by degrees (+5 C), message format kept for ratio channels |
| `policy.py` | `ml_gate()` (stop -> inspect until `ML_VALIDATED=true`) |
| `llm.py` | `stream_text()` |
| `clients/*` | `get_telemetry(asset_id, minutes, end=None)` |
| `configs/config.yaml`, `.env.example` | safe defaults (LLM off, no weak key) |
| `scripts/replay_csv.py`, `tests/data/` | replay a recording through the rules |
| tests | 76 -> 96; old tests updated for the rule-only design |

Apply: unzip over the repo root (`unzip -o`), `pip install -r requirements.txt`, `pytest`, `bash check_agents.sh`.

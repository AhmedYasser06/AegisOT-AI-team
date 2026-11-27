# AegisOT: JSON contracts and trigger rules (schema 2.0)

Every JSON below is defined and validated in `src/agents/schemas.py`. An invalid answer from a service is reported in
`notes` and lowers the confidence; it never crashes the incident or is silently accepted.

Rule of thumb: **every team sends the agents plain measurements, never conclusions.**
The agents turn measurements into votes (rules, or one LLM call for `diagnosis`), the supervisor decides, and the solver
searches for what to do about it.

## 1. ML team -> Agents   `POST /api/v1/incident/analyze` (or `/analyze/stream`)

```json
{
  "incident_id": "INC-001",
  "asset_id": "esp32-conveyor-01",
  "timestamp": "2026-10-01T12:30:00Z",
  "window": {"start": "2026-10-01T12:29:50Z", "end": "2026-10-01T12:30:00Z", "n_samples": 50},
  "is_anomaly": true,
  "anomaly_score": 0.93,
  "raw_score": -0.21,
  "threshold": 0.5,
  "model": {"name": "isolation_forest", "version": "0.1", "feature_version": "f1"},
  "top_features": [
    {"name": "gyro_mag", "contribution": 0.42, "value": 0.26, "baseline": 0.036, "direction": "high"}
  ]
}
```
* `model` may also be just a string (`"isolation_forest"`). `window` only needs `end`; extra window fields are ignored.
* Required: `incident_id`, `anomaly_score` (0..1). Everything else is optional; unknown fields are ignored.
* `asset_id` missing -> `DEFAULT_ASSET_ID` (single ESP). Send it as soon as there is more than one device.
* `anomaly_score` = **how far from normal** (calibrated: 0.5 = the model's threshold, e.g. percentile against
  validation-normal scores). It is **not** "hardware failure": the agents decide the cause.
* `timestamp` (or `window.end`) = end of the evidence window. The agents ask the backend for telemetry ending there,
  so a retry or a late run sees the same data.
* `top_features`: use the same channel names as the telemetry (`temperature_c`, `gyro_mag`, ...).

**One call instead of two:** `POST /api/v1/anomaly/detect` takes a window `{incident_id, asset_id, samples:[{timestamp, sensor_data}]}`,
runs the ML model, and only if it is an anomaly runs the agents. Answer: `{"is_anomaly", "ml": <incident JSON>, "assessment": <section 5 or null>}`.

## 2. Backend (Laravel) -> Agents   (read-only GET)

| Endpoint | Returns |
|---|---|
| `GET /api/devices/{id}` | `asset_id, type, station, enrollment_state (unclaimed/claimed/operational/degraded/revoked), firmware, baseline{channel: mean}` |
| `GET /api/devices/{id}/telemetry?minutes=5&end=<ISO>` | `sampling_interval_s, window_minutes, baseline, samples[{ts, <channel>: number or null}]` |
| `GET /api/devices/{id}/events?kind=security&minutes=60` | `events[{ts, type: broker_denied/enroll_rejected/auth_failed, detail}]` |
| `GET /api/devices/{id}/maintenance` | `health_score (0..1), days_since_maintenance, open_work_items` |
| `POST /api/agent/assessments` | receives the final assessment (section 5) |

Notes: `window_minutes` must be the window you really return (if the device has only 100 s of history, say so),
otherwise missing samples look like a network gap. Use `null` (not NaN) for an invalid reading.
`baseline` should contain every channel the rules use, including `gyro_mag` and `accel_mag`.

## 3. Network team -> Agents   `GET /network/devices/{id}/health`

```json
{"asset_id": "esp32-conveyor-01", "reachable": true, "latency_ms": 12, "packet_loss_pct": 0.0,
 "link_status": "up", "interface": {"name": "Gi0/1", "status": "up"}, "zone": "ot-segment",
 "acl_violations": 0, "measured_at": "2026-10-01T12:30:00Z"}
```
Also agree the list of template names the backend may run (`restore_path`, `quarantine`, ...).

## 4. Hardware team -> everyone
Stable channel names + units + sampling rate + sensor error values (DS18B20: -127 / `temperature_valid=0`).

## 5. Agents -> Backend / Frontend (final output, schema 2.0)

```json
{"schema_version": "2.0", "incident_id": "INC-001", "asset_id": "esp32-conveyor-01",
 "status": "complete", "likely_cause": "physical_fault", "confidence": 0.98, "severity": "high",
 "recommended_action": "stop_equipment", "action_params": {"mode": "controlled_stop"}, "approval_required": true,
 "findings": ["[diagnosis] vibration_rms is 2.59x its baseline over the recent window.", "[network] Network healthy ..."],
 "summary": "Bearing wear is likely: vibration is 2.6x normal while the network is healthy.",
 "solutions": [{"step": "Check belt tension and bearing play", "source": "https://stackoverflow.com/..."},
               {"step": "Compare vibration with the baseline", "source": null}],
 "verified": true, "engine": "llm", "notes": [], "created_at": "2026-10-01T12:30:05Z"}
```
* `status`: complete | inconclusive (then `likely_cause` = unknown and the action is `escalate_to_human`).
* `recommended_action` + `approval_required`: decided by plain code (policy allowlist), never by an LLM.
  The backend executes only after a human approves when `approval_required` is true.
* `solutions`: what to do. `source` is a URL only if a real web search returned it, otherwise `null`.
* `verified`: true = the solutions were written by the solver and passed the check loop. false = default steps
  (solver off, no search key, or no answer passed after `SOLVER_MAX_ROUNDS` tries; see `notes`).
* `engine`: `llm` if the diagnosis vote or the solutions came from an LLM, else `rules`.
* Removed in 2.0 (they repeated other fields): `alternatives`, `votes`, `evidence_refs`, `checks_performed`, `diagnosis`,
  `evidence`, `explanation`, `recommended_action_text`, `risk_level` (now `severity`), `auto_executable`, `alert_id`,
  `llm_models_used`, `anomaly_score`, `ml_model_version`, `ml_validated`, `latency_ms`, `timings_ms`;
  the endpoint `/diagnosis/analyze` (use `/incident/analyze`).

## 6. Streaming (SSE) events

`accepted -> evidence_collected -> vote x4 -> decision -> search xN / note xN -> done` (or `error`).
* `decision` already has the cause, the action and default steps: show it at once.
* Show Approve/Reject only after `decision`, and only if `assessment.approval_required`.
* `search` = a web search the solver made; `note` = an answer the check loop rejected (it tries again).
* `done` carries the final stored assessment (the cause and action never change after `decision`).
* The browser must talk to **Laravel**, which proxies the stream (the browser cannot send `X-API-Key`).

## 7. When to call what  (data is 5 Hz)

**Backend -> ML model (every window, automatic)**
* Window = **50 samples (10 s)**, stride = **10 samples (2 s)** -> the model scores the newest window every 2 s per asset.
* Features per window: mean / std / RMS of `gyro_mag`, `accel_mag`, mean + slope of `temperature_c`.
  (FFT harmonics need kHz sampling; at 5 Hz they carry no information.)

**ML -> Agents (only when it is a real incident, NOT every window)**
* Open an incident when **3 of the last 5 windows** are anomalous (about 6-10 s), or the score is >= 0.9 twice in a row.
* One open incident per asset: while it is open, do not call the agents again for the same asset.
* Cool-down 5 min after it is closed. Call again earlier only if the score rises by >= 0.15 (use `force=true`).
* Always send the same `incident_id` on retries (the agents return the stored result).
* Slow faults (overheating ramps over ~100 s) will fire later than sudden ones (vibration): that is expected.

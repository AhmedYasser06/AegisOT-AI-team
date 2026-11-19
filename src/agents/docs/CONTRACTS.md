# AegisOT: JSON contracts and trigger rules (schema 1.1)

Rule of thumb: **every team sends the agents plain measurements, never conclusions.**
The agents turn measurements into votes (rules, or one LLM call for `diagnosis`), the supervisor decides.

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
* Required: `incident_id`, `anomaly_score` (0..1). Everything else is optional; unknown fields are ignored.
* `asset_id` missing -> `DEFAULT_ASSET_ID` (single ESP). Send it as soon as there is more than one device.
* `anomaly_score` = **how far from normal** (calibrated: 0.5 = the model's threshold, e.g. percentile against
  validation-normal scores). It is **not** "hardware failure": the agents decide the cause.
* `timestamp` (or `window.end`) = end of the evidence window. The agents ask the backend for telemetry ending there,
  so a retry or a late run sees the same data.
* `top_features`: use the same channel names as the telemetry (`temperature_c`, `gyro_mag`, ...).

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

## 5. Agents -> Backend / Frontend (final output, additive over 1.0)

Key fields: `status` (complete | inconclusive), `likely_cause`, `confidence`, `alternatives`, `diagnosis`, `evidence`,
`evidence_refs`, `recommended_action` (id) + `recommended_action_text`, `risk_level`, `approval_required`,
`auto_executable`, `explanation`, `engine` (rules | llm | mixed), `notes`.
New in 1.1: `votes[{role, engine, scores, findings}]`, `anomaly_score`, `ml_model_version`, `ml_validated`,
`latency_ms`, `timings_ms{node: ms}`.

## 6. Streaming (SSE) events

`accepted -> evidence_collected -> preliminary -> vote x4 -> decision -> explanation_token xN -> done` (or `error`).
* `preliminary` is rules-only and **not final**: never show Approve/Reject for it.
* Show Approve/Reject only after `decision`, and only if `assessment.approval_required`.
* `explanation_token` only rewrites the explanation; `done` carries the final stored assessment.
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

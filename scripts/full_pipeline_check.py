"""Full pipeline check: recorded CSV -> ML model -> agent layer -> (stream) -> stored result.

    python scripts/full_pipeline_check.py --key dev-key                 # servers already running
    python scripts/full_pipeline_check.py --key dev-key --start         # start the missing servers for you
    python scripts/full_pipeline_check.py --key dev-key --start --no-llm      # force rules only
    python scripts/full_pipeline_check.py --key dev-key --expect-llm    # FAIL if the LLM did not answer

One server (port 8001) serves both the ML and the agents; change it with --url. Run it from the repo root.
Exit code 0 = every check passed, 1 = something failed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing_extensions import final
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAUSES = {"physical_fault", "sensor_fault", "network_issue", "security_incident", "unknown"}
ACTIONS = {"notify_operator", "run_network_diagnostics", "check_motor_temperature", "check_motor_vibration",
           "check_power_supply", "check_motor_speed", "recheck_temperature_sensor", "request_sensor_recheck",
           "inspect_equipment", "escalate_to_human", "stop_equipment", "isolate_device", "apply_network_template"}
FAULT_TYPES = {"overheating", "abnormal_vibration", "electrical_fault", "speed_deviation", "machine_fault",
               "sensor_fault", "network_issue", "security_incident", "unknown"}
BACKEND = {"mode": "?"}       # what the agent server reads its telemetry from (csv = the same recording as the ML)
EXPECT_FAULT = {"overheated": "overheating", "vibration_issue": "abnormal_vibration"}      # what the labels should give
RISKY = {"stop_equipment", "isolate_device", "apply_network_template"}
EXPECT_ANOMALY = {"normal": False, "overheated": True, "vibration_issue": True}      # labels in final.csv

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool | None, detail: str = "") -> bool:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


def call(method: str, url: str, body: dict | None = None, key: str | None = None, timeout: int = 180):
    headers = {"Content-Type": "application/json", **({"X-API-Key": key} if key else {})}
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as e:                                   # connection refused, timeout, ...
        return 0, {"error": str(e)}


def stream(url: str, body: dict, key: str, timeout: int = 300) -> list[tuple[str, dict]]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "X-API-Key": key})
    events, ev = [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line[5:])))
    return events


def wait_health(url: str, seconds: int = 40) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if call("GET", url + "/health", timeout=3)[0] == 200:
            return True
        time.sleep(1)
    return False


def start_server(a, procs: list) -> None:
    env = {**os.environ, "AGENT_API_KEY": a.key, **({"LLM_ENABLED": "false"} if a.no_llm else {})}
    if not a.mock_backend:      # the agents must read the SAME recorded rows the ML model scored, not made-up mock data
        env.update({"BACKEND_MODE": "csv", "BACKEND_CSV": str(Path(a.csv).resolve())})
    if call("GET", a.url + "/health", timeout=3)[0] == 200:
        print(f"(using the server that is already running on {a.url})")
        return
    port = a.url.rsplit(":", 1)[-1]
    print(f"(starting the server on port {port} ...)")
    procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", port,
                                   "--log-level", "warning"], cwd=ROOT, env=env))
    if not wait_health(a.url, 90):
        print("  the server did not come up within 90 s (see its error above)")


def samples_of(rows: pd.DataFrame) -> list[dict]:
    return [{"timestamp": r["host_received_timestamp_utc"],
             "sensor_data": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                             for k, v in r.drop(labels=["host_received_timestamp_utc"]).items()}}
            for _, r in rows.iterrows()]


def contiguous_window(df: pd.DataFrame, cond: str, size: int) -> pd.DataFrame:
    """`size` consecutive rows of one condition with no pause > 2 s (so a window never spans two recording days)."""
    d = df[df["condition"] == cond].copy()
    t = pd.to_datetime(d["host_received_timestamp_utc"], utc=True, format="ISO8601").map(lambda x: x.timestamp())
    seg = (t.diff().fillna(0) > 2.0).cumsum()
    best = None
    for _, g in d.groupby(seg.values):
        if len(g) >= size:
            return g.head(size)
        if best is None or len(g) > len(best):
            best = g
    return best.head(size) if best is not None else d.head(size)


def check_assessment(tag: str, out: dict, ml: dict) -> None:
    record(f"{tag}: contract fields", all(k in out for k in (
        "schema_version", "incident_id", "asset_id", "status", "likely_cause", "confidence", "recommended_action",
        "approval_required", "evidence_refs", "findings", "explanation", "engine", "notes", "fault_type", "headline",
        "recommended_solution", "severity", "llm")))
    record(f"{tag}: fault type is named and the answer says what to do",
           out.get("fault_type") in FAULT_TYPES and bool(out.get("headline")) and bool(out.get("recommended_solution")),
           f"{out.get('fault_type')} | {str(out.get('headline'))[:110]}")
    want = EXPECT_FAULT.get(tag)
    if want and out.get("status") == "complete":
        if BACKEND["mode"] == "csv":
            record(f"{tag}: fault type matches the recorded condition", out.get("fault_type") == want,
                   f"got {out.get('fault_type')}, expected {want}")
        else:
            record(f"{tag}: fault type vs the recorded condition", None,
                   f"skipped: the agents read '{BACKEND['mode']}' telemetry, not the recording (start with BACKEND_MODE=csv)")
    record(f"{tag}: cause and action are in the allowed lists",
           out.get("likely_cause") in CAUSES and out.get("recommended_action") in ACTIONS,
           f"{out.get('likely_cause')} -> {out.get('recommended_action')}")
    record(f"{tag}: risky action requires approval",
           out.get("recommended_action") not in RISKY or out.get("approval_required") is True)
    record(f"{tag}: has evidence refs", bool(out.get("evidence_refs")), f"{len(out.get('evidence_refs', []))} refs")
    record(f"{tag}: ML result carried through",
           out.get("incident_id") == ml["incident_id"] and abs(float(out.get("anomaly_score", -1)) - ml["anomaly_score"]) < 1e-6,
           f"ML score {ml['anomaly_score']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full pipeline check: CSV -> ML -> agents")
    ap.add_argument("--csv", default=str(ROOT / "data" / "processed" / "final.csv"))
    ap.add_argument("--key", required=True, help="agent API key (AGENT_API_KEY)")
    ap.add_argument("--url", default="http://localhost:8001", help="the one server (ML + agents)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--asset", default="esp32-conveyor-01")
    ap.add_argument("--start", action="store_true", help="start the servers that are not running")
    ap.add_argument("--no-llm", action="store_true", help="with --start: force LLM_ENABLED=false")
    ap.add_argument("--mock-backend", action="store_true",
                    help="with --start: let the agents read MOCK telemetry instead of the CSV (the fault-type check is skipped)")
    ap.add_argument("--expect-llm", action="store_true", help="FAIL unless the LLM took part in a decision")
    a = ap.parse_args()

    procs: list = []
    try:
        if a.start:
            start_server(a, procs)
        run = f"run-{int(time.time())}"

        print("\n== 1. servers ==")
        s, ag_h = call("GET", a.url + "/health")
        if not record("server is up", s == 200 and ag_h.get("status") == "ok", str(ag_h)):
            print("Start it: uvicorn src.api.main:app --port 8001   (or add --start)")
        s, ml_h = call("GET", a.url + "/api/v1/anomaly/health")
        record("ML model file is loaded", s == 200 and ml_h.get("model_loaded") is True, str(ml_h))
        if any(st == "FAIL" for st, _, _ in results):
            return finish()
        record("LLM enabled in the agent server", None, f"llm_enabled={ag_h.get('llm_enabled')}")
        BACKEND["mode"] = (ag_h.get("integrations") or {}).get("backend", "?")
        record("agents read the recorded CSV (same rows as the ML)", None if BACKEND["mode"] == "csv" else True,
               f"backend={BACKEND['mode']}" + ("" if BACKEND["mode"] == "csv" else
               "  -> NOT the recording: restart with BACKEND_MODE=csv, or let --start launch it"))

        print("\n== 2. ML model on recorded data ==")
        df = pd.read_csv(a.csv)
        s, single = call("POST", a.url + "/api/v1/anomaly/predict", {
            "incident_id": f"{run}-single", "asset_id": a.asset,
            "sensor_data": samples_of(df.head(1))[0]["sensor_data"]}, a.key)
        record("ML single-reading endpoint answers with the incident JSON", s == 200 and all(
            k in single for k in ("incident_id", "is_anomaly", "anomaly_score", "model", "top_features")), f"HTTP {s}")
        anomalies = []
        for cond in df["condition"].unique():
            s, ml = call("POST", a.url + "/api/v1/anomaly/window", {
                "incident_id": f"{run}-{cond}", "asset_id": a.asset,
                "samples": samples_of(contiguous_window(df, cond, a.window))}, a.key)
            if s != 200:
                record(f"ML window '{cond}'", False, f"HTTP {s}: {str(ml)[:200]}")
                continue
            in_range = 0.0 <= ml["anomaly_score"] <= 1.0
            expected = EXPECT_ANOMALY.get(cond)
            record(f"ML window '{cond}': score in 0..1 and verdict matches the label",
                   in_range and (expected is None or ml["is_anomaly"] == expected),
                   f"anomaly={ml['is_anomaly']} score={ml['anomaly_score']:.2f} top={[f['name'] for f in ml['top_features'][:3]]}")
            if ml["is_anomaly"]:
                anomalies.append((cond, ml))
        if not anomalies:
            record("at least one anomalous window to forward", False, "nothing reached the agents")
            return finish()

        print("\n== 3. agent layer ==")
        record("ML endpoint rejects a call without the key (401)",
               call("POST", a.url + "/api/v1/anomaly/window", {"incident_id": "x", "samples": []})[0] == 401)
        record("agent API rejects a call without the key (401)",
               call("POST", a.url + "/api/v1/incident/analyze", anomalies[0][1])[0] == 401)
        record("agent API rejects is_anomaly=false (422)", call(
            "POST", a.url + "/api/v1/incident/analyze", {**anomalies[0][1], "is_anomaly": False}, a.key)[0] == 422)
        llm_used = False
        for cond, ml in anomalies:
            s, out = call("POST", a.url + "/api/v1/incident/analyze?view=full", ml, a.key)
            if not record(f"agents analyze '{cond}'", s == 200, f"HTTP {s}"):
                continue
            check_assessment(cond, out, ml)
            llm_used |= out.get("llm", {}).get("status") == "ok"        # the diagnosis LLM really answered
            record(f"{cond}: decision", None, f"{out['fault_type']} ({out['confidence']}) severity={out['severity']} -> "
                   f"{out['recommended_action']}, approval={out['approval_required']}, status={out['status']}, "
                   f"llm={out['llm']['status']}")
            for step in out.get("recommended_solution", []):
                record(f"{cond}: step", None, step[:160])
            if out.get("reason"):
                record(f"{cond}: inconclusive because", None, out["reason"][:160])
            for n in out.get("notes", []):
                record(f"{cond}: note", None, n[:160])
        cond0, ml0 = anomalies[0]
        s, again = call("POST", a.url + "/api/v1/incident/analyze", ml0, a.key)
        record("same incident_id returns the stored result (no second analysis)",
               s == 200 and again.get("created_at") == call("GET", a.url + f"/api/v1/incident/{ml0['incident_id']}", key=a.key)[1].get("created_at"))
        s, short = call("GET", a.url + f"/api/v1/incident/{ml0['incident_id']}", key=a.key)
        record("stored incident can be read back (compact view by default)", s == 200 and all(k in short for k in (
            "headline", "fault_type", "confidence", "summary", "evidence", "recommended_solution", "action", "llm")))
        record("compact view has no duplicated blocks", s == 200 and not any(k in short for k in (
            "findings", "votes", "evidence_refs", "timings_ms", "checks_performed")))
        record("unknown incident returns 404", call("GET", a.url + "/api/v1/incident/NOPE-404", key=a.key)[0] == 404)

        print("\n== 4. live stream ==")
        ml_stream = {**ml0, "incident_id": f"{run}-stream"}
        try:
            ev = stream(a.url + "/api/v1/incident/analyze/stream", ml_stream, a.key)
        except Exception as e:
            record("stream endpoint", False, str(e)[:200])
            ev = []
        names = [n for n, _ in ev]
        if ev:
            order = ["accepted", "evidence_collected", "preliminary", "vote", "decision", "done"]
            pos = [names.index(x) for x in order if x in names]
            record("stream: events arrive in the agreed order", len(pos) == len(order) and pos == sorted(pos), " > ".join(dict.fromkeys(names)))
            record("stream: 4 specialist votes", names.count("vote") == 4)
            dec, done = dict(ev).get("decision", {}), dict(ev).get("done", {})
            da, fa = dec.get("assessment", {}), done.get("assessment", {})
            record("stream: the decision never changes after the explanation",
                   all(da.get(k) == fa.get(k) for k in ("likely_cause", "recommended_action", "approval_required", "confidence")))
            tokens = [d["t"] for n, d in ev if n == "explanation_token"]
            record("stream: explanation tokens", None, f"{len(tokens)} pieces" if tokens else "none (LLM off or it failed; template used)")
            llm_used |= fa.get("llm", {}).get("status") == "ok"
            diag = [d for n, d in ev if n == "vote" and d.get("role") == "diagnosis"]
            if diag:
                record("diagnosis vote", None, f"engine={diag[0]['engine']} model={diag[0].get('model')} scores={diag[0]['scores']}")
                if a.expect_llm and diag[0]["engine"] != "llm":
                    record("stream: diagnosis answered by the LLM", False, f"engine={diag[0]['engine']} (see notes: LLM failed?)")
            record("stream: result is stored like a normal run",
                   call("GET", a.url + f"/api/v1/incident/{ml_stream['incident_id']}", key=a.key)[0] == 200)

        print("\n== 5. LLM ==")
        if a.expect_llm:
            record("the LLM took part in at least one decision", llm_used)
        else:
            record("the LLM took part in at least one decision", None, "yes" if llm_used else "no (rules only)")
    finally:
        for p in procs:
            p.terminate()
    return finish()


def finish() -> int:
    n = {k: sum(1 for s, _, _ in results if s == k) for k in ("PASS", "FAIL", "INFO")}
    print(f"\n{'=' * 60}\nPASS {n['PASS']}   FAIL {n['FAIL']}   INFO {n['INFO']}")
    for s, name, detail in results:
        if s == "FAIL":
            print(f"  FAILED: {name} {detail}")
    print("RESULT:", "ALL CHECKS PASSED" if n["FAIL"] == 0 else "SOMETHING FAILED")
    return 0 if n["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

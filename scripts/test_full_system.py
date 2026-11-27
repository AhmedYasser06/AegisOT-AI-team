"""STEP 2: test the FULL system.   recorded CSV  ->  ML model  ->  agent layer  ->  agent JSON

    python scripts/test_full_system.py --start                 # starts the server for you (rules only, no API key needed)
    python scripts/test_full_system.py --start --llm           # same, with the Gemini LLM on (needs GEMINI_API_KEY in .env)
    python scripts/test_full_system.py --url http://localhost:8001 --key <AGENT_API_KEY>      # server already running

What happens, for every window of rows in the CSV:
  1. the rows go to the ML endpoint            POST /api/v1/anomaly/window   -> "incident JSON" (score, top features)
  2. if the ML says anomaly, that JSON is sent UNCHANGED to the agents
                                               POST /api/v1/incident/analyze -> the agents' answer (cause, action, ...)
The agents read their evidence (telemetry) from the SAME csv (BACKEND_MODE=csv), so the diagnosis is made on the
recorded data, not on mock data. With --start this is configured for you.

At the end: a table of what the ML said and what the agents decided for each real condition, plus the full agent
JSON of the first anomaly of every condition (--show 0 to hide, --save out.json to keep every answer).
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
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAP_S = 2.0                                   # a pause longer than this ends a window (rows must be consecutive)
NOT_SENT = {"host_received_timestamp_utc", "condition", "_t"}      # the label is for scoring only, never sent to the system


def call(url: str, body: dict | None, key: str, timeout: int = 300) -> tuple[int, dict]:
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}
    except Exception as e:
        return 0, {"error": str(e)}


def make_windows(df: pd.DataFrame, size: int, min_rows: int) -> list[tuple[str, pd.DataFrame]]:
    """Cut the CSV into windows: consecutive rows (no pause > GAP_S), one condition, `size` rows each."""
    df = df.copy()
    df["_t"] = pd.to_datetime(df["host_received_timestamp_utc"], utc=True, format="ISO8601")
    df = df.sort_values("_t").reset_index(drop=True)
    new_segment = (df["_t"].diff().dt.total_seconds() > GAP_S) | (df["condition"] != df["condition"].shift())
    out = []
    for _, seg in df.groupby(new_segment.cumsum()):
        for start in range(0, len(seg), size):
            chunk = seg.iloc[start:start + size]
            if len(chunk) >= min_rows:
                out.append((str(seg["condition"].iloc[0]), chunk))
    return out


def to_samples(chunk: pd.DataFrame) -> list[dict]:
    return [{"timestamp": row["host_received_timestamp_utc"],
             "sensor_data": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                             for k, v in row.items() if k not in NOT_SENT}}
            for _, row in chunk.iterrows()]


def start_server(a) -> subprocess.Popen:
    env = {**os.environ, "BACKEND_MODE": "csv", "BACKEND_CSV": str(Path(a.csv).resolve()), "ANOMALY_MODE": "inline",
           "TELEMETRY_CHANNELS": a.channels, "EVIDENCE_WINDOW_MINUTES": "1", "AGENT_API_KEY": a.key,
           "LLM_ENABLED": "true" if a.llm else "false", "ML_VALIDATED": "true" if a.ml_validated else "false",
           "PUSH_RESULT_TO_BACKEND": "false"}
    port = a.url.rsplit(":", 1)[-1]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", port, "--log-level", "warning"],
                            cwd=ROOT, env=env)
    for _ in range(120):
        if proc.poll() is not None:
            raise SystemExit("the server stopped at start-up (see the error above)")
        if call(f"{a.url}/api/v1/anomaly/health", None, a.key, 5)[0] == 200:
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit("the server did not start in 60 s")


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV -> ML -> agents, with the agents' full JSON")
    ap.add_argument("--csv", default=str(ROOT / "data" / "processed" / "final.csv"))
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--key", default="dev-key", help="AGENT_API_KEY of the server (with --start it is set for you)")
    ap.add_argument("--start", action="store_true", help="start the server for this test (CSV as the agents' backend)")
    ap.add_argument("--llm", action="store_true", help="with --start: turn the LLM on (slow; use --limit)")
    ap.add_argument("--ml-validated", action="store_true", help="with --start: allow stop_equipment (ML_VALIDATED=true)")
    ap.add_argument("--channels", default="temperature_c,gyro_mag", help="with --start: channels the agent rules judge")
    ap.add_argument("--window", type=int, default=10, help="rows per window (10 rows = 2 s at 5 Hz)")
    ap.add_argument("--min-rows", type=int, default=3, help="skip shorter windows")
    ap.add_argument("--limit", type=int, default=0, help="max windows per condition (0 = all)")
    ap.add_argument("--show", type=int, default=1, help="print the full agent JSON for the first N anomalies per condition")
    ap.add_argument("--save", help="write every ML + agent answer to this .json file")
    ap.add_argument("--full", action="store_true", help="ask the agents for the full record (votes, refs, timings) instead of the compact answer")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    windows = make_windows(df, a.window, a.min_rows)
    if a.limit:
        seen, keep = Counter(), []
        for cond, chunk in windows:
            if seen[cond] < a.limit:
                keep.append((cond, chunk)); seen[cond] += 1
        windows = keep
    server = start_server(a) if a.start else None
    try:
        results, shown = [], Counter()
        ml_count, agent_count = defaultdict(Counter), defaultdict(Counter)
        for n, (cond, chunk) in enumerate(windows, 1):
            body = {"incident_id": f"FS-{n:04d}", "asset_id": "esp32-conveyor-01", "samples": to_samples(chunk)}
            code, ml = call(f"{a.url}/api/v1/anomaly/window", body, a.key)
            if code != 200:
                raise SystemExit(f"ML endpoint answered {code}: {ml}")
            ml_count[cond]["anomaly" if ml["is_anomaly"] else "normal"] += 1
            record = {"truth": cond, "ml": ml, "agents": None}
            if ml["is_anomaly"]:
                view = "full" if a.full else "compact"
                code, out = call(f"{a.url}/api/v1/incident/analyze?view={view}", ml, a.key)       # the ML JSON, unchanged
                if code != 200:
                    raise SystemExit(f"agent endpoint answered {code}: {out}")
                record["agents"] = out
                act = out["action"]["id"] if "action" in out else out["recommended_action"]
                appr = out["action"]["approval_required"] if "action" in out else out["approval_required"]
                llm = (out.get("llm") or {}).get("status", "?")
                agent_count[cond][f"{out['fault_type']} ({out['status']}) -> {act}" + (" (needs approval)" if appr else "")
                                  + f"  [llm={llm}]"] += 1
                if shown[cond] < a.show:
                    shown[cond] += 1
                    print(f"\n{'=' * 78}\nREAL CONDITION: {cond}   window {n}: {len(chunk)} rows, ML score {ml['anomaly_score']:.2f}, "
                          f"top features {[f['name'] for f in ml['top_features'][:3]]}\n{'-' * 78}\nAGENT JSON:")
                    print(json.dumps(out, indent=2, ensure_ascii=False))
            results.append(record)

        print(f"\n{'=' * 78}\nSUMMARY  ({len(windows)} windows of {a.window} rows)\n{'=' * 78}")
        for cond in sorted(ml_count):
            n_anom, n_all = ml_count[cond]["anomaly"], sum(ml_count[cond].values())
            print(f"\nreal condition = {cond}:   ML flagged {n_anom} of {n_all} windows as anomaly")
            for decision, count in agent_count[cond].most_common():
                print(f"    agents: {count:3d} x {decision}")
        if a.save:
            Path(a.save).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
            print(f"\nall answers saved to {a.save}")
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()

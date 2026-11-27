"""Watch the streaming endpoint live: prints the stages, then the explanation word by word.

    python -m scripts.stream_demo --asset esp32-hot-01 --score 0.91
    python -m scripts.stream_demo --url http://localhost:8002 --key <AGENT_API_KEY> --incident T-1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description="Consume POST /api/v1/incident/analyze/stream and print it live.")
    ap.add_argument("--url", default=os.getenv("AGENT_URL", "http://localhost:8001"))
    ap.add_argument("--key", default=os.getenv("AGENT_API_KEY", "dev-key"))
    ap.add_argument("--asset", default="esp32-hot-01")
    ap.add_argument("--score", type=float, default=0.91)
    ap.add_argument("--incident", default=f"STREAM-{int(time.time())}")
    ap.add_argument("--force", action="store_true", help="re-run even if the incident_id was analysed")
    a = ap.parse_args()

    body = {"asset_id": a.asset, "incident_id": a.incident, "anomaly_score": a.score, "model": "isolation_forest"}
    t0, event, in_text = time.perf_counter(), None, False
    with httpx.stream("POST", f"{a.url}/api/v1/incident/analyze/stream", json=body, params={"force": a.force},
                      headers={"X-API-Key": a.key}, timeout=None) as r:
        if r.status_code != 200:
            r.read()
            raise SystemExit(f"HTTP {r.status_code}: {r.text}")
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event == "explanation_token":
                    if not in_text:
                        print("\nexplanation: ", end="", flush=True)
                        in_text = True
                    print(data["t"], end="", flush=True)        # word by word, no newline
                    continue
                if in_text:
                    print()
                    in_text = False
                t = time.perf_counter() - t0
                if event in ("vote",):
                    print(f"[{t:5.2f}s] vote       {data['role']:<12} {data['engine']:<6} {data['scores']}")
                elif event in ("preliminary",):
                    print(f"[{t:5.2f}s] preliminary (NOT final) {data['likely_cause']} {data['confidence']}")
                elif event in ("decision", "done"):
                    x = data["assessment"]
                    print(f"[{t:5.2f}s] {event:<10} {x['likely_cause']} conf={x['confidence']} "
                          f"action={x['recommended_action']} approval={x['approval_required']} engine={x['engine']}")
                else:
                    print(f"[{t:5.2f}s] {event:<10} {json.dumps(data)[:100]}")
    print()


if __name__ == "__main__":
    main()

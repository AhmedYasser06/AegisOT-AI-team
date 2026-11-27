"""End-to-end check: recorded CSV rows -> ML model -> agent layer, printing the agent's FULL answer.

Start the server once (one app serves both the ML and the agents), from the repo root:
    uvicorn src.api.main:app --port 8001

Then:
    python scripts/e2e_ml_to_agents.py --key <AGENT_API_KEY>              # the agents' compact answer (the API default)
    python scripts/e2e_ml_to_agents.py --key <AGENT_API_KEY> --full       # the complete record (?view=full)
    python scripts/e2e_ml_to_agents.py --key <AGENT_API_KEY> --brief      # one summary block per condition
    python scripts/e2e_ml_to_agents.py --key <AGENT_API_KEY> --show-ml    # also print the ML's JSON

For each condition in the CSV it sends a window of rows to the ML endpoint. When the ML says "anomaly" it forwards the
ML's JSON, unchanged, to the agent endpoint and prints what the agents answered.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

import pandas as pd


def post(url: str, body: dict, key: str, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "_body": e.read().decode()[:400]}
    except urllib.error.URLError as e:
        raise SystemExit(f"Cannot reach {url}: {e.reason}. Is the server running?  uvicorn src.api.main:app --port 8001")


def show(title: str, data: dict) -> None:
    print(f"\n----- {title} " + "-" * max(3, 66 - len(title)))
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV rows -> ML -> agents, with the full agent answer")
    ap.add_argument("--csv", default="data/processed/final.csv")
    ap.add_argument("--key", required=True, help="the API key (AGENT_API_KEY in .env, default dev-key)")
    ap.add_argument("--url", default="http://localhost:8001", help="the one server (ML + agents)")
    ap.add_argument("--window", type=int, default=20, help="rows per window")
    ap.add_argument("--asset", default="esp32-conveyor-01")
    ap.add_argument("--brief", action="store_true", help="print a short summary instead of the compact answer")
    ap.add_argument("--full", action="store_true", help="print the complete record (votes, refs, timings...)")
    ap.add_argument("--show-ml", action="store_true", help="also print the ML's full JSON")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    for cond in df["condition"].unique():
        rows = df[df["condition"] == cond].head(a.window)
        samples = [{"timestamp": r["host_received_timestamp_utc"],
                    "sensor_data": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                                    for k, v in r.drop(labels=["host_received_timestamp_utc"]).items()}}
                   for _, r in rows.iterrows()]
        ml = post(f"{a.url}/api/v1/anomaly/window", {"incident_id": f"E2E-{cond}", "asset_id": a.asset, "samples": samples}, a.key)
        print("\n" + "=" * 72)
        if "_http" in ml:
            print(f"{cond}: ML error {ml['_http']}: {ml['_body']}")
            continue
        top = [f["name"] for f in ml["top_features"][:3]]
        print(f"{cond}  |  ML: anomaly={ml['is_anomaly']}  score={ml['anomaly_score']:.2f}  top={top}")
        if a.show_ml:
            show("ML answer (this exact JSON goes to the agents)", ml)
        if not ml["is_anomaly"]:
            print("not an anomaly, so nothing is sent to the agents")
            continue
        out = post(f"{a.url}/api/v1/incident/analyze?view={'full' if a.full else 'compact'}", ml, a.key)
        if "_http" in out:
            print(f"AGENTS error {out['_http']}: {out['_body']}")
            continue
        if a.brief:
            act = out["action"]["id"] if "action" in out else out["recommended_action"]
            appr = out["action"]["approval_required"] if "action" in out else out["approval_required"]
            print(f"AGENTS: {out['fault_type']} ({out['confidence']}) severity={out['severity']} -> {act}, "
                  f"approval={appr}, status={out['status']}, llm={out['llm']['status']}")
            print(f"   {out['headline']}")
            for step in out["recommended_solution"]:
                print(f"   - {step}")
            for n in out["notes"]:
                print(f"   note: {n}")
        else:
            show("AGENTS full answer" if a.full else "AGENTS answer", out)


if __name__ == "__main__":
    main()

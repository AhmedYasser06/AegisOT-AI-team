"""Terminal demo of the agent workflow.

    python -m src.agents.cli                                   # the 5 demo scenarios
    python -m src.agents.cli --asset esp32-conveyor-01 --score 0.93 --json
"""
from __future__ import annotations

import argparse
import textwrap

from .clients import build_clients
from .graph import build_graph, run_incident
from .schemas import AlertIn, IncidentAssessment
from .settings import get_settings

DEMO_CASES = [("esp32-conveyor-01", 0.93), ("esp32-conveyor-net", 0.70), ("esp32-rogue-01", 0.88),
              ("esp32-sensor-01", 0.80), ("esp32-unknown-99", 0.30)]


def _print(a: IncidentAssessment) -> None:
    print(f"\n=== {a.asset_id}  [{a.status.upper()}]  engine={a.engine}")
    print(f"cause      : {a.likely_cause.value} ({a.confidence:.0%})   alternatives: "
          f"{[(x.cause.value, x.confidence) for x in a.alternatives]}")
    print(f"action     : {a.recommended_action.value}  -> {a.recommended_action_text}")
    print(f"approval   : {'REQUIRED' if a.approval_required else 'not required'}   risk={a.risk_level.value}")
    print("explain    : " + textwrap.fill(a.explanation, 100, subsequent_indent=" " * 13))
    print(f"evidence   : {', '.join(a.evidence_refs)}")
    for n in a.notes:
        print(f"note       : {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the AegisOT agent workflow on demo or custom incidents.")
    ap.add_argument("--asset", help="asset_id to analyse (default: run the 5 demo scenarios)")
    ap.add_argument("--score", type=float, default=0.9, help="anomaly score 0-1 for --asset")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of the readable summary")
    args = ap.parse_args()

    settings = get_settings()
    graph = build_graph(settings, build_clients(settings))
    cases = [(args.asset, args.score)] if args.asset else DEMO_CASES
    print(f"LLM enabled: {settings.llm_enabled} | backend={settings.backend_mode} "
          f"anomaly={settings.anomaly_mode} network={settings.network_mode}")
    for i, (asset, score) in enumerate(cases, 1):
        alert = AlertIn(incident_id=f"DEMO-{i:03d}", asset_id=asset, anomaly_score=score, model_version="demo")
        result = run_incident(graph, alert)
        print(result.model_dump_json(indent=2)) if args.json else _print(result)


if __name__ == "__main__":
    main()

"""Replay a recorded CSV through the diagnosis rules (no server, no LLM, no API key).

    python -m scripts.replay_csv path/to/aegisot_web_dataset.csv
    python -m scripts.replay_csv data.csv --channels temperature_c,gyro_mag --window 50 --stride 10

What it does: the `normal` rows give the baseline (the mean of every channel); then each condition is cut into
sliding windows (default 50 samples = 10 s at 5 Hz) and every window is judged by the same rules the agents use.
It answers: "how often does the diagnosis agent call a window a physical fault?" per condition.

HONEST LIMIT: the baseline comes from the same recording, so this is a sanity check, not an evaluation.
Judge the model only on NEW recordings.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.agents.enums import Cause
from src.agents.rules import rules_diagnosis
from src.agents.testing import make_alert
from src.agents.tools import EvidencePack, _add_derived, parse_channels, summarize_telemetry

SAMPLE_HZ = 5.0
COLS = ["temperature_c", "current_ma", "rpm", "speed_percent", "bus_voltage_v", "power_mw",
        "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]


def to_samples(d: pd.DataFrame) -> list[dict]:
    """DataFrame rows -> the sample dicts the backend would send (NaN becomes None, like JSON null)."""
    cols = [c for c in COLS if c in d.columns]
    return [{**{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}, "ts": i}
            for i, r in enumerate(d[cols].to_dict("records"))]


def baseline_from(normal: pd.DataFrame) -> dict[str, float]:
    """Mean of every channel (including gyro_mag / accel_mag) over the normal rows."""
    rows = _add_derived(to_samples(normal))
    keys = {k for r in rows for k in r if k != "ts"}
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def judge_window(window: pd.DataFrame, baseline: dict, channels: tuple[str, ...]) -> tuple[float, float]:
    """(physical_fault score, sensor_fault score) the diagnosis rules give this window."""
    raw = {"sampling_interval_s": 1 / SAMPLE_HZ, "window_minutes": len(window) / SAMPLE_HZ / 60,
           "baseline": baseline, "samples": to_samples(window)}
    pack = EvidencePack(asset_id="replay", telemetry=summarize_telemetry(raw, channels),
                        refs={"telemetry:replay:last1m": "replay window"})
    rep = rules_diagnosis(make_alert("replay", 0.9), pack)
    return rep.scores[Cause.physical_fault], rep.scores[Cause.sensor_fault]


def replay(df: pd.DataFrame, channels: tuple[str, ...] = (), window: int = 50, stride: int = 10) -> dict[str, dict]:
    """{condition: {"windows": n, "physical": n_flagged, "sensor": n_flagged}} for every condition in the CSV."""
    base = baseline_from(df[df.condition == "normal"])
    result = {}
    for cond, d in df.groupby("condition", sort=False):
        d = d.reset_index(drop=True)
        scores = [judge_window(d.iloc[s:s + window], base, channels) for s in range(0, len(d) - window + 1, stride)]
        result[cond] = {"windows": len(scores), "physical": sum(p >= 0.5 for p, _ in scores),
                        "sensor": sum(s >= 0.5 for _, s in scores)}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--channels", default="", help="allowlist, e.g. temperature_c,gyro_mag (empty = all channels)")
    ap.add_argument("--window", type=int, default=50, help="samples per window (50 = 10 s at 5 Hz)")
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    for label, ch in (("ALL channels", ()), (f"allowlist: {a.channels}", parse_channels(a.channels))):
        if label.startswith("allowlist") and not ch:
            continue
        print(f"\n== {label}")
        for cond, r in replay(df, ch, a.window, a.stride).items():
            n = max(1, r["windows"])
            print(f"  {cond:16s} windows={r['windows']:3d}  physical_fault={r['physical']:3d} ({r['physical']/n:.0%})"
                  f"  sensor_fault={r['sensor']}")


if __name__ == "__main__":
    main()

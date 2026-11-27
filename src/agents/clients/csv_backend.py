"""Backend that REPLAYS a recorded CSV  (BACKEND_MODE=csv, BACKEND_CSV=path/to/file.csv).

Why: with the normal mock backend the agents diagnose made-up telemetry, whatever the ML model scored. With this
client the agents read the SAME recorded rows the ML model scored, so an ML -> agents test is honest.

What it serves (same contract as the real backend, see docs/CONTRACTS.md):
  get_telemetry(asset, minutes, end)  the rows of the recording that END at `end` (the ML timestamp), going back at
                                      most `minutes`, but never across a pause longer than GAP_S seconds
  baseline                            the average of the rows labelled "normal" (a stand-in for a calibration run)
Everything else (maintenance, events, asset) is healthy and neutral: a CSV has no data for them.

The `condition` label and other labels/counters are NEVER sent to the agents.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

from .base import ClientError
from .mock import MockBackendClient

ROOT = Path(__file__).resolve().parents[3]
GAP_S = 2.0
SESSION_GAP_S = 600.0                                  # a longer pause between two rows = a new recording segment
HIDDEN = {"condition", "last_motor_stop_reason", "host_received_timestamp_utc", "stage_elapsed_ms",
          "device_time_ms", "encoder_count"}


def _to_epoch(text: str) -> float:
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


class CsvBackendClient(MockBackendClient):
    def __init__(self, path: str):
        import numpy as np
        import pandas as pd

        p = Path(path) if Path(path).is_absolute() else ROOT / path
        if not p.exists():
            raise ClientError(f"BACKEND_CSV not found: {p}")
        df = pd.read_csv(p)
        df["_t"] = pd.to_datetime(df["host_received_timestamp_utc"], utc=True, format="ISO8601").map(lambda x: x.timestamp())
        df = df.sort_values("_t").reset_index(drop=True)
        if {"temperature_c", "temperature_valid"} <= set(df.columns):       # an invalid reading is "missing", not a number
            df.loc[df["temperature_valid"] != 1, "temperature_c"] = np.nan

        self.times = df["_t"].tolist()
        channels = [c for c in df.columns if c not in HIDDEN and c != "_t" and pd.api.types.is_numeric_dtype(df[c])]
        self.records = [{"ts": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                         **{c: (None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 4))
                            for c, v in zip(channels, row)}}
                        for t, row in zip(self.times, df[channels].itertuples(index=False, name=None))]
        gaps = np.diff(df["_t"].to_numpy())
        self.interval = float(np.median(gaps[gaps <= GAP_S])) if (gaps <= GAP_S).any() else 1.0

        # baseline = average of the "normal" rows of the SAME recording session (a pause > 10 min starts a new one:
        # supply voltage etc. differ between days, which would otherwise look like a fault)
        sess = (df["_t"].diff().fillna(0) > SESSION_GAP_S).cumsum().to_numpy()
        is_normal = (df["condition"] == "normal").to_numpy() if "condition" in df.columns else np.ones(len(df), bool)
        if not is_normal.any():
            is_normal = np.ones(len(df), bool)
        self._baselines = {int(k): self._baseline_of(df[is_normal & (sess == k)] if (is_normal & (sess == k)).any() else df[is_normal], channels)
                           for k in np.unique(sess)}
        self._session = sess
        self.baseline = self._baseline_of(df[is_normal], channels)       # overall (used by get_asset)
        super().__init__()

    @staticmethod
    def _baseline_of(normal, channels):
        import numpy as np
        base = {c: float(normal[c].mean()) for c in channels if normal[c].notna().any()}
        for name, parts in (("gyro_mag", ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")),
                            ("accel_mag", ("accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"))):
            if all(p_ in normal.columns for p_ in parts):
                base[name] = float(np.sqrt(sum(normal[p_] ** 2 for p_ in parts)).mean())
        return {k: round(v, 4) for k, v in base.items()}

    def get_asset(self, asset_id):
        return {"asset_id": asset_id, "station": "B", "type": "conveyor_drive", "enrollment_state": "operational",
                "firmware": "csv-replay", "profile": "conveyor-drive-v1", "baseline": self.baseline}

    def get_telemetry(self, asset_id, minutes, end=None):
        last = (bisect_right(self.times, _to_epoch(end)) if end else len(self.times)) - 1
        first = last
        while (first > 0 and self.times[last] - self.times[first - 1] <= minutes * 60
               and self.times[first] - self.times[first - 1] <= GAP_S):
            first -= 1
        rows = self.records[first:last + 1] if last >= 0 else []
        baseline = self._baselines.get(int(self._session[last]), self.baseline) if last >= 0 else self.baseline
        # window_minutes = the window REALLY returned (a short segment must not look like missing samples)
        return {"asset_id": asset_id, "sampling_interval_s": self.interval,
                "window_minutes": max(len(rows), 1) * self.interval / 60, "baseline": baseline, "samples": rows}

    def get_maintenance(self, asset_id):
        return {"asset_id": asset_id, "health_score": 0.9, "days_since_maintenance": 15, "open_work_items": 0}

    def get_events(self, asset_id, kind, minutes):
        return {"asset_id": asset_id, "events": []}

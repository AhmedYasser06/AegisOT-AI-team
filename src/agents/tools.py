"""Read-only evidence tools. One table (`SOURCES`) drives everything:

    * collect_evidence()      -> fetches every source once and fills an EvidencePack
    * ToolBox.run(key)        -> the same fetch, exposed to an LLM agent (errors returned as data)
    * crew_tools.py           -> wraps the same table as crewAI tools

Function names follow the team README: get_sensor_data, get_machine_status, get_network_status,
get_security_events, get_machine_history (+ get_anomaly_details).

Safety rules (README "AI Safety Rules"):
  * tools are read-only, typed and allowlisted; no shell / SQL / router / device access exists here
  * every tool is bound to ONE asset (the incident's): the LLM cannot pass another asset id
  * every successful fetch registers an evidence ref; agents may only cite refs that exist
  * errors are returned as data, never raised into the LLM loop
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from .clients import ClientError, Clients
from .schemas import AlertIn
from .settings import Settings

PLAUSIBLE = {"temperature": (-40.0, 125.0), "temperature_c": (-40.0, 125.0)}   # outside = sensor error (DS18B20 reads -127)
FLATLINE_MIN_SECONDS = 30                     # a flat line only means "dead sensor" if the window is long enough
                                              # (a DS18B20 legitimately repeats the same value for several seconds)
INVALID_RATIO = 0.5                           # a channel that is None/NaN in >= 50% of samples = failing sensor
# Channels computed from other channels: vibration shows up as a jump in the MAGNITUDE of the gyro/accel vector.
DERIVED = {"accel_mag": ("accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"),
           "gyro_mag": ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")}
MAX_DETAIL = 120                              # device-provided text is truncated (prompt-injection hygiene)

# A fetch result: (data, evidence ref id, human description of the ref)
Fetched = tuple[dict, str, str]


class EvidencePack(BaseModel):
    asset_id: str
    asset: dict | None = None
    telemetry: dict | None = None
    anomaly: dict | None = None
    network: dict | None = None
    security: dict | None = None
    maintenance: dict | None = None
    unavailable: list[str] = Field(default_factory=list)    # sources that failed to load
    refs: dict[str, str] = Field(default_factory=dict)      # ref id -> description


@dataclass(frozen=True)
class InlineAnomaly:
    """The ML result that arrived inside the request itself.

    preferred=True (anomaly_mode=inline): never call the anomaly service, use this.
    preferred=False: ask the service first and fall back to this if it fails.
    """
    data: dict
    preferred: bool

    @classmethod
    def from_alert(cls, alert: AlertIn, settings: Settings) -> "InlineAnomaly":
        return cls(alert.inline_anomaly(), settings.anomaly_mode == "inline")


# ───────── telemetry summary ─────────
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v) -> bool:
    return _is_num(v) and math.isfinite(v)


def _add_derived(samples: list[dict]) -> list[dict]:
    """Return copies of the samples with accel_mag / gyro_mag added when their x,y,z parts exist."""
    out = []
    for s in samples:
        s = dict(s)
        for name, parts in DERIVED.items():
            vals = [s.get(p) for p in parts]
            if all(_finite(v) for v in vals):
                s[name] = math.sqrt(sum(v * v for v in vals))
        out.append(s)
    return out


def summarize_telemetry(raw: dict[str, Any], channels: tuple[str, ...] = ()) -> dict[str, Any]:
    """Turn raw samples into per-channel statistics. `channels` = allowlist (empty = all channels).

    * None / NaN values are ignored (a sensor that is mostly None is reported in `implausible`)
    * `ratio_to_baseline` and `delta_to_baseline` let the rules compare with normal behaviour
    """
    samples = _add_derived(raw.get("samples") or [])
    interval = float(raw.get("sampling_interval_s") or 1.0)
    minutes = float(raw.get("window_minutes") or 5)
    expected = max(1, int(minutes * 60 / interval))
    n = len(samples)
    baseline = raw.get("baseline") or {}
    keys = sorted({k for s in samples for k, v in s.items() if k != "ts" and _is_num(v)})
    if channels:
        keys = [k for k in keys if k in channels]
    ch_out: dict[str, dict] = {}
    flat, implausible = [], []
    for k in keys:
        present = [s[k] for s in samples if k in s and (_is_num(s[k]) or s[k] is None)]
        vals = [float(v) for v in present if _finite(v)]
        invalid_ratio = 1 - len(vals) / len(present) if present else 0.0
        if not vals:                                  # nothing usable: failing sensor
            implausible.append(k)
            continue
        recent = vals[-max(1, len(vals) // 3):]
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        b, src = baseline.get(k), "provided"
        if not b and len(vals) >= 9:      # no baseline from backend -> use the start of the window (weaker)
            b, src = statistics.fmean(vals[: len(vals) // 3]), "window_start"
        rec = statistics.fmean(recent)
        ch_out[k] = {"mean": round(statistics.fmean(vals), 3), "min": round(min(vals), 3),
                     "max": round(max(vals), 3), "std": round(std, 4), "recent_mean": round(rec, 3),
                     "baseline": round(b, 3) if b else None, "baseline_source": src if b else None,
                     "ratio_to_baseline": round(rec / b, 2) if b else None,
                     "delta_to_baseline": round(rec - b, 3) if b is not None else None,
                     "invalid_ratio": round(invalid_ratio, 3)}
        if len(vals) >= 10 and std < 1e-9 and n * interval >= FLATLINE_MIN_SECONDS:
            flat.append(k)
        lo, hi = PLAUSIBLE.get(k, (float("-inf"), float("inf")))
        if any(v < lo or v > hi for v in vals) or invalid_ratio >= INVALID_RATIO:
            implausible.append(k)
    return {"n_samples": n, "expected_samples": expected,
            "missing_ratio": round(max(0.0, 1 - n / expected), 3),
            "channels": ch_out, "flatlined": flat, "implausible": implausible}


# ───────── the tools: each returns (data, ref, description) or raises ClientError ─────────
def get_machine_status(c: Clients, asset_id: str) -> Fetched:
    return c.backend.get_asset(asset_id), f"asset:{asset_id}", "device profile and enrollment state"


def get_sensor_data(c: Clients, asset_id: str, minutes: int = 5, end: str | None = None,
                    channels: tuple[str, ...] = ()) -> Fetched:
    """`end` = ISO time the window ends at (the ML detection time). None = "now"."""
    minutes = max(1, min(int(minutes), 30))
    d = summarize_telemetry(c.backend.get_telemetry(asset_id, minutes, end), channels)
    return d, f"telemetry:{asset_id}:last{minutes}m", f"telemetry summary, last {minutes} min"


def get_network_status(c: Clients, asset_id: str) -> Fetched:
    d = c.network.get_health(asset_id)
    return d, f"network:{asset_id}:health", "network health (reachability, latency, loss, interface, ACL)"


def get_security_events(c: Clients, asset_id: str, enrollment_state: str | None = None) -> Fetched:
    ev = c.backend.get_events(asset_id, "security", 60).get("events", [])
    counts = Counter(e.get("type", "unknown") for e in ev)
    sample = [{"type": e.get("type"), "detail": str(e.get("detail", ""))[:MAX_DETAIL]} for e in ev[:5]]
    d = {"enrollment_state": enrollment_state, "event_counts_60m": dict(counts), "events_sample": sample}
    return d, f"security:{asset_id}:events60m", "enrollment state and broker/enrollment security events"


def get_machine_history(c: Clients, asset_id: str) -> Fetched:
    return c.backend.get_maintenance(asset_id), f"maintenance:{asset_id}", "health score and maintenance history"


def get_anomaly_details(c: Clients, alert_id: str, inline: InlineAnomaly | None = None) -> Fetched:
    ref, desc = f"anomaly:{alert_id}", "anomaly model score and top contributing features"
    if inline and inline.preferred:
        return inline.data, ref, desc
    try:
        return c.anomaly.get_alert(alert_id), ref, desc
    except ClientError:
        if inline:
            return inline.data, ref, desc
        raise


# ───────── per-asset toolbox ─────────
class ToolBox:
    """Everything one incident's tools need. Bound to ONE asset; shares `refs` with the EvidencePack."""

    def __init__(self, clients: Clients, asset_id: str, alert_id: str | None, refs: dict[str, str],
                 enrollment_state: str | None = None, inline: InlineAnomaly | None = None,
                 end: str | None = None, channels: tuple[str, ...] = ()):
        self.clients, self.asset_id, self.alert_id = clients, asset_id, alert_id
        self.refs, self.enrollment_state, self.inline = refs, enrollment_state, inline
        self.end, self.channels = end, channels
        # source key -> how to fetch it. Keys are the EvidencePack fields and the names used in roles.py.
        self._fetch: dict[str, Callable[..., Fetched]] = {
            "asset":       lambda: get_machine_status(clients, asset_id),
            "telemetry":   lambda minutes=5: get_sensor_data(clients, asset_id, minutes, self.end, self.channels),
            "network":     lambda: get_network_status(clients, asset_id),
            "security":    lambda: get_security_events(clients, asset_id, self.enrollment_state),
            "maintenance": lambda: get_machine_history(clients, asset_id),
            "anomaly":     lambda: get_anomaly_details(clients, self.alert_id, self.inline),
        }

    def fetch(self, key: str, **kw) -> tuple[dict, str]:
        """Fetch one source and register its evidence ref. Raises ClientError."""
        data, ref, desc = self._fetch[key](**kw)
        self.refs[ref] = desc
        return data, ref

    def run(self, key: str, **kw) -> dict[str, Any]:
        """Same as fetch, but for LLM agents: errors come back as data instead of exceptions."""
        try:
            data, ref = self.fetch(key, **kw)
        except ClientError as e:
            return {"error": str(e)}
        return {"ref": ref, "data": data}


SOURCE_KEYS = ("asset", "telemetry", "network", "security", "maintenance", "anomaly")   # fetch order


def parse_channels(text: str | None) -> tuple[str, ...]:
    """"temperature_c, gyro_mag" -> ("temperature_c", "gyro_mag"). Empty = all channels."""
    return tuple(x.strip() for x in (text or "").split(",") if x.strip())


def collect_evidence(c: Clients, asset_id: str, alert_id: str | None = None, minutes: int = 5,
                     inline: InlineAnomaly | None = None, end: str | None = None,
                     channels: tuple[str, ...] = ()) -> EvidencePack:
    """Fetch every source once. A failing source is listed in `pack.unavailable`, never raised."""
    pack = EvidencePack(asset_id=asset_id)
    box = ToolBox(c, asset_id, alert_id, pack.refs, inline=inline, end=end, channels=channels)
    for key in SOURCE_KEYS:
        if key == "anomaly" and not alert_id:
            continue
        try:
            data, _ = box.fetch(key, **({"minutes": minutes} if key == "telemetry" else {}))
        except ClientError:
            pack.unavailable.append(key)
            continue
        setattr(pack, key, data)
        if key == "asset":                       # the security tool reports the asset's enrollment state
            box.enrollment_state = data.get("enrollment_state")
    return pack

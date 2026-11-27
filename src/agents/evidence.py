"""Evidence collection: ask the other teams' services for data and put it in one typed EvidencePack.

    collect_evidence()  ->  asset, telemetry, network, security, maintenance, anomaly   (each may fail on its own)

Called by: graph.py (node `collect_evidence`) and router.py (debug endpoint). The agents never call a service
themselves, they only read the pack, so they can only cite evidence that really exists (`pack.refs`).

A source that fails (service down, invalid JSON) is skipped and listed in `pack.unavailable`; the supervisor then
lowers its confidence. It never crashes the incident.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .clients import ClientError, Clients
from .schemas import (Asset, AnomalyInfo, ChannelStats, Maintenance, NetworkHealth, SecurityEvent, SecurityInfo,
                      Telemetry, EvidencePack)
from .settings import Settings

logger = logging.getLogger("uvicorn.error")

PLAUSIBLE = {"temperature": (-40.0, 125.0), "temperature_c": (-40.0, 125.0)}   # outside = sensor error (DS18B20: -127)
FLATLINE_MIN_SECONDS = 30      # a flat line only means "dead sensor" if the window is long enough
                               # (a DS18B20 legitimately repeats the same value for several seconds)
INVALID_RATIO = 0.5            # a channel that is null/NaN in >= 50% of the samples = failing sensor
# Channels computed from other channels: vibration shows up as a jump in the MAGNITUDE of the gyro/accel vector.
DERIVED = {"accel_mag": ("accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"),
           "gyro_mag": ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")}


# ───────── telemetry: raw samples -> per-channel statistics ─────────
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v) -> bool:
    return _is_num(v) and math.isfinite(v)


def add_derived_channels(samples: list[dict]) -> list[dict]:
    """Copies of the samples with accel_mag / gyro_mag added when their x, y, z parts exist."""
    out = []
    for sample in samples:
        sample = dict(sample)
        for name, parts in DERIVED.items():
            values = [sample.get(p) for p in parts]
            if all(_finite(v) for v in values):
                sample[name] = math.sqrt(sum(v * v for v in values))
        out.append(sample)
    return out


def summarize_telemetry(raw: dict[str, Any], channels: tuple[str, ...] = ()) -> Telemetry:
    """Raw backend telemetry -> Telemetry. `channels` = allowlist (empty = all channels).

    * null / NaN values are ignored; a channel that is mostly invalid is reported in `implausible`
    * `ratio_to_baseline` / `delta_to_baseline` let the rules compare with normal behaviour
    """
    samples = add_derived_channels(raw.get("samples") or [])
    interval = float(raw.get("sampling_interval_s") or 1.0)
    minutes = float(raw.get("window_minutes") or 5)
    expected = max(1, int(minutes * 60 / interval))
    n = len(samples)
    baseline = raw.get("baseline") or {}
    names = sorted({k for s in samples for k, v in s.items() if k != "ts" and _is_num(v)})
    if channels:
        names = [k for k in names if k in channels]

    stats: dict[str, ChannelStats] = {}
    flatlined, implausible = [], []
    for name in names:
        present = [s[name] for s in samples if name in s and (_is_num(s[name]) or s[name] is None)]
        values = [float(v) for v in present if _finite(v)]
        invalid_ratio = 1 - len(values) / len(present) if present else 0.0
        if not values:                                  # nothing usable: failing sensor
            implausible.append(name)
            continue
        recent = statistics.fmean(values[-max(1, len(values) // 3):])
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        ref, ref_source = baseline.get(name), "provided"
        if not ref and len(values) >= 9:                # no baseline from the backend: use the start of the window (weaker)
            ref, ref_source = statistics.fmean(values[: len(values) // 3]), "window_start"
        stats[name] = ChannelStats(
            mean=round(statistics.fmean(values), 3), min=round(min(values), 3), max=round(max(values), 3),
            std=round(std, 4), recent_mean=round(recent, 3),
            baseline=round(ref, 3) if ref else None, baseline_source=ref_source if ref else None,
            ratio_to_baseline=round(recent / ref, 2) if ref else None,
            delta_to_baseline=round(recent - ref, 3) if ref is not None else None,
            invalid_ratio=round(invalid_ratio, 3))
        if len(values) >= 10 and std < 1e-9 and n * interval >= FLATLINE_MIN_SECONDS:
            flatlined.append(name)
        low, high = PLAUSIBLE.get(name, (float("-inf"), float("inf")))
        if any(v < low or v > high for v in values) or invalid_ratio >= INVALID_RATIO:
            implausible.append(name)
    return Telemetry(n_samples=n, expected_samples=expected, missing_ratio=round(max(0.0, 1 - n / expected), 3),
                     channels=stats, flatlined=flatlined, implausible=implausible)


# ───────── security: raw events -> counts + a short sample ─────────
def summarize_security(raw_events: dict[str, Any], enrollment_state: str | None) -> SecurityInfo:
    events = [SecurityEvent.model_validate(e) for e in raw_events.get("events", [])]
    return SecurityInfo(enrollment_state=enrollment_state,
                        event_counts_60m=dict(Counter(e.type for e in events)), events_sample=events[:5])


# ───────── anomaly: the ML result ─────────
def _anomaly(clients: Clients, settings: Settings, alert_id: str, from_request: AnomalyInfo | None) -> AnomalyInfo:
    """anomaly_mode=inline: use the ML result that came with the request. Otherwise ask the anomaly service
    first and fall back to the request's copy if the service fails."""
    if from_request and settings.anomaly_mode == "inline":
        return from_request
    try:
        return AnomalyInfo.model_validate(clients.anomaly.get_alert(alert_id))
    except (ClientError, ValidationError):
        if from_request:
            return from_request
        raise


# ───────── the collector ─────────
def collect_evidence(clients: Clients, settings: Settings, asset_id: str, alert_id: str | None = None,
                     from_request: AnomalyInfo | None = None, end: str | None = None) -> EvidencePack:
    """Fetch every source once. `end` = ISO time the telemetry window ends at (None = now)."""
    pack = EvidencePack(asset_id=asset_id)
    minutes = max(1, min(settings.evidence_window_minutes, 30))

    def load(source: str, ref: str, description: str, fetch: Callable[[], BaseModel]) -> None:
        try:
            data = fetch()
        except (ClientError, ValidationError) as e:
            logger.warning("evidence source %r unavailable for %s: %s", source, asset_id, e)
            pack.unavailable.append(source)
            return
        setattr(pack, source, data)
        pack.refs[ref] = description

    load("asset", f"asset:{asset_id}", "device profile and enrollment state",
         lambda: Asset.model_validate(clients.backend.get_asset(asset_id)))
    load("telemetry", f"telemetry:{asset_id}:last{minutes}m", f"telemetry summary, last {minutes} min",
         lambda: summarize_telemetry(clients.backend.get_telemetry(asset_id, minutes, end), settings.channels))
    load("network", f"network:{asset_id}:health", "network health (reachability, latency, loss, interface, ACL)",
         lambda: NetworkHealth.model_validate(clients.network.get_health(asset_id)))
    load("security", f"security:{asset_id}:events60m", "enrollment state and broker/enrollment security events",
         lambda: summarize_security(clients.backend.get_events(asset_id, "security", 60),
                                    pack.asset.enrollment_state if pack.asset else None))
    load("maintenance", f"maintenance:{asset_id}", "health score and maintenance history",
         lambda: Maintenance.model_validate(clients.backend.get_maintenance(asset_id)))
    if alert_id:
        load("anomaly", f"anomaly:{alert_id}", "anomaly model score and top contributing features",
             lambda: _anomaly(clients, settings, alert_id, from_request))
    return pack

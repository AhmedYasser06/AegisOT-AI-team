"""Measured deviations: the numbers behind every diagnosis (no LLM, no canned sentences).

A *signal* is one channel that is off its baseline, e.g.
    {"group": "temperature", "channel": "temperature_c", "value": 31.8, "baseline": 25.9, "delta": 5.9,
     "ratio": 1.23, "direction": "high", "change": "+5.9", "strength": 1.18}

"""
from __future__ import annotations

import re

from .enums import FaultType

# group -> channel names (telemetry names, derived names and the ML's feature names)
GROUPS: dict[str, frozenset[str]] = {
    "temperature": frozenset({"temperature", "temperature_c"}),
    "vibration": frozenset({"vibration_rms", "gyro_mag", "accel_mag", "gyro_magnitude", "acceleration_magnitude"}),
    "electrical": frozenset({"current", "current_ma", "bus_voltage_v", "shunt_voltage_mv", "power_mw"}),
    "speed": frozenset({"speed", "speed_percent", "rpm"}),
}
CHANNEL_GROUP = {ch: g for g, chans in GROUPS.items() for ch in chans}
PRIORITY = ("temperature", "vibration", "electrical", "speed", "other")     # safety first: heat > vibration > ...
FAULT_OF_GROUP = {"temperature": FaultType.overheating, "vibration": FaultType.abnormal_vibration,
                  "electrical": FaultType.electrical_fault, "speed": FaultType.speed_deviation,
                  "other": FaultType.machine_fault}

# what to look at for each group (a label, not a sentence)
TARGET_OF_GROUP = {"temperature": "motor temperature", "vibration": "motor vibration", "electrical": "power supply",
                   "speed": "motor speed"}

TEMP_RISE_DEGREES = 3.0                                   # Celsius is not a ratio scale: judge by degrees (a DS18B20 jitters ~0.25)
# group -> (ratio that counts as HIGH, ratio that counts as LOW or None when a drop is not a fault)
RATIO_LIMITS = {"vibration": (1.5, None), "electrical": (1.5, 0.6), "speed": (1.5, 0.6), "other": (1.5, 0.6)}
MAX_SIGNALS = 6                                           # the strongest few; the rest is noise for an operator

_RAW_AXIS = re.compile(r"^(gyro|accel)_[xyz]_")           # components of the magnitude channels: ignored
# flags, commands and raw encoder counters are not measurements of a fault: motor_running, rpm_valid,
# temperature_valid, pwm_duty_8bit (the command), encoder_counts_* / encoder_delta (follow rpm, already judged)
_NOT_MEASUREMENTS = re.compile(r"^(ts|pwm_.*|encoder_.*)$|(_valid|_running)$")


def group_of(channel: str) -> str | None:
    """None = ignore this channel (raw axis or a flag, not a measurement)."""
    if channel in CHANNEL_GROUP:
        return CHANNEL_GROUP[channel]
    if _RAW_AXIS.match(channel) or _NOT_MEASUREMENTS.search(channel):
        return None
    return "other"


def _signal(group: str, channel: str, value: float, baseline: float | None, delta: float | None,
            ratio: float | None) -> dict | None:
    if baseline is None:
        return None
    if group == "temperature":
        if delta is None or delta < TEMP_RISE_DEGREES:
            return None
        return {"group": group, "channel": channel, "value": round(value, 3), "baseline": round(baseline, 3),
                "delta": round(delta, 3), "ratio": ratio, "direction": "high", "change": f"{delta:+.1f}",
                "strength": round(delta / TEMP_RISE_DEGREES, 2)}
    if ratio is None:
        return None
    hi, lo = RATIO_LIMITS[group]
    if ratio >= hi:
        direction, strength = "high", (ratio - 1) / (hi - 1)
    elif lo is not None and 0 <= ratio <= lo:
        direction, strength = "low", (1 - ratio) / (1 - lo)
    else:
        return None
    return {"group": group, "channel": channel, "value": round(value, 3), "baseline": round(baseline, 3),
            "delta": None if delta is None else round(delta, 3), "ratio": ratio, "direction": direction,
            "change": f"{(ratio - 1) * 100:+.0f}%", "strength": round(strength, 2)}


def build_signals(telemetry: dict | None, anomaly: dict | None = None, skip: set[str] | None = None) -> list[dict]:
    """Deviations from the telemetry summary. If there is no telemetry, fall back to the ML's own top features
    (they carry value + baseline). Strongest first."""
    skip, out = skip or set(), []
    if telemetry:
        for name, ch in telemetry.get("channels", {}).items():
            g = group_of(name)
            if g is None or name in skip:
                continue
            s = _signal(g, name, ch.get("recent_mean"), ch.get("baseline"), ch.get("delta_to_baseline"),
                        ch.get("ratio_to_baseline"))
            if s:
                out.append(s)
    elif anomaly:
        for f in anomaly.get("top_features", []):
            name, value, base = f.get("name", ""), f.get("value"), f.get("baseline")
            g = group_of(name)
            if g is None or name in skip or value is None or base is None:
                continue
            s = _signal(g, name, float(value), float(base), float(value) - float(base),
                        round(float(value) / float(base), 2) if base else None)
            if s:
                out.append(s)
    return sorted(out, key=lambda s: (PRIORITY.index(s["group"]), -s["strength"]))[:MAX_SIGNALS]


def primary_fault(signals: list[dict]) -> tuple[FaultType | None, list[str]]:
    """(specific fault type, affected channels). Highest-priority group wins; every deviating channel is listed."""
    if not signals:
        return None, []
    return FAULT_OF_GROUP[signals[0]["group"]], [s["channel"] for s in signals]


def corroborated(signals: list[dict]) -> bool:
    """True when two or more kinds of measurement changed together (heat + current, rpm + voltage ...).
    A broken sensor changes ONE channel; a real fault moves its neighbours too."""
    return len({s["group"] for s in signals}) >= 2


def evidence_confidence(anomaly_score: float, signals: list[dict]) -> float:
    """Confidence in a physical fault from the DATA (it moves with the evidence, it is never a constant):
    starts from the ML score, +0.04 per extra corroborating group (max 2), -0.05 when the only deviation is
    barely over its threshold."""
    groups = len({s["group"] for s in signals})
    conf = 0.5 + 0.5 * anomaly_score + 0.04 * min(2, max(0, groups - 1))
    if groups == 1 and max(s["strength"] for s in signals) < 1.25:
        conf -= 0.05
    return round(max(0.0, min(0.97, conf)), 3)


def _fmt(v: float) -> str:
    """Readable number: 998.667 -> 998.7, 7.1296 -> 7.13, 0.0325 -> 0.0325."""
    a = abs(v)
    return f"{v:.1f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3g}"


def describe(s: dict) -> str:
    """One measured sentence, composed from the numbers: 'rpm is 49.8, 71% below its baseline 171.5'."""
    way = "above" if s["direction"] == "high" else "below"
    if s["group"] == "temperature":
        return f"{s['channel']} is {_fmt(s['value'])}, {abs(s['delta']):.1f} \u00b0C above its baseline {_fmt(s['baseline'])}"
    amount = s["change"].lstrip("+-")
    return f"{s['channel']} is {_fmt(s['value'])}, {amount} {way} its baseline {_fmt(s['baseline'])}"

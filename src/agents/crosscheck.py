from __future__ import annotations

from .enums import Cause
from .schemas import SpecialistReport

CONFIDENT = 0.5
SENSOR_CAP = 0.3             # max sensor_fault vote when the evidence is corroborated by other channels
UNSUPPORTED_CAP = 0.3        # max physical_fault vote when every measured channel is within baseline
RULES_WEIGHT = 0.7


def _top(scores: dict):
    return max(scores.items(), key=lambda kv: kv[1])


def crosscheck(llm: SpecialistReport, rules: SpecialistReport) -> tuple[SpecialistReport, str | None]:
    """Return (report to use, note or None). `note` is None when nothing needs to be said."""
    authority = {"fault_type": rules.fault_type, "affected_channels": rules.affected_channels,
                 "signals": rules.signals, "corroborated": rules.corroborated}
    llm = llm.model_copy(update=authority if rules.fault_type or rules.signals else {})
    if not llm.scores or not rules.scores:
        return llm, None

    notes: list[str] = []
    scores = dict(llm.scores)
    if rules.corroborated and rules.scores.get(Cause.sensor_fault, 0.0) < CONFIDENT \
            and scores.get(Cause.sensor_fault, 0.0) > SENSOR_CAP:
        notes.append(f"{llm.role}: LLM sensor_fault vote {scores[Cause.sensor_fault]:.2f} capped to {SENSOR_CAP} "
                     f"because {len({s['group'] for s in rules.signals})} kinds of measurement changed together "
                     f"({', '.join(rules.affected_channels)}).")
        scores[Cause.sensor_fault] = SENSOR_CAP
    if "telemetry_all_within_baseline" in rules.checks_performed and not rules.signals \
            and scores.get(Cause.physical_fault, 0.0) > UNSUPPORTED_CAP:
        notes.append(f"{llm.role}: LLM physical_fault vote {scores[Cause.physical_fault]:.2f} capped to "
                     f"{UNSUPPORTED_CAP} because no measured channel deviates from its baseline.")
        scores[Cause.physical_fault] = UNSUPPORTED_CAP
    llm = llm.model_copy(update={"scores": scores})

    (lc, ls), (rc, rs) = _top(llm.scores), _top(rules.scores)
    if rs < CONFIDENT:
        return llm, "; ".join(notes) or None                       # rules see nothing: nothing to compare
    if lc == rc and ls >= CONFIDENT:                               # agree: blend, so the score follows the data
        blended = {c: round(RULES_WEIGHT * rules.scores.get(c, 0.0) + (1 - RULES_WEIGHT) * llm.scores.get(c, 0.0), 3)
                   for c in {*llm.scores, *rules.scores}}
        return llm.model_copy(update={"scores": blended}), "; ".join(notes) or None
    merged = {c: max(llm.scores.get(c, 0.0), rules.scores.get(c, 0.0)) for c in {*llm.scores, *rules.scores}}
    if ls >= CONFIDENT:
        notes.append(f"{llm.role}: LLM says {lc.value} ({ls:.2f}) but rules say {rc.value} ({rs:.2f}); "
                     f"votes merged, the supervisor will escalate if the margin is too small.")
    else:
        notes.append(f"{llm.role}: LLM was not confident ({lc.value} {ls:.2f}) while rules say {rc.value} ({rs:.2f}); "
                     f"the rules vote was kept.")
    out = llm.model_copy(update={"scores": merged,
                                 "findings": [*llm.findings, *[f"[rules check] {f}" for f in rules.findings]]})
    return out, "; ".join(notes)

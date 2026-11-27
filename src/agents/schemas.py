from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .enums import ActionId, Cause, FaultType, Risk
from .settings import get_settings

SCHEMA_VERSION = "1.2"   # additive only. 1.1 = votes/latency/ml fields; 1.2 = fault_type, severity, headline, signals,
                         # recommended_solution, reason, llm + the compact view (views.py)


# ───────── input: the AI team's anomaly result (+ optional extras) ─────────
class TopFeature(BaseModel):
    """One feature that pushed the ML score up. Only `name` is required."""
    name: str
    contribution: float = 0.0
    value: float | None = None          # the feature's current value
    baseline: float | None = None       # its normal value (so the agent can see "2.6 vs 1.0")
    direction: Literal["high", "low"] | None = None


class MlWindow(BaseModel):
    """The window of samples the ML model scored (optional: `timestamp` alone is enough)."""
    start: datetime | None = None
    end: datetime | None = None
    n_samples: int | None = None


def _default_asset() -> str:
    return get_settings().default_asset_id


class AlertIn(BaseModel):
    """The ML team's result. Only `incident_id` and `anomaly_score` are really required.

    * asset_id   : optional. If missing, DEFAULT_ASSET_ID is used (single-ESP setup).
    * timestamp  : when the anomaly was detected. The agents fetch evidence ending at this time.
    * window     : optional; if window.end is present it is used as the evidence end instead.
    * model      : either a string ("isolation_forest") or an object {name, version, feature_version}.
    * extra fields are ignored, so the ML team can add fields without breaking us."""
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = SCHEMA_VERSION
    incident_id: str = Field(min_length=1)
    alert_id: str | None = None
    asset_id: str = Field(default_factory=_default_asset, min_length=1,
                          description="Same identifier used by device, backend and twin")
    station: str | None = None
    is_anomaly: bool = True
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                  validation_alias=AliasChoices("detected_at", "timestamp"))
    window: MlWindow | None = None
    anomaly_score: float = Field(ge=0.0, le=1.0)
    raw_score: float | None = None      # the model's own score before calibration (for debugging only)
    threshold: float | None = None
    model_name: str | None = None
    model_version: str = "unknown"
    feature_version: str | None = None
    summary: str | None = None
    features: dict[str, float] = Field(default_factory=dict)
    top_features: list[TopFeature] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_model_field(cls, data):
        """`model` may be a string or {name, version, feature_version}; flatten it into plain fields."""
        if not isinstance(data, dict) or "model" not in data:
            return data
        data = dict(data)
        m = data.pop("model")
        if isinstance(m, dict):
            data.setdefault("model_name", m.get("name"))
            data.setdefault("model_version", str(m.get("version") or "unknown"))
            data.setdefault("feature_version", m.get("feature_version"))
        elif m:
            data.setdefault("model_name", str(m))
            data.setdefault("model_version", str(m))
        return data

    @model_validator(mode="after")
    def _default_alert_id(self):
        if not self.alert_id:
            self.alert_id = self.incident_id
        return self

    @property
    def evidence_end(self) -> datetime:
        """The moment the evidence window ends: the ML window end if given, else the detection time."""
        return (self.window.end if self.window and self.window.end else None) or self.detected_at

    def inline_anomaly(self) -> dict[str, Any]:
        """The ML result as evidence, taken straight from the request (no extra call needed)."""
        return {"alert_id": self.alert_id, "score": self.anomaly_score, "raw_score": self.raw_score,
                "threshold": self.threshold, "model_name": self.model_name, "model_version": self.model_version,
                "feature_version": self.feature_version, "source": "request",
                "top_features": [f.model_dump(exclude_none=True) for f in self.top_features]}


# ───────── specialist agent output ─────────
class CauseScore(BaseModel):
    cause: Cause
    score: float = Field(ge=0.0, le=1.0)


class SpecialistLLMOutput(BaseModel):
    """What a crewAI specialist must return (kept flat: works with strict JSON-schema providers)."""
    scores: list[CauseScore]
    findings: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    checks_performed: list[str] = Field(default_factory=list)
    # written by the diagnosis LLM (all optional: an older/odd answer without them is still valid)
    fault_type: str | None = None                  # one of enums.LLM_FAULT_TYPES (validated in agents.sanitize)
    headline: str | None = None                    # ONE line naming the faulty component with measured numbers
    solution_steps: list[str] = Field(default_factory=list)   # ordered, concrete, advisory steps


class SpecialistReport(BaseModel):
    role: str
    engine: Literal["llm", "rules"]
    scores: dict[Cause, float]
    findings: list[str]
    evidence_refs: list[str]
    checks_performed: list[str]
    model: str | None = None
    # diagnosis only
    fault_type: FaultType | None = None
    affected_channels: list[str] = Field(default_factory=list)
    signals: list[dict[str, Any]] = Field(default_factory=list)    # measured deviations (see signals.py)
    corroborated: bool = False                     # two or more kinds of measurement moved together
    headline: str | None = None
    solution_steps: list[str] = Field(default_factory=list)
    llm_error: str | None = None                   # set on the rules report that replaced a failed LLM call


# ───────── final output (proposal §10.1) ─────────
class Alternative(BaseModel):
    cause: Cause
    confidence: float


class Vote(BaseModel):
    """What one specialist said (shown in the dashboard and used in the streaming `vote` events)."""
    role: str
    engine: Literal["llm", "rules"]
    scores: dict[Cause, float]
    findings: list[str] = Field(default_factory=list)


class LlmInfo(BaseModel):
    """Did the LLM really answer? (`engine` alone cannot tell: it is `mixed` whenever the diagnosis used the LLM.)"""
    status: Literal["ok", "fallback_rules", "disabled"] = "disabled"
    model: str | None = None              # the model that answered the diagnosis
    narrative_model: str | None = None    # the model that wrote the explanation
    error: str | None = None              # why the diagnosis fell back to rules


class IncidentAssessment(BaseModel):
    schema_version: str = SCHEMA_VERSION
    incident_id: str
    alert_id: str
    asset_id: str
    status: Literal["complete", "inconclusive"]
    likely_cause: Cause
    confidence: float
    alternatives: list[Alternative]
    evidence_refs: list[str]
    findings: list[str]
    recommended_action: ActionId
    action_params: dict[str, Any] = Field(default_factory=dict)
    risk_level: Risk
    approval_required: bool
    auto_executable: bool
    checks_performed: list[str]
    explanation: str
    diagnosis: str = ""                      # = headline (team's "diagnosis" field); written by the LLM or composed from data
    evidence: list[str] = Field(default_factory=list)    # human-readable evidence sentences
    recommended_action_text: str = ""        # = the first solution step; `recommended_action` is the stable machine id
    engine: Literal["llm", "rules", "mixed"]
    llm_models_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # added in schema 1.1 (all optional, so older readers keep working)
    anomaly_score: float | None = None
    ml_model_version: str | None = None
    ml_validated: bool = False           # false = ML not yet proven on new recordings (stop_equipment is capped)
    votes: list[Vote] = Field(default_factory=list)
    latency_ms: int | None = None        # total time of the run
    timings_ms: dict[str, int] = Field(default_factory=dict)   # time per node (where is it slow?)
    # added in schema 1.2 (additive): the specific fault, the numbers behind it, and what to do
    fault_type: FaultType = FaultType.unknown
    severity: Literal["low", "medium", "high", "critical"] = "low"
    headline: str = ""
    affected_channels: list[str] = Field(default_factory=list)
    signals: list[dict[str, Any]] = Field(default_factory=list)
    recommended_solution: list[str] = Field(default_factory=list)
    reason: str | None = None            # why the result is inconclusive (None when complete)
    llm: LlmInfo = Field(default_factory=LlmInfo)


class DiagnosisOut(BaseModel):
    """Compact format from the team's README ("Agent diagnosis can return")."""
    incident_id: str
    asset_id: str
    diagnosis: str
    confidence: float
    evidence: list[str]
    recommended_action: str                  # human-readable
    action_id: ActionId                      # machine-readable
    approval_required: bool
    status: Literal["complete", "inconclusive"]


# ───────── compact view: the default answer of the HTTP API (what a dashboard / mobile app shows) ─────────
class CompactAction(BaseModel):
    id: ActionId
    params: dict[str, Any] = Field(default_factory=dict)
    risk: Risk
    approval_required: bool
    auto_executable: bool


class CompactEvidence(BaseModel):
    channel: str
    value: float
    baseline: float
    change: str
    direction: Literal["high", "low"]


class IncidentCompact(BaseModel):
    """Short answer: WHAT is wrong, HOW sure, WHAT to do. Nothing is repeated. `?view=full` gives IncidentAssessment."""
    incident_id: str
    asset_id: str
    status: Literal["complete", "inconclusive"]
    severity: Literal["low", "medium", "high", "critical"]
    likely_cause: Cause
    fault_type: FaultType
    headline: str
    confidence: float
    summary: str
    affected_channels: list[str] = Field(default_factory=list)
    evidence: list[CompactEvidence] = Field(default_factory=list)
    recommended_solution: list[str] = Field(default_factory=list)
    action: CompactAction
    alternatives: list[Alternative] = Field(default_factory=list)
    other_checks: list[str] = Field(default_factory=list)     # maintenance / network / security results
    reason: str | None = None
    llm: LlmInfo
    notes: list[str] = Field(default_factory=list)
    anomaly_score: float | None = None
    latency_ms: int | None = None
    created_at: datetime

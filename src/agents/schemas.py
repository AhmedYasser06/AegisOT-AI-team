"""Versioned contracts (proposal §3.1). Share these with the backend / anomaly teams.

The enums live in enums.py and are re-exported here, so `from .schemas import Cause` keeps working."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from .enums import ActionId, Cause, Risk
from .settings import get_settings

SCHEMA_VERSION = "1.1"   # 1.1 = additive only: new optional input fields + votes/latency/ml fields in the output


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


class SpecialistReport(BaseModel):
    role: str
    engine: Literal["llm", "rules"]
    scores: dict[Cause, float]
    findings: list[str]
    evidence_refs: list[str]
    checks_performed: list[str]
    model: str | None = None


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
    diagnosis: str = ""                      # short human-readable cause (team's "diagnosis" field)
    evidence: list[str] = Field(default_factory=list)    # human-readable evidence sentences
    recommended_action_text: str = ""        # human-readable action; `recommended_action` is the stable machine id
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

"""Versioned contracts (proposal §3.1). Share these with the backend / anomaly teams."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class Cause(str, Enum):
    physical_fault = "physical_fault"
    sensor_fault = "sensor_fault"
    network_issue = "network_issue"
    security_incident = "security_incident"
    unknown = "unknown"


class ActionId(str, Enum):
    # read-only / low impact (may be automated by policy)
    notify_operator = "notify_operator"
    run_network_diagnostics = "run_network_diagnostics"
    request_sensor_recheck = "request_sensor_recheck"
    inspect_equipment = "inspect_equipment"
    escalate_to_human = "escalate_to_human"
    # high impact (ALWAYS needs human approval)
    stop_equipment = "stop_equipment"
    isolate_device = "isolate_device"
    apply_network_template = "apply_network_template"


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ───────── input: the AI team's anomaly result (+ optional extras) ─────────
class TopFeature(BaseModel):
    name: str
    contribution: float = 0.0


class AlertIn(BaseModel):
    """Accepts the team's "AI Result" format: asset_id, incident_id, is_anomaly, anomaly_score, model, timestamp.
    `alert_id` defaults to `incident_id`; `model`/`timestamp` are accepted as aliases."""
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = SCHEMA_VERSION
    incident_id: str = Field(min_length=1)
    alert_id: str | None = None
    asset_id: str = Field(min_length=1, description="Same identifier used by device, backend and twin")
    station: str | None = None
    is_anomaly: bool = True
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                  validation_alias=AliasChoices("detected_at", "timestamp"))
    anomaly_score: float = Field(ge=0.0, le=1.0)
    threshold: float | None = None
    model_version: str = Field(default="unknown", validation_alias=AliasChoices("model_version", "model"))
    summary: str | None = None
    features: dict[str, float] = Field(default_factory=dict)
    top_features: list[TopFeature] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_alert_id(self):
        if not self.alert_id:
            self.alert_id = self.incident_id
        return self

    def inline_anomaly(self) -> dict[str, Any]:
        """The ML result as evidence, taken straight from the request (no extra call needed)."""
        return {"alert_id": self.alert_id, "score": self.anomaly_score, "threshold": self.threshold,
                "model_version": self.model_version, "source": "request",
                "top_features": [f.model_dump() for f in self.top_features]}


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

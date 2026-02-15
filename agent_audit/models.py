"""Data models for agent execution traces, explanations, and policy violations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    DECISION = "decision"
    OBSERVATION = "observation"


class Step(BaseModel):
    """A single step in an agent execution trace."""

    step_id: str
    type: StepType
    description: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
    parent_step_id: str | None = None


class Trace(BaseModel):
    """A complete agent execution trace."""

    trace_id: str
    agent_name: str
    steps: list[Step] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Violation(BaseModel):
    """A policy violation found during evaluation."""

    rule_id: str
    rule_description: str
    step_id: str
    severity: str = "medium"
    details: str = ""


class StepExplanation(BaseModel):
    """LLM-generated explanation for a single step."""

    step_id: str
    summary: str
    rationale: str = ""
    alternatives: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    data_accessed: list[str] = Field(default_factory=list)
    raw_response: str = ""


class TraceSummary(BaseModel):
    """LLM-generated summary of the entire trace."""

    overview: str
    key_actions: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    raw_response: str = ""


class AuditResult(BaseModel):
    """Complete audit result combining all analysis."""

    trace: Trace
    explanations: list[StepExplanation] = Field(default_factory=list)
    summary: TraceSummary | None = None
    violations: list[Violation] = Field(default_factory=list)
    risk_score: int = 0


class BatchResult(BaseModel):
    """Result of auditing multiple traces in batch mode."""

    results: list[AuditResult]
    skipped: list[dict] = Field(default_factory=list)  # {"file": str, "error": str}

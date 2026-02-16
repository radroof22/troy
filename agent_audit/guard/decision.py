"""Decision model returned by the guard on each check."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_audit.models import Violation


class Decision(BaseModel):
    """Result of evaluating a single step against the policy."""

    step_id: str
    allowed: bool
    violations: list[Violation] = Field(default_factory=list)
    risk_score: int = 0
    mode: str = "enforce"

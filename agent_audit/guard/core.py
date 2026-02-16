"""AgentAuditGuard — real-time interception for agent tool calls."""

from __future__ import annotations

import functools
import uuid
from pathlib import Path
from typing import Any, Callable

from agent_audit.guard.decision import Decision
from agent_audit.models import Step, StepType, Trace, Violation
from agent_audit.policy.engine import (
    PolicyRule,
    _make_step_dict,
    compute_risk_score,
    evaluate_step,
    load_policy,
)

VALID_MODES = ("enforce", "monitor", "dry-run")


class AgentAuditGuard:
    """Real-time guard that evaluates policy rules before tool calls execute.

    Modes:
        enforce  — blocks violating steps (raises PermissionError via check())
        monitor  — allows all steps but fires on_violation callback
        dry-run  — logs only, no callbacks and no blocking
    """

    def __init__(
        self,
        policy: str | Path | list[PolicyRule],
        agent_name: str = "agent",
        mode: str = "enforce",
        on_violation: Callable[[Decision], None] | None = None,
        agent_metadata: dict[str, Any] | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

        if isinstance(policy, (str, Path)):
            self._rules = load_policy(Path(policy))
        else:
            self._rules = list(policy)

        self._agent_name = agent_name
        self._mode = mode
        self._on_violation = on_violation
        self._agent_metadata = agent_metadata or {}

        # Accumulated state
        self._steps: list[Step] = []
        self._step_dicts: list[dict[str, Any]] = []
        self._violations: list[Violation] = []

    @property
    def mode(self) -> str:
        return self._mode

    def check(
        self,
        action: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        step_type: StepType = StepType.TOOL_CALL,
    ) -> Decision:
        """Evaluate a proposed step against the policy.

        Returns a Decision. In enforce mode, the caller should check
        decision.allowed before executing the tool.
        """
        step_id = f"step-{len(self._steps)}"
        step = Step(
            step_id=step_id,
            type=step_type,
            description=action,
            input=input or {},
            output={},
            metadata=metadata or {},
        )
        step_dict = _make_step_dict(step)

        agent_dict = {"name": self._agent_name, "metadata": self._agent_metadata}
        trace_dict = {"agent_name": self._agent_name, "metadata": self._agent_metadata}

        violations = evaluate_step(
            step_dict=step_dict,
            rules=self._rules,
            prev_steps=list(self._step_dicts),
            agent=agent_dict,
            trace=trace_dict,
        )

        # Always record the step (even if blocked) so subsequent rules can reference it
        self._steps.append(step)
        self._step_dicts.append(step_dict)
        self._violations.extend(violations)

        all_violations = list(self._violations)
        risk = compute_risk_score(all_violations, self._rules)

        allowed = len(violations) == 0
        decision = Decision(
            step_id=step_id,
            allowed=allowed,
            violations=violations,
            risk_score=risk,
            mode=self._mode,
        )

        if not allowed and self._mode == "monitor":
            if self._on_violation:
                self._on_violation(decision)
            decision = decision.model_copy(update={"allowed": True})
        elif not allowed and self._mode == "dry-run":
            decision = decision.model_copy(update={"allowed": True})

        return decision

    def record_output(self, step_id: str, output: dict[str, Any]) -> None:
        """Attach output to a previously checked step after it executes."""
        for i, step in enumerate(self._steps):
            if step.step_id == step_id:
                self._steps[i] = step.model_copy(update={"output": output})
                self._step_dicts[i]["output"] = output
                return
        raise KeyError(f"Unknown step_id: {step_id}")

    def get_trace(self) -> Trace:
        """Return the accumulated trace for post-session audit."""
        if not self._steps:
            raise ValueError("No steps recorded yet")
        return Trace(
            trace_id=str(uuid.uuid4()),
            agent_name=self._agent_name,
            steps=list(self._steps),
            metadata=self._agent_metadata,
        )

    def reset(self) -> None:
        """Clear all accumulated state for a new session."""
        self._steps.clear()
        self._step_dicts.clear()
        self._violations.clear()


def audited(guard: AgentAuditGuard):
    """Decorator that wraps a function with guard check → execute → record_output.

    Usage:
        @audited(guard)
        def send_email(to, body):
            ...

    Raises PermissionError in enforce mode if the step is blocked.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            action = fn.__name__
            input_data = {"args": list(args), "kwargs": kwargs}
            decision = guard.check(action=action, input=input_data)

            if not decision.allowed:
                raise PermissionError(
                    f"Blocked by policy: {[v.rule_description for v in decision.violations]}"
                )

            result = fn(*args, **kwargs)
            guard.record_output(decision.step_id, {"result": result})
            return result

        return wrapper
    return decorator

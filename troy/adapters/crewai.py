"""CrewAI adapter for TroyGuard — global hook-based integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from crewai.hooks import (
        register_before_tool_call_hook,
        register_after_tool_call_hook,
        unregister_before_tool_call_hook,
        unregister_after_tool_call_hook,
    )
except ImportError:
    raise ImportError(
        "Install crewai to use the CrewAI adapter: "
        "pip install 'troy[crewai]'"
    )

from troy.guard.core import TroyGuard
from troy.guard.decision import Decision
from troy.models import StepType
from troy.policy.engine import PolicyRule

# Module-level state for hook registration/cleanup
_before_hook: Callable | None = None
_after_hook: Callable | None = None
_guard: TroyGuard | None = None
_step_ids: dict[str, str] = {}


def enable_troy(
    policy: str | Path | list[PolicyRule],
    mode: str = "enforce",
    agent_name: str = "crewai-agent",
    on_violation: Callable[[Decision], None] | None = None,
    agent_metadata: dict[str, Any] | None = None,
    metadata_fn: Callable[[str, dict[str, Any], StepType], dict[str, Any]] | None = None,
) -> TroyGuard:
    """Enable audit hooks for CrewAI tool calls. Returns the guard instance."""
    global _before_hook, _after_hook, _guard, _step_ids

    _guard = TroyGuard(
        policy=policy,
        agent_name=agent_name,
        mode=mode,
        on_violation=on_violation,
        agent_metadata=agent_metadata,
    )
    _step_ids = {}

    def before_hook(context: Any) -> bool | None:
        tool_name = getattr(context, "tool_name", "unknown_tool")
        tool_input = getattr(context, "tool_input", {})
        if isinstance(tool_input, str):
            tool_input = {"input": tool_input}

        meta = metadata_fn(tool_name, tool_input, StepType.TOOL_CALL) if metadata_fn else None
        decision = _guard.check(
            action=tool_name,
            input=tool_input,
            metadata=meta,
            step_type=StepType.TOOL_CALL,
        )
        _step_ids[tool_name] = decision.step_id

        if not decision.allowed:
            return False
        return None

    def after_hook(context: Any) -> None:
        tool_name = getattr(context, "tool_name", "unknown_tool")
        step_id = _step_ids.pop(tool_name, None)
        if step_id:
            tool_result = getattr(context, "tool_result", None)
            _guard.record_output(step_id, {"result": str(tool_result)})

    _before_hook = before_hook
    _after_hook = after_hook

    register_before_tool_call_hook(before_hook)
    register_after_tool_call_hook(after_hook)

    return _guard


def disable_troy() -> None:
    """Unregister audit hooks from CrewAI."""
    global _before_hook, _after_hook, _guard, _step_ids

    if _before_hook is not None:
        unregister_before_tool_call_hook(_before_hook)
        _before_hook = None
    if _after_hook is not None:
        unregister_after_tool_call_hook(_after_hook)
        _after_hook = None
    _guard = None
    _step_ids = {}

"""OpenAI Agents SDK hooks adapter for TroyGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from agents import AgentHooks
except ImportError:
    raise ImportError(
        "Install openai-agents to use the OpenAI Agents adapter: "
        "pip install 'troy[openai-agents]'"
    )

from troy.guard.core import TroyGuard
from troy.guard.decision import Decision
from troy.models import StepType
from troy.policy.engine import PolicyRule


class TroyHooks(AgentHooks):
    """OpenAI Agents SDK hooks that intercept tool and LLM calls via TroyGuard."""

    def __init__(
        self,
        policy: str | Path | list[PolicyRule],
        mode: str = "enforce",
        agent_name: str = "openai-agent",
        on_violation: Callable[[Decision], None] | None = None,
        agent_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._guard = TroyGuard(
            policy=policy,
            agent_name=agent_name,
            mode=mode,
            on_violation=on_violation,
            agent_metadata=agent_metadata,
        )
        self._tool_step_ids: dict[str, str] = {}

    @property
    def guard(self) -> TroyGuard:
        return self._guard

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        decision = self._guard.check(
            action=tool_name,
            input={"tool": tool_name},
            step_type=StepType.TOOL_CALL,
        )
        self._tool_step_ids[tool_name] = decision.step_id

        if not decision.allowed:
            raise PermissionError(
                f"Blocked by policy: {[v.rule_description for v in decision.violations]}"
            )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        tool_name = getattr(tool, "name", str(tool))
        step_id = self._tool_step_ids.pop(tool_name, None)
        if step_id:
            self._guard.record_output(step_id, {"result": str(result)})

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: str | None = None, input_items: Any = None) -> None:
        agent_name = getattr(agent, "name", "unknown_agent")
        decision = self._guard.check(
            action=agent_name,
            input={"system_prompt": system_prompt or "", "input_items": str(input_items or [])},
            step_type=StepType.LLM_CALL,
        )
        self._tool_step_ids["__llm__"] = decision.step_id

        if not decision.allowed:
            raise PermissionError(
                f"Blocked by policy: {[v.rule_description for v in decision.violations]}"
            )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        step_id = self._tool_step_ids.pop("__llm__", None)
        if step_id:
            self._guard.record_output(step_id, {"result": str(response)})

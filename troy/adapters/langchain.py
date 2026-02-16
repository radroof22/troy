"""LangChain callback handler adapter for TroyGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "Install langchain-core to use the LangChain adapter: "
        "pip install 'troy[langchain]'"
    )

from troy.guard.core import TroyGuard
from troy.guard.decision import Decision
from troy.models import StepType
from troy.policy.engine import PolicyRule


class TroyHandler(BaseCallbackHandler):
    """LangChain callback handler that intercepts tool and LLM calls via TroyGuard."""

    def __init__(
        self,
        policy: str | Path | list[PolicyRule],
        mode: str = "enforce",
        agent_name: str = "langchain-agent",
        on_violation: Callable[[Decision], None] | None = None,
        agent_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._guard = TroyGuard(
            policy=policy,
            agent_name=agent_name,
            mode=mode,
            on_violation=on_violation,
            agent_metadata=agent_metadata,
        )
        self._run_to_step: dict[UUID, str] = {}

    @property
    def guard(self) -> TroyGuard:
        return self._guard

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        input_data = kwargs.get("inputs", {"input": input_str})
        if isinstance(input_data, str):
            input_data = {"input": input_data}

        decision = self._guard.check(
            action=tool_name,
            input=input_data,
            step_type=StepType.TOOL_CALL,
        )
        self._run_to_step[run_id] = decision.step_id

        if not decision.allowed:
            raise PermissionError(
                f"Blocked by policy: {[v.rule_description for v in decision.violations]}"
            )

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._run_to_step.pop(run_id, None)
        if step_id:
            self._guard.record_output(step_id, {"result": output})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._run_to_step.pop(run_id, None)
        if step_id:
            self._guard.record_output(step_id, {"error": str(error)})

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown_llm")
        decision = self._guard.check(
            action=model_name,
            input={"prompts": prompts},
            step_type=StepType.LLM_CALL,
        )
        self._run_to_step[run_id] = decision.step_id

        if not decision.allowed:
            raise PermissionError(
                f"Blocked by policy: {[v.rule_description for v in decision.violations]}"
            )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._run_to_step.pop(run_id, None)
        if step_id:
            text = ""
            if response.generations:
                gen = response.generations[0]
                if gen:
                    text = gen[0].text
            self._guard.record_output(step_id, {"result": text})

"""Tests for framework adapters — all framework imports are mocked."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from troy.guard.core import TroyGuard
from troy.models import StepType
from troy.policy.engine import PolicyRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_all_rule() -> list[PolicyRule]:
    """Policy that blocks every tool call."""
    return [
        PolicyRule(
            rule_id="block-all",
            description="Block all tool calls",
            condition="step['type'] == 'tool_call'",
            severity="critical",
        )
    ]


def _allow_all_rules() -> list[PolicyRule]:
    """Empty policy — everything is allowed."""
    return []


# ---------------------------------------------------------------------------
# LangChain Adapter
# ---------------------------------------------------------------------------

class _FakeBaseCallbackHandler:
    """Minimal stand-in for langchain_core.callbacks.base.BaseCallbackHandler."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@pytest.fixture(autouse=False)
def _mock_langchain(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake langchain_core module so the adapter can import."""
    callbacks_base = types.ModuleType("langchain_core.callbacks.base")
    callbacks_base.BaseCallbackHandler = _FakeBaseCallbackHandler

    outputs_mod = types.ModuleType("langchain_core.outputs")
    outputs_mod.LLMResult = type("LLMResult", (), {})

    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    lc = types.ModuleType("langchain_core")
    lc.callbacks = callbacks_mod
    lc.callbacks.base = callbacks_base
    lc.outputs = outputs_mod

    monkeypatch.setitem(sys.modules, "langchain_core", lc)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks.base", callbacks_base)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs_mod)

    # Force re-import so the adapter picks up the fakes
    sys.modules.pop("troy.adapters.langchain", None)
    from troy.adapters.langchain import TroyHandler
    yield TroyHandler


class TestLangChainAdapter:
    @pytest.fixture(autouse=True)
    def _setup(self, _mock_langchain):
        self.TroyHandler = _mock_langchain

    def test_on_tool_start_allowed(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        run_id = uuid4()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        assert run_id in handler._run_to_step

    def test_on_tool_start_blocked_enforce(self):
        handler = self.TroyHandler(policy=_block_all_rule(), mode="enforce")
        with pytest.raises(PermissionError, match="Blocked by policy"):
            handler.on_tool_start({"name": "search"}, "query", run_id=uuid4())

    def test_on_tool_start_allowed_monitor(self):
        violations_seen = []
        handler = self.TroyHandler(
            policy=_block_all_rule(),
            mode="monitor",
            on_violation=lambda d: violations_seen.append(d),
        )
        # Should NOT raise
        handler.on_tool_start({"name": "search"}, "query", run_id=uuid4())
        assert len(violations_seen) == 1

    def test_on_tool_end_records_output(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        run_id = uuid4()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        handler.on_tool_end("result text", run_id=run_id)
        # Step should have output recorded
        step = handler.guard._steps[0]
        assert step.output == {"result": "result text"}

    def test_on_llm_start_and_end(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        run_id = uuid4()
        handler.on_llm_start({"name": "gpt-4"}, ["Hello"], run_id=run_id)
        assert run_id in handler._run_to_step
        # Create a fake LLMResult
        fake_gen = MagicMock()
        fake_gen.text = "Hi there"
        fake_response = MagicMock()
        fake_response.generations = [[fake_gen]]
        handler.on_llm_end(fake_response, run_id=run_id)
        step = handler.guard._steps[0]
        assert step.output == {"result": "Hi there"}

    def test_run_id_correlation(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        run1 = uuid4()
        run2 = uuid4()
        handler.on_tool_start({"name": "tool_a"}, "input_a", run_id=run1)
        handler.on_tool_start({"name": "tool_b"}, "input_b", run_id=run2)
        handler.on_tool_end("output_b", run_id=run2)
        handler.on_tool_end("output_a", run_id=run1)
        assert handler.guard._steps[0].output == {"result": "output_a"}
        assert handler.guard._steps[1].output == {"result": "output_b"}

    def test_guard_property(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        assert isinstance(handler.guard, TroyGuard)

    def test_on_tool_error_records(self):
        handler = self.TroyHandler(policy=_allow_all_rules())
        run_id = uuid4()
        handler.on_tool_start({"name": "flaky"}, "x", run_id=run_id)
        handler.on_tool_error(RuntimeError("boom"), run_id=run_id)
        assert handler.guard._steps[0].output == {"error": "boom"}

    def test_import_error_message(self, monkeypatch):
        # Remove the fake langchain_core so the real ImportError fires
        for key in list(sys.modules):
            if key.startswith("langchain_core") or key == "troy.adapters.langchain":
                monkeypatch.delitem(sys.modules, key, raising=False)
        with pytest.raises(ImportError, match="Install langchain-core"):
            import importlib
            importlib.import_module("troy.adapters.langchain")


# ---------------------------------------------------------------------------
# OpenAI Agents SDK Adapter
# ---------------------------------------------------------------------------

class _FakeAgentHooks:
    """Minimal stand-in for agents.AgentHooks."""
    pass


@pytest.fixture(autouse=False)
def _mock_openai_agents(monkeypatch: pytest.MonkeyPatch):
    agents_mod = types.ModuleType("agents")
    agents_mod.AgentHooks = _FakeAgentHooks
    monkeypatch.setitem(sys.modules, "agents", agents_mod)

    sys.modules.pop("troy.adapters.openai_agents", None)
    from troy.adapters.openai_agents import TroyHooks
    yield TroyHooks


class TestOpenAIAgentsAdapter:
    @pytest.fixture(autouse=True)
    def _setup(self, _mock_openai_agents):
        self.TroyHooks = _mock_openai_agents

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_on_tool_start_allowed(self):
        hooks = self.TroyHooks(policy=_allow_all_rules())
        tool = MagicMock()
        tool.name = "search"
        self._run(hooks.on_tool_start(MagicMock(), MagicMock(), tool))
        assert "search" in hooks._tool_step_ids

    def test_on_tool_start_blocked(self):
        hooks = self.TroyHooks(policy=_block_all_rule(), mode="enforce")
        tool = MagicMock()
        tool.name = "search"
        with pytest.raises(PermissionError, match="Blocked by policy"):
            self._run(hooks.on_tool_start(MagicMock(), MagicMock(), tool))

    def test_on_tool_end_records(self):
        hooks = self.TroyHooks(policy=_allow_all_rules())
        tool = MagicMock()
        tool.name = "search"
        self._run(hooks.on_tool_start(MagicMock(), MagicMock(), tool))
        self._run(hooks.on_tool_end(MagicMock(), MagicMock(), tool, "result"))
        assert hooks.guard._steps[0].output == {"result": "result"}

    def test_on_llm_start_and_end(self):
        hooks = self.TroyHooks(policy=_allow_all_rules())
        agent = MagicMock()
        agent.name = "my-agent"
        self._run(hooks.on_llm_start(MagicMock(), agent, "You are helpful", []))
        assert "__llm__" in hooks._tool_step_ids
        self._run(hooks.on_llm_end(MagicMock(), agent, "Hello!"))
        assert hooks.guard._steps[0].output == {"result": "Hello!"}

    def test_monitor_mode(self):
        violations_seen = []
        hooks = self.TroyHooks(
            policy=_block_all_rule(),
            mode="monitor",
            on_violation=lambda d: violations_seen.append(d),
        )
        tool = MagicMock()
        tool.name = "search"
        # Should NOT raise
        self._run(hooks.on_tool_start(MagicMock(), MagicMock(), tool))
        assert len(violations_seen) == 1

    def test_guard_property(self):
        hooks = self.TroyHooks(policy=_allow_all_rules())
        assert isinstance(hooks.guard, TroyGuard)

    def test_import_error_message(self, monkeypatch):
        for key in list(sys.modules):
            if key == "agents" or key == "troy.adapters.openai_agents":
                monkeypatch.delitem(sys.modules, key, raising=False)
        with pytest.raises(ImportError, match="Install openai-agents"):
            import importlib
            importlib.import_module("troy.adapters.openai_agents")


# ---------------------------------------------------------------------------
# CrewAI Adapter
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _mock_crewai(monkeypatch: pytest.MonkeyPatch):
    _registered_before: list = []
    _registered_after: list = []

    hooks_mod = types.ModuleType("crewai.hooks")
    hooks_mod.register_before_tool_call_hook = lambda fn: _registered_before.append(fn)
    hooks_mod.register_after_tool_call_hook = lambda fn: _registered_after.append(fn)
    hooks_mod.unregister_before_tool_call_hook = lambda fn: _registered_before.remove(fn) if fn in _registered_before else None
    hooks_mod.unregister_after_tool_call_hook = lambda fn: _registered_after.remove(fn) if fn in _registered_after else None

    crewai_mod = types.ModuleType("crewai")
    crewai_mod.hooks = hooks_mod

    monkeypatch.setitem(sys.modules, "crewai", crewai_mod)
    monkeypatch.setitem(sys.modules, "crewai.hooks", hooks_mod)

    sys.modules.pop("troy.adapters.crewai", None)
    from troy.adapters.crewai import enable_troy, disable_troy
    yield enable_troy, disable_troy, _registered_before, _registered_after


class TestCrewAIAdapter:
    @pytest.fixture(autouse=True)
    def _setup(self, _mock_crewai):
        self.enable_troy, self.disable_troy, self._before, self._after = _mock_crewai

    def test_enable_troy_registers_hooks(self):
        self.enable_troy(policy=_allow_all_rules())
        assert len(self._before) == 1
        assert len(self._after) == 1

    def test_before_hook_allows(self):
        self.enable_troy(policy=_allow_all_rules())
        ctx = MagicMock()
        ctx.tool_name = "safe_tool"
        ctx.tool_input = {"x": 1}
        result = self._before[0](ctx)
        assert result is None  # None = allow

    def test_before_hook_blocks(self):
        self.enable_troy(policy=_block_all_rule(), mode="enforce")
        ctx = MagicMock()
        ctx.tool_name = "dangerous_tool"
        ctx.tool_input = {}
        result = self._before[0](ctx)
        assert result is False  # False = block

    def test_after_hook_records_output(self):
        guard = self.enable_troy(policy=_allow_all_rules())
        ctx_before = MagicMock()
        ctx_before.tool_name = "search"
        ctx_before.tool_input = {}
        self._before[0](ctx_before)

        ctx_after = MagicMock()
        ctx_after.tool_name = "search"
        ctx_after.tool_result = "found it"
        self._after[0](ctx_after)

        assert guard._steps[0].output == {"result": "found it"}

    def test_monitor_mode_allows(self):
        violations_seen = []
        self.enable_troy(
            policy=_block_all_rule(),
            mode="monitor",
            on_violation=lambda d: violations_seen.append(d),
        )
        ctx = MagicMock()
        ctx.tool_name = "tool"
        ctx.tool_input = {}
        result = self._before[0](ctx)
        assert result is None  # monitor mode allows
        assert len(violations_seen) == 1

    def test_disable_troy(self):
        self.enable_troy(policy=_allow_all_rules())
        assert len(self._before) == 1
        self.disable_troy()
        assert len(self._before) == 0
        assert len(self._after) == 0

    def test_import_error_message(self, monkeypatch):
        for key in list(sys.modules):
            if key.startswith("crewai") or key == "troy.adapters.crewai":
                monkeypatch.delitem(sys.modules, key, raising=False)
        with pytest.raises(ImportError, match="Install crewai"):
            import importlib
            importlib.import_module("troy.adapters.crewai")

"""Tests for the real-time interception SDK (guard)."""

from __future__ import annotations

import pytest

from agent_audit.guard import AgentAuditGuard, Decision, audited
from agent_audit.models import StepType, Violation
from agent_audit.policy.engine import PolicyRule, evaluate_step, _make_step_dict


# ---------------------------------------------------------------------------
# evaluate_step (unit tests for the new engine function)
# ---------------------------------------------------------------------------

class TestEvaluateStep:
    """Tests for evaluate_step() added to policy/engine.py."""

    SIMPLE_RULE = PolicyRule(
        rule_id="block-external",
        description="Block external calls",
        condition="get(step, 'metadata.network_zone') == 'external'",
        severity="high",
        weight=40,
    )

    def test_single_step_violation(self):
        step = {"step_id": "s1", "type": "tool_call", "metadata": {"network_zone": "external"}}
        violations = evaluate_step(step, [self.SIMPLE_RULE])
        assert len(violations) == 1
        assert violations[0].rule_id == "block-external"

    def test_single_step_no_violation(self):
        step = {"step_id": "s1", "type": "tool_call", "metadata": {"network_zone": "internal"}}
        violations = evaluate_step(step, [self.SIMPLE_RULE])
        assert len(violations) == 0

    def test_any_next_always_false(self):
        """In real-time mode, any_next() should always return False."""
        rule = PolicyRule(
            rule_id="future-check",
            description="Needs future",
            condition="any_next(lambda s: True)",
        )
        step = {"step_id": "s1", "type": "tool_call", "metadata": {}}
        violations = evaluate_step(step, [rule])
        assert len(violations) == 0

    def test_any_prev_with_history(self):
        """any_prev() should see previously accumulated steps."""
        rule = PolicyRule(
            rule_id="pii-send",
            description="PII then send",
            condition="get(step, 'metadata.network_zone') == 'external' and any_prev(lambda s: get(s, 'metadata.data_classification') == 'pii')",
            severity="critical",
            weight=50,
        )
        prev = [{"step_id": "s0", "type": "tool_call", "metadata": {"data_classification": "pii"}}]
        step = {"step_id": "s1", "type": "tool_call", "metadata": {"network_zone": "external"}}
        violations = evaluate_step(step, [rule], prev_steps=prev)
        assert len(violations) == 1

    def test_agent_metadata(self):
        rule = PolicyRule(
            rule_id="priv-esc",
            description="Privilege escalation",
            condition="get(agent, 'metadata.permission_level') != 'admin' and get(step, 'metadata.permission_level') == 'admin'",
            severity="critical",
            weight=50,
        )
        step = {"step_id": "s1", "type": "tool_call", "metadata": {"permission_level": "admin"}}
        violations = evaluate_step(step, [rule], agent={"name": "bot", "metadata": {"permission_level": "user"}})
        assert len(violations) == 1

    def test_malformed_condition_skipped(self):
        rule = PolicyRule(rule_id="bad", description="broken", condition="not valid python!!!")
        step = {"step_id": "s1", "type": "tool_call", "metadata": {}}
        violations = evaluate_step(step, [rule])
        assert violations == []


# ---------------------------------------------------------------------------
# Guard init
# ---------------------------------------------------------------------------

class TestGuardInit:
    def test_from_rule_list(self):
        rules = [PolicyRule("r1", "test", "True")]
        guard = AgentAuditGuard(policy=rules)
        assert guard.mode == "enforce"

    def test_from_file_path(self, tmp_path):
        policy_file = tmp_path / "policy.json"
        policy_file.write_text('{"rules": [{"rule_id": "r1", "description": "d", "condition": "True"}]}')
        guard = AgentAuditGuard(policy=str(policy_file))
        assert guard.mode == "enforce"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            AgentAuditGuard(policy=[], mode="invalid")


# ---------------------------------------------------------------------------
# Guard check
# ---------------------------------------------------------------------------

class TestGuardCheck:
    BLOCK_RULE = PolicyRule(
        rule_id="block-external",
        description="Block external calls",
        condition="get(step, 'metadata.network_zone') == 'external'",
        severity="high",
        weight=40,
    )

    def test_returns_decision(self):
        guard = AgentAuditGuard(policy=[self.BLOCK_RULE])
        decision = guard.check("read_file", metadata={"network_zone": "internal"})
        assert isinstance(decision, Decision)
        assert decision.allowed is True

    def test_enforce_blocks(self):
        guard = AgentAuditGuard(policy=[self.BLOCK_RULE], mode="enforce")
        decision = guard.check("send_data", metadata={"network_zone": "external"})
        assert decision.allowed is False
        assert len(decision.violations) == 1

    def test_monitor_allows_with_violations(self):
        guard = AgentAuditGuard(policy=[self.BLOCK_RULE], mode="monitor")
        decision = guard.check("send_data", metadata={"network_zone": "external"})
        assert decision.allowed is True
        assert len(decision.violations) == 1

    def test_dry_run_allows(self):
        guard = AgentAuditGuard(policy=[self.BLOCK_RULE], mode="dry-run")
        decision = guard.check("send_data", metadata={"network_zone": "external"})
        assert decision.allowed is True
        assert len(decision.violations) == 1

    def test_monitor_fires_callback(self):
        received = []
        guard = AgentAuditGuard(
            policy=[self.BLOCK_RULE],
            mode="monitor",
            on_violation=lambda d: received.append(d),
        )
        guard.check("send_data", metadata={"network_zone": "external"})
        assert len(received) == 1
        assert received[0].violations[0].rule_id == "block-external"

    def test_dry_run_no_callback(self):
        received = []
        guard = AgentAuditGuard(
            policy=[self.BLOCK_RULE],
            mode="dry-run",
            on_violation=lambda d: received.append(d),
        )
        guard.check("send_data", metadata={"network_zone": "external"})
        assert len(received) == 0


# ---------------------------------------------------------------------------
# Cross-step real-time detection
# ---------------------------------------------------------------------------

class TestCrossStepRealtime:
    """PII exfiltration detected in real-time using any_prev() pattern."""

    PII_REALTIME_RULE = PolicyRule(
        rule_id="pii-exfiltration-rt",
        description="PII exfiltration (real-time)",
        condition="get(step, 'metadata.network_zone') == 'external' and step['type'] == 'tool_call' and any_prev(lambda s: get(s, 'metadata.data_classification') == 'pii')",
        severity="critical",
        weight=50,
    )

    def test_blocked_at_send_step(self):
        guard = AgentAuditGuard(policy=[self.PII_REALTIME_RULE], mode="enforce")
        # Step 1: access PII — allowed (no rule fires on this alone)
        d1 = guard.check("read_pii", metadata={"data_classification": "pii"})
        assert d1.allowed is True
        # Step 2: send externally — blocked
        d2 = guard.check("send_external", metadata={"network_zone": "external"})
        assert d2.allowed is False
        assert d2.violations[0].rule_id == "pii-exfiltration-rt"

    def test_safe_workflow_allowed(self):
        guard = AgentAuditGuard(policy=[self.PII_REALTIME_RULE], mode="enforce")
        d1 = guard.check("read_public", metadata={"data_classification": "public"})
        assert d1.allowed is True
        d2 = guard.check("send_internal", metadata={"network_zone": "internal"})
        assert d2.allowed is True


# ---------------------------------------------------------------------------
# Risk score accumulation
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_cumulative_scoring(self):
        r1 = PolicyRule("r1", "d1", "True", weight=30)
        r2 = PolicyRule("r2", "d2", "True", weight=25)
        guard = AgentAuditGuard(policy=[r1, r2])
        d1 = guard.check("action1")
        assert d1.risk_score == 55
        d2 = guard.check("action2")
        # Same rules fire again, but compute_risk_score deduplicates rule_ids
        assert d2.risk_score == 55

    def test_capped_at_100(self):
        rules = [PolicyRule(f"r{i}", "d", "True", weight=40) for i in range(5)]
        guard = AgentAuditGuard(policy=rules)
        d = guard.check("action")
        assert d.risk_score == 100


# ---------------------------------------------------------------------------
# Trace accumulation
# ---------------------------------------------------------------------------

class TestTraceAccumulation:
    def test_record_output(self):
        guard = AgentAuditGuard(policy=[])
        d = guard.check("read_file", input={"path": "/data"})
        guard.record_output(d.step_id, {"content": "hello"})
        trace = guard.get_trace()
        assert trace.steps[0].output == {"content": "hello"}

    def test_get_trace(self):
        guard = AgentAuditGuard(policy=[], agent_name="my-agent")
        guard.check("step1")
        guard.check("step2")
        trace = guard.get_trace()
        assert trace.agent_name == "my-agent"
        assert len(trace.steps) == 2

    def test_empty_trace_raises(self):
        guard = AgentAuditGuard(policy=[])
        with pytest.raises(ValueError, match="No steps recorded"):
            guard.get_trace()

    def test_record_output_unknown_step_raises(self):
        guard = AgentAuditGuard(policy=[])
        with pytest.raises(KeyError):
            guard.record_output("nonexistent", {})


# ---------------------------------------------------------------------------
# @audited decorator
# ---------------------------------------------------------------------------

class TestAuditedDecorator:
    def test_allowed_executes(self):
        guard = AgentAuditGuard(policy=[])

        @audited(guard)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_blocked_raises_permission_error(self):
        rule = PolicyRule("always-block", "Always block", "True", severity="critical")
        guard = AgentAuditGuard(policy=[rule], mode="enforce")

        @audited(guard)
        def dangerous():
            return "should not run"

        with pytest.raises(PermissionError, match="Blocked by policy"):
            dangerous()

    def test_records_output(self):
        guard = AgentAuditGuard(policy=[])

        @audited(guard)
        def greet(name):
            return f"Hello, {name}!"

        greet("Alice")
        trace = guard.get_trace()
        assert trace.steps[0].output == {"result": "Hello, Alice!"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_policy(self):
        guard = AgentAuditGuard(policy=[])
        d = guard.check("anything", metadata={"network_zone": "external"})
        assert d.allowed is True
        assert d.risk_score == 0

    def test_reset_clears_state(self):
        rule = PolicyRule("r1", "d", "True", weight=10)
        guard = AgentAuditGuard(policy=[rule])
        guard.check("action1")
        guard.reset()
        with pytest.raises(ValueError, match="No steps recorded"):
            guard.get_trace()
        # After reset, new checks start fresh
        d = guard.check("action2")
        assert d.step_id == "step-0"
